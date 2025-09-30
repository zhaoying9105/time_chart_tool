"""
重构后的通信性能分析器
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ...parser import PyTorchProfilerParser
from ...models import ActivityEvent, ProfilerData
from .base import BaseCommunicationAnalyzer, EventAnalysisMixin, _readable_timestamp_to_microseconds
from .alignment import EventAlignmentAnalyzer
from .event_analysis import EventAnalyzer
from .excel_generator import ExcelGenerator
from .visualization import AlignmentVisualizer


class CommunicationAnalyzerV2(BaseCommunicationAnalyzer, EventAnalysisMixin):
    """重构后的通信性能分析器"""
    
    def __init__(self):
        super().__init__()
        self.parser = PyTorchProfilerParser()
        self.event_analyzer = EventAnalyzer()
        self.alignment_analyzer = EventAlignmentAnalyzer()
        self.excel_generator = ExcelGenerator()
        self.visualizer = AlignmentVisualizer()
        # 保存parser实例用于调用栈分析
        self.fastest_parser = None
        self.slowest_parser = None
    
    def analyze_communication_performance(self, pod_dir: str, step: Optional[int] = None, 
                                         comm_idx: Optional[int] = None,
                                         fastest_card_idx: Optional[int] = None, 
                                         slowest_card_idx: Optional[int] = None,
                                         kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL",
                                         prev_kernel_pattern: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL_BF16_ADD",
                                         output_dir: str = ".",
                                         show_dtype: bool = False,
                                         show_shape: bool = False,
                                         show_kernel_names: bool = False,
                                         show_kernel_duration: bool = False,
                                         show_timestamp: bool = False,
                                         show_readable_timestamp: bool = False,
                                         show_kernel_timestamp: bool = False) -> List[Path]:
        """
        分析通信性能的主入口
        
        Args:
            pod_dir: Pod目录路径
            step: 指定要分析的step
            comm_idx: 指定要分析的通信操作索引
            fastest_card_idx: 指定最快卡的索引
            slowest_card_idx: 指定最慢卡的索引
            kernel_prefix: 通信kernel前缀
            prev_kernel_pattern: 上一个通信kernel的匹配模式
            output_dir: 输出目录
            show_dtype: 是否显示数据类型
            show_shape: 是否显示形状信息
            show_kernel_names: 是否显示kernel名称
            show_kernel_duration: 是否显示kernel持续时间
            show_timestamp: 是否显示时间戳
            show_readable_timestamp: 是否显示可读时间戳
            show_kernel_timestamp: 是否显示kernel时间戳
            
        Returns:
            List[Path]: 生成的文件路径列表
        """
        print("=== 通信性能分析 ===")
        
        # 扫描executor文件夹
        executor_folders = self._scan_executor_folders(pod_dir)
        if not executor_folders:
            print("错误: 没有找到executor文件夹")
            return []
        
        print(f"找到 {len(executor_folders)} 个executor文件夹")
        
        # 提取通信数据
        comm_data = self._extract_communication_data(executor_folders, step, kernel_prefix)
        if not comm_data:
            print("错误: 没有找到通信数据")
            return []
        
        # 生成输出文件
        generated_files = []
        
        # 生成原始数据Excel
        raw_data_file = self.excel_generator.generate_raw_data_excel(comm_data, output_dir)
        generated_files.append(raw_data_file)
        
        # 生成统计信息Excel
        stats_file = self.excel_generator.generate_statistics_excel(comm_data, output_dir)
        generated_files.append(stats_file)
        
        # 深度分析（如果指定了step和comm_idx）
        if step is not None and comm_idx is not None:
            deep_analysis_files = self._perform_deep_analysis(
                comm_data, executor_folders, step, comm_idx, output_dir, 
                kernel_prefix, prev_kernel_pattern, fastest_card_idx, slowest_card_idx,
                show_timestamp, show_readable_timestamp
            )
            if deep_analysis_files:
                if isinstance(deep_analysis_files, list):
                    generated_files.extend(deep_analysis_files)
                else:
                    generated_files.append(deep_analysis_files)
        
        return generated_files
    
    def _extract_communication_data(self, executor_folders: List[str], step: Optional[int] = None, 
                                   kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL") -> Dict[int, Dict[int, List[float]]]:
        """
        提取通信数据
        
        Args:
            executor_folders: executor文件夹列表
            step: 指定要分析的step
            kernel_prefix: 通信kernel前缀
            
        Returns:
            Dict[int, Dict[int, List[float]]]: 通信数据 {step: {card_idx: [durations]}}
        """
        comm_data = {}
        
        for executor_folder in executor_folders:
            # 查找JSON文件
            json_files = glob.glob(os.path.join(executor_folder, "*.json"))
            
            for json_file in json_files:
                step_num, card_idx = self._parse_json_filename(os.path.basename(json_file))
                
                if step_num is None or card_idx is None:
                    continue
                
                if step is not None and step_num != step:
                    continue
                
                # 提取通信持续时间
                durations = self._extract_communication_durations(json_file, kernel_prefix)
                
                if durations:
                    if step_num not in comm_data:
                        comm_data[step_num] = {}
                    comm_data[step_num][card_idx] = durations
        
        return comm_data
    
    def _perform_deep_analysis(self, comm_data: Dict[int, Dict[int, List[float]]], 
                              executor_folders: List[str],
                              step: int, comm_idx: int, output_dir: str, 
                              kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL",
                              prev_kernel_pattern: str = "TCDP_.*", 
                              fastest_card_idx: Optional[int] = None, 
                              slowest_card_idx: Optional[int] = None,
                              show_timestamp: bool = False, 
                              show_readable_timestamp: bool = False) -> Optional[List[Path]]:
        """
        执行深度分析，比较快卡和慢卡的详细差异
        """
        print(f"=== 开始深度分析 ===")
        print(f"分析step={step}, comm_idx={comm_idx}")
        
        # 2. 确定快慢卡
        if fastest_card_idx is not None and slowest_card_idx is not None:
            fastest_card = (fastest_card_idx, None)
            slowest_card = (slowest_card_idx, None)
            print(f"使用指定的快慢卡:")
        else:
            # 自动查找差别最大的两个card
            fastest_card, slowest_card = self._find_fastest_slowest_cards(
                comm_data, step, comm_idx
            )
            if not fastest_card or not slowest_card:
                return None
        
        # 3. 找到对应的JSON文件
        fastest_json_file = self._find_json_file(executor_folders, step, fastest_card[0])
        slowest_json_file = self._find_json_file(executor_folders, step, slowest_card[0])
        
        if not fastest_json_file or not slowest_json_file:
            print("错误: 无法找到对应的JSON文件")
            return None
        
        print(f"最快card JSON文件: {fastest_json_file}")
        print(f"最慢card JSON文件: {slowest_json_file}")
        
        # 4. 加载并解析JSON文件
        fastest_parser = PyTorchProfilerParser()
        slowest_parser = PyTorchProfilerParser()
        
        # 保存为实例变量，供后续调用栈分析使用
        self.fastest_parser = fastest_parser
        self.slowest_parser = slowest_parser
        
        fastest_data = fastest_parser.load_json_file(fastest_json_file)
        slowest_data = slowest_parser.load_json_file(slowest_json_file)
        
        # 构建调用栈树
        print("构建调用栈树...")
        fastest_parser.build_call_stack_trees()
        slowest_parser.build_call_stack_trees()
        
        # 5. 进行深度对比分析
        comparison_result = self._compare_card_performance(
            fastest_data, slowest_data, fastest_card[0], slowest_card[0], 
            step, comm_idx, fastest_card[1], slowest_card[1], kernel_prefix, 
            prev_kernel_pattern, show_timestamp, show_readable_timestamp, output_dir
        )
        
        if not comparison_result:
            print("错误: 深度对比分析失败")
            return None
        
        # 6. 生成深度分析Excel文件
        excel_result = self.excel_generator.generate_deep_analysis_excel(
            comparison_result, step, comm_idx, output_dir
        )
        
        # 7. 添加可视化文件到返回结果
        if isinstance(excel_result, tuple):
            # 返回了两个文件（主文件 + CPU Start Time分析文件）
            excel_file, cpu_excel_file = excel_result
            generated_files = [excel_file, cpu_excel_file]
        else:
            # 只返回了一个文件
            generated_files = [excel_result]
            
        if 'visualization_files' in comparison_result:
            generated_files.extend([Path(f) for f in comparison_result['visualization_files']])
        
        print("=== 深度分析完成 ===")
        return generated_files
    
    def _find_fastest_slowest_cards(self, comm_data: Dict[int, Dict[int, List[float]]], 
                                   step: int, comm_idx: int) -> Tuple[Tuple[int, float], Tuple[int, float]]:
        """找到最快和最慢的card"""
        if step not in comm_data:
            print(f"错误: 没有找到step={step}的数据")
            return None, None
        
        step_data = comm_data[step]
        card_durations = {}
        
        for card_idx, durations in step_data.items():
            if comm_idx < len(durations):
                card_durations[card_idx] = durations[comm_idx]
        
        if len(card_durations) < 2:
            print(f"错误: step={step}, comm_idx={comm_idx}只有{len(card_durations)}个card的数据，无法比较")
            return None, None
        
        # 自动查找差别最大的两个card
        sorted_cards = sorted(card_durations.items(), key=lambda x: x[1])
        fastest_card = sorted_cards[0]  # (card_idx, duration)
        slowest_card = sorted_cards[-1]  # (card_idx, duration)
        print(f"自动查找的快慢卡:")
        print(f"最快card: {fastest_card[0]}, duration: {fastest_card[1]:.2f}")
        print(f"最慢card: {slowest_card[0]}, duration: {slowest_card[1]:.2f}")
        print(f"性能差异: {slowest_card[1] / fastest_card[1]:.2f}倍")
        
        return fastest_card, slowest_card
    
    def _compare_card_performance(self, fastest_data: ProfilerData, 
                                 slowest_data: ProfilerData, 
                                 fastest_card_idx: int, slowest_card_idx: int,
                                 step: int, comm_idx: int, 
                                 fastest_duration: float, slowest_duration: float,
                                 kernel_prefix: str, prev_kernel_pattern: str, 
                                 show_timestamp: bool = False, 
                                 show_readable_timestamp: bool = False, 
                                 output_dir: str = ".") -> Optional[Dict[str, Any]]:
        """比较快卡和慢卡的性能差异"""
        print("开始比较快卡和慢卡的性能差异...")
        
        # 1. 检查通信kernel一致性
        if not self.event_analyzer.check_communication_kernel_consistency(
            fastest_data, slowest_data, kernel_prefix, comm_idx):
            print("错误: 通信kernel一致性检查失败")
            return None
        
        # 2. 对两个数据进行Stage1处理
        fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id = \
            self.event_analyzer.postprocessor.stage1_data_postprocessing(fastest_data)
        slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id = \
            self.event_analyzer.postprocessor.stage1_data_postprocessing(slowest_data)
        
        # 3. 找到目标通信kernel操作的时间范围
        fastest_comm_event = self._find_communication_event(fastest_data, comm_idx, kernel_prefix)
        slowest_comm_event = self._find_communication_event(slowest_data, comm_idx, kernel_prefix)
        
        if not fastest_comm_event or not slowest_comm_event:
            print("错误: 无法找到目标通信kernel操作")
            return None
        
        # 4. 找到上一个通信kernel操作的时间范围
        print("=== 查找上一个通信kernel ===")
        fastest_prev_comm_events = self.event_analyzer.find_prev_communication_kernel_events(
            fastest_data, [fastest_comm_event], prev_kernel_pattern)
        slowest_prev_comm_events = self.event_analyzer.find_prev_communication_kernel_events(
            slowest_data, [slowest_comm_event], prev_kernel_pattern)
        
        # 5. 验证上一个通信kernel名称是否一致
        self._verify_prev_comm_kernel_consistency(fastest_prev_comm_events, slowest_prev_comm_events)
        
        # 6. 确定分析的时间范围并提取事件
        fastest_start_time, fastest_end_time = self._get_analysis_time_ranges(
            fastest_data, fastest_prev_comm_events, fastest_comm_event)
        slowest_start_time, slowest_end_time = self._get_analysis_time_ranges(
            slowest_data, slowest_prev_comm_events, slowest_comm_event)
        
        print(f"最快card分析时间范围: {fastest_start_time:.2f} - {fastest_end_time:.2f}")
        print(f"最慢card分析时间范围: {slowest_start_time:.2f} - {slowest_end_time:.2f}")
        
        # 7. 提取时间范围内的events并与filtered events取交集
        fastest_events_by_external_id, fastest_kernel_events_by_external_id = \
            self._extract_events_in_range_with_intersection(
                fastest_data, fastest_start_time, fastest_end_time, 
                fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id)
        slowest_events_by_external_id, slowest_kernel_events_by_external_id = \
            self._extract_events_in_range_with_intersection(
                slowest_data, slowest_start_time, slowest_end_time,
                slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id)
        
        # 8. 合并CPU和kernel events
        fastest_events_by_external_id = self._merge_cpu_and_kernel_events(
            fastest_events_by_external_id, fastest_kernel_events_by_external_id)
        slowest_events_by_external_id = self._merge_cpu_and_kernel_events(
            slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id)
        
        # 9. 使用事件分析器进行比较
        comparison_result = self.event_analyzer.compare_events_by_time_sequence(
            fastest_events_by_external_id, slowest_events_by_external_id,
            fastest_card_idx, slowest_card_idx, fastest_duration, slowest_duration,
            fastest_data, slowest_data, show_timestamp, show_readable_timestamp,
            output_dir, step, comm_idx
        )
        
        # 添加额外的元数据
        comparison_result.update({
            'step': step,
            'comm_idx': comm_idx,
            'fastest_card_idx': fastest_card_idx,
            'slowest_card_idx': slowest_card_idx,
            'fastest_duration': fastest_duration,
            'slowest_duration': slowest_duration,
            'duration_ratio': slowest_duration / fastest_duration if fastest_duration > 0 else 0
        })
        
        return comparison_result
    
    def _verify_prev_comm_kernel_consistency(self, fastest_prev_comm_events, slowest_prev_comm_events):
        """验证上一个通信kernel一致性"""
        if fastest_prev_comm_events and slowest_prev_comm_events:
            fastest_prev_kernel_name = fastest_prev_comm_events[0].name
            slowest_prev_kernel_name = slowest_prev_comm_events[0].name
            print(f"    最快卡上一个通信kernel: {fastest_prev_kernel_name}")
            print(f"    最慢卡上一个通信kernel: {slowest_prev_kernel_name}")
            
            if fastest_prev_kernel_name != slowest_prev_kernel_name:
                print(f"    错误: 快慢卡的上一个通信kernel名称不一致!")
                raise ValueError("上一个通信kernel名称不一致")
            else:
                print(f"    ✓ 上一个通信kernel名称一致: {fastest_prev_kernel_name}")
        elif not fastest_prev_comm_events and not slowest_prev_comm_events:
            print("    警告: 快慢卡都没有找到上一个通信kernel")
        else:
            print("    错误: 快慢卡的上一个通信kernel查找结果不一致!")
            raise ValueError("上一个通信kernel查找结果不一致")
    
    def _get_analysis_time_ranges(self, data: ProfilerData, prev_comm_events, comm_event):
        """获取分析时间范围"""
        prev_event = prev_comm_events[0] if prev_comm_events else None
        start_time = self._get_analysis_start_time(data, prev_event, comm_event)
        end_time = self._get_analysis_end_time(data, comm_event)
        return start_time, end_time
    
    def _get_analysis_start_time(self, data, prev_event, comm_event):
        """获取分析开始时间"""
        return prev_event.ts + prev_event.dur if prev_event else comm_event.ts
    
    def _get_analysis_end_time(self, data, comm_event):
        """获取分析结束时间"""
        return comm_event.ts
    
    def _extract_events_in_range_with_intersection(self, data, start_time, end_time, 
                                                 filtered_cpu_events_by_external_id, 
                                                 filtered_kernel_events_by_external_id):
        """提取时间范围内的events并与filtered events取交集"""
        # 提取时间范围内的所有events
        time_range_events = []
        for event in data.events:
            if event.ts >= start_time and event.ts <= end_time:
                time_range_events.append(event)
        
        # 按external_id分组时间范围内的events
        time_range_events_by_external_id = self._group_events_by_external_id(time_range_events)
        
        # 取交集：只保留在filtered events中存在的external_id
        intersected_cpu_events = {}
        intersected_kernel_events = {}
        
        for external_id, events in time_range_events_by_external_id.items():
            # 检查该external_id是否在filtered events中存在
            has_cpu_events = external_id in filtered_cpu_events_by_external_id
            has_kernel_events = external_id in filtered_kernel_events_by_external_id
            
            if has_cpu_events or has_kernel_events:
                # 分离CPU和kernel events
                cpu_events = [e for e in events if e.cat == 'cpu_op']
                kernel_events = [e for e in events if e.cat == 'kernel']
                
                if cpu_events and has_cpu_events:
                    intersected_cpu_events[external_id] = cpu_events
                
                if kernel_events and has_kernel_events:
                    intersected_kernel_events[external_id] = kernel_events
        
        print(f"时间范围内找到 {len(time_range_events_by_external_id)} 个external_id")
        print(f"与filtered events取交集后保留 {len(intersected_cpu_events)} 个CPU external_id, {len(intersected_kernel_events)} 个kernel external_id")
        
        return intersected_cpu_events, intersected_kernel_events
    
    def _group_events_by_external_id(self, events):
        """按external_id分组events"""
        events_by_external_id = defaultdict(list)
        for event in events:
            if event.external_id is not None:
                events_by_external_id[event.external_id].append(event)
        return dict(events_by_external_id)
    
    def _merge_cpu_and_kernel_events(self, cpu_events_by_external_id, kernel_events_by_external_id):
        """合并CPU和kernel events"""
        merged_events = {}
        
        for external_id, events in cpu_events_by_external_id.items():
            merged_events[external_id] = events.copy()
        
        for external_id, events in kernel_events_by_external_id.items():
            if external_id in merged_events:
                merged_events[external_id].extend(events)
            else:
                merged_events[external_id] = events.copy()
        
        return merged_events

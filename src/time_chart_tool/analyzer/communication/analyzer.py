"""
通信性能分析器
"""

import os
import glob
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from ...parser import PyTorchProfilerParser


def _readable_timestamp_to_microseconds(readable_timestamp: str) -> float:
    """将readable_timestamp转换为微秒时间戳"""
    try:
        dt = datetime.strptime(readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp() * 1_000_000
    except Exception as e:
        print(f"警告: 时间戳转换失败: {e}")
        return 0.0


def _format_timestamp_display(event, show_readable: bool = True) -> str:
    """格式化时间戳显示"""
    if show_readable:
        return event.readable_timestamp
    else:
        return f"{event.ts:.2f}μs"


def _calculate_end_time_display(event) -> str:
    """计算并格式化结束时间显示"""
    start_dt = datetime.strptime(event.readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    end_dt = start_dt + timedelta(microseconds=event.dur)
    return end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def _calculate_time_diff_readable(event1, event2) -> float:
    """使用readable_timestamp计算时间差（微秒）"""
    start1 = _readable_timestamp_to_microseconds(event1.readable_timestamp)
    start2 = _readable_timestamp_to_microseconds(event2.readable_timestamp)
    return abs(start1 - start2)


class CommunicationAnalyzer:
    COMMUNICATION_BLACKLIST_PATTERNS = [
        'TCDP_RING_ALLGATHER_',
        'TCDP_RING_REDUCESCATTER_',
        'ALLREDUCELL',
        'TCDP_RING_ALLREDUCE_SIMPLE_BF16_ADD'
    ]

    """通信性能分析器"""
    
    def __init__(self):
        self.parser = PyTorchProfilerParser()
    
    def analyze_communication_performance(self, pod_dir: str, step: Optional[int] = None, comm_idx: Optional[int] = None,
                                         fastest_card_idx: Optional[int] = None, slowest_card_idx: Optional[int] = None,
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
        分析通信性能
        
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
        raw_data_file = self._generate_raw_data_excel(comm_data, output_dir)
        generated_files.append(raw_data_file)
        
        # 生成统计信息Excel
        stats_file = self._generate_statistics_excel(comm_data, output_dir)
        generated_files.append(stats_file)
        
        # 深度分析（如果指定了step和comm_idx）
        if step is not None and comm_idx is not None:
            deep_analysis_file = self._perform_deep_analysis(
                comm_data, executor_folders, step, comm_idx, output_dir, 
                kernel_prefix, prev_kernel_pattern, fastest_card_idx, slowest_card_idx,
                show_timestamp, show_readable_timestamp
            )
            if deep_analysis_file:
                generated_files.append(deep_analysis_file)
        
        return generated_files
    
    def _perform_deep_analysis(self, comm_data: Dict[int, Dict[int, List[float]]], executor_folders: List[str],
                              step: int, comm_idx: int, output_dir: str, kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL",
                              prev_kernel_pattern: str = "TCDP_.*", fastest_card_idx: Optional[int] = None, slowest_card_idx: Optional[int] = None,
                              show_timestamp: bool = False, show_readable_timestamp: bool = False) -> Optional[Path]:
        """
        执行深度分析，比较快卡和慢卡的详细差异
        
        Args:
            comm_data: {step: {card_idx: [duration1, duration2, ...]}}
            executor_folders: executor文件夹路径列表
            step: 要分析的step
            comm_idx: 要分析的通信操作索引
            output_dir: 输出目录
            kernel_prefix: 要检测的通信kernel前缀
            prev_kernel_pattern: 上一个通信kernel的匹配模式，用于确定对比区间
            fastest_card_idx: 指定最快卡的索引，如果为None则自动查找
            slowest_card_idx: 指定最慢卡的索引，如果为None则自动查找
            
        Returns:
            Optional[Path]: 生成的深度分析文件路径
        """
        print(f"=== 开始深度分析 ===")
        print(f"分析step={step}, comm_idx={comm_idx}")
        
        # 2. 确定快慢卡
        if fastest_card_idx is not None and slowest_card_idx is not None:
            fastest_card = (fastest_card_idx, None)
            slowest_card = (slowest_card_idx, None)
            print(f"使用指定的快慢卡:")
        else:
            # 1. 找到指定step和comm_idx的duration数据
            if step not in comm_data:
                print(f"错误: 没有找到step={step}的数据")
                return None
            step_data = comm_data[step]
            card_durations = {}
            
            for card_idx, durations in step_data.items():
                if comm_idx < len(durations):
                    card_durations[card_idx] = durations[comm_idx]
            
            if len(card_durations) < 2:
                print(f"错误: step={step}, comm_idx={comm_idx}只有{len(card_durations)}个card的数据，无法比较")
                return None
            # 自动查找差别最大的两个card
            sorted_cards = sorted(card_durations.items(), key=lambda x: x[1])
            fastest_card = sorted_cards[0]  # (card_idx, duration)
            slowest_card = sorted_cards[-1]  # (card_idx, duration)
            print(f"自动查找的快慢卡:")
            print(f"最快card: {fastest_card[0]}, duration: {fastest_card[1]:.2f}")
            print(f"最慢card: {slowest_card[0]}, duration: {slowest_card[1]:.2f}")
            print(f"性能差异: {slowest_card[1] / fastest_card[1]:.2f}倍")
        
        # 3. 找到对应的JSON文件
        fastest_json_file = self._find_json_file(executor_folders, step, fastest_card[0])
        slowest_json_file = self._find_json_file(executor_folders, step, slowest_card[0])
        
        if not fastest_json_file or not slowest_json_file:
            print("错误: 无法找到对应的JSON文件")
            return None
        
        print(f"最快card JSON文件: {fastest_json_file}")
        print(f"最慢card JSON文件: {slowest_json_file}")
        
        # 4. 加载并解析JSON文件
        fastest_data = self.parser.load_json_file(fastest_json_file)
        slowest_data = self.parser.load_json_file(slowest_json_file)
        
        # 5. 进行深度对比分析
        comparison_result = self._compare_card_performance(
            fastest_data, slowest_data, fastest_card[0], slowest_card[0], 
            step, comm_idx, fastest_card[1], slowest_card[1], kernel_prefix, prev_kernel_pattern,
            show_timestamp, show_readable_timestamp
        )
        
        if not comparison_result:
            print("错误: 深度对比分析失败")
            return None
        
        # 6. 生成深度分析Excel文件
        excel_file = self._generate_deep_analysis_excel(comparison_result, step, comm_idx, output_dir)
        
        print("=== 深度分析完成 ===")
        return excel_file
    
    def _find_json_file(self, executor_folders: List[str], step: int, card_idx: int) -> Optional[str]:
        """
        根据step和card_idx找到对应的JSON文件
        
        Args:
            executor_folders: executor文件夹路径列表
            step: step值
            card_idx: card索引
            
        Returns:
            Optional[str]: JSON文件路径
        """
        for executor_folder in executor_folders:
            json_files = list(Path(executor_folder).glob("*.json"))
            for json_file in json_files:
                parsed_step, parsed_card_idx = self._parse_json_filename(json_file.name)
                if parsed_step == step and parsed_card_idx == card_idx:
                    return str(json_file)
        return None
    
    def _compare_card_performance(self, fastest_data, slowest_data, fastest_card_idx, slowest_card_idx,
                                 step, comm_idx, fastest_duration, slowest_duration,
                                 kernel_prefix, prev_kernel_pattern, show_timestamp: bool = False, 
                                 show_readable_timestamp: bool = False):
        """比较快卡和慢卡的性能差异"""
        print("开始比较快卡和慢卡的性能差异...")
        
        # 1. 检查通信kernel一致性
        if not self._check_communication_kernel_consistency(fastest_data, slowest_data, kernel_prefix, comm_idx):
            print("错误: 通信kernel一致性检查失败")
            return None
        
        # 2. 对两个数据进行Stage1处理
        from ..stages.postprocessor import DataPostProcessor
        postprocessor = DataPostProcessor()
        fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id = postprocessor.stage1_data_postprocessing(fastest_data)
        slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id = postprocessor.stage1_data_postprocessing(slowest_data)
        
        # 3. 找到目标通信kernel操作的时间范围
        fastest_comm_event = self._find_communication_event(fastest_data, comm_idx, kernel_prefix)
        slowest_comm_event = self._find_communication_event(slowest_data, comm_idx, kernel_prefix)
        
        if not fastest_comm_event or not slowest_comm_event:
            print("错误: 无法找到目标通信kernel操作")
            return None
        
        # 4. 找到上一个通信kernel操作的时间范围
        print("=== 查找上一个通信kernel ===")
        fastest_prev_comm_events = self._find_prev_communication_kernel_events(fastest_data, [fastest_comm_event], prev_kernel_pattern)
        slowest_prev_comm_events = self._find_prev_communication_kernel_events(slowest_data, [slowest_comm_event], prev_kernel_pattern)
        
        # 5. 验证上一个通信kernel名称是否一致
        if fastest_prev_comm_events and slowest_prev_comm_events:
            fastest_prev_kernel_name = fastest_prev_comm_events[0].name
            slowest_prev_kernel_name = slowest_prev_comm_events[0].name
            print(f"    最快卡上一个通信kernel: {fastest_prev_kernel_name}")
            print(f"    最慢卡上一个通信kernel: {slowest_prev_kernel_name}")
            
            if fastest_prev_kernel_name != slowest_prev_kernel_name:
                print(f"    错误: 快慢卡的上一个通信kernel名称不一致!")
                return None
            else:
                print(f"    ✓ 上一个通信kernel名称一致: {fastest_prev_kernel_name}")
        elif not fastest_prev_comm_events and not slowest_prev_comm_events:
            print("    警告: 快慢卡都没有找到上一个通信kernel")
        else:
            print("    错误: 快慢卡的上一个通信kernel查找结果不一致!")
            return None
        
        # 6. 确定分析的时间范围
        fastest_prev_event = fastest_prev_comm_events[0] if fastest_prev_comm_events else None
        slowest_prev_event = slowest_prev_comm_events[0] if slowest_prev_comm_events else None
        
        fastest_start_time = self._get_analysis_start_time(fastest_data, fastest_prev_event, fastest_comm_event)
        slowest_start_time = self._get_analysis_start_time(slowest_data, slowest_prev_event, slowest_comm_event)
        
        fastest_end_time = self._get_analysis_end_time(fastest_data, fastest_comm_event)
        slowest_end_time = self._get_analysis_end_time(slowest_data, slowest_comm_event)
        
        print(f"最快card分析时间范围: {fastest_start_time:.2f} - {fastest_end_time:.2f}")
        print(f"最慢card分析时间范围: {slowest_start_time:.2f} - {slowest_end_time:.2f}")
        
        # 7. 提取时间范围内的events并与filtered events取交集
        fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id = self._extract_events_in_range_with_intersection(
            fastest_data, fastest_start_time, fastest_end_time, 
            fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id
        )
        slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id = self._extract_events_in_range_with_intersection(
            slowest_data, slowest_start_time, slowest_end_time,
            slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id
        )
        
        # 8. 合并CPU和kernel events
        fastest_events_by_external_id = self._merge_cpu_and_kernel_events(fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id)
        slowest_events_by_external_id = self._merge_cpu_and_kernel_events(slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id)
        
        # 9. 计算通信kernel的实际duration
        fastest_comm_duration = fastest_comm_event.dur if fastest_comm_event and fastest_comm_event.dur is not None else 0.0
        slowest_comm_duration = slowest_comm_event.dur if slowest_comm_event and slowest_comm_event.dur is not None else 0.0
        
        # 10. 按时间顺序比较CPU操作
        comparison_rows = self._compare_events_by_time_sequence(
            fastest_events_by_external_id, slowest_events_by_external_id,
            fastest_card_idx, slowest_card_idx, fastest_comm_duration, slowest_comm_duration,
            show_timestamp, show_readable_timestamp
        )
        
        return {
            'step': step,
            'comm_idx': comm_idx,
            'fastest_card_idx': fastest_card_idx,
            'slowest_card_idx': slowest_card_idx,
            'fastest_duration': fastest_comm_duration,
            'slowest_duration': slowest_comm_duration,
            'duration_ratio': slowest_comm_duration / fastest_comm_duration if fastest_comm_duration > 0 else 0,
            'comparison_rows': comparison_rows['comparison_rows'],
            'top_kernel_duration_ratios': comparison_rows['top_kernel_duration_ratios'],
            'top_cpu_start_time_differences': comparison_rows['top_cpu_start_time_differences']
        }
    
    def _generate_deep_analysis_excel(self, comparison_result, step, comm_idx, output_dir):
        """生成深度分析Excel文件"""
        if pd is None:
            raise ImportError("pandas is required for Excel output. Please install pandas and openpyxl.")
        
        print("生成深度分析Excel文件...")
        
        # 创建DataFrame
        df = pd.DataFrame(comparison_result['comparison_rows'])
        
        # 添加汇总信息
        summary_data = {
            'step': step,
            'comm_idx': comm_idx,
            'fastest_card_idx': comparison_result['fastest_card_idx'],
            'slowest_card_idx': comparison_result['slowest_card_idx'],
            'fastest_duration': comparison_result['fastest_duration'],
            'slowest_duration': comparison_result['slowest_duration'],
            'duration_ratio': comparison_result['duration_ratio'],
            'total_cpu_start_time_diff': df['cpu_start_time_diff'].sum() if not df.empty else 0,
            'total_kernel_duration_diff': df['kernel_duration_diff'].sum() if not df.empty else 0,
            'total_cpu_start_time_diff_ratio': df['cpu_start_time_diff_ratio'].sum() if not df.empty else 0,
            'total_kernel_duration_diff_ratio': df['kernel_duration_diff_ratio'].sum() if not df.empty else 0
        }
        
        # 保存到Excel文件
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        excel_file = output_path / f"comm_deep_analysis_step_{step}_idx_{comm_idx}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 写入详细对比数据
            df.to_excel(writer, sheet_name='详细对比', index=False)
            
            # 写入汇总信息
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='汇总信息', index=False)
            
            # 写入kernel duration ratio分析数据
            if 'top_kernel_duration_ratios' in comparison_result and comparison_result['top_kernel_duration_ratios']:
                kernel_data = []
                for i, (idx, ratio, row) in enumerate(comparison_result['top_kernel_duration_ratios']):
                    kernel_data.append({
                        '排名': i + 1,
                        '事件序列': idx + 1,
                        '最快Card CPU操作名': row['cpu_op_name'],
                        '最慢Card CPU操作名': row.get('slowest_cpu_op_name', 'N/A'),
                        '最快Card操作形状': row['cpu_op_shape'],
                        '最慢Card操作形状': row.get('slowest_cpu_op_shape', 'N/A'),
                        '最快Card操作类型': row['cpu_op_dtype'],
                        '最慢Card操作类型': row.get('slowest_cpu_op_dtype', 'N/A'),
                        'Kernel Duration Ratio': ratio,
                        '最快Card CPU开始时间(可读)': row['fastest_cpu_readable_timestamp'],
                        '最快Card CPU开始时间(ts)': row.get('fastest_cpu_start_time', 'N/A'),
                        '最慢Card CPU开始时间(可读)': row['slowest_cpu_readable_timestamp'],
                        '最慢Card CPU开始时间(ts)': row.get('slowest_cpu_start_time', 'N/A'),
                        '最快Card CPU持续时间': row['fastest_cpu_duration'],
                        '最慢Card CPU持续时间': row['slowest_cpu_duration'],
                        'CPU持续时间差异': row['cpu_duration_diff'],
                        '最快Card Kernel持续时间': row['fastest_kernel_duration'],
                        '最慢Card Kernel持续时间': row['slowest_kernel_duration'],
                        'Kernel持续时间差异': row['kernel_duration_diff']
                    })
                kernel_df = pd.DataFrame(kernel_data)
                kernel_df.to_excel(writer, sheet_name='Kernel Duration分析', index=False)
            
            # 写入CPU start time相邻差值分析数据
            if 'top_cpu_start_time_differences' in comparison_result and comparison_result['top_cpu_start_time_differences']:
                cpu_data = []
                for i, diff_info in enumerate(comparison_result['top_cpu_start_time_differences']):
                    prev_row = diff_info['prev_row']
                    current_row = diff_info['current_row']
                    cpu_data.append({
                        '排名': i + 1,
                        '事件对': f"{diff_info['index_pair'][0]+1}-{diff_info['index_pair'][1]+1}",
                        '前一个最快Card操作名': prev_row['cpu_op_name'],
                        '前一个最慢Card操作名': prev_row.get('slowest_cpu_op_name', 'N/A'),
                        '前一个Ratio': diff_info['prev_ratio'],
                        '前一个最快Card时间(可读)': prev_row['fastest_cpu_readable_timestamp'],
                        '前一个最快Card时间(ts)': prev_row.get('fastest_cpu_start_time', 'N/A'),
                        '前一个最慢Card时间(可读)': prev_row['slowest_cpu_readable_timestamp'],
                        '前一个最慢Card时间(ts)': prev_row.get('slowest_cpu_start_time', 'N/A'),
                        '当前最快Card操作名': current_row['cpu_op_name'],
                        '当前最慢Card操作名': current_row.get('slowest_cpu_op_name', 'N/A'),
                        '当前Ratio': diff_info['current_ratio'],
                        '当前最快Card时间(可读)': current_row['fastest_cpu_readable_timestamp'],
                        '当前最快Card时间(ts)': current_row.get('fastest_cpu_start_time', 'N/A'),
                        '当前最慢Card时间(可读)': current_row['slowest_cpu_readable_timestamp'],
                        '当前最慢Card时间(ts)': current_row.get('slowest_cpu_start_time', 'N/A'),
                        '相邻差值': diff_info['difference'],
                        '前一个CPU启动时间差异': prev_row['cpu_start_time_diff'],
                        '当前CPU启动时间差异': current_row['cpu_start_time_diff']
                    })
                cpu_df = pd.DataFrame(cpu_data)
                cpu_df.to_excel(writer, sheet_name='CPU Start Time分析', index=False)
        
        print(f"深度分析Excel文件已生成: {excel_file}")
        return excel_file
    
    def _find_all_communication_events(self, data: 'ProfilerData') -> List['ActivityEvent']:
        """
        找到所有TCDP_开头的通信kernel events，按结束时间排序
        
        Args:
            data: ProfilerData对象
            
        Returns:
            List[ActivityEvent]: 所有通信kernel events，按结束时间排序
        """
        comm_events = []
        
        for event in data.events:
            if (event.cat == 'kernel' and 
                event.name.startswith('TCDP_')):
                # 检查是否在黑名单中
                is_blacklisted = any(pattern in event.name for pattern in self.COMMUNICATION_BLACKLIST_PATTERNS)
                if not is_blacklisted:
                    comm_events.append(event)
        
        # 按结束时间排序（从早到晚）
        comm_events.sort(key=lambda x: x.ts + x.dur)
        
        return comm_events

    def _print_communication_events_table(self, fastest_events, slowest_events) -> bool:
        """打印通信事件对比表格并返回一致性检查结果"""
        print("    通信事件对比表格:")
        print("    | 序号 | 最快卡Kernel名称 | 最快卡开始时间 | 最快卡结束时间 | 最慢卡Kernel名称 | 最慢卡开始时间 | 最慢卡结束时间 | 名称一致 | 开始时间差(ms) | 结束时间差(ms) | 超过阈值 |")
        print("    |------|----------------|-------------|-------------|----------------|-------------|-------------|----------|---------------|---------------|----------|")
        
        max_events = max(len(fastest_events), len(slowest_events))
        consistency_check_passed = True
        
        for i in range(max_events):
            fastest_event = fastest_events[i] if i < len(fastest_events) else None
            slowest_event = slowest_events[i] if i < len(slowest_events) else None
            
            # 基本信息
            fastest_name = fastest_event.name if fastest_event else "N/A"
            slowest_name = slowest_event.name if slowest_event else "N/A"
            name_consistent = "✓" if fastest_name == slowest_name else "✗"
            
            # 时间戳显示
            fastest_start_ts = _format_timestamp_display(fastest_event) if fastest_event else "N/A"
            slowest_start_ts = _format_timestamp_display(slowest_event) if slowest_event else "N/A"
            fastest_end_ts = _calculate_end_time_display(fastest_event) if fastest_event else "N/A"
            slowest_end_ts = _calculate_end_time_display(slowest_event) if slowest_event else "N/A"
            
            # 时间差计算
            start_time_diff, end_time_diff, threshold_exceeded = self._calculate_time_differences(
                fastest_event, slowest_event
            )
            
            if threshold_exceeded == "✗":
                consistency_check_passed = False
            
            print(f"    | {i+1} | {fastest_name} | {fastest_start_ts} | {fastest_end_ts} | {slowest_name} | {slowest_start_ts} | {slowest_end_ts} | {name_consistent} | {start_time_diff} | {end_time_diff} | {threshold_exceeded} |")
        
        return consistency_check_passed

    def _calculate_time_differences(self, fastest_event, slowest_event) -> tuple:
        """计算时间差并检查阈值"""
        
        # 开始时间差
        start_diff_us = _calculate_time_diff_readable(fastest_event, slowest_event)
        start_time_diff = f"{start_diff_us / 1000.0:.3f}"
        
        # 结束时间差
        fastest_start_us = _readable_timestamp_to_microseconds(fastest_event.readable_timestamp)
        slowest_start_us = _readable_timestamp_to_microseconds(slowest_event.readable_timestamp)
        fastest_end_us = fastest_start_us + fastest_event.dur
        slowest_end_us = slowest_start_us + slowest_event.dur
        end_diff_ms = abs(fastest_end_us - slowest_end_us) / 1000.0
        end_time_diff = f"{end_diff_ms:.3f}"
        
        # 阈值检查
        threshold_exceeded = "✗" if end_diff_ms > 5.0 else "✓"
        
        return start_time_diff, end_time_diff, threshold_exceeded

    def _check_communication_kernel_consistency(self, fastest_data: 'ProfilerData', slowest_data: 'ProfilerData',
                                              kernel_prefix: str, comm_idx: int) -> bool:
        """
        检查快慢卡的所有通信kernel顺序和时间一致性
        
        Args:
            fastest_data: 最快card的ProfilerData
            slowest_data: 最慢card的ProfilerData
            kernel_prefix: 通信kernel前缀（用于查找目标通信操作）
            comm_idx: 通信操作索引
            
        Returns:
            bool: 是否一致
        """
        print("=== 检查通信kernel一致性 ===")
        
        # 1. 找到所有TCDP_开头的通信kernel events（用于检查顺序和同步性）
        fastest_all_comm_events = self._find_all_communication_events(fastest_data)
        slowest_all_comm_events = self._find_all_communication_events(slowest_data)
        
        # 2. 找到目标通信kernel events（用于确定分析范围）
        fastest_target_comm_event = self._find_communication_event(fastest_data, comm_idx, kernel_prefix)
        slowest_target_comm_event = self._find_communication_event(slowest_data, comm_idx, kernel_prefix)
        
        if not fastest_target_comm_event or not slowest_target_comm_event:
            print("    错误: 无法找到目标通信kernel events")
            return False
        
        if not fastest_all_comm_events or not slowest_all_comm_events:
            print("    错误: 无法找到任何通信kernel events")
            return False
        
        # 2. 时间戳标准化已在parser中完成
        print("    时间戳标准化已在parser中完成")
        
        # 3. 打印通信事件对比表格
        consistency_check_passed = self._print_communication_events_table(
            fastest_all_comm_events, slowest_all_comm_events
        )
        
        # 4. 检查kernel名称顺序一致性
        fastest_kernel_names = [event.name for event in fastest_all_comm_events]
        slowest_kernel_names = [event.name for event in slowest_all_comm_events]
        
        print(f"    最快卡通信kernel序列: {fastest_kernel_names}")
        print(f"    最慢卡通信kernel序列: {slowest_kernel_names}")
        
        sequence_consistent = (fastest_kernel_names == slowest_kernel_names)
        if not sequence_consistent:
            print("    警告: 通信kernel名称序列不一致!")
            print(f"    最快卡: {fastest_kernel_names}")
            print(f"    最慢卡: {slowest_kernel_names}")
        else:
            print("    ✓ 通信kernel名称序列一致")
        
        # 5. 检查结果
        if not consistency_check_passed:
            print("    ✗ 通信kernel一致性检查失败: 存在超过5000us阈值的事件")
            return False
        else:
            print("    ✓ 通信kernel一致性检查通过")
            return True
    
    def _find_communication_event(self, data, comm_idx, kernel_prefix):
        """找到指定comm_idx的通信kernel操作event"""
        events = self._find_events_by_criteria(
            data.events, 
            lambda e: e.cat == 'kernel' and e.name.startswith(kernel_prefix)
        )
        
        if comm_idx < len(events):
            print(f"    找到目标通信kernel event: {events[comm_idx].name}")
            return events[comm_idx]
        
        return None
    
    def _find_events_by_criteria(self, events, criteria_func):
        """根据条件查找事件"""
        return [event for event in events if criteria_func(event)]
    
    def _find_prev_communication_kernel_events(self, data, target_kernel_events, prev_kernel_pattern):
        """找到目标通信kernel之前的通信kernel events"""
        if not target_kernel_events:
            print("    警告: 目标通信kernel events为空")
            return []
        
        target_start_time = min(event.ts for event in target_kernel_events)
        target_kernel_name = target_kernel_events[0].name
        print(f"    目标通信kernel: {target_kernel_name}, 开始时间: {target_start_time:.2f}")
        
        # 查找匹配条件的通信kernel events
        communication_kernels = self._find_events_by_criteria(
            data.events,
            lambda e: (e.cat == 'kernel' and e.ts < target_start_time and
                      re.match(prev_kernel_pattern, e.name) and
                      not any(pattern in e.name for pattern in self.COMMUNICATION_BLACKLIST_PATTERNS))
        )
        
        print(f"    找到 {len(communication_kernels)} 个匹配的通信kernel events")
        
        if communication_kernels:
            # 按结束时间排序，取最后一个
            communication_kernels.sort(key=lambda x: x.ts + x.dur)
            last_kernel_event = communication_kernels[-1]
            
            print(f"    上一个通信kernel: {last_kernel_event.name}")
            print(f"    开始时间: {_format_timestamp_display(last_kernel_event)}")
            print(f"    结束时间: {_calculate_end_time_display(last_kernel_event)}")
            
            return [last_kernel_event]
        else:
            print(f"    警告: 没有找到匹配模式 '{prev_kernel_pattern}' 的上一个通信kernel")
        
        return []
    
    def _get_analysis_start_time(self, data, prev_event, current_event):
        """获取分析开始时间（简化版本）"""
        return prev_event.ts + prev_event.dur
    
    def _get_analysis_end_time(self, data, comm_event):
        """获取分析结束时间（简化版本）"""
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
        from collections import defaultdict
        events_by_external_id = defaultdict(list)
        for event in events:
            if event.external_id is not None:
                events_by_external_id[event.external_id].append(event)
        return dict(events_by_external_id)
    
    def _merge_cpu_and_kernel_events(self, cpu_events_by_external_id, kernel_events_by_external_id):
        """合并CPU和kernel events（简化版本）"""
        merged_events = {}
        
        for external_id, events in cpu_events_by_external_id.items():
            merged_events[external_id] = events.copy()
        
        for external_id, events in kernel_events_by_external_id.items():
            if external_id in merged_events:
                merged_events[external_id].extend(events)
            else:
                merged_events[external_id] = events.copy()
        
        return merged_events
    
    def _compare_events_by_time_sequence(self, fastest_events_by_external_id, slowest_events_by_external_id,
                                        fastest_card_idx, slowest_card_idx, fastest_duration, slowest_duration,
                                        show_timestamp: bool = False, show_readable_timestamp: bool = False):
        """按照时间顺序比较快卡和慢卡的events"""
        comparison_rows = []
        
        # 1. 提取所有CPU events并按时间排序
        fastest_cpu_events = []
        slowest_cpu_events = []
        
        for events in fastest_events_by_external_id.values():
            fastest_cpu_events.extend([e for e in events if e.cat == 'cpu_op'])
        for events in slowest_events_by_external_id.values():
            slowest_cpu_events.extend([e for e in events if e.cat == 'cpu_op'])
        
        # 按开始时间排序
        fastest_cpu_events.sort(key=lambda x: x.ts)
        slowest_cpu_events.sort(key=lambda x: x.ts)
        
        print(f"最快card找到 {len(fastest_cpu_events)} 个CPU events")
        print(f"最慢card找到 {len(slowest_cpu_events)} 个CPU events")
        
        # 2. 检查CPU操作是否一致
        cpu_ops_consistent = self._check_cpu_ops_consistency(fastest_cpu_events, slowest_cpu_events)
        if not cpu_ops_consistent:
            print("警告: 快卡和慢卡的CPU操作不完全一致，比较结果可能不准确")
        else:
            print("CPU操作一致性检查通过")
        
        # 3. 检查kernel操作是否一致
        kernel_ops_consistent = self._check_kernel_ops_consistency(fastest_events_by_external_id, slowest_events_by_external_id)
        if not kernel_ops_consistent:
            print("警告: 快卡和慢卡的kernel操作不完全一致，比较结果可能不准确")
        else:
            print("Kernel操作一致性检查通过")
        
        # 4. 按照时间顺序进行配对比较
        min_events_count = min(len(fastest_cpu_events), len(slowest_cpu_events))
        
        for i in range(min_events_count):
            fastest_cpu_event = fastest_cpu_events[i]
            slowest_cpu_event = slowest_cpu_events[i]
            
            # 找到对应的kernel events
            fastest_kernel_events = self._find_kernel_events_for_cpu_event(fastest_cpu_event, fastest_events_by_external_id)
            slowest_kernel_events = self._find_kernel_events_for_cpu_event(slowest_cpu_event, slowest_events_by_external_id)
            
            # 计算时间差异 - 使用readable_timestamp
            fastest_start_us = _readable_timestamp_to_microseconds(fastest_cpu_event.readable_timestamp)
            slowest_start_us = _readable_timestamp_to_microseconds(slowest_cpu_event.readable_timestamp)
            cpu_start_time_diff = slowest_start_us - fastest_start_us
            
            cpu_duration_diff = slowest_cpu_event.dur - fastest_cpu_event.dur
            
            # 计算kernel时间差异
            fastest_kernel_duration = sum(e.dur for e in fastest_kernel_events)
            slowest_kernel_duration = sum(e.dur for e in slowest_kernel_events)
            kernel_duration_diff = slowest_kernel_duration - fastest_kernel_duration
            
            # 计算时间差异占all2all duration差异的比例
            all2all_duration_diff = slowest_duration - fastest_duration
            cpu_start_time_diff_ratio = cpu_start_time_diff / all2all_duration_diff if all2all_duration_diff > 0 else 0
            kernel_duration_diff_ratio = kernel_duration_diff / all2all_duration_diff if all2all_duration_diff > 0 else 0
            
            # 构建行数据
            # 根据show参数决定展示的时间格式和列名
            if show_readable_timestamp:
                # 使用readable_timestamp转换后的值
                fastest_display_time = _readable_timestamp_to_microseconds(fastest_cpu_event.readable_timestamp)
                slowest_display_time = _readable_timestamp_to_microseconds(slowest_cpu_event.readable_timestamp)
                fastest_time_col = 'fastest_cpu_start_time_readable'
                slowest_time_col = 'slowest_cpu_start_time_readable'
            else:
                # 使用原始ts值（适合chrome tracing显示）
                fastest_display_time = fastest_cpu_event.ts
                slowest_display_time = slowest_cpu_event.ts
                fastest_time_col = 'fastest_cpu_start_time'
                slowest_time_col = 'slowest_cpu_start_time'
            
            row = {
                'event_sequence': i + 1,
                'cpu_op_name': fastest_cpu_event.name,
                'slowest_cpu_op_name': slowest_cpu_event.name,
                'cpu_op_shape': str(fastest_cpu_event.args.get('Input Dims', '') if fastest_cpu_event.args else ''),
                'slowest_cpu_op_shape': str(slowest_cpu_event.args.get('Input Dims', '') if slowest_cpu_event.args else ''),
                'cpu_op_dtype': str(fastest_cpu_event.args.get('Input type', '') if fastest_cpu_event.args else ''),
                'slowest_cpu_op_dtype': str(slowest_cpu_event.args.get('Input type', '') if slowest_cpu_event.args else ''),
                fastest_time_col: fastest_display_time,
                slowest_time_col: slowest_display_time,
                'cpu_start_time_diff': cpu_start_time_diff,
                'cpu_start_time_diff_ratio': cpu_start_time_diff_ratio,
                'fastest_cpu_duration': fastest_cpu_event.dur,
                'slowest_cpu_duration': slowest_cpu_event.dur,
                'cpu_duration_diff': cpu_duration_diff,
                'fastest_kernel_duration': fastest_kernel_duration,
                'slowest_kernel_duration': slowest_kernel_duration,
                'kernel_duration_diff': kernel_duration_diff,
                'kernel_duration_diff_ratio': kernel_duration_diff_ratio,
                'fastest_card_idx': fastest_card_idx,
                'slowest_card_idx': slowest_card_idx,
                'fastest_cpu_readable_timestamp': fastest_cpu_event.readable_timestamp,
                'slowest_cpu_readable_timestamp': slowest_cpu_event.readable_timestamp
            }
            
            comparison_rows.append(row)
        
        print(f"成功比较了 {len(comparison_rows)} 个CPU events")
        
        # 分析kernel duration ratio - 找出前5个最大的
        print("\n=== Kernel Duration Ratio 分析 ===")
        top_kernel_duration_ratios = self.find_top_kernel_duration_ratios(comparison_rows, top_n=5)
        print(f"找到 {len(top_kernel_duration_ratios)} 个最大的kernel duration ratio事件:")
        for i, (idx, ratio, row) in enumerate(top_kernel_duration_ratios):
            print(f"  {i+1}. 事件序列 {idx+1}:")
            print(f"     最快Card CPU操作: {row['cpu_op_name']}")
            print(f"     最慢Card CPU操作: {row.get('slowest_cpu_op_name', 'N/A')}")
            print(f"     最快Card Timestamp: {row['fastest_cpu_readable_timestamp']} (ts: {row.get('fastest_cpu_start_time', 'N/A')})")
            print(f"     最慢Card Timestamp: {row['slowest_cpu_readable_timestamp']} (ts: {row.get('slowest_cpu_start_time', 'N/A')})")
            print(f"     Kernel Duration Ratio: {ratio:.4f}")
            print(f"     最快Card操作形状: {row['cpu_op_shape']}, 类型: {row['cpu_op_dtype']}")
            print(f"     最慢Card操作形状: {row.get('slowest_cpu_op_shape', 'N/A')}, 类型: {row.get('slowest_cpu_op_dtype', 'N/A')}")
            print()
        
        # 分析cpu start time相邻差值 - 找出前5对最大的
        print("\n=== CPU Start Time 相邻差值分析 ===")
        top_cpu_start_time_differences = self.find_top_cpu_start_time_differences(comparison_rows, top_n=5)
        print(f"找到 {len(top_cpu_start_time_differences)} 对最大的CPU start time相邻差值:")
        for i, diff_info in enumerate(top_cpu_start_time_differences):
            prev_idx, current_idx = diff_info['index_pair']
            prev_row = diff_info['prev_row']
            current_row = diff_info['current_row']
            print(f"  {i+1}. 事件对 ({prev_idx+1}, {current_idx+1}):")
            print(f"     前一个事件:")
            print(f"       最快Card CPU操作: {prev_row['cpu_op_name']}")
            print(f"       最慢Card CPU操作: {prev_row.get('slowest_cpu_op_name', 'N/A')}")
            print(f"       最快Card Timestamp: {prev_row['fastest_cpu_readable_timestamp']} (ts: {prev_row.get('fastest_cpu_start_time', 'N/A')})")
            print(f"       最慢Card Timestamp: {prev_row['slowest_cpu_readable_timestamp']} (ts: {prev_row.get('slowest_cpu_start_time', 'N/A')})")
            print(f"       Ratio: {diff_info['prev_ratio']:.4f}")
            print(f"     当前事件:")
            print(f"       最快Card CPU操作: {current_row['cpu_op_name']}")
            print(f"       最慢Card CPU操作: {current_row.get('slowest_cpu_op_name', 'N/A')}")
            print(f"       最快Card Timestamp: {current_row['fastest_cpu_readable_timestamp']} (ts: {current_row.get('fastest_cpu_start_time', 'N/A')})")
            print(f"       最慢Card Timestamp: {current_row['slowest_cpu_readable_timestamp']} (ts: {current_row.get('slowest_cpu_start_time', 'N/A')})")
            print(f"       Ratio: {diff_info['current_ratio']:.4f}")
            print(f"     相邻差值: {diff_info['difference']:.4f}")
            print()
        
        return {
            'comparison_rows': comparison_rows,
            'top_kernel_duration_ratios': top_kernel_duration_ratios,
            'top_cpu_start_time_differences': top_cpu_start_time_differences
        }
    
    def _find_kernel_events_for_cpu_event(self, cpu_event, events_by_external_id):
        """为CPU event找到对应的kernel events"""
        if cpu_event.external_id is None:
            return []
        
        if cpu_event.external_id not in events_by_external_id:
            return []
        
        events = events_by_external_id[cpu_event.external_id]
        kernel_events = [e for e in events if e.cat == 'kernel']
        
        return kernel_events
    
    def _check_cpu_ops_consistency(self, fastest_cpu_events, slowest_cpu_events):
        """检查快卡和慢卡的CPU操作是否一致"""
        if len(fastest_cpu_events) != len(slowest_cpu_events):
            print(f"CPU events数量不一致: 最快card {len(fastest_cpu_events)} 个, 最慢card {len(slowest_cpu_events)} 个")
            return False
        
        # 检查每个位置的CPU操作是否一致
        for i, (fastest_event, slowest_event) in enumerate(zip(fastest_cpu_events, slowest_cpu_events)):
            # 比较操作名称
            if fastest_event.name != slowest_event.name:
                print(f"位置 {i}: CPU操作名称不一致 - 最快: {fastest_event.name}, 最慢: {slowest_event.name}")
                return False
            
            # 比较操作参数
            fastest_args = fastest_event.args or {}
            slowest_args = slowest_event.args or {}
            
            # 比较Input Dims
            fastest_dims = fastest_args.get('Input Dims', [])
            slowest_dims = slowest_args.get('Input Dims', [])
            if fastest_dims != slowest_dims:
                print(f"位置 {i}: Input Dims不一致 - 最快: {fastest_dims}, 最慢: {slowest_dims}")
                return False
        
        print(f"CPU操作一致性检查通过: 共 {len(fastest_cpu_events)} 个操作")
        return True
    
    def _check_kernel_ops_consistency(self, fastest_events_by_external_id, slowest_events_by_external_id):
        """检查快卡和慢卡的kernel操作是否一致"""
        # 提取所有kernel events
        fastest_kernel_events = []
        slowest_kernel_events = []
        
        for events in fastest_events_by_external_id.values():
            fastest_kernel_events.extend([e for e in events if e.cat == 'kernel'])
        for events in slowest_events_by_external_id.values():
            slowest_kernel_events.extend([e for e in events if e.cat == 'kernel'])
        
        # 按结束时间排序
        fastest_kernel_events.sort(key=lambda x: (x.ts + x.dur) if (x.ts is not None and x.dur is not None) else 0)
        slowest_kernel_events.sort(key=lambda x: (x.ts + x.dur) if (x.ts is not None and x.dur is not None) else 0)
        
        if len(fastest_kernel_events) != len(slowest_kernel_events):
            print(f"Kernel events数量不一致: 最快card {len(fastest_kernel_events)} 个, 最慢card {len(slowest_kernel_events)} 个")
            return False
        
        # 检查每个位置的kernel操作是否一致
        for i, (fastest_event, slowest_event) in enumerate(zip(fastest_kernel_events, slowest_kernel_events)):
            # 比较kernel名称
            if fastest_event.name != slowest_event.name:
                print(f"位置 {i}: Kernel名称不一致 - 最快: {fastest_event.name}, 最慢: {slowest_event.name}")
                return False
        
        print(f"Kernel操作一致性检查通过: 共 {len(fastest_kernel_events)} 个操作")
        return True
    
    def detect_change_points(self, data, ratio_column, threshold=0.3):
        """检测数据中的突变点"""
        if len(data) < 3:
            return []
        
        change_points = []
        ratios = [row.get(ratio_column, 0) for row in data]
        
        # 计算一阶导数（差分）
        for i in range(1, len(ratios) - 1):
            # 计算前向和后向的差分
            prev_diff = ratios[i] - ratios[i-1]
            next_diff = ratios[i+1] - ratios[i]
            
            # 如果前向和后向差分符号相反且绝对值都超过阈值，认为是突变点
            if (prev_diff * next_diff < 0) and (abs(prev_diff) > threshold or abs(next_diff) > threshold):
                change_points.append(i)
        
        # 也检测边界处的突变
        if len(ratios) >= 2:
            # 检测第一个点
            if abs(ratios[1] - ratios[0]) > threshold:
                change_points.append(0)
            # 检测最后一个点
            if abs(ratios[-1] - ratios[-2]) > threshold:
                change_points.append(len(ratios) - 1)
        
        return sorted(list(set(change_points)))
    
    def find_top_kernel_duration_ratios(self, comparison_rows, top_n=5):
        """找出kernel duration ratio最大的前N个事件"""
        if not comparison_rows:
            return []
        
        # 提取kernel_duration_diff_ratio并添加索引
        kernel_ratios_with_index = []
        for i, row in enumerate(comparison_rows):
            ratio = row.get('kernel_duration_diff_ratio', 0)
            kernel_ratios_with_index.append((i, ratio, row))
        
        # 按ratio降序排序
        kernel_ratios_with_index.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前N个
        return kernel_ratios_with_index[:top_n]
    
    def find_top_cpu_start_time_differences(self, comparison_rows, top_n=5):
        """找出cpu start time相邻差值最大的前N对"""
        if len(comparison_rows) < 2:
            return []
        
        # 计算相邻行的cpu_start_time_diff_ratio差值
        differences = []
        for i in range(1, len(comparison_rows)):
            current_ratio = comparison_rows[i].get('cpu_start_time_diff_ratio', 0)
            prev_ratio = comparison_rows[i-1].get('cpu_start_time_diff_ratio', 0)
            diff = abs(current_ratio - prev_ratio)
            
            differences.append({
                'index_pair': (i-1, i),
                'difference': diff,
                'prev_ratio': prev_ratio,
                'current_ratio': current_ratio,
                'prev_row': comparison_rows[i-1],
                'current_row': comparison_rows[i]
            })
        
        # 按差值降序排序
        differences.sort(key=lambda x: x['difference'], reverse=True)
        
        # 返回前N对
        return differences[:top_n]
    
    def _print_combined_change_points(self, comparison_rows, cpu_start_time_change_points, kernel_duration_change_points):
        """打印合并的突变点信息，按照操作执行顺序排列"""
        print("\n=== 突变点详情（按执行顺序） ===")
        
        # 合并所有突变点并按索引排序
        all_change_points = sorted(list(set(cpu_start_time_change_points + kernel_duration_change_points)))
        
        if not all_change_points:
            print("未发现突变点")
            return
        
        for cp_idx in all_change_points:
            if cp_idx >= len(comparison_rows):
                continue
                
            row = comparison_rows[cp_idx]
            is_cpu_change = cp_idx in cpu_start_time_change_points
            is_kernel_change = cp_idx in kernel_duration_change_points
            
            # 确定突变类型
            change_types = []
            if is_cpu_change:
                change_types.append("CPU操作")
            if is_kernel_change:
                change_types.append("Kernel操作")
            
            change_type_str = " + ".join(change_types)
            
            print(f"\n突变点 #{cp_idx + 1} ({change_type_str}):")
            print(f"  操作名称: {row.get('cpu_op_name', 'N/A')}")
            print(f"  操作形状: {row.get('cpu_op_shape', 'N/A')}")
            print(f"  操作类型: {row.get('cpu_op_dtype', 'N/A')}")
            
            if is_cpu_change:
                print(f"  CPU启动时间差异: {row.get('cpu_start_time_diff', 0):.2f}μs")
                print(f"  CPU启动时间差异比例: {row.get('cpu_start_time_diff_ratio', 0):.3f}")
            
            if is_kernel_change:
                print(f"  Kernel持续时间差异: {row.get('kernel_duration_diff', 0):.2f}μs")
                print(f"  Kernel持续时间差异比例: {row.get('kernel_duration_diff_ratio', 0):.3f}")
            
            print(f"  最快卡启动时间: {row.get('fastest_cpu_start_time', 0):.2f}μs")
            print(f"  最慢卡启动时间: {row.get('slowest_cpu_start_time', 0):.2f}μs")
    
    def extract_change_point_data(self, data, change_points, context_size=3):
        """提取突变点前后的数据"""
        if not change_points:
            return []
        
        change_point_data = []
        
        for cp_idx in change_points:
            if cp_idx >= len(data):
                continue
            
            # 计算上下文范围
            start_idx = max(0, cp_idx - context_size)
            end_idx = min(len(data), cp_idx + context_size + 1)
            
            # 提取上下文数据
            context_data = data[start_idx:end_idx]
            
            # 标记突变点
            for i, row in enumerate(context_data):
                row_copy = row.copy()
                row_copy['is_change_point'] = (start_idx + i) == cp_idx
                row_copy['context_position'] = i - context_size if i >= context_size else i
                change_point_data.append(row_copy)
        
        return change_point_data
    
    def _scan_executor_folders(self, pod_dir: str) -> List[str]:
        """
        扫描Pod目录下的executor文件夹
        
        Args:
            pod_dir: Pod目录路径
            
        Returns:
            List[str]: executor文件夹路径列表
        """
        executor_folders = []
        
        for item in os.listdir(pod_dir):
            item_path = os.path.join(pod_dir, item)
            if os.path.isdir(item_path) and item.startswith('executor_trainer-runner_'):
                executor_folders.append(item_path)
        
        return sorted(executor_folders)
    
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
    
    def _parse_json_filename(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """
        解析JSON文件名，提取step和card索引
        
        Args:
            filename: JSON文件名
            
        Returns:
            Tuple[Optional[int], Optional[int]]: (step, card_idx)
        """
        # 匹配多种文件名模式
        patterns = [
            r'(\d+)_(\d+)\.json',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    card_idx = int(match.group(1))
                    step = int(match.group(2))
                    return step, card_idx
                elif len(match.groups()) == 1:
                    # 只有step信息，card_idx设为0
                    step = int(match.group(1))
                    return step, 0
        
        return None, None
    
    def _extract_communication_durations(self, json_file_path: str, kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL") -> List[float]:
        """
        从JSON文件中提取指定通信kernel前缀的duration
        
        Args:
            json_file_path: JSON文件路径
            kernel_prefix: 要检测的通信kernel前缀
            
        Returns:
            List[float]: duration列表，最多6个
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用正则表达式匹配指定的通信kernel前缀
            # 支持多种kernel前缀，包括TCDP_开头的各种通信操作
            escaped_kernel_prefix = re.escape(kernel_prefix)
            pattern = rf'"ph":\s*"X",\s*"cat":\s*"kernel",\s*"name":\s*"{escaped_kernel_prefix}[^"]*",\s*"pid":\s*\d+,\s*"tid":\s*\d+,\s*"ts":\s*[\d.]+,\s*"dur":\s*([\d.]+)'
            
            matches = re.findall(pattern, content, re.DOTALL)
            
            # 转换为浮点数，最多取6个
            durations = [float(match) for match in matches[:6]]
            
            return durations
            
        except Exception as e:
            print(f"    读取JSON文件失败 {json_file_path}: {e}")
            return []
    
    def _generate_raw_data_excel(self, all2all_data: Dict[int, Dict[int, List[float]]], output_dir: str) -> Path:
        """生成原始数据Excel文件"""
        return self._generate_excel_file(
            all2all_data, output_dir, "communication_raw_data.xlsx", 
            self._create_raw_data_sheets
        )
    
    def _generate_statistics_excel(self, all2all_data: Dict[int, Dict[int, List[float]]], output_dir: str) -> Path:
        """生成统计信息Excel文件"""
        return self._generate_excel_file(
            all2all_data, output_dir, "communication_statistics.xlsx",
            self._create_statistics_sheets
        )
    
    def _generate_excel_file(self, data, output_dir: str, filename: str, sheet_creator) -> Path:
        """通用的Excel文件生成方法"""
        if pd is None:
            raise ImportError("pandas is required for Excel output. Please install pandas and openpyxl.")
        
        output_path = Path(output_dir) / filename
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            sheet_creator(data, writer)
        
        print(f"生成文件: {output_path}")
        return output_path
    
    def _create_raw_data_sheets(self, all2all_data, writer):
        """创建原始数据表格"""
        for step, card_data in all2all_data.items():
            rows = []
            for card_idx, durations in card_data.items():
                for i, duration in enumerate(durations):
                    rows.append({
                        'Step': step, 'Card_Index': card_idx, 
                        'Comm_Index': i, 'Duration_us': duration
                    })
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=f'Step_{step}', index=False)
    
    def _create_statistics_sheets(self, all2all_data, writer):
        """创建统计信息表格"""
        # Step统计
        step_stats = self._calculate_step_statistics(all2all_data)
        if step_stats:
            pd.DataFrame(step_stats).to_excel(writer, sheet_name='Step_Statistics', index=False)
        
        # Card统计
        card_stats = self._calculate_card_statistics(all2all_data)
        if card_stats:
            pd.DataFrame(card_stats).to_excel(writer, sheet_name='Card_Statistics', index=False)
    
    def _calculate_step_statistics(self, all2all_data):
        """计算step级别统计信息"""
        step_stats = []
        for step, card_data in all2all_data.items():
            all_durations = [d for durations in card_data.values() for d in durations]
            if all_durations:
                mean_dur = sum(all_durations) / len(all_durations)
                step_stats.append({
                    'Step': step, 'Total_Cards': len(card_data), 'Total_Comm_Ops': len(all_durations),
                    'Min_Duration_us': min(all_durations), 'Max_Duration_us': max(all_durations),
                    'Mean_Duration_us': mean_dur,
                    'Std_Duration_us': (sum((x - mean_dur)**2 for x in all_durations) / len(all_durations))**0.5
                })
        return step_stats
    
    def _calculate_card_statistics(self, all2all_data):
        """计算card级别统计信息"""
        card_stats = []
        for step, card_data in all2all_data.items():
            for card_idx, durations in card_data.items():
                if durations:
                    mean_dur = sum(durations) / len(durations)
                    card_stats.append({
                        'Step': step, 'Card_Index': card_idx, 'Comm_Ops_Count': len(durations),
                        'Min_Duration_us': min(durations), 'Max_Duration_us': max(durations),
                        'Mean_Duration_us': mean_dur,
                        'Std_Duration_us': (sum((x - mean_dur)**2 for x in durations) / len(durations))**0.5
                    })
        return card_stats
    
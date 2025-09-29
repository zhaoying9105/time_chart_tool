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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ...parser import PyTorchProfilerParser
from ...models import ActivityEvent, ProfilerData


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
        # 为每个文件创建独立的parser实例，避免调用栈树被覆盖
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
            step, comm_idx, fastest_card[1], slowest_card[1], kernel_prefix, prev_kernel_pattern,
            show_timestamp, show_readable_timestamp, output_dir
        )
        
        if not comparison_result:
            print("错误: 深度对比分析失败")
            return None
        
        # 6. 生成深度分析Excel文件
        excel_result = self._generate_deep_analysis_excel(comparison_result, step, comm_idx, output_dir)
        
        # 7. 添加可视化文件到返回结果
        if isinstance(excel_result, tuple):
            # 返回了两个文件（主文件 + CPU Start Time分析文件）
            excel_file, cpu_excel_file = excel_result
            generated_files = [excel_file, cpu_excel_file]
        else:
            # 只返回了一个文件
            generated_files = [excel_result]
            
        if 'visualization_files' in comparison_result:
            generated_files.extend(comparison_result['visualization_files'])
        
        print("=== 深度分析完成 ===")
        return generated_files
    
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
                                 show_readable_timestamp: bool = False, output_dir: str = "."):
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
            fastest_data, slowest_data, show_timestamp, show_readable_timestamp,
            output_dir, step, comm_idx
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
        
        # 生成单独的CPU Start Time相邻差值分析Excel文件
        if 'top_cpu_start_time_differences' in comparison_result and comparison_result['top_cpu_start_time_differences']:
            cpu_excel_file = output_path / f"cpu_start_time_differences_step_{step}_idx_{comm_idx}.xlsx"
            
            with pd.ExcelWriter(cpu_excel_file, engine='openpyxl') as writer:
                # 写入CPU start time相邻差值分析数据
                cpu_data = []
                for i, diff_info in enumerate(comparison_result['top_cpu_start_time_differences']):
                    prev_row = diff_info['prev_row']
                    current_row = diff_info['current_row']
                    
                    # 获取调用栈信息
                    prev_fastest_call_stack = []
                    prev_slowest_call_stack = []
                    current_fastest_call_stack = []
                    current_slowest_call_stack = []
                    
                    if diff_info.get('prev_fastest_event'):
                        prev_fastest_call_stack = diff_info['prev_fastest_event'].get_call_stack('tree') or []
                    if diff_info.get('prev_slowest_event'):
                        prev_slowest_call_stack = diff_info['prev_slowest_event'].get_call_stack('tree') or []
                    if diff_info.get('current_fastest_event'):
                        current_fastest_call_stack = diff_info['current_fastest_event'].get_call_stack('tree') or []
                    if diff_info.get('current_slowest_event'):
                        current_slowest_call_stack = diff_info['current_slowest_event'].get_call_stack('tree') or []
                    
                    # 计算最大调用栈长度
                    max_stack_len = max(len(prev_fastest_call_stack), len(prev_slowest_call_stack), 
                                      len(current_fastest_call_stack), len(current_slowest_call_stack))
                    
                    # 为每一行添加调用栈信息
                    for j in range(max_stack_len):
                        cpu_data.append({
                            '排名': i + 1 if j == 0 else '',
                            '事件对': f"{diff_info['index_pair'][0]+1}-{diff_info['index_pair'][1]+1}" if j == 0 else '',
                            '相邻差值': diff_info['difference'] if j == 0 else '',
                            '前一个最快Card操作名': prev_row['cpu_op_name'] if j == 0 else '',
                            '前一个最慢Card操作名': prev_row.get('slowest_cpu_op_name', 'N/A') if j == 0 else '',
                            '前一个Ratio': diff_info['prev_ratio'] if j == 0 else '',
                            '前一个最快Card时间(可读)': prev_row['fastest_cpu_readable_timestamp'] if j == 0 else '',
                            '前一个最慢Card时间(可读)': prev_row['slowest_cpu_readable_timestamp'] if j == 0 else '',
                            '当前最快Card操作名': current_row['cpu_op_name'] if j == 0 else '',
                            '当前最慢Card操作名': current_row.get('slowest_cpu_op_name', 'N/A') if j == 0 else '',
                            '当前Ratio': diff_info['current_ratio'] if j == 0 else '',
                            '当前最快Card时间(可读)': current_row['fastest_cpu_readable_timestamp'] if j == 0 else '',
                            '当前最慢Card时间(可读)': current_row['slowest_cpu_readable_timestamp'] if j == 0 else '',
                            '前一个最快Card调用栈': prev_fastest_call_stack[j] if j < len(prev_fastest_call_stack) else '',
                            '前一个最慢Card调用栈': prev_slowest_call_stack[j] if j < len(prev_slowest_call_stack) else '',
                            '当前最快Card调用栈': current_fastest_call_stack[j] if j < len(current_fastest_call_stack) else '',
                            '当前最慢Card调用栈': current_slowest_call_stack[j] if j < len(current_slowest_call_stack) else ''
                        })
                
                cpu_df = pd.DataFrame(cpu_data)
                cpu_df.to_excel(writer, sheet_name='CPU Start Time相邻差值分析', index=False)
            
            print(f"CPU Start Time相邻差值分析Excel文件已生成: {cpu_excel_file}")
            return excel_file, str(cpu_excel_file)
        
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
                                        fastest_data=None, slowest_data=None,
                                        show_timestamp: bool = False, show_readable_timestamp: bool = False,
                                        output_dir: str = ".", step: int = None, comm_idx: int = None):
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
        
        # 2. 使用LCS模糊对齐算法
        print("\n=== 执行LCS模糊对齐 ===")
        aligned_fastest, aligned_slowest, alignment_info = self.fuzzy_align_cpu_events(
            fastest_cpu_events, slowest_cpu_events
        )
        
        # 3. 生成对齐可视化结果
        print("\n=== 生成对齐可视化 ===")
        visualization_files = self.visualize_alignment_results(
            fastest_cpu_events, slowest_cpu_events,
            aligned_fastest, aligned_slowest, alignment_info,
            output_dir, step, comm_idx
        )
        
        # 4. 检查kernel操作是否一致
        kernel_ops_consistent = self._check_kernel_ops_consistency(fastest_events_by_external_id, slowest_events_by_external_id)
        if not kernel_ops_consistent:
            print("警告: 快卡和慢卡的kernel操作不完全一致，比较结果可能不准确")
        else:
            print("Kernel操作一致性检查通过")
        
        # 5. 使用对齐后的事件进行配对比较
        matched_events_count = len([info for info in alignment_info if info['match_type'] == 'exact_match'])
        
        for i in range(len(aligned_fastest)):
            fastest_cpu_event = aligned_fastest[i]
            slowest_cpu_event = aligned_slowest[i]
            alignment = alignment_info[i]
            
            # 跳过未匹配的事件
            if fastest_cpu_event is None or slowest_cpu_event is None:
                continue
            
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
                'slowest_cpu_readable_timestamp': slowest_cpu_event.readable_timestamp,
                # 对齐信息
                'alignment_type': alignment['match_type'],
                'alignment_fast_idx': alignment['fast_idx'],
                'alignment_slow_idx': alignment['slow_idx']
            }
            
            comparison_rows.append(row)
        
        print(f"成功比较了 {len(comparison_rows)} 个CPU events")
        
        # 对齐质量评估
        print("\n=== 对齐质量评估 ===")
        total_aligned = len(comparison_rows)
        exact_count = sum(1 for row in comparison_rows if row['alignment_type'] == 'exact_match')
        
        print(f"总对齐事件数: {total_aligned}")
        print(f"精确匹配: {exact_count} ({exact_count/total_aligned*100:.1f}%)")
        
        # 显示一些对齐示例
        print("\n=== 对齐示例 ===")
        matched_examples = [row for row in comparison_rows if row['alignment_type'] == 'exact_match'][:5]
        for i, row in enumerate(matched_examples):
            print(f"  {i+1}. 精确匹配: {row['cpu_op_name']} <-> {row['slowest_cpu_op_name']}")
        
        # 分析kernel duration ratio - 找出前5个最大的
        print("\n=== Kernel Duration Ratio 分析 ===")
        top_kernel_duration_ratios = self.find_top_kernel_duration_ratios(comparison_rows, top_n=5)
        print(f"找到 {len(top_kernel_duration_ratios)} 个最大的kernel duration ratio事件:")
        for i, (idx, ratio, row) in enumerate(top_kernel_duration_ratios):
            alignment_type_str = {
                'exact_match': '精确匹配',
                'unmatched_fast': '快卡未匹配',
                'unmatched_slow': '慢卡未匹配'
            }.get(row['alignment_type'], row['alignment_type'])
            
            print(f"  {i+1}. 事件序列 {idx+1} ({alignment_type_str}):")
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
        top_cpu_start_time_differences = self.find_top_cpu_start_time_differences(
            comparison_rows, aligned_fastest, aligned_slowest, top_n=5
        )
        print(f"找到 {len(top_cpu_start_time_differences)} 对最大的CPU start time相邻差值:")
        for i, diff_info in enumerate(top_cpu_start_time_differences):
            prev_idx, current_idx = diff_info['index_pair']
            prev_row = diff_info['prev_row']
            current_row = diff_info['current_row']
            
            prev_alignment_str = {
                'exact_match': '精确匹配',
                'unmatched_fast': '快卡未匹配',
                'unmatched_slow': '慢卡未匹配'
            }.get(prev_row['alignment_type'], prev_row['alignment_type'])
            
            current_alignment_str = {
                'exact_match': '精确匹配',
                'unmatched_fast': '快卡未匹配',
                'unmatched_slow': '慢卡未匹配'
            }.get(current_row['alignment_type'], current_row['alignment_type'])
            
            print(f"  {i+1}. 事件对 ({prev_idx+1}, {current_idx+1}):")
            print(f"     相邻差值: {diff_info['difference']:.4f}")
            print()
            
            # 四列竖排格式 - 扩展宽度以容纳长调用栈
            col_width = 50  # 每列宽度从30扩展到50
            total_width = col_width * 4 + 9  # 4列 + 3个分隔符(3个|) + 6个空格
            
            print("     " + "="*total_width)
            print("     " + f"{'最快Card前一个事件':<{col_width}} | {'最慢Card前一个事件':<{col_width}} | {'最快Card当前事件':<{col_width}} | {'最慢Card当前事件':<{col_width}}")
            print("     " + "="*total_width)
            
            # 第一行：事件名称
            prev_fastest_name = prev_row['cpu_op_name']
            prev_slowest_name = prev_row.get('slowest_cpu_op_name', 'N/A')
            current_fastest_name = current_row['cpu_op_name']
            current_slowest_name = current_row.get('slowest_cpu_op_name', 'N/A')
            
            print("     " + f"{prev_fastest_name:<{col_width}} | {prev_slowest_name:<{col_width}} | {current_fastest_name:<{col_width}} | {current_slowest_name:<{col_width}}")
            
            # 第二行：对齐类型
            print("     " + f"{prev_alignment_str:<{col_width}} | {prev_alignment_str:<{col_width}} | {current_alignment_str:<{col_width}} | {current_alignment_str:<{col_width}}")
            
            # 第三行：时间戳
            prev_fastest_ts = prev_row['fastest_cpu_readable_timestamp']
            prev_slowest_ts = prev_row['slowest_cpu_readable_timestamp']
            current_fastest_ts = current_row['fastest_cpu_readable_timestamp']
            current_slowest_ts = current_row['slowest_cpu_readable_timestamp']
            
            print("     " + f"{prev_fastest_ts:<{col_width}} | {prev_slowest_ts:<{col_width}} | {current_fastest_ts:<{col_width}} | {current_slowest_ts:<{col_width}}")
            
            # 第四行：Ratio
            print("     " + f"Ratio: {diff_info['prev_ratio']:.4f}{'':<{col_width-12}} | {'':<{col_width}} | Ratio: {diff_info['current_ratio']:.4f}{'':<{col_width-12}} | {'':<{col_width}}")
            
            # 第五行：事件名称匹配验证
            prev_name_match = "✅ 匹配" if prev_fastest_name == prev_slowest_name else f"❌ 不匹配"
            current_name_match = "✅ 匹配" if current_fastest_name == current_slowest_name else f"❌ 不匹配"
            
            print("     " + f"{prev_name_match:<{col_width}} | {'':<{col_width}} | {current_name_match:<{col_width}} | {'':<{col_width}}")
            
            # 调用栈信息
            print("     " + "-"*total_width)
            print("     " + f"{'调用栈 (from tree)':<{col_width}} | {'调用栈 (from tree)':<{col_width}} | {'调用栈 (from tree)':<{col_width}} | {'调用栈 (from tree)':<{col_width}}")
            print("     " + "-"*total_width)
            
            # 获取调用栈信息
            prev_fastest_call_stack = []
            prev_slowest_call_stack = []
            current_fastest_call_stack = []
            current_slowest_call_stack = []
            
            if diff_info.get('prev_fastest_event'):
                prev_fastest_call_stack = diff_info['prev_fastest_event'].get_call_stack('tree') or []
            if diff_info.get('prev_slowest_event'):
                prev_slowest_call_stack = diff_info['prev_slowest_event'].get_call_stack('tree') or []
            if diff_info.get('current_fastest_event'):
                current_fastest_call_stack = diff_info['current_fastest_event'].get_call_stack('tree') or []
            if diff_info.get('current_slowest_event'):
                current_slowest_call_stack = diff_info['current_slowest_event'].get_call_stack('tree') or []
            
            # 计算最大调用栈长度
            max_stack_len = max(len(prev_fastest_call_stack), len(prev_slowest_call_stack), 
                              len(current_fastest_call_stack), len(current_slowest_call_stack))
            
            if max_stack_len == 0:
                print("     " + f"{'无调用栈':<{col_width}} | {'无调用栈':<{col_width}} | {'无调用栈':<{col_width}} | {'无调用栈':<{col_width}}")
            else:
                for j in range(max_stack_len):
                    prev_fastest_frame = prev_fastest_call_stack[j] if j < len(prev_fastest_call_stack) else ""
                    prev_slowest_frame = prev_slowest_call_stack[j] if j < len(prev_slowest_call_stack) else ""
                    current_fastest_frame = current_fastest_call_stack[j] if j < len(current_fastest_call_stack) else ""
                    current_slowest_frame = current_slowest_call_stack[j] if j < len(current_slowest_call_stack) else ""
                    
                    print("     " + f"{prev_fastest_frame:<{col_width}} | {prev_slowest_frame:<{col_width}} | {current_fastest_frame:<{col_width}} | {current_slowest_frame:<{col_width}}")
            
            print("     " + "="*total_width)
            
            # 打印两个事件之间的调用栈分析（只有当两个事件都不是None时才分析）
            prev_fastest = diff_info.get('prev_fastest_event')
            current_fastest = diff_info.get('current_fastest_event')
            prev_slowest = diff_info.get('prev_slowest_event')
            current_slowest = diff_info.get('current_slowest_event')
            
            if prev_fastest and current_fastest:
                self._print_events_between_call_stacks(
                    prev_fastest_event=prev_fastest,
                    current_fastest_event=current_fastest,
                    prev_slowest_event=prev_slowest,
                    current_slowest_event=current_slowest,
                    fastest_parser=self.fastest_parser,
                    slowest_parser=self.slowest_parser
                )
            else:
                print("[信息] 跳过调用栈分析：存在None事件")
            
            print()
        
        return {
            'comparison_rows': comparison_rows,
            'top_kernel_duration_ratios': top_kernel_duration_ratios,
            'top_cpu_start_time_differences': top_cpu_start_time_differences,
            'visualization_files': visualization_files
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
    
    def find_top_cpu_start_time_differences(self, comparison_rows, aligned_fastest=None, aligned_slowest=None, top_n=5):
        """找出cpu start time相邻差值最大的前N对"""
        if len(comparison_rows) < 2:
            return []
        
        # 计算相邻行的cpu_start_time_diff_ratio差值
        differences = []
        for i in range(1, len(comparison_rows)):
            current_ratio = comparison_rows[i].get('cpu_start_time_diff_ratio', 0)
            prev_ratio = comparison_rows[i-1].get('cpu_start_time_diff_ratio', 0)
            diff = abs(current_ratio - prev_ratio)
            
            # 获取对应的事件对象
            prev_fastest_event = None
            current_fastest_event = None
            prev_slowest_event = None
            current_slowest_event = None
            
            if aligned_fastest and aligned_slowest:
                # 使用 event_sequence 作为对齐后列表的索引
                prev_event_sequence = comparison_rows[i-1].get('event_sequence')
                current_event_sequence = comparison_rows[i].get('event_sequence')
                
                if prev_event_sequence is not None and current_event_sequence is not None:
                    # event_sequence 从1开始，需要减1转换为0基索引
                    prev_idx = prev_event_sequence - 1
                    current_idx = current_event_sequence - 1
                    
                    if prev_idx < len(aligned_fastest) and prev_idx < len(aligned_slowest):
                        prev_fastest_event = aligned_fastest[prev_idx]
                        prev_slowest_event = aligned_slowest[prev_idx]
                        
                        # 检查是否为None事件
                        if prev_fastest_event is None:
                            print(f"[调试] prev_fastest_event is None at index {prev_idx}")
                            continue
                    
                    if current_idx < len(aligned_fastest) and current_idx < len(aligned_slowest):
                        current_fastest_event = aligned_fastest[current_idx]
                        current_slowest_event = aligned_slowest[current_idx]
                        
                        # 检查是否为None事件
                        if current_fastest_event is None:
                            print(f"[调试] current_fastest_event is None at index {current_idx}")
                            continue
            
            # 检查事件名称一致性
            if prev_fastest_event and prev_slowest_event:
                if prev_fastest_event.name != prev_slowest_event.name:
                    print(f"[警告] find_top_cpu_start_time_differences: 前一个事件名称不匹配: '{prev_fastest_event.name}' vs '{prev_slowest_event.name}'")
            
            if current_fastest_event and current_slowest_event:
                if current_fastest_event.name != current_slowest_event.name:
                    print(f"[警告] find_top_cpu_start_time_differences: 当前事件名称不匹配: '{current_fastest_event.name}' vs '{current_slowest_event.name}'")
            
            differences.append({
                'index_pair': (i-1, i),
                'difference': diff,
                'prev_ratio': prev_ratio,
                'current_ratio': current_ratio,
                'prev_row': comparison_rows[i-1],
                'current_row': comparison_rows[i],
                # 添加事件对象
                'prev_fastest_event': prev_fastest_event,
                'current_fastest_event': current_fastest_event,
                'prev_slowest_event': prev_slowest_event,
                'current_slowest_event': current_slowest_event
            })
        
        # 按差值降序排序
        differences.sort(key=lambda x: x['difference'], reverse=True)
        
        # 返回前N对
        return differences[:top_n]
    
    def fuzzy_align_cpu_events(self, fastest_cpu_events, slowest_cpu_events):
        """
        基于LCS的模糊对齐CPU事件
        使用操作名称 + call stack 进行匹配，提高对齐精度
        
        Args:
            fastest_cpu_events: 最快卡的CPU事件列表
            slowest_cpu_events: 最慢卡的CPU事件列表
            
        Returns:
            tuple: (aligned_fastest, aligned_slowest, alignment_info)
        """
        if not fastest_cpu_events or not slowest_cpu_events:
            return [], [], []
        
        print(f"开始LCS模糊对齐: 快卡{len(fastest_cpu_events)}个事件, 慢卡{len(slowest_cpu_events)}个事件")
        
        # 1. 构建操作名称 + call stack 序列
        fast_signatures = [self._build_event_signature(e) for e in fastest_cpu_events]
        slow_signatures = [self._build_event_signature(e) for e in slowest_cpu_events]
        
        # 2. 计算LCS矩阵
        lcs_matrix = self._compute_lcs_matrix(fast_signatures, slow_signatures)
        
        # 3. 回溯LCS路径，构建对齐方案
        alignment = self._backtrack_lcs(fast_signatures, slow_signatures, lcs_matrix)
        
        # 4. 构建最终对齐结果
        aligned_fastest, aligned_slowest, alignment_info = self._build_aligned_events(
            alignment, fastest_cpu_events, slowest_cpu_events
        )
        
        # 5. 检查对齐结果中相同位置的事件名称是否一致
        print("检查对齐结果中事件名称一致性...")
        name_mismatches = 0
        for i, (fast_event, slow_event) in enumerate(zip(aligned_fastest, aligned_slowest)):
            if fast_event is not None and slow_event is not None:
                if fast_event.name != slow_event.name:
                    print(f"[警告] fuzzy_align_cpu_events: 位置 {i} 事件名称不匹配: '{fast_event.name}' vs '{slow_event.name}'")
                    name_mismatches += 1
        
        if name_mismatches == 0:
            print("✅ 对齐结果中所有事件名称都匹配")
        else:
            print(f"❌ 发现 {name_mismatches} 个事件名称不匹配")
        
        # 6. 统计对齐质量
        exact_matches = sum(1 for info in alignment_info if info['match_type'] == 'exact_match')
        unmatched_fast = sum(1 for info in alignment_info if info['match_type'] == 'unmatched_fast')
        unmatched_slow = sum(1 for info in alignment_info if info['match_type'] == 'unmatched_slow')
        
        print(f"LCS对齐完成: 精确匹配 {exact_matches} 个, 快卡未匹配 {unmatched_fast} 个, 慢卡未匹配 {unmatched_slow} 个")
        print(f"匹配率: {exact_matches / len(alignment_info) * 100:.1f}%")
        
        return aligned_fastest, aligned_slowest, alignment_info
    
    def _build_event_signature(self, event):
        """
        构建事件的签名，包含操作名称和过滤后的call stack
        
        Args:
            event: CPU事件对象
            
        Returns:
            str: 事件签名
        """
        # 获取操作名称
        name = event.name if event.name else ""
        return name
        
        # 获取call stack
        call_stack = event.get_call_stack("tree")
        if not call_stack:
            return name
        
        # 过滤call stack，去掉地址信息
        filtered_call_stack = self._filter_call_stack(call_stack)
        
        # 构建签名：操作名称 + 过滤后的call stack
        signature = name
        if filtered_call_stack:
            # 将call stack转换为字符串，用特殊分隔符连接
            call_stack_str = " | ".join(filtered_call_stack)
            signature = f"{name} [{call_stack_str}]"
        
        return signature
    
    def _filter_call_stack(self, call_stack):
        """
        过滤call stack，去掉地址信息
        
        Args:
            call_stack: call stack列表
            
        Returns:
            list: 过滤后的call stack
        """
        if not call_stack:
            return []
        
        import re
        filtered = []
        
        for frame in call_stack:
            if not frame:
                continue
            
            # 去掉 "object at 0x*****" 这种形式的地址信息
            # 匹配模式：object at 0x后跟十六进制数字
            filtered_frame = re.sub(r'object at 0x[0-9a-fA-F]+', 'object', frame)
            
            # 去掉其他可能的地址信息
            # 匹配模式：0x后跟十六进制数字
            filtered_frame = re.sub(r'0x[0-9a-fA-F]+', '0x****', filtered_frame)
            
            # 去掉行号信息（可选，根据需要调整）
            # filtered_frame = re.sub(r':\d+\)', ')', filtered_frame)
            
            filtered.append(filtered_frame)
        
        return filtered
    
    def _compute_lcs_matrix(self, seq1, seq2):
        """计算LCS矩阵"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:  # 事件签名完全相等
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp
    
    def _backtrack_lcs(self, fast_signatures, slow_signatures, lcs_matrix):
        """回溯LCS路径，构建对齐方案"""
        alignment = []
        i, j = len(fast_signatures), len(slow_signatures)
        
        while i > 0 and j > 0:
            if fast_signatures[i-1] == slow_signatures[j-1]:
                # 精确匹配
                alignment.append((i-1, j-1, 'exact_match'))
                i -= 1
                j -= 1
            elif lcs_matrix[i-1][j] > lcs_matrix[i][j-1]:
                # 快卡事件未匹配
                alignment.append((i-1, None, 'unmatched_fast'))
                i -= 1
            else:
                # 慢卡事件未匹配
                alignment.append((None, j-1, 'unmatched_slow'))
                j -= 1
        
        # 处理剩余事件
        while i > 0:
            alignment.append((i-1, None, 'unmatched_fast'))
            i -= 1
        while j > 0:
            alignment.append((None, j-1, 'unmatched_slow'))
            j -= 1
        
        return alignment[::-1]  # 反转得到正确顺序
    
    def _build_aligned_events(self, alignment, fastest_events, slowest_events):
        """构建对齐后的事件序列"""
        aligned_fastest = []
        aligned_slowest = []
        alignment_info = []
        
        for fast_idx, slow_idx, match_type in alignment:
            if fast_idx is not None and slow_idx is not None:
                # 匹配成功 - 检查事件名称是否一致
                fast_event = fastest_events[fast_idx]
                slow_event = slowest_events[slow_idx]
                
                if fast_event.name != slow_event.name:
                    print(f"[警告] _build_aligned_events: exact_match 事件名称不匹配: '{fast_event.name}' vs '{slow_event.name}'")
                    # 将不匹配的事件标记为 unmatched
                    aligned_fastest.append(fast_event)
                    aligned_slowest.append(None)
                    alignment_info.append({
                        'match_type': 'unmatched_fast',
                        'fast_idx': fast_idx,
                        'slow_idx': None,
                        'fast_name': fast_event.name,
                        'slow_name': None
                    })
                    
                    aligned_fastest.append(None)
                    aligned_slowest.append(slow_event)
                    alignment_info.append({
                        'match_type': 'unmatched_slow',
                        'fast_idx': None,
                        'slow_idx': slow_idx,
                        'fast_name': None,
                        'slow_name': slow_event.name
                    })
                else:
                    aligned_fastest.append(fast_event)
                    aligned_slowest.append(slow_event)
                alignment_info.append({
                    'match_type': 'exact_match',
                    'fast_idx': fast_idx,
                    'slow_idx': slow_idx,
                        'fast_name': fast_event.name,
                        'slow_name': slow_event.name
                })
            elif fast_idx is not None:
                # 快卡事件未匹配
                aligned_fastest.append(fastest_events[fast_idx])
                aligned_slowest.append(None)
                alignment_info.append({
                    'match_type': 'unmatched_fast',
                    'fast_idx': fast_idx,
                    'slow_idx': None,
                    'fast_name': fastest_events[fast_idx].name,
                    'slow_name': None
                })
            else:
                # 慢卡事件未匹配
                aligned_fastest.append(None)
                aligned_slowest.append(slowest_events[slow_idx])
                alignment_info.append({
                    'match_type': 'unmatched_slow',
                    'fast_idx': None,
                    'slow_idx': slow_idx,
                    'fast_name': None,
                    'slow_name': slowest_events[slow_idx].name
                })
        
        return aligned_fastest, aligned_slowest, alignment_info
    
    def visualize_alignment_results(self, fastest_cpu_events, slowest_cpu_events, 
                                   aligned_fastest, aligned_slowest, alignment_info,
                                   output_dir: str = ".", step: int = None, comm_idx: int = None):
        """
        可视化对齐结果
        
        Args:
            fastest_cpu_events: 原始最快卡CPU事件列表
            slowest_cpu_events: 原始最慢卡CPU事件列表
            aligned_fastest: 对齐后的最快卡事件列表
            aligned_slowest: 对齐后的最慢卡事件列表
            alignment_info: 对齐信息列表
            output_dir: 输出目录
            step: 步骤编号
            comm_idx: 通信索引
        """
        print("=== 生成对齐可视化图表 ===")
        
        # 设置英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 生成LCS对齐矩阵图
        # lcs_matrix_file = self._create_lcs_matrix_visualization(
        #     fastest_cpu_events, slowest_cpu_events, alignment_info, output_path, step, comm_idx
        # )
        lcs_matrix_file = 'None'
        
        # 2. 生成时间线对齐图（文本版本）
        timeline_file = self._create_timeline_alignment_text_visualization(
            aligned_fastest, aligned_slowest, alignment_info, output_path, step, comm_idx
        )
        
        # 3. 生成对齐统计图
        stats_file = self._create_alignment_statistics_visualization(
            alignment_info, output_path, step, comm_idx
        )
        
        print(f"对齐可视化文件已生成:")
        print(f"  - LCS矩阵图: {lcs_matrix_file}")
        print(f"  - 时间线对齐文本: {timeline_file}")
        print(f"  - 对齐统计图: {stats_file}")
        
        return [lcs_matrix_file, timeline_file, stats_file]
    
    def _create_lcs_matrix_visualization(self, fastest_cpu_events, slowest_cpu_events, 
                                       alignment_info, output_path, step, comm_idx):
        """创建LCS对齐矩阵可视化"""
        # 构建事件签名序列
        fast_signatures = [self._build_event_signature(e) for e in fastest_cpu_events]
        slow_signatures = [self._build_event_signature(e) for e in slowest_cpu_events]
        
        # 计算LCS矩阵
        lcs_matrix = self._compute_lcs_matrix(fast_signatures, slow_signatures)
        
        # 创建图形 - 根据矩阵大小调整
        matrix_size = (len(fast_signatures) + 1) * (len(slow_signatures) + 1)
        if matrix_size > 1000:
            # 大矩阵使用较小的图形和字体
            fig, ax = plt.subplots(figsize=(max(8, len(slow_signatures) * 0.4), max(6, len(fast_signatures) * 0.3)))
        else:
            fig, ax = plt.subplots(figsize=(max(12, len(slow_signatures) * 0.8), max(8, len(fast_signatures) * 0.6)))
        
        # 绘制矩阵
        matrix_array = np.array(lcs_matrix)
        im = ax.imshow(matrix_array, cmap='Blues', aspect='auto')
        
        # 设置坐标轴标签 - 根据矩阵大小调整
        if matrix_size > 1000:
            # 大矩阵简化标签
            ax.set_xticks(range(0, len(slow_signatures) + 1, max(1, len(slow_signatures) // 10)))
            ax.set_yticks(range(0, len(fast_signatures) + 1, max(1, len(fast_signatures) // 10)))
            ax.set_xticklabels([f'{i}' for i in range(0, len(slow_signatures) + 1, max(1, len(slow_signatures) // 10))])
            ax.set_yticklabels([f'{i}' for i in range(0, len(fast_signatures) + 1, max(1, len(fast_signatures) // 10))])
        else:
            ax.set_xticks(range(len(slow_signatures) + 1))
            ax.set_yticks(range(len(fast_signatures) + 1))
            ax.set_xticklabels([''] + [sig[:20] + '...' if len(sig) > 20 else sig for sig in slow_signatures], 
                              rotation=45, ha='right')
            ax.set_yticklabels([''] + [sig[:20] + '...' if len(sig) > 20 else sig for sig in fast_signatures])
        
        # 在矩阵中显示数值 - 优化版本
        self._add_matrix_text_optimized(ax, lcs_matrix, matrix_array)
        
        # 高亮对齐路径
        self._highlight_alignment_path(ax, alignment_info, fast_signatures, slow_signatures)
        
        # 设置标题和标签
        ax.set_title(f'LCS Alignment Matrix (Step {step}, Comm {comm_idx})\n'
                    f'Fastest Card: {len(fast_signatures)} events, Slowest Card: {len(slow_signatures)} events', 
                    fontsize=14, pad=20)
        ax.set_xlabel('Slowest Card CPU Event Sequence', fontsize=12)
        ax.set_ylabel('Fastest Card CPU Event Sequence', fontsize=12)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('LCS Length', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存文件
        filename = f"lcs_alignment_matrix_step_{step}_comm_{comm_idx}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _add_matrix_text_optimized(self, ax, lcs_matrix, matrix_array):
        """优化的矩阵文本添加方法"""
        rows, cols = matrix_array.shape
        
        # 如果矩阵太大，跳过文本显示以提高性能
        if rows * cols > 500:  # 超过500个元素时跳过文本显示
            return
        
        # 计算阈值（只计算一次）
        max_val = matrix_array.max()
        threshold = max_val * 0.5
        
        # 创建颜色矩阵
        color_matrix = np.where(matrix_array < threshold, 'black', 'white')
        
        # 批量添加文本
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, str(lcs_matrix[i][j]), ha="center", va="center", 
                       color=color_matrix[i, j], fontsize=8)
    
    def _highlight_alignment_path(self, ax, alignment_info, fast_signatures, slow_signatures):
        """在LCS矩阵中高亮对齐路径"""
        for i, info in enumerate(alignment_info):
            if info['match_type'] == 'exact_match':
                # 精确匹配用绿色圆圈标记
                fast_idx = info['fast_idx'] + 1  # +1 因为矩阵有0行/列
                slow_idx = info['slow_idx'] + 1
                circle = patches.Circle((slow_idx, fast_idx), 0.3, 
                                      facecolor='green', edgecolor='darkgreen', alpha=0.7)
                ax.add_patch(circle)
            elif info['match_type'] == 'unmatched_fast':
                # 快卡未匹配用红色三角形标记
                fast_idx = info['fast_idx'] + 1
                triangle = patches.RegularPolygon((0, fast_idx), 3, radius=0.3,
                                                facecolor='red', edgecolor='darkred', alpha=0.7)
                ax.add_patch(triangle)
            elif info['match_type'] == 'unmatched_slow':
                # 慢卡未匹配用橙色三角形标记
                slow_idx = info['slow_idx'] + 1
                triangle = patches.RegularPolygon((slow_idx, 0), 3, radius=0.3,
                                                facecolor='orange', edgecolor='darkorange', alpha=0.7)
                ax.add_patch(triangle)
    
    def _create_timeline_alignment_text_visualization(self, aligned_fastest, aligned_slowest, 
                                                    alignment_info, output_path, step, comm_idx):
        """创建时间线对齐文本可视化"""
        print("=== 生成时间线对齐文本可视化 ===")
        
        # 创建文本文件
        filename = f"timeline_alignment_step_{step}_comm_{comm_idx}.txt"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"CPU事件对齐时间线 (Step {step}, Comm {comm_idx})\n")
            f.write("=" * 80 + "\n\n")
            
            # 统计信息
            total_events = len(alignment_info)
            exact_matches = sum(1 for info in alignment_info if info['match_type'] == 'exact_match')
            unmatched_fast = sum(1 for info in alignment_info if info['match_type'] == 'unmatched_fast')
            unmatched_slow = sum(1 for info in alignment_info if info['match_type'] == 'unmatched_slow')
            
            f.write(f"对齐统计:\n")
            f.write(f"  总事件数: {total_events}\n")
            f.write(f"  精确匹配: {exact_matches} ({exact_matches/total_events*100:.1f}%)\n")
            f.write(f"  快卡未匹配: {unmatched_fast} ({unmatched_fast/total_events*100:.1f}%)\n")
            f.write(f"  慢卡未匹配: {unmatched_slow} ({unmatched_slow/total_events*100:.1f}%)\n\n")
            
            # 符号说明
            f.write("符号说明:\n")
            f.write("  ✓ 精确匹配\n")
            f.write("  ✗ 快卡未匹配\n")
            f.write("  ○ 慢卡未匹配\n\n")
            
            # 时间线对齐图
            f.write("时间线对齐图:\n")
            f.write("-" * 80 + "\n")
            
            # 计算时间范围用于缩放
            all_times = []
            for fast_event, slow_event in zip(aligned_fastest, aligned_slowest):
                if fast_event:
                    all_times.extend([fast_event.ts, fast_event.ts + fast_event.dur])
                if slow_event:
                    all_times.extend([slow_event.ts, slow_event.ts + slow_event.dur])
            
            if all_times:
                min_time = min(all_times)
                max_time = max(all_times)
                time_range = max_time - min_time
                scale_factor = 60.0 / time_range if time_range > 0 else 1.0
            else:
                min_time = 0
                scale_factor = 1.0
            
            # 生成时间线
            for i, (fast_event, slow_event, info) in enumerate(zip(aligned_fastest, aligned_slowest, alignment_info)):
                # 选择符号
                if info['match_type'] == 'exact_match':
                    symbol = '✓'
                elif info['match_type'] == 'unmatched_fast':
                    symbol = '✗'
                else:
                    symbol = '○'
                
                # 最快卡信息
                if fast_event:
                    fast_start = (fast_event.ts - min_time) * scale_factor
                    fast_duration = fast_event.dur * scale_factor
                    fast_name = fast_event.name[:40] + '...' if len(fast_event.name) > 40 else fast_event.name
                    fast_bar = '█' * max(1, int(fast_duration))
                    f.write(f"最快卡 {i+1:3d}: {symbol} {fast_bar:<20} | {fast_name}\n")
                else:
                    f.write(f"最快卡 {i+1:3d}: {symbol} {'':<20} | [未匹配]\n")
                
                # 最慢卡信息
                if slow_event:
                    slow_start = (slow_event.ts - min_time) * scale_factor
                    slow_duration = slow_event.dur * scale_factor
                    slow_name = slow_event.name[:40] + '...' if len(slow_event.name) > 40 else slow_event.name
                    slow_bar = '█' * max(1, int(slow_duration))
                    f.write(f"最慢卡 {i+1:3d}: {symbol} {slow_bar:<20} | {slow_name}\n")
                else:
                    f.write(f"最慢卡 {i+1:3d}: {symbol} {'':<20} | [未匹配]\n")
                
                f.write("\n")
            
            # 添加详细的对齐信息
            f.write("\n详细对齐信息:\n")
            f.write("-" * 80 + "\n")
            
            for i, (fast_event, slow_event, info) in enumerate(zip(aligned_fastest, aligned_slowest, alignment_info)):
                f.write(f"事件 {i+1}: {info['match_type']}\n")
                
                if fast_event:
                    f.write(f"  最快卡: {fast_event.name}\n")
                    f.write(f"    时间: {fast_event.ts:.2f} - {fast_event.ts + fast_event.dur:.2f} (持续: {fast_event.dur:.2f}μs)\n")
                else:
                    f.write(f"  最快卡: [未匹配]\n")
                
                if slow_event:
                    f.write(f"  最慢卡: {slow_event.name}\n")
                    f.write(f"    时间: {slow_event.ts:.2f} - {slow_event.ts + slow_event.dur:.2f} (持续: {slow_event.dur:.2f}μs)\n")
                else:
                    f.write(f"  最慢卡: [未匹配]\n")
                
                f.write("\n")
        
        print(f"时间线对齐文本文件已生成: {filepath}")
        return filepath
    
    def _plot_timeline(self, ax, timeline_data, title, card_type):
        """绘制单个时间线"""
        colors = {
            'exact_match': 'green',
            'unmatched_fast': 'red',
            'unmatched_slow': 'orange'
        }
        
        y_positions = []
        bars = []
        
        for i, event in enumerate(timeline_data):
            y_pos = i
            y_positions.append(y_pos)
            
            if event['duration'] > 0:
                # 绘制时间条
                bar = ax.barh(y_pos, event['duration'], left=event['start'], 
                             color=colors.get(event['match_type'], 'gray'), 
                             alpha=0.7, height=0.8)
                bars.append(bar)
                
                # 添加事件名称标签
                ax.text(event['start'] + event['duration']/2, y_pos, event['name'], 
                       ha='center', va='center', fontsize=8, rotation=0)
            else:
                # 未匹配事件用虚线表示
                ax.barh(y_pos, 1, left=0, color='lightgray', alpha=0.3, height=0.8, 
                       linestyle='--', edgecolor='black')
                ax.text(0.5, y_pos, event['name'], ha='center', va='center', 
                       fontsize=8, style='italic')
        
        # 设置坐标轴
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"事件 {i+1}" for i in range(len(timeline_data))])
        ax.set_xlabel('时间 (微秒)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='精确匹配'),
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='快卡未匹配'),
            plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label='慢卡未匹配')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _create_alignment_statistics_visualization(self, alignment_info, output_path, step, comm_idx):
        """创建对齐统计可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 对齐类型分布饼图
        match_types = [info['match_type'] for info in alignment_info]
        type_counts = {
            'exact_match': match_types.count('exact_match'),
            'unmatched_fast': match_types.count('unmatched_fast'),
            'unmatched_slow': match_types.count('unmatched_slow')
        }
        
        labels = ['Exact Match', 'Unmatched Fast', 'Unmatched Slow']
        sizes = [type_counts['exact_match'], type_counts['unmatched_fast'], type_counts['unmatched_slow']]
        colors = ['green', 'red', 'orange']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Alignment Type Distribution')
        
        # 2. 对齐质量指标
        total_events = len(alignment_info)
        exact_matches = type_counts['exact_match']
        match_rate = exact_matches / total_events * 100 if total_events > 0 else 0
        
        metrics = ['Total Events', 'Exact Matches', 'Match Rate(%)']
        values = [total_events, exact_matches, match_rate]
        
        bars = ax2.bar(metrics, values, color=['blue', 'green', 'orange'])
        ax2.set_title('Alignment Quality Metrics')
        ax2.set_ylabel('Count/Percentage')
        
        # 在柱状图上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 3. 连续匹配长度分布
        consecutive_lengths = self._calculate_consecutive_match_lengths(alignment_info)
        if consecutive_lengths:
            ax3.hist(consecutive_lengths, bins=min(10, len(set(consecutive_lengths))), 
                    color='skyblue', alpha=0.7, edgecolor='black')
            ax3.set_title('Consecutive Match Length Distribution')
            ax3.set_xlabel('Consecutive Match Length')
            ax3.set_ylabel('Frequency')
        else:
            ax3.text(0.5, 0.5, 'No Consecutive Matches', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Consecutive Match Length Distribution')
        
        # 4. 对齐位置分布
        positions = list(range(len(alignment_info)))
        match_status = [1 if info['match_type'] == 'exact_match' else 0 for info in alignment_info]
        
        ax4.scatter(positions, match_status, c=match_status, cmap='RdYlGn', alpha=0.7)
        ax4.set_title('Alignment Position Distribution')
        ax4.set_xlabel('Event Position')
        ax4.set_ylabel('Match Status (1=Match, 0=Unmatched)')
        ax4.set_ylim(-0.1, 1.1)
        ax4.grid(True, alpha=0.3)
        
        # 设置总标题
        fig.suptitle(f'Alignment Statistics Analysis (Step {step}, Comm {comm_idx})', fontsize=16, y=0.95)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存文件
        filename = f"alignment_statistics_step_{step}_comm_{comm_idx}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _calculate_consecutive_match_lengths(self, alignment_info):
        """计算连续精确匹配的长度"""
        consecutive_lengths = []
        current_length = 0
        
        for info in alignment_info:
            if info['match_type'] == 'exact_match':
                current_length += 1
            else:
                if current_length > 0:
                    consecutive_lengths.append(current_length)
                    current_length = 0
        
        # 处理最后一个连续序列
        if current_length > 0:
            consecutive_lengths.append(current_length)
        
        return consecutive_lengths
    
    def _extract_events_between_cpu_ops(self, data, prev_cpu_event, current_cpu_event):
        """
        提取两个CPU操作之间的所有事件
        
        Args:
            data: ProfilerData对象
            prev_cpu_event: 前一个CPU操作事件
            current_cpu_event: 当前CPU操作事件
            
        Returns:
            dict: 按进程ID和线程ID组织的事件 {pid: {tid: [events]}}
        """
        if not prev_cpu_event or not current_cpu_event:
            return {}
        
        # 计算时间窗口
        # 前一个CPU操作的结束时间
        prev_end_time = prev_cpu_event.ts + prev_cpu_event.dur
        # 当前CPU操作的开始时间
        current_start_time = current_cpu_event.ts
        
        print(f"    提取事件时间窗口: {prev_end_time:.2f} - {current_start_time:.2f} (窗口大小: {current_start_time - prev_end_time:.2f}μs)")
        
        # 提取时间窗口内的所有事件
        events_in_window = []
        # 需要过滤掉的事件类型
        filtered_cats = {'mtia_ccp_events', 'ac2g', 'flow', 'gpu_user_annotation'}
        
        for event in data.events:
            # 断言检查事件基本字段
            assert event.ts is not None, f"event.ts is None for event: {event}"
            
            # 过滤掉特定类型的事件
            if event.cat in filtered_cats:
                continue
            
            event_start = event.ts
            event_end = event.ts + (event.dur if event.dur is not None else 0)
            
            # 检查事件是否与时间窗口有交集
            # 事件开始时间在窗口内，或者事件结束时间在窗口内，或者事件完全包含窗口
            if (event_start < current_start_time and event_end > prev_end_time):
                events_in_window.append(event)
        
        # 按进程ID和线程ID组织事件，过滤掉PID或TID不是整数的事件
        events_by_pid_tid = {}
        for event in events_in_window:
            pid = getattr(event, 'pid', 0)
            tid = getattr(event, 'tid', 0)
            
            # 检查PID和TID是否为整数
            try:
                pid = int(pid) if pid is not None else 0
                tid = int(tid) if tid is not None else 0
            except (ValueError, TypeError):
                # 如果转换失败，跳过这个事件
                continue
            
            if pid not in events_by_pid_tid:
                events_by_pid_tid[pid] = {}
            if tid not in events_by_pid_tid[pid]:
                events_by_pid_tid[pid][tid] = []
            
            events_by_pid_tid[pid][tid].append(event)
        
        # 对每个线程内的事件按时间排序
        for pid in events_by_pid_tid:
            for tid in events_by_pid_tid[pid]:
                events_by_pid_tid[pid][tid].sort(key=lambda x: x.ts)
        
        return events_by_pid_tid
    
    def _print_events_between_cpu_ops(self, fastest_data, slowest_data, prev_fastest_cpu, current_fastest_cpu, 
                                     prev_slowest_cpu, current_slowest_cpu):
        """
        打印两个CPU操作之间的事件信息
        """
        print(f"    === 快卡事件 (PID: {getattr(prev_fastest_cpu, 'pid', 'N/A')}) ===")
        fastest_events = self._extract_events_between_cpu_ops(fastest_data, prev_fastest_cpu, current_fastest_cpu)
        self._print_events_by_pid_tid(fastest_events, "快卡")
        
        print(f"    === 慢卡事件 (PID: {getattr(prev_slowest_cpu, 'pid', 'N/A')}) ===")
        slowest_events = self._extract_events_between_cpu_ops(slowest_data, prev_slowest_cpu, current_slowest_cpu)
        self._print_events_by_pid_tid(slowest_events, "慢卡")
    
    def _print_events_by_pid_tid(self, events_by_pid_tid, card_name):
        """
        按进程ID和线程ID打印事件
        """
        if not events_by_pid_tid:
            print(f"      {card_name}: 无事件")
            return
        
        for pid in sorted(events_by_pid_tid.keys()):
            for tid in sorted(events_by_pid_tid[pid].keys()):
                events = events_by_pid_tid[pid][tid]
                if not events:
                    continue
                
                print(f"      {card_name} PID={pid}, TID={tid} ({len(events)} 个事件):")
                
                for i, event in enumerate(events): 
                    # 断言检查关键字段
                    assert event.cat is not None, f"event.cat is None for event: {event}"
                    assert event.name is not None, f"event.name is None for event: {event}"
                    assert event.ts is not None, f"event.ts is None for event: {event}"
                    
                    # 安全处理duration
                    event_dur = event.dur if event.dur is not None else 0
                    event_end = event.ts + event_dur
                    
                    print(f"        {i+1:2d}. {event.cat:8s} | {event.name:30s} | {event.ts:12.2f} - {event_end:12.2f} | {event_dur:8.2f}μs")
                
                # if len(events) > 10:
                #     print(f"        ... 还有 {len(events) - 10} 个事件")
                print()
    
    
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
    
    def _print_events_between_call_stacks(self, prev_fastest_event, current_fastest_event, 
                                          prev_slowest_event=None, current_slowest_event=None,
                                          fastest_parser=None, slowest_parser=None):
        """
        打印两个事件之间的调用栈树分析，使用时间窗口过滤
        
        Args:
            prev_fastest_event: 前一个最快卡事件对象
            current_fastest_event: 当前最快卡事件对象
            prev_slowest_event: 前一个最慢卡事件对象
            current_slowest_event: 当前最慢卡事件对象
            fastest_parser: 最快卡的parser对象
            slowest_parser: 最慢卡的parser对象
        """
        if not prev_fastest_event or not current_fastest_event:
            print("    事件之间调用栈分析：无法获取事件对象")
            return
        
        print("    事件之间调用栈分析:")
        print("    " + "=" * 80)
        
        # 为最快卡分析两个事件之间的调用栈关系
        print(f"    最快Card ({prev_fastest_event.name} → {current_fastest_event.name}) 的调用栈关系:")
        self._print_time_window_call_stack_tree(
            prev_event=prev_fastest_event, 
            current_event=current_fastest_event, 
            parser=fastest_parser, 
            card_name="最快Card"
        )
        
        # 如果提供了最慢卡事件，也进行分析
        if prev_slowest_event and current_slowest_event and slowest_parser:
            print(f"\n    最慢Card ({prev_slowest_event.name} → {current_slowest_event.name}) 的调用栈关系:")
            self._print_time_window_call_stack_tree(
                prev_event=prev_slowest_event, 
                current_event=current_slowest_event, 
                parser=slowest_parser, 
                card_name="最慢Card"
            )
        
        print("    " + "=" * 80)
    
    def _print_time_window_call_stack_tree(self, prev_event, current_event, parser, card_name):
        """
        打印时间窗口内的调用栈树
        
        Args:
            prev_event: 前一个事件对象
            current_event: 当前事件对象  
            parser: PyTorchProfilerParser 对象
            card_name: Card名称标识
        """
        if not parser:
            print(f"      {card_name}: 无法获取parser对象")
            return
        
        # 获取调用栈树
        call_stack_trees = parser.get_call_stack_trees()
        if not call_stack_trees:
            print(f"      {card_name}: 未找到调用栈树")
            return
        
        # 找到前一个和当前事件对应的节点
        prev_event_id = f"{prev_event.name}:{prev_event.ts}:{prev_event.dur}:{prev_event.pid}:{prev_event.tid}"
        current_event_id = f"{current_event.name}:{current_event.ts}:{current_event.dur}:{current_event.pid}:{current_event.tid}"
        
        prev_node = parser.call_stack_builder.event_to_node_map.get(prev_event_id)
        current_node = parser.call_stack_builder.event_to_node_map.get(current_event_id)
        
        if not prev_node:
            # 添加调试信息
            print(f"      {card_name}: 未找到前一个事件 '{prev_event.name}' 的节点")
            print(f"        调试信息 - 查询的event_id: {prev_event_id}")
            print(f"        调试信息 - 前事件属性: ts={prev_event.ts}, dur={prev_event.dur}, pid={prev_event.pid}, tid={prev_event.tid}")
            print(f"        调试信息 - event_to_node_map大小: {len(parser.call_stack_builder.event_to_node_map)}")
            
            # 检查event_to_node_map中的前几个条目
            print(f"        event_to_node_map中的前几个条目:")
            for i, (key, value) in enumerate(list(parser.call_stack_builder.event_to_node_map.items())[:3]):
                print(f"          {i+1}. {key}")
            
            return
        
        if not current_node:
            print(f"      {card_name}: 未找到当前事件 '{current_event.name}' 的节点")
            return
        
        print(f"      {card_name}: 找到事件节点")
        print(f"        前一个事件: {prev_event.name} (ts={prev_event.ts:.6f}, dur={prev_event.dur:.6f})")
        print(f"        当前事件: {current_event.name} (ts={current_event.ts:.6f}, dur={current_event.dur:.6f})")
        
        # 定义时间窗口：从前一个事件开始到当前事件结束
        time_start = prev_event.ts
        time_end = current_event.ts + current_event.dur
        
        print(f"        时间窗口: [{time_start:.6f}, {time_end:.6f}]")
        
        # 收集时间窗口内的事件
        window_events = []
        
        # 获取前一个事件所属的(pid, tid)组
        pid_tid_key = (prev_event.pid, prev_event.tid)
        root_node = call_stack_trees.get(pid_tid_key)
        
        if root_node:
            window_events = self._collect_events_in_time_window(
                root_node, time_start, time_end, 
                prev_node.event_id, current_node.event_id
            )
        
        print(f"        时间窗口内事件数: {len(window_events)}")
        
        if window_events:
            print(f"        时间窗口内事件列表:")
            for i, event in enumerate(window_events):
                duration_ms = event.dur / 1000 if event.dur else 0
                is_prev = (event.name == prev_event.name and event.ts == prev_event.ts)
                is_curr = (event.name == current_event.name and event.ts == current_event.ts)
                
                marker = " [START]" if is_prev else " [END]" if is_curr else ""
                print(f"          {i+1:2d}. {event.name}{marker} (ts={event.ts:.6f}, dur={duration_ms:.3f}ms)")
            
            # 重新构建调用栈树并打印
            print(f"        时间窗口调用栈树:")
            self._build_and_print_time_window_tree(
                window_events, prev_event, current_event, card_name
            )
        else:
            print(f"        时间窗口内无其他事件")
    
    def _collect_events_in_time_window(self, root_node, time_start, time_end, prev_event_id, current_event_id):
        """收集时间窗口内的所有事件"""
        events = []
        event_ids = set()
        
        def traverse_node(node):
            event = node.event
            
            # 检查事件是否在时间窗口内
            event_start = event.ts
            event_end = event.ts + event.dur
            
            # 事件与时间窗口有重叠就包含（支持重叠检查）
            if (event_end > time_start and event_start < time_end):
                event_id = node.event_id
                
                # 避免重复添加相同事件
                if event_id not in event_ids:
                    events.append(event)
                    event_ids.add(event_id)
            
            # 递归遍历子节点
            for child in node.children:
                traverse_node(child)
        
        traverse_node(root_node)
        
        # 按时间戳排序
        events.sort(key=lambda e: e.ts)
        return events
    
    def _build_and_print_time_window_tree(self, events, prev_event, current_event, card_name):
        """为时间窗口内的事件重建调用栈树并打印"""
        if not events:
            return
        
        # 使用build_call_stacks_subtree方法重建调用栈树，保留原始映射
        builder = self.fastest_parser.call_stack_builder if card_name == "最快Card" else self.slowest_parser.call_stack_builder
        time_window_trees = builder.build_call_stacks_subtree(events, preserve_mapping=True)
        
        # 找到对应的(pid, tid)组的树
        pid_tid_key = (prev_event.pid, prev_event.tid)
        tree_root = time_window_trees.get(pid_tid_key)
        
        if tree_root:
            print(f"        ===== 调用栈树结构 =====")
            self._print_time_window_tree_recursive(
                tree_root, prev_event, current_event, depth=0, prefix=""
            )
        else:
            print(f"        {card_name}: 未能重建调用栈树")
    
    def _print_time_window_tree_recursive(self, node, prev_event, current_event, depth=0, prefix=""):
        """递归打印时间窗口调用栈树"""
        if depth > 20:  # 限制深度避免过深
            return
        
        event = node.event
        duration_ms = event.dur / 1000 if event.dur else 0
        
        # 标记起始和结束事件
        is_prev = (event.name == prev_event.name and event.ts == prev_event.ts)
        is_curr = (event.name == current_event.name and event.ts == current_event.ts)
        
        markers = []
        if is_prev:
            markers.append(" [START]")
        if is_curr:
            markers.append(" [END]")
        
        marker_str = "".join(markers)
        
        print(f"        {prefix}{event.name}{marker_str} (ts={event.ts:.6f}, dur={duration_ms:.3f}ms)")
        
        # 按开始时间排序子节点，便于跟踪时序
        sorted_children = sorted(node.children, key=lambda n: n.event.ts)
        
        for i, child in enumerate(sorted_children):
            is_last = i == len(sorted_children) - 1
            child_prefix = prefix + ("└── " if is_last else "├── ")
            self._print_time_window_tree_recursive(
                child, prev_event, current_event, depth + 1, child_prefix
            )
    
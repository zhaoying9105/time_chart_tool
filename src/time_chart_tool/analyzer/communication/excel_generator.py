"""
Excel文件生成模块
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union
from collections import defaultdict

from ....models import ActivityEvent


class ExcelGenerator:
    """Excel文件生成器"""
    
    def __init__(self):
        if pd is None:
            raise ImportError("pandas is required for Excel output. Please install pandas and openpyxl.")
    
    def generate_raw_data_excel(self, comm_data: Dict[int, Dict[int, List[float]]], 
                               output_dir: str) -> Path:
        """生成原始数据Excel文件"""
        return self._generate_excel_file(
            comm_data, output_dir, "communication_raw_data.xlsx", 
            self._create_raw_data_sheets
        )
    
    def generate_statistics_excel(self, comm_data: Dict[int, Dict[int, List[float]]], 
                                 output_dir: str) -> Path:
        """生成统计信息Excel文件"""
        return self._generate_excel_file(
            comm_data, output_dir, "communication_statistics.xlsx",
            self._create_statistics_sheets
        )
    
    def generate_deep_analysis_excel(self, comparison_result: Dict, 
                                   step: int, comm_idx: int, 
                                   output_dir: str) -> Union[Path, Tuple[Path, str]]:
        """生成深度分析Excel文件"""
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
            cpu_excel_file = self._generate_cpu_start_time_differences_excel(
                comparison_result, step, comm_idx, output_dir
            )
            return excel_file, str(cpu_excel_file)
        
        return excel_file
    
    def _generate_cpu_start_time_differences_excel(self, comparison_result: Dict, 
                                                  step: int, comm_idx: int, 
                                                  output_dir: str) -> Path:
        """生成CPU Start Time相邻差值分析Excel文件"""
        cpu_excel_file = Path(output_dir) / f"cpu_start_time_differences_card_idx_{comm_idx}.xlsx"
        
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
        return cpu_excel_file
    
    def _generate_excel_file(self, data: Any, output_dir: str, filename: str, 
                            sheet_creator) -> Path:
        """通用的Excel文件生成方法"""
        output_path = Path(output_dir) / filename
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            sheet_creator(data, writer)
        
        print(f"生成文件: {output_path}")
        return output_path
    
    def _create_raw_data_sheets(self, comm_data: Dict[int, Dict[int, List[float]]], 
                               writer):
        """创建原始数据表格"""
        for step, card_data in comm_data.items():
            rows = []
            for card_idx, durations in card_data.items():
                for i, duration in enumerate(durations):
                    rows.append({
                        '': step, 'Card_Index': card_idx, 
                        'Comm_Index': i, 'Duration_us': duration
                    })
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=f'Step_{step}', index=False)
    
    def _create_statistics_sheets(self, comm_data: Dict[int, Dict[int, List[float]]], 
                                 writer):
        """创建统计信息表格"""
        # Step统计
        step_stats = self._calculate_step_statistics(comm_data)
        if step_stats:
            pd.DataFrame(step_stats).to_excel(writer, sheet_name='Step_Statistics', index=False)
        
        # Card统计
        card_stats = self._calculate_card_statistics(comm_data)
        if card_stats:
            pd.DataFrame(card_stats).to_excel(writer, sheet_name='Card_Statistics', index=False)
    
    def _calculate_step_statistics(self, comm_data: Dict[int, Dict[int, List[float]]]) -> List[Dict]:
        """计算step级别统计信息"""
        step_stats = []
        for step, card_data in comm_data.items():
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
    
    def _calculate_card_statistics(self, comm_data: Dict[int, Dict[int, List[float]]]) -> List[Dict]:
        """计算card级别统计信息"""
        card_stats = []
        for step, card_data in comm_data.items():
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

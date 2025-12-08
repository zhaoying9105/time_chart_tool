
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from ...parser import parse_profiler_data

from .utils import (
    _extract_events_in_range_with_intersection,
    _get_analysis_start_time,
    _get_analysis_end_time,
    _find_prev_communication_kernel_events,
    _find_communication_event,
    _merge_cpu_and_kernel_events,
    _check_communication_kernel_consistency,
    _calculate_step_statistics,
)
from .event_comparator import compare_events_by_time_sequence
from .data_extractor import _parse_json_filename

def _auto_select_comm_target(comm_data: Dict[int, Dict[int, List[float]]], step: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    自动选择要分析的通信操作目标
    
    Returns:
        Tuple[comm_idx, fastest_card_idx, slowest_card_idx]
    """
    step_stats = _calculate_step_statistics(comm_data)
    # 筛选出指定step的统计信息
    step_specific_stats = [stat for stat in step_stats if stat["Step"] == step]
    
    if not step_specific_stats:
        return None, None, None
    
    # 找到Duration_Ratio最大的comm_idx
    max_ratio_stat = max(step_specific_stats, key=lambda x: x["Duration_Ratio"])
    
    return (
        max_ratio_stat["Comm_Op_Index"],
        max_ratio_stat["Fastest_Card_Index"],
        max_ratio_stat["Slowest_Card_Index"]
    )

def _perform_deep_analysis(comm_data: Dict[int, Dict[int, List[float]]], executor_folders: List[str],
                          step: int, comm_idx: int, output_dir: str, kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL",
                          prev_kernel_pattern: str = "TCDP_.*", fastest_card_idx: Optional[int] = None, slowest_card_idx: Optional[int] = None,
                          show_timestamp: bool = False, show_readable_timestamp: bool = False) -> Optional[Union[Path, List[Path]]]:
    """
    执行深度分析，比较快卡和慢卡的详细差异
    
    Args:
        comm_data: {step: {card_idx: [CommunicationData]}}
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
        
        for card_idx, entries in step_data.items():
            if comm_idx < len(entries):
                # entries是CommunicationData对象列表，我们需要获取dur属性
                card_durations[card_idx] = entries[comm_idx].dur
        
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

    fastest_json_file = _find_json_file(executor_folders, step, fastest_card[0])
    slowest_json_file = _find_json_file(executor_folders, step, slowest_card[0])
    
    if not fastest_json_file or not slowest_json_file:
        print("错误: 无法找到对应的JSON文件")
        return None
    
    print(f"最快card JSON文件: {fastest_json_file}")
    print(f"最慢card JSON文件: {slowest_json_file}")
    
    # 4. 加载并解析JSON文件
    fastest_events,fastest_call_stack_trees,fast_base_time = parse_profiler_data(fastest_json_file)
    slowest_events,slowest_call_stack_trees,slow_base_time = parse_profiler_data(slowest_json_file)
    assert fast_base_time is not None, "最快卡基准时间不能为空"
    assert slow_base_time is not None, "最慢卡基准时间不能为空"
    from ...utils.time import calculate_readable_timestamps
    fastest_events = calculate_readable_timestamps(fastest_events,fast_base_time)
    slowest_events = calculate_readable_timestamps(slowest_events,slow_base_time)
    
    # 5. 进行深度对比分析
    comparison_result = _compare_card_performance(
        fastest_events, slowest_events, 
        fastest_call_stack_trees, slowest_call_stack_trees,
        fast_base_time, slow_base_time,
        fastest_card[0], slowest_card[0], 
        step, comm_idx, 
        fastest_card[1] if fastest_card[1] is not None else 0, 
        slowest_card[1] if slowest_card[1] is not None else 0, 
        kernel_prefix, prev_kernel_pattern,
        show_timestamp, show_readable_timestamp, output_dir
    )
    
    if not comparison_result:
        print("错误: 深度对比分析失败")
        return None
    
    # 6. 生成深度分析Excel文件
    excel_result = _generate_deep_analysis_excel(comparison_result, step, comm_idx, output_dir)
    
    generated_files = []
    
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

def _find_json_file(executor_folders: List[str], step: int, card_idx: int) -> Optional[str]:
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
            parsed_step, parsed_card_idx = _parse_json_filename(json_file.name)
            if parsed_step == step and parsed_card_idx == card_idx:
                return str(json_file)
    return None

def _compare_card_performance(fastest_events, slowest_events, 
                            fastest_call_stack_trees, slowest_call_stack_trees,
                            fast_base_time, slow_base_time,
                             fastest_card_idx, slowest_card_idx,
                             step, comm_idx, fastest_duration, slowest_duration,
                             kernel_prefix, prev_kernel_pattern, show_timestamp: bool = False, 
                             show_readable_timestamp: bool = False, output_dir: str = "."):
    """比较快卡和慢卡的性能差异"""
    print("开始比较快卡和慢卡的性能差异...")
    
    # 1. 检查通信kernel一致性
    if not _check_communication_kernel_consistency(fastest_events, slowest_events, kernel_prefix, comm_idx):
        print("错误: 通信kernel一致性检查失败")
        # return None
    
    # 2. 对两个数据进行Stage1处理
    from ..op.postprocessor import postprocessing
    fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id = postprocessing(fastest_events,fastest_call_stack_trees,fast_base_time)
    slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id = postprocessing(slowest_events,slowest_call_stack_trees,slow_base_time)
    
    # 3. 找到目标通信kernel操作的时间范围
    fastest_comm_event = _find_communication_event(fastest_events, comm_idx, kernel_prefix)
    slowest_comm_event = _find_communication_event(slowest_events, comm_idx, kernel_prefix)
    
    if not fastest_comm_event or not slowest_comm_event:
        print("错误: 无法找到目标通信kernel操作")
        return None
    
    # 4. 找到上一个通信kernel操作的时间范围
    print("=== 查找上一个通信kernel ===")
    fastest_prev_comm_events = _find_prev_communication_kernel_events(fastest_events, [fastest_comm_event], prev_kernel_pattern)
    slowest_prev_comm_events = _find_prev_communication_kernel_events(slowest_events, [slowest_comm_event], prev_kernel_pattern)
    
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
    
    fastest_start_time = _get_analysis_start_time(fastest_events, fastest_prev_event, fastest_comm_event)
    slowest_start_time = _get_analysis_start_time(slowest_events, slowest_prev_event, slowest_comm_event)
    
    fastest_end_time = _get_analysis_end_time(fastest_comm_event)
    slowest_end_time = _get_analysis_end_time(slowest_comm_event)
    
    print(f"最快card分析时间范围: {fastest_start_time:.2f} - {fastest_end_time:.2f}")
    print(f"最慢card分析时间范围: {slowest_start_time:.2f} - {slowest_end_time:.2f}")
    
    # 7. 提取时间范围内的events并与filtered events取交集
    fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id = _extract_events_in_range_with_intersection(
        fastest_events, fastest_start_time, fastest_end_time, 
        fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id
    )
    slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id = _extract_events_in_range_with_intersection(
        slowest_events, slowest_start_time, slowest_end_time,
        slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id
    )
    
    # 8. 合并CPU和kernel events
    fastest_events_by_external_id = _merge_cpu_and_kernel_events(fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id)
    slowest_events_by_external_id = _merge_cpu_and_kernel_events(slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id)
    
    # 9. 计算通信kernel的实际duration
    fastest_comm_duration = fastest_comm_event.dur if fastest_comm_event and fastest_comm_event.dur is not None else 0.0
    slowest_comm_duration = slowest_comm_event.dur if slowest_comm_event and slowest_comm_event.dur is not None else 0.0
    
    # 10. 按时间顺序比较CPU操作
    comparison_rows = compare_events_by_time_sequence(
        fastest_events_by_external_id, slowest_events_by_external_id,
        fastest_card_idx, slowest_card_idx, fastest_comm_duration, slowest_comm_duration,
        fastest_events, slowest_events, show_timestamp, show_readable_timestamp,
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
        'top_cpu_start_time_differences': comparison_rows['top_cpu_start_time_differences'],
        'visualization_files': comparison_rows.get('visualization_files', [])
    }

def _generate_deep_analysis_excel(comparison_result, step, comm_idx, output_dir):
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
                    prev_fastest_call_stack = diff_info['prev_fastest_event'].call_stack
                if diff_info.get('prev_slowest_event'):
                    prev_slowest_call_stack = diff_info['prev_slowest_event'].call_stack
                if diff_info.get('current_fastest_event'):
                    current_fastest_call_stack = diff_info['current_fastest_event'].call_stack
                if diff_info.get('current_slowest_event'):
                    current_slowest_call_stack = diff_info['current_slowest_event'].call_stack
                
                # 计算最大调用栈长度，确保至少输出一行
                max_stack_len = max(len(prev_fastest_call_stack), len(prev_slowest_call_stack), 
                                  len(current_fastest_call_stack), len(current_slowest_call_stack), 1)
                
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


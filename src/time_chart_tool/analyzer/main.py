"""
主分析器模块 - 函数式重构
"""

import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from collections import defaultdict
import pandas as pd

from ..parser import parse_profiler_data
from .op import (
    postprocessing, 
    classify_events_by_external_id, 
    attach_call_stacks, 
    process_triton_names_step, 
    filter_and_select_events,
    data_aggregation, 
    compare, 
)
from .op.presenter import _present_single_file, _present_multiple_files
from .comm import analyze_communication_performance
from .utils import AggregatedData

logger = logging.getLogger(__name__)

def _process_single_file_internal(args):
    """处理单个文件的内部函数，用于并行处理"""
    file_path, aggregation_spec, include_op_patterns, exclude_op_patterns, include_kernel_patterns, exclude_kernel_patterns, step_idx, coarse_call_stack = args
    
    try:
        print(f"开始处理文件: {file_path}")
        # 1. 解析数据
        events, call_stack_trees, base_time_nanoseconds = parse_profiler_data(file_path, step_idx=step_idx)
        if events is None:
            logger.error(f"解析文件 {file_path} 失败，返回None")
            return file_path, None
        print(f"解析文件 {file_path} 成功，包含 {len(events)} 个事件")
        
        # 2. 后处理
        cpu_events, kernel_events = postprocessing(
            events,call_stack_trees,base_time_nanoseconds,
            include_op_patterns, exclude_op_patterns,
            include_kernel_patterns, exclude_kernel_patterns,
            coarse_call_stack
        )
        
        cpu_events_count = sum(len(events) for events in cpu_events.values())
        kernel_events_count = sum(len(events) for events in kernel_events.values())
        print(f"后处理完成: CPU事件 {cpu_events_count} 个, Kernel事件 {kernel_events_count} 个")

        if cpu_events_count == 0:
            logger.warning(f"后处理后没有CPU事件，请检查过滤条件或文件内容")
        
        # 3. 聚合
        aggregated = data_aggregation(
            cpu_events, kernel_events, aggregation_spec
        )
        
        if not aggregated:
            logger.warning(f"聚合后结果为空，请检查聚合逻辑或call stack配置")
        else:
            print(f"聚合完成: {len(aggregated)} 个聚合键")
        
        print(f"完成文件处理: {file_path}")
        return file_path, aggregated
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return file_path, None

def _process_files_parallel(file_paths, aggregation_spec, num_workers, 
                          include_op_patterns, exclude_op_patterns, 
                          include_kernel_patterns, exclude_kernel_patterns,
                           step_idx, coarse_call_stack):
    """并行处理多个文件"""
    tasks = []
    for file_path in file_paths:
        tasks.append((
            file_path, aggregation_spec, 
            include_op_patterns, exclude_op_patterns, 
            include_kernel_patterns, exclude_kernel_patterns,
             step_idx, coarse_call_stack
        ))
        
    aggregated_results = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_single_file_internal, task) for task in tasks]
        
        for future in as_completed(futures):
            file_path, result = future.result()
            if result is not None:
                aggregated_results[str(file_path)] = result
                
    return aggregated_results

def _merge_aggregated_data_for_key(aggregated_data_list: List[AggregatedData]) -> AggregatedData:
    """
    合并同一个键的多个聚合数据
    
    Args:
        aggregated_data_list: 聚合数据列表
        
    Returns:
        AggregatedData: 合并后的聚合数据
    """
    if not aggregated_data_list:
        return AggregatedData([], [], "")
    
    if len(aggregated_data_list) == 1:
        return aggregated_data_list[0]
    
    # 合并所有CPU事件和kernel事件
    all_cpu_events = []
    all_kernel_events = []
    
    for agg_data in aggregated_data_list:
        all_cpu_events.extend(agg_data.cpu_events)
        all_kernel_events.extend(agg_data.kernel_events)
    
    # 使用第一个数据的键
    key = aggregated_data_list[0].key
    
    return AggregatedData(
        cpu_events=all_cpu_events,
        kernel_events=all_kernel_events,
        key=key
    )

def _merge_same_label_files(aggregated_data_list: List[Dict[Union[str, tuple], AggregatedData]], 
                           aggregation_spec: List[str]) -> Dict[Union[str, tuple], AggregatedData]:
    """
    合并相同标签的文件数据
    
    Args:
        aggregated_data_list: 每个文件的聚合数据列表
        aggregation_spec: 聚合字段组合
        
    Returns:
        Dict[Union[str, tuple], AggregatedData]: 合并后的数据
    """
    print("合并相同标签的文件数据")
    
    merged_data = {}
    
    # aggregated_data_list 是一个字典，key是文件路径，value是该文件的聚合数据
    # 我们需要遍历所有的value
    data_values = list(aggregated_data_list.values()) if isinstance(aggregated_data_list, dict) else aggregated_data_list
    
    for aggregated_data in data_values:
        for key, agg_data in aggregated_data.items():
            if key in merged_data:
                # 合并数据
                merged_data[key] = _merge_aggregated_data_for_key([merged_data[key], agg_data])
            else:
                merged_data[key] = agg_data
    
    print(f"合并后得到 {len(merged_data)} 个唯一的键")
    return merged_data

def analyze_single_file_with_glob(file_paths: List[str], aggregation_spec: List[str] = ['name'],
                                 show_attributes: List[str] = [],
                                 output_dir: str = ".", label: str = None, 
                                 include_op_patterns: List[str] = None, 
                                 exclude_op_patterns: List[str] = None, include_kernel_patterns: List[str] = None, 
                                 exclude_kernel_patterns: List[str] = None, 
                                 step_idx: Optional[int] = None,
                                 coarse_call_stack: bool = False) -> List[Path]:
    """
    分析多个文件（视为同一个label进行合并）
    
    Args:
        file_paths: 文件路径列表
        aggregation_spec: 聚合字段组合
        show_attributes: 显示属性列表
        output_dir: 输出目录
        label: 文件标签
        
    Returns:
        List[Path]: 生成的文件路径列表
    """
    print(f"开始分析 {len(file_paths)} 个文件")
    
    # 1. 并行处理文件
    num_workers = min(len(file_paths), mp.cpu_count())
    print(f"使用 {num_workers} 个进程进行并行处理")
    
    aggregated_results = _process_files_parallel(
        file_paths, aggregation_spec, num_workers,
        include_op_patterns, exclude_op_patterns,
        include_kernel_patterns, exclude_kernel_patterns,
         step_idx, coarse_call_stack
    )
    
    if not aggregated_results:
        print("警告: 没有成功处理任何文件")
        return []
    
    # 2. 合并数据 (Stage 3)
    # 即使是多个文件，也视为同一个label（如DDP的多个rank）进行合并
    merged_data = _merge_same_label_files(aggregated_results, aggregation_spec)
    
    if not merged_data:
        print("警告: 合并后数据为空")
        return []
        
    # 3. 生成报告 (Stage 4)
    # 准备数据用于展示
    # 注意：Stage 4 需要的数据结构是 {label: aggregated_data}
    # 这里只有一个label
    # label_name = label if label else "merged"
    
    output_path = Path(output_dir)
    generated_files = _present_single_file(
        merged_data, 
        output_path, 
        show_attributes, 
        aggregation_spec,
        label=label
    )
    
    return generated_files


def analyze_multiple_files(file_labels: List[Tuple[List[str], str]], aggregation_spec: List[str] = ['name'], 
                          show_attributes: List[str] = [], 
                          output_dir: str = ".", 
                          compare_attributes: List[str] = [], 
                          max_workers: int = None, 
                          include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None, 
                          include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None, 
                          call_stack_source: str = 'tree',
                          coarse_call_stack: bool = False) -> List[Path]:
    """ 
    分析多个文件并对比 
    
    Args: 
        file_labels: 文件标签列表 [(file_paths, label), ...] 
        aggregation_spec: 聚合字段组合 
        show_attributes: 显示属性列表
        output_dir: 输出目录 
        compare_attributes: 比较属性列表
        max_workers: 最大工作进程数 
        include_op_patterns: 包含的操作名称模式列表 
        exclude_op_patterns: 排除的操作名称模式列表 
        include_kernel_patterns: 包含的kernel名称模式列表 
        exclude_kernel_patterns: 排除的kernel名称模式列表 
        call_stack_source: 调用栈来源，'args' 或 'tree' 
        coarse_call_stack: 是否使用粗粒度调用栈
        
    Returns: 
        List[Path]: 生成的文件路径列表 
    """ 
    print(f"=== 分析多个文件并对比: {len(file_labels)} 个标签 ===") 
    
    # 处理每个标签的文件 
    multiple_files_data = {} 
    
    for file_paths, label in file_labels: 
        # 并行处理文件 
        aggregated_data_list = _process_files_parallel(
            file_paths, aggregation_spec, max_workers or mp.cpu_count(), 
            include_op_patterns, exclude_op_patterns, 
            include_kernel_patterns, exclude_kernel_patterns, 
            step_idx=None, coarse_call_stack=coarse_call_stack
        ) 
        
        # 合并相同标签的文件 
        merged_data = _merge_same_label_files(aggregated_data_list, aggregation_spec) 
        
        multiple_files_data[label] = merged_data 
    
    # Stage 3: 数据比较 
    comparison_result = compare(multiple_files_data=multiple_files_data, aggregation_spec=aggregation_spec) 
    
    # Stage 4: 数据展示 
    generated_files = _present_multiple_files(
        data=comparison_result, 
        output_dir=output_dir, 
        show_attributes=show_attributes,
        aggregation_spec=aggregation_spec, 
        compare_attributes=compare_attributes,
        file_labels=[label for _, label in file_labels], 
        include_op_patterns=include_op_patterns, 
        exclude_op_patterns=exclude_op_patterns, 
        include_kernel_patterns=include_kernel_patterns, 
        exclude_kernel_patterns=exclude_kernel_patterns
    ) 
    
    return generated_files

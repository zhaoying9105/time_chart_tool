"""
并行处理工具
"""

from typing import Dict, List, Tuple, Union, Optional
from ...parser import PyTorchProfilerParser
from .data_structures import AggregatedData


def process_single_file_parallel(file_path: str, aggregation_spec: str,
                                include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None,
                                include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None,
                                call_stack_source: str = 'args', step_idx: Optional[int] = None) -> Tuple[str, Dict[Union[str, tuple], AggregatedData]]:
    """
    并行处理单个文件的函数
    
    Args:
        file_path: JSON文件路径
        aggregation_spec: 聚合字段组合
        include_op_patterns: 包含的操作名称模式列表
        exclude_op_patterns: 排除的操作名称模式列表
        include_kernel_patterns: 包含的kernel名称模式列表
        exclude_kernel_patterns: 排除的kernel名称模式列表
        call_stack_source: 调用栈来源，'args' 或 'tree'
        step_idx: 指定要分析的step索引，如果不指定则分析所有step
        
    Returns:
        Tuple[str, Dict]: (文件路径, 聚合后的数据)
    """
    try:
        # 创建新的解析器实例（每个进程需要独立的实例）
        parser = PyTorchProfilerParser(step_idx=step_idx)
        
        # 加载数据
        data = parser.load_json_file(file_path)
        
        # 如果需要使用调用栈树，先构建调用栈树
        if call_stack_source == 'tree':
            parser.build_call_stack_trees()
        
        # 创建分析器实例
        from ..main import Analyzer
        analyzer = Analyzer(step_idx=step_idx)
        
        # Stage 1: 数据后处理
        cpu_events_by_external_id, kernel_events_by_external_id = analyzer.stage1_data_postprocessing(
            data,
            include_op_patterns=include_op_patterns, exclude_op_patterns=exclude_op_patterns,
            include_kernel_patterns=include_kernel_patterns, exclude_kernel_patterns=exclude_kernel_patterns
        )
        
        # Stage 2: 数据聚合
        aggregated_data = analyzer.stage2_data_aggregation(cpu_events_by_external_id, kernel_events_by_external_id, aggregation_spec, call_stack_source)
        
        return file_path, aggregated_data
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return file_path, {}

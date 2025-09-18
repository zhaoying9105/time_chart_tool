"""
并行处理工具
"""

from typing import Dict, List, Tuple, Union
from ...parser import PyTorchProfilerParser
from .data_structures import AggregatedData


def process_single_file_parallel(file_path: str, aggregation_spec: str,
                                include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None,
                                include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None) -> Tuple[str, Dict[Union[str, tuple], AggregatedData]]:
    """
    并行处理单个文件的函数
    
    Args:
        file_path: JSON文件路径
        aggregation_spec: 聚合字段组合
        include_op_patterns: 包含的操作名称模式列表
        exclude_op_patterns: 排除的操作名称模式列表
        include_kernel_patterns: 包含的kernel名称模式列表
        exclude_kernel_patterns: 排除的kernel名称模式列表
        
    Returns:
        Tuple[str, Dict]: (文件路径, 聚合后的数据)
    """
    try:
        # 创建新的解析器实例（每个进程需要独立的实例）
        parser = PyTorchProfilerParser()
        
        # 加载数据
        data = parser.load_json_file(file_path)
        
        # 创建分析器实例
        from ..main import Analyzer
        analyzer = Analyzer()
        
        # Stage 1: 数据后处理
        cpu_events_by_external_id, kernel_events_by_external_id = analyzer.stage1_data_postprocessing(
            data,
            include_op_patterns=include_op_patterns, exclude_op_patterns=exclude_op_patterns,
            include_kernel_patterns=include_kernel_patterns, exclude_kernel_patterns=exclude_kernel_patterns
        )
        
        # Stage 2: 数据聚合
        aggregated_data = analyzer.stage2_data_aggregation(cpu_events_by_external_id, kernel_events_by_external_id, aggregation_spec)
        
        return file_path, aggregated_data
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return file_path, {}

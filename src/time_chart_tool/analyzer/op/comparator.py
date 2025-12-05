"""
数据比较阶段 (纯函数实现)
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging

from ..utils.data_structures import AggregatedData

logger = logging.getLogger(__name__)


def compare( multiple_files_data: Optional[Dict[str, Dict[Union[str, tuple], AggregatedData]]] = None,
                     aggregation_spec: List[str] = ['name']) -> Dict[str, Any]:
    """
    Stage 3: 数据比较
    支持单文件和多文件比较
    
    Args:
        single_file_data: 单文件聚合数据
        multiple_files_data: 多文件聚合数据
        aggregation_spec: 聚合字段列表
        
    Returns:
        Dict[str, Any]: 比较结果
    """
    print("=== Stage 3: 数据比较 ===")
    
    if multiple_files_data is not None:
        if len(multiple_files_data) <= 1:
            logger.warning("多文件模式下文件数量少于2，跳过比较")
            # 如果只有一个文件，可以直接返回，或者根据需求处理
            # 这里我们返回一个包含单个文件数据的字典，结构与多文件合并后的一致
            if len(multiple_files_data) == 1:
                label, data = next(iter(multiple_files_data.items()))
                # 转换为合并后的格式: {key: {label: data}}
                merged_data = {}
                for key, value in data.items():
                    merged_data[key] = {label: value}
                return merged_data
            return {}

        print("多文件模式")
        if 'call_stack' in aggregation_spec:
            return _merge_multiple_files_call_stack(multiple_files_data)
        else:
            return _merge_multiple_files_data(multiple_files_data, aggregation_spec)
    
    return {}


def _merge_multiple_files_data(multiple_files_data: Dict[str, Dict[Union[str, tuple], AggregatedData]]) -> Dict[str, Any]:
    """
    合并多个文件的数据（非调用栈模式）
    
    Args:
        multiple_files_data: 多文件聚合数据
        aggregation_spec: 聚合字段列表
        
    Returns:
        Dict[str, Any]: 合并后的数据
    """
    print(f"合并 {len(multiple_files_data)} 个文件的数据")
    
    # 收集所有唯一的键
    all_keys = set()
    for file_data in multiple_files_data.values():
        all_keys.update(file_data.keys())
    
    print(f"找到 {len(all_keys)} 个唯一的聚合键")
    
    # 创建合并后的数据结构
    merged_data = {}
    for key in all_keys:
        merged_data[key] = {}
        for file_label, file_data in multiple_files_data.items():
            if key in file_data:
                merged_data[key][file_label] = file_data[key]
            else:
                # 如果某个文件没有这个键，创建空的数据
                merged_data[key][file_label] = AggregatedData([], [], key)
    
    return merged_data


def _merge_multiple_files_call_stack(multiple_files_data: Dict[str, Dict[Union[str, tuple], AggregatedData]]) -> Dict[str, Any]:
    """
    合并多个文件的数据（调用栈模式）
    
    Args:
        multiple_files_data: 多文件聚合数据
        
    Returns:
        Dict[str, Any]: 合并后的数据
    """
    print(f"合并 {len(multiple_files_data)} 个文件的数据（调用栈模式）")
    
    # 收集所有唯一的键，并处理调用栈相似性
    all_keys = set()
    key_mapping = {}  # 原始键 -> 标准化键的映射
    
    for file_label, file_data in multiple_files_data.items():
        for key in file_data.keys():
            # 这里需要实现调用栈相似性检查
            # 简化版本：直接使用原始键
            normalized_key = key
            all_keys.add(normalized_key)
            key_mapping[key] = normalized_key
    
    print(f"找到 {len(all_keys)} 个唯一的聚合键")
    
    # 创建合并后的数据结构
    merged_data = {}
    for normalized_key in all_keys:
        merged_data[normalized_key] = {}
        for file_label, file_data in multiple_files_data.items():
            # 查找匹配的键
            matched_key = None
            for original_key, mapped_key in key_mapping.items():
                if mapped_key == normalized_key and original_key in file_data:
                    matched_key = original_key
                    break
            
            if matched_key:
                merged_data[normalized_key][file_label] = file_data[matched_key]
            else:
                # 如果某个文件没有这个键，创建空的数据
                merged_data[normalized_key][file_label] = AggregatedData([], [], normalized_key)
    
    return merged_data

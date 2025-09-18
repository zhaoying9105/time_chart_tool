"""
CLI验证器模块
"""

from typing import List, Dict


def validate_aggregation_fields(aggregation_spec: str) -> List[str]:
    """
    验证聚合字段组合是否合规
    
    Args:
        aggregation_spec: 聚合字段组合字符串
        
    Returns:
        List[str]: 验证后的字段列表
        
    Raises:
        ValueError: 如果字段组合不合法
    """
    if not aggregation_spec or not aggregation_spec.strip():
        raise ValueError("聚合字段不能为空")
    
    # 解析字段组合：逗号分隔的字段
    if ',' in aggregation_spec:
        fields = [field.strip() for field in aggregation_spec.split(',')]
    else:
        fields = [aggregation_spec.strip()]
    
    # 验证字段
    valid_fields = {'call_stack', 'name', 'shape', 'dtype', 'op_index'}
    for field in fields:
        if not field:
            raise ValueError("聚合字段不能为空字符串")
        if field not in valid_fields:
            raise ValueError(f"不支持的聚合字段: {field}。支持的字段: {', '.join(sorted(valid_fields))}")
    
    # 检查字段重复
    if len(fields) != len(set(fields)):
        raise ValueError("聚合字段不能重复")
    
    return fields


def parse_show_options(show_spec: str) -> Dict[str, bool]:
    """
    解析show选项
    
    Args:
        show_spec: show参数字符串，逗号分隔
        
    Returns:
        Dict[str, bool]: 各show选项的开关状态
    """
    valid_show_options = {
        'dtype', 'shape', 'kernel-names', 'kernel-duration', 
        'timestamp', 'readable-timestamp', 'kernel-timestamp', 'name'
    }
    
    show_options = {
        'dtype': False,
        'shape': False, 
        'kernel_names': False,
        'kernel_duration': False,
        'timestamp': False,
        'readable_timestamp': False,
        'kernel_timestamp': False,
        'name': False
    }
    
    if not show_spec or not show_spec.strip():
        return show_options
    
    # 解析逗号分隔的选项
    show_args = [arg.strip() for arg in show_spec.split(',')]
    
    for arg in show_args:
        if not arg:
            continue
        if arg not in valid_show_options:
            raise ValueError(f"不支持的show选项: {arg}。支持的选项: {', '.join(sorted(valid_show_options))}")
        
        if arg == 'kernel-names':
            show_options['kernel_names'] = True
        elif arg == 'kernel-duration':
            show_options['kernel_duration'] = True
        elif arg == 'readable-timestamp':
            show_options['readable_timestamp'] = True
        elif arg == 'kernel-timestamp':
            show_options['kernel_timestamp'] = True
        else:
            show_options[arg] = True
    
    return show_options


def parse_compare_options(compare_spec: str) -> Dict[str, bool]:
    """
    解析compare选项
    
    Args:
        compare_spec: compare参数字符串，逗号分隔
        
    Returns:
        Dict[str, bool]: 各compare选项的开关状态
    """
    valid_compare_options = {
        'dtype', 'shape', 'name', 'kernel_name'
    }
    
    compare_options = {
        'dtype': False,
        'shape': False,
        'name': False,
        'kernel_name': False
    }
    
    if not compare_spec or not compare_spec.strip():
        return compare_options
    
    # 解析逗号分隔的选项
    compare_args = [arg.strip() for arg in compare_spec.split(',')]
    
    for arg in compare_args:
        if not arg:
            continue
        if arg not in valid_compare_options:
            raise ValueError(f"不支持的compare选项: {arg}。支持的选项: {', '.join(sorted(valid_compare_options))}")
        
        compare_options[arg] = True
    
    return compare_options


def validate_file(file_path: str) -> bool:
    """验证文件是否存在且为 JSON 格式"""
    from pathlib import Path
    
    path = Path(file_path)
    if not path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return False
    
    if not path.suffix.lower() == '.json':
        print(f"警告: 文件可能不是 JSON 格式: {file_path}")
    
    return True

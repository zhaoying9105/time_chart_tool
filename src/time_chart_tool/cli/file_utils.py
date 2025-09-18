"""
文件处理工具模块
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple


def parse_file_paths(file_pattern: str) -> List[str]:
    """
    解析文件路径，支持 glob 模式
    
    Args:
        file_pattern: 文件路径模式，支持 glob 通配符
        
    Returns:
        List[str]: 匹配的文件路径列表
    """
    # 检查是否包含 glob 通配符
    if '*' in file_pattern or '?' in file_pattern or '[' in file_pattern:
        # 使用 glob 模式匹配
        matched_files = glob.glob(file_pattern)
        if not matched_files:
            raise ValueError(f"glob 模式 {file_pattern} 没有匹配到任何文件")
        
        # 过滤出 JSON 文件
        json_files = [f for f in matched_files if f.lower().endswith('.json')]
        if not json_files:
            raise ValueError(f"glob 模式 {file_pattern} 没有匹配到任何 JSON 文件")
        
        return sorted(json_files)
    else:
        # 单个文件路径
        if not os.path.exists(file_pattern):
            raise ValueError(f"文件不存在: {file_pattern}")
        
        if not file_pattern.lower().endswith('.json'):
            raise ValueError(f"文件不是 JSON 格式: {file_pattern}")
        
        return [file_pattern]


def parse_file_label(file_label: str) -> Tuple[List[str], str]:
    """解析 file:label 格式的字符串，支持混合模式
    
    支持的模式：
    1. 单文件: file.json:label
    2. 多文件: file1.json,file2.json,file3.json:label  
    3. 目录: dir/:label (自动查找所有*.json文件)
    4. 通配符: "dir/*.json":label
    
    Returns:
        Tuple[List[str], str]: (文件路径列表, 标签)
    """
    if ':' in file_label:
        file_part, label = file_label.rsplit(':', 1)
        file_part = file_part.strip()
        label = label.strip()
        
        # 检查是否是目录
        if os.path.isdir(file_part):
            # 目录模式：查找所有json文件
            json_files = glob.glob(os.path.join(file_part, "*.json"))
            if not json_files:
                raise ValueError(f"目录 {file_part} 中没有找到任何 .json 文件")
            return sorted(json_files), label
        elif ',' in file_part:
            # 多文件模式：逗号分隔
            files = [f.strip() for f in file_part.split(',')]
            return files, label
        elif '*' in file_part or '?' in file_part:
            # 通配符模式
            matched_files = glob.glob(file_part)
            if not matched_files:
                raise ValueError(f"通配符 {file_part} 没有匹配到任何文件")
            return sorted(matched_files), label
        else:
            # 单文件模式
            return [file_part], label
    else:
        # 没有标签，使用文件名作为标签
        return [file_label], Path(file_label).stem

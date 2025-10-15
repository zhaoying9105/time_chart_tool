"""
调用栈处理工具模块
"""

import re
from typing import List


def _remove_module_suffix(module_name: str) -> str:
    """
    去除模块名后缀，将精确的 call stack 转换为粗糙的 call stack
    例如: Dense_10 -> Dense, DenseTower_5 -> DenseTower, AllEmbedding_0 -> AllEmbedding
    
    Args:
        module_name: 模块名称，可能包含后缀
        
    Returns:
        str: 去除后缀后的模块名称
    """
    # 使用正则表达式匹配以下划线+数字结尾的模块名
    # 例如: Dense_10, DenseTower_5, AllEmbedding_0
    match = re.match(r'^(.+)_\d+$', module_name)
    if match:
        return match.group(1)
    return module_name


class CallStackWrapper:
    """Call stack 包装类，用于区分 call_stack 的 tuple 和多字段聚合的 tuple"""
    
    def __init__(self, call_stack: List[str]):
        self.call_stack = call_stack
    
    def __eq__(self, other):
        if isinstance(other, CallStackWrapper):
            return self.call_stack == other.call_stack
        return False
    
    def __hash__(self):
        return hash(tuple(self.call_stack))
    
    def __repr__(self):
        return f"CallStackWrapper({self.call_stack})"
    
    def __str__(self):
        return " -> ".join(self.call_stack)


def normalize_call_stack(call_stack: List[str], coarse_call_stack: bool = False) -> CallStackWrapper:
    """
    标准化 call stack，优先保留包含 nn.Module 的有价值部分
    特殊处理：去掉 Runstep 模块及其之后的模块（面向 lg-torch 的特殊逻辑）
    如果没有 nn.Module，则保留原始 call stack
    同时去掉内存地址信息（如 object at 0x7f6f0a70efc0）以确保相同逻辑的调用栈被识别为相同
    
    Args:
        call_stack: 原始 call stack
        coarse_call_stack: 是否生成粗糙的 call stack（去掉模块名后缀如 _10, _11 等）
        
    Returns:
        CallStackWrapper: 标准化后的 call stack 包装对象，优先包含模型相关的层级，去掉 'nn.Module: ' 前缀
                         如果没有 nn.Module，则返回原始 call stack
    """
    if not call_stack:
        # print("DEBUG normalize_call_stack: 输入call_stack为空")
        return CallStackWrapper([])
    
    # print(f"DEBUG normalize_call_stack: 输入call_stack长度={len(call_stack)}")
    # print(f"DEBUG normalize_call_stack: 原始call_stack内容:")
    # for i, frame in enumerate(call_stack):
    #     print(f"  [{i}] {frame}")
    
    # 过滤出所有的 nn.Module
    nn_modules = []
    for i, frame in enumerate(call_stack):
        if 'nn.Module:' in frame:
            # 去掉 'nn.Module: ' 前缀
            module_name = frame.replace('nn.Module: ', '')
            
            # 如果需要生成粗糙的 call stack，去除后缀（如 _10, _11 等）
            if coarse_call_stack:
                module_name = _remove_module_suffix(module_name)
            
            nn_modules.append(module_name)
            # print(f"DEBUG normalize_call_stack: 找到nn.Module[{i}]: {module_name}")
    
    if not nn_modules:
        # print("DEBUG normalize_call_stack: 没有找到nn.Module，保留原始call_stack")
        # 如果没有 nn.Module，保留原始 call stack，但去掉内存地址信息
        normalized = []
        for frame in call_stack:
            # 去掉内存地址信息（如 object at 0x7f6f0a70efc0）
            # 匹配并去掉 "object at 0x..." 这样的内存地址信息
            cleaned_frame = re.sub(r' object at 0x[0-9a-fA-F]+>', '>', frame)
            normalized.append(cleaned_frame)
        # print(f"DEBUG normalize_call_stack: 返回标准化后的call_stack长度={len(normalized)}")
        return CallStackWrapper(normalized)
    
    # # 找到包含 Runstep 的模块
    # runstep_idx = -1
    # for i, module_name in enumerate(nn_modules):
    #     if 'Runstep' in module_name:
    #         runstep_idx = i
    #         break
    
    # # 如果找到 Runstep，去掉它及其之后的模块
    # if runstep_idx != -1:
    #     normalized = nn_modules[:runstep_idx]
    # else:
    #     normalized = nn_modules
    
    # # 对结果也去掉内存地址信息
    # cleaned_normalized = []
    # for frame in normalized:
    #     # 去掉内存地址信息（如 object at 0x7f6f0a70efc0）
    #     cleaned_frame = re.sub(r' object at 0x[0-9a-fA-F]+>', '>', frame)
    #     cleaned_normalized.append(cleaned_frame)
    
    # print(f"DEBUG normalize_call_stack: 找到{len(nn_modules)}个nn.Module，返回nn_modules")
    # print(f"DEBUG normalize_call_stack: nn_modules内容:")
    # for i, module in enumerate(nn_modules):
    #     print(f"  [{i}] {module}")
    return CallStackWrapper(nn_modules)

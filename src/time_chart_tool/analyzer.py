"""
PyTorch Profiler 高级分析器 - 重构版本
按照4个stage重新组织：数据后处理、数据聚合、比较、展示
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
import statistics
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from .models import ActivityEvent, ProfilerData
from .parser import PyTorchProfilerParser


@dataclass
class KernelStatistics:
    """Kernel 统计信息"""
    kernel_name: str
    min_duration: float
    max_duration: float
    mean_duration: float
    variance: float
    count: int
    
    def __str__(self):
        return f"KernelStatistics({self.kernel_name}, count={self.count}, mean={self.mean_duration:.3f}, std={self.variance**0.5:.3f})"


@dataclass
class AggregatedData:
    """聚合后的数据结构"""
    cpu_events: List[ActivityEvent]
    kernel_events: List[ActivityEvent]
    key: str  # 聚合键（op_name, op_shape, call_stack等）


def _process_single_file_parallel(file_path: str, aggregation_spec: str) -> Tuple[str, Dict[Union[str, tuple], 'AggregatedData']]:
    """
    并行处理单个文件的函数
    
    Args:
        file_path: JSON文件路径
        aggregation_spec: 聚合字段组合
        
    Returns:
        Tuple[str, Dict]: (文件路径, 聚合后的数据)
    """
    try:
        # 创建新的解析器实例（每个进程需要独立的实例）
        parser = PyTorchProfilerParser()
        
        # 加载数据
        data = parser.load_json_file(file_path)
        
        # 创建分析器实例
        analyzer = Analyzer()
        
        # Stage 1: 数据后处理
        cpu_events_by_external_id, kernel_events_by_external_id = analyzer.stage1_data_postprocessing(data)
        
        # Stage 2: 数据聚合
        aggregated_data = analyzer.stage2_data_aggregation(cpu_events_by_external_id, kernel_events_by_external_id, aggregation_spec)
        
        return file_path, aggregated_data
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return file_path, {}


class Analyzer:
    """重构后的分析器，按照4个stage组织"""
    
    # 通信kernel黑名单：过滤机内通信，只关注全局通信
    COMMUNICATION_BLACKLIST_PATTERNS = [
        'TCDP_RING_ALLGATHER_',
        'TCDP_RING_REDUCESCATTER_',
        'ALLREDUCELL',
        'TCDP_RING_ALLREDUCE_SIMPLE_BF16_ADD'
    ]
    
    def __init__(self):
        self.parser = PyTorchProfilerParser()
    
    # ==================== Stage 1: 数据后处理 ====================
    
    def stage1_data_postprocessing(self, data: ProfilerData) -> Tuple[Dict[Union[int, str], List[ActivityEvent]], Dict[Union[int, str], List[ActivityEvent]]]:
        """
        Stage 1: 数据后处理
        1. 时间戳标准化已在parser中完成
        2. 根据 cpu_op event & kernel event 的external id 分类，形成两个map
        3. 然后对一个 external id 下的 cpu_event 进行call stack 合并，保留call stack 最长的哪个 event
        
        Args:
            data: ProfilerData 对象
            
        Returns:
            Tuple[Dict[external_id, cpu_events], Dict[external_id, kernel_events]]: 
                两个映射字典，确保 external id 和 cpu_event 是一对一的关系
        """
        print("=== Stage 1: 数据后处理 ===")
        print("时间戳标准化已在parser中完成")
        
        # 2. 根据 external id 分类
        cpu_events_by_external_id = defaultdict(list)
        kernel_events_by_external_id = defaultdict(list)
        
        for event in data.events:
            if event.external_id is not None:
                if event.cat == 'cpu_op':
                    cpu_events_by_external_id[event.external_id].append(event)
                elif event.cat == 'kernel':
                    kernel_events_by_external_id[event.external_id].append(event)
        
        print(f"找到 {len(cpu_events_by_external_id)} 个有 cpu_op 的 external_id")
        print(f"找到 {len(kernel_events_by_external_id)} 个有 kernel 的 external_id")
        
        # 2. 过滤：保留有对应 kernel 事件的 cpu_op，如果没有 kernel 事件则保留所有 cpu_op
        filtered_cpu_events = {}
        filtered_kernel_events = {}
        
        if kernel_events_by_external_id:
            # 有 kernel 事件，只保留有对应 kernel 的 cpu_op
            for external_id in cpu_events_by_external_id:
                if external_id in kernel_events_by_external_id:
                    # 3. 对同一 external_id 下的 cpu_op 进行 call stack 合并
                    cpu_events = self._merge_cpu_events_by_call_stack(cpu_events_by_external_id[external_id])
                    filtered_cpu_events[external_id] = cpu_events
                    filtered_kernel_events[external_id] = kernel_events_by_external_id[external_id]
            print(f"过滤后保留 {len(filtered_cpu_events)} 个有对应 kernel 的 external_id")
        else:
            # 没有 kernel 事件，保留所有 cpu_op
            for external_id in cpu_events_by_external_id:
                # 3. 对同一 external_id 下的 cpu_op 进行 call stack 合并
                cpu_events = self._merge_cpu_events_by_call_stack(cpu_events_by_external_id[external_id])
                filtered_cpu_events[external_id] = cpu_events
                filtered_kernel_events[external_id] = []  # 空的 kernel 事件列表
            print(f"没有找到 kernel 事件，保留所有 {len(filtered_cpu_events)} 个 cpu_op external_id")
        
        return filtered_cpu_events, filtered_kernel_events
    
    def _merge_cpu_events_by_call_stack(self, cpu_events: List[ActivityEvent]) -> List[ActivityEvent]:
        """
        对同一 external_id 下的多个 cpu_op 事件进行 call stack 合并
        保留 call stack 最长的那个 event
        
        Args:
            cpu_events: 同一 external_id 下的 cpu_op 事件列表
            
        Returns:
            List[ActivityEvent]: 合并后的事件列表
        """
        if len(cpu_events) <= 1:
            return cpu_events
        
        # 按 name 分组
        name_groups = defaultdict(list)
        for event in cpu_events:
            name_groups[event.name].append(event)
        
        merged_events = []
        
        for name, events in name_groups.items():
            if len(events) == 1:
                merged_events.append(events[0])
            else:
                # 多个事件，检查 call stack 关系
                merged_events.extend(self._filter_events_by_call_stack_prefix(events))
        
        return merged_events
    
    def _filter_events_by_call_stack_prefix(self, events: List[ActivityEvent]) -> List[ActivityEvent]:
        """
        过滤具有相同 name 的多个 cpu_op 事件
        如果它们的 call stack 存在前缀关系，只保留最长的
        
        Args:
            events: 具有相同 name 的 cpu_op 事件列表
            
        Returns:
            List[ActivityEvent]: 过滤后的事件列表
        """
        if len(events) <= 1:
            return events
        
        # 检查是否所有事件都有 call stack
        events_with_call_stack = [e for e in events if e.call_stack is not None]
        events_without_call_stack = [e for e in events if e.call_stack is None]
        
        if not events_with_call_stack:
            return events
        
        if len(events_without_call_stack) > 0:
            raise ValueError(
                f"在同一个 external_id 下发现 {len(events)} 个相同 name 的 cpu_op 事件，"
                f"其中 {len(events_with_call_stack)} 个有 call stack，{len(events_without_call_stack)} 个没有 call stack。"
                f"这不符合预期，请检查数据。"
            )
        
        # 标准化 call stack 并检查前缀关系
        normalized_events = []
        for event in events_with_call_stack:
            normalized_call_stack = self._normalize_call_stack(event.call_stack)
            if normalized_call_stack:
                normalized_events.append((event, normalized_call_stack))
        
        if not normalized_events:
            return events
        
        # 检查是否存在前缀关系
        call_stack_strings = []
        for event, normalized_call_stack in normalized_events:
            call_stack_str = ' -> '.join(normalized_call_stack)
            call_stack_strings.append((event, call_stack_str))
        
        # 按长度排序，从长到短
        call_stack_strings.sort(key=lambda x: len(x[1]), reverse=True)
        
        # 检查前缀关系
        filtered_events = []
        for i, (event, call_stack_str) in enumerate(call_stack_strings):
            is_prefix_of_longer = False
            
            # 检查是否是更长的 call stack 的前缀
            for j in range(i):
                longer_call_stack_str = call_stack_strings[j][1]
                if longer_call_stack_str.startswith(call_stack_str):
                    is_prefix_of_longer = True
                    break
            
            if not is_prefix_of_longer:
                filtered_events.append(event)
        
        if not filtered_events:
            filtered_events = [call_stack_strings[0][0]]
        
        return filtered_events
    
    def _normalize_call_stack(self, call_stack: List[str]) -> List[str]:
        """
        标准化 call stack，优先保留包含 nn.Module 的有价值部分
        特殊处理：去掉 Runstep 模块及其之后的模块（面向 lg-torch 的特殊逻辑）
        如果没有 nn.Module，则保留原始 call stack
        
        Args:
            call_stack: 原始 call stack
            
        Returns:
            List[str]: 标准化后的 call stack，优先包含模型相关的层级，去掉 'nn.Module: ' 前缀
                      如果没有 nn.Module，则返回原始 call stack
        """
        if not call_stack:
            return call_stack
        
        # 过滤出所有的 nn.Module
        nn_modules = []
        for frame in call_stack:
            if 'nn.Module:' in frame:
                # 去掉 'nn.Module: ' 前缀
                module_name = frame.replace('nn.Module: ', '')
                nn_modules.append(module_name)
        
        if not nn_modules:
            # 如果没有 nn.Module，保留原始 call stack
            return call_stack
        
        # 找到包含 Runstep 的模块
        runstep_idx = -1
        for i, module_name in enumerate(nn_modules):
            if 'Runstep' in module_name:
                runstep_idx = i
                break
        
        # 如果找到 Runstep，去掉它及其之后的模块
        if runstep_idx != -1:
            normalized = nn_modules[:runstep_idx]
        else:
            normalized = nn_modules
        
        return normalized
    
    # ==================== Stage 2: 数据聚合 ====================
    
    def _generate_aggregation_key(self, cpu_event: ActivityEvent, aggregation_fields: List[str]) -> Union[str, tuple]:
        """
        根据指定的字段生成聚合键
        
        Args:
            cpu_event: CPU事件
            aggregation_fields: 聚合字段列表，支持: call_stack, name, shape, dtype
            
        Returns:
            Union[str, tuple]: 聚合键
        """
        key_parts = []
        
        for field in aggregation_fields:
            if field == 'call_stack':
                if cpu_event.call_stack is not None:
                    normalized_call_stack = self._normalize_call_stack(cpu_event.call_stack)
                    if normalized_call_stack:
                        key_parts.append(tuple(normalized_call_stack))
                    else:
                        key_parts.append(None)
                else:
                    key_parts.append(None)
            elif field == 'name':
                key_parts.append(cpu_event.name)
            elif field == 'shape':
                args = cpu_event.args or {}
                input_dims = args.get('Input Dims', [])
                # 递归地将嵌套列表转换为元组，使其可哈希
                def list_to_tuple(obj):
                    if isinstance(obj, list):
                        return tuple(list_to_tuple(item) for item in obj)
                    elif isinstance(obj, (int, float, str, bool)) or obj is None:
                        return obj
                    else:
                        return str(obj)  # 对于其他类型，转换为字符串
                key_parts.append(list_to_tuple(input_dims))
            elif field == 'dtype':
                args = cpu_event.args or {}
                dtype = args.get('Input type', 'unknown')
                # 确保dtype是可哈希的
                if isinstance(dtype, (list, dict)):
                    dtype = str(dtype)
                key_parts.append(dtype)
            else:
                raise ValueError(f"不支持的聚合字段: {field}")
        
        # 如果只有一个字段且不是None，直接返回该字段
        if len(key_parts) == 1 and key_parts[0] is not None:
            return key_parts[0]
        
        # 否则返回元组
        return tuple(key_parts)
    
    def _parse_aggregation_fields(self, aggregation_spec: str) -> List[str]:
        """
        解析聚合字段组合
        
        Args:
            aggregation_spec: 聚合字段组合字符串，如 "name,shape" 或 "call_stack,name"
            
        Returns:
            List[str]: 聚合字段列表
        """
        # 解析字段组合：逗号分隔的字段
        if ',' in aggregation_spec:
            fields = [field.strip() for field in aggregation_spec.split(',')]
        else:
            fields = [aggregation_spec.strip()]
        
        # 验证字段
        valid_fields = {'call_stack', 'name', 'shape', 'dtype'}
        for field in fields:
            if field not in valid_fields:
                raise ValueError(f"不支持的聚合字段: {field}。支持的字段: {', '.join(valid_fields)}")
        
        return fields
    
    def stage2_data_aggregation(self, cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                               kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                               aggregation_spec: str = 'name') -> Dict[Union[str, tuple], AggregatedData]:
        """
        Stage 2: 数据聚合
        支持灵活的字段组合聚合：
        - 支持的字段: call_stack, name, shape, dtype
        - 使用逗号分隔的字段组合，如 "name,shape" 或 "call_stack,name,dtype"
        
        Args:
            cpu_events_by_external_id: external_id -> cpu_events 映射
            kernel_events_by_external_id: external_id -> kernel_events 映射
            aggregation_spec: 聚合字段组合字符串
            
        Returns:
            Dict[Union[str, tuple], AggregatedData]: 聚合后的数据
        """
        print(f"=== Stage 2: 数据聚合 ({aggregation_spec}) ===")
        
        # 解析聚合字段组合
        aggregation_fields = self._parse_aggregation_fields(aggregation_spec)
        
        if 'call_stack' in aggregation_fields:
            # 包含调用栈的聚合需要特殊处理，因为需要合并相似的调用栈
            return self._aggregate_with_call_stack(cpu_events_by_external_id, kernel_events_by_external_id, aggregation_fields)
        
        # 普通聚合处理
        aggregated_data = defaultdict(lambda: AggregatedData([], [], ""))
        
        for external_id in cpu_events_by_external_id:
            cpu_events = cpu_events_by_external_id[external_id]
            kernel_events = kernel_events_by_external_id.get(external_id, [])
            
            for cpu_event in cpu_events:
                try:
                    key = self._generate_aggregation_key(cpu_event, aggregation_fields)
                    aggregated_data[key].cpu_events.append(cpu_event)
                    aggregated_data[key].kernel_events.extend(kernel_events)
                    aggregated_data[key].key = key
                except Exception as e:
                    print(f"警告: 跳过事件 {cpu_event.name}，错误: {e}")
                    continue
        
        print(f"聚合后得到 {len(aggregated_data)} 个不同的键")
        
        return dict(aggregated_data)
    
    def _aggregate_with_call_stack(self, cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                                  kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                                  aggregation_fields: List[str]) -> Dict[Union[str, tuple], AggregatedData]:
        """
        处理包含调用栈的聚合，实现 startswith 合并逻辑
        
        Args:
            cpu_events_by_external_id: external_id -> cpu_events 映射
            kernel_events_by_external_id: external_id -> kernel_events 映射
            aggregation_fields: 聚合字段列表
            
        Returns:
            Dict[Union[str, tuple], AggregatedData]: 聚合后的数据
        """
        aggregated_data = defaultdict(lambda: AggregatedData([], [], ""))
        
        for external_id in cpu_events_by_external_id:
            cpu_events = cpu_events_by_external_id[external_id]
            kernel_events = kernel_events_by_external_id.get(external_id, [])
            
            for cpu_event in cpu_events:
                if cpu_event.call_stack is None:
                    continue  # 跳过没有 call stack 的事件
                
                try:
                    new_key = self._generate_aggregation_key(cpu_event, aggregation_fields)
                    
                    # 检查是否已有相似的调用栈（使用startswith判断）
                    existing_key = None
                    for existing_key_candidate in aggregated_data.keys():
                        if self._is_similar_call_stack_key(new_key, existing_key_candidate, aggregation_fields):
                            # 保留较长的调用栈
                            if self._should_keep_new_key(new_key, existing_key_candidate, aggregation_fields):
                                existing_key = new_key
                            else:
                                existing_key = existing_key_candidate
                            break
                    
                    if existing_key:
                        # 合并到现有键
                        aggregated_data[existing_key].cpu_events.append(cpu_event)
                        aggregated_data[existing_key].kernel_events.extend(kernel_events)
                    else:
                        # 创建新键
                        aggregated_data[new_key] = AggregatedData(
                            cpu_events=[cpu_event],
                            kernel_events=kernel_events,
                            key=new_key
                        )
                except Exception as e:
                    print(f"警告: 跳过事件 {cpu_event.name}，错误: {e}")
                    continue
        
        print(f"聚合后得到 {len(aggregated_data)} 个不同的键")
        
        return dict(aggregated_data)
    
    def _is_similar_call_stack_key(self, key1: Union[str, tuple], key2: Union[str, tuple], aggregation_fields: List[str]) -> bool:
        """
        判断两个键是否相似（调用栈部分使用startswith判断）
        
        Args:
            key1: 第一个键
            key2: 第二个键
            aggregation_fields: 聚合字段列表
            
        Returns:
            bool: 是否相似
        """
        # 确保键是元组格式
        if not isinstance(key1, tuple):
            key1 = (key1,)
        if not isinstance(key2, tuple):
            key2 = (key2,)
        
        # 检查非调用栈字段是否相同
        for i, field in enumerate(aggregation_fields):
            if field != 'call_stack':
                if i < len(key1) and i < len(key2) and key1[i] != key2[i]:
                    return False
        
        # 检查调用栈字段
        call_stack_idx = aggregation_fields.index('call_stack') if 'call_stack' in aggregation_fields else -1
        if call_stack_idx >= 0 and call_stack_idx < len(key1) and call_stack_idx < len(key2):
            call_stack1 = key1[call_stack_idx]
            call_stack2 = key2[call_stack_idx]
            
            if call_stack1 is None or call_stack2 is None:
                return False
            
            # 将调用栈转换为字符串进行比较
            call_stack1_str = ' -> '.join(call_stack1) if isinstance(call_stack1, tuple) else str(call_stack1)
            call_stack2_str = ' -> '.join(call_stack2) if isinstance(call_stack2, tuple) else str(call_stack2)
            
            return (call_stack1_str.startswith(call_stack2_str) or 
                    call_stack2_str.startswith(call_stack1_str))
        
        return False
    
    def _should_keep_new_key(self, new_key: Union[str, tuple], existing_key: Union[str, tuple], aggregation_fields: List[str]) -> bool:
        """
        判断是否应该保留新键（保留较长的调用栈）
        
        Args:
            new_key: 新键
            existing_key: 现有键
            aggregation_fields: 聚合字段列表
            
        Returns:
            bool: 是否保留新键
        """
        call_stack_idx = aggregation_fields.index('call_stack') if 'call_stack' in aggregation_fields else -1
        if call_stack_idx < 0:
            return False
        
        # 确保键是元组格式
        if not isinstance(new_key, tuple):
            new_key = (new_key,)
        if not isinstance(existing_key, tuple):
            existing_key = (existing_key,)
        
        if (call_stack_idx < len(new_key) and call_stack_idx < len(existing_key) and
            new_key[call_stack_idx] is not None and existing_key[call_stack_idx] is not None):
            
            new_call_stack = new_key[call_stack_idx]
            existing_call_stack = existing_key[call_stack_idx]
            
            new_len = len(new_call_stack) if isinstance(new_call_stack, tuple) else 1
            existing_len = len(existing_call_stack) if isinstance(existing_call_stack, tuple) else 1
            
            return new_len > existing_len
        
        return False
    
    def _aggregate_on_call_stack(self, cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                                kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]]) -> Dict[Union[str, tuple], AggregatedData]:
        """
        专门处理 on_call_stack 聚合，实现 startswith 合并逻辑
        
        Args:
            cpu_events_by_external_id: external_id -> cpu_events 映射
            kernel_events_by_external_id: external_id -> kernel_events 映射
            
        Returns:
            Dict[Union[str, tuple], AggregatedData]: 聚合后的数据
        """
        aggregated_data = defaultdict(lambda: AggregatedData([], [], ""))
        
        for external_id in cpu_events_by_external_id:
            cpu_events = cpu_events_by_external_id[external_id]
            kernel_events = kernel_events_by_external_id.get(external_id, [])
            
            for cpu_event in cpu_events:
                if cpu_event.call_stack is not None:
                    normalized_call_stack = self._normalize_call_stack(cpu_event.call_stack)
                    if normalized_call_stack:
                        # 组合键：调用栈 + 操作名称 为 tuple
                        new_key = (tuple(normalized_call_stack), cpu_event.name)
                        
                        # 检查是否已有相似的调用栈（使用startswith判断）
                        existing_key = None
                        for existing_call_stack_tuple, existing_op_name in aggregated_data.keys():
                            if existing_op_name == cpu_event.name:
                                # 只有当操作名称相同时才比较调用栈
                                existing_call_stack_list = list(existing_call_stack_tuple)
                                new_call_stack_list = list(normalized_call_stack)
                                
                                # 检查是否是 startswith 关系（将列表转换为字符串进行比较）
                                new_call_stack_str = ' -> '.join(new_call_stack_list)
                                existing_call_stack_str = ' -> '.join(existing_call_stack_list)
                                if (new_call_stack_str.startswith(existing_call_stack_str) or 
                                    existing_call_stack_str.startswith(new_call_stack_str)):
                                    # 保留较长的调用栈
                                    if len(new_call_stack_list) > len(existing_call_stack_list):
                                        existing_key = new_key
                                    else:
                                        existing_key = (existing_call_stack_tuple, existing_op_name)
                                    break
                        
                        if existing_key:
                            # 合并到现有键
                            aggregated_data[existing_key].cpu_events.append(cpu_event)
                            aggregated_data[existing_key].kernel_events.extend(kernel_events)
                        else:
                            # 创建新键
                            aggregated_data[new_key] = AggregatedData(
                                cpu_events=[cpu_event],
                                kernel_events=kernel_events,
                                key=new_key
                            )
                    else:
                        continue  # 跳过没有有效 call stack 的事件
                else:
                    continue  # 跳过没有 call stack 的事件
        
        print(f"聚合后得到 {len(aggregated_data)} 个不同的键")
        
        return dict(aggregated_data)
    
    # ==================== Stage 3: 比较 ====================
    
    def stage3_comparison(self, single_file_data: Optional[Dict[Union[str, tuple], AggregatedData]] = None,
                         multiple_files_data: Optional[Dict[str, Dict[Union[str, tuple], AggregatedData]]] = None,
                         aggregation_spec: str = 'name') -> Dict[str, Any]:
        """
        Stage 3: 比较
        区分处理单个 time tracing json 还是 多个 time tracing json
        
        Args:
            single_file_data: 单文件聚合数据
            multiple_files_data: 多文件聚合数据 {label: {key: AggregatedData}}
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        print("=== Stage 3: 比较 ===")
        
        if single_file_data is not None and multiple_files_data is None:
            # 单个 time tracing json - 没有比较，直接返回
            print("处理单个文件，无需比较")
            return {
                'type': 'single_file',
                'data': single_file_data
            }
        
        elif single_file_data is None and multiple_files_data is not None:
            # 多个 time tracing json - 进行跨文件聚合
            print(f"处理多个文件，共 {len(multiple_files_data)} 个文件")
            return self._merge_multiple_files_data(multiple_files_data, aggregation_spec)
        
        else:
            raise ValueError("必须提供 single_file_data 或 multiple_files_data 中的一个")
    
    def _merge_multiple_files_data(self, multiple_files_data: Dict[str, Dict[Union[str, tuple], AggregatedData]], aggregation_spec: str = 'name') -> Dict[str, Any]:
        """
        合并多个文件的聚合数据
        
        Args:
            multiple_files_data: {label: {key: AggregatedData}}
            
        Returns:
            Dict[str, Any]: 合并后的数据
        """
        # 检查是否包含调用栈字段，需要特殊处理
        aggregation_fields = self._parse_aggregation_fields(aggregation_spec)
        if 'call_stack' in aggregation_fields:
            # 包含调用栈的聚合需要特殊处理，合并相似的调用栈
            return self._merge_multiple_files_call_stack(multiple_files_data)
        
        # 收集所有唯一的键
        all_keys = set()
        for file_data in multiple_files_data.values():
            all_keys.update(file_data.keys())
        
        merged_data = {}
        
        for key in all_keys:
            merged_entry = {
                'key': key,
                'files': {}
            }
            
            # 为每个文件收集该键的数据
            for label, file_data in multiple_files_data.items():
                if key in file_data:
                    aggregated_data = file_data[key]
                    merged_entry['files'][label] = {
                        'cpu_events': aggregated_data.cpu_events,
                        'kernel_events': aggregated_data.kernel_events
                    }
                else:
                    merged_entry['files'][label] = {
                        'cpu_events': [],
                        'kernel_events': []
                    }
            
            merged_data[key] = merged_entry
        
        return {
            'type': 'multiple_files',
            'data': merged_data
        }
    
    def _merge_multiple_files_call_stack(self, multiple_files_data: Dict[str, Dict[Union[str, tuple], AggregatedData]]) -> Dict[str, Any]:
        """
        专门处理 on_call_stack 的多文件合并，实现 startswith 合并逻辑
        
        Args:
            multiple_files_data: {label: {key: AggregatedData}}
            
        Returns:
            Dict[str, Any]: 合并后的数据
        """
        # 收集所有唯一的键
        all_keys = set()
        for file_data in multiple_files_data.values():
            all_keys.update(file_data.keys())
        
        # 合并相似的调用栈
        merged_keys = {}
        for key in all_keys:
            call_stack_tuple, op_name = key
            
            # 检查是否已有相似的调用栈
            existing_key = None
            for existing_call_stack_tuple, existing_op_name in merged_keys.keys():
                if existing_op_name == op_name:
                    # 只有当操作名称相同时才比较调用栈
                    existing_call_stack_list = list(existing_call_stack_tuple)
                    new_call_stack_list = list(call_stack_tuple)
                    
                    # 检查是否是 startswith 关系（将列表转换为字符串进行比较）
                    new_call_stack_str = ' -> '.join(new_call_stack_list)
                    existing_call_stack_str = ' -> '.join(existing_call_stack_list)
                    if (new_call_stack_str.startswith(existing_call_stack_str) or 
                        existing_call_stack_str.startswith(new_call_stack_str)):
                        # 保留较长的调用栈
                        if len(new_call_stack_list) > len(existing_call_stack_list):
                            # 删除旧的键，使用新的键
                            del merged_keys[(existing_call_stack_tuple, existing_op_name)]
                            existing_key = key
                        else:
                            existing_key = (existing_call_stack_tuple, existing_op_name)
                        break
            
            if existing_key:
                # 合并到现有键
                merged_keys[existing_key] = True
            else:
                # 创建新键
                merged_keys[key] = True
        
        # 构建合并后的数据
        merged_data = {}
        for key in merged_keys.keys():
            merged_entry = {
                'key': key,
                'files': {}
            }
            
            # 为每个文件收集该键的数据
            for label, file_data in multiple_files_data.items():
                # 查找该文件中是否有与当前键相似的调用栈
                found_data = None
                for file_key, aggregated_data in file_data.items():
                    file_call_stack_tuple, file_op_name = file_key
                    if file_op_name == key[1]:  # 操作名称相同
                        # 检查调用栈是否是 startswith 关系（将列表转换为字符串进行比较）
                        file_call_stack_str = ' -> '.join(list(file_call_stack_tuple))
                        key_call_stack_str = ' -> '.join(list(key[0]))
                        if (file_call_stack_str.startswith(key_call_stack_str) or 
                            key_call_stack_str.startswith(file_call_stack_str)):
                            found_data = aggregated_data
                            break
                
                if found_data:
                    merged_entry['files'][label] = {
                        'cpu_events': found_data.cpu_events,
                        'kernel_events': found_data.kernel_events
                    }
                else:
                    merged_entry['files'][label] = {
                        'cpu_events': [],
                        'kernel_events': []
                    }
            
            merged_data[key] = merged_entry
        
        return {
            'type': 'multiple_files',
            'data': merged_data
        }
    
    # ==================== Stage 4: 展示 ====================
    
    def stage4_presentation(self, comparison_result: Dict[str, Any], 
                           output_dir: str = ".", 
                           show_dtype: bool = True,
                           show_shape: bool = True,
                           show_kernel_names: bool = True,
                           show_kernel_duration: bool = True,
                           show_timestamp: bool = False,
                           show_readable_timestamp: bool = False,
                           show_kernel_timestamp: bool = False,
                           special_matmul: bool = False,
                           aggregation_spec: str = 'name',
                           compare_dtype: bool = False,
                           compare_shape: bool = False,
                           file_labels: List[str] = None,
                           label: str = None,
                           print_markdown: bool = False) -> List[Path]:
        """
        Stage 4: 展示
        根据配置决定是否展示 dtype 信息，是否展示 shape 信息，是否展示kernel 信息，是否展示时间戳信息
        如果展示kernel 信息，那么包括kernel event 的name，duration的统计值，kernel event 的数量
        展示的结果是 包括xlsx 和 json
        
        Args:
            comparison_result: Stage 3 的比较结果
            output_dir: 输出目录
            show_dtype: 是否展示 dtype 信息
            show_shape: 是否展示 shape 和 strides 信息
            show_kernel_names: 是否展示 kernel 名称信息
            show_kernel_duration: 是否展示 kernel 持续时间信息
            show_timestamp: 是否展示 CPU 操作启动时间戳
            special_matmul: 是否进行特殊的 matmul 展示
        """
        print("=== Stage 4: 展示 ===")
        
        if comparison_result['type'] == 'single_file':
            return self._present_single_file(comparison_result['data'], output_dir, show_dtype, show_shape, show_kernel_names, show_kernel_duration, show_timestamp, show_readable_timestamp, show_kernel_timestamp, aggregation_spec, label, print_markdown)
        elif comparison_result['type'] == 'multiple_files':
            return self._present_multiple_files(comparison_result['data'], output_dir, show_dtype, show_shape, show_kernel_names, show_kernel_duration, special_matmul, show_timestamp, show_readable_timestamp, aggregation_spec, compare_dtype, compare_shape, file_labels, print_markdown)
    
    def _present_single_file(self, data: Dict[Union[str, tuple], AggregatedData], output_dir: str, show_dtype: bool, show_shape: bool, show_kernel_names: bool, show_kernel_duration: bool, show_timestamp: bool = False, show_readable_timestamp: bool = False, show_kernel_timestamp: bool = False, aggregation_spec: str = 'name', label: str = None, print_markdown: bool = False) -> List[Path]:
        """展示单文件数据"""
        print("生成单文件展示结果...")
        
        # 如果显示kernel duration，先计算总耗时
        total_duration = 0.0
        if show_kernel_duration:
            for aggregated_data in data.values():
                kernel_stats = self._calculate_kernel_statistics(aggregated_data.kernel_events)
                if kernel_stats:
                    total_duration += sum(stats.mean_duration * stats.count for stats in kernel_stats)
        
        rows = []
        for key, aggregated_data in data.items():
            # 初始化行数据
            row = {}
            
            # 添加时间戳列（如果启用，作为第一列）
            if show_timestamp:
                # 获取第一个CPU事件的启动时间戳
                if aggregated_data.cpu_events:
                    first_cpu_event = aggregated_data.cpu_events[0]
                    if show_readable_timestamp and first_cpu_event.readable_timestamp:
                        row['cpu_start_timestamp'] = first_cpu_event.readable_timestamp
                    else:
                        row['cpu_start_timestamp'] = first_cpu_event.ts if first_cpu_event.ts is not None else 0.0
                else:
                    row['cpu_start_timestamp'] = 0.0
            
            # 根据聚合字段组合生成相应的列
            aggregation_fields = self._parse_aggregation_fields(aggregation_spec)
            
            # 解析键的各个部分
            if isinstance(key, tuple):
                key_parts = list(key)
            else:
                key_parts = [key]
            
            # 为每个字段添加对应的列
            for i, field in enumerate(aggregation_fields):
                if i < len(key_parts) and key_parts[i] is not None:
                    if field == 'name':
                        row['op_name'] = key_parts[i]
                    elif field == 'shape':
                        row['input_dims'] = str(key_parts[i])
                    elif field == 'call_stack':
                        call_stack_str = ' -> '.join(key_parts[i]) if isinstance(key_parts[i], tuple) else str(key_parts[i])
                        row['call_stack'] = call_stack_str
                    elif field == 'dtype':
                        row['input_type'] = str(key_parts[i])
                else:
                    if field == 'name':
                        row['op_name'] = 'None'
                    elif field == 'shape':
                        row['input_dims'] = 'None'
                    elif field == 'call_stack':
                        row['call_stack'] = 'None'
                    elif field == 'dtype':
                        row['input_type'] = 'None'
            
            row['cpu_event_count'] = len(aggregated_data.cpu_events)
            
            if show_dtype:
                # 检查聚合字段是否已经包含dtype，如果包含则不重复添加dtypes列
                aggregation_fields = self._parse_aggregation_fields(aggregation_spec)
                if 'dtype' not in aggregation_fields:
                    # 收集 dtype 种类信息
                    dtypes = set()
                    for cpu_event in aggregated_data.cpu_events:
                        args = cpu_event.args or {}
                        input_types = args.get('Input type', [])
                        # 将每个event的输入类型信息作为一个整体来记录
                        if input_types:
                            dtypes.add(str(input_types))
                    row['dtypes'] = '\n'.join(sorted(dtypes)) if dtypes else ''
            
            if show_shape:
                # 收集 shape 和 strides 种类信息
                shapes = set()
                strides = set()
                for cpu_event in aggregated_data.cpu_events:
                    args = cpu_event.args or {}
                    input_dims = args.get('Input Dims', [])
                    input_strides = args.get('Input Strides', [])
                    # 将每个event的输入维度信息作为一个整体来记录
                    if input_dims:
                        shapes.add(str(input_dims))
                    if input_strides:
                        strides.add(str(input_strides))
                row['shapes'] = '\n'.join(sorted(shapes)) if shapes else ''
                row['strides'] = '\n'.join(sorted(strides)) if strides else ''
            
            if show_kernel_names or show_kernel_duration or show_kernel_timestamp:
                # 收集 kernel 信息
                kernel_stats = self._calculate_kernel_statistics(aggregated_data.kernel_events)
                if kernel_stats:
                    row['kernel_count'] = sum(stats.count for stats in kernel_stats)
                    if show_kernel_names:
                        row['kernel_names'] = '\n'.join(stats.kernel_name for stats in kernel_stats)
                    if show_kernel_duration:
                        op_total_duration = sum(stats.mean_duration * stats.count for stats in kernel_stats)
                        row['kernel_mean_duration'] = op_total_duration / sum(stats.count for stats in kernel_stats)
                        row['kernel_min_duration'] = min(stats.min_duration for stats in kernel_stats)
                        row['kernel_max_duration'] = max(stats.max_duration for stats in kernel_stats)
                        row['kernel_total_duration'] = op_total_duration
                        row['kernel_duration_ratio'] = (op_total_duration / total_duration * 100) if total_duration > 0 else 0.0
                    if show_kernel_timestamp:
                        # 收集kernel时间戳信息
                        kernel_timestamps = []
                        for kernel_event in aggregated_data.kernel_events:
                            if kernel_event.ts is not None:
                                kernel_timestamps.append(kernel_event.ts)
                        if kernel_timestamps:
                            row['kernel_timestamps'] = '\n'.join(map(str, sorted(kernel_timestamps)))
                        else:
                            row['kernel_timestamps'] = ''
                else:
                    row['kernel_count'] = 0
                    if show_kernel_names:
                        row['kernel_names'] = ''
                    if show_kernel_duration:
                        row['kernel_mean_duration'] = 0.0
                        row['kernel_min_duration'] = 0.0
                        row['kernel_max_duration'] = 0.0
                        row['kernel_total_duration'] = 0.0
                        row['kernel_duration_ratio'] = 0.0
                    if show_kernel_timestamp:
                        row['kernel_timestamps'] = ''
            
            rows.append(row)
        
        # 如果启用markdown打印，在stdout中打印表格
        if print_markdown:
            # 按照kernel_duration_ratio排序（从大到小）
            sorted_rows = sorted(rows, key=lambda x: x.get('kernel_duration_ratio', 0), reverse=True)
            self._print_markdown_table(sorted_rows, f"{label} 分析结果" if label else "单文件分析结果")
        
        # 生成文件，使用label信息命名
        base_name = f"{label}_analysis" if label else "single_file_analysis"
        return self._generate_output_files(rows, output_dir, base_name)
    
    def _present_multiple_files(self, data: Dict[str, Any], output_dir: str, show_dtype: bool, show_shape: bool, show_kernel_names: bool, show_kernel_duration: bool, special_matmul: bool, show_timestamp: bool = False, show_readable_timestamp: bool = False, aggregation_spec: str = 'name', compare_dtype: bool = False, compare_shape: bool = False, file_labels: List[str] = None, print_markdown: bool = False) -> List[Path]:
        """展示多文件数据"""
        print("生成多文件展示结果...")
        
        # 如果显示kernel duration，先计算每个文件的总耗时
        file_total_durations = {}
        if show_kernel_duration:
            for key, entry in data.items():
                for label, file_data in entry['files'].items():
                    if label not in file_total_durations:
                        file_total_durations[label] = 0.0
                    kernel_events = file_data['kernel_events']
                    kernel_stats = self._calculate_kernel_statistics(kernel_events)
                    if kernel_stats:
                        file_total_durations[label] += sum(stats.mean_duration * stats.count for stats in kernel_stats)
        
        rows = []
        for key, entry in data.items():
            # 初始化行数据
            row = {}
            
            # 添加时间戳列（如果启用，作为第一列）
            if show_timestamp:
                # 获取第一个文件的第一个CPU事件的启动时间戳
                first_timestamp = None
                for label, file_data in entry['files'].items():
                    cpu_events = file_data['cpu_events']
                    if cpu_events:
                        first_cpu_event = cpu_events[0]
                        if show_readable_timestamp and first_cpu_event.readable_timestamp:
                            first_timestamp = first_cpu_event.readable_timestamp
                            break
                        elif first_cpu_event.ts is not None:
                            first_timestamp = first_cpu_event.ts
                            break
                row['cpu_start_timestamp'] = first_timestamp if first_timestamp is not None else 0.0
            
            # 根据聚合字段组合生成相应的列
            aggregation_fields = self._parse_aggregation_fields(aggregation_spec)
            
            # 解析键的各个部分
            if isinstance(key, tuple):
                key_parts = list(key)
            else:
                key_parts = [key]
            
            # 为每个字段添加对应的列
            for i, field in enumerate(aggregation_fields):
                if i < len(key_parts) and key_parts[i] is not None:
                    if field == 'name':
                        row['op_name'] = key_parts[i]
                    elif field == 'shape':
                        row['input_dims'] = str(key_parts[i])
                    elif field == 'call_stack':
                        call_stack_str = ' -> '.join(key_parts[i]) if isinstance(key_parts[i], tuple) else str(key_parts[i])
                        row['call_stack'] = call_stack_str
                    elif field == 'dtype':
                        row['input_type'] = str(key_parts[i])
                else:
                    if field == 'name':
                        row['op_name'] = 'None'
                    elif field == 'shape':
                        row['input_dims'] = 'None'
                    elif field == 'call_stack':
                        row['call_stack'] = 'None'
                    elif field == 'dtype':
                        row['input_type'] = 'None'
            
            # 为每个文件添加数据
            for label, file_data in entry['files'].items():
                cpu_events = file_data['cpu_events']
                kernel_events = file_data['kernel_events']
                
                row[f'{label}_cpu_event_count'] = len(cpu_events)
                
                if show_dtype:
                    dtypes = set()
                    for cpu_event in cpu_events:
                        args = cpu_event.args or {}
                        input_types = args.get('Input type', [])
                        # 将每个event的输入类型信息作为一个整体来记录
                        if input_types:
                            dtypes.add(str(input_types))
                    row[f'{label}_dtypes'] = '\n'.join(sorted(dtypes)) if dtypes else ''
                
                if show_shape:
                    shapes = set()
                    strides = set()
                    for cpu_event in cpu_events:
                        args = cpu_event.args or {}
                        input_dims = args.get('Input Dims', [])
                        input_strides = args.get('Input Strides', [])
                        # 将每个event的输入维度信息作为一个整体来记录
                        if input_dims:
                            shapes.add(str(input_dims))
                        if input_strides:
                            strides.add(str(input_strides))
                    row[f'{label}_shapes'] = '\n'.join(sorted(shapes)) if shapes else ''
                    row[f'{label}_strides'] = '\n'.join(sorted(strides)) if strides else ''
                
                if show_kernel_names or show_kernel_duration:
                    kernel_stats = self._calculate_kernel_statistics(kernel_events)
                    if kernel_stats:
                        row[f'{label}_kernel_count'] = sum(stats.count for stats in kernel_stats)
                        if show_kernel_names:
                            row[f'{label}_kernel_names'] = '\n'.join(stats.kernel_name for stats in kernel_stats)
                        if show_kernel_duration:
                            op_total_duration = sum(stats.mean_duration * stats.count for stats in kernel_stats)
                            row[f'{label}_kernel_mean_duration'] = op_total_duration / sum(stats.count for stats in kernel_stats)
                            row[f'{label}_kernel_min_duration'] = min(stats.min_duration for stats in kernel_stats)
                            row[f'{label}_kernel_max_duration'] = max(stats.max_duration for stats in kernel_stats)
                            row[f'{label}_kernel_total_duration'] = op_total_duration
                            row[f'{label}_kernel_duration_ratio'] = (op_total_duration / file_total_durations[label] * 100) if file_total_durations[label] > 0 else 0.0
                    else:
                        row[f'{label}_kernel_count'] = 0
                        if show_kernel_names:
                            row[f'{label}_kernel_names'] = ''
                        if show_kernel_duration:
                            row[f'{label}_kernel_mean_duration'] = 0.0
                            row[f'{label}_kernel_min_duration'] = 0.0
                            row[f'{label}_kernel_max_duration'] = 0.0
                            row[f'{label}_kernel_total_duration'] = 0.0
                            row[f'{label}_kernel_duration_ratio'] = 0.0
            
            # 添加比较列
            if compare_dtype or compare_shape or show_kernel_duration:
                # 获取所有文件的标签
                file_labels = list(entry['files'].keys())
                if len(file_labels) >= 2:
                    # 比较前两个文件的 dtype 和 shape
                    label1, label2 = file_labels[0], file_labels[1]
                    
                    if compare_dtype:
                        dtypes1 = set()
                        dtypes2 = set()
                        for cpu_event in entry['files'][label1]['cpu_events']:
                            args = cpu_event.args or {}
                            input_types = args.get('Input type', [])
                            if input_types:
                                # 将单个event的input_types转换为字符串
                                dtypes1.add(str(input_types))
                        for cpu_event in entry['files'][label2]['cpu_events']:
                            args = cpu_event.args or {}
                            input_types = args.get('Input type', [])
                            if input_types:
                                # 将单个event的input_types转换为字符串
                                dtypes2.add(str(input_types))
                        row['dtype_equal'] = dtypes1 == dtypes2
                    
                    if compare_shape:
                        shapes1 = set()
                        shapes2 = set()
                        for cpu_event in entry['files'][label1]['cpu_events']:
                            args = cpu_event.args or {}
                            input_dims = args.get('Input Dims', [])
                            if input_dims:
                                # 将单个event的input_dims转换为字符串
                                shapes1.add(str(input_dims))
                        for cpu_event in entry['files'][label2]['cpu_events']:
                            args = cpu_event.args or {}
                            input_dims = args.get('Input Dims', [])
                            if input_dims:
                                # 将单个event的input_dims转换为字符串
                                shapes2.add(str(input_dims))
                        row['shape_equal'] = shapes1 == shapes2
                    
                    # 添加 mean duration ratio 比较
                    if show_kernel_duration:
                        mean_duration1 = row.get(f'{label1}_kernel_mean_duration', 0.0)
                        mean_duration2 = row.get(f'{label2}_kernel_mean_duration', 0.0)
                        
                        if mean_duration1 > 0 and mean_duration2 > 0:
                            # 计算 ratio: label2 / label1 (即第二个标签相对于第一个标签的性能比例)
                            ratio = mean_duration2 / mean_duration1
                            row[f'{label2}_vs_{label1}_mean_duration_ratio'] = ratio
                            
                            # 计算性能提升百分比
                            if ratio < 1:
                                improvement = (1 - ratio) * 100
                                row[f'{label2}_vs_{label1}_performance_improvement'] = f"{improvement:.1f}%"
                            else:
                                degradation = (ratio - 1) * 100
                                row[f'{label2}_vs_{label1}_performance_improvement'] = f"-{degradation:.1f}%"
                        else:
                            row[f'{label2}_vs_{label1}_mean_duration_ratio'] = 0.0
                            row[f'{label2}_vs_{label1}_performance_improvement'] = "N/A"
            
            rows.append(row)
        
        # 如果启用markdown打印，在stdout中打印表格
        if print_markdown:
            title = f"{file_labels[0]} vs {file_labels[1]} 对比分析" if file_labels and len(file_labels) >= 2 else "多文件对比分析"
            # 对于多文件分析，优先按照mean_duration_ratio排序，如果没有则按照第一个文件的kernel_duration_ratio排序
            if file_labels and len(file_labels) >= 2:
                ratio_key = f'{file_labels[1]}_vs_{file_labels[0]}_mean_duration_ratio'
                if any(ratio_key in row for row in rows):
                    sorted_rows = sorted(rows, key=lambda x: x.get(ratio_key, 0), reverse=True)
                else:
                    first_label = file_labels[0]
                    ratio_key = f'{first_label}_kernel_duration_ratio'
                    sorted_rows = sorted(rows, key=lambda x: x.get(ratio_key, 0), reverse=True)
            elif file_labels and len(file_labels) >= 1:
                first_label = file_labels[0]
                ratio_key = f'{first_label}_kernel_duration_ratio'
                sorted_rows = sorted(rows, key=lambda x: x.get(ratio_key, 0), reverse=True)
            else:
                sorted_rows = rows
            self._print_markdown_table(sorted_rows, title)
        
        # 生成文件，使用标签信息命名
        if file_labels and len(file_labels) >= 2:
            base_name = f"{file_labels[0]}_vs_{file_labels[1]}_analysis"
        else:
            base_name = "multiple_files_analysis"
        
        generated_files = self._generate_output_files(rows, output_dir, base_name)
        
        # 特殊的 matmul 展示
        if special_matmul:
            matmul_files = self._present_special_matmul(data, output_dir)
            generated_files.extend(matmul_files)
        
        return generated_files
    
    def _present_special_matmul(self, data: Dict[str, Any], output_dir: str) -> List[Path]:
        """特殊的 matmul 展示：抓取matmul 系列kernel 信息，形成 kernel mean duration ratio vs min(m,k,n) 的表格和折线图"""
        print("生成特殊的 matmul 展示...")
        
        matmul_data = {}
        
        for key, entry in data.items():
            # 检查是否是 matmul 相关的键
            if 'aten::mm' in key or 'matmul' in key.lower():
                # 提取维度信息
                dimensions = self._extract_matmul_dimensions(key)
                if dimensions:
                    m, k, n = dimensions
                    min_dim = min(m, k, n)
                    
                    if min_dim not in matmul_data:
                        matmul_data[min_dim] = {}
                    
                    # 为每个文件收集数据
                    for label, file_data in entry['files'].items():
                        kernel_events = file_data['kernel_events']
                        kernel_stats = self._calculate_kernel_statistics(kernel_events)
                        
                        if kernel_stats:
                            mean_duration = sum(stats.mean_duration * stats.count for stats in kernel_stats) / sum(stats.count for stats in kernel_stats)
                            matmul_data[min_dim][label] = mean_duration
        
        if matmul_data:
            # 生成 matmul 分析文件
            return self._generate_matmul_analysis(matmul_data, output_dir)
        else:
            return []
    
    def _extract_matmul_dimensions(self, key: str) -> Optional[Tuple[int, int, int]]:
        """从键中提取 matmul 维度信息"""
        try:
            # 使用正则表达式提取数字
            pattern = r'\(\((\d+),\s*(\d+)\)\s*,\s*\((\d+),\s*(\d+)\)\)'
            match = re.search(pattern, key)
            
            if match:
                m, k1, k2, n = map(int, match.groups())
                # 验证k1 == k2 (矩阵乘法的要求)
                if k1 == k2:
                    return (m, k1, n)
            
            return None
        except Exception:
            return None
    
    def _calculate_kernel_statistics(self, kernel_events: List[ActivityEvent]) -> List[KernelStatistics]:
        """计算 kernel 事件的统计信息"""
        if not kernel_events:
            return []
        
        # 按 kernel name 分组
        kernel_groups = defaultdict(list)
        for event in kernel_events:
            kernel_groups[event.name].append(event)
        
        statistics_list = []
        
        for kernel_name, events in kernel_groups.items():
            if len(events) == 0:
                continue
                
            # 计算持续时间统计
            durations = [event.dur for event in events if event.dur is not None]
            
            if not durations:
                continue
                
            mean_duration = statistics.mean(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            # 计算方差
            if len(durations) > 1:
                variance = statistics.variance(durations)
            else:
                variance = 0.0
            
            stats = KernelStatistics(
                kernel_name=kernel_name,
                min_duration=min_duration,
                max_duration=max_duration,
                mean_duration=mean_duration,
                variance=variance,
                count=len(events)
            )
            
            statistics_list.append(stats)
        
        return statistics_list
    
    def _generate_output_files(self, rows: List[Dict], output_dir: str, base_name: str) -> List[Path]:
        """生成输出文件（JSON 和 XLSX）"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # 生成 JSON 文件
        json_file = output_path / f"{base_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"JSON 文件已生成: {json_file}")
        generated_files.append(json_file)
        
        # 生成 Excel 文件
        if rows:
            df = pd.DataFrame(rows)
            xlsx_file = output_path / f"{base_name}.xlsx"
            try:
                df.to_excel(xlsx_file, index=False)
                print(f"Excel 文件已生成: {xlsx_file}")
                generated_files.append(xlsx_file)
            except ImportError:
                csv_file = output_path / f"{base_name}.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8')
                print(f"CSV 文件已生成: {csv_file}")
                generated_files.append(csv_file)
        else:
            print("没有数据可以生成 Excel 文件")
        
        return generated_files
    
    def _generate_matmul_analysis(self, matmul_data: Dict[int, Dict[str, float]], output_dir: str) -> List[Path]:
        """生成 matmul 分析文件"""
        output_path = Path(output_dir)
        
        # 生成数据行
        rows = []
        for min_dim in sorted(matmul_data.keys()):
            row = {'min_dim': min_dim}
            for label, duration in matmul_data[min_dim].items():
                row[f'{label}_mean_duration'] = duration
            rows.append(row)
        
        # 生成文件
        generated_files = self._generate_output_files(rows, output_dir, "matmul_analysis")
        
        # 生成图表
        chart_file = self._generate_matmul_chart(matmul_data, output_path)
        if chart_file:
            generated_files.append(chart_file)
        
        return generated_files
    
    def _generate_matmul_chart(self, matmul_data: Dict[int, Dict[str, float]], output_path: Path) -> Optional[Path]:
        """生成 matmul 性能图表"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 收集所有标签
            all_labels = set()
            for data in matmul_data.values():
                all_labels.update(data.keys())
            
            # 为每个标签绘制一条线
            for label in sorted(all_labels):
                x_values = []
                y_values = []
                
                for min_dim in sorted(matmul_data.keys()):
                    if label in matmul_data[min_dim]:
                        x_values.append(min_dim)
                        y_values.append(matmul_data[min_dim][label])
                
                if x_values and y_values:
                    plt.plot(x_values, y_values, marker='o', label=label, linewidth=2, markersize=6)
            
            plt.xlabel('Matmul Min Dimension (m/k/n)', fontsize=12)
            plt.ylabel('Mean Duration (μs)', fontsize=12)
            plt.title('Matmul Performance Analysis by Min Dimension', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图表
            chart_file = output_path / "matmul_performance_chart.jpg"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Matmul 性能图表已生成: {chart_file}")
            return chart_file
        except Exception as e:
            print(f"生成 matmul 图表时出错: {e}")
            return None
    
    # ==================== 完整的分析流程 ====================
    
    def analyze_single_file(self, file_path: str, aggregation_spec: str = 'name', 
                           show_dtype: bool = True, show_shape: bool = True, show_kernel_names: bool = True, show_kernel_duration: bool = True, 
                           show_timestamp: bool = False, show_readable_timestamp: bool = False, show_kernel_timestamp: bool = False,
                           output_dir: str = ".", label: str = None, print_markdown: bool = False) -> List[Path]:
        """
        分析单个文件的完整流程
        
        Args:
            file_path: JSON 文件路径
            aggregation_spec: 聚合字段组合
            show_dtype: 是否展示 dtype 信息
            show_shape: 是否展示 shape 和 strides 信息
            show_kernel_names: 是否展示 kernel 名称信息
            show_kernel_duration: 是否展示 kernel 持续时间信息
            show_timestamp: 是否展示 CPU 操作启动时间戳
            show_readable_timestamp: 是否展示可读时间戳
            show_kernel_timestamp: 是否展示 kernel 时间戳
            output_dir: 输出目录
            label: 文件标签
            print_markdown: 是否打印markdown表格
        """
        print(f"=== 开始分析单个文件: {file_path} ===")
        
        # 加载数据
        data = self.parser.load_json_file(file_path)
        
        # Stage 1: 数据后处理
        cpu_events_by_external_id, kernel_events_by_external_id = self.stage1_data_postprocessing(data)
        
        # Stage 2: 数据聚合
        aggregated_data = self.stage2_data_aggregation(cpu_events_by_external_id, kernel_events_by_external_id, aggregation_spec)
        
        # Stage 3: 比较（单文件无需比较）
        comparison_result = self.stage3_comparison(single_file_data=aggregated_data, aggregation_spec=aggregation_spec)
        
        # Stage 4: 展示
        generated_files = self.stage4_presentation(comparison_result, output_dir, show_dtype, show_shape, show_kernel_names, show_kernel_duration, show_timestamp, show_readable_timestamp, show_kernel_timestamp, aggregation_spec=aggregation_spec, label=label, print_markdown=print_markdown)
        
        print("=== 单文件分析完成 ===")
        return generated_files
    
    def analyze_single_file_with_glob(self, file_paths: List[str], aggregation_spec: str = 'name', 
                                    show_dtype: bool = True, show_shape: bool = True, show_kernel_names: bool = True, show_kernel_duration: bool = True, 
                                    show_timestamp: bool = False, show_readable_timestamp: bool = False, show_kernel_timestamp: bool = False,
                                    output_dir: str = ".", label: str = None, print_markdown: bool = False) -> List[Path]:
        """
        分析多个文件的完整流程，每个文件独立解析，然后一起聚合
        
        Args:
            file_paths: JSON 文件路径列表
            aggregation_spec: 聚合字段组合
            show_dtype: 是否展示 dtype 信息
            show_shape: 是否展示 shape 和 strides 信息
            show_kernel_names: 是否展示 kernel 名称信息
            show_kernel_duration: 是否展示 kernel 持续时间信息
            show_timestamp: 是否展示 CPU 操作启动时间戳
            show_readable_timestamp: 是否展示可读时间戳
            show_kernel_timestamp: 是否展示 kernel 时间戳
            output_dir: 输出目录
            label: 文件标签
            print_markdown: 是否打印markdown表格
        """
        print(f"=== 开始分析多个文件，共 {len(file_paths)} 个文件 ===")
        
        if len(file_paths) == 1:
            # 只有一个文件，直接使用原来的方法
            return self.analyze_single_file(
                file_path=file_paths[0],
                aggregation_spec=aggregation_spec,
                show_dtype=show_dtype,
                show_shape=show_shape,
                show_kernel_names=show_kernel_names,
                show_kernel_duration=show_kernel_duration,
                show_timestamp=show_timestamp,
                show_readable_timestamp=show_readable_timestamp,
                show_kernel_timestamp=show_kernel_timestamp,
                output_dir=output_dir,
                label=label,
                print_markdown=print_markdown
            )
        
        # 多个文件，每个文件独立解析
        print("每个文件独立解析，然后一起聚合...")
        
        # 使用多进程并行处理文件
        max_workers = min(mp.cpu_count(), len(file_paths))
        print(f"使用 {max_workers} 个进程并行处理文件")
        
        # 并行处理所有文件
        aggregated_data_list = self._process_files_parallel(file_paths, aggregation_spec, max_workers)
        
        # 合并所有文件的聚合数据
        print("合并所有文件的聚合数据...")
        merged_aggregated_data = self._merge_same_label_files(aggregated_data_list, aggregation_spec)
        
        # Stage 3: 比较（单标签多文件无需比较）
        comparison_result = self.stage3_comparison(single_file_data=merged_aggregated_data, aggregation_spec=aggregation_spec)
        
        # Stage 4: 展示
        generated_files = self.stage4_presentation(comparison_result, output_dir, show_dtype, show_shape, show_kernel_names, show_kernel_duration, show_timestamp, show_readable_timestamp, show_kernel_timestamp, aggregation_spec=aggregation_spec, label=label, print_markdown=print_markdown)
        
        print("=== 多文件分析完成 ===")
        return generated_files
    
    def analyze_multiple_files(self, file_labels: List[Tuple[List[str], str]], aggregation_spec: str = 'name',
                              show_dtype: bool = True, show_shape: bool = True, show_kernel_names: bool = True, show_kernel_duration: bool = True, 
                              show_timestamp: bool = False, show_readable_timestamp: bool = False, show_kernel_timestamp: bool = False, special_matmul: bool = False, output_dir: str = ".", compare_dtype: bool = False, compare_shape: bool = False, print_markdown: bool = False, 
                              max_workers: Optional[int] = None) -> List[Path]:
        """
        分析多个文件的完整流程，支持同一label下多个文件的聚合，使用多进程并行读取JSON文件
        
        Args:
            file_labels: [(file_paths, label), ...] 文件路径列表和标签的列表
                        每个label可以对应多个文件，会自动聚合
            aggregation_spec: 聚合字段组合
            show_dtype: 是否展示 dtype 信息
            show_shape: 是否展示 shape 和 strides 信息
            show_kernel_names: 是否展示 kernel 名称信息
            show_kernel_duration: 是否展示 kernel 持续时间信息
            show_timestamp: 是否展示 CPU 操作启动时间戳
            show_readable_timestamp: 是否展示可读时间戳
            show_kernel_timestamp: 是否展示 kernel 时间戳
            special_matmul: 是否进行特殊的 matmul 展示
            output_dir: 输出目录
            compare_dtype: 是否添加 dtype 比较列
            compare_shape: 是否添加 shape 比较列
            print_markdown: 是否打印markdown表格
            max_workers: 最大工作进程数，默认为CPU核心数
        """
        # 计算总文件数
        total_files = sum(len(file_paths) for file_paths, _ in file_labels)
        print(f"=== 开始分析多个文件，共 {len(file_labels)} 个标签，{total_files} 个文件 ===")
        
        # 设置进程数
        if max_workers is None:
            max_workers = min(mp.cpu_count(), total_files)
        
        print(f"使用 {max_workers} 个进程并行处理文件")
        
        # 分析每个标签下的文件
        multiple_files_data = {}
        
        for file_paths, label in file_labels:
            print(f"正在分析标签: {label} ({len(file_paths)} 个文件)")
            
            # 使用多进程并行处理文件
            label_aggregated_data_list = self._process_files_parallel(file_paths, aggregation_spec, max_workers)
            
            # 聚合同一标签下的多个文件
            if len(label_aggregated_data_list) == 1:
                # 只有一个文件，直接使用
                multiple_files_data[label] = label_aggregated_data_list[0]
            else:
                # 多个文件，进行聚合
                print(f"  聚合 {len(label_aggregated_data_list)} 个文件的数据...")
                multiple_files_data[label] = self._merge_same_label_files(label_aggregated_data_list, aggregation_spec)
        
        # Stage 3: 比较（多文件比较）
        comparison_result = self.stage3_comparison(multiple_files_data=multiple_files_data, aggregation_spec=aggregation_spec)
        
        # Stage 4: 展示
        file_labels_list = [label for _, label in file_labels]
        generated_files = self.stage4_presentation(comparison_result, output_dir, show_dtype, show_shape, show_kernel_names, show_kernel_duration, show_timestamp, show_readable_timestamp, show_kernel_timestamp, special_matmul, aggregation_spec, compare_dtype, compare_shape, file_labels_list, print_markdown=print_markdown)
        
        print("=== 多文件分析完成 ===")
        return generated_files
    
    def _process_files_parallel(self, file_paths: List[str], aggregation_spec: str, max_workers: int) -> List[Dict[Union[str, tuple], 'AggregatedData']]:
        """
        使用多进程并行处理文件列表
        
        Args:
            file_paths: 文件路径列表
            aggregation_spec: 聚合字段组合
            max_workers: 最大工作进程数
            
        Returns:
            List[Dict]: 聚合后的数据列表
        """
        if len(file_paths) == 1:
            # 只有一个文件，直接处理
            file_path = file_paths[0]
            print(f"  处理文件: {file_path}")
            _, aggregated_data = _process_single_file_parallel(file_path, aggregation_spec)
            return [aggregated_data]
        
        # 多个文件，使用进程池并行处理
        aggregated_data_list = []
        completed_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(_process_single_file_parallel, file_path, aggregation_spec): file_path 
                for file_path in file_paths
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed_count += 1
                
                try:
                    _, aggregated_data = future.result()
                    aggregated_data_list.append(aggregated_data)
                    print(f"  完成文件 {completed_count}/{len(file_paths)}: {file_path}")
                except Exception as e:
                    print(f"  处理文件失败 {completed_count}/{len(file_paths)}: {file_path}, 错误: {e}")
                    # 添加空数据以保持索引一致
                    aggregated_data_list.append({})
        
        return aggregated_data_list
    
    def _merge_same_label_files(self, aggregated_data_list: List[Dict[Union[str, tuple], 'AggregatedData']], aggregation_spec: str) -> Dict[Union[str, tuple], 'AggregatedData']:
        """
        聚合同一标签下的多个文件的聚合数据
        
        Args:
            aggregated_data_list: 多个文件的聚合数据列表
            aggregation_spec: 聚合字段组合
            
        Returns:
            合并后的聚合数据
        """
        if not aggregated_data_list:
            return {}
        
        if len(aggregated_data_list) == 1:
            return aggregated_data_list[0]
        
        # 收集所有可能的键
        all_keys = set()
        for data in aggregated_data_list:
            all_keys.update(data.keys())
        
        merged_data = {}
        
        for key in all_keys:
            # 收集该键在所有文件中的数据
            key_data_list = []
            for data in aggregated_data_list:
                if key in data:
                    key_data_list.append(data[key])
            
            if not key_data_list:
                continue
            
            # 合并该键的数据
            merged_aggregated_data = self._merge_aggregated_data_for_key(key_data_list)
            merged_data[key] = merged_aggregated_data
        
        return merged_data
    
    def _merge_aggregated_data_for_key(self, aggregated_data_list: List['AggregatedData']) -> 'AggregatedData':
        """
        合并同一个键在多个文件中的AggregatedData
        
        Args:
            aggregated_data_list: 同一个键在多个文件中的AggregatedData列表
            
        Returns:
            合并后的AggregatedData
        """
        if not aggregated_data_list:
            return None
        
        if len(aggregated_data_list) == 1:
            return aggregated_data_list[0]
        
        # 取第一个作为模板
        template = aggregated_data_list[0]
        
        # 合并所有kernel事件
        all_kernel_events = []
        for data in aggregated_data_list:
            all_kernel_events.extend(data.kernel_events)
        
        # 合并所有CPU事件
        all_cpu_events = []
        for data in aggregated_data_list:
            all_cpu_events.extend(data.cpu_events)
        
        # 创建合并后的AggregatedData
        merged_data = AggregatedData(
            cpu_events=all_cpu_events,
            kernel_events=all_kernel_events,
            key=template.key  # 使用第一个数据的key
        )
        
        return merged_data
    
    def _print_markdown_table(self, rows: List[Dict], title: str):
        """在stdout中以markdown格式打印表格"""
        if not rows:
            print(f"\n## {title}\n\n无数据可显示\n")
            return
        
        print(f"\n## {title}\n")
        
        # 获取所有列名
        columns = list(rows[0].keys())
        
        # 打印表头
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        print(header)
        print(separator)
        
        # 打印数据行
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, "")
                # 处理特殊值
                if isinstance(value, float):
                    if col.endswith('_ratio'):
                        if 'mean_duration_ratio' in col:
                            # mean_duration_ratio 显示为比例形式
                            values.append(f"{value:.3f}")
                        else:
                            values.append(f"{value:.2f}%")
                    elif col.endswith('_duration'):
                        values.append(f"{value:.2f}")
                    else:
                        values.append(f"{value:.2f}")
                elif isinstance(value, str) and '\n' in value:
                    # 对于包含换行符的字符串，只显示前50个字符
                    display_value = value.replace('\n', ', ')[:50]
                    if len(value) > 50:
                        display_value += "..."
                    values.append(display_value)
                else:
                    values.append(str(value))
            
            data_row = "| " + " | ".join(values) + " |"
            print(data_row)
        
        print()  # 添加空行
    
    # ==================== 突变点检测工具函数 ====================
    
    def _print_combined_change_points(self, comparison_rows: List[Dict], cpu_start_time_change_points: List[int], 
                                    kernel_duration_change_points: List[int]):
        """
        打印合并的突变点信息，按照操作执行顺序排列
        
        Args:
            comparison_rows: 对比分析结果行
            cpu_start_time_change_points: CPU启动时间差异突变点索引
            kernel_duration_change_points: Kernel持续时间差异突变点索引
        """
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
                change_types.append("Kernel")
            change_type_str = " + ".join(change_types)
            
            print(f"\n事件 {cp_idx + 1}: {row['cpu_op_name']} [{change_type_str}突变]")
            
            # 显示当前事件信息
            print(f"  当前事件:")
            print(f"    最快卡时间戳: {row['fastest_cpu_start_time']:.2f}μs")
            print(f"    最慢卡时间戳: {row['slowest_cpu_start_time']:.2f}μs")
            print(f"    时间戳差异: {row['cpu_start_time_diff']:.2f}μs")
            print(f"    CPU启动时间差异比例: {row['cpu_start_time_diff_ratio']:.4f}")
            print(f"    Kernel持续时间差异比例: {row['kernel_duration_diff_ratio']:.4f}")
            if row.get('fastest_cpu_readable_timestamp'):
                print(f"    最快卡可读时间戳: {row['fastest_cpu_readable_timestamp']}")
            if row.get('slowest_cpu_readable_timestamp'):
                print(f"    最慢卡可读时间戳: {row['slowest_cpu_readable_timestamp']}")
            
            # 显示突变前后的对比
            if cp_idx > 0:
                prev_row = comparison_rows[cp_idx - 1]
                print(f"  突变前事件 {cp_idx}: {prev_row['cpu_op_name']}")
                print(f"    CPU启动时间差异比例: {prev_row['cpu_start_time_diff_ratio']:.4f}")
                print(f"    Kernel持续时间差异比例: {prev_row['kernel_duration_diff_ratio']:.4f}")
            
            if cp_idx < len(comparison_rows) - 1:
                next_row = comparison_rows[cp_idx + 1]
                print(f"  突变后事件 {cp_idx + 2}: {next_row['cpu_op_name']}")
                print(f"    CPU启动时间差异比例: {next_row['cpu_start_time_diff_ratio']:.4f}")
                print(f"    Kernel持续时间差异比例: {next_row['kernel_duration_diff_ratio']:.4f}")
            
            # 计算突变幅度
            if is_cpu_change and cp_idx > 0:
                prev_cpu_ratio = comparison_rows[cp_idx - 1]['cpu_start_time_diff_ratio']
                curr_cpu_ratio = row['cpu_start_time_diff_ratio']
                cpu_change_magnitude = curr_cpu_ratio - prev_cpu_ratio
                print(f"  CPU启动时间差异比例变化: {cpu_change_magnitude:.4f}")
            
            if is_kernel_change and cp_idx > 0:
                prev_kernel_ratio = comparison_rows[cp_idx - 1]['kernel_duration_diff_ratio']
                curr_kernel_ratio = row['kernel_duration_diff_ratio']
                kernel_change_magnitude = curr_kernel_ratio - prev_kernel_ratio
                print(f"  Kernel持续时间差异比例变化: {kernel_change_magnitude:.4f}")

    # ==================== 突变点检测工具函数 ====================
    
    def detect_change_points(self, data: List[Dict], ratio_column: str, threshold: float = 0.3) -> List[int]:
        """
        检测数据中的突变点
        
        Args:
            data: 数据列表，每个元素是包含ratio_column的字典
            ratio_column: 要检测的比率列名（如'cpu_start_time_diff_ratio'或'kernel_duration_diff_ratio'）
            threshold: 突变阈值，超过此值认为是突变点
            
        Returns:
            List[int]: 突变点的索引列表
        """
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
    
    def extract_change_point_data(self, data: List[Dict], change_points: List[int], context_size: int = 3) -> List[Dict]:
        """
        提取突变点前后的数据
        
        Args:
            data: 原始数据列表
            change_points: 突变点索引列表
            context_size: 每个突变点前后保留的行数
            
        Returns:
            List[Dict]: 包含突变点上下文的数据列表
        """
        if not change_points:
            return []
        
        extracted_data = []
        for cp_idx in change_points:
            # 计算上下文范围
            start_idx = max(0, cp_idx - context_size)
            end_idx = min(len(data), cp_idx + context_size + 1)
            
            # 添加分隔符
            if extracted_data:
                extracted_data.append({
                    'cpu_op_name': '--- 突变点分隔符 ---',
                    'cpu_start_time_diff_ratio': '',
                    'kernel_duration_diff_ratio': '',
                    'fastest_cpu_start_time': '',
                    'slowest_cpu_start_time': '',
                    'cpu_start_time_diff': '',
                    'fastest_cpu_readable_timestamp': '',
                    'slowest_cpu_readable_timestamp': '',
                    'is_change_point': False,
                    'change_point_index': -1
                })
            
            # 添加上下文数据
            for i in range(start_idx, end_idx):
                row = data[i].copy()
                row['is_change_point'] = (i == cp_idx)
                row['change_point_index'] = cp_idx
                extracted_data.append(row)
        
        return extracted_data

    # ==================== All-to-All 通信性能分析 ====================
    
    def analyze_communication_performance(self, pod_dir: str, step: Optional[int] = None, comm_idx: Optional[int] = None, 
                                         fastest_card_idx: Optional[int] = None, slowest_card_idx: Optional[int] = None,
                                         kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL", prev_kernel_pattern: str = "TCDP_.*", output_dir: str = ".",
                                         show_dtype: bool = False, show_shape: bool = False, show_kernel_names: bool = False, 
                                         show_kernel_duration: bool = False, show_timestamp: bool = False, 
                                         show_readable_timestamp: bool = False, show_kernel_timestamp: bool = False) -> List[Path]:
        """
        分析分布式训练中的通信性能
        
        Args:
            pod_dir: Pod文件夹路径
            step: 指定要分析的step，如果为None则分析所有step
            comm_idx: 指定要分析的通信操作索引，如果为None则分析所有通信操作
            fastest_card_idx: 指定最快卡的索引，如果为None则自动查找
            slowest_card_idx: 指定最慢卡的索引，如果为None则自动查找
            kernel_prefix: 要检测的通信kernel前缀
            prev_kernel_pattern: 上一个通信kernel的匹配模式，用于确定对比区间
            output_dir: 输出目录
            show_dtype: 是否显示数据类型信息
            show_shape: 是否显示形状和步长信息
            show_kernel_names: 是否显示kernel名称
            show_kernel_duration: 是否显示kernel持续时间
            show_timestamp: 是否显示时间戳
            show_readable_timestamp: 是否显示可读时间戳
            show_kernel_timestamp: 是否显示kernel时间戳
            
        Returns:
            List[Path]: 生成的文件路径列表
        """
        print(f"=== 开始通信性能分析 ===")
        print(f"Pod目录: {pod_dir}")
        print(f"Step: {step if step is not None else '所有step'}")
        print(f"通信操作索引: {comm_idx if comm_idx is not None else '所有通信操作'}")
        print(f"通信Kernel前缀: {kernel_prefix}")
        print(f"上一个通信Kernel模式: {prev_kernel_pattern}")
        
        # 1. 扫描pod目录，找到所有executor文件夹
        executor_folders = self._scan_executor_folders(pod_dir)
        if not executor_folders:
            print(f"错误: 在 {pod_dir} 中没有找到executor文件夹")
            return []
        
        print(f"找到 {len(executor_folders)} 个executor文件夹")
        
        # 2. 解析所有JSON文件，提取通信数据
        comm_data = None
        if step is None or comm_idx is None or fastest_card_idx is None or slowest_card_idx is None:
            comm_data = self._extract_communication_data(executor_folders, step, kernel_prefix)
            if not comm_data:
                print("错误: 没有找到任何通信数据")
                return []
        
            print(f"成功提取通信数据，共 {len(comm_data)} 个step")
        
        generated_files = []
        
        # 3. 如果指定了step和comm_idx，进行深度分析
        if step is not None and comm_idx is not None:
            print(f"进行深度分析: step={step}, comm_idx={comm_idx}")
            deep_analysis_file = self._perform_deep_analysis(comm_data, executor_folders, step, comm_idx, output_dir, kernel_prefix, prev_kernel_pattern, fastest_card_idx, slowest_card_idx)
            if deep_analysis_file:
                generated_files.append(deep_analysis_file)
        else:
            # 4. 生成原始数据Excel文件
            raw_data_file = self._generate_raw_data_excel(comm_data, output_dir)
            generated_files.append(raw_data_file)
            
            # 5. 生成统计分析Excel文件
            stats_file = self._generate_statistics_excel(comm_data, output_dir)
            generated_files.append(stats_file)
        
        print("=== 通信性能分析完成 ===")
        
        return generated_files
    
    def _scan_executor_folders(self, pod_dir: str) -> List[str]:
        """
        扫描pod目录，找到所有executor文件夹
        
        Args:
            pod_dir: Pod目录路径
            
        Returns:
            List[str]: executor文件夹路径列表
        """
        executor_folders = []
        pod_path = Path(pod_dir)
        
        # 查找所有executor_trainer-runner_*_*_*格式的文件夹
        pattern = "executor_trainer-runner_*_*_*"
        for folder in pod_path.glob(pattern):
            if folder.is_dir():
                executor_folders.append(str(folder))
        
        return sorted(executor_folders)
    
    def _extract_communication_data(self, executor_folders: List[str], step: Optional[int] = None, kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL") -> Dict[int, Dict[int, List[float]]]:
        """
        从executor文件夹中提取通信数据
        
        Args:
            executor_folders: executor文件夹路径列表
            step: 指定要处理的step，如果为None则处理所有step
            kernel_prefix: 要检测的通信kernel前缀
            
        Returns:
            Dict[int, Dict[int, List[float]]]: {step: {card_idx: [duration1, duration2, ...]}}
        """
        comm_data = defaultdict(lambda: defaultdict(list))
        
        for executor_folder in executor_folders:
            print(f"  处理文件夹: {executor_folder}")
            
            # 查找所有JSON文件
            json_files = list(Path(executor_folder).glob("*.json"))
            if not json_files:
                print(f"    警告: 文件夹中没有JSON文件")
                continue
            
            for json_file in json_files:
                # 解析文件名获取step和card_idx
                file_step, card_idx = self._parse_json_filename(json_file.name)
                if file_step is None or card_idx is None:
                    print(f"    警告: 无法解析文件名 {json_file.name}")
                    continue
                
                # 如果指定了step参数，只处理匹配的step
                if step is not None and file_step != step:
                    continue
                
                # 提取通信kernel durations
                durations = self._extract_communication_durations(str(json_file), kernel_prefix)
                if durations:
                    comm_data[file_step][card_idx].extend(durations)
                    print(f"    文件 {json_file.name}: step={file_step}, card={card_idx}, 提取到 {len(durations)} 个duration")
        
        return dict(comm_data)
    
    def _parse_json_filename(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """
        解析JSON文件名，提取step和card_idx
        
        Args:
            filename: JSON文件名，格式如 trace_ctr_torch_v310_hot_start01_r10495354_0_7_321.json
            
        Returns:
            Tuple[Optional[int], Optional[int]]: (step, card_idx)
        """
        try:
            # 使用正则表达式提取step和card_idx
            # 文件名格式: trace_ctr_torch_v310_hot_start01_r10495354_0_7_321.json
            # 其中7是card_idx，321是step
            pattern = r'.*_(\d+)_(\d+)\.json$'
            match = re.match(pattern, filename)
            
            if match:
                card_idx = int(match.group(1))
                step = int(match.group(2))
                return step, card_idx
            else:
                return None, None
        except Exception as e:
            print(f"    解析文件名失败 {filename}: {e}")
            return None, None
    
    def _find_communication_kernels(self, json_file_path: str, kernel_pattern: str = "TCDP_.*") -> List[Dict]:
        """
        从JSON文件中找到所有匹配模式的通信kernel，返回其时间戳和持续时间
        
        Args:
            json_file_path: JSON文件路径
            kernel_pattern: kernel名称的匹配模式（正则表达式）
            
        Returns:
            List[Dict]: 包含kernel信息的列表，每个元素包含 {'name': str, 'ts': float, 'dur': float, 'idx': int}
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用正则表达式匹配所有通信kernel
            pattern = rf'"ph":\s*"X",\s*"cat":\s*"kernel",\s*"name":\s*"({kernel_pattern}[^"]*)",\s*"pid":\s*\d+,\s*"tid":\s*\d+,\s*"ts":\s*([\d.]+),\s*"dur":\s*([\d.]+)'
            
            matches = re.findall(pattern, content, re.DOTALL)
            
            # 转换为字典列表
            kernels = []
            for idx, (name, ts, dur) in enumerate(matches):
                kernels.append({
                    'name': name,
                    'ts': float(ts),
                    'dur': float(dur),
                    'idx': idx
                })
            
            # 按时间戳排序
            kernels.sort(key=lambda x: x['ts'])
            
            return kernels
            
        except Exception as e:
            print(f"    读取JSON文件失败 {json_file_path}: {e}")
            return []

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
        """
        生成原始数据Excel文件
        
        Args:
            all2all_data: {step: {card_idx: [duration1, duration2, ...]}}
            output_dir: 输出目录
            
        Returns:
            Path: 生成的文件路径
        """
        print("生成原始数据Excel文件...")
        
        rows = []
        for step in sorted(all2all_data.keys()):
            for card_idx in sorted(all2all_data[step].keys()):
                durations = all2all_data[step][card_idx]
                for all2all_idx, duration in enumerate(durations):
                    rows.append({
                        'step': step,
                        'card_idx': card_idx,
                        'all2all_idx': all2all_idx,
                        'duration': duration
                    })
        
        # 创建DataFrame并保存
        df = pd.DataFrame(rows)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        excel_file = output_path / "comm_raw_data.xlsx"
        df.to_excel(excel_file, index=False)
        print(f"原始数据Excel文件已生成: {excel_file}")
        
        return excel_file
    
    def _generate_statistics_excel(self, all2all_data: Dict[int, Dict[int, List[float]]], output_dir: str) -> Path:
        """
        生成统计分析Excel文件
        
        Args:
            all2all_data: {step: {card_idx: [duration1, duration2, ...]}}
            output_dir: 输出目录
            
        Returns:
            Path: 生成的文件路径
        """
        print("生成统计分析Excel文件...")
        
        rows = []
        for step in sorted(all2all_data.keys()):
            step_data = all2all_data[step]
            
            # 为每个all2all_idx生成统计信息
            max_all2all_idx = max(max(len(durations) for durations in card_data.values()) for card_data in [step_data])
            
            for all2all_idx in range(max_all2all_idx):
                # 收集该all2all_idx下所有card的duration
                card_durations = {}
                for card_idx, durations in step_data.items():
                    if all2all_idx < len(durations):
                        card_durations[card_idx] = durations[all2all_idx]
                
                if not card_durations:
                    continue
                
                # 计算统计信息
                durations_list = list(card_durations.values())
                durations_list.sort()
                
                # 最快和最慢的5个card
                fastest_5_cards = durations_list[:5]
                slowest_5_cards = durations_list[-5:][::-1]  # 反转，从最慢到最快
                
                # 统计信息
                mean_duration = statistics.mean(durations_list)
                min_duration = min(durations_list)
                max_duration = max(durations_list)
                variance = statistics.variance(durations_list) if len(durations_list) > 1 else 0.0
                std_dev = variance ** 0.5
                
                # 最大最小值的差值和比例
                duration_diff = max_duration - min_duration
                duration_ratio = max_duration / min_duration if min_duration > 0 else 0.0
                
                # 找到对应的card_idx
                fastest_5_cards_with_idx = []
                slowest_5_cards_with_idx = []
                
                for duration in fastest_5_cards:
                    for card_idx, dur in card_durations.items():
                        if dur == duration and (card_idx, dur) not in fastest_5_cards_with_idx:
                            fastest_5_cards_with_idx.append((card_idx, dur))
                            break
                
                for duration in slowest_5_cards:
                    for card_idx, dur in card_durations.items():
                        if dur == duration and (card_idx, dur) not in slowest_5_cards_with_idx:
                            slowest_5_cards_with_idx.append((card_idx, dur))
                            break
                
                # 构建行数据
                row = {
                    'step': step,
                    'all2all_idx': all2all_idx,
                    'card_count': len(card_durations),
                    'mean_duration': mean_duration,
                    'min_duration': min_duration,
                    'max_duration': max_duration,
                    'std_deviation': std_dev,
                    'variance': variance,
                    'duration_diff': duration_diff,
                    'duration_ratio': duration_ratio
                }
                
                # 添加最快和最慢的5个card
                for i in range(5):
                    if i < len(fastest_5_cards_with_idx):
                        card_idx, duration = fastest_5_cards_with_idx[i]
                        row[f'fastest_card_{i+1}_idx'] = card_idx
                        row[f'fastest_card_{i+1}_duration'] = duration
                    else:
                        row[f'fastest_card_{i+1}_idx'] = None
                        row[f'fastest_card_{i+1}_duration'] = None
                
                for i in range(5):
                    if i < len(slowest_5_cards_with_idx):
                        card_idx, duration = slowest_5_cards_with_idx[i]
                        row[f'slowest_card_{i+1}_idx'] = card_idx
                        row[f'slowest_card_{i+1}_duration'] = duration
                    else:
                        row[f'slowest_card_{i+1}_idx'] = None
                        row[f'slowest_card_{i+1}_duration'] = None
                
                rows.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(rows)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        excel_file = output_path / "comm_statistics.xlsx"
        df.to_excel(excel_file, index=False)
        print(f"统计分析Excel文件已生成: {excel_file}")
        
        return excel_file
    
    def _perform_deep_analysis(self, comm_data: Dict[int, Dict[int, List[float]]], executor_folders: List[str], 
                              step: int, comm_idx: int, output_dir: str, kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL", 
                              prev_kernel_pattern: str = "TCDP_.*", fastest_card_idx: Optional[int] = None, slowest_card_idx: Optional[int] = None) -> Optional[Path]:
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
        fastest_data = self.parser.load_json_file(fastest_json_file)
        slowest_data = self.parser.load_json_file(slowest_json_file)
        
        # 5. 进行深度对比分析
        comparison_result = self._compare_card_performance(
            fastest_data, slowest_data, fastest_card[0], slowest_card[0], 
            step, comm_idx, fastest_card[1], slowest_card[1], kernel_prefix, prev_kernel_pattern
        )
        
        if not comparison_result:
            print("错误: 深度对比分析失败")
            return None
        
        # 6. 生成深度分析Excel文件
        excel_file = self._generate_deep_analysis_excel(comparison_result, step, comm_idx, output_dir)
        
        print("=== 深度分析完成 ===")
        return excel_file
    
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
    
    def _compare_card_performance(self, fastest_data: 'ProfilerData', slowest_data: 'ProfilerData',
                                 fastest_card_idx: int, slowest_card_idx: int,
                                 step: int, comm_idx: int, fastest_duration: float, slowest_duration: float,
                                 kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL", 
                                 prev_kernel_pattern: str = "TCDP_.*") -> Optional[Dict]:
        """
        比较快卡和慢卡的性能差异
        
        Args:
            fastest_data: 最快card的ProfilerData
            slowest_data: 最慢card的ProfilerData
            fastest_card_idx: 最快card的索引
            slowest_card_idx: 最慢card的索引
            step: step值
            comm_idx: 通信操作索引
            fastest_duration: 最快card的通信kernel duration
            slowest_duration: 最慢card的通信kernel duration
            kernel_prefix: 要检测的通信kernel前缀
            prev_kernel_pattern: 上一个通信kernel的匹配模式，用于确定对比区间
            
        Returns:
            Optional[Dict]: 对比分析结果
        """
        print("开始比较快卡和慢卡的性能差异...")
        
        # 1. 检查通信kernel一致性
        if not self._check_communication_kernel_consistency(fastest_data, slowest_data, kernel_prefix, comm_idx):
            print("错误: 通信kernel一致性检查失败")
            return None
        
        # 2. 对两个数据进行Stage1处理
        fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id = self.stage1_data_postprocessing(fastest_data)
        slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id = self.stage1_data_postprocessing(slowest_data)
        
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
                print(f"    最快卡: {fastest_prev_kernel_name}")
                print(f"    最慢卡: {slowest_prev_kernel_name}")
                print(f"    要求: 两个卡的上一个通信kernel名称必须完全一致")
                return None
            else:
                print(f"    ✓ 上一个通信kernel名称一致: {fastest_prev_kernel_name}")
        elif not fastest_prev_comm_events and not slowest_prev_comm_events:
            print("    警告: 快慢卡都没有找到上一个通信kernel")
        else:
            print("    错误: 快慢卡的上一个通信kernel查找结果不一致!")
            print(f"    最快卡找到: {len(fastest_prev_comm_events)} 个events")
            print(f"    最慢卡找到: {len(slowest_prev_comm_events)} 个events")
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
        
        # 7. 时间戳已经在stage1_data_postprocessing中标准化，无需额外处理
        
        # 8. 提取时间范围内的events并与filtered events取交集
        fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id = self._extract_events_in_range_with_intersection(
            fastest_data, fastest_start_time, fastest_end_time, 
            fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id
        )
        slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id = self._extract_events_in_range_with_intersection(
            slowest_data, slowest_start_time, slowest_end_time,
            slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id
        )
        
        # 9. 合并CPU和kernel events
        fastest_events_by_external_id = self._merge_cpu_and_kernel_events(fastest_cpu_events_by_external_id, fastest_kernel_events_by_external_id)
        slowest_events_by_external_id = self._merge_cpu_and_kernel_events(slowest_cpu_events_by_external_id, slowest_kernel_events_by_external_id)
        
        # 10. 计算通信kernel的实际duration
        fastest_comm_duration = fastest_comm_event.dur if fastest_comm_event and fastest_comm_event.dur is not None else 0.0
        slowest_comm_duration = slowest_comm_event.dur if slowest_comm_event and slowest_comm_event.dur is not None else 0.0
        
        # 11. 按时间顺序比较CPU操作
        comparison_rows = self._compare_events_by_time_sequence(
            fastest_events_by_external_id, slowest_events_by_external_id,
            fastest_card_idx, slowest_card_idx, fastest_comm_duration, slowest_comm_duration
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
            'cpu_start_time_change_points': comparison_rows['cpu_start_time_change_points'],
            'kernel_duration_change_points': comparison_rows['kernel_duration_change_points'],
            'all_change_points': comparison_rows['all_change_points'],
            'change_point_data': comparison_rows['change_point_data']
        }
    
    def _find_communication_event(self, data: 'ProfilerData', comm_idx: int, kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL") -> Optional['ActivityEvent']:
        """
        找到指定comm_idx的通信kernel操作event
        
        Args:
            data: ProfilerData对象
            comm_idx: 通信操作索引
            kernel_prefix: 要检测的通信kernel前缀
            
        Returns:
            Optional[ActivityEvent]: 通信kernel event，如果找到则返回单个event，否则返回None
        """
        comm_events = []
        comm_count = 0
        
        for event in data.events:
            if (event.cat == 'kernel' and 
                event.name.startswith(kernel_prefix)):
                # 检查是否在黑名单中
                is_blacklisted = any(pattern in event.name for pattern in self.COMMUNICATION_BLACKLIST_PATTERNS)
                if not is_blacklisted:
                    if comm_count == comm_idx:
                        comm_events.append(event)
                    comm_count += 1
        
        # 按结束时间排序（从早到晚）
        comm_events.sort(key=lambda x: (x.ts + x.dur) if (x.ts is not None and x.dur is not None) else 0)
        
        # 打印找到的通信事件的可读时间戳
        if comm_events:
            print(f"    找到 1 个目标通信kernel event:")
            event = comm_events[0]  # 取第一个（按结束时间排序后的第一个）
            start_time = event.readable_timestamp if event.readable_timestamp else f"{event.ts:.2f}μs"
            end_time = "N/A"
            if event.ts is not None and event.dur is not None:
                if event.readable_timestamp:
                    import datetime
                    start_dt = datetime.datetime.strptime(event.readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
                    end_dt = start_dt + datetime.timedelta(microseconds=event.dur)
                    end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                else:
                    end_time = f"{event.ts + event.dur:.2f}μs"
            print(f"      {event.name}")
            print(f"         开始时间: {start_time}")
            print(f"         结束时间: {end_time}")
            return event
        
        return None
    
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
        comm_events.sort(key=lambda x: (x.ts + x.dur) if (x.ts is not None and x.dur is not None) else 0)
        
        return comm_events
    
    def _find_prev_communication_kernel_events(self, data: 'ProfilerData', target_kernel_events: List['ActivityEvent'], 
                                             prev_kernel_pattern: str = "TCDP_.*") -> List['ActivityEvent']:
        """
        找到目标通信kernel之前的通信kernel events
        
        Args:
            data: ProfilerData对象
            target_kernel_events: 目标通信kernel events
            prev_kernel_pattern: 上一个通信kernel的匹配模式
            
        Returns:
            List[ActivityEvent]: 上一个通信kernel events列表
        """
        if not target_kernel_events:
            print("    警告: 目标通信kernel events为空")
            return []
        
        # 找到目标kernel的最早开始时间
        target_start_time = min(event.ts for event in target_kernel_events if event.ts is not None)
        target_kernel_name = target_kernel_events[0].name
        print(f"    目标通信kernel: {target_kernel_name}, 开始时间: {target_start_time:.2f}")
        
        # 找到所有匹配模式的通信kernel events，按结束时间排序
        communication_kernels = []
        for event in data.events:
            if (event.cat == 'kernel' and
                event.ts is not None and
                event.ts < target_start_time and
                re.match(prev_kernel_pattern, event.name)):
                # 检查是否在黑名单中
                is_blacklisted = any(pattern in event.name for pattern in self.COMMUNICATION_BLACKLIST_PATTERNS)
                if not is_blacklisted:
                    communication_kernels.append(event)
        
        print(f"    找到 {len(communication_kernels)} 个匹配模式 '{prev_kernel_pattern}' 的通信kernel events（已过滤机内通信）")
        
        # 按结束时间排序，取最后一个（最接近目标kernel的）
        communication_kernels.sort(key=lambda x: (x.ts + x.dur) if (x.ts is not None and x.dur is not None) else 0)
        
        if communication_kernels:
            # 返回最后一个通信kernel（最接近目标kernel的）
            last_kernel_event = communication_kernels[-1]
            print(f"    上一个通信kernel: {last_kernel_event.name}")
            
            # 打印找到的上一个通信事件的可读时间戳
            start_time = last_kernel_event.readable_timestamp if last_kernel_event.readable_timestamp else f"{last_kernel_event.ts:.2f}μs"
            end_time = "N/A"
            if last_kernel_event.ts is not None and last_kernel_event.dur is not None:
                if last_kernel_event.readable_timestamp:
                    import datetime
                    start_dt = datetime.datetime.strptime(last_kernel_event.readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
                    end_dt = start_dt + datetime.timedelta(microseconds=last_kernel_event.dur)
                    end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                else:
                    end_time = f"{last_kernel_event.ts + last_kernel_event.dur:.2f}μs"
            print(f"    开始时间: {start_time}")
            print(f"    结束时间: {end_time}")
            
            return [last_kernel_event]
        else:
            print(f"    警告: 没有找到匹配模式 '{prev_kernel_pattern}' 的上一个通信kernel")
        
        return []
    
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
        
        # 3. 打印通信事件的markdown表格
        print("    通信事件对比表格:")
        print("    | 序号 | 最快卡Kernel名称 | 最快卡开始时间 | 最快卡结束时间 | 最慢卡Kernel名称 | 最慢卡开始时间 | 最慢卡结束时间 | 名称一致 | 开始时间差(ms) | 结束时间差(ms) |")
        print("    |------|----------------|-------------|-------------|----------------|-------------|-------------|----------|---------------|---------------|")
        
        # 按顺序对比通信事件
        max_events = max(len(fastest_all_comm_events), len(slowest_all_comm_events))
        
        for i in range(max_events):
            # 获取当前索引的事件
            fastest_event = fastest_all_comm_events[i] if i < len(fastest_all_comm_events) else None
            slowest_event = slowest_all_comm_events[i] if i < len(slowest_all_comm_events) else None
            
            # 获取kernel名称
            fastest_name = fastest_event.name if fastest_event else "N/A"
            slowest_name = slowest_event.name if slowest_event else "N/A"
            
            # 获取开始时间戳
            if fastest_event and fastest_event.readable_timestamp:
                fastest_start_ts = fastest_event.readable_timestamp
            elif fastest_event:
                fastest_start_ts = f"{fastest_event.ts:.2f}μs"
            else:
                fastest_start_ts = "N/A"
            
            if slowest_event and slowest_event.readable_timestamp:
                slowest_start_ts = slowest_event.readable_timestamp
            elif slowest_event:
                slowest_start_ts = f"{slowest_event.ts:.2f}μs"
            else:
                slowest_start_ts = "N/A"
            
            # 计算结束时间戳
            if fastest_event and fastest_event.ts is not None and fastest_event.dur is not None:
                fastest_end_ts = fastest_event.ts + fastest_event.dur
                if fastest_event.readable_timestamp:
                    # 如果有可读时间戳，计算结束时间戳
                    import datetime
                    start_dt = datetime.datetime.strptime(fastest_event.readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
                    end_dt = start_dt + datetime.timedelta(microseconds=fastest_event.dur)
                    fastest_end_ts_str = end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                else:
                    fastest_end_ts_str = f"{fastest_end_ts:.2f}μs"
            else:
                fastest_end_ts_str = "N/A"
                fastest_end_ts = None
            
            if slowest_event and slowest_event.ts is not None and slowest_event.dur is not None:
                slowest_end_ts = slowest_event.ts + slowest_event.dur
                if slowest_event.readable_timestamp:
                    # 如果有可读时间戳，计算结束时间戳
                    import datetime
                    start_dt = datetime.datetime.strptime(slowest_event.readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
                    end_dt = start_dt + datetime.timedelta(microseconds=slowest_event.dur)
                    slowest_end_ts_str = end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                else:
                    slowest_end_ts_str = f"{slowest_end_ts:.2f}μs"
            else:
                slowest_end_ts_str = "N/A"
                slowest_end_ts = None
            
            # 检查名称是否一致
            name_consistent = "✓" if fastest_name == slowest_name else "✗"
            
            # 计算开始时间差（如果有两个事件）
            start_time_diff = "N/A"
            if fastest_event and slowest_event and fastest_event.ts is not None and slowest_event.ts is not None:
                diff_ms = abs(fastest_event.ts - slowest_event.ts) / 1000.0  # 转换为毫秒
                start_time_diff = f"{diff_ms:.3f}"
            
            # 计算结束时间差（如果有两个事件）
            end_time_diff = "N/A"
            if fastest_end_ts is not None and slowest_end_ts is not None:
                diff_ms = abs(fastest_end_ts - slowest_end_ts) / 1000.0  # 转换为毫秒
                end_time_diff = f"{diff_ms:.3f}"
            
            print(f"    | {i+1} | {fastest_name} | {fastest_start_ts} | {fastest_end_ts_str} | {slowest_name} | {slowest_start_ts} | {slowest_end_ts_str} | {name_consistent} | {start_time_diff} | {end_time_diff} |")
        
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
        
        # 3. 检查kernel结束时间一致性（1ms以内）
        print("    检查通信kernel结束时间一致性...")
        
        if sequence_consistent:
            # 如果序列一致，按顺序比较
            print("    按顺序比较通信kernel结束时间...")
            for i, (fastest_event, slowest_event) in enumerate(zip(fastest_all_comm_events, slowest_all_comm_events)):
                fastest_end_time = fastest_event.ts + fastest_event.dur if fastest_event.ts is not None and fastest_event.dur is not None else 0
                slowest_end_time = slowest_event.ts + slowest_event.dur if slowest_event.ts is not None and slowest_event.dur is not None else 0
                
                time_diff = abs(fastest_end_time - slowest_end_time)
                print(f"      Kernel {i+1} ({fastest_event.name}):")
                print(f"        最快卡结束时间: {fastest_end_time:.2f}")
                print(f"        最慢卡结束时间: {slowest_end_time:.2f}")
                print(f"        时间差: {time_diff:.2f} ms")
                
                if time_diff > 5000.0:  # 5ms以内
                    print(f"        错误: 时间差超过5000us阈值!")
                    return False
                else:
                    print(f"        ✓ 时间差在5000us以内")
        else:
            # 如果序列不一致，检查相同kernel的结束时间
            print("    序列不一致，检查相同kernel的结束时间...")
            
            # 创建最快卡和最慢卡的kernel映射 {name: [events]}
            fastest_kernel_map = {}
            for event in fastest_all_comm_events:
                if event.name not in fastest_kernel_map:
                    fastest_kernel_map[event.name] = []
                fastest_kernel_map[event.name].append(event)
            
            slowest_kernel_map = {}
            for event in slowest_all_comm_events:
                if event.name not in slowest_kernel_map:
                    slowest_kernel_map[event.name] = []
                slowest_kernel_map[event.name].append(event)
            
            # 检查共同的kernel
            common_kernels = set(fastest_kernel_map.keys()) & set(slowest_kernel_map.keys())
            print(f"    共同kernel: {sorted(common_kernels)}")
            
            if not common_kernels:
                print("    错误: 没有共同的通信kernel!")
                return False
            
            # 对每个共同的kernel，比较结束时间
            for kernel_name in sorted(common_kernels):
                fastest_kernels = fastest_kernel_map[kernel_name]
                slowest_kernels = slowest_kernel_map[kernel_name]
                
                print(f"      检查kernel: {kernel_name}")
                print(f"        最快卡有 {len(fastest_kernels)} 个实例")
                print(f"        最慢卡有 {len(slowest_kernels)} 个实例")
                
                # 如果实例数量不同，取最小值进行比较
                min_count = min(len(fastest_kernels), len(slowest_kernels))
                
                for i in range(min_count):
                    fastest_event = fastest_kernels[i]
                    slowest_event = slowest_kernels[i]
                    
                    fastest_end_time = fastest_event.ts + fastest_event.dur if fastest_event.ts is not None and fastest_event.dur is not None else 0
                    slowest_end_time = slowest_event.ts + slowest_event.dur if slowest_event.ts is not None and slowest_event.dur is not None else 0
                    
                    time_diff = abs(fastest_end_time - slowest_end_time)
                    print(f"        实例 {i+1}:")
                    print(f"          最快卡结束时间: {fastest_end_time:.2f}")
                    print(f"          最慢卡结束时间: {slowest_end_time:.2f}")
                    print(f"          时间差: {time_diff:.2f} ms")
                    
                    if time_diff > 5000.0:  # 5ms以内
                        print(f"          错误: 时间差超过5000us阈值!")
                        return False
                    else:
                        print(f"          ✓ 时间差在5000us以内")
        
        print("    ✓ 通信kernel一致性检查通过")
        return True
    
    def _get_analysis_start_time(self, data: 'ProfilerData', prev_all2all_event: Optional['ActivityEvent'], 
                                current_all2all_event: 'ActivityEvent') -> float:
        """
        获取分析开始时间
        
        Args:
            data: ProfilerData对象
            prev_all2all_event: 上一次all2all event
            current_all2all_event: 当前all2all event
            
        Returns:
            float: 分析开始时间
        """
        if prev_all2all_event and prev_all2all_event.dur is not None:
            # 使用上一次all2all kernel的结束时间
            return prev_all2all_event.ts + prev_all2all_event.dur
        else:
            return 0.0
    
    def _get_analysis_end_time(self, data: 'ProfilerData', all2all_event: 'ActivityEvent') -> float:
        """
        获取分析结束时间
        
        Args:
            data: ProfilerData对象
            all2all_event: all2all event
            
        Returns:
            float: 分析结束时间（使用目标通信kernel的开始时间）
        """
        if all2all_event and all2all_event.ts is not None:
            # 使用目标通信kernel的开始时间，而不是结束时间
            return all2all_event.ts
        else:
            return float('inf')
    
    def _extract_events_in_range_with_intersection(self, data: 'ProfilerData', start_time: float, end_time: float,
                                                 filtered_cpu_events_by_external_id: Dict[Union[int, str], List['ActivityEvent']],
                                                 filtered_kernel_events_by_external_id: Dict[Union[int, str], List['ActivityEvent']]) -> Tuple[Dict[Union[int, str], List['ActivityEvent']], Dict[Union[int, str], List['ActivityEvent']]]:
        """
        提取时间范围内的events并与filtered events取交集
        
        Args:
            data: ProfilerData对象
            start_time: 开始时间
            end_time: 结束时间
            filtered_cpu_events_by_external_id: 过滤后的CPU events分组
            filtered_kernel_events_by_external_id: 过滤后的kernel events分组
            
        Returns:
            Tuple[Dict[external_id, List[ActivityEvent]], Dict[external_id, List[ActivityEvent]]]: 
                (CPU events分组, kernel events分组)
        """
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
    
    def _merge_cpu_and_kernel_events(self, cpu_events_by_external_id: Dict[Union[int, str], List['ActivityEvent']], 
                                   kernel_events_by_external_id: Dict[Union[int, str], List['ActivityEvent']]) -> Dict[Union[int, str], List['ActivityEvent']]:
        """
        合并CPU和kernel events
        
        Args:
            cpu_events_by_external_id: CPU events分组
            kernel_events_by_external_id: kernel events分组
            
        Returns:
            Dict[external_id, List[ActivityEvent]]: 合并后的events分组
        """
        merged_events = {}
        
        # 添加所有CPU events
        for external_id, events in cpu_events_by_external_id.items():
            merged_events[external_id] = events.copy()
        
        # 添加所有kernel events
        for external_id, events in kernel_events_by_external_id.items():
            if external_id in merged_events:
                merged_events[external_id].extend(events)
            else:
                merged_events[external_id] = events.copy()
        
        return merged_events
    
    def _group_events_by_external_id(self, events: List['ActivityEvent']) -> Dict[Union[int, str], List['ActivityEvent']]:
        """
        按external_id分组events
        
        Args:
            events: events列表
            
        Returns:
            Dict[external_id, List[ActivityEvent]]: 按external_id分组的events
        """
        events_by_external_id = defaultdict(list)
        for event in events:
            if event.external_id is not None:
                events_by_external_id[event.external_id].append(event)
        return dict(events_by_external_id)
    
    def _compare_events_by_time_sequence(self, fastest_events_by_external_id: Dict[Union[int, str], List['ActivityEvent']],
                                        slowest_events_by_external_id: Dict[Union[int, str], List['ActivityEvent']],
                                        fastest_card_idx: int, slowest_card_idx: int,
                                        fastest_duration: float, slowest_duration: float) -> List[Dict]:
        """
        按照时间顺序比较快卡和慢卡的events（不依赖external_id）
        
        Args:
            fastest_events_by_external_id: 最快card的events分组
            slowest_events_by_external_id: 最慢card的events分组
            fastest_card_idx: 最快card索引
            slowest_card_idx: 最慢card索引
            fastest_duration: 最快card all2all duration
            slowest_duration: 最慢card all2all duration
            
        Returns:
            List[Dict]: 对比分析结果行
        """
        comparison_rows = []
        
        # 1. 提取所有CPU events并按时间排序
        fastest_cpu_events = []
        slowest_cpu_events = []
        
        for events in fastest_events_by_external_id.values():
            fastest_cpu_events.extend([e for e in events if e.cat == 'cpu_op'])
        for events in slowest_events_by_external_id.values():
            slowest_cpu_events.extend([e for e in events if e.cat == 'cpu_op'])
        
        # 按开始时间排序（ts已经标准化，直接使用）
        fastest_cpu_events.sort(key=lambda x: x.ts)
        slowest_cpu_events.sort(key=lambda x: x.ts)
        
        print(f"最快card找到 {len(fastest_cpu_events)} 个CPU events")
        print(f"最慢card找到 {len(slowest_cpu_events)} 个CPU events")
        
        # 2. 检查CPU操作是否一致（除了external_id）
        cpu_ops_consistent = self._check_cpu_ops_consistency(fastest_cpu_events, slowest_cpu_events)
        if not cpu_ops_consistent:
            print("警告: 快卡和慢卡的CPU操作不完全一致，比较结果可能不准确")
        else:
            print("CPU操作一致性检查通过")
        
        # 3. 检查kernel操作是否一致
        kernel_ops_consistent = self._check_kernel_ops_consistency(fastest_events_by_external_id, slowest_events_by_external_id)
        if not kernel_ops_consistent:
            print("警告: 快卡和慢卡的kernel操作不完全一致，比较结果可能不准确")
        else:
            print("Kernel操作一致性检查通过")
        
        # 4. 按照时间顺序进行配对比较
        # 假设两个card的CPU操作顺序相同，按时间顺序配对
        min_events_count = min(len(fastest_cpu_events), len(slowest_cpu_events))
        
        for i in range(min_events_count):
            fastest_cpu_event = fastest_cpu_events[i]
            slowest_cpu_event = slowest_cpu_events[i]
            
            # 4. 找到对应的kernel events
            fastest_kernel_events = self._find_kernel_events_for_cpu_event(
                fastest_cpu_event, fastest_events_by_external_id
            )
            slowest_kernel_events = self._find_kernel_events_for_cpu_event(
                slowest_cpu_event, slowest_events_by_external_id
            )
            
            # 5. 计算时间差异（ts已经标准化，直接使用）
            cpu_start_time_diff = slowest_cpu_event.ts - fastest_cpu_event.ts
            cpu_duration_diff = (slowest_cpu_event.dur or 0) - (fastest_cpu_event.dur or 0)
            
            # 6. 计算kernel时间差异
            fastest_kernel_duration = sum(e.dur for e in fastest_kernel_events if e.dur is not None)
            slowest_kernel_duration = sum(e.dur for e in slowest_kernel_events if e.dur is not None)
            kernel_duration_diff = slowest_kernel_duration - fastest_kernel_duration
            
            # 7. 计算时间差异占all2all duration差异的比例
            all2all_duration_diff = slowest_duration - fastest_duration
            cpu_start_time_diff_ratio = cpu_start_time_diff / all2all_duration_diff if all2all_duration_diff > 0 else 0
            kernel_duration_diff_ratio = kernel_duration_diff / all2all_duration_diff if all2all_duration_diff > 0 else 0
            
            # 8. 构建行数据
            row = {
                'event_sequence': i + 1,  # 事件序号
                'cpu_op_name': fastest_cpu_event.name,
                'cpu_op_shape': str(fastest_cpu_event.args.get('Input Dims', '') if fastest_cpu_event.args else ''),
                'cpu_op_dtype': str(fastest_cpu_event.args.get('Input type', '') if fastest_cpu_event.args else ''),
                'fastest_cpu_start_time': fastest_cpu_event.ts,  # 使用标准化的时间戳（毫秒）
                'slowest_cpu_start_time': slowest_cpu_event.ts,  # 使用标准化的时间戳（毫秒）
                'cpu_start_time_diff': cpu_start_time_diff,
                'cpu_start_time_diff_ratio': cpu_start_time_diff_ratio,
                'fastest_cpu_duration': fastest_cpu_event.dur or 0,
                'slowest_cpu_duration': slowest_cpu_event.dur or 0,
                'cpu_duration_diff': cpu_duration_diff,
                'fastest_kernel_duration': fastest_kernel_duration,
                'slowest_kernel_duration': slowest_kernel_duration,
                'kernel_duration_diff': kernel_duration_diff,
                'kernel_duration_diff_ratio': kernel_duration_diff_ratio,
                'fastest_card_idx': fastest_card_idx,
                'slowest_card_idx': slowest_card_idx,
                'fastest_cpu_readable_timestamp': fastest_cpu_event.readable_timestamp,
                'slowest_cpu_readable_timestamp': slowest_cpu_event.readable_timestamp
            }
            
            comparison_rows.append(row)
        
        print(f"成功比较了 {len(comparison_rows)} 个CPU events")
        
        # 9. 检测突变点
        print("\n=== 检测突变点 ===")
        cpu_start_time_change_points = self.detect_change_points(comparison_rows, 'cpu_start_time_diff_ratio', threshold=0.3)
        kernel_duration_change_points = self.detect_change_points(comparison_rows, 'kernel_duration_diff_ratio', threshold=0.3)
        
        print(f"CPU启动时间差异突变点: {cpu_start_time_change_points}")
        print(f"Kernel持续时间差异突变点: {kernel_duration_change_points}")
        
        # 10. 打印合并的突变点信息
        self._print_combined_change_points(comparison_rows, cpu_start_time_change_points, kernel_duration_change_points)
        
        # 11. 合并所有突变点
        all_change_points = sorted(list(set(cpu_start_time_change_points + kernel_duration_change_points)))
        
        # 12. 提取突变点数据
        change_point_data = self.extract_change_point_data(comparison_rows, all_change_points, context_size=3)
        
        return {
            'comparison_rows': comparison_rows,
            'cpu_start_time_change_points': cpu_start_time_change_points,
            'kernel_duration_change_points': kernel_duration_change_points,
            'all_change_points': all_change_points,
            'change_point_data': change_point_data
        }
    
    def _find_kernel_events_for_cpu_event(self, cpu_event: 'ActivityEvent', 
                                        events_by_external_id: Dict[Union[int, str], List['ActivityEvent']]) -> List['ActivityEvent']:
        """
        为CPU event找到对应的kernel events
        
        Args:
            cpu_event: CPU event
            events_by_external_id: 按external_id分组的events
            
        Returns:
            List[ActivityEvent]: 对应的kernel events
        """
        if cpu_event.external_id is None:
            return []
        
        if cpu_event.external_id not in events_by_external_id:
            return []
        
        events = events_by_external_id[cpu_event.external_id]
        kernel_events = [e for e in events if e.cat == 'kernel']
        
        return kernel_events
    
    def _check_cpu_ops_consistency(self, fastest_cpu_events: List['ActivityEvent'], 
                                 slowest_cpu_events: List['ActivityEvent']) -> bool:
        """
        检查快卡和慢卡的CPU操作是否一致（除了external_id）
        
        Args:
            fastest_cpu_events: 最快card的CPU events列表
            slowest_cpu_events: 最慢card的CPU events列表
            
        Returns:
            bool: 是否一致
        """
        if len(fastest_cpu_events) != len(slowest_cpu_events):
            print(f"CPU events数量不一致: 最快card {len(fastest_cpu_events)} 个, 最慢card {len(slowest_cpu_events)} 个")
            return False
        
        # 检查每个位置的CPU操作是否一致
        for i, (fastest_event, slowest_event) in enumerate(zip(fastest_cpu_events, slowest_cpu_events)):
            # 比较操作名称
            if fastest_event.name != slowest_event.name:
                print(f"位置 {i}: CPU操作名称不一致 - 最快: {fastest_event.name}, 最慢: {slowest_event.name}")
                return False
            
            # 比较操作参数（除了external_id）
            fastest_args = fastest_event.args or {}
            slowest_args = slowest_event.args or {}
            
            # 比较Input Dims
            fastest_dims = fastest_args.get('Input Dims', [])
            slowest_dims = slowest_args.get('Input Dims', [])
            if fastest_dims != slowest_dims:
                print(f"位置 {i}: Input Dims不一致 - 最快: {fastest_dims}, 最慢: {slowest_dims}")
                return False
            
            # 比较Input type
            fastest_type = fastest_args.get('Input type', [])
            slowest_type = slowest_args.get('Input type', [])
            if fastest_type != slowest_type:
                print(f"位置 {i}: Input type不一致 - 最快: {fastest_type}, 最慢: {slowest_type}")
                return False
            
            # 比较Input Strides
            fastest_strides = fastest_args.get('Input Strides', [])
            slowest_strides = slowest_args.get('Input Strides', [])
            if fastest_strides != slowest_strides:
                print(f"位置 {i}: Input Strides不一致 - 最快: {fastest_strides}, 最慢: {slowest_strides}")
                return False
        
        print(f"CPU操作一致性检查通过: 共 {len(fastest_cpu_events)} 个操作")
        return True
    
    def _check_kernel_ops_consistency(self, fastest_events_by_external_id: Dict[Union[int, str], List['ActivityEvent']],
                                    slowest_events_by_external_id: Dict[Union[int, str], List['ActivityEvent']]) -> bool:
        """
        检查快卡和慢卡的kernel操作是否一致（除了external_id）
        
        Args:
            fastest_events_by_external_id: 最快card的events分组
            slowest_events_by_external_id: 最慢card的events分组
            
        Returns:
            bool: 是否一致
        """
        # 提取所有kernel events
        fastest_kernel_events = []
        slowest_kernel_events = []
        
        for events in fastest_events_by_external_id.values():
            fastest_kernel_events.extend([e for e in events if e.cat == 'kernel'])
        for events in slowest_events_by_external_id.values():
            slowest_kernel_events.extend([e for e in events if e.cat == 'kernel'])
        
        # 按结束时间排序（ts已经标准化，直接使用）
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
    
    def _generate_deep_analysis_excel(self, comparison_result: Dict, step: int, comm_idx: int, output_dir: str) -> Path:
        """
        生成深度分析Excel文件
        
        Args:
            comparison_result: 对比分析结果
            step: step值
            comm_idx: 通信操作索引
            output_dir: 输出目录
            
        Returns:
            Path: 生成的Excel文件路径
        """
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
            'total_cpu_start_time_diff': df['cpu_start_time_diff'].sum(),
            'total_kernel_duration_diff': df['kernel_duration_diff'].sum(),
            'total_cpu_start_time_diff_ratio': df['cpu_start_time_diff_ratio'].sum(),
            'total_kernel_duration_diff_ratio': df['kernel_duration_diff_ratio'].sum()
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
            
            # 写入突变点数据
            if 'change_point_data' in comparison_result and comparison_result['change_point_data']:
                change_point_df = pd.DataFrame(comparison_result['change_point_data'])
                change_point_df.to_excel(writer, sheet_name='突变点分析', index=False)
                
                # 写入突变点汇总信息
                change_point_summary = {
                    'cpu_start_time_change_points': str(comparison_result.get('cpu_start_time_change_points', [])),
                    'kernel_duration_change_points': str(comparison_result.get('kernel_duration_change_points', [])),
                    'all_change_points': str(comparison_result.get('all_change_points', [])),
                    'total_change_points': len(comparison_result.get('all_change_points', []))
                }
                change_point_summary_df = pd.DataFrame([change_point_summary])
                change_point_summary_df.to_excel(writer, sheet_name='突变点汇总', index=False)
        
        print(f"深度分析Excel文件已生成: {excel_file}")
        
        return excel_file
    
    def _aggregate_on_op_timestamp(self, cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                                 kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]]) -> Dict[Union[str, tuple], AggregatedData]:
        """
        按CPU操作启动时间排序的聚合方法
        每个CPU操作单独一个条目，按启动时间排序
        
        Args:
            cpu_events_by_external_id: external_id -> cpu_events 映射
            kernel_events_by_external_id: external_id -> kernel_events 映射
            
        Returns:
            Dict[Union[str, tuple], AggregatedData]: 按时间排序的聚合数据
        """
        print("=== 按CPU操作启动时间排序聚合 ===")
        
        # 收集所有CPU事件
        all_cpu_events = []
        for external_id in cpu_events_by_external_id:
            cpu_events = cpu_events_by_external_id[external_id]
            all_cpu_events.extend(cpu_events)
        
        # 按启动时间排序
        all_cpu_events.sort(key=lambda x: x.ts if x.ts is not None else 0)
        
        print(f"找到 {len(all_cpu_events)} 个CPU操作，按启动时间排序")
        
        # 创建聚合数据，每个操作单独一个条目
        aggregated_data = {}
        
        for i, cpu_event in enumerate(all_cpu_events):
            # 使用序号作为键，确保唯一性
            key = f"op_{i:06d}_{cpu_event.name}"
            
            # 找到对应的kernel事件
            kernel_events = []
            if cpu_event.external_id is not None:
                kernel_events = kernel_events_by_external_id.get(cpu_event.external_id, [])
            
            # 创建聚合数据条目
            aggregated_data[key] = AggregatedData(
                cpu_events=[cpu_event],
                kernel_events=kernel_events,
                key=key
            )
        
        print(f"聚合后得到 {len(aggregated_data)} 个按时间排序的操作")
        
        return aggregated_data

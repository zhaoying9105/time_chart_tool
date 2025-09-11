"""
PyTorch Profiler 高级分析器 - 重构版本
按照4个stage重新组织：数据后处理、数据聚合、比较、展示
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
import statistics
from dataclasses import dataclass

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


class Analyzer:
    """重构后的分析器，按照4个stage组织"""
    
    def __init__(self):
        self.parser = PyTorchProfilerParser()
    
    # ==================== Stage 1: 数据后处理 ====================
    
    def stage1_data_postprocessing(self, data: ProfilerData) -> Tuple[Dict[Union[int, str], List[ActivityEvent]], Dict[Union[int, str], List[ActivityEvent]]]:
        """
        Stage 1: 数据后处理
        根据 cpu_op event & kernel event 的external id 分类，形成两个map
        然后对一个 external id 下的 cpu_event 进行call stack 合并，保留call stack 最长的哪个 event
        
        Args:
            data: ProfilerData 对象
            
        Returns:
            Tuple[Dict[external_id, cpu_events], Dict[external_id, kernel_events]]: 
                两个映射字典，确保 external id 和 cpu_event 是一对一的关系
        """
        print("=== Stage 1: 数据后处理 ===")
        
        # 1. 根据 external id 分类
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
        标准化 call stack，只保留包含 nn.Module 的有价值部分
        特殊处理：去掉 Runstep 模块及其之后的模块（面向 lg-torch 的特殊逻辑）
        
        Args:
            call_stack: 原始 call stack
            
        Returns:
            List[str]: 标准化后的 call stack，只包含模型相关的层级，去掉 'nn.Module: ' 前缀
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
            return []
        
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
    
    def stage2_data_aggregation(self, cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                               kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                               aggregation_type: str = 'on_op_name') -> Dict[Union[str, tuple], AggregatedData]:
        """
        Stage 2: 数据聚合
        分成三种聚合方式：
        - on_op_name: 形成cpu_op event name -> cpu_op event list & 对应 kernel event list extend
        - on_op_shape: 形成cpu_op event name & shape & strides -> cpu_op event list & 对应 kernel event list extend  
        - on_call_stack: 形成(cpu_op call stack & cpu_op_name) -> cpu_op event list & 对应 kernel event list extend
        
        Args:
            cpu_events_by_external_id: external_id -> cpu_events 映射
            kernel_events_by_external_id: external_id -> kernel_events 映射
            aggregation_type: 聚合类型 ('on_op_name', 'on_op_shape', 'on_call_stack')
            
        Returns:
            Dict[str, AggregatedData]: 聚合后的数据
        """
        print(f"=== Stage 2: 数据聚合 ({aggregation_type}) ===")
        
        if aggregation_type == 'on_call_stack':
            # on_call_stack 需要特殊处理，因为需要合并相似的调用栈
            return self._aggregate_on_call_stack(cpu_events_by_external_id, kernel_events_by_external_id)
        
        aggregated_data = defaultdict(lambda: AggregatedData([], [], ""))
        
        for external_id in cpu_events_by_external_id:
            cpu_events = cpu_events_by_external_id[external_id]
            kernel_events = kernel_events_by_external_id.get(external_id, [])
            
            for cpu_event in cpu_events:
                if aggregation_type == 'on_op_name':
                    key = cpu_event.name
                elif aggregation_type == 'on_op_shape':
                    # 组合 name, shape, strides 为 tuple
                    args = cpu_event.args or {}
                    input_dims = args.get('Input Dims', [])
                    input_strides = args.get('Input Strides', [])
                    # 递归地将嵌套列表转换为元组，使其可哈希
                    def list_to_tuple(obj):
                        if isinstance(obj, list):
                            return tuple(list_to_tuple(item) for item in obj)
                        return obj
                    key = (cpu_event.name, list_to_tuple(input_dims), list_to_tuple(input_strides))
                else:
                    raise ValueError(f"未知的聚合类型: {aggregation_type}")
                
                aggregated_data[key].cpu_events.append(cpu_event)
                aggregated_data[key].kernel_events.extend(kernel_events)
                aggregated_data[key].key = key
        
        print(f"聚合后得到 {len(aggregated_data)} 个不同的键")
        
        return dict(aggregated_data)
    
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
                         aggregation_type: str = 'on_op_name') -> Dict[str, Any]:
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
            return self._merge_multiple_files_data(multiple_files_data, aggregation_type)
        
        else:
            raise ValueError("必须提供 single_file_data 或 multiple_files_data 中的一个")
    
    def _merge_multiple_files_data(self, multiple_files_data: Dict[str, Dict[Union[str, tuple], AggregatedData]], aggregation_type: str = 'on_op_name') -> Dict[str, Any]:
        """
        合并多个文件的聚合数据
        
        Args:
            multiple_files_data: {label: {key: AggregatedData}}
            
        Returns:
            Dict[str, Any]: 合并后的数据
        """
        if aggregation_type == 'on_call_stack':
            # on_call_stack 需要特殊处理，合并相似的调用栈
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
                           special_matmul: bool = False,
                           aggregation_type: str = 'on_op_name',
                           compare_dtype: bool = False,
                           compare_shape: bool = False,
                           file_labels: List[str] = None,
                           label: str = None,
                           print_markdown: bool = False) -> List[Path]:
        """
        Stage 4: 展示
        根据配置决定是否展示 dtype 信息，是否展示 shape 信息，是否展示kernel 信息
        如果展示kernel 信息，那么包括kernel event 的name，duration的统计值，kernel event 的数量
        展示的结果是 包括xlsx 和 json
        
        Args:
            comparison_result: Stage 3 的比较结果
            output_dir: 输出目录
            show_dtype: 是否展示 dtype 信息
            show_shape: 是否展示 shape 和 strides 信息
            show_kernel: 是否展示 kernel 信息
            special_matmul: 是否进行特殊的 matmul 展示
        """
        print("=== Stage 4: 展示 ===")
        
        if comparison_result['type'] == 'single_file':
            return self._present_single_file(comparison_result['data'], output_dir, show_dtype, show_shape, show_kernel_names, show_kernel_duration, aggregation_type, label, print_markdown)
        elif comparison_result['type'] == 'multiple_files':
            return self._present_multiple_files(comparison_result['data'], output_dir, show_dtype, show_shape, show_kernel_names, show_kernel_duration, special_matmul, aggregation_type, compare_dtype, compare_shape, file_labels, print_markdown)
    
    def _present_single_file(self, data: Dict[Union[str, tuple], AggregatedData], output_dir: str, show_dtype: bool, show_shape: bool, show_kernel_names: bool, show_kernel_duration: bool, aggregation_type: str = 'on_op_name', label: str = None, print_markdown: bool = False) -> List[Path]:
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
            # 根据聚合类型生成相应的列
            if aggregation_type == 'on_op_name':
                # 字符串键：操作名称
                row = {
                    'op_name': key,
                    'cpu_event_count': len(aggregated_data.cpu_events)
                }
            elif aggregation_type == 'on_op_shape':
                # tuple键：(op_name, input_dims, input_strides)
                op_name, input_dims, input_strides = key
                row = {
                    'op_name': op_name,
                    'input_dims': str(input_dims),
                    'input_strides': str(input_strides),
                    'cpu_event_count': len(aggregated_data.cpu_events)
                }
            elif aggregation_type == 'on_call_stack':
                # tuple键：(call_stack_tuple, op_name)
                call_stack_tuple, op_name = key
                call_stack_str = ' -> '.join(call_stack_tuple)
                row = {
                    'call_stack': call_stack_str,
                    'op_name': op_name,
                    'cpu_event_count': len(aggregated_data.cpu_events)
                }
            else:
                # 未知聚合类型，使用通用格式
                row = {
                    'key': str(key),
                    'cpu_event_count': len(aggregated_data.cpu_events)
                }
            
            if show_dtype:
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
            
            if show_kernel_names or show_kernel_duration:
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
            
            rows.append(row)
        
        # 如果启用markdown打印，在stdout中打印表格
        if print_markdown:
            # 按照kernel_duration_ratio排序（从大到小）
            sorted_rows = sorted(rows, key=lambda x: x.get('kernel_duration_ratio', 0), reverse=True)
            self._print_markdown_table(sorted_rows, f"{label} 分析结果" if label else "单文件分析结果")
        
        # 生成文件，使用label信息命名
        base_name = f"{label}_analysis" if label else "single_file_analysis"
        return self._generate_output_files(rows, output_dir, base_name)
    
    def _present_multiple_files(self, data: Dict[str, Any], output_dir: str, show_dtype: bool, show_shape: bool, show_kernel_names: bool, show_kernel_duration: bool, special_matmul: bool, aggregation_type: str = 'on_op_name', compare_dtype: bool = False, compare_shape: bool = False, file_labels: List[str] = None, print_markdown: bool = False) -> List[Path]:
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
            # 根据聚合类型生成相应的列
            if aggregation_type == 'on_op_name':
                # 字符串键：操作名称
                row = {'op_name': key}
            elif aggregation_type == 'on_op_shape':
                # tuple键：(op_name, input_dims, input_strides)
                op_name, input_dims, input_strides = key
                row = {
                    'op_name': op_name,
                    'input_dims': str(input_dims),
                    'input_strides': str(input_strides)
                }
            elif aggregation_type == 'on_call_stack':
                # tuple键：(call_stack_tuple, op_name)
                call_stack_tuple, op_name = key
                call_stack_str = ' -> '.join(call_stack_tuple)
                row = {
                    'call_stack': call_stack_str,
                    'op_name': op_name
                }
            else:
                # 未知聚合类型，使用通用格式
                row = {'key': str(key)}
            
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
            if compare_dtype or compare_shape:
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
            
            rows.append(row)
        
        # 如果启用markdown打印，在stdout中打印表格
        if print_markdown:
            title = f"{file_labels[0]} vs {file_labels[1]} 对比分析" if file_labels and len(file_labels) >= 2 else "多文件对比分析"
            # 对于多文件分析，按照第一个文件的kernel_duration_ratio排序（从大到小）
            if file_labels and len(file_labels) >= 1:
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
    
    def analyze_single_file(self, file_path: str, aggregation_type: str = 'on_op_name', 
                           show_dtype: bool = True, show_shape: bool = True, show_kernel_names: bool = True, show_kernel_duration: bool = True, 
                           output_dir: str = ".", label: str = None, print_markdown: bool = False) -> List[Path]:
        """
        分析单个文件的完整流程
        
        Args:
            file_path: JSON 文件路径
            aggregation_type: 聚合类型
            show_dtype: 是否展示 dtype 信息
            show_shape: 是否展示 shape 和 strides 信息
            show_kernel: 是否展示 kernel 信息
            output_dir: 输出目录
        """
        print(f"=== 开始分析单个文件: {file_path} ===")
        
        # 加载数据
        data = self.parser.load_json_file(file_path)
        
        # Stage 1: 数据后处理
        cpu_events_by_external_id, kernel_events_by_external_id = self.stage1_data_postprocessing(data)
        
        # Stage 2: 数据聚合
        aggregated_data = self.stage2_data_aggregation(cpu_events_by_external_id, kernel_events_by_external_id, aggregation_type)
        
        # Stage 3: 比较（单文件无需比较）
        comparison_result = self.stage3_comparison(single_file_data=aggregated_data, aggregation_type=aggregation_type)
        
        # Stage 4: 展示
        generated_files = self.stage4_presentation(comparison_result, output_dir, show_dtype, show_shape, show_kernel_names, show_kernel_duration, aggregation_type=aggregation_type, label=label, print_markdown=print_markdown)
        
        print("=== 单文件分析完成 ===")
        return generated_files
    
    def analyze_multiple_files(self, file_labels: List[Tuple[List[str], str]], aggregation_type: str = 'on_op_name',
                              show_dtype: bool = True, show_shape: bool = True, show_kernel_names: bool = True, show_kernel_duration: bool = True, special_matmul: bool = False,
                              output_dir: str = ".", compare_dtype: bool = False, compare_shape: bool = False, print_markdown: bool = False) -> List[Path]:
        """
        分析多个文件的完整流程，支持同一label下多个文件的聚合
        
        Args:
            file_labels: [(file_paths, label), ...] 文件路径列表和标签的列表
                        每个label可以对应多个文件，会自动聚合
            aggregation_type: 聚合类型
            show_dtype: 是否展示 dtype 信息
            show_shape: 是否展示 shape 和 strides 信息
            show_kernel: 是否展示 kernel 信息
            special_matmul: 是否进行特殊的 matmul 展示
            output_dir: 输出目录
        """
        # 计算总文件数
        total_files = sum(len(file_paths) for file_paths, _ in file_labels)
        print(f"=== 开始分析多个文件，共 {len(file_labels)} 个标签，{total_files} 个文件 ===")
        
        # 分析每个标签下的文件
        multiple_files_data = {}
        
        for file_paths, label in file_labels:
            print(f"正在分析标签: {label} ({len(file_paths)} 个文件)")
            
            # 存储该标签下所有文件的聚合数据
            label_aggregated_data_list = []
            
            for i, file_path in enumerate(file_paths):
                print(f"  处理文件 {i+1}/{len(file_paths)}: {file_path}")
                
                # 加载数据
                data = self.parser.load_json_file(file_path)
                
                # Stage 1: 数据后处理
                cpu_events_by_external_id, kernel_events_by_external_id = self.stage1_data_postprocessing(data)
                
                # Stage 2: 数据聚合
                aggregated_data = self.stage2_data_aggregation(cpu_events_by_external_id, kernel_events_by_external_id, aggregation_type)
                
                label_aggregated_data_list.append(aggregated_data)
            
            # 聚合同一标签下的多个文件
            if len(label_aggregated_data_list) == 1:
                # 只有一个文件，直接使用
                multiple_files_data[label] = label_aggregated_data_list[0]
            else:
                # 多个文件，进行聚合
                print(f"  聚合 {len(label_aggregated_data_list)} 个文件的数据...")
                multiple_files_data[label] = self._merge_same_label_files(label_aggregated_data_list, aggregation_type)
        
        # Stage 3: 比较（多文件比较）
        comparison_result = self.stage3_comparison(multiple_files_data=multiple_files_data, aggregation_type=aggregation_type)
        
        # Stage 4: 展示
        file_labels_list = [label for _, label in file_labels]
        generated_files = self.stage4_presentation(comparison_result, output_dir, show_dtype, show_shape, show_kernel_names, show_kernel_duration, special_matmul, aggregation_type, compare_dtype, compare_shape, file_labels_list, print_markdown=print_markdown)
        
        print("=== 多文件分析完成 ===")
        return generated_files
    
    def _merge_same_label_files(self, aggregated_data_list: List[Dict[Union[str, tuple], 'AggregatedData']], aggregation_type: str) -> Dict[Union[str, tuple], 'AggregatedData']:
        """
        聚合同一标签下的多个文件的聚合数据
        
        Args:
            aggregated_data_list: 多个文件的聚合数据列表
            aggregation_type: 聚合类型
            
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

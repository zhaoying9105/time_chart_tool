"""
数据后处理阶段
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from ...models import ActivityEvent, ProfilerData
from ..utils.data_structures import AggregatedData


class DataPostProcessor:
    """数据后处理器"""
    
    def __init__(self):
        pass
    
    def _process_triton_names(self, cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                             kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]]) -> Tuple[Dict[Union[int, str], List[ActivityEvent]], Dict[Union[int, str], List[ActivityEvent]]]:
        """
        处理triton开头的op和kernel名称，去除suffix部分
        
        triton名称格式: triton_{kernel_category}_{fused_name}_{suffix}
        其中suffix是递增的数字后缀，用于区分不同的kernel实例
        
        Args:
            cpu_events_by_external_id: CPU events 按 external_id 分组
            kernel_events_by_external_id: Kernel events 按 external_id 分组
            
        Returns:
            Tuple: 处理后的 CPU events 和 Kernel events
        """
        print("处理triton名称，去除suffix...")
        
        # 处理CPU events中的triton名称
        for external_id, cpu_events in cpu_events_by_external_id.items():
            for cpu_event in cpu_events:
                if cpu_event.name.startswith('triton_'):
                    cpu_event.name = self._remove_triton_suffix(cpu_event.name)
        
        # 处理Kernel events中的triton名称
        for external_id, kernel_events in kernel_events_by_external_id.items():
            for kernel_event in kernel_events:
                if kernel_event.name.startswith('triton_'):
                    kernel_event.name = self._remove_triton_suffix(kernel_event.name)
        
        return cpu_events_by_external_id, kernel_events_by_external_id
    
    def _remove_triton_suffix(self, name: str) -> str:
        """
        去除triton名称中的suffix部分
        
        Args:
            name: 原始名称
            
        Returns:
            str: 去除suffix后的名称
        """
        if not name.startswith('triton_'):
            return name
        
        # 按'_'分割名称
        parts = name.split('_')
        
        # triton名称格式: triton_{kernel_category}_{fused_name}_{suffix}
        # 需要至少4个部分: triton, category, fused_name, suffix
        if len(parts) < 4:
            return name
        
        # 检查最后一部分是否为数字（suffix）
        last_part = parts[-1]
        if last_part.isdigit():
            # 去除最后的suffix部分
            return '_'.join(parts[:-1])
        
        return name
    
    def stage1_data_postprocessing(self, data: ProfilerData, 
                                 include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None,
                                 include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None) -> Tuple[Dict[Union[int, str], List[ActivityEvent]], Dict[Union[int, str], List[ActivityEvent]]]:
        """
        Stage 1: 数据后处理
        1. 时间戳标准化已在parser中完成
        2. 根据 cpu_op event & kernel event 的external id 分类，形成两个map
        3. 然后对一个 external id 下的 cpu_event 进行call stack 合并，保留call stack 最长的哪个 event
        4. 根据过滤模式过滤操作和kernel事件
        
        Args:
            data: ProfilerData 对象
            include_op_patterns: 包含的操作名称模式列表
            exclude_op_patterns: 排除的操作名称模式列表
            include_kernel_patterns: 包含的kernel名称模式列表
            exclude_kernel_patterns: 排除的kernel名称模式列表
            
        Returns:
            Tuple[Dict[external_id, cpu_events], Dict[external_id, kernel_events]]: 
                两个映射字典，确保 external id 和 cpu_event 是一对一的关系
        """
        print("=== Stage 1: 数据后处理 ===")
        print("时间戳标准化已在parser中完成")
        
        # 2. 根据 external id 分类
        cpu_events_by_external_id = defaultdict(list)
        kernel_events_by_external_id = defaultdict(list)
        
        # 首先统计所有 external_id 下的 cpu_op 和 kernel 事件
        for event in data.events:
            if event.external_id is not None:
                if event.cat == 'cpu_op':
                    cpu_events_by_external_id[event.external_id].append(event)
                elif event.cat == 'kernel':
                    kernel_events_by_external_id[event.external_id].append(event)

        # 只保留那些同时拥有 cpu_op 和 kernel 的 external_id
        valid_external_ids = set(cpu_events_by_external_id.keys()) & set(kernel_events_by_external_id.keys())
        cpu_events_by_external_id = {eid: cpu_events_by_external_id[eid] for eid in valid_external_ids}
        kernel_events_by_external_id = {eid: kernel_events_by_external_id[eid] for eid in valid_external_ids}

        print(f"找到 {len(cpu_events_by_external_id)} 个同时有 cpu_op 和 kernel 的 external_id")
        
        # 处理triton名称，去除suffix
        cpu_events_by_external_id, kernel_events_by_external_id = self._process_triton_names(
            cpu_events_by_external_id, kernel_events_by_external_id
        )
        
        # 根据过滤模式过滤事件
        if include_op_patterns or exclude_op_patterns or include_kernel_patterns or exclude_kernel_patterns:
            print("根据过滤模式过滤事件...")
            cpu_events_by_external_id, kernel_events_by_external_id = self._filter_events_by_patterns(
                cpu_events_by_external_id, kernel_events_by_external_id,
                include_op_patterns, exclude_op_patterns,
                include_kernel_patterns, exclude_kernel_patterns
            )
            print(f"过滤后剩余 {len(cpu_events_by_external_id)} 个有 cpu_op 的 external_id")
            print(f"过滤后剩余 {len(kernel_events_by_external_id)} 个有 kernel 的 external_id")
        
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
    
    def _filter_events_by_patterns(self, cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                                 kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]],
                                 include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None,
                                 include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None) -> Tuple[Dict[Union[int, str], List[ActivityEvent]], Dict[Union[int, str], List[ActivityEvent]]]:
        """
        根据过滤模式过滤事件
        
        Args:
            cpu_events_by_external_id: CPU events 按 external_id 分组
            kernel_events_by_external_id: Kernel events 按 external_id 分组
            include_op_patterns: 包含的操作名称模式列表
            exclude_op_patterns: 排除的操作名称模式列表
            include_kernel_patterns: 包含的kernel名称模式列表
            exclude_kernel_patterns: 排除的kernel名称模式列表
            
        Returns:
            Tuple: 过滤后的 CPU events 和 Kernel events
        """
        # 编译正则表达式模式
        include_op_regexes = self._compile_patterns(include_op_patterns) if include_op_patterns else []
        exclude_op_regexes = self._compile_patterns(exclude_op_patterns) if exclude_op_patterns else []
        include_kernel_regexes = self._compile_patterns(include_kernel_patterns) if include_kernel_patterns else []
        exclude_kernel_regexes = self._compile_patterns(exclude_kernel_patterns) if exclude_kernel_patterns else []
        
        # 找到需要保留的 external_id
        keep_external_ids = set()
        
        # 检查 CPU events
        for external_id, cpu_events in cpu_events_by_external_id.items():
            should_keep = True
            
            # 如果设置了包含模式，需要至少有一个操作匹配
            if include_op_regexes:
                has_matching_op = any(
                    any(regex.search(cpu_event.name) for regex in include_op_regexes)
                    for cpu_event in cpu_events
                )
                if not has_matching_op:
                    should_keep = False
            
            # 如果设置了排除模式，任何操作都不能匹配
            if exclude_op_regexes and should_keep:
                has_excluded_op = any(
                    any(regex.search(cpu_event.name) for regex in exclude_op_regexes)
                    for cpu_event in cpu_events
                )
                if has_excluded_op:
                    should_keep = False
            
            if should_keep:
                keep_external_ids.add(external_id)
        
        # 检查 Kernel events
        for external_id, kernel_events in kernel_events_by_external_id.items():
            should_keep = True
            
            # 检查kernel名称过滤
            for kernel_event in kernel_events:
                kernel_name = kernel_event.name
                
                # 检查包含模式
                if include_kernel_regexes:
                    if not any(regex.search(kernel_name) for regex in include_kernel_regexes):
                        should_keep = False
                        break
                
                # 检查排除模式
                if exclude_kernel_regexes:
                    if any(regex.search(kernel_name) for regex in exclude_kernel_regexes):
                        should_keep = False
                        break
            
            if should_keep:
                keep_external_ids.add(external_id)
        
        # 过滤事件
        filtered_cpu_events = {}
        filtered_kernel_events = {}
        
        for external_id in keep_external_ids:
            # 过滤 CPU events
            if external_id in cpu_events_by_external_id:
                cpu_events = cpu_events_by_external_id[external_id]
                filtered_cpu_events_list = []
                
                for cpu_event in cpu_events:
                    op_name = cpu_event.name
                    should_include = True
                    
                    # 检查包含模式
                    if include_op_regexes:
                        if not any(regex.search(op_name) for regex in include_op_regexes):
                            should_include = False
                    
                    # 检查排除模式
                    if exclude_op_regexes and should_include:
                        if any(regex.search(op_name) for regex in exclude_op_regexes):
                            should_include = False
                    
                    if should_include:
                        filtered_cpu_events_list.append(cpu_event)
                
                if filtered_cpu_events_list:  # 只有当过滤后还有操作时才保留
                    filtered_cpu_events[external_id] = filtered_cpu_events_list
            
            # 过滤 Kernel events
            if external_id in kernel_events_by_external_id:
                kernel_events = kernel_events_by_external_id[external_id]
                filtered_kernel_events_list = []
                
                for kernel_event in kernel_events:
                    kernel_name = kernel_event.name
                    should_include = True
                    
                    # 检查包含模式
                    if include_kernel_regexes:
                        if not any(regex.search(kernel_name) for regex in include_kernel_regexes):
                            should_include = False
                    
                    # 检查排除模式
                    if exclude_kernel_regexes and should_include:
                        if any(regex.search(kernel_name) for regex in exclude_kernel_regexes):
                            should_include = False
                    
                    if should_include:
                        filtered_kernel_events_list.append(kernel_event)
                
                if filtered_kernel_events_list:  # 只有当过滤后还有kernel时才保留
                    filtered_kernel_events[external_id] = filtered_kernel_events_list
        
        return filtered_cpu_events, filtered_kernel_events
    
    def _compile_patterns(self, patterns: List[str]) -> List[re.Pattern]:
        """
        编译正则表达式模式，支持通配符语法
        
        Args:
            patterns: 模式字符串列表
            
        Returns:
            List[re.Pattern]: 编译后的正则表达式列表
        """
        compiled_patterns = []
        for pattern in patterns:
            # 首先检查是否包含通配符语法
            if '*' in pattern or '?' in pattern:
                # 将通配符转换为正则表达式
                # * -> .*
                # ? -> .
                regex_pattern = pattern.replace('*', '.*').replace('?', '.')
                try:
                    compiled_patterns.append(re.compile(regex_pattern))
                    continue
                except re.error:
                    pass
            
            # 尝试作为正则表达式编译
            try:
                compiled_patterns.append(re.compile(pattern))
            except re.error:
                # 如果编译失败，作为普通字符串匹配
                escaped_pattern = re.escape(pattern)
                compiled_patterns.append(re.compile(escaped_pattern))
        
        return compiled_patterns
    
    def _drop_communication_events(self, cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                                 kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]]) -> Tuple[Dict[Union[int, str], List[ActivityEvent]], Dict[Union[int, str], List[ActivityEvent]]]:
        """
        丢弃包含 TCDP 的通信 kernel events 及其对应的 CPU events
        
        Args:
            cpu_events_by_external_id: CPU events 按 external_id 分组
            kernel_events_by_external_id: Kernel events 按 external_id 分组
            
        Returns:
            Tuple: 过滤后的 CPU events 和 Kernel events
        """
        # 找到包含 TCDP 的 external_id
        tcdp_external_ids = set()
        
        for external_id, kernel_events in kernel_events_by_external_id.items():
            for kernel_event in kernel_events:
                if 'TCDP' in kernel_event.name:
                    tcdp_external_ids.add(external_id)
                    break
        
        print(f"找到 {len(tcdp_external_ids)} 个包含 TCDP kernel 的 external_id")
        
        # 过滤掉包含 TCDP 的 external_id
        filtered_cpu_events = {}
        filtered_kernel_events = {}
        
        for external_id in cpu_events_by_external_id:
            if external_id not in tcdp_external_ids:
                filtered_cpu_events[external_id] = cpu_events_by_external_id[external_id]
        
        for external_id in kernel_events_by_external_id:
            if external_id not in tcdp_external_ids:
                filtered_kernel_events[external_id] = kernel_events_by_external_id[external_id]
        
        return filtered_cpu_events, filtered_kernel_events
    
    def _collect_per_rank_statistics(self, file_paths: List[str], aggregated_data_list: List[Dict[Union[str, tuple], 'AggregatedData']]) -> Dict[str, Dict[str, int]]:
        """
        收集每个文件的统计信息
        
        Args:
            file_paths: 文件路径列表
            aggregated_data_list: 每个文件的聚合数据列表
            
        Returns:
            Dict[str, Dict[str, int]]: 每个文件的统计信息 {file_path: {cpu_count: int, kernel_count: int}}
        """
        per_rank_stats = {}
        
        for i, (file_path, aggregated_data) in enumerate(zip(file_paths, aggregated_data_list)):
            cpu_count = 0
            kernel_count = 0
            
            for key, agg_data in aggregated_data.items():
                cpu_count += len(agg_data.cpu_events)
                kernel_count += len(agg_data.kernel_events)
            
            # 提取文件名作为 rank 标识
            file_name = Path(file_path).name
            per_rank_stats[file_name] = {
                'cpu_count': cpu_count,
                'kernel_count': kernel_count
            }
        
        return per_rank_stats
    
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

"""
数据后处理阶段 (纯函数实现)
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import logging

from datetime import datetime
from ...models import ActivityEvent
from ...utils.event_utils import remove_triton_suffix
from ..utils.call_stack_utils import normalize_call_stack
from ...utils.tree_utils import attach_call_stacks_to_events
from ...utils.time import calculate_readable_timestamps



logger = logging.getLogger(__name__)




def normalize_timestamps(events: List[ActivityEvent]) -> List[ActivityEvent]:
    """
    标准化时间戳：找到最小ts值，将所有事件的ts减去这个值（纯函数）
    
    Args:
        events: 事件列表
        
    Returns:
        List[ActivityEvent]: 标准化时间戳后的事件列表（新对象）
    """
    if not events:
        return events
    
    # 找到最小时间戳
    valid_ts = [event.ts for event in events if event.ts is not None]
    if not valid_ts:
        logger.warning("没有找到有效的时间戳，跳过标准化")
        return events
    
    min_ts = min(valid_ts)
    print(f"=== Stage 1.-1: 时间戳标准化 (min_ts: {min_ts:.2f} us) ===")
    
    # 标准化所有事件的时间戳
    processed_events = []
    for event in events:
        if event.ts is not None:
            # 创建新对象以保持纯函数特性
            event.ts = event.ts - min_ts
        processed_events.append(event)
    
    return processed_events


def process_triton_names(cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
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
                cpu_event.name = remove_triton_suffix(cpu_event.name)
    
    # 处理Kernel events中的triton名称
    for external_id, kernel_events in kernel_events_by_external_id.items():
        for kernel_event in kernel_events:
            if kernel_event.name.startswith('triton_'):
                kernel_event.name = remove_triton_suffix(kernel_event.name)
    
    return cpu_events_by_external_id, kernel_events_by_external_id

def attach_call_stacks(events: List[ActivityEvent], call_stack_trees: Dict[Tuple[int, int], Any]) -> List[ActivityEvent]:
    """
    为 CPU 事件添加调用栈信息（纯函数）
    
    Args:
        events: 事件列表
        call_stack_trees: 调用栈树字典 { (pid, tid): CallStackNode }
        
    Returns:
        List[ActivityEvent]: 带有调用栈信息的事件列表
    """
    if not call_stack_trees:
        return events

    print("=== Stage 1.0: 为 CPU 事件添加调用栈信息 ===")
    
    # 筛选出需要处理的 CPU 事件
    cpu_events = [e for e in events if e.cat == 'cpu_op']
    other_events = [e for e in events if e.cat != 'cpu_op']
    
    if not cpu_events:
        return events
        
    # 获取带有调用栈的新事件列表
    processed_cpu_events = attach_call_stacks_to_events(cpu_events, call_stack_trees)
    
    # 合并结果（保持原始顺序可能比较困难，这里简单合并，后续处理不依赖原始顺序）
    # 如果需要保持原始顺序，可以在 attach_call_stacks_to_events 中处理所有事件
    
    # 为了保持纯函数且不修改输入列表，我们返回一个新的列表
    return processed_cpu_events + other_events


def classify_events_by_external_id(events: List[ActivityEvent]) -> Tuple[Dict[Union[int, str], List[ActivityEvent]], Dict[Union[int, str], List[ActivityEvent]]]:
    """
    1. 根据 cpu_op event & kernel event 的external id 分类，形成两个map
    """
    print("=== Stage 1.1: 根据 external_id 分类事件 ===")
    print("时间戳标准化已在parser中完成")
    
    # 根据 external id 分类
    cpu_events_by_external_id = defaultdict(list)
    kernel_events_by_external_id = defaultdict(list)
    
    # 首先统计所有 external_id 下的 cpu_op 和 kernel 事件
    for event in events:
        if event.external_id is not None:
            if event.cat == 'cpu_op':
                cpu_events_by_external_id[event.external_id].append(event)
            elif event.cat == 'kernel':
                kernel_events_by_external_id[event.external_id].append(event)
    
    logger.debug(f"初始分类: {len(cpu_events_by_external_id)} 个external_id有CPU事件, {len(kernel_events_by_external_id)} 个external_id有Kernel事件")

    # 只保留那些同时拥有 cpu_op 和 kernel 的 external_id
    valid_external_ids = set(cpu_events_by_external_id.keys()) & set(kernel_events_by_external_id.keys())
    
    logger.debug(f"交集: {len(valid_external_ids)} 个external_id同时有CPU和Kernel事件")
    
    # 使用交集过滤，确保每个 external_id 在两个 map 中都存在
    cpu_events_by_external_id = {eid: cpu_events_by_external_id[eid] for eid in valid_external_ids}
    kernel_events_by_external_id = {eid: kernel_events_by_external_id[eid] for eid in valid_external_ids}

    print(f"找到 {len(cpu_events_by_external_id)} 个同时有 cpu_op 和 kernel 的 external_id")
    
    return cpu_events_by_external_id, kernel_events_by_external_id


def process_triton_names_step(cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                             kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]]) -> Tuple[Dict[Union[int, str], List[ActivityEvent]], Dict[Union[int, str], List[ActivityEvent]]]:
    """
    3. 处理triton名称，去除suffix
    """
    print("=== Stage 1.3: 处理triton名称，去除suffix ===")
    return process_triton_names(cpu_events_by_external_id, kernel_events_by_external_id)


def filter_cpu_events_by_patterns(
    cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]],
    include_op_patterns: List[str] = None,
    exclude_op_patterns: List[str] = None,
) -> Dict[Union[int, str], List[ActivityEvent]]:
    include_op_regexes = _compile_patterns(include_op_patterns) if include_op_patterns else []
    exclude_op_regexes = _compile_patterns(exclude_op_patterns) if exclude_op_patterns else []

    result: Dict[Union[int, str], List[ActivityEvent]] = {}
    for external_id, cpu_events in cpu_events_by_external_id.items():
        filtered_list: List[ActivityEvent] = []
        for cpu_event in cpu_events:
            name = cpu_event.name
            keep = True
            if include_op_regexes:
                if not any(regex.search(name) for regex in include_op_regexes):
                    keep = False
            if exclude_op_regexes and keep:
                if any(regex.search(name) for regex in exclude_op_regexes):
                    keep = False
            if keep:
                filtered_list.append(cpu_event)
        if filtered_list:
            result[external_id] = filtered_list
    return result


def filter_kernel_events_by_patterns(
    kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]],
    include_kernel_patterns: List[str] = None,
    exclude_kernel_patterns: List[str] = None,
) -> Dict[Union[int, str], List[ActivityEvent]]:
    include_kernel_regexes = _compile_patterns(include_kernel_patterns) if include_kernel_patterns else []
    exclude_kernel_regexes = _compile_patterns(exclude_kernel_patterns) if exclude_kernel_patterns else []

    result: Dict[Union[int, str], List[ActivityEvent]] = {}
    for external_id, kernel_events in kernel_events_by_external_id.items():
        filtered_list: List[ActivityEvent] = []
        for kernel_event in kernel_events:
            name = kernel_event.name
            keep = True
            if include_kernel_regexes:
                if not any(regex.search(name) for regex in include_kernel_regexes):
                    keep = False
            if exclude_kernel_regexes and keep:
                if any(regex.search(name) for regex in exclude_kernel_regexes):
                    keep = False
            if keep:
                filtered_list.append(kernel_event)
        if filtered_list:
            result[external_id] = filtered_list
    return result


def restrict_to_common_external_ids(
    cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]],
    kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]],
) -> Tuple[Dict[Union[int, str], List[ActivityEvent]], Dict[Union[int, str], List[ActivityEvent]]]:
    common_ids = set(cpu_events_by_external_id.keys()) & set(kernel_events_by_external_id.keys())
    cpu_filtered = {eid: cpu_events_by_external_id[eid] for eid in common_ids}
    kernel_filtered = {eid: kernel_events_by_external_id[eid] for eid in common_ids}
    return cpu_filtered, kernel_filtered


def select_representative_cpu_events(
    cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]],
    coarse_call_stack: bool = False,
) -> Dict[Union[int, str], List[ActivityEvent]]:
    result: Dict[Union[int, str], List[ActivityEvent]] = {}
    for external_id, cpu_events in cpu_events_by_external_id.items():
        selected_cpu_events = _merge_cpu_events_by_call_stack(cpu_events, coarse_call_stack)
        if selected_cpu_events:
            result[external_id] = selected_cpu_events
    return result


def _set_fwd_bwd_type_from_call_stack(event: ActivityEvent) -> ActivityEvent:
    """根据 call stack 设置 fwd_bwd_type，并增加 debug 信息"""
    if event.cat == 'cpu_op':
        if event.call_stack:
            # 检查 call stack 中是否包含 ':forward' 或 ':backward'
            for i, frame in enumerate(event.call_stack):
                if ': forward' in frame:
                    event.fwd_bwd_type = 'fwd'
                    return event
                elif ': backward' in frame or 'Backward' in frame:
                    event.fwd_bwd_type = 'bwd'
                    return event
                elif 'Grad' in frame:
                    event.fwd_bwd_type = 'grad'
                    return event
            # 没有找到 ':forward' 或 ':backward'
            event.fwd_bwd_type = 'none'
        else:
            # 没有 call stack
            event.fwd_bwd_type = 'none'
    return event


def _propagate_fwd_call_stack_to_bwd(events: List[ActivityEvent], call_stack_trees: Dict[Tuple[int, int], Any]) -> List[ActivityEvent]:
    """
    将 fwd 的 call stack 传播给 bwd
    Logic: fwdbwd event -> ts -> got event ->  形成 fwd event  map to bwd event -> fwd event  call stack copy to bwd event
    """
    print("=== Stage 1.1.2: Propagate fwd call stack to bwd ===")
    
    # 1. 建立 ts -> cpu_op event 的映射 (只包含有 call stack 的 cpu_op)
    # 这样可以通过 timestamp 快速找到对应的 cpu_op
    ts_to_cpu_op = {e.ts: e for e in events if e.cat == 'cpu_op'}

    # 2. 处理 fwdbwd events，找到 fwd 和 bwd 的对应关系
    # fwdbwd events 是用来连接 fwd 和 bwd 的桥梁
    # 它们成对出现，拥有相同的 id
    # ph='s' 是 fwd start, ph='f' 是 bwd start
    fwdbwd_events = [e for e in events if e.cat == 'fwdbwd']
    
    # key: id
    # value: { 'fwd_ts': float, 'bwd_ts': float }
    flow_map = defaultdict(dict)
    
    for e in fwdbwd_events:
        if e.id is None: continue
        if e.ph == 's':
            flow_map[e.id]['fwd_ts'] = e.ts
        elif e.ph == 'f':
            flow_map[e.id]['bwd_ts'] = e.ts
            
    # 3. 构建 fwd_event -> bwd_event 的映射
    # 只有当 fwd_ts 和 bwd_ts 都存在，且都能在 ts_to_cpu_op 中找到对应的 event 时，才建立映射
    # 使用 list 存储 pairs，避免 unhashable type: 'ActivityEvent' 错误
    fwd_bwd_pairs = []
    
    for _, flow in flow_map.items():
        fwd_ts = flow.get('fwd_ts')
        bwd_ts = flow.get('bwd_ts')
        
        if fwd_ts is not None and bwd_ts is not None:
            fwd_op = ts_to_cpu_op.get(fwd_ts)
            bwd_op = ts_to_cpu_op.get(bwd_ts)
            
            if fwd_op and bwd_op and getattr(fwd_op, 'call_stack', None):
                fwd_bwd_pairs.append((fwd_op, bwd_op))

    print(f"Found {len(fwd_bwd_pairs)} fwd-bwd pairs")

    # 4. 复制 call stack 并传播
    count_propagated = 0
    bwd_ops_to_update_children = []
    
    for fwd_op, bwd_op in fwd_bwd_pairs:
        # Copy call stack
        bwd_op.call_stack = fwd_op.call_stack
        bwd_op.fwd_bwd_type = 'bwd'
        bwd_ops_to_update_children.append(bwd_op)
        count_propagated += 1
        
    print(f"Propagated call stack to {count_propagated} bwd events")

    # 5. 将 bwd event 的 call stack 传播给其子孙节点
    if bwd_ops_to_update_children and call_stack_trees:
        _propagate_stack_to_children(bwd_ops_to_update_children, call_stack_trees)
        
    return events

def _propagate_stack_to_children(parent_events: List[ActivityEvent], call_stack_trees: Dict[Tuple[int, int], Any]):
    """
    将父节点的 call stack 传播给子孙节点
    """
    # 1. 建立 event_id 到 node 的映射，以便快速找到 parent_events 对应的 nodes
    # 由于 call_stack_trees 是按 (pid, tid) 分组的，我们需要遍历查找
    
    # 优化：我们只关心 parent_events 涉及的 (pid, tid)
    involved_pid_tids = set((e.pid, e.tid) for e in parent_events)
    
    # 建立 event_id -> event 的映射，用于快速查找 parent event
    # 注意：我们需要使用 create_event_id 来匹配
    from ...utils.event_utils import create_event_id
    parent_event_ids = set(create_event_id(e) for e in parent_events)
    
    # 遍历相关的 trees
    count_children_updated = 0
    
    for pid_tid in involved_pid_tids:
        if pid_tid in call_stack_trees:
            root = call_stack_trees[pid_tid]
            # 遍历树找到 parent nodes
            nodes_to_process = [root]
            while nodes_to_process:
                node = nodes_to_process.pop()
                if node.event.name != "ROOT": # Skip virtual root
                    node_event_id = create_event_id(node.event)
                    if node_event_id in parent_event_ids:
                        # 这是一个 bwd root node，它的 event 已经被更新了 call stack
                        # 我们需要把这个 call stack 传播给它的所有子孙
                        # 注意：node.event 引用的是 events 列表中的同一个对象吗？
                        # attach_call_stacks_to_events 中似乎是创建了新对象或者是修改了属性
                        # 我们假设 node.event 和 parent_events 中的对象是对应的（通过引用或ID）
                        
                        # 获取正确的 call stack (从 parent_events 中获取，或者直接从 node.event 获取如果已经更新)
                        # 在 _propagate_fwd_call_stack_to_bwd 中我们直接修改了 bwd_op.call_stack
                        # 如果 node.event 就是那个 bwd_op，那么直接用 node.event.call_stack
                        
                        # 验证 node.event 是否有 call stack
                        if hasattr(node.event, 'call_stack') and node.event.call_stack:
                             count_children_updated += _apply_stack_to_descendants(node, node.event.call_stack)
                    
                nodes_to_process.extend(node.children)
                
    print(f"Propagated call stack to {count_children_updated} children events")

def _apply_stack_to_descendants(node: Any, call_stack: List[str]) -> int:
    """递归将 stack 应用于所有子孙节点"""
    count = 0
    stack = list(node.children)
    while stack:
        child = stack.pop()
        # 更新子节点的 call stack
        child.event.call_stack = call_stack
        child.event.fwd_bwd_type = 'bwd' # 标记为 bwd
        count += 1
        stack.extend(child.children)
    return count


def select_representative_kernel_events(
    kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]]
) -> Dict[Union[int, str], List[ActivityEvent]]:
    result: Dict[Union[int, str], List[ActivityEvent]] = {}
    for external_id, kernel_events in kernel_events_by_external_id.items():
        if not kernel_events:
            continue
        selected_kernel_event = max(kernel_events, key=lambda x: x.dur if x.dur is not None else 0)
        result[external_id] = [selected_kernel_event]
    return result


def filter_and_select_events(cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                           kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]],
                           include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None,
                           include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None,
                           coarse_call_stack: bool = False) -> Tuple[Dict[Union[int, str], ActivityEvent], Dict[Union[int, str], ActivityEvent]]:
    """
    4. 根据过滤模式过滤操作和kernel事件，并选择代表事件（map[cpu op event: kernel event]）
    """
    print("=== Stage 1.4: 过滤并选择代表事件 ===")
    
    if include_op_patterns or exclude_op_patterns or include_kernel_patterns or exclude_kernel_patterns:
        print("根据过滤模式过滤事件...")
        cpu_events_by_external_id = filter_cpu_events_by_patterns(
            cpu_events_by_external_id, include_op_patterns, exclude_op_patterns
        )
        kernel_events_by_external_id = filter_kernel_events_by_patterns(
            kernel_events_by_external_id, include_kernel_patterns, exclude_kernel_patterns
        )
        cpu_events_by_external_id, kernel_events_by_external_id = restrict_to_common_external_ids(
            cpu_events_by_external_id, kernel_events_by_external_id
        )
        print(f"过滤后剩余 {len(cpu_events_by_external_id)} 个有 cpu_op 的 external_id")
        print(f"过滤后剩余 {len(kernel_events_by_external_id)} 个有 kernel 的 external_id")
    
    # 选择代表事件：对同一个 external_id 下的 cpu_op 、kernel 都各取一个
    # 返回 map[cpu op event: kernel event] 的形式其实不太好用，因为后续还需要根据 external_id 聚合
    # 但根据用户要求：1.1 返回 filtered_cpu_events, filtered_kernel_events，改成返回 map[cpu op event: kernel event]
    # 实际上这里应该是返回每个 external_id 对应的单个 cpu_event 和单个 kernel_event
    
    # 由于原始接口返回的是 Dict[external_id, List[Event]]，而后续 data_aggregation 接收的也是 Dict[external_id, List[Event]]
    # 所以为了兼容性，我们这里返回 Dict[external_id, List[Event]]，但在 List 中只保留一个选中的 Event
    
    cpu_selected = select_representative_cpu_events(cpu_events_by_external_id, coarse_call_stack)
    kernel_selected = select_representative_kernel_events(kernel_events_by_external_id)
    print(f"最终处理完成，保留 {len(cpu_selected)} 个 external_id 的事件对")
    return cpu_selected, kernel_selected

# 保持原来的函数名以便兼容，但在内部调用新的函数
def postprocessing(events, call_stack_trees,base_time_nanoseconds,
                             include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None,
                             include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None,
                             coarse_call_stack: bool = False) -> Tuple[Dict[Union[int, str], List[ActivityEvent]], Dict[Union[int, str], List[ActivityEvent]]]:
    """
    Stage 1: 数据后处理
    """

    # 1.-2 计算可读时间戳
    events = calculate_readable_timestamps(events, base_time_nanoseconds)

    # 1.-1 时间戳标准化
    events = normalize_timestamps(events)

    # 1.1 Attach call stack (在分类前执行)
    # 纯函数：接受 events 和 call_stack_trees，返回新的 events 列表
    events = attach_call_stacks(events, call_stack_trees)

    # 1.1.1 Set fwd_bwd_type from call stack
    events = [_set_fwd_bwd_type_from_call_stack(e) for e in events]

    # 1.1.2 Propagate fwd call stack to bwd
    events = _propagate_fwd_call_stack_to_bwd(events, call_stack_trees)

    # 1.2 分类
    cpu_events, kernel_events = classify_events_by_external_id(events)
    
    # 1.3 处理 Triton names
    cpu_events, kernel_events = process_triton_names_step(cpu_events, kernel_events)
    
    # 1.4 过滤和选择
    return filter_and_select_events(cpu_events, kernel_events, 
                                  include_op_patterns, exclude_op_patterns,
                                  include_kernel_patterns, exclude_kernel_patterns,
                                  coarse_call_stack)



def filter_events_by_patterns(cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
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
    include_op_regexes = _compile_patterns(include_op_patterns) if include_op_patterns else []
    exclude_op_regexes = _compile_patterns(exclude_op_patterns) if exclude_op_patterns else []
    include_kernel_regexes = _compile_patterns(include_kernel_patterns) if include_kernel_patterns else []
    exclude_kernel_regexes = _compile_patterns(exclude_kernel_patterns) if exclude_kernel_patterns else []
    
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


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
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


def _merge_cpu_events_by_call_stack(cpu_events: List[ActivityEvent], coarse_call_stack: bool = False) -> List[ActivityEvent]:
    """
    对同一 external_id 下的多个 cpu_op 事件进行 call stack 合并
    保留 call stack 最长的那个 event
    
    Args:
        cpu_events: 同一 external_id 下的 cpu_op 事件列表
        coarse_call_stack: 是否使用粗糙的call stack匹配
        
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
            merged_events.extend(_filter_events_by_call_stack_prefix(events, coarse_call_stack))
    
    return merged_events


def _filter_events_by_call_stack_prefix(events: List[ActivityEvent], coarse_call_stack: bool = False) -> List[ActivityEvent]:
    """
    过滤具有相同 name 的多个 cpu_op 事件
    如果它们的 call stack 存在前缀关系，只保留最长的
    
    Args:
        events: 具有相同 name 的 cpu_op 事件列表
        coarse_call_stack: 是否使用粗糙的call stack匹配
        
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
        normalized_call_stack_wrapper = normalize_call_stack(event.call_stack, coarse_call_stack=coarse_call_stack)
        if normalized_call_stack_wrapper.call_stack:
            normalized_events.append((event, normalized_call_stack_wrapper.call_stack))
    
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


from typing import List, Optional, Dict, Tuple, Set, Any
from ...parser import PyTorchProfilerParser, CallStackNode
from ...models import ActivityEvent

def _print_events_between_call_stacks(prev_fastest_event, current_fastest_event, 
                                      prev_slowest_event=None, current_slowest_event=None,
                                      fastest_parser=None, slowest_parser=None):
    """
    打印两个事件之间的调用栈树分析，使用时间窗口过滤
    
    Args:
        prev_fastest_event: 前一个最快卡事件对象
        current_fastest_event: 当前最快卡事件对象
        prev_slowest_event: 前一个最慢卡事件对象
        current_slowest_event: 当前最慢卡事件对象
        fastest_parser: 最快卡的parser对象
        slowest_parser: 最慢卡的parser对象
    """
    if not prev_fastest_event or not current_fastest_event:
        print("    事件之间调用栈分析：无法获取事件对象")
        return
    
    print("    事件之间调用栈分析:")
    print("    " + "=" * 80)
    
    # 为最快卡分析两个事件之间的调用栈关系
    print(f"    最快Card ({prev_fastest_event.name} → {current_fastest_event.name}) 的调用栈关系:")
    _print_time_window_call_stack_tree(
        prev_event=prev_fastest_event, 
        current_event=current_fastest_event, 
        parser=fastest_parser, 
        card_name="最快Card"
    )
    
    # 如果提供了最慢卡事件，也进行分析
    if prev_slowest_event and current_slowest_event and slowest_parser:
        print(f"\n    最慢Card ({prev_slowest_event.name} → {current_slowest_event.name}) 的调用栈关系:")
        _print_time_window_call_stack_tree(
            prev_event=prev_slowest_event, 
            current_event=current_slowest_event, 
            parser=slowest_parser, 
            card_name="最慢Card"
        )
    
    print("    " + "=" * 80)

def _print_time_window_call_stack_tree(prev_event, current_event, parser, card_name):
    """
    打印时间窗口内的调用栈树
    
    Args:
        analyzer: Analyzer instance
        prev_event: 前一个事件对象
        current_event: 当前事件对象  
        parser: PyTorchProfilerParser 对象
        card_name: Card名称标识
    """
    if not parser:
        print(f"      {card_name}: 无法获取parser对象")
        return
    
    # 获取调用栈树
    call_stack_trees = parser.get_call_stack_trees()
    if not call_stack_trees:
        print(f"      {card_name}: 未找到调用栈树")
        return
    
    # 找到前一个和当前事件对应的节点
    # 使用与call_stack_builder相同的triton名称处理逻辑
    prev_event_id = _create_event_id_for_lookup(prev_event)
    current_event_id = _create_event_id_for_lookup(current_event)
    
    prev_node = parser.call_stack_builder.event_to_node_map.get(prev_event_id)
    current_node = parser.call_stack_builder.event_to_node_map.get(current_event_id)
    
    if not prev_node:
        # 添加调试信息
        print(f"      {card_name}: 未找到前一个事件 '{prev_event.name}' 的节点")
        print(f"        调试信息 - 查询的event_id: {prev_event_id}")
        print(f"        调试信息 - 前事件属性: ts={prev_event.ts}, dur={prev_event.dur}, pid={prev_event.pid}, tid={prev_event.tid}")
        print(f"        调试信息 - event_to_node_map大小: {len(parser.call_stack_builder.event_to_node_map)}")
        
        # 检查event_to_node_map中的前几个条目
        print(f"        event_to_node_map中的前几个条目:")
        for i, (key, value) in enumerate(list(parser.call_stack_builder.event_to_node_map.items())[:3]):
            print(f"          {i+1}. {key}")
        
        return
    
    if not current_node:
        print(f"      {card_name}: 未找到当前事件 '{current_event.name}' 的节点")
        return
    
    print(f"      {card_name}: 找到事件节点")
    print(f"        前一个事件: {prev_event.name} (ts={prev_event.ts/1000:.3f}ms, dur={prev_event.dur/1000:.3f}ms)")
    print(f"        当前事件: {current_event.name} (ts={current_event.ts/1000:.3f}ms, dur={current_event.dur/1000:.3f}ms)")
    
    # 定义时间窗口：从前一个事件开始到当前事件结束
    time_start = prev_event.ts
    time_end = current_event.ts + current_event.dur
    
    print(f"        时间窗口: [{time_start:.6f}, {time_end:.6f}]")
    
    # 收集时间窗口内的事件
    window_events = []
    
    # 获取前一个事件所属的(pid, tid)组
    pid_tid_key = (prev_event.pid, prev_event.tid)
    root_node = call_stack_trees.get(pid_tid_key)
    
    if root_node:
        window_events = _collect_events_in_time_window(
            root_node, time_start, time_end, 
            prev_node.event_id, current_node.event_id
        )
    
    print(f"        时间窗口内事件数: {len(window_events)}")
    
    if window_events:
        # 重新构建调用栈树并打印
        print(f"        时间窗口调用栈树:")
        _build_and_print_time_window_tree(
            window_events, prev_event, current_event, card_name, parser
        )
    else:
        print(f"        时间窗口内无其他事件")

def _collect_events_in_time_window(root_node, time_start, time_end, prev_event_id, current_event_id):
    """收集时间窗口内的所有事件"""
    events = []
    event_ids = set()
    
    def traverse_node(node):
        event = node.event
        
        # 检查事件是否在时间窗口内
        event_start = event.ts
        event_end = event.ts + event.dur
        
        # 事件与时间窗口有重叠就包含（支持重叠检查）
        if (event_end >= time_start and event_start <= time_end):
            event_id = node.event_id
            
            # 避免重复添加相同事件
            if event_id not in event_ids:
                events.append(event)
                event_ids.add(event_id)
        
        # 递归遍历子节点
        for child in node.children:
            traverse_node(child)
    
    traverse_node(root_node)
    
    # 按时间戳排序
    events.sort(key=lambda e: e.ts)
    return events

def _build_and_print_time_window_tree(events, prev_event, current_event, card_name, parser):
    """为时间窗口内的事件重建调用栈树并打印"""
    if not events:
        return
    
    # 使用build_call_stacks_subtree方法重建调用栈树，保留原始映射
    builder = parser.call_stack_builder
    
    # 扩展事件列表，包含所有相关父节点
    extended_events = _extend_events_with_parents(events, builder)
    
    time_window_trees = builder.build_call_stacks_subtree(extended_events, preserve_mapping=True)
    
    if not time_window_trees:
        print(f"        {card_name}: 未能重建调用栈树")
        return
    
    # 分析时间间隔，找到最大间隔的事件对（只分析原始事件，不包含扩展的父节点）
    max_gap_events = _find_max_time_gap_events(events, prev_event, current_event)
    
    # 打印最大时间间隔信息
    if max_gap_events:
        first_event, second_event = max_gap_events
        gap_ms = (second_event.ts - first_event.ts) / 1000
        print(f"        ===== 最大时间间隔分析 =====")
        print(f"        最大时间间隔: {gap_ms:.3f}ms")
        print(f"        事件对:")
        first_dur_ms = first_event.dur / 1000 if first_event.dur else 0
        second_dur_ms = second_event.dur / 1000 if second_event.dur else 0
        print(f"          {first_event.name} (ts={first_event.ts:.6f}, dur={first_dur_ms:.3f}ms, pid={first_event.pid}, tid={first_event.tid})")
        print(f"          ↓ 间隔 {gap_ms:.3f}ms")
        print(f"          {second_event.name} (ts={second_event.ts:.6f}, dur={second_dur_ms:.3f}ms, pid={second_event.pid}, tid={second_event.tid})")
        print()
    
    print(f"        ===== 调用栈树结构 =====")
    
    # 获取prev_event所在的pid & tid，优先显示
    main_pid_tid = (prev_event.pid, prev_event.tid)
    
    # 首先打印主要的pid & tid（prev_event所在的）
    if main_pid_tid in time_window_trees:
        print(f"        【主要进程/线程】 PID: {main_pid_tid[0]}, TID: {main_pid_tid[1]}")
        _print_time_window_tree_recursive(
            time_window_trees[main_pid_tid], prev_event, current_event, max_gap_events, depth=0, prefix=""
        )
        print()  # 添加空行分隔
    
    # 然后打印其他相关的pid & tid
    other_pid_tids = [(pid, tid) for (pid, tid) in time_window_trees.keys() if (pid, tid) != main_pid_tid]
    if other_pid_tids:
        print(f"        【其他相关进程/线程】")
        for pid, tid in other_pid_tids:
            print(f"        PID: {pid}, TID: {tid}")
            _print_time_window_tree_recursive(
                time_window_trees[(pid, tid)], prev_event, current_event, max_gap_events, depth=0, prefix=""
            )
            print()  # 添加空行分隔

def _find_max_time_gap_events(events, prev_event, current_event):
    """找到时间间隔最大的连续事件对（只分析指定时间窗口内的原始事件）"""
    if len(events) < 2:
        return None
    
    # 确定时间窗口边界
    time_start = min(prev_event.ts, current_event.ts)
    time_end = max(prev_event.ts, current_event.ts)
    
    # 过滤出时间窗口内的事件（只比较ts，不考虑dur）
    filtered_events = []
    for event in events:
        if time_start <= event.ts <= time_end:
            filtered_events.append(event)
    
    if len(filtered_events) < 2:
        return None
    
    # 按时间戳排序过滤后的事件
    sorted_events = sorted(filtered_events, key=lambda e: e.ts)
    
    # 计算连续事件的时间间隔
    max_gap_ms = 0
    max_gap_pair = None
    
    for i in range(len(sorted_events) - 1):
        current_event_obj = sorted_events[i]
        next_event_obj = sorted_events[i + 1]
        gap_ms = (next_event_obj.ts - current_event_obj.ts) / 1000  # 转换为毫秒
        
        if gap_ms > max_gap_ms:
            max_gap_ms = gap_ms
            max_gap_pair = (current_event_obj, next_event_obj)
    
    return max_gap_pair

def _print_time_window_tree_recursive(node, prev_event, current_event, max_gap_events, depth=0, prefix=""):
    """递归打印时间窗口调用栈树"""
    # if depth > 20:  # 限制深度避免过深
    #     return
    
    event = node.event
    duration_ms = event.dur / 1000 if event.dur else 0
    
    # 标记起始和结束事件
    is_prev = (event.name == prev_event.name and event.ts == prev_event.ts)
    is_curr = (event.name == current_event.name and event.ts == current_event.ts)
    
    # 标记最大时间间隔的事件对
    is_max_gap_first = False
    is_max_gap_second = False
    if max_gap_events:
        first_event, second_event = max_gap_events
        is_max_gap_first = (event.name == first_event.name and event.ts == first_event.ts and 
                           event.pid == first_event.pid and event.tid == first_event.tid)
        is_max_gap_second = (event.name == second_event.name and event.ts == second_event.ts and 
                            event.pid == second_event.pid and event.tid == second_event.tid)
    
    markers = []
    if is_prev:
        markers.append(" [START]")
    if is_curr:
        markers.append(" [END]")
    if is_max_gap_first:
        markers.append(" [MAX_GAP_FIRST]")
    if is_max_gap_second:
        markers.append(" [MAX_GAP_SECOND]")
    
    marker_str = "".join(markers)
    
    print(f"        {prefix}{event.name}{marker_str} (ts={event.ts:.6f}, dur={duration_ms:.3f}ms, pid={event.pid}, tid={event.tid})")
    
    # 按开始时间排序子节点，便于跟踪时序
    sorted_children = sorted(node.children, key=lambda n: n.event.ts)
    
    for i, child in enumerate(sorted_children):
        is_last = i == len(sorted_children) - 1
        child_prefix = prefix + ("└── " if is_last else "├── ")
        _print_time_window_tree_recursive(
            child, prev_event, current_event, max_gap_events, depth + 1, child_prefix
        )

def _create_event_id_for_lookup(event) -> str:
    """
    创建事件ID用于查找，与call_stack_builder保持一致
    
    Args:
        event: ActivityEvent对象
        
    Returns:
        str: 事件ID
    """
    # 对triton事件应用相同的名称处理逻辑
    processed_name = _remove_triton_suffix(event.name)
    return f"{processed_name}:{event.ts}:{event.dur}:{event.pid}:{event.tid}"

def _extend_events_with_parents(events, builder):
    """
    扩展事件列表，包含所有相关父节点
    
    Args:
        events: 原始事件列表
        builder: CallStackBuilder实例
        
    Returns:
        List[ActivityEvent]: 扩展后的事件列表
    """
    extended_events = []
    added_event_ids = set()
    
    # 首先添加所有原始事件到去重集合中
    for event in events:
        event_id = builder._create_event_id_for_lookup(event)
        if event_id not in added_event_ids:
            extended_events.append(event)
            added_event_ids.add(event_id)
    
    # 为每个事件添加其父节点
    for event in events:
        event_id = builder._create_event_id_for_lookup(event)
        node = builder.event_to_node_map.get(event_id)
        
        if node:
            # 向上遍历父节点链
            current = node.parent
            while current and current.parent:  # 排除根节点
                parent_event = current.event
                parent_event_id = builder._create_event_id_for_lookup(parent_event)
                
                # 如果父节点还没有被添加
                if parent_event_id not in added_event_ids:
                    extended_events.append(parent_event)
                    added_event_ids.add(parent_event_id)
                    
                current = current.parent
    
    return extended_events

def _remove_triton_suffix(name: str) -> str:
    """
    去除triton名称中的suffix部分，与postprocessor和call_stack_builder保持一致
    
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

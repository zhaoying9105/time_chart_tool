import re
import os
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from ...models import ActivityEvent

def _readable_timestamp_to_microseconds(readable_timestamp: str) -> float:
    """将readable_timestamp转换为微秒时间戳"""
    try:
        dt = datetime.strptime(readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp() * 1_000_000
    except Exception as e:
        print(f"警告: 时间戳转换失败: {e}")
        return 0.0

def _format_timestamp_display(event, show_readable: bool = True) -> str:
    """格式化时间戳显示"""
    if show_readable:
        return event.readable_timestamp
    else:
        return f"{event.ts:.2f}μs"

def _calculate_end_time_display(event) -> str:
    """计算并格式化结束时间显示"""
    start_dt = datetime.strptime(event.readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    end_dt = start_dt + timedelta(microseconds=event.dur)
    return end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")

def _calculate_time_diff_readable(event1, event2) -> float:
    """使用readable_timestamp计算时间差（微秒）"""
    start1 = _readable_timestamp_to_microseconds(event1.readable_timestamp)
    start2 = _readable_timestamp_to_microseconds(event2.readable_timestamp)
    return abs(start1 - start2)

def _extract_events_in_range_with_intersection(events: List[ActivityEvent], start_time: float, end_time: float, 
                                              filtered_cpu_events_by_external_id: Dict[str, List[ActivityEvent]],
                                              filtered_kernel_events_by_external_id: Dict[str, List[ActivityEvent]]) -> Tuple[Dict[str, List[ActivityEvent]], Dict[str, List[ActivityEvent]]]:
    """
    提取指定时间范围内的events，并与已筛选的events取交集
    
    Args:
        start_time: 开始时间
        end_time: 结束时间
        filtered_cpu_events_by_external_id: 已筛选的CPU events字典
        filtered_kernel_events_by_external_id: 已筛选的Kernel events字典
        
    Returns:
        Tuple[Dict, Dict]: (intersected_cpu_events, intersected_kernel_events)
    """
    # 提取时间范围内的events
    range_cpu_events = []
    range_kernel_events = []
    
    for event in events:
        if event.ts is not None:
            event_end = event.ts + (event.dur if event.dur is not None else 0)
            # 只要event与时间范围有交集就保留
            if event.ts <= end_time and event_end >= start_time:
                if event.cat == 'cpu_op':
                    range_cpu_events.append(event)
                elif event.cat == 'kernel':
                    range_kernel_events.append(event)
    
    print(f"    时间范围内找到 {len(range_cpu_events)} 个CPU events, {len(range_kernel_events)} 个Kernel events")
    
    # 构建结果字典
    intersected_cpu_events = {}
    intersected_kernel_events = {}
    
    # 处理CPU events
    matched_cpu_count = 0
    for event in range_cpu_events:
        if event.external_id in filtered_cpu_events_by_external_id:
            if event.external_id not in intersected_cpu_events:
                intersected_cpu_events[event.external_id] = []
            intersected_cpu_events[event.external_id].append(event)
            matched_cpu_count += 1
            
    # 处理Kernel events
    matched_kernel_count = 0
    for event in range_kernel_events:
        if event.external_id in filtered_kernel_events_by_external_id:
            if event.external_id not in intersected_kernel_events:
                intersected_kernel_events[event.external_id] = []
            intersected_kernel_events[event.external_id].append(event)
            matched_kernel_count += 1
            
    print(f"    与筛选结果取交集后: {matched_cpu_count} 个CPU events, {matched_kernel_count} 个Kernel events")
    return intersected_cpu_events, intersected_kernel_events

def _get_analysis_start_time(events, prev_event: Optional[ActivityEvent], current_event: ActivityEvent) -> float:
    """
    确定分析的开始时间
    如果找到上一个通信kernel，则从其结束时间开始
    否则，从数据开始时间开始
    """
    if prev_event:
        return prev_event.ts + (prev_event.dur if prev_event.dur else 0)
    else:
        # 如果没有上一个通信kernel，则从数据最早的时间开始
        min_ts = float('inf')
        for event in events:
            if event.ts is not None and event.ts < min_ts:
                min_ts = event.ts
        return min_ts

def _get_analysis_end_time(current_event: ActivityEvent) -> float:
    """
    确定分析的结束时间
    即当前通信kernel的开始时间
    """
    return current_event.ts


def _find_prev_communication_kernel_events(events, target_kernel_events, prev_kernel_pattern):
    """找到目标通信kernel之前的通信kernel events，必须找到，否则抛出异常"""
    if not target_kernel_events:
        raise RuntimeError("    错误: 目标通信kernel events为空，无法查找上一个通信kernel")
    
    target_start_time = min(event.ts for event in target_kernel_events)
    target_kernel_name = target_kernel_events[0].name
    print(f"    目标通信kernel: {target_kernel_name}, 开始时间: {target_start_time/1000.0:.2f} ms")
    
    # 查找匹配条件的通信kernel events
    communication_kernels = _find_events_by_criteria(
        events,
        lambda e: (e.cat == 'kernel' and e.ts < target_start_time and
                    re.match(prev_kernel_pattern, e.name))
    )
    
    print(f"    找到 {len(communication_kernels)} 个匹配的通信kernel events")
    
    if communication_kernels:
        # 按结束时间排序，取最后一个
        communication_kernels.sort(key=lambda x: x.ts + x.dur)
        last_kernel_event = communication_kernels[-1]
        
        print(f"    上一个通信kernel: {last_kernel_event.name}")
        print(f"    开始时间: {last_kernel_event.ts/1000.0:.6f} ms")
        print(f"    结束时间: {(last_kernel_event.ts + last_kernel_event.dur)/1000.0:.6f} ms")
        
        return [last_kernel_event]
    else:
        error_msg = f"    错误: 没有找到匹配模式 '{prev_kernel_pattern}' 的上一个通信kernel"
        print(error_msg)
        raise RuntimeError(error_msg)

def _find_events_by_criteria(events, criteria_func) -> List[ActivityEvent]:
    """根据自定义条件查找事件"""
    return [event for event in events if criteria_func(event)]

def _find_communication_event(events, comm_idx: int, kernel_prefix: str) -> Optional[ActivityEvent]:
    """找到指定索引的通信kernel事件"""
    comm_events = _find_all_communication_events(events,kernel_prefix)
    if 0 <= comm_idx < len(comm_events):
        return comm_events[comm_idx]
    return None

def _find_all_communication_events(events, kernel_prefix: str) -> List[ActivityEvent]:
    """找到所有通信kernel事件并按时间排序"""
    comm_events = []
    escaped_prefix = re.escape(kernel_prefix)
    pattern = re.compile(rf"^{escaped_prefix}")
    
    for event in events:
        if event.cat == 'kernel' and pattern.match(event.name):
            comm_events.append(event)
    
    # 按时间戳排序
    comm_events.sort(key=lambda x: x.ts)
    return comm_events

def _print_communication_events_table(self, fastest_events, slowest_events) -> bool:
    """打印通信事件对比表格并返回一致性检查结果"""
    print("    通信事件对比表格:")
    
    # 定义列名
    headers = [
        "序号", "最快卡Kernel名称", "最快卡开始时间", "最快卡结束时间", 
        "最慢卡Kernel名称", "最慢卡开始时间", "最慢卡结束时间", 
        "名称一致", "开始时间差(ms)", "结束时间差(ms)", "超过阈值",
        "In msg nelems (Fast)", "In msg nelems (Slow)", "Nelems一致",
        "Group size (Fast)", "Group size (Slow)", "Group Size一致"
    ]
    
    # 定义每列的宽度
    widths = [
        6, 60, 30,30, 
        60, 20, 20, 
        10, 15, 15, 10,
        20, 20, 10,
        18, 18, 15
    ]
    
    # 打印表头
    header_row = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
    separator_row = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    
    print(f"    {header_row}")
    print(f"    {separator_row}")
    
    if len(fastest_events) != len(slowest_events):  
        print(f"    WARN: 最快卡和最慢卡的通信kernel events数量不一致 ({len(fastest_events)} vs {len(slowest_events)})")
    
    max_events = min(len(fastest_events), len(slowest_events))
    consistency_check_passed = True
    
    for i in range(max_events):
        fastest_event = fastest_events[i] if i < len(fastest_events) else None
        slowest_event = slowest_events[i] if i < len(slowest_events) else None
        
        # 基本信息
        fastest_name = fastest_event.name if fastest_event else "N/A"
        slowest_name = slowest_event.name if slowest_event else "N/A"
        name_consistent = "✓" if fastest_name == slowest_name else "✗"
        
        # 时间戳显示
        fastest_start_ts = _format_timestamp_display(fastest_event)
        slowest_start_ts = _format_timestamp_display(slowest_event)
        fastest_end_ts = _calculate_end_time_display(fastest_event)
        slowest_end_ts = _calculate_end_time_display(slowest_event)
        
        # 时间差计算
        start_time_diff, end_time_diff, threshold_exceeded = self._calculate_time_differences(
            fastest_event, slowest_event
        )
        
        if threshold_exceeded == "✗":
            consistency_check_passed = False
        
        # 提取 args 信息
        fastest_args = fastest_event.args if fastest_event else {}
        slowest_args = slowest_event.args if slowest_event else {}
        
        # 提取 'In msg nelems'
        fast_nelems = str(fastest_args.get('In msg nelems', 'N/A'))
        slow_nelems = str(slowest_args.get('In msg nelems', 'N/A'))
        nelems_consistent = "✓" if fast_nelems == slow_nelems else "✗"
        
        # 提取 'Group size'
        fast_group_size = str(fastest_args.get('Group size', 'N/A'))
        slow_group_size = str(slowest_args.get('Group size', 'N/A'))
        group_size_consistent = "✓" if fast_group_size == slow_group_size else "✗"
        
        # 构建数据行
        row_data = [
            str(i+1), fastest_name, fastest_start_ts, fastest_end_ts,
            slowest_name, slowest_start_ts, slowest_end_ts,
            name_consistent, start_time_diff, end_time_diff, threshold_exceeded,
            fast_nelems, slow_nelems, nelems_consistent,
            fast_group_size, slow_group_size, group_size_consistent
        ]
        
        # 格式化并打印行
        row_str = "| " + " | ".join(f"{d:<{w}}" for d, w in zip(row_data, widths)) + " |"
        print(f"    {row_str}")
    
    return consistency_check_passed

def _merge_cpu_and_kernel_events(cpu_events_by_external_id, kernel_events_by_external_id):
    """合并CPU和kernel events到一个字典中"""
    merged = {}
    
    # 首先添加CPU events
    for ext_id, events in cpu_events_by_external_id.items():
        if ext_id not in merged:
            merged[ext_id] = []
        merged[ext_id].extend(events)
        
    # 然后添加Kernel events
    for ext_id, events in kernel_events_by_external_id.items():
        if ext_id not in merged:
            merged[ext_id] = []
        merged[ext_id].extend(events)
        
    return merged

def _check_kernel_ops_consistency(fastest_events_by_external_id: Dict[str, List['ActivityEvent']], slowest_events_by_external_id: Dict[str, List['ActivityEvent']]) -> bool:
    """检查快卡和慢卡的kernel操作是否一致"""
    # 提取所有kernel events
    fastest_kernel_events = []
    slowest_kernel_events = []
    
    for events in fastest_events_by_external_id.values():
        fastest_kernel_events.extend([e for e in events if e.cat == 'kernel'])
    for events in slowest_events_by_external_id.values():
        slowest_kernel_events.extend([e for e in events if e.cat == 'kernel'])
    
    # 按结束时间排序
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

def _check_communication_kernel_consistency(fastest_events: List['ActivityEvent'], slowest_events: List['ActivityEvent'],
                                              kernel_prefix: str, comm_idx: int) -> bool:
    """
    检查快慢卡的所有通信kernel顺序和时间一致性
    
    Args:
        fastest_events: 最快card的所有ActivityEvent
        slowest_events: 最慢card的所有ActivityEvent
        kernel_prefix: 通信kernel前缀（用于查找目标通信操作）
        comm_idx: 通信操作索引
        
    Returns:
        bool: 是否一致
    """
    print("=== 检查通信kernel一致性 ===")
    
    # 1. 找到所有TCDP_开头的通信kernel events（用于检查顺序和同步性）
    fastest_all_comm_events = _find_all_communication_events(fastest_events, kernel_prefix)
    slowest_all_comm_events = _find_all_communication_events(slowest_events, kernel_prefix)
    for i, (fastest_event, slowest_event) in enumerate(zip(fastest_all_comm_events, slowest_all_comm_events)):
        print(f"    对比 {i+1}/{len(fastest_all_comm_events)}, 最快卡: {fastest_event.name}, 最慢卡: {slowest_event.name}")
    
    # 2. 找到目标通信kernel events（用于确定分析范围）
    fastest_target_comm_event = _find_communication_event(fastest_events, comm_idx, kernel_prefix)
    slowest_target_comm_event = _find_communication_event(slowest_events, comm_idx, kernel_prefix)
    
    # 打印 fastest_target_comm_event 和 slowest_target_comm_event 的 duration
    if fastest_target_comm_event:
        print(f"fastest_target_comm_event duration: {fastest_target_comm_event.dur}")
    if slowest_target_comm_event:
        print(f"slowest_target_comm_event duration: {slowest_target_comm_event.dur}")
    
    if not fastest_target_comm_event or not slowest_target_comm_event:
        print("    错误: 无法找到目标通信kernel events")
        return False
    
    if len(fastest_all_comm_events) == 0:
        print("    错误: 无法找到任何通信kernel events (fastest card)")
        return False
    else:
        print(f"    找到 {len(fastest_all_comm_events)} 个通信kernel events (fastest card)")
    
    if len(slowest_all_comm_events) == 0:
        print("    错误: 无法找到任何通信kernel events (slowest card)")
        return False
    else:
        print(f"    找到 {len(slowest_all_comm_events)} 个通信kernel events (slowest card)")
    
    # 2. 时间戳标准化已在parser中完成
    print("    时间戳标准化已在parser中完成")
    # 4. 检查kernel名称顺序一致性
    fastest_kernel_names = [event.name for event in fastest_all_comm_events]
    slowest_kernel_names = [event.name for event in slowest_all_comm_events]
    
    print(f"    最快卡通信kernel序列: {fastest_kernel_names}")
    print(f"    最慢卡通信kernel序列: {slowest_kernel_names}")
    
    # 3. 打印通信事件对比表格
    consistency_check_passed = _print_communication_events_table_helper(
        fastest_all_comm_events, slowest_all_comm_events
    )
    
    
    sequence_consistent = (fastest_kernel_names == slowest_kernel_names)
    if not sequence_consistent:
        print("    警告: 通信kernel名称序列不一致!")
        print(f"    最快卡: {fastest_kernel_names}")
        print(f"    最慢卡: {slowest_kernel_names}")
    else:
        print("    ✓ 通信kernel名称序列一致")
    
    # 5. 检查结果
    if not consistency_check_passed:
        print("    ✗ 通信kernel一致性检查失败: 存在超过5000us阈值的事件")
        return False
    else:
        print("    ✓ 通信kernel一致性检查通过")
        return True

def _print_communication_events_table_helper(fastest_events, slowest_events) -> bool:
    """打印通信事件对比表格并返回一致性检查结果"""
    print("    通信事件对比表格:")
    
    # 定义列名
    headers = [
        "序号", "最快卡Kernel名称", "最快卡开始时间", "最快卡结束时间", 
        "最慢卡Kernel名称", "最慢卡开始时间", "最慢卡结束时间", 
        "名称一致", "开始时间差(ms)", "结束时间差(ms)", "超过阈值",
        "In msg nelems (Fast)", "In msg nelems (Slow)", "Nelems一致",
        "Group size (Fast)", "Group size (Slow)", "Group Size一致"
    ]
    
    # 定义每列的宽度
    widths = [
        6, 60, 30,30, 
        60, 20, 20, 
        10, 15, 15, 10,
        20, 20, 10,
        18, 18, 15
    ]
    
    # 打印表头
    header_row = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
    separator_row = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    
    print(f"    {header_row}")
    print(f"    {separator_row}")
    
    if len(fastest_events) != len(slowest_events):  
        print(f"    WARN: 最快卡和最慢卡的通信kernel events数量不一致 ({len(fastest_events)} vs {len(slowest_events)})")
    
    min_events = min(len(fastest_events), len(slowest_events))
    consistency_check_passed = True
    
    print(f"    对比 {min_events} 个通信kernel events")
    for i in range(min_events):
        print(f"    对比 {i+1}/{min_events}")
        fastest_event = fastest_events[i] 
        slowest_event = slowest_events[i] 
        
        # 基本信息
        fastest_name = fastest_event.name 
        slowest_name = slowest_event.name 
        name_consistent = "✓" if fastest_name == slowest_name else "✗"
        
        # 时间戳显示
        print(f"fastest_event.readable_timestamp: {fastest_event.readable_timestamp}")
        fastest_start_ts = _format_timestamp_display(fastest_event) 
        slowest_start_ts = _format_timestamp_display(slowest_event) 
        print(f"fastest_event.readable_timestamp: {fastest_event.readable_timestamp}")
        fastest_end_ts = _calculate_end_time_display(fastest_event) 
        slowest_end_ts = _calculate_end_time_display(slowest_event) 
        
        # 时间差计算
        start_time_diff, end_time_diff, threshold_exceeded = _calculate_time_differences(
            fastest_event, slowest_event
        )
        
        if threshold_exceeded == "✗":
            consistency_check_passed = False
        
        # 提取 args 信息
        fastest_args = fastest_event.args if fastest_event else {}
        slowest_args = slowest_event.args if slowest_event else {}
        
        # 提取 'In msg nelems'
        fast_nelems = str(fastest_args.get('In msg nelems', 'N/A'))
        slow_nelems = str(slowest_args.get('In msg nelems', 'N/A'))
        nelems_consistent = "✓" if fast_nelems == slow_nelems else "✗"
        
        # 提取 'Group size'
        fast_group_size = str(fastest_args.get('Group size', 'N/A'))
        slow_group_size = str(slowest_args.get('Group size', 'N/A'))
        group_size_consistent = "✓" if fast_group_size == slow_group_size else "✗"
        
        # 构建数据行
        row_data = [
            str(i+1), fastest_name, fastest_start_ts, fastest_end_ts,
            slowest_name, slowest_start_ts, slowest_end_ts,
            name_consistent, start_time_diff, end_time_diff, threshold_exceeded,
            fast_nelems, slow_nelems, nelems_consistent,
            fast_group_size, slow_group_size, group_size_consistent
        ]
        
        # 格式化并打印行
        row_str = "| " + " | ".join(f"{d:<{w}}" for d, w in zip(row_data, widths)) + " |"
        print(f"    {row_str}")
    
    return consistency_check_passed

def _calculate_time_differences(fastest_event, slowest_event):
    """计算时间差，返回 (start_diff, end_diff, threshold_exceeded)"""
    if not fastest_event or not slowest_event:
        return "N/A", "N/A", "N/A"
        
    # 计算readable timestamp的微秒差值
    start_diff = _calculate_time_diff_readable(fastest_event, slowest_event)
    
    # 结束时间差值 (duration + start_time)
    fast_end = _readable_timestamp_to_microseconds(fastest_event.readable_timestamp) + fastest_event.dur
    slow_end = _readable_timestamp_to_microseconds(slowest_event.readable_timestamp) + slowest_event.dur
    end_diff = abs(fast_end - slow_end)
    
    threshold_exceeded = "✗" if start_diff > 5000 or end_diff > 5000 else "✓"
    
    return f"{start_diff:.2f}", f"{end_diff:.2f}", threshold_exceeded

def _scan_executor_folders(pod_dir: str) -> List[str]:
    """
    扫描Pod目录下的executor文件夹
    
    Args:
        pod_dir: Pod目录路径
        
    Returns:
        List[str]: executor文件夹路径列表
    """
    executor_folders = []
    
    for item in os.listdir(pod_dir):
        item_path = os.path.join(pod_dir, item)
        if os.path.isdir(item_path) and item.startswith('executor_trainer-runner_'):
            executor_folders.append(item_path)
    
    return sorted(executor_folders)

def _calculate_step_statistics(all2all_data):
    """计算step级别统计信息 - 对每个step和Comm_Op_Index统计最快最慢card"""
    step_stats = []
    
    for step, card_data in all2all_data.items():
        # 按Comm_Op_Index分组统计
        comm_op_groups = {}
        
        # 收集所有card的duration数据
        for card_idx, entries in card_data.items():
            for entry_idx, entry in enumerate(entries):
                if entry_idx not in comm_op_groups:
                    comm_op_groups[entry_idx] = []
                comm_op_groups[entry_idx].append({
                    'card_idx': card_idx,
                    'entry': entry
                })
        
        # 对每个Comm_Op_Index计算统计信息
        for comm_op_idx, card_entries in comm_op_groups.items():
            if not card_entries:
                continue
            
            # 找到最快和最慢的card
            fastest_card = min(card_entries, key=lambda x: x['entry'].dur)
            slowest_card = max(card_entries, key=lambda x: x['entry'].dur)
            
            # 所有的entries应该有相同的元数据，取第一个
            meta_entry = card_entries[0]['entry']
            
            # 计算倍数
            ratio = slowest_card['entry'].dur / fastest_card['entry'].dur if fastest_card['entry'].dur > 0 else float('inf')
            
            step_stats.append({
                "Step": step,
                "Comm_Op_Index": comm_op_idx,
                "Name": meta_entry.name,
                "In_Msg_Nelems": meta_entry.in_msg_nelems,
                "Out_Msg_Nelems": meta_entry.out_msg_nelems,
                "Group_Size": meta_entry.group_size,
                "Dtype": meta_entry.dtype,
                "Process_Group_Ranks": meta_entry.process_group_ranks,
                "Fastest_Card_Index": fastest_card['card_idx'],
                "Fastest_Duration_us": fastest_card['entry'].dur,
                "Slowest_Card_Index": slowest_card['card_idx'],
                "Slowest_Duration_us": slowest_card['entry'].dur,
                "Duration_Ratio": ratio,
                "Total_Cards": len(card_entries)
            })
    
    return step_stats

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from .event_aligner import EventAligner

# 导入必要的类型
try:
    from ...parser.profiler_data import ActivityEvent
except ImportError:
    # Fallback for when running as a standalone script or in different context
    pass

def _readable_timestamp_to_microseconds(readable_timestamp: str) -> float:
    """将readable_timestamp转换为微秒时间戳"""
    try:
        dt = datetime.strptime(readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp() * 1_000_000
    except Exception as e:
        print(f"警告: 时间戳转换失败: {e}")
        return 0.0

def _find_kernel_events_for_cpu_event(cpu_event: 'ActivityEvent', events_by_external_id: Dict[str, List['ActivityEvent']]) -> List['ActivityEvent']:
    """为CPU event找到对应的kernel events"""
    if cpu_event.external_id is None:
        return []
    
    if cpu_event.external_id not in events_by_external_id:
        return []
    
    events = events_by_external_id[cpu_event.external_id]
    kernel_events = [e for e in events if e.cat == 'kernel']
    
    return kernel_events


def find_top_kernel_duration_ratios(comparison_rows: List[Dict], top_n: int = 5) -> List[Tuple[int, float, Dict]]:
    """找出kernel duration ratio最大的前N个事件"""
    if not comparison_rows:
        return []
    
    # 提取kernel_duration_diff_ratio并添加索引
    kernel_ratios_with_index = []
    for i, row in enumerate(comparison_rows):
        ratio = row.get('kernel_duration_diff_ratio', 0)
        kernel_ratios_with_index.append((i, ratio, row))
    
    # 按ratio降序排序
    kernel_ratios_with_index.sort(key=lambda x: x[1], reverse=True)
    
    # 返回前N个
    return kernel_ratios_with_index[:top_n]

def _calculate_timing_analysis(prev_fastest_event, current_fastest_event, prev_slowest_event, current_slowest_event):
    """计算时间差分析"""
    analysis = {}
    
    if not (prev_fastest_event and current_fastest_event and prev_slowest_event and current_slowest_event):
        return analysis

    # 1. 快卡分析
    fast_ts_diff = current_fastest_event.ts - prev_fastest_event.ts
    fast_end_to_start = current_fastest_event.ts - (prev_fastest_event.ts + prev_fastest_event.dur)
    
    analysis['fastest_card'] = {
        'event_pair_ts_diff': fast_ts_diff,
        'end_to_start_gap': fast_end_to_start
    }
    
    # 2. 慢卡分析
    slow_ts_diff = current_slowest_event.ts - prev_slowest_event.ts
    slow_end_to_start = current_slowest_event.ts - (prev_slowest_event.ts + prev_slowest_event.dur)
    
    analysis['slowest_card'] = {
        'event_pair_ts_diff': slow_ts_diff,
        'end_to_start_gap': slow_end_to_start
    }
    
    # 3. 快慢卡差距分析
    analysis['fast_slow_diff'] = {
        'event_pair_ts_diff': slow_ts_diff - fast_ts_diff,
        'end_to_start_gap': slow_end_to_start - fast_end_to_start
    }
    
    return analysis

def find_top_cpu_start_time_differences(comparison_rows: List[Dict], aligned_fastest: List = None, aligned_slowest: List = None, top_n: int = 5) -> List[Dict]:
    """找出cpu start time相邻差值最大的前N对"""
    if len(comparison_rows) < 2:
        return []
    
    # 计算相邻行的cpu_start_time_diff_ratio差值
    differences = []
    for i in range(1, len(comparison_rows)):
        current_ratio = comparison_rows[i].get('cpu_start_time_diff_ratio', 0)
        prev_ratio = comparison_rows[i-1].get('cpu_start_time_diff_ratio', 0)
        diff = abs(current_ratio - prev_ratio)
        
        # 获取对应的事件对象
        prev_fastest_event = None
        current_fastest_event = None
        prev_slowest_event = None
        current_slowest_event = None
        
        if aligned_fastest and aligned_slowest:
            # 使用 event_sequence 作为对齐后列表的索引
            prev_event_sequence = comparison_rows[i-1].get('event_sequence')
            current_event_sequence = comparison_rows[i].get('event_sequence')
            
            if prev_event_sequence is not None and current_event_sequence is not None:
                # event_sequence 从1开始，需要减1转换为0基索引
                prev_idx = prev_event_sequence - 1
                current_idx = current_event_sequence - 1
                
                if prev_idx < len(aligned_fastest) and prev_idx < len(aligned_slowest):
                    prev_fastest_event = aligned_fastest[prev_idx]
                    prev_slowest_event = aligned_slowest[prev_idx]
                    
                    # 检查是否为None事件
                    if prev_fastest_event is None:
                        print(f"[调试] prev_fastest_event is None at index {prev_idx}")
                        continue
                
                if current_idx < len(aligned_fastest) and current_idx < len(aligned_slowest):
                    current_fastest_event = aligned_fastest[current_idx]
                    current_slowest_event = aligned_slowest[current_idx]
                    
                    # 检查是否为None事件
                    if current_fastest_event is None:
                        print(f"[调试] current_fastest_event is None at index {current_idx}")
                        continue
        
        # 检查事件名称一致性
        if prev_fastest_event and prev_slowest_event:
            if prev_fastest_event.name != prev_slowest_event.name:
                print(f"[警告] find_top_cpu_start_time_differences: 前一个事件名称不匹配: '{prev_fastest_event.name}' vs '{prev_slowest_event.name}'")
        
        if current_fastest_event and current_slowest_event:
            if current_fastest_event.name != current_slowest_event.name:
                print(f"[警告] find_top_cpu_start_time_differences: 当前事件名称不匹配: '{current_fastest_event.name}' vs '{current_slowest_event.name}'")
        
        # 计算时间差分析
        timing_analysis = _calculate_timing_analysis(
            prev_fastest_event, current_fastest_event,
            prev_slowest_event, current_slowest_event
        )
        
        differences.append({
            'index_pair': (i-1, i),
            'difference': diff,
            'prev_ratio': prev_ratio,
            'current_ratio': current_ratio,
            'prev_row': comparison_rows[i-1],
            'current_row': comparison_rows[i],
            # 添加事件对象
            'prev_fastest_event': prev_fastest_event,
            'current_fastest_event': current_fastest_event,
            'prev_slowest_event': prev_slowest_event,
            'current_slowest_event': current_slowest_event,
            # 添加时间差分析
            'timing_analysis': timing_analysis
        })
    
    # 按差值降序排序
    differences.sort(key=lambda x: x['difference'], reverse=True)
    
    # 返回前N对
    return differences[:top_n]

def compare_events_by_time_sequence(fastest_events_by_external_id: Dict[str, List['ActivityEvent']], 
                                  slowest_events_by_external_id: Dict[str, List['ActivityEvent']],
                                  fastest_card_idx: int, slowest_card_idx: int, 
                                  fastest_duration: float, slowest_duration: float,
                                  fastest_events=None, slowest_events=None,
                                  show_timestamp: bool = False, show_readable_timestamp: bool = False,
                                  output_dir: str = ".", step: int = None, comm_idx: int = None) -> Dict:
    """按照时间顺序比较快卡和慢卡的events"""
    comparison_rows = []
    
    # 1. 提取所有CPU events并按时间排序
    fastest_cpu_events = []
    slowest_cpu_events = []
    
    for events in fastest_events_by_external_id.values():
        fastest_cpu_events.extend([e for e in events if e.cat == 'cpu_op'])
    for events in slowest_events_by_external_id.values():
        slowest_cpu_events.extend([e for e in events if e.cat == 'cpu_op'])
    
    # 按开始时间排序
    fastest_cpu_events.sort(key=lambda x: x.ts)
    slowest_cpu_events.sort(key=lambda x: x.ts)
    
    print(f"最快card找到 {len(fastest_cpu_events)} 个CPU events")
    print(f"最慢card找到 {len(slowest_cpu_events)} 个CPU events")
    
    # 2. 使用LCS模糊对齐算法
    print("\n=== 执行LCS模糊对齐 ===")
    event_aligner = EventAligner()
    aligned_fastest, aligned_slowest, alignment_info = event_aligner.fuzzy_align_cpu_events(
        fastest_cpu_events, slowest_cpu_events
    )
    
    # 3. 检查kernel操作是否一致
    from .utils import _check_kernel_ops_consistency
    kernel_ops_consistent = _check_kernel_ops_consistency(fastest_events_by_external_id, slowest_events_by_external_id)
    if not kernel_ops_consistent:
        print("警告: 快卡和慢卡的kernel操作不完全一致，比较结果可能不准确")
    else:
        print("Kernel操作一致性检查通过")
    
    # 4. 使用对齐后的事件进行配对比较
    matched_events_count = len([info for info in alignment_info if info['match_type'] == 'exact_match'])
    
    for i in range(len(aligned_fastest)):
        fastest_cpu_event = aligned_fastest[i]
        slowest_cpu_event = aligned_slowest[i]
        alignment = alignment_info[i]
        
        # 跳过未匹配的事件
        if fastest_cpu_event is None or slowest_cpu_event is None:
            continue
        
        # 找到对应的kernel events
        fastest_kernel_events = _find_kernel_events_for_cpu_event(fastest_cpu_event, fastest_events_by_external_id)
        slowest_kernel_events = _find_kernel_events_for_cpu_event(slowest_cpu_event, slowest_events_by_external_id)
        
        # 计算时间差异 - 使用readable_timestamp
        fastest_start_us = _readable_timestamp_to_microseconds(fastest_cpu_event.readable_timestamp)
        slowest_start_us = _readable_timestamp_to_microseconds(slowest_cpu_event.readable_timestamp)
        cpu_start_time_diff = slowest_start_us - fastest_start_us
        
        cpu_duration_diff = slowest_cpu_event.dur - fastest_cpu_event.dur
        
        # 计算kernel时间差异
        fastest_kernel_duration = sum(e.dur for e in fastest_kernel_events)
        slowest_kernel_duration = sum(e.dur for e in slowest_kernel_events)
        kernel_duration_diff = slowest_kernel_duration - fastest_kernel_duration
        
        # 计算时间差异占all2all duration差异的比例
        all2all_duration_diff = slowest_duration - fastest_duration
        cpu_start_time_diff_ratio = cpu_start_time_diff / all2all_duration_diff if all2all_duration_diff > 0 else 0
        kernel_duration_diff_ratio = kernel_duration_diff / all2all_duration_diff if all2all_duration_diff > 0 else 0
        
        # 构建行数据
        # 根据show参数决定展示的时间格式和列名
        if show_readable_timestamp:
            # 使用readable_timestamp转换后的值
            fastest_display_time = _readable_timestamp_to_microseconds(fastest_cpu_event.readable_timestamp)
            slowest_display_time = _readable_timestamp_to_microseconds(slowest_cpu_event.readable_timestamp)
            fastest_time_col = 'fastest_cpu_start_time_readable'
            slowest_time_col = 'slowest_cpu_start_time_readable'
        else:
            # 使用原始ts值（适合chrome tracing显示）
            fastest_display_time = fastest_cpu_event.ts
            slowest_display_time = slowest_cpu_event.ts
            fastest_time_col = 'fastest_cpu_start_time'
            slowest_time_col = 'slowest_cpu_start_time'
        
        row = {
            'event_sequence': i + 1,
            'cpu_op_name': fastest_cpu_event.name,
            'slowest_cpu_op_name': slowest_cpu_event.name,
            'cpu_op_shape': str(fastest_cpu_event.args.get('Input Dims', '') if fastest_cpu_event.args else ''),
            'slowest_cpu_op_shape': str(slowest_cpu_event.args.get('Input Dims', '') if slowest_cpu_event.args else ''),
            'cpu_op_dtype': str(fastest_cpu_event.args.get('Input type', '') if fastest_cpu_event.args else ''),
            'slowest_cpu_op_dtype': str(slowest_cpu_event.args.get('Input type', '') if slowest_cpu_event.args else ''),
            fastest_time_col: fastest_display_time,
            slowest_time_col: slowest_display_time,
            'cpu_start_time_diff': cpu_start_time_diff,
            'cpu_start_time_diff_ratio': cpu_start_time_diff_ratio,
            'fastest_cpu_duration': fastest_cpu_event.dur,
            'slowest_cpu_duration': slowest_cpu_event.dur,
            'cpu_duration_diff': cpu_duration_diff,
            'fastest_kernel_duration': fastest_kernel_duration,
            'slowest_kernel_duration': slowest_kernel_duration,
            'kernel_duration_diff': kernel_duration_diff,
            'kernel_duration_diff_ratio': kernel_duration_diff_ratio,
            'fastest_card_idx': fastest_card_idx,
            'slowest_card_idx': slowest_card_idx,
            'fastest_cpu_readable_timestamp': fastest_cpu_event.readable_timestamp,
            'slowest_cpu_readable_timestamp': slowest_cpu_event.readable_timestamp,
            # 对齐信息
            'alignment_type': alignment['match_type'],
            'alignment_fast_idx': alignment['fast_idx'],
            'alignment_slow_idx': alignment['slow_idx']
        }
        
        comparison_rows.append(row)
    
    print(f"成功比较了 {len(comparison_rows)} 个CPU events")
    
    # 5. 可视化对齐结果
    print("\n=== 生成对齐结果可视化 ===")
    visualization_files = event_aligner.visualize_alignment_results(
        fastest_cpu_events, slowest_cpu_events,
        aligned_fastest, aligned_slowest, alignment_info,
        output_dir, step, comm_idx
    )
    print(f"已生成 {len(visualization_files)} 个可视化文件")
    
    # 对齐质量评估
    print("\n=== 对齐质量评估 ===")
    total_aligned = len(comparison_rows)
    exact_count = sum(1 for row in comparison_rows if row['alignment_type'] == 'exact_match')
    
    print(f"总对齐事件数: {total_aligned}")
    if total_aligned > 0:
        print(f"精确匹配: {exact_count} ({exact_count/total_aligned*100:.1f}%)")
    else:
        print(f"精确匹配: 0 (0.0%)")
    
    # 显示一些对齐示例
    print("\n=== 对齐示例 ===")
    matched_examples = [row for row in comparison_rows if row['alignment_type'] == 'exact_match'][:5]
    for i, row in enumerate(matched_examples):
        print(f"  {i+1}. 精确匹配: {row['cpu_op_name']} <-> {row['slowest_cpu_op_name']}")
    
    # 分析kernel duration ratio - 找出前5个最大的
    print("\n=== Kernel Duration Ratio 分析 ===")
    top_kernel_duration_ratios = find_top_kernel_duration_ratios(comparison_rows, top_n=5)
    print(f"找到 {len(top_kernel_duration_ratios)} 个最大的kernel duration ratio事件:")
    for i, (idx, ratio, row) in enumerate(top_kernel_duration_ratios):
        alignment_type_str = {
            'exact_match': '精确匹配',
            'unmatched_fast': '快卡未匹配',
            'unmatched_slow': '慢卡未匹配'
        }.get(row['alignment_type'], row['alignment_type'])
        
        print(f"  {i+1}. 事件序列 {idx+1} ({alignment_type_str}):")
        print(f"     最快Card CPU操作: {row['cpu_op_name']}")
        print(f"     最慢Card CPU操作: {row.get('slowest_cpu_op_name', 'N/A')}")
        print(f"     最快Card Timestamp: {row['fastest_cpu_readable_timestamp']} (ts: {row.get('fastest_cpu_start_time', 'N/A')})")
        print(f"     最慢Card Timestamp: {row['slowest_cpu_readable_timestamp']} (ts: {row.get('slowest_cpu_start_time', 'N/A')})")
        print(f"     Kernel Duration Ratio: {ratio:.4f}")
        print(f"     最快Card操作形状: {row['cpu_op_shape']}, 类型: {row['cpu_op_dtype']}")
        print(f"     最慢Card操作形状: {row.get('slowest_cpu_op_shape', 'N/A')}, 类型: {row.get('slowest_cpu_op_dtype', 'N/A')}")
        print()
    
    # 分析cpu start time相邻差值 - 找出前5对最大的
    print("\n=== CPU Start Time 相邻差值分析 ===")
    top_cpu_start_time_differences = find_top_cpu_start_time_differences(
        comparison_rows, aligned_fastest, aligned_slowest, top_n=5
    )
    print(f"找到 {len(top_cpu_start_time_differences)} 对最大的CPU start time相邻差值:")
    for i, diff_info in enumerate(top_cpu_start_time_differences):
        prev_idx, current_idx = diff_info['index_pair']
        prev_row = diff_info['prev_row']
        current_row = diff_info['current_row']
        
        prev_alignment_str = {
            'exact_match': '精确匹配',
            'unmatched_fast': '快卡未匹配',
            'unmatched_slow': '慢卡未匹配'
        }.get(prev_row['alignment_type'], prev_row['alignment_type'])
        
        current_alignment_str = {
            'exact_match': '精确匹配',
            'unmatched_fast': '快卡未匹配',
            'unmatched_slow': '慢卡未匹配'
        }.get(current_row['alignment_type'], current_row['alignment_type'])
        
        print(f"  {i+1}. 事件对 ({prev_idx+1}, {current_idx+1}):")
        print(f"     相邻差值: {diff_info['difference']:.4f}")
        
        # 添加时间差分析信息
        timing_analysis = diff_info.get('timing_analysis', {})
        if timing_analysis:
            print(f"     时间差分析:")
            
            # 快卡分析
            fastest_card = timing_analysis.get('fastest_card', {})
            if fastest_card.get('event_pair_ts_diff') is not None:
                print(f"       快卡事件对ts差值: {fastest_card['event_pair_ts_diff']:.4f}")
            if fastest_card.get('end_to_start_gap') is not None:
                print(f"       快卡end到start间隔: {fastest_card['end_to_start_gap']:.4f}")
            
            # 慢卡分析
            slowest_card = timing_analysis.get('slowest_card', {})
            if slowest_card.get('event_pair_ts_diff') is not None:
                print(f"       慢卡事件对ts差值: {slowest_card['event_pair_ts_diff']:.4f}")
            if slowest_card.get('end_to_start_gap') is not None:
                print(f"       慢卡end到start间隔: {slowest_card['end_to_start_gap']:.4f}")
            
            # 快慢卡差距分析
            fast_slow_diff = timing_analysis.get('fast_slow_diff', {})
            if fast_slow_diff.get('event_pair_ts_diff') is not None:
                print(f"       快慢卡ts差值差距: {fast_slow_diff['event_pair_ts_diff']:.4f}")
            if fast_slow_diff.get('end_to_start_gap') is not None:
                print(f"       快慢卡间隔差距: {fast_slow_diff['end_to_start_gap']:.4f}")
        
        print()
        
        # 四列竖排格式 - 扩展宽度以容纳长调用栈
        col_width = 120  # 每列宽度从30扩展到50
        total_width = col_width * 4 + 9  # 4列 + 3个分隔符(3个|) + 6个空格
        
        print("     " + "="*total_width)
        print("     " + f"{'最快Card前一个事件':<{col_width}} | {'最慢Card前一个事件':<{col_width}} | {'最快Card当前事件':<{col_width}} | {'最慢Card当前事件':<{col_width}}")
        print("     " + "="*total_width)
        
        # 第一行：事件名称
        prev_fastest_name = prev_row['cpu_op_name']
        prev_slowest_name = prev_row.get('slowest_cpu_op_name', 'N/A')
        current_fastest_name = current_row['cpu_op_name']
        current_slowest_name = current_row.get('slowest_cpu_op_name', 'N/A')
        
        print("     " + f"{prev_fastest_name:<{col_width}} | {prev_slowest_name:<{col_width}} | {current_fastest_name:<{col_width}} | {current_slowest_name:<{col_width}}")
        
        # 第二行：对齐类型
        print("     " + f"{prev_alignment_str:<{col_width}} | {prev_alignment_str:<{col_width}} | {current_alignment_str:<{col_width}} | {current_alignment_str:<{col_width}}")
        
        # 第三行：时间戳
        prev_fastest_ts = prev_row['fastest_cpu_readable_timestamp']
        prev_slowest_ts = prev_row['slowest_cpu_readable_timestamp']
        current_fastest_ts = current_row['fastest_cpu_readable_timestamp']
        current_slowest_ts = current_row['slowest_cpu_readable_timestamp']
        
        print("     " + f"{prev_fastest_ts:<{col_width}} | {prev_slowest_ts:<{col_width}} | {current_fastest_ts:<{col_width}} | {current_slowest_ts:<{col_width}}")
        
        # 第四行：Ratio
        print("     " + f"Ratio: {diff_info['prev_ratio']:.4f}{'':<{col_width-12}} | {'':<{col_width}} | Ratio: {diff_info['current_ratio']:.4f}{'':<{col_width-12}} | {'':<{col_width}}")
        
        # 第五行：事件名称匹配验证
        prev_name_match = "✅ 匹配" if prev_fastest_name == prev_slowest_name else f"❌ 不匹配"
        current_name_match = "✅ 匹配" if current_fastest_name == current_slowest_name else f"❌ 不匹配"
        
        print("     " + f"{prev_name_match:<{col_width}} | {'':<{col_width}} | {current_name_match:<{col_width}} | {'':<{col_width}}")
        
        # 调用栈信息
        print("     " + "-"*total_width)
        print("     " + f"{'调用栈 (from tree)':<{col_width}} | {'调用栈 (from tree)':<{col_width}} | {'调用栈 (from tree)':<{col_width}} | {'调用栈 (from tree)':<{col_width}}")
        print("     " + "-"*total_width)
        
        # 获取调用栈信息
        prev_fastest_call_stack = []
        prev_slowest_call_stack = []
        current_fastest_call_stack = []
        current_slowest_call_stack = []
        
        if diff_info.get('prev_fastest_event'):
            prev_fastest_call_stack = diff_info['prev_fastest_event'].call_stack or []
        if diff_info.get('prev_slowest_event'):
            prev_slowest_call_stack = diff_info['prev_slowest_event'].call_stack or []
        if diff_info.get('current_fastest_event'):
            current_fastest_call_stack = diff_info['current_fastest_event'].call_stack or []
        if diff_info.get('current_slowest_event'):
            current_slowest_call_stack = diff_info['current_slowest_event'].call_stack or []
        
        # 计算最大调用栈长度
        max_stack_len = max(len(prev_fastest_call_stack), len(prev_slowest_call_stack), 
                          len(current_fastest_call_stack), len(current_slowest_call_stack))
        
        for stack_idx in range(max_stack_len):
            c1 = prev_fastest_call_stack[stack_idx] if stack_idx < len(prev_fastest_call_stack) else ""
            c2 = prev_slowest_call_stack[stack_idx] if stack_idx < len(prev_slowest_call_stack) else ""
            c3 = current_fastest_call_stack[stack_idx] if stack_idx < len(current_fastest_call_stack) else ""
            c4 = current_slowest_call_stack[stack_idx] if stack_idx < len(current_slowest_call_stack) else ""
            
            # 截断过长的调用栈行
            if len(c1) > col_width - 2: c1 = c1[:col_width-5] + "..."
            if len(c2) > col_width - 2: c2 = c2[:col_width-5] + "..."
            if len(c3) > col_width - 2: c3 = c3[:col_width-5] + "..."
            if len(c4) > col_width - 2: c4 = c4[:col_width-5] + "..."
            
            print("     " + f"{c1:<{col_width}} | {c2:<{col_width}} | {c3:<{col_width}} | {c4:<{col_width}}")
        
        print("     " + "="*total_width)
        print()

    return {
        'comparison_rows': comparison_rows,
        'top_kernel_duration_ratios': top_kernel_duration_ratios,
        'top_cpu_start_time_differences': top_cpu_start_time_differences
    }

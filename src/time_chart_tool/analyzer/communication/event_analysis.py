"""
事件分析模块
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import re
from collections import defaultdict

from ....models import ActivityEvent, ProfilerData
from ....parser import PyTorchProfilerParser
from ..stages.postprocessor import DataPostProcessor
from .base import _format_timestamp_display, _calculate_end_time_display, _calculate_time_diff_readable


class EventAnalyzer:
    """事件分析器"""
    
    def __init__(self):
        self.postprocessor = DataPostProcessor()
    
    def find_prev_communication_kernel_events(self, data: ProfilerData, 
                                            target_kernel_events: List[ActivityEvent], 
                                            prev_kernel_pattern: str) -> List[ActivityEvent]:
        """找到目标通信kernel之前的通信kernel events"""
        if not target_kernel_events:
            print("    警告: 目标通信kernel events为空")
            return []
        
        target_start_time = min(event.ts for event in target_kernel_events)
        target_kernel_name = target_kernel_events[0].name
        print(f"    目标通信kernel: {target_kernel_name}, 开始时间: {target_start_time:.2f}")
        
        # 查找匹配条件的通信kernel events
        communication_kernels = self._find_events_by_criteria(
            data.events,
            lambda e: (e.cat == 'kernel' and e.ts < target_start_time and
                      re.match(prev_kernel_pattern, e.name) and
                      not any(pattern in e.name for pattern in self.COMMUNICATION_BLACKLIST_PATTERNS))
        )
        
        print(f"    找到 {len(communication_kernels)} 个匹配的通信kernel events")
        
        if communication_kernels:
            # 按结束时间排序，取最后一个
            communication_kernels.sort(key=lambda x: x.ts + x.dur)
            last_kernel_event = communication_kernels[-1]
            
            print(f"    上一个通信kernel: {last_kernel_event.name}")
            print(f"    开始时间: {_format_timestamp_display(last_kernel_event)}")
            print(f"    结束时间: {_calculate_end_time_display(last_kernel_event)}")
            
            return [last_kernel_event]
        else:
            print(f"    警告: 没有找到匹配模式 '{prev_kernel_pattern}' 的上一个通信kernel")
        
        return []
    
    def check_communication_kernel_consistency(self, fastest_data: ProfilerData, 
                                             slowest_data: ProfilerData,
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
        
        # 3. 打印通信事件对比表格
        consistency_check_passed = self._print_communication_events_table(
            fastest_all_comm_events, slowest_all_comm_events
        )
        
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
        
        # 5. 检查结果
            if not consistency_check_passed:
                print("    ✗ 通信kernel一致性检查失败: 存在超过5000us阈值的事件")
                return False
        else:
            print("    ✓ 通信kernel一致性检查通过")
            return True
    
    def compare_events_by_time_sequence(self, fastest_events_by_external_id: Dict, 
                                      slowest_events_by_external_id: Dict,
                                      fastest_card_idx: int, slowest_card_idx: int, 
                                      fastest_duration: float, slowest_duration: float,
                                      fastest_data: ProfilerData = None, 
                                      slowest_data: ProfilerData = None,
                                      show_timestamp: bool = False, 
                                      show_readable_timestamp: bool = False,
                                      output_dir: str = ".", step: int = None, 
                                      comm_idx: int = None) -> Dict[str, Any]:
        """按照时间顺序比较快卡和慢卡的events"""
        from .alignment import EventAlignmentAnalyzer, EventComparisonAnalyzer
        from .visualization import AlignmentVisualizer
        from .base import _readable_timestamp_to_microseconds
        
        comparison_rows = []
        alignment_analyzer = EventAlignmentAnalyzer()
        comparison_analyzer = EventComparisonAnalyzer()
        visualizer = AlignmentVisualizer()
        
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
        aligned_fastest, aligned_slowest, alignment_info = alignment_analyzer.fuzzy_align_cpu_events(
            fastest_cpu_events, slowest_cpu_events
        )
        
        # 3. 生成对齐可视化结果
        print("\n=== 生成对齐可视化 ===")
        visualization_files = visualizer.visualize_alignment_results(
            fastest_cpu_events, slowest_cpu_events,
            aligned_fastest, aligned_slowest, alignment_info,
            output_dir, step, comm_idx
        )
        
        # 5. 检查kernel操作是否一致
        kernel_ops_consistent = self._check_kernel_ops_consistency(fastest_events_by_external_id, slowest_events_by_external_id)
        if not kernel_ops_consistent:
            print("警告: 快卡和慢卡的kernel操作不完全一致，比较结果可能不准确")
        else:
            print("Kernel操作一致性检查通过")
        
        # 5. 使用对齐后的事件进行配对比较
        matched_events_count = len([info for info in alignment_info if info['match_type'] == 'exact_match'])
        
        for i in range(len(aligned_fastest)):
            fastest_cpu_event = aligned_fastest[i]
            slowest_cpu_event = aligned_slowest[i]
            alignment = alignment_info[i]
            
            # 跳过未匹配的事件
            if fastest_cpu_event is None or slowest_cpu_event is None:
                continue
            
            # 找到对应的kernel events
            fastest_kernel_events = self._find_kernel_events_for_cpu_event(fastest_cpu_event, fastest_events_by_external_id)
            slowest_kernel_events = self._find_kernel_events_for_cpu_event(slowest_cpu_event, slowest_events_by_external_id)
            
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
        
        # 对齐质量评估
        print("\n=== 对齐质量评估 ===")
        total_aligned = len(comparison_rows)
        exact_count = sum(1 for row in comparison_rows if row['alignment_type'] == 'exact_match')
        
        print(f"总对齐事件数: {total_aligned}")
        print(f"精确匹配: {exact_count} ({exact_count/total_aligned*100:.1f}%)")
        
        # 显示一些对齐示例
        print("\n=== 对齐示例 ===")
        matched_examples = [row for row in comparison_rows if row['alignment_type'] == 'exact_match'][:5]
        for i, row in enumerate(matched_examples):
            print(f"  {i+1}. 精确匹配: {row['cpu_op_name']} <-> {row['slowest_cpu_op_name']}")
        
        # 分析kernel duration ratio - 找出前5个最大的
        print("\n=== Kernel Duration Ratio 分析 ===")
        top_kernel_duration_ratios = comparison_analyzer.find_top_kernel_duration_ratios(comparison_rows, top_n=5)
        self._print_kernel_duration_analysis(top_kernel_duration_ratios)
        
        # 分析cpu start time相邻差值 - 找出前5对最大的
        print("\n=== CPU Start Time 相邻差值分析 ===")
        top_cpu_start_time_differences = comparison_analyzer.find_top_cpu_start_time_differences(
            comparison_rows, aligned_fastest, aligned_slowest, top_n=5
        )
        
        print(f"找到 {len(top_cpu_start_time_differences)} 对最大的CPU start time相邻差值:")
        self._print_cpu_start_time_differences_analysis(top_cpu_start_time_differences)
        
        return {
            'comparison_rows': comparison_rows,
            'top_kernel_duration_ratios': top_kernel_duration_ratios,
            'top_cpu_start_time_differences': top_cpu_start_time_differences,
            'visualization_files': visualization_files
        }
    
    def _print_kernel_duration_analysis(self, top_kernel_duration_ratios):
        """打印kernel duration分析结果"""
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
    
    def _print_cpu_start_time_differences_analysis(self, top_cpu_start_time_differences):
        """打印CPU start time相邻差值分析结果"""
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
            print()
            
            # 四列竖排格式 - 扩展宽度以容纳长调用栈
            col_width = 50  # 每列宽度从30扩展到50
            total_width = col_width * 4 + 9  # 4列 + 3个分隔符(3个|) + 6个空格
            
            print("     " + "="*total_width)
            print("     " + f"{'最快Card前一个事件':<{col_width}} | {'最慢Card前一个事件':<{col_width}} | {'最快Card当前事件':<{col_width}} | {'最慢Card当前事件':<{col_width}}")
            print("     " + "="*total_width)
            
            # 第一行：事件名称
            prev_fastest_name = prev_row['cpu_op_name']
            prev_slowest_name = prev_row.get('slowest_cpu_op_name', 'N/A')
            current_fastest_name = current_row['cpu_op_name']
            current_slowest_name = current_row.get('slowest_cpu_op_name', 'N/A')
            
            print("     " + f"{prev_fastest_name:<{col_width}} | {prev_slowest_name:<{col_width}} | {current_fastest_name:<{col_width}} |"

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
            prev_name_match = "✅ 匹配" if prev_fastest_name == prev_slowest_name else "❌ 不匹配"
            current_name_match = "✅ 匹配" if current_fastest_name == current_slowest_name else "❌ 不匹配"
            
            print("     " + f"{prev_name_match:<{col_width}} | {'':<{col_width}} | {current_name_match:<{col_width}} | {'':<{col_width}}")
            
            # 调用栈信息
            if 'prev_fastest_event' in diff_info and diff_info['prev_fastest_event']:
                # 这里可以添加调用栈的打印逻辑
                pass
            
            print("     " + "="*total_width)
            print()
    
    def _find_all_communication_events(self, data: ProfilerData) -> List[ActivityEvent]:
        """找到所有TCDP_开头的通信kernel events，按结束时间排序"""
        comm_events = []
        
        for event in data.events:
            if (event.cat == 'kernel' and 
                event.name.startswith('TCDP_')):
                # 检查是否在黑名单中
                is_blacklisted = any(pattern in event.name for pattern in self.COMMUNICATION_BLACKLIST_PATTERNS)
                if not is_blacklisted:
                    comm_events.append(event)
        
        # 按结束时间排序（从早到晚）
        comm_events.sort(key=lambda x: x.ts + x.dur)
        
        return comm_events
    
    def _find_communication_event(self, data, comm_idx, kernel_prefix):
        """找到指定comm_idx的通信kernel操作event"""
        events = self._find_events_by_criteria(
            data.events, 
            lambda e: e.cat == 'kernel' and e.name.startswith(kernel_prefix)
        )
        
        if comm_idx < len(events):
            print(f"    找到目标通信kernel event: {events[comm_idx].name}")
            return events[comm_idx]
        
        return None
    
    def _find_events_by_criteria(self, events, criteria_func):
        """根据条件查找事件"""
        return [event for event in events if criteria_func(event)]
    
    def _find_kernel_events_for_cpu_event(self, cpu_event: ActivityEvent, 
                                        events_by_external_id: Dict):
        """为CPU event找到对应的kernel events"""
        if cpu_event.external_id is None:
            return []
        
        if cpu_event.external_id not in events_by_external_id:
            return []
        
        events = events_by_external_id[cpu_event.external_id]
        kernel_events = [e for e in events if e.cat == 'kernel']
        
        return kernel_events
    
    def _check_kernel_ops_consistency(self, fastest_events_by_external_id: Dict, 
                                     slowest_events_by_external_id: Dict) -> bool:
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
    
    def _print_communication_events_table(self, fastest_events: List[ActivityEvent], 
                                        slowest_events: List[ActivityEvent]) -> bool:
        """打印通信事件对比表格并返回一致性检查结果"""
        print("    通信事件对比表格:")
        print("    | 序号 | 最快卡Kernel名称 | 最快卡开始时间 | 最快卡结束时间 | 最慢卡Kernel名称 | 最慢卡开始时间 | 最慢卡结束时间 | 名称一致 | 开始时间差(ms) | 结束时间差(ms) | 超过阈值 |")
        print("    |------|----------------|-------------|-------------|----------------|-------------|-------------|----------|---------------|---------------|----------|")
        
        max_events = max(len(fastest_events), len(slowest_events))
        consistency_check_passed = True
        
        for i in range(max_events):
            fastest_event = fastest_events[i] if i < len(fastest_events) else None
            slowest_event = slowest_events[i] if i < len(slowest_events) else None
            
            # 基本信息
            fastest_name = fastest_event.name if fastest_event else "N/A"
            slowest_name = slowest_event.name if slowest_event else "N/A"
            name_consistent = "✓" if fastest_name == slowest_name else "✗"
            
            # 时间戳显示
            fastest_start_ts = _format_timestamp_display(fastest_event) if fastest_event else "N/A"
            slowest_start_ts = _format_timestamp_display(slowest_event) if slowest_event else "N/A"
            fastest_end_ts = _calculate_end_time_display(fastest_event) if fastest_event else "N/A"
            slowest_end_ts = _calculate_end_time_display(slowest_event) if slowest_event else "N/A"
            
            # 时间差计算
            start_time_diff, end_time_diff, threshold_exceeded = self._calculate_time_differences(
                fastest_event, slowest_event
            )
            
            if threshold_exceeded == "✗":
                consistency_check_passed = False
            
            print(f"    | {i+1} | {fastest_name} | {fastest_start_ts} | {fastest_end_ts} | {slowest_name} | {slowest_start_ts} | {slowest_end_ts} | {name_consistent} | {start_time_diff} | {end_time_diff} | {threshold_exceeded} |")
        
        return consistency_check_passed
    
    def _calculate_time_differences(self, fastest_event, slowest_event) -> tuple:
        """计算时间差并检查阈值"""
        from .base import _readable_timestamp_to_microseconds
        
        # 开始时间差
        start_diff_us = _calculate_time_diff_readable(fastest_event, slowest_event)
        start_time_diff = f"{start_diff_us / 1000.0:.3f}"
        
        # 结束时间差
        fastest_start_us = _readable_timestamp_to_microseconds(fastest_event.readable_timestamp)
        slowest_start_us = _readable_timestamp_to_microseconds(slowest_event.readable_timestamp)
        fastest_end_us = fastest_start_us + fastest_event.dur
        slowest_end_us = slowest_start_us + slowest_event.dur
        end_diff_ms = abs(fastest_end_us - slowest_end_us) / 1000.0
        end_time_diff = f"{end_diff_ms:.3f}"
        
        # 阈值检查
        threshold_exceeded = "✗" if end_diff_ms > 5.0 else "✓"
        
        return start_time_diff, end_time_diff, threshold_exceeded

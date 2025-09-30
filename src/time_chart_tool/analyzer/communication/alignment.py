"""
LCS事件对齐模块
"""

from typing import List, Tuple, Dict, Any
import re
from ..stages.postprocessor import DataPostProcessor

from ....models import ActivityEvent
from ..base import _readable_timestamp_to_microseconds

class EventAlignmentAnalyzer:
    """事件对齐分析器"""
    
    def __init__(self):
        pass
    
    def fuzzy_align_cpu_events(self, fastest_cpu_events: List[ActivityEvent], 
                              slowest_cpu_events: List[ActivityEvent]):
        """
        基于LCS的模糊对齐CPU事件
        使用操作名称 + call stack 进行匹配，提高对齐精度
        
        Args:
            fastest_cpu_events: 最快卡的CPU事件列表
            slowest_cpu_events: 最慢卡的CPU事件列表
            
        Returns:
            tuple: (aligned_fastest, aligned_slowest, alignment_info)
        """
        if not fastest_cpu_events or not slowest_cpu_events:
            return [], [], []
        
        print(f"开始LCS模糊对齐: 快卡{len(fastest_cpu_events)}个事件, 慢卡{len(slowest_cpu_events)}个事件")
        
        # 1. 构建操作名称 + call stack 序列
        fast_signatures = [self._build_event_signature(e) for e in fastest_cpu_events]
        slow_signatures = [self._build_event_signature(e) for e in slowest_cpu_events]
        
        # 2. 计算LCS矩阵
        lcs_matrix = self._compute_lcs_matrix(fast_signatures, slow_signatures)
        
        # 3. 回溯LCS路径，构建对齐方案
        alignment = self._backtrack_lcs(fast_signatures, slow_signatures, lcs_matrix)
        
        # 4. 构建最终对齐结果
        aligned_fastest, aligned_slowest, alignment_info = self._build_aligned_events(
            alignment, fastest_cpu_events, slowest_cpu_events
        )
        
        # 5. 检查对齐结果中相同位置的事件名称是否一致
        print("检查对齐结果中事件名称一致性...")
        name_mismatches = 0
        for i, (fast_event, slow_event) in enumerate(zip(aligned_fastest, aligned_slowest)):
            if fast_event is not None and slow_event is not None:
                if fast_event.name != slow_event.name:
                    print(f"[警告] fuzzy_align_cpu_events: 位置 {i} 事件名称不匹配: '{fast_event.name}' vs '{slow_event.name}'")
                    name_mismatches += 1
        
        if name_mismatches == 0:
            print("✅ 对齐结果中所有事件名称都匹配")
        else:
            print(f"❌ 发现 {name_mismatches} 个事件名称不匹配")
        
        # 6. 统计对齐质量
        exact_matches = sum(1 for info in alignment_info if info['match_type'] == 'exact_match')
        unmatched_fast = sum(1 for info in alignment_info if info['match_type'] == 'unmatched_fast')
        unmatched_slow = sum(1 for info in alignment_info if info['match_type'] == 'unmatched_slow')
        
        print(f"LCS对齐完成: 精确匹配 {exact_matches} 个, 快卡未匹配 {unmatched_fast} 个, 慢卡未匹配 {unmatched_slow} 个")
        print(f"匹配率: {exact_matches / len(alignment_info) * 100:.1f}%")
        
        return aligned_fastest, aligned_slowest, alignment_info
    
    def _build_event_signature(self, event: ActivityEvent) -> str:
        """
        构建事件的签名，包含操作名称和过滤后的call stack
        
        Args:
            event: CPU事件对象
            
        Returns:
            str: 事件签名
        """
        # 获取操作名称
        name = event.name if event.name else ""
        return name
        
        # 获取call stack
        call_stack = event.get_call_stack("tree")
        if not call_stack:
            return name
        
        # 过滤call stack，去掉地址信息
        filtered_call_stack = self._filter_call_stack(call_stack)
        
        # 构建签名：操作名称 + 过滤后的call stack
        signature = name
        if filtered_call_stack:
            # 将call stack转换为字符串，用特殊分隔符连接
            call_stack_str = " | ".join(filtered_call_stack)
            signature = f"{name} [{call_stack_str}]"
        
        return signature
    
    def _filter_call_stack(self, call_stack: List[str]) -> List[str]:
        """
        过滤call stack，去掉地址信息
        
        Args:
            call_stack: call stack列表
            
        Returns:
            list: 过滤后的call stack
        """
        if not call_stack:
            return []
        
        filtered = []
        
        for frame in call_stack:
            if not frame:
                continue
            
            # 去掉 "object at 0x*****" 这种形式的地址信息
            # 匹配模式：object at 0x后跟十六进制数字
            filtered_frame = re.sub(r'object at 0x[0-9a-fA-F]+', 'object', frame)
            
            # 去掉其他可能的地址信息
            # 匹配模式：0x后跟十六进制数字
            filtered_frame = re.sub(r'0x[0-9a-fA-F]+', '0x****', filtered_frame)
            
            # 去掉行号信息（可选，根据需要调整）
            # filtered_frame = re.sub(r':\d+\)', ')', filtered_frame)
            
            filtered.append(filtered_frame)
        
        return filtered
    
    def _compute_lcs_matrix(self, seq1: List[str], seq2: List[str]) -> List[List[int]]:
        """计算LCS矩阵"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:  # 事件签名完全相等
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp
    
    def _backtrack_lcs(self, fast_signatures: List[str], slow_signatures: List[str], 
                      lcs_matrix: List[List[int]]) -> List[Tuple[int, int, str]]:
        """回溯LCS路径，构建对齐方案"""
        alignment = []
        i, j = len(fast_signatures), len(slow_signatures)
        
        while i > 0 and j > 0:
            if fast_signatures[i-1] == slow_signatures[j-1]:
                # 精确匹配
                alignment.append((i-1, j-1, 'exact_match'))
                i -= 1
                j -= 1
            elif lcs_matrix[i-1][j] > lcs_matrix[i][j-1]:
                # 快卡事件未匹配
                alignment.append((i-1, None, 'unmatched_fast'))
                i -= 1
            else:
                # 慢卡事件未匹配
                alignment.append((None, j-1, 'unmatched_slow'))
                j -= 1
        
        # 处理剩余事件
        while i > 0:
            alignment.append((i-1, None, 'unmatched_fast'))
            i -= 1
        while j > 0:
            alignment.append((None, j-1, 'unmatched_slow'))
            j -= 1
        
        return alignment[::-1]  # 反转得到正确顺序
    
    def _build_aligned_events(self, alignment: List[Tuple[int, int, str]], 
                              fastest_events: List[ActivityEvent], 
                              slowest_events: List[ActivityEvent]):
        """构建对齐后的事件序列"""
        aligned_fastest = []
        aligned_slowest = []
        alignment_info = []
        
        for fast_idx, slow_idx, match_type in alignment:
            if fast_idx is not None and slow_idx is not None:
                # 匹配成功 - 检查事件名称是否一致
                fast_event = fastest_events[fast_idx]
                slow_event = slowest_events[slow_idx]
                
                if fast_event.name != slow_event.name:
                    print(f"[警告] _build_aligned_events: exact_match 事件名称不匹配: '{fast_event.name}' vs '{slow_event.name}'")
                    # 将不匹配的事件标记为 unmatched
                    aligned_fastest.append(fast_event)
                    aligned_slowest.append(None)
                    alignment_info.append({
                        'match_type': 'unmatched_fast',
                        'fast_idx': fast_idx,
                        'slow_idx': None,
                        'fast_name': fast_event.name,
                        'slow_name': None
                    })
                    
                    aligned_fastest.append(None)
                    aligned_slowest.append(slow_event)
                    alignment_info.append({
                        'match_type': 'unmatched_slow',
                        'fast_idx': None,
                        'slow_idx': slow_idx,
                        'fast_name': None,
                        'slow_name': slow_event.name
                    })
                else:
                    aligned_fastest.append(fast_event)
                    aligned_slowest.append(slow_event)
                    alignment_info.append({
                        'match_type': 'exact_match',
                        'fast_idx': fast_idx,
                        'slow_idx': slow_idx,
                        'fast_name': fast_event.name,
                        'slow_name': slow_event.name
                    })
            elif fast_idx is not None:
                # 快卡事件未匹配
                aligned_fastest.append(fastest_events[fast_idx])
                aligned_slowest.append(None)
                alignment_info.append({
                    'match_type': 'unmatched_fast',
                    'fast_idx': fast_idx,
                    'slow_idx': None,
                    'fast_name': fastest_events[fast_idx].name,
                    'slow_name': None
                })
            else:
                # 慢卡事件未匹配
                aligned_fastest.append(None)
                aligned_slowest.append(slowest_events[slow_idx])
                alignment_info.append({
                    'match_type': 'unmatched_slow',
                    'fast_idx': None,
                    'slow_idx': slow_idx,
                    'fast_name': None,
                    'slow_name': slowest_events[slow_idx].name
                })
        
        return aligned_fastest, aligned_slowest, alignment_info


class EventComparisonAnalyzer:
    """事件对比分析器"""
    
    def __init__(self):
        self.postprocessor = DataPostProcessor()
    
    def find_top_cpu_start_time_differences(self, comparison_rows: List[Dict], 
                                          aligned_fastest: List[ActivityEvent] = None, 
                                          aligned_slowest: List[ActivityEvent] = None, 
                                          top_n: int = 5) -> List[Dict]:
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
                'current_slowest_event': current_slowest_event
            })
        
        # 按差值降序排序
        differences.sort(key=lambda x: x['difference'], reverse=True)
        
        # 返回前N对
        return differences[:top_n]
    
    def find_top_kernel_duration_ratios(self, comparison_rows: List[Dict], top_n<｜tool▁sep｜>5):
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

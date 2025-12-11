import re
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class EventAligner:
    """
    事件对齐工具类
    """
    
    def fuzzy_align_cpu_events(self, fastest_cpu_events, slowest_cpu_events):
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
    
    def _build_event_signature(self, event):
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
        call_stack = event.call_stack
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
    
    def _filter_call_stack(self, call_stack):
        """
        过滤call stack，去掉地址信息
        
        Args:
            call_stack: call stack列表
            
        Returns:
            list: 过滤后的call stack
        """
        if not call_stack:
            return []
        
        import re
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
    
    def _compute_lcs_matrix(self, seq1, seq2):
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
    
    def _backtrack_lcs(self, fast_signatures, slow_signatures, lcs_matrix):
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
    
    def _build_aligned_events(self, alignment, fastest_events, slowest_events):
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
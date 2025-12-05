import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    
    def visualize_alignment_results(self, fastest_cpu_events, slowest_cpu_events, 
                                   aligned_fastest, aligned_slowest, alignment_info,
                                   output_dir: str = ".", step: int = None, comm_idx: int = None):
        """
        可视化对齐结果
        
        Args:
            fastest_cpu_events: 原始最快卡CPU事件列表
            slowest_cpu_events: 原始最慢卡CPU事件列表
            aligned_fastest: 对齐后的最快卡事件列表
            aligned_slowest: 对齐后的最慢卡事件列表
            alignment_info: 对齐信息列表
            output_dir: 输出目录
            step: 步骤编号
            comm_idx: 通信索引
        """
        print("=== 生成对齐可视化图表 ===")
        
        # 设置英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 生成LCS对齐矩阵图
        # lcs_matrix_file = self._create_lcs_matrix_visualization(
        #     fastest_cpu_events, slowest_cpu_events, alignment_info, output_path, step, comm_idx
        # )
        lcs_matrix_file = 'None'
        
        # 2. 生成时间线对齐图（文本版本）
        timeline_file = self._create_timeline_alignment_text_visualization(
            aligned_fastest, aligned_slowest, alignment_info, output_path, step, comm_idx
        )
        
        # 3. 生成对齐统计图
        stats_file = self._create_alignment_statistics_visualization(
            alignment_info, output_path, step, comm_idx
        )
        
        print(f"对齐可视化文件已生成:")
        print(f"  - LCS矩阵图: {lcs_matrix_file}")
        print(f"  - 时间线对齐文本: {timeline_file}")
        print(f"  - 对齐统计图: {stats_file}")
        
        return [lcs_matrix_file, timeline_file, stats_file]
    
    def _create_lcs_matrix_visualization(self, fastest_cpu_events, slowest_cpu_events, 
                                       alignment_info, output_path, step, comm_idx):
        """创建LCS对齐矩阵可视化"""
        # 构建事件签名序列
        fast_signatures = [self._build_event_signature(e) for e in fastest_cpu_events]
        slow_signatures = [self._build_event_signature(e) for e in slowest_cpu_events]
        
        # 计算LCS矩阵
        lcs_matrix = self._compute_lcs_matrix(fast_signatures, slow_signatures)
        
        # 创建图形 - 根据矩阵大小调整
        matrix_size = (len(fast_signatures) + 1) * (len(slow_signatures) + 1)
        if matrix_size > 1000:
            # 大矩阵使用较小的图形和字体
            fig, ax = plt.subplots(figsize=(max(8, len(slow_signatures) * 0.4), max(6, len(fast_signatures) * 0.3)))
        else:
            fig, ax = plt.subplots(figsize=(max(12, len(slow_signatures) * 0.8), max(8, len(fast_signatures) * 0.6)))
        
        # 绘制矩阵
        matrix_array = np.array(lcs_matrix)
        im = ax.imshow(matrix_array, cmap='Blues', aspect='auto')
        
        # 设置坐标轴标签 - 根据矩阵大小调整
        if matrix_size > 1000:
            # 大矩阵简化标签
            ax.set_xticks(range(0, len(slow_signatures) + 1, max(1, len(slow_signatures) // 10)))
            ax.set_yticks(range(0, len(fast_signatures) + 1, max(1, len(fast_signatures) // 10)))
            ax.set_xticklabels([f'{i}' for i in range(0, len(slow_signatures) + 1, max(1, len(slow_signatures) // 10))])
            ax.set_yticklabels([f'{i}' for i in range(0, len(fast_signatures) + 1, max(1, len(fast_signatures) // 10))])
        else:
            ax.set_xticks(range(len(slow_signatures) + 1))
            ax.set_yticks(range(len(fast_signatures) + 1))
            ax.set_xticklabels([''] + [sig[:20] + '...' if len(sig) > 20 else sig for sig in slow_signatures], 
                              rotation=45, ha='right')
            ax.set_yticklabels([''] + [sig[:20] + '...' if len(sig) > 20 else sig for sig in fast_signatures])
        
        # 在矩阵中显示数值 - 优化版本
        self._add_matrix_text_optimized(ax, lcs_matrix, matrix_array)
        
        # 高亮对齐路径
        self._highlight_alignment_path(ax, alignment_info, fast_signatures, slow_signatures)
        
        # 设置标题和标签
        ax.set_title(f'LCS Alignment Matrix (Step {step}, Comm {comm_idx})\n'
                    f'Fastest Card: {len(fast_signatures)} events, Slowest Card: {len(slow_signatures)} events', 
                    fontsize=14, pad=20)
        ax.set_xlabel('Slowest Card CPU Event Sequence', fontsize=12)
        ax.set_ylabel('Fastest Card CPU Event Sequence', fontsize=12)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('LCS Length', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存文件
        filename = f"lcs_alignment_matrix_step_{step}_comm_{comm_idx}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _add_matrix_text_optimized(self, ax, lcs_matrix, matrix_array):
        """优化的矩阵文本添加方法"""
        rows, cols = matrix_array.shape
        
        # 如果矩阵太大，跳过文本显示以提高性能
        if rows * cols > 500:  # 超过500个元素时跳过文本显示
            return
        
        # 计算阈值（只计算一次）
        max_val = matrix_array.max()
        threshold = max_val * 0.5
        
        # 创建颜色矩阵
        color_matrix = np.where(matrix_array < threshold, 'black', 'white')
        
        # 批量添加文本
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, str(lcs_matrix[i][j]), ha="center", va="center", 
                       color=color_matrix[i, j], fontsize=8)
    
    def _highlight_alignment_path(self, ax, alignment_info, fast_signatures, slow_signatures):
        """在LCS矩阵中高亮对齐路径"""
        for i, info in enumerate(alignment_info):
            if info['match_type'] == 'exact_match':
                # 精确匹配用绿色圆圈标记
                fast_idx = info['fast_idx'] + 1  # +1 因为矩阵有0行/列
                slow_idx = info['slow_idx'] + 1
                circle = patches.Circle((slow_idx, fast_idx), 0.3, 
                                      facecolor='green', edgecolor='darkgreen', alpha=0.7)
                ax.add_patch(circle)
            elif info['match_type'] == 'unmatched_fast':
                # 快卡未匹配用红色三角形标记
                fast_idx = info['fast_idx'] + 1
                triangle = patches.RegularPolygon((0, fast_idx), 3, radius=0.3,
                                                facecolor='red', edgecolor='darkred', alpha=0.7)
                ax.add_patch(triangle)
            elif info['match_type'] == 'unmatched_slow':
                # 慢卡未匹配用橙色三角形标记
                slow_idx = info['slow_idx'] + 1
                triangle = patches.RegularPolygon((slow_idx, 0), 3, radius=0.3,
                                                facecolor='orange', edgecolor='darkorange', alpha=0.7)
                ax.add_patch(triangle)
    
    def _create_timeline_alignment_text_visualization(self, aligned_fastest, aligned_slowest, 
                                                    alignment_info, output_path, step, comm_idx):
        """创建时间线对齐文本可视化"""
        print("=== 生成时间线对齐文本可视化 ===")
        
        # 创建文本文件
        filename = f"timeline_alignment_step_{step}_comm_{comm_idx}.txt"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"CPU事件对齐时间线 (Step {step}, Comm {comm_idx})\n")
            f.write("=" * 80 + "\n\n")
            
            # 统计信息
            total_events = len(alignment_info)
            exact_matches = sum(1 for info in alignment_info if info['match_type'] == 'exact_match')
            unmatched_fast = sum(1 for info in alignment_info if info['match_type'] == 'unmatched_fast')
            unmatched_slow = sum(1 for info in alignment_info if info['match_type'] == 'unmatched_slow')
            
            f.write(f"对齐统计:\n")
            f.write(f"  总事件数: {total_events}\n")
            f.write(f"  精确匹配: {exact_matches} ({exact_matches/total_events*100:.1f}%)\n")
            f.write(f"  快卡未匹配: {unmatched_fast} ({unmatched_fast/total_events*100:.1f}%)\n")
            f.write(f"  慢卡未匹配: {unmatched_slow} ({unmatched_slow/total_events*100:.1f}%)\n\n")
            
            # 符号说明
            f.write("符号说明:\n")
            f.write("  ✓ 精确匹配\n")
            f.write("  ✗ 快卡未匹配\n")
            f.write("  ○ 慢卡未匹配\n\n")
            
            # 时间线对齐图
            f.write("时间线对齐图:\n")
            f.write("-" * 80 + "\n")
            
            # 计算时间范围用于缩放
            all_times = []
            for fast_event, slow_event in zip(aligned_fastest, aligned_slowest):
                if fast_event:
                    all_times.extend([fast_event.ts, fast_event.ts + fast_event.dur])
                if slow_event:
                    all_times.extend([slow_event.ts, slow_event.ts + slow_event.dur])
            
            if all_times:
                min_time = min(all_times)
                max_time = max(all_times)
                time_range = max_time - min_time
                scale_factor = 60.0 / time_range if time_range > 0 else 1.0
            else:
                min_time = 0
                scale_factor = 1.0
            
            # 生成时间线
            for i, (fast_event, slow_event, info) in enumerate(zip(aligned_fastest, aligned_slowest, alignment_info)):
                # 选择符号
                if info['match_type'] == 'exact_match':
                    symbol = '✓'
                elif info['match_type'] == 'unmatched_fast':
                    symbol = '✗'
                else:
                    symbol = '○'
                
                # 最快卡信息
                if fast_event:
                    fast_start = (fast_event.ts - min_time) * scale_factor
                    fast_duration = fast_event.dur * scale_factor
                    fast_name = fast_event.name[:40] + '...' if len(fast_event.name) > 40 else fast_event.name
                    fast_bar = '█' * max(1, int(fast_duration))
                    f.write(f"最快卡 {i+1:3d}: {symbol} {fast_bar:<20} | {fast_name}\n")
                else:
                    f.write(f"最快卡 {i+1:3d}: {symbol} {'':<20} | [未匹配]\n")
                
                # 最慢卡信息
                if slow_event:
                    slow_start = (slow_event.ts - min_time) * scale_factor
                    slow_duration = slow_event.dur * scale_factor
                    slow_name = slow_event.name[:40] + '...' if len(slow_event.name) > 40 else slow_event.name
                    slow_bar = '█' * max(1, int(slow_duration))
                    f.write(f"最慢卡 {i+1:3d}: {symbol} {slow_bar:<20} | {slow_name}\n")
                else:
                    f.write(f"最慢卡 {i+1:3d}: {symbol} {'':<20} | [未匹配]\n")
                
                f.write("\n")
            
            # 添加详细的对齐信息
            f.write("\n详细对齐信息:\n")
            f.write("-" * 80 + "\n")
            
            for i, (fast_event, slow_event, info) in enumerate(zip(aligned_fastest, aligned_slowest, alignment_info)):
                f.write(f"事件 {i+1}: {info['match_type']}\n")
                
                if fast_event:
                    f.write(f"  最快卡: {fast_event.name}\n")
                    f.write(f"    时间: {fast_event.ts:.2f} - {fast_event.ts + fast_event.dur:.2f} (持续: {fast_event.dur:.2f}μs)\n")
                else:
                    f.write(f"  最快卡: [未匹配]\n")
                
                if slow_event:
                    f.write(f"  最慢卡: {slow_event.name}\n")
                    f.write(f"    时间: {slow_event.ts:.2f} - {slow_event.ts + slow_event.dur:.2f} (持续: {slow_event.dur:.2f}μs)\n")
                else:
                    f.write(f"  最慢卡: [未匹配]\n")
                
                f.write("\n")
        
        print(f"时间线对齐文本文件已生成: {filepath}")
        return filepath
    
    def _create_alignment_statistics_visualization(self, alignment_info, output_path, step, comm_idx):
        """创建对齐统计可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 对齐类型分布饼图
        match_types = [info['match_type'] for info in alignment_info]
        type_counts = {
            'exact_match': match_types.count('exact_match'),
            'unmatched_fast': match_types.count('unmatched_fast'),
            'unmatched_slow': match_types.count('unmatched_slow')
        }
        
        labels = ['Exact Match', 'Unmatched Fast', 'Unmatched Slow']
        sizes = [type_counts['exact_match'], type_counts['unmatched_fast'], type_counts['unmatched_slow']]
        colors = ['green', 'red', 'orange']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Alignment Type Distribution')
        
        # 2. 对齐质量指标
        total_events = len(alignment_info)
        exact_matches = type_counts['exact_match']
        match_rate = exact_matches / total_events * 100 if total_events > 0 else 0
        
        metrics = ['Total Events', 'Exact Matches', 'Match Rate(%)']
        values = [total_events, exact_matches, match_rate]
        
        bars = ax2.bar(metrics, values, color=['blue', 'green', 'orange'])
        ax2.set_title('Alignment Quality Metrics')
        ax2.set_ylabel('Count/Percentage')
        
        # 在柱状图上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 3. 连续匹配长度分布
        consecutive_lengths = self._calculate_consecutive_match_lengths(alignment_info)
        if consecutive_lengths:
            ax3.hist(consecutive_lengths, bins=min(10, len(set(consecutive_lengths))), 
                    color='skyblue', alpha=0.7, edgecolor='black')
            ax3.set_title('Consecutive Match Length Distribution')
            ax3.set_xlabel('Consecutive Match Length')
            ax3.set_ylabel('Frequency')
        else:
            ax3.text(0.5, 0.5, 'No Consecutive Matches', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Consecutive Match Length Distribution')
        
        # 4. 对齐位置分布
        positions = list(range(len(alignment_info)))
        match_status = [1 if info['match_type'] == 'exact_match' else 0 for info in alignment_info]
        
        ax4.scatter(positions, match_status, c=match_status, cmap='RdYlGn', alpha=0.7)
        ax4.set_title('Alignment Position Distribution')
        ax4.set_xlabel('Event Position')
        ax4.set_ylabel('Match Status (1=Match, 0=Unmatched)')
        ax4.set_ylim(-0.1, 1.1)
        ax4.grid(True, alpha=0.3)
        
        # 设置总标题
        fig.suptitle(f'Alignment Statistics Analysis (Step {step}, Comm {comm_idx})', fontsize=16, y=0.95)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存文件
        filename = f"alignment_statistics_step_{step}_comm_{comm_idx}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _calculate_consecutive_match_lengths(self, alignment_info):
        """计算连续精确匹配的长度"""
        consecutive_lengths = []
        current_length = 0
        
        for info in alignment_info:
            if info['match_type'] == 'exact_match':
                current_length += 1
            else:
                if current_length > 0:
                    consecutive_lengths.append(current_length)
                    current_length = 0
        
        # 处理最后一个连续序列
        if current_length > 0:
            consecutive_lengths.append(current_length)
        
        return consecutive_lengths

"""
可视化模块
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import List, Tuple, Any

from ....models import ActivityEvent


class AlignmentVisualizer:
    """对齐结果可视化器"""
    
    def __init__(self):
        # 设置英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    
    def visualize_alignment_results(self, fastest_cpu_events: List[ActivityEvent], 
                                   slowest_cpu_events: List[ActivityEvent],
                                   aligned_fastest: List[ActivityEvent], 
                                   aligned_slowest: List[ActivityEvent], 
                                   alignment_info: List[Dict],
                                   output_dir: str = ".", 
                                   step: int = None, 
                                   comm_idx: int = None) -> List[Path]:
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
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 生成时间线对齐图（文本版本）
        timeline_file = self._create_timeline_alignment_text_visualization(
            aligned_fastest, aligned_slowest, alignment_info, output_path, step, comm_idx
        )
        
        # 2. 生成对齐统计图
        stats_file = self._create_alignment_statistics_visualization(
            alignment_info, output_path, step, comm_idx
        )
        
        print(f"对齐可视化文件已生成:")
        print(f"  - 时间线对齐文本: {timeline_file}")
        print(f"  - 对齐统计图: {stats_file}")
        
        return [str(timeline_file), str(stats_file)]
    
    def _create_timeline_alignment_text_visualization(self, aligned_fastest: List[ActivityEvent], 
                                                    aligned_slowest: List[ActivityEvent], 
                                                    alignment_info: List[Dict], 
                                                    output_path: Path, step: int, comm_idx: int) -> Path:
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
    
    def _create_alignment_statistics_visualization(self, alignment_info: List[Dict], 
                                                  output_path: Path, 
                                                  step: int, 
                                                  comm_idx: int) -> Path:
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
    
    def _calculate_consecutive_match_lengths(self, alignment_info: List[Dict]) -> List[int]:
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



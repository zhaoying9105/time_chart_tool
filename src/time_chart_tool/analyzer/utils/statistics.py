"""
统计相关工具
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ...models import ActivityEvent
from collections import defaultdict
import statistics

@dataclass
class KernelStatistics:
    """Kernel 统计信息"""
    kernel_name: str
    min_duration: float
    max_duration: float
    mean_duration: float
    variance: float
    count: int
    
    def __str__(self):
        return f"KernelStatistics({self.kernel_name}, count={self.count}, mean={self.mean_duration:.3f}, std={self.variance**0.5:.3f})"

def calculate_kernel_statistics(kernel_events: List[ActivityEvent]) -> List[KernelStatistics]:
    
    if not kernel_events:
        return []
    
    # 按 kernel name 分组
    kernel_groups = defaultdict(list)
    for event in kernel_events:
        kernel_groups[event.name].append(event)
    
    statistics_list = []
    
    for kernel_name, events in kernel_groups.items():
        if len(events) == 0:
            continue
            
        # 计算持续时间统计
        durations = [event.dur for event in events if event.dur is not None]
        
        if not durations:
            continue
            
        mean_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        # 计算方差
        if len(durations) > 1:
            variance = statistics.variance(durations)
        else:
            variance = 0.0
        
        stats = KernelStatistics(
            kernel_name=kernel_name,
            min_duration=min_duration,
            max_duration=max_duration,
            mean_duration=mean_duration,
            variance=variance,
            count=len(events)
        )
        
        statistics_list.append(stats)
    
    return statistics_list
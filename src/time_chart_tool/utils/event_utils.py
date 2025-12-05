"""
调用栈处理工具模块
"""

from typing import List, Dict, Any, Optional
import logging

from ..models import ActivityEvent

logger = logging.getLogger(__name__)

def normalize_timestamps(events: List[ActivityEvent]) -> List[ActivityEvent]:
    """
    标准化时间戳：找到最小ts值，将所有事件的ts减去这个值
    
    Args:
        events: 事件列表
        
    Returns:
        List[ActivityEvent]: 标准化时间戳后的事件列表（原地修改并返回）
    """
    if not events:
        return events
    
    # 找到最小时间戳
    valid_ts = [event.ts for event in events if event.ts is not None]
    if not valid_ts:
        logger.warning("没有找到有效的时间戳，跳过标准化")
        return events
    
    min_ts = min(valid_ts)
    print(f"找到最小时间戳: {min_ts:.2f} 微秒，开始标准化...")
    
    # 标准化所有事件的时间戳
    for event in events:
        if event.ts is not None:
            event.ts = event.ts - min_ts
    
    print("时间戳标准化完成")
    return events


def remove_triton_suffix(name: str) -> str:
    """
    去除triton名称中的suffix部分
    
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


def create_event_id(event: ActivityEvent) -> str:
    """
    创建事件ID用于查找
    
    Args:
        event: ActivityEvent对象
        
    Returns:
        str: 事件ID
    """
    # 对triton事件应用相同的名称处理逻辑
    processed_name = remove_triton_suffix(event.name)
    return f"{processed_name}:{event.ts}:{event.dur}:{event.pid}:{event.tid}"

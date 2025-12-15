"""
时间工具函数（顶层utils，避免解析器与analyzer之间的循环依赖）
"""

from datetime import datetime
from typing import Optional,List
from ..models import ActivityEvent
import logging

logger = logging.getLogger(__name__)


def _nanoseconds_to_readable_timestamp(nanoseconds: int) -> str:
    """
    将纳秒时间戳转换为可读的时间格式
    
    Args:
        nanoseconds: 纳秒时间戳
        
    Returns:
        str: 格式化的时间字符串 (YYYY-MM-DD HH:MM:SS.ffffff)
    """
    try:
        # 将纳秒转换为秒（浮点数）
        seconds = nanoseconds / 1_000_000_000.0
        
        # 创建datetime对象
        dt = datetime.fromtimestamp(seconds)
        
        # 格式化为字符串，包含微秒
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        logger.warning(f"时间戳转换失败: {e}")
        return f"Invalid timestamp: {nanoseconds}"


def calculate_readable_timestamps(events: List[ActivityEvent], base_time_nanoseconds: Optional[int]) -> List[ActivityEvent]:
    """
    计算事件的可读时间戳（纯函数）
    
    Args:
        events: 事件列表
        base_time_nanoseconds: 基准时间（纳秒）
        
    Returns:
        List[ActivityEvent]: 带有可读时间戳的事件列表（新对象）
    """
    print("===  计算可读时间戳 ===")
    
    processed_events = []
    for event in events:
        
        # ts是微秒，base_time_nanoseconds是纳秒
        # 将ts转换为纳秒，然后加上基准时间
        total_nanoseconds = base_time_nanoseconds + int(event.ts * 1000)  # ts * 1000 将微秒转换为纳秒
        event.readable_timestamp = _nanoseconds_to_readable_timestamp(total_nanoseconds)
        processed_events.append(event)
            
    return processed_events
def readable_timestamp_to_microseconds(readable_timestamp: str) -> float:
    try:
        dt = datetime.strptime(readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp() * 1_000_000
    except Exception as e:
        logger.warning(f"时间戳转换失败: {e}")
        return 0.0


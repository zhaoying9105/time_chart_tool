"""
时间工具函数
"""

from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def nanoseconds_to_readable_timestamp(nanoseconds: int) -> str:
    """
    将纳秒时间戳转换为可读的时间字符串 (YYYY-MM-DD HH:MM:SS.ffffff)
    """
    try:
        seconds = nanoseconds / 1_000_000_000.0
        dt = datetime.fromtimestamp(seconds)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        logger.warning(f"时间戳转换失败: {e}")
        return f"Invalid timestamp: {nanoseconds}"


def readable_timestamp_to_microseconds(readable_timestamp: str) -> float:
    """
    将可读时间字符串转换为微秒时间戳
    """
    try:
        dt = datetime.strptime(readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp() * 1_000_000
    except Exception as e:
        logger.warning(f"时间戳转换失败: {e}")
        return 0.0


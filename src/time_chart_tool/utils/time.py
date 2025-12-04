"""
时间工具函数（顶层utils，避免解析器与analyzer之间的循环依赖）
"""

from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def nanoseconds_to_readable_timestamp(nanoseconds: int) -> str:
    try:
        seconds = nanoseconds / 1_000_000_000.0
        dt = datetime.fromtimestamp(seconds)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        logger.warning(f"时间戳转换失败: {e}")
        return f"Invalid timestamp: {nanoseconds}"


def readable_timestamp_to_microseconds(readable_timestamp: str) -> float:
    try:
        dt = datetime.strptime(readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp() * 1_000_000
    except Exception as e:
        logger.warning(f"时间戳转换失败: {e}")
        return 0.0


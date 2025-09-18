"""
数据结构定义
"""

from dataclasses import dataclass
from typing import List, Union
from ...models import ActivityEvent


@dataclass
class AggregatedData:
    """聚合后的数据结构"""
    cpu_events: List[ActivityEvent]
    kernel_events: List[ActivityEvent]
    key: Union[str, tuple]  # 聚合键（op_name, op_shape, call_stack等）

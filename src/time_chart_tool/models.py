# -*- coding: utf-8 -*-
"""
PyTorch Profiler 数据模型定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union


@dataclass
class ActivityEvent:
    """活动事件数据模型"""
    name: str
    cat: str
    ph: str
    pid: int
    tid: int
    ts: float
    dur: Optional[float] = None
    args: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    stream_id: Optional[int] = None
    readable_timestamp: Optional[str] = None
    fwd_bwd_type = 'none'
    call_stack_from_tree: Optional[List[str]] = None
    
    def __post_init__(self):
        
        if self.stream_id is None and self.args:
            self.stream_id = self.args.get('stream', None)
    
    @property
    def external_id(self) -> Optional[Union[int, str]]:
        """获取 External id"""
        return self.args.get('External id', None) if self.args else None
    
    @property
    def correlation_id(self) -> Optional[Union[int, str]]:
        """获取 correlation id"""
        return self.args.get('correlation id', None) if self.args else None
    
    @property
    def correlation(self) -> Optional[Union[int, str]]:
        """获取 correlation"""
        return self.args.get('correlation', None) if self.args else None
    
    @property
    def ev_idx(self) -> Optional[Union[int, str]]:
        """获取 Ev Idx"""
        return self.args.get('Ev Idx', None) if self.args else None
    
    @property
    def python_id(self) -> Optional[Union[int, str]]:
        """获取 Python id"""
        return self.args.get('Python id', None) if self.args else None
    
    @property
    def python_parent_id(self) -> Optional[Union[int, str]]:
        """获取 Python parent id"""
        return self.args.get('Python parent id', None) if self.args else None
    
    @property
    def fwd_thread_id(self) -> Optional[Union[int, str]]:
        """获取 Fwd thread id"""
        return self.args.get('Fwd thread id', None) if self.args else None
    
    @property
    def is_kernel(self) -> bool:
        """判断是否为 kernel 事件"""
        return self.cat == 'kernel' or 'kernel' in self.name.lower()

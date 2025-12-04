"""
PyTorch Profiler 数据模型定义
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


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
    args: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    stream_id: Optional[int] = None
    readable_timestamp: Optional[str] = None
    fwd_bwd_type: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.args is None:
            self.args = {}
        
        # 从 args 中提取 stream_id
        if self.stream_id is None and self.args:
            self.stream_id = self.args.get('stream', None)
        
        # 初始化 fwd_bwd_type 为 None，稍后在 CallStackBuilder 执行后设置
        self.fwd_bwd_type = None
        
        # 如果 args 中没有 Call stack，直接设置为 none
        if self.cat == 'cpu_op' and (not self.args or 'Call stack' not in self.args):
            self.fwd_bwd_type = 'none'
    
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
    def call_stack(self) -> Optional[List[str]]:
        """获取 call stack 信息"""
        if not self.args:
            return None
        call_stack_str = self.args.get('Call stack', None)
        if call_stack_str is None:
            return None
        # 将字符串按分号分割成列表
        call_stack_list = [frame.strip() for frame in call_stack_str.split(';') if frame.strip()]
        
        # 如果 fwd_bwd_type 还没有设置，现在设置它
        if self.fwd_bwd_type is None:
            self._set_fwd_bwd_type_from_call_stack()
        
        return call_stack_list
    
    @property
    def call_stack_from_args(self) -> Optional[List[str]]:
        """从args中获取call stack信息（原始方式）"""
        if not self.args:
            return None
        call_stack_str = self.args.get('Call stack', None)
        if call_stack_str is None:
            return None
        # 将字符串按分号分割成列表
        return [frame.strip() for frame in call_stack_str.split(';') if frame.strip()]
    
    def set_call_stack_from_tree(self, call_stack: List[str]):
        """设置从调用栈树生成的call stack信息（反向存储）"""
        if not hasattr(self, '_call_stack_from_tree'):
            self._call_stack_from_tree = None
        # 反向存储 call_stack
        self._call_stack_from_tree = list(reversed(call_stack)) if call_stack is not None else None
        
        # 设置 fwd_bwd_type
        self._set_fwd_bwd_type_from_call_stack()
    
    @property
    def call_stack_from_tree(self) -> Optional[List[str]]:
        """获取从调用栈树生成的call stack信息"""
        return getattr(self, '_call_stack_from_tree', None)
    
    def _set_fwd_bwd_type_from_call_stack(self):
        """根据 call stack 设置 fwd_bwd_type，并增加 debug 信息"""
        if self.cat == 'cpu_op':
            # 优先使用 tree 来源的 call stack，如果没有则使用 args 来源的
            call_stack = self.call_stack_from_tree or self.call_stack_from_args
            assert call_stack is not None, f"cpu_op 事件 {self.name} 没有 call stack"
            if call_stack:
                # 检查 call stack 中是否包含 ':forward' 或 ':backward'
                for i, frame in enumerate(call_stack):
                    if ': forward' in frame:
                        self.fwd_bwd_type = 'fwd'
                        return
                    elif ': backward' in frame or 'Backward' in frame:
                        self.fwd_bwd_type = 'bwd'
                        return
                    elif 'Grad' in frame:
                        
                        self.fwd_bwd_type = 'grad'
                        return
                # 没有找到 ':forward' 或 ':backward'
                self.fwd_bwd_type = 'none'
            else:
                # 没有 call stack
                self.fwd_bwd_type = 'none'
        else:
            # 非 cpu_op 事件
            self.fwd_bwd_type = 'none'
    
    def get_call_stack(self, source: str = 'args') -> Optional[List[str]]:
        """
        统一的调用栈获取接口
        
        Args:
            source: 调用栈来源，'args' 或 'tree'
            
        Returns:
            Optional[List[str]]: 调用栈列表，如果不存在则返回None
        """
        if source == 'args':
            return self.call_stack_from_args
        elif source == 'tree':
            return self.call_stack_from_tree
        else:
            raise ValueError(f"不支持的调用栈来源: {source}，支持的值: 'args', 'tree'")
    
    @property
    def is_kernel(self) -> bool:
        """判断是否为 kernel 事件"""
        return self.cat == 'kernel' or 'kernel' in self.name.lower()
    
    @property
    def is_cuda_event(self) -> bool:
        """判断是否为 CUDA 事件"""
        return self.cat == 'cuda' or 'cuda' in self.name.lower()


@dataclass
class ProfilerData:
    """Profiler 数据模型"""
    metadata: Dict[str, Any]
    events: List[ActivityEvent]
    trace_events: List[Dict[str, Any]]
    
    def __post_init__(self):
        """初始化后处理"""
        if self.metadata is None:
            self.metadata = {}
        if self.events is None:
            self.events = []
        if self.trace_events is None:
            self.trace_events = []
        
        # 初始化索引缓存
        self._indexes = {}
        self._indexes_built = False
    
    @property
    def total_events(self) -> int:
        """总事件数"""
        return len(self.events)
    
    @property
    def kernel_events(self) -> List[ActivityEvent]:
        """获取所有 kernel 事件"""
        return [event for event in self.events if event.is_kernel]
    
    @property
    def cuda_events(self) -> List[ActivityEvent]:
        """获取所有 CUDA 事件"""
        return [event for event in self.events if event.is_cuda_event]
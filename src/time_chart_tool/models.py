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
    
    def __post_init__(self):
        """初始化后处理"""
        if self.args is None:
            self.args = {}
        
        # 从 args 中提取 stream_id
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
    def call_stack(self) -> Optional[List[str]]:
        """获取 call stack 信息"""
        if not self.args:
            return None
        call_stack_str = self.args.get('Call stack', None)
        if call_stack_str is None:
            return None
        # 将字符串按分号分割成列表
        return [frame.strip() for frame in call_stack_str.split(';') if frame.strip()]
    
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
    
    @property
    def unique_processes(self) -> List[Union[int, str]]:
        """获取唯一的进程 ID 列表"""
        pids = list(set(event.pid for event in self.events))
        # 分离数字和字符串，分别排序后合并
        numeric_pids = []
        string_pids = []
        for pid in pids:
            if isinstance(pid, int) or (isinstance(pid, str) and pid.isdigit()):
                numeric_pids.append(int(pid) if isinstance(pid, str) else pid)
            else:
                string_pids.append(pid)
        
        return sorted(numeric_pids) + sorted(string_pids)
    
    @property
    def unique_threads(self) -> List[Union[int, str]]:
        """获取唯一的线程 ID 列表"""
        tids = list(set(event.tid for event in self.events))
        # 分离数字和字符串，分别排序后合并
        numeric_tids = []
        string_tids = []
        for tid in tids:
            if isinstance(tid, int) or (isinstance(tid, str) and tid.isdigit()):
                numeric_tids.append(int(tid) if isinstance(tid, str) else tid)
            else:
                string_tids.append(tid)
        
        return sorted(numeric_tids) + sorted(string_tids)
    
    @property
    def unique_streams(self) -> List[Union[int, str]]:
        """获取唯一的流 ID 列表"""
        streams = [event.stream_id for event in self.events if event.stream_id is not None]
        streams = list(set(streams))
        # 分离数字和字符串，分别排序后合并
        numeric_streams = []
        string_streams = []
        for stream in streams:
            if isinstance(stream, int) or (isinstance(stream, str) and stream.isdigit()):
                numeric_streams.append(int(stream) if isinstance(stream, str) else stream)
            else:
                string_streams.append(stream)
        
        return sorted(numeric_streams) + sorted(string_streams)
    
    def get_events_by_process(self, pid: Union[int, str]) -> List[ActivityEvent]:
        """根据进程 ID 获取事件"""
        return [event for event in self.events if event.pid == pid]
    
    def get_events_by_thread(self, tid: Union[int, str]) -> List[ActivityEvent]:
        """根据线程 ID 获取事件"""
        return [event for event in self.events if event.tid == tid]
    
    def get_events_by_stream(self, stream_id: Union[int, str]) -> List[ActivityEvent]:
        """根据流 ID 获取事件"""
        return [event for event in self.events if event.stream_id == stream_id]
    
    def get_events_by_external_id(self, external_id: Union[int, str]) -> List[ActivityEvent]:
        """根据 External id 获取事件"""
        return [event for event in self.events if event.external_id == external_id]
    
    def get_events_by_correlation_id(self, correlation_id: Union[int, str]) -> List[ActivityEvent]:
        """根据 correlation id 获取事件"""
        return [event for event in self.events if event.correlation_id == correlation_id]
    
    def get_events_by_correlation(self, correlation: Union[int, str]) -> List[ActivityEvent]:
        """根据 correlation 获取事件"""
        return [event for event in self.events if event.correlation == correlation]
    
    def get_events_by_ev_idx(self, ev_idx: Union[int, str]) -> List[ActivityEvent]:
        """根据 Ev Idx 获取事件"""
        return [event for event in self.events if event.ev_idx == ev_idx]
    
    def get_events_by_python_id(self, python_id: Union[int, str]) -> List[ActivityEvent]:
        """根据 Python id 获取事件"""
        return [event for event in self.events if event.python_id == python_id]
    
    def get_events_by_any_id(self, id_value: Union[int, str]) -> List[ActivityEvent]:
        """根据任意 ID 字段获取事件（External id, correlation, Ev Idx, Python id）"""
        events = []
        for event in self.events:
            if (event.external_id == id_value or 
                event.correlation == id_value or 
                event.ev_idx == id_value or 
                event.python_id == id_value):
                events.append(event)
        return events

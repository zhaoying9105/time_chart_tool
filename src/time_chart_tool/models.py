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
    
    def _build_indexes(self) -> None:
        """构建所有索引"""
        if self._indexes_built:
            return
        
        # 初始化索引字典
        self._indexes = {
            'by_process': {},
            'by_thread': {},
            'by_stream': {},
            'by_external_id': {},
            'by_correlation_id': {},
            'by_correlation': {},
            'by_ev_idx': {},
            'by_python_id': {},
            'by_any_id': {}
        }
        
        # 遍历所有事件构建索引
        for event in self.events:
            # 进程索引
            if event.pid not in self._indexes['by_process']:
                self._indexes['by_process'][event.pid] = []
            self._indexes['by_process'][event.pid].append(event)
            
            # 线程索引
            if event.tid not in self._indexes['by_thread']:
                self._indexes['by_thread'][event.tid] = []
            self._indexes['by_thread'][event.tid].append(event)
            
            # 流索引
            if event.stream_id is not None:
                if event.stream_id not in self._indexes['by_stream']:
                    self._indexes['by_stream'][event.stream_id] = []
                self._indexes['by_stream'][event.stream_id].append(event)
            
            # External id 索引
            if event.external_id is not None:
                if event.external_id not in self._indexes['by_external_id']:
                    self._indexes['by_external_id'][event.external_id] = []
                self._indexes['by_external_id'][event.external_id].append(event)
            
            # correlation id 索引
            if event.correlation_id is not None:
                if event.correlation_id not in self._indexes['by_correlation_id']:
                    self._indexes['by_correlation_id'][event.correlation_id] = []
                self._indexes['by_correlation_id'][event.correlation_id].append(event)
            
            # correlation 索引
            if event.correlation is not None:
                if event.correlation not in self._indexes['by_correlation']:
                    self._indexes['by_correlation'][event.correlation] = []
                self._indexes['by_correlation'][event.correlation].append(event)
            
            # Ev Idx 索引
            if event.ev_idx is not None:
                if event.ev_idx not in self._indexes['by_ev_idx']:
                    self._indexes['by_ev_idx'][event.ev_idx] = []
                self._indexes['by_ev_idx'][event.ev_idx].append(event)
            
            # Python id 索引
            if event.python_id is not None:
                if event.python_id not in self._indexes['by_python_id']:
                    self._indexes['by_python_id'][event.python_id] = []
                self._indexes['by_python_id'][event.python_id].append(event)
            
            # 任意 ID 索引
            for id_value in [event.external_id, event.correlation, event.ev_idx, event.python_id]:
                if id_value is not None:
                    if id_value not in self._indexes['by_any_id']:
                        self._indexes['by_any_id'][id_value] = []
                    self._indexes['by_any_id'][id_value].append(event)
        
        self._indexes_built = True
    
    def get_events_by_process(self, pid: Union[int, str]) -> List[ActivityEvent]:
        """根据进程 ID 获取事件"""
        self._build_indexes()
        return self._indexes['by_process'].get(pid, [])
    
    def get_events_by_thread(self, tid: Union[int, str]) -> List[ActivityEvent]:
        """根据线程 ID 获取事件"""
        self._build_indexes()
        return self._indexes['by_thread'].get(tid, [])
    
    def get_events_by_stream(self, stream_id: Union[int, str]) -> List[ActivityEvent]:
        """根据流 ID 获取事件"""
        self._build_indexes()
        return self._indexes['by_stream'].get(stream_id, [])
    
    def get_events_by_external_id(self, external_id: Union[int, str]) -> List[ActivityEvent]:
        """根据 External id 获取事件"""
        self._build_indexes()
        return self._indexes['by_external_id'].get(external_id, [])
    
    def get_events_by_correlation_id(self, correlation_id: Union[int, str]) -> List[ActivityEvent]:
        """根据 correlation id 获取事件"""
        self._build_indexes()
        return self._indexes['by_correlation_id'].get(correlation_id, [])
    
    def get_events_by_correlation(self, correlation: Union[int, str]) -> List[ActivityEvent]:
        """根据 correlation 获取事件"""
        self._build_indexes()
        return self._indexes['by_correlation'].get(correlation, [])
    
    def get_events_by_ev_idx(self, ev_idx: Union[int, str]) -> List[ActivityEvent]:
        """根据 Ev Idx 获取事件"""
        self._build_indexes()
        return self._indexes['by_ev_idx'].get(ev_idx, [])
    
    def get_events_by_python_id(self, python_id: Union[int, str]) -> List[ActivityEvent]:
        """根据 Python id 获取事件"""
        self._build_indexes()
        return self._indexes['by_python_id'].get(python_id, [])
    
    def get_events_by_any_id(self, id_value: Union[int, str]) -> List[ActivityEvent]:
        """根据任意 ID 字段获取事件（External id, correlation, Ev Idx, Python id）"""
        self._build_indexes()
        return self._indexes['by_any_id'].get(id_value, [])

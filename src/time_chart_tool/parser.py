"""
PyTorch Profiler JSON 解析器
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from .models import ActivityEvent, ProfilerData

logger = logging.getLogger(__name__)


class PyTorchProfilerParser:
    """PyTorch Profiler JSON 解析器"""
    
    def __init__(self):
        self.data: Optional[ProfilerData] = None
    
    def load_json_file(self, file_path: Union[str, Path]) -> ProfilerData:
        """
        加载 PyTorch profiler JSON 文件
        
        Args:
            file_path: JSON 文件路径
            
        Returns:
            ProfilerData: 解析后的数据对象
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        logger.info(f"正在加载文件: {file_path}")
        
        # 尝试读取文件内容
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析错误: {e}")
        except Exception as e:
            raise RuntimeError(f"文件读取错误: {e}")
        
        # 解析数据
        self.data = self._parse_data(data)
        logger.info(f"成功加载文件，包含 {self.data.total_events} 个事件")
        
        return self.data
    
    def _parse_data(self, data: Dict[str, Any]) -> ProfilerData:
        """
        解析 JSON 数据
        
        Args:
            data: JSON 数据字典
            
        Returns:
            ProfilerData: 解析后的数据对象
        """
        # 提取元数据
        metadata = data.get('metadata', {})
        
        # 提取 traceEvents
        trace_events = data.get('traceEvents', [])
        
        # 解析事件
        events = []
        for event_data in trace_events:
            try:
                event = self._parse_event(event_data)
                if event:
                    events.append(event)
            except Exception as e:
                logger.warning(f"解析事件失败: {e}, 事件数据: {event_data}")
                continue
        
        return ProfilerData(
            metadata=metadata,
            events=events,
            trace_events=trace_events
        )
    
    def _parse_event(self, event_data: Dict[str, Any]) -> Optional[ActivityEvent]:
        """
        解析单个事件
        
        Args:
            event_data: 事件数据字典
            
        Returns:
            ActivityEvent: 解析后的事件对象，如果解析失败返回 None
        """
        try:
            # 提取必需字段
            name = event_data.get('name', '')
            cat = event_data.get('cat', '')
            ph = event_data.get('ph', '')
            pid = event_data.get('pid', 0)
            tid = event_data.get('tid', 0)
            ts = event_data.get('ts', 0.0)
            
            # 提取可选字段
            dur = event_data.get('dur')
            args = event_data.get('args', {})
            event_id = event_data.get('id')
            stream_id = event_data.get('stream')
            
            # 创建事件对象
            event = ActivityEvent(
                name=name,
                cat=cat,
                ph=ph,
                pid=pid,
                tid=tid,
                ts=ts,
                dur=dur,
                args=args,
                id=event_id,
                stream_id=stream_id
            )
            
            return event
            
        except Exception as e:
            logger.warning(f"解析事件失败: {e}")
            return None
    
    def print_metadata(self) -> None:
        """打印元数据信息"""
        if not self.data:
            print("未加载数据，请先调用 load_json_file()")
            return
        
        print("=== PyTorch Profiler 元数据 ===")
        for key, value in self.data.metadata.items():
            print(f"{key}: {value}")
        print()
    
    def print_statistics(self) -> None:
        """打印统计信息"""
        if not self.data:
            print("未加载数据，请先调用 load_json_file()")
            return
        
        print("=== PyTorch Profiler 统计信息 ===")
        print(f"总事件数: {self.data.total_events}")
        print(f"Kernel 事件数: {len(self.data.kernel_events)}")
        print(f"CUDA 事件数: {len(self.data.cuda_events)}")
        print(f"唯一进程数: {len(self.data.unique_processes)}")
        print(f"唯一线程数: {len(self.data.unique_threads)}")
        print(f"唯一流数: {len(self.data.unique_streams)}")
        
        if self.data.unique_processes:
            print(f"进程 ID 列表: {self.data.unique_processes}")
        if self.data.unique_threads:
            print(f"线程 ID 列表: {self.data.unique_threads[:10]}{'...' if len(self.data.unique_threads) > 10 else ''}")
        if self.data.unique_streams:
            print(f"流 ID 列表: {self.data.unique_streams}")
        print()
    
    def search_by_process(self, pid: Union[int, str]) -> List[ActivityEvent]:
        """
        根据进程 ID 搜索事件
        
        Args:
            pid: 进程 ID
            
        Returns:
            List[ActivityEvent]: 匹配的事件列表
        """
        if not self.data:
            return []
        
        events = self.data.get_events_by_process(pid)
        print(f"进程 {pid} 的事件数: {len(events)}")
        return events
    
    def search_by_thread(self, tid: Union[int, str]) -> List[ActivityEvent]:
        """
        根据线程 ID 搜索事件
        
        Args:
            tid: 线程 ID
            
        Returns:
            List[ActivityEvent]: 匹配的事件列表
        """
        if not self.data:
            return []
        
        events = self.data.get_events_by_thread(tid)
        print(f"线程 {tid} 的事件数: {len(events)}")
        return events
    
    def search_by_stream(self, stream_id: Union[int, str]) -> List[ActivityEvent]:
        """
        根据流 ID 搜索事件
        
        Args:
            stream_id: 流 ID
            
        Returns:
            List[ActivityEvent]: 匹配的事件列表
        """
        if not self.data:
            return []
        
        events = self.data.get_events_by_stream(stream_id)
        print(f"流 {stream_id} 的事件数: {len(events)}")
        return events
    
    def search_by_external_id(self, external_id: Union[int, str]) -> List[ActivityEvent]:
        """
        根据 External id 搜索事件
        
        Args:
            external_id: External id 值
            
        Returns:
            List[ActivityEvent]: 匹配的事件列表
        """
        if not self.data:
            return []
        
        events = self.data.get_events_by_external_id(external_id)
        print(f"External id '{external_id}' 的事件数: {len(events)}")
        return events
    
    def search_by_correlation_id(self, correlation_id: Union[int, str]) -> List[ActivityEvent]:
        """
        根据 correlation id 搜索事件
        
        Args:
            correlation_id: correlation id 值
            
        Returns:
            List[ActivityEvent]: 匹配的事件列表
        """
        if not self.data:
            return []
        
        events = self.data.get_events_by_correlation_id(correlation_id)
        print(f"Correlation id '{correlation_id}' 的事件数: {len(events)}")
        return events
    
    def search_by_correlation(self, correlation: Union[int, str]) -> List[ActivityEvent]:
        """
        根据 correlation 搜索事件
        
        Args:
            correlation: correlation 值
            
        Returns:
            List[ActivityEvent]: 匹配的事件列表
        """
        if not self.data:
            return []
        
        events = self.data.get_events_by_correlation(correlation)
        print(f"Correlation '{correlation}' 的事件数: {len(events)}")
        return events
    
    def search_by_ev_idx(self, ev_idx: Union[int, str]) -> List[ActivityEvent]:
        """
        根据 Ev Idx 搜索事件
        
        Args:
            ev_idx: Ev Idx 值
            
        Returns:
            List[ActivityEvent]: 匹配的事件列表
        """
        if not self.data:
            return []
        
        events = self.data.get_events_by_ev_idx(ev_idx)
        print(f"Ev Idx '{ev_idx}' 的事件数: {len(events)}")
        return events
    
    def search_by_python_id(self, python_id: Union[int, str]) -> List[ActivityEvent]:
        """
        根据 Python id 搜索事件
        
        Args:
            python_id: Python id 值
            
        Returns:
            List[ActivityEvent]: 匹配的事件列表
        """
        if not self.data:
            return []
        
        events = self.data.get_events_by_python_id(python_id)
        print(f"Python id '{python_id}' 的事件数: {len(events)}")
        return events
    
    def search_by_any_id(self, id_value: Union[int, str]) -> List[ActivityEvent]:
        """
        根据任意 ID 字段搜索事件（External id, correlation, Ev Idx, Python id）
        
        Args:
            id_value: ID 值
            
        Returns:
            List[ActivityEvent]: 匹配的事件列表
        """
        if not self.data:
            return []
        
        events = self.data.get_events_by_any_id(id_value)
        print(f"任意 ID '{id_value}' 的事件数: {len(events)}")
        return events
    
    def search_by_id(self, id_value: Union[int, str]) -> Tuple[List[ActivityEvent], List[ActivityEvent]]:
        """
        根据 External id 或 correlation id 搜索事件
        
        Args:
            id_value: External id 或 correlation id 值
            
        Returns:
            Tuple[List[ActivityEvent], List[ActivityEvent]]: 
                (External id 匹配的事件列表, correlation id 匹配的事件列表)
        """
        if not self.data:
            return [], []
        
        external_events = self.search_by_external_id(id_value)
        correlation_events = self.search_by_correlation_id(id_value)
        
        return external_events, correlation_events
    
    def search_by_call_stack(self, call_stack: List[str]) -> List[ActivityEvent]:
        """
        根据 call stack 搜索事件
        
        Args:
            call_stack: call stack 列表
            
        Returns:
            List[ActivityEvent]: 匹配的事件列表
        """
        if not self.data:
            return []
        
        events = []
        for event in self.data.events:
            if event.call_stack == call_stack:
                events.append(event)
        
        print(f"Call stack 匹配的事件数: {len(events)}")
        return events
    
    def get_cpu_op_events_with_call_stack(self) -> List[ActivityEvent]:
        """
        获取所有包含 call stack 的 cpu_op 事件
        
        Returns:
            List[ActivityEvent]: 包含 call stack 的 cpu_op 事件列表
        """
        if not self.data:
            return []
        
        events = []
        for event in self.data.events:
            if event.cat == 'cpu_op' and event.call_stack is not None:
                events.append(event)
        
        print(f"包含 call stack 的 cpu_op 事件数: {len(events)}")
        return events
    
    def get_unique_call_stacks(self) -> List[List[str]]:
        """
        获取所有唯一的 call stack
        
        Returns:
            List[List[str]]: 唯一的 call stack 列表
        """
        if not self.data:
            return []
        
        call_stacks = set()
        for event in self.data.events:
            if event.call_stack is not None:
                # 将 list 转换为 tuple 以便可以放入 set
                call_stacks.add(tuple(event.call_stack))
        
        # 转换回 list 并排序（按第一次出现的时间）
        unique_call_stacks = [list(cs) for cs in call_stacks]
        
        # 按第一次出现的时间排序
        call_stack_first_occurrence = {}
        for event in self.data.events:
            if event.call_stack is not None:
                cs_tuple = tuple(event.call_stack)
                if cs_tuple not in call_stack_first_occurrence:
                    call_stack_first_occurrence[cs_tuple] = event.ts
        
        unique_call_stacks.sort(key=lambda cs: call_stack_first_occurrence[tuple(cs)])
        
        print(f"唯一的 call stack 数量: {len(unique_call_stacks)}")
        return unique_call_stacks

    def print_events_summary(self, events: List[ActivityEvent], title: str = "事件摘要") -> None:
        """
        打印事件摘要信息
        
        Args:
            events: 事件列表
            title: 标题
        """
        if not events:
            print(f"{title}: 无事件")
            return
        
        print(f"=== {title} ===")
        print(f"事件数量: {len(events)}")
        
        # 按类型统计
        event_types = {}
        for event in events:
            event_type = f"{event.cat}:{event.name}"
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print("事件类型统计:")
        for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {event_type}: {count}")
        
        if len(event_types) > 10:
            print(f"  ... 还有 {len(event_types) - 10} 种类型")
        
        print()

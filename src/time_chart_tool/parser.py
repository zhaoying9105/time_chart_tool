"""
PyTorch Profiler JSON 解析器
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime

from .models import ActivityEvent, ProfilerData
from .call_stack_builder import CallStackBuilder

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


def _parse_event(event_data: Dict[str, Any], base_time_nanoseconds: Optional[int] = None) -> Optional[ActivityEvent]:
    """
    解析单个事件
    
    Args:
        event_data: 事件数据字典
        base_time_nanoseconds: 基准时间（纳秒），用于计算可读时间戳
        
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
        
        # 计算可读时间戳
        readable_timestamp = None
        if base_time_nanoseconds is not None and ts is not None:
            # ts是微秒，base_time_nanoseconds是纳秒
            # 将ts转换为纳秒，然后加上基准时间
            total_nanoseconds = base_time_nanoseconds + int(ts * 1000)  # ts * 1000 将微秒转换为纳秒
            readable_timestamp = _nanoseconds_to_readable_timestamp(total_nanoseconds)
        
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
            stream_id=stream_id,
            readable_timestamp=readable_timestamp
        )
        
        return event
        
    except Exception as e:
        logger.warning(f"解析事件失败: {e}")
        return None


class PyTorchProfilerParser:
    """PyTorch Profiler JSON 解析器"""
    
    def __init__(self, step_idx: Optional[int] = None):
        self.data: Optional[ProfilerData] = None
        self.call_stack_builder = CallStackBuilder()
        self._call_stack_trees: Optional[Dict[Tuple[int, int], Any]] = None
        self.step_idx: Optional[int] = step_idx
    
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
        # 提取元数据（从根级别提取相关属性）
        metadata = {
            'schemaVersion': data.get('schemaVersion'),
            'with_modules': data.get('with_modules'),
            'distributedInfo': data.get('distributedInfo'),
            'record_shapes': data.get('record_shapes'),
            'with_stack': data.get('with_stack'),
            'traceName': data.get('traceName'),
            'displayTimeUnit': data.get('displayTimeUnit'),
            'baseTimeNanoseconds': data.get('baseTimeNanoseconds')
        }
        
        # 提取 traceEvents
        trace_events = data.get('traceEvents', [])
        
        # 串行解析事件
        base_time_nanoseconds = metadata.get('baseTimeNanoseconds')
        events = self._parse_events(trace_events, base_time_nanoseconds)
        
        # 标准化时间戳
        events = self._normalize_timestamps(events)
        
        # 如果指定了 step_idx，则进行 step 事件过滤
        if self.step_idx is not None:
            logger.info(f"开始进行 step 事件过滤，step_idx: {self.step_idx}")
            
            # 检测所有 step 事件
            step_events = self._detect_step_events(events)
            
            if not step_events:
                logger.warning("未找到任何 train_runner.py step 事件，将使用所有事件")
            elif self.step_idx >= len(step_events):
                logger.warning(f"step_idx {self.step_idx} 超出范围（总共 {len(step_events)} 个 step），将使用所有事件")
            else:
                # 选择指定的 step 事件
                selected_step_event = step_events[self.step_idx]
                logger.info(f"选择第 {self.step_idx} 个 step 事件: {selected_step_event.name}")
                logger.info(f"Step 事件时间区间: [{selected_step_event.ts:.2f}, {selected_step_event.ts + (selected_step_event.dur or 0):.2f}] 微秒")
                
                # 根据选定的 step 事件过滤其他事件
                events = self._filter_events_by_step(events, selected_step_event)
        
        return ProfilerData(
            metadata=metadata,
            events=events,
            trace_events=trace_events
        )
    
    def _parse_events(self, trace_events: List[Dict[str, Any]], base_time_nanoseconds: Optional[int] = None) -> List[ActivityEvent]:
        """
        串行解析事件列表
        
        Args:
            trace_events: 原始事件数据列表
            base_time_nanoseconds: 基准时间（纳秒），用于计算可读时间戳
            
        Returns:
            List[ActivityEvent]: 解析后的事件列表
        """
        events = []
        for event_data in trace_events:
            try:
                event = _parse_event(event_data, base_time_nanoseconds)
                if event:
                    events.append(event)
            except Exception as e:
                logger.warning(f"解析事件失败: {e}, 事件数据: {event_data}")
                continue
        return events
    
    def _normalize_timestamps(self, events: List[ActivityEvent]) -> List[ActivityEvent]:
        """
        标准化时间戳：找到最小ts值，将所有事件的ts减去这个值
        
        Args:
            events: 事件列表
            
        Returns:
            List[ActivityEvent]: 标准化时间戳后的事件列表
        """
        if not events:
            return events
        
        # 找到最小时间戳
        valid_ts = [event.ts for event in events if event.ts is not None]
        if not valid_ts:
            logger.warning("没有找到有效的时间戳，跳过标准化")
            return events
        
        min_ts = min(valid_ts)
        logger.info(f"找到最小时间戳: {min_ts:.2f} 微秒，开始标准化...")
        
        # 标准化所有事件的时间戳
        for event in events:
            if event.ts is not None:
                event.ts = event.ts - min_ts
        
        logger.info("时间戳标准化完成")
        return events
    
    def _detect_step_events(self, events: List[ActivityEvent]) -> List[ActivityEvent]:
        """
        检测 train_runner.py step 事件
        
        Args:
            events: 事件列表
            
        Returns:
            List[ActivityEvent]: train_runner.py step 事件列表
        """
        step_events = []
        for event in events:
            if event.name and "train_runner.py" in event.name and "step" in event.name:
                step_events.append(event)
        
        logger.info(f"检测到 {len(step_events)} 个 step 事件")
        return step_events
    
    def _has_time_intersection(self, event1: ActivityEvent, event2: ActivityEvent) -> bool:
        """
        检查两个事件的时间区间是否有交集
        
        Args:
            event1: 第一个事件
            event2: 第二个事件
            
        Returns:
            bool: 是否有交集
        """
        if event1.ts is None or event2.ts is None:
            return False
        
        # 计算事件1的时间区间
        event1_start = event1.ts
        event1_end = event1.ts + (event1.dur if event1.dur is not None else 0)
        
        # 计算事件2的时间区间
        event2_start = event2.ts
        event2_end = event2.ts + (event2.dur if event2.dur is not None else 0)
        
        # 检查是否有交集：两个区间有交集当且仅当 max(start1, start2) < min(end1, end2)
        return max(event1_start, event2_start) < min(event1_end, event2_end)
    
    def _filter_events_by_step(self, events: List[ActivityEvent], step_event: ActivityEvent) -> List[ActivityEvent]:
        """
        根据指定的 step 事件过滤其他事件，只保留与 step 事件有时间交集的事件
        
        Args:
            events: 所有事件列表
            step_event: 指定的 step 事件
            
        Returns:
            List[ActivityEvent]: 过滤后的事件列表
        """
        filtered_events = []
        for event in events:
            if self._has_time_intersection(event, step_event):
                filtered_events.append(event)
        
        logger.info(f"根据 step 事件过滤后，保留 {len(filtered_events)} 个事件（原始 {len(events)} 个）")
        return filtered_events
    
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
    
    def build_call_stack_trees(self) -> Dict[Tuple[int, int], Any]:
        """
        构建调用栈树
        
        Returns:
            Dict[Tuple[int, int], Any]: 按(pid, tid)分组的调用栈树
        """
        if not self.data:
            logger.warning("未加载数据，请先调用 load_json_file()")
            return {}
        
        logger.info("开始构建调用栈树...")
        self._call_stack_trees = self.call_stack_builder.build_call_stacks(self.data.events)
        
        # 为每个事件设置调用栈信息
        for (pid, tid), tree_root in self._call_stack_trees.items():
            events_in_group = self.data.get_events_by_process(pid)
            events_in_group = [e for e in events_in_group if e.tid == tid]
            
            for event in events_in_group:
                call_stack = self.call_stack_builder.get_call_stack_for_event(event, self._call_stack_trees)
                if call_stack:
                    event.set_call_stack_from_tree(call_stack)
        
        logger.info(f"调用栈树构建完成，共 {len(self._call_stack_trees)} 个树")
        return self._call_stack_trees
    
    def get_call_stack_trees(self) -> Optional[Dict[Tuple[int, int], Any]]:
        """获取调用栈树"""
        return self._call_stack_trees
    
    def print_call_stack_statistics(self) -> None:
        """打印调用栈统计信息"""
        if not self._call_stack_trees:
            print("调用栈树未构建，请先调用 build_call_stack_trees()")
            return
        
        stats = self.call_stack_builder.get_tree_statistics(self._call_stack_trees)
        
        print("=== 调用栈树统计信息 ===")
        print(f"总树数: {stats['total_trees']}")
        print(f"总节点数: {stats['total_nodes']}")
        print(f"最大深度: {stats['max_depth']}")
        print(f"平均深度: {stats['avg_depth']:.2f}")
        
        print("\n各树详细信息:")
        for tree_info in stats['tree_sizes']:
            print(f"  进程 {tree_info['pid']} 线程 {tree_info['tid']}: "
                  f"节点数 {tree_info['size']}, 深度 {tree_info['depth']}")
        print()
    
    def print_call_stack_tree(self, pid: int, tid: int, max_depth: int = 10) -> None:
        """
        打印指定进程和线程的调用栈树
        
        Args:
            pid: 进程ID
            tid: 线程ID
            max_depth: 最大打印深度
        """
        if not self._call_stack_trees:
            print("调用栈树未构建，请先调用 build_call_stack_trees()")
            return
        
        tree_root = self._call_stack_trees.get((pid, tid))
        if not tree_root:
            print(f"未找到进程 {pid} 线程 {tid} 的调用栈树")
            return
        
        print(f"=== 进程 {pid} 线程 {tid} 的调用栈树 ===")
        self.call_stack_builder.print_call_stack_tree(tree_root, max_depth)
        print()

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

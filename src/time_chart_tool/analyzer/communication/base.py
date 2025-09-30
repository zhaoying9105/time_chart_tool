"""
通信分析器基础模块
"""

from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from pathlib import Path

from ....models import ActivityEvent, ProfilerData

class BaseCommunicationAnalyzer(ABC):
    """通信分析器基类"""
    
    COMMUNICATION_BLACKLIST_PATTERNS = [
        'TCDP_RING_ALLGATHER_',
        'TCDP_RING_REDUCESCATTER_',
        'ALLREDUCELL',
        'TCDP_RING_ALLREDUCE_SIMPLE_BF16_ADD'
    ]
    
    def __init__(self):
        self.parser = None  # 由子类初始化
    
    @abstractmethod
    def analyze_communication_performance(self, pod_dir: str, **kwargs) -> List[Path]:
        """分析通信性能的主入口"""
        pass
    
    def _scan_executor_folders(self, pod_dir: str) -> List[str]:
        """扫描Pod目录下的executor文件夹"""
        import os
        executor_folders = []
        
        for item in os.listdir(pod_dir):
            item_path = os.path.join(pod_dir, item)
            if os.path.isdir(item_path) and item.startswith('executor_trainer-runner_'):
                executor_folders.append(item_path)
        
        return sorted(executor_folders)
    
    def _parse_json_filename(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        """解析JSON文件名，提取step和card索引"""
        import re
        
        # 匹配多种文件名模式
        patterns = [
            r'(\d+)_(\d+)\.json',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    card_idx = int(match.group(1))
                    step = int(match.group(2))
                    return step, card_idx
                elif len(match.groups()) == 1:
                    # 只有step信息，card_idx设为0
                    step = int(match.group(1))
                    return step, 0
        
        return None, None
    
    def _find_json_file(self, executor_folders: List[str], step: int, card_idx: int) -> Optional[str]:
        """根据step和card_idx找到对应的JSON文件"""
        for executor_folder in executor_folders:
            json_files = list(Path(executor_folder).glob("*.json"))
            for json_file in json_files:
                parsed_step, parsed_card_idx = self._parse_json_filename(json_file.name)
                if parsed_step == step and parsed_card_idx == card_idx:
                    return str(json_file)
        return None
    
    def _extract_communication_durations(self, json_file_path: str, kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL") -> List[float]:
        """从JSON文件中提取指定通信kernel前缀的duration"""
        import re
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用正则表达式匹配指定的通信kernel前缀
            escaped_kernel_prefix = re.escape(kernel_prefix)
            pattern = rf'"ph":\s*"X",\s*"cat":\s*"kernel",\s*"name":\s*"{escaped_kernel_prefix}[^"]*",\s*"pid":\s*\d+,\s*"tid":\s*\d+,\s*"ts":\s*[\d.]+,\s*"dur":\s*([\d.]+)'
            
            matches = re.findall(pattern, content, re.DOTALL)
            
            # 转换为浮点数，最多取6个
            durations = [float(match) for match in matches[:6]]
            
            return durations
            
        except Exception as e:
            print(f"    读取JSON文件失败 {json_file_path}: {e}")
            return []


class EventAnalysisMixin:
    """事件分析混合类"""
    
    def _find_all_communication_events(self, data: ProfilerData) -> List[ActivityEvent]:
        """找到所有TCDP_开头的通信kernel events，按结束时间排序"""
        comm_events = []
        
        for event in data.events:
            if (event.cat == 'kernel' and 
                event.name.startswith('TCDP_')):
                # 检查是否在黑名单中
                is_blacklisted = any(pattern in event.name for pattern in self.COMMUNICATION_BLACKLIST_PATTERNS)
                if not is_blacklisted:
                    comm_events.append(event)
        
        # 按结束时间排序（从早到晚）
        comm_events.sort(key=lambda x: x.ts + x.dur)
        
        return comm_events
    
    def _find_communication_event(self, data, comm_idx, kernel_prefix):
        """找到指定comm_idx的通信kernel操作event"""
        events = self._find_events_by_criteria(
            data.events, 
            lambda e: e.cat == 'kernel' and e.name.startswith(kernel_prefix)
        )
        
        if comm_idx < len(events):
            print(f"    找到目标通信kernel event: {events[comm_idx].name}")
            return events[comm_idx]
        
        return None
    
    def _find_events_by_criteria(self, events, criteria_func):
        """根据条件查找事件"""
        return [event for event in events if criteria_func(event)]


def _readable_timestamp_to_microseconds(readable_timestamp: str) -> float:
    """将readable_timestamp转换为微秒时间戳"""
    from datetime import datetime
    
    try:
        dt = datetime.strptime(readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp() * 1_000_000
    except Exception as e:
        print(f"警告: 时间戳转换失败: {e}")
        return 0.0


def _format_timestamp_display(event, show_readable: bool = True) -> str:
    """格式化时间戳显示"""
    if show_readable:
        return event.readable_timestamp
    else:
        return f"{event.ts:.2f}μs"


def _calculate_end_time_display(event) -> str:
    """计算并格式化结束时间显示"""
    from datetime import datetime, timedelta
    
    start_dt = datetime.strptime(event.readable_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    end_dt = start_dt + timedelta(microseconds=event.dur)
    return end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def _calculate_time_diff_readable(event1, event2) -> float:
    """使用readable_timestamp计算时间差（微秒）"""
    start1 = _readable_timestamp_to_microseconds(event1.readable_timestamp)
    start2 = _readable_timestamp_to_microseconds(event2.readable_timestamp)
    return abs(start1 - start2)



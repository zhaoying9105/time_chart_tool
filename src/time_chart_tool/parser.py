"""
PyTorch Profiler JSON 解析器
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime

from .models import ActivityEvent
from .call_stack_builder import build_call_stack_trees, CallStackBuilder, CallStackNode

logger = logging.getLogger(__name__)


def _parse_event(event_data: Dict[str, Any]) -> Optional[ActivityEvent]:
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
            stream_id=stream_id,
            readable_timestamp=None # 将在后处理阶段计算
        )
        
        return event
        
    except Exception as e:
        logger.warning(f"解析事件失败: {e}")
        return None


def parse_profiler_data(file_path: Union[str, Path], step_idx: Optional[int] = None):
    """
    解析 PyTorch profiler JSON 文件
    
    Args:
        file_path: JSON 文件路径
        step_idx: 指定要分析的step索引，如果为None则分析所有step
        
    Returns:
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return None
    
    try:
        print(f"正在解析文件: {file_path}")
        
        # 读取文件
        open_func = gzip.open if file_path.suffix == '.gz' else open
        with open_func(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        raw_events = data['traceEvents']
        print(f"读取到 {len(raw_events)} 个原始事件")
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
        
        # 串行解析事件
        base_time_nanoseconds = metadata.get('baseTimeNanoseconds')
        
        # 解析事件
        events = []
        for raw_event in raw_events:
            event = _parse_event(raw_event)
            events.append(event)
        
        
        # 使用 build_call_stack_trees 构建调用栈树
        call_stack_trees = build_call_stack_trees(events)
        print(f"构建了 {len(call_stack_trees)} 个调用栈树")
        
        return events, call_stack_trees, base_time_nanoseconds
        
    except Exception as e:
        logger.error(f"解析文件出错: {e}", exc_info=True)
        return None

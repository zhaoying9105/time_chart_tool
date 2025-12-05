"""
树结构处理工具模块
"""

from typing import List, Dict, Tuple, Optional, Any
import logging

from ..models import ActivityEvent
from .event_utils import create_event_id

logger = logging.getLogger(__name__)

def extract_call_stacks_from_tree(root) -> Dict[str, List[str]]:
    """
    从树中提取所有事件的调用栈
    
    Args:
        root: 树根节点 (CallStackNode)
        
    Returns:
        Dict[str, List[str]]: 结果映射，键为事件ID，值为调用栈列表
    """
    event_map = {}
    # 深度优先遍历
    stack = [root]
    while stack:
        current = stack.pop()
        
        # 记录当前节点的调用栈
        if current.event.name != "ROOT":
            event_id = create_event_id(current.event)
            event_map[event_id] = current.get_call_stack()
        
        # 将子节点加入栈
        stack.extend(current.children)
    return event_map


def attach_call_stacks_to_events(events: List[ActivityEvent], call_stack_trees: Dict[Tuple[int, int], Any]) -> List[ActivityEvent]:
    """
    提取调用栈并附加到事件属性中（纯函数）
    
    Args:
        events: 事件列表
        call_stack_trees: 调用栈树字典 { (pid, tid): CallStackNode }
        
    Returns:
        List[ActivityEvent]: 附加了调用栈信息的事件列表
    """
    # 1. 构建所有树的事件ID到调用栈的映射
    event_call_stacks = {}
    for root in call_stack_trees.values():
        event_call_stacks.update(extract_call_stacks_from_tree(root))
    
    # 2. 将调用栈附加到事件对象
    # 为了保持纯函数特性，我们应该创建新的事件对象，而不是修改传入的事件对象
    # 使用 copy() 方法创建浅拷贝，只修改 call_stack_from_tree 属性
    processed_events = []
    for event in events:
        # 只有CPU op需要调用栈
        if event.dur is not None and event.dur > 0:
            event_id = create_event_id(event)
            if event_id in event_call_stacks:
                event.call_stack = event_call_stacks[event_id]
            processed_events.append(event)
        else:
            processed_events.append(event)
            
    return processed_events

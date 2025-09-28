"""
基于扫描线和线段树的调用栈构建算法
时间复杂度: O(n log n)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set, Any
import logging
from collections import defaultdict

from .models import ActivityEvent

logger = logging.getLogger(__name__)


@dataclass
class EventInterval:
    """事件时间区间"""
    event: ActivityEvent
    start: float  # ts
    end: float    # ts + dur
    index: int    # 原始事件索引，用于排序稳定性
    
    def __post_init__(self):
        """确保end >= start"""
        if self.end < self.start:
            self.end = self.start


@dataclass
class CallStackNode:
    """调用栈树节点"""
    event: ActivityEvent
    children: List['CallStackNode']
    parent: Optional['CallStackNode']
    depth: int
    
    def __init__(self, event: ActivityEvent, parent: Optional['CallStackNode'] = None):
        self.event = event
        self.children = []
        self.parent = parent
        self.depth = parent.depth + 1 if parent else 0
        # 为快速查找创建事件标识符
        self.event_id = self._create_event_id(event)
    
    def _create_event_id(self, event: ActivityEvent) -> str:
        """创建事件的唯一标识符"""
        return f"{event.name}:{event.ts}:{event.dur}:{event.pid}:{event.tid}"
    
    def add_child(self, child: 'CallStackNode'):
        """添加子节点"""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def get_call_stack(self) -> List[str]:
        """获取从根到当前节点的调用栈路径"""
        path = []
        current = self
        while current is not None:
            path.append(current.event.name)
            current = current.parent
        return list(reversed(path))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'event': {
                'name': self.event.name,
                'cat': self.event.cat,
                'ts': self.event.ts,
                'dur': self.event.dur,
                'pid': self.event.pid,
                'tid': self.event.tid
            },
            'children': [child.to_dict() for child in self.children],
            'depth': self.depth,
            'call_stack': self.get_call_stack()
        }


class SegmentTree:
    """线段树，用于快速查询区间内的最大结束时间"""
    
    def __init__(self, intervals: List[EventInterval]):
        """初始化线段树"""
        if not intervals:
            self.n = 0
            self.tree = []
            return
        
        # 获取所有时间点并排序
        time_points = set()
        for interval in intervals:
            time_points.add(interval.start)
            time_points.add(interval.end)
        
        self.time_points = sorted(time_points)
        self.n = len(self.time_points)
        
        # 构建线段树
        self.tree = [0.0] * (4 * self.n)
        self._build(intervals)
    
    def _build(self, intervals: List[EventInterval]):
        """构建线段树"""
        # 为每个区间更新对应的叶子节点
        for interval in intervals:
            start_idx = self._get_index(interval.start)
            end_idx = self._get_index(interval.end)
            self._update_range(0, 0, self.n - 1, start_idx, end_idx, interval.end)
    
    def _get_index(self, time_point: float) -> int:
        """获取时间点在数组中的索引"""
        left, right = 0, self.n - 1
        while left <= right:
            mid = (left + right) // 2
            if self.time_points[mid] == time_point:
                return mid
            elif self.time_points[mid] < time_point:
                left = mid + 1
            else:
                right = mid - 1
        return left
    
    def _update_range(self, node: int, start: int, end: int, l: int, r: int, value: float):
        """更新区间[l, r]的最大值"""
        if start > r or end < l:
            return
        
        if start >= l and end <= r:
            self.tree[node] = max(self.tree[node], value)
            return
        
        mid = (start + end) // 2
        self._update_range(2 * node + 1, start, mid, l, r, value)
        self._update_range(2 * node + 2, mid + 1, end, l, r, value)
        self.tree[node] = max(self.tree[2 * node + 1], self.tree[2 * node + 2])
    
    def query_max_end(self, start_time: float, end_time: float) -> float:
        """查询区间[start_time, end_time]内的最大结束时间"""
        if self.n == 0:
            return 0.0
        
        start_idx = self._get_index(start_time)
        end_idx = self._get_index(end_time)
        return self._query(0, 0, self.n - 1, start_idx, end_idx)
    
    def _query(self, node: int, start: int, end: int, l: int, r: int) -> float:
        """查询区间[l, r]的最大值"""
        if start > r or end < l:
            return 0.0
        
        if start >= l and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_max = self._query(2 * node + 1, start, mid, l, r)
        right_max = self._query(2 * node + 2, mid + 1, end, l, r)
        return max(left_max, right_max)


class CallStackBuilder:
    """基于扫描线和线段树的调用栈构建器"""
    
    def __init__(self):
        self.logger = logger
        # 事件到节点的映射，用于快速查找
        self.event_to_node_map: Dict[str, CallStackNode] = {}
    
    def build_call_stacks(self, events: List[ActivityEvent]) -> Dict[Tuple[int, int], CallStackNode]:
        """
        为所有事件构建调用栈树
        
        Args:
            events: 事件列表
            
        Returns:
            Dict[Tuple[int, int], CallStackNode]: 按(pid, tid)分组的调用栈树根节点
        """
        # 清空事件映射
        self.event_to_node_map.clear()
        
        # 1. 按pid和tid分组事件
        events_by_pid_tid = self._group_events_by_pid_tid(events)
        
        # 2. 为每个(pid, tid)组构建调用栈树
        call_stack_trees = {}
        
        for (pid, tid), group_events in events_by_pid_tid.items():
            self.logger.info(f"为进程 {pid} 线程 {tid} 构建调用栈，事件数: {len(group_events)}")
            
            # 过滤掉没有duration的事件
            valid_events = [e for e in group_events if e.dur is not None and e.dur > 0]
            
            if not valid_events:
                self.logger.warning(f"进程 {pid} 线程 {tid} 没有有效的事件")
                continue
            
            # 构建调用栈树
            root = self._build_call_stack_tree(valid_events)
            if root:
                call_stack_trees[(pid, tid)] = root
                # 构建事件到节点的映射
                self._build_event_mapping(root)
        
        self.logger.info(f"成功构建 {len(call_stack_trees)} 个调用栈树，映射了 {len(self.event_to_node_map)} 个事件")
        return call_stack_trees
    
    def _group_events_by_pid_tid(self, events: List[ActivityEvent]) -> Dict[Tuple[int, int], List[ActivityEvent]]:
        """
        按pid和tid分组事件，过滤掉没有pid/tid的事件
        
        Args:
            events: 事件列表
            
        Returns:
            Dict[Tuple[int, int], List[ActivityEvent]]: 按(pid, tid)分组的事件
        """
        events_by_pid_tid = defaultdict(list)
        
        for event in events:
            # 检查pid和tid是否有效
            if event.pid is None or event.tid is None:
                continue
            
            # 尝试转换为整数
            try:
                pid = int(event.pid) if not isinstance(event.pid, int) else event.pid
                tid = int(event.tid) if not isinstance(event.tid, int) else event.tid
            except (ValueError, TypeError):
                continue
            
            events_by_pid_tid[(pid, tid)].append(event)
        
        # 对每个组内的事件按时间排序
        for (pid, tid), group_events in events_by_pid_tid.items():
            group_events.sort(key=lambda e: e.ts)
        
        self.logger.info(f"按pid/tid分组完成，共 {len(events_by_pid_tid)} 个组")
        return dict(events_by_pid_tid)
    
    def _build_event_mapping(self, root: CallStackNode):
        """
        构建事件到节点的映射
        
        Args:
            root: 调用栈树根节点
        """
        # 使用深度优先遍历构建映射
        stack = [root]
        
        while stack:
            current = stack.pop()
            
            # 将当前节点添加到映射中
            self.event_to_node_map[current.event_id] = current
            
            # 将子节点加入栈
            stack.extend(current.children)
    
    def _build_call_stack_tree(self, events: List[ActivityEvent]) -> Optional[CallStackNode]:
        """
        使用扫描线算法构建调用栈树
        
        Args:
            events: 已按时间排序的事件列表
            
        Returns:
            CallStackNode: 调用栈树的根节点
        """
        if not events:
            return None
        
        # 1. 创建事件区间
        intervals = []
        for i, event in enumerate(events):
            if event.dur is not None and event.dur > 0:
                interval = EventInterval(
                    event=event,
                    start=event.ts,
                    end=event.ts + event.dur,
                    index=i
                )
                intervals.append(interval)
        
        if not intervals:
            return None
        
        # 2. 按开始时间排序（稳定排序）
        intervals.sort(key=lambda x: (x.start, x.index))
        
        # 3. 构建线段树
        segment_tree = SegmentTree(intervals)
        
        # 4. 使用扫描线算法构建调用栈树
        return self._sweep_line_algorithm(intervals, segment_tree)
    
    def _sweep_line_algorithm(self, intervals: List[EventInterval], segment_tree: SegmentTree) -> CallStackNode:
        """
        扫描线算法构建调用栈树
        
        Args:
            intervals: 按开始时间排序的事件区间列表
            segment_tree: 线段树
            
        Returns:
            CallStackNode: 调用栈树的根节点
        """
        # 创建虚拟根节点
        root = CallStackNode(ActivityEvent(
            name="ROOT",
            cat="root",
            ph="X",
            pid=intervals[0].event.pid,
            tid=intervals[0].event.tid,
            ts=0.0,
            dur=0.0
        ))
        
        # 维护当前活跃的节点栈
        active_stack = [root]
        
        for interval in intervals:
            # 处理当前区间
            self._process_interval(interval, active_stack, segment_tree)
        
        return root
    
    def _process_interval(self, interval: EventInterval, active_stack: List[CallStackNode], segment_tree: SegmentTree):
        """
        处理单个事件区间
        
        Args:
            interval: 当前事件区间
            active_stack: 当前活跃的节点栈
            segment_tree: 线段树
        """
        # 1. 弹出所有已经结束的节点
        while len(active_stack) > 1:
            top_node = active_stack[-1]
            top_event = top_node.event
            
            # 检查顶部节点是否已经结束
            if top_event.dur is not None and top_event.ts + top_event.dur <= interval.start:
                active_stack.pop()
            else:
                break
        
        # 2. 查找当前区间的父节点
        parent = self._find_parent_node(interval, active_stack, segment_tree)
        
        # 3. 创建新节点并添加到树中
        new_node = CallStackNode(interval.event, parent)
        parent.add_child(new_node)
        
        # 4. 将新节点推入活跃栈
        active_stack.append(new_node)
    
    def _find_parent_node(self, interval: EventInterval, active_stack: List[CallStackNode], segment_tree: SegmentTree) -> CallStackNode:
        """
        为当前区间找到合适的父节点
        
        Args:
            interval: 当前事件区间
            active_stack: 当前活跃的节点栈
            segment_tree: 线段树
            
        Returns:
            CallStackNode: 父节点
        """
        # 从栈顶开始向下查找
        for i in range(len(active_stack) - 1, -1, -1):
            candidate = active_stack[i]
            candidate_event = candidate.event
            
            # 检查候选节点是否包含当前区间
            if self._contains(candidate_event, interval.event):
                return candidate
        
        # 如果没有找到包含的节点，返回根节点
        return active_stack[0]
    
    def _contains(self, parent_event: ActivityEvent, child_event: ActivityEvent) -> bool:
        """
        检查父事件是否包含子事件
        
        Args:
            parent_event: 父事件
            child_event: 子事件
            
        Returns:
            bool: 是否包含
        """
        if parent_event.dur is None or child_event.dur is None:
            return False
        
        parent_start = parent_event.ts
        parent_end = parent_event.ts + parent_event.dur
        child_start = child_event.ts
        child_end = child_event.ts + child_event.dur
        
        # 检查是否包含：父事件的开始时间 <= 子事件的开始时间 且 父事件的结束时间 >= 子事件的结束时间
        return parent_start <= child_start and parent_end >= child_end
    
    def get_call_stack_for_event(self, event: ActivityEvent, call_stack_trees: Dict[Tuple[int, int], CallStackNode]) -> Optional[List[str]]:
        """
        为指定事件获取调用栈
        
        Args:
            event: 目标事件
            call_stack_trees: 调用栈树字典
            
        Returns:
            Optional[List[str]]: 调用栈路径，如果找不到返回None
        """
        # 创建事件标识符
        event_id = f"{event.name}:{event.ts}:{event.dur}:{event.pid}:{event.tid}"
        
        # 直接从映射中查找节点
        target_node = self.event_to_node_map.get(event_id)
        if target_node:
            return target_node.get_call_stack()
        
        return None
    
    def _find_event_node(self, target_event: ActivityEvent, root: CallStackNode) -> Optional[CallStackNode]:
        """
        在调用栈树中查找包含指定事件的节点（已弃用，使用映射查找）
        
        Args:
            target_event: 目标事件
            root: 树根节点
            
        Returns:
            Optional[CallStackNode]: 找到的节点，如果找不到返回None
        """
        # 创建事件标识符
        event_id = f"{target_event.name}:{target_event.ts}:{target_event.dur}:{target_event.pid}:{target_event.tid}"
        
        # 直接从映射中查找
        return self.event_to_node_map.get(event_id)
    
    def print_call_stack_tree(self, root: CallStackNode, max_depth: int = 10):
        """
        打印调用栈树结构
        
        Args:
            root: 树根节点
            max_depth: 最大打印深度
        """
        def _print_node(node: CallStackNode, depth: int, prefix: str = ""):
            if depth > max_depth:
                return
            
            event = node.event
            print(f"{prefix}{event.name} (ts={event.ts:.2f}, dur={event.dur:.2f})")
            
            for i, child in enumerate(node.children):
                is_last = i == len(node.children) - 1
                child_prefix = prefix + ("└── " if is_last else "├── ")
                _print_node(child, depth + 1, child_prefix)
        
        _print_node(root, 0)
    
    def get_tree_statistics(self, call_stack_trees: Dict[Tuple[int, int], CallStackNode]) -> Dict[str, Any]:
        """
        获取调用栈树的统计信息
        
        Args:
            call_stack_trees: 调用栈树字典
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'total_trees': len(call_stack_trees),
            'total_nodes': 0,
            'max_depth': 0,
            'avg_depth': 0.0,
            'tree_sizes': []
        }
        
        total_depth = 0
        
        for (pid, tid), root in call_stack_trees.items():
            tree_size = self._count_nodes(root)
            tree_depth = self._get_tree_depth(root)
            
            stats['total_nodes'] += tree_size
            stats['max_depth'] = max(stats['max_depth'], tree_depth)
            total_depth += tree_depth
            stats['tree_sizes'].append({
                'pid': pid,
                'tid': tid,
                'size': tree_size,
                'depth': tree_depth
            })
        
        if stats['total_trees'] > 0:
            stats['avg_depth'] = total_depth / stats['total_trees']
        
        return stats
    
    def _count_nodes(self, root: CallStackNode) -> int:
        """计算树中的节点数"""
        count = 1  # 包含根节点
        for child in root.children:
            count += self._count_nodes(child)
        return count
    
    def _get_tree_depth(self, root: CallStackNode) -> int:
        """获取树的深度"""
        if not root.children:
            return 0
        
        max_child_depth = 0
        for child in root.children:
            max_child_depth = max(max_child_depth, self._get_tree_depth(child))
        
        return max_child_depth + 1

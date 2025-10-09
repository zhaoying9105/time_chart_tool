"""
数据聚合阶段
"""

from collections import defaultdict
from typing import Dict, List, Union

from ...models import ActivityEvent
from ..utils.data_structures import AggregatedData
from ..utils.call_stack_utils import CallStackWrapper


class DataAggregator:
    """数据聚合器"""
    
    def __init__(self):
        pass
    
    def _generate_aggregation_key(self, cpu_event: ActivityEvent, aggregation_fields: List[str], 
                                 call_stack_source: str = 'tree') -> Union[str, tuple]:
        """
        根据指定的字段生成聚合键
        
        Args:
            cpu_event: CPU事件
            aggregation_fields: 聚合字段列表，支持: call_stack, name, shape, dtype
            call_stack_source: 调用栈来源，'args' 或 'tree'
            
        Returns:
            Union[str, tuple]: 聚合键
        """
        key_parts = []
        
        for field in aggregation_fields:
            if field == 'call_stack':
                call_stack = cpu_event.get_call_stack(call_stack_source)
                
                if call_stack is not None:
                    normalized_call_stack = self._normalize_call_stack(call_stack)
                    if normalized_call_stack.call_stack:
                        key_parts.append(normalized_call_stack)
                    else:
                        key_parts.append(None)
                else:
                    key_parts.append(None)
            elif field == 'name':
                key_parts.append(cpu_event.name)
            elif field == 'shape':
                args = cpu_event.args or {}
                input_dims = args.get('Input Dims', [])
                # 递归地将嵌套列表转换为元组，使其可哈希
                def list_to_tuple(obj):
                    if isinstance(obj, list):
                        return tuple(list_to_tuple(item) for item in obj)
                    elif isinstance(obj, (int, float, str, bool)) or obj is None:
                        return obj
                    else:
                        return str(obj)  # 对于其他类型，转换为字符串
                key_parts.append(list_to_tuple(input_dims))
            elif field == 'dtype':
                args = cpu_event.args or {}
                dtype = args.get('Input type', 'unknown')
                # 确保dtype是可哈希的
                if isinstance(dtype, (list, dict)):
                    dtype = str(dtype)
                key_parts.append(dtype)
            elif field == 'op_index':
                # op_index需要特殊处理，这里先返回None，后续在聚合时处理
                key_parts.append(None)
            else:
                raise ValueError(f"不支持的聚合字段: {field}")
        
        # 如果只有一个字段且不是None，直接返回该字段
        if len(key_parts) == 1 and key_parts[0] is not None:
            return key_parts[0]
        
        # 否则返回元组
        return tuple(key_parts)
    
    def _parse_aggregation_fields(self, aggregation_spec: str) -> List[str]:
        """
        解析聚合字段组合
        
        Args:
            aggregation_spec: 聚合字段组合字符串，如 "name,shape" 或 "call_stack,name"
            
        Returns:
            List[str]: 聚合字段列表
        """
        # 解析字段组合：逗号分隔的字段
        if ',' in aggregation_spec:
            fields = [field.strip() for field in aggregation_spec.split(',')]
        else:
            fields = [aggregation_spec.strip()]
        
        # 验证字段
        valid_fields = {'call_stack', 'name', 'shape', 'dtype', 'op_index'}
        for field in fields:
            if field not in valid_fields:
                raise ValueError(f"不支持的聚合字段: {field}。支持的字段: {', '.join(valid_fields)}")
        
        return fields
    
    def stage2_data_aggregation(self, cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                               kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                               aggregation_spec: str = 'name', call_stack_source: str = 'tree') -> Dict[Union[str, tuple], AggregatedData]:
        """
        Stage 2: 数据聚合
        支持灵活的字段组合聚合：
        - 支持的字段: call_stack, name, shape, dtype
        - 使用逗号分隔的字段组合，如 "name,shape" 或 "call_stack,name,dtype"
        
        Args:
            cpu_events_by_external_id: external_id -> cpu_events 映射
            kernel_events_by_external_id: external_id -> kernel_events 映射
            aggregation_spec: 聚合字段组合字符串
            call_stack_source: 调用栈来源，'args' 或 'tree'
            
        Returns:
            Dict[Union[str, tuple], AggregatedData]: 聚合后的数据
        """
        print(f"=== Stage 2: 数据聚合 ({aggregation_spec}) ===")
        
        # 解析聚合字段组合
        aggregation_fields = self._parse_aggregation_fields(aggregation_spec)
        
        if 'call_stack' in aggregation_fields:
            # 包含调用栈的聚合需要特殊处理，因为需要合并相似的调用栈
            return self._aggregate_with_call_stack(cpu_events_by_external_id, kernel_events_by_external_id, aggregation_fields, call_stack_source)
        
        if 'op_index' in aggregation_fields:
            # 包含op_index的聚合需要特殊处理，按时间顺序排列
            return self._aggregate_with_op_index(cpu_events_by_external_id, kernel_events_by_external_id, aggregation_fields)
        
        # 普通聚合处理
        aggregated_data = defaultdict(lambda: AggregatedData([], [], ""))
        
        for external_id in cpu_events_by_external_id:
            cpu_events = cpu_events_by_external_id[external_id]
            kernel_events = kernel_events_by_external_id.get(external_id, [])
            
            for cpu_event in cpu_events:
                try:
                    key = self._generate_aggregation_key(cpu_event, aggregation_fields, call_stack_source)
                    aggregated_data[key].cpu_events.append(cpu_event)
                    aggregated_data[key].kernel_events.extend(kernel_events)
                    aggregated_data[key].key = key
                except Exception as e:
                    print(f"警告: 跳过事件 {cpu_event.name}，错误: {e}")
                    continue
        
        print(f"聚合后得到 {len(aggregated_data)} 个不同的键")
        
        # 调试：统计跳过的事件数量
        total_events = sum(len(cpu_events_by_external_id[ext_id]) for ext_id in cpu_events_by_external_id)
        valid_events = 0
        skipped_events = 0
        for external_id in cpu_events_by_external_id:
            cpu_events = cpu_events_by_external_id[external_id]
            for cpu_event in cpu_events:
                call_stack = cpu_event.get_call_stack(call_stack_source)
                if call_stack is None:
                    skipped_events += 1
                else:
                    valid_events += 1
        
        print(f"DEBUG: 总事件数={total_events}, 有效事件数={valid_events}, 跳过事件数={skipped_events}")
        
        return dict(aggregated_data)
    
    def _aggregate_with_op_index(self, cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                                kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                                aggregation_fields: List[str]) -> Dict[Union[str, tuple], AggregatedData]:
        """
        处理包含op_index的聚合，按执行顺序(ts从小到大)排列cpu_op
        
        Args:
            cpu_events_by_external_id: external_id -> cpu_events 映射
            kernel_events_by_external_id: external_id -> kernel_events 映射
            aggregation_fields: 聚合字段列表
            
        Returns:
            Dict[Union[str, tuple], AggregatedData]: 聚合后的数据
        """
        print("=== 按op_index聚合，按执行顺序排列 ===")
        
        # 收集所有CPU事件并按时间排序
        all_cpu_events = []
        for external_id in cpu_events_by_external_id:
            cpu_events = cpu_events_by_external_id[external_id]
            all_cpu_events.extend(cpu_events)
        
        # 按启动时间排序
        all_cpu_events.sort(key=lambda x: x.ts if x.ts is not None else 0)
        
        print(f"找到 {len(all_cpu_events)} 个CPU操作，按启动时间排序")
        
        # 创建聚合数据，每个操作按索引排列
        aggregated_data = {}
        
        for i, cpu_event in enumerate(all_cpu_events):
            # 生成聚合键，将op_index替换为实际的索引
            key_parts = []
            for field in aggregation_fields:
                if field == 'op_index':
                    key_parts.append(i)  # 使用实际的操作索引
                elif field == 'call_stack':
                    if cpu_event.call_stack is not None:
                        normalized_call_stack = self._normalize_call_stack(cpu_event.call_stack)
                        if normalized_call_stack:
                            key_parts.append(tuple(normalized_call_stack))
                        else:
                            key_parts.append(None)
                    else:
                        key_parts.append(None)
                elif field == 'name':
                    key_parts.append(cpu_event.name)
                elif field == 'shape':
                    args = cpu_event.args or {}
                    input_dims = args.get('Input Dims', [])
                    def list_to_tuple(obj):
                        if isinstance(obj, list):
                            return tuple(list_to_tuple(item) for item in obj)
                        elif isinstance(obj, (int, float, str, bool)) or obj is None:
                            return obj
                        else:
                            return str(obj)
                    key_parts.append(list_to_tuple(input_dims))
                elif field == 'dtype':
                    args = cpu_event.args or {}
                    dtype = args.get('Input type', 'unknown')
                    if isinstance(dtype, (list, dict)):
                        dtype = str(dtype)
                    key_parts.append(dtype)
            
            # 生成键
            if len(key_parts) == 1 and key_parts[0] is not None:
                key = key_parts[0]
            else:
                key = tuple(key_parts)
            
            # 找到对应的kernel事件
            kernel_events = []
            if cpu_event.external_id is not None:
                kernel_events = kernel_events_by_external_id.get(cpu_event.external_id, [])
            
            # 创建聚合数据条目
            aggregated_data[key] = AggregatedData(
                cpu_events=[cpu_event],
                kernel_events=kernel_events,
                key=key
            )
        
        print(f"聚合后得到 {len(aggregated_data)} 个按索引排序的操作")
        
        return aggregated_data
    
    def _aggregate_with_call_stack(self, cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                                  kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                                  aggregation_fields: List[str], call_stack_source: str = 'tree') -> Dict[Union[str, tuple], AggregatedData]:
        """
        处理包含调用栈的聚合，实现 startswith 合并逻辑
        
        Args:
            cpu_events_by_external_id: external_id -> cpu_events 映射
            kernel_events_by_external_id: external_id -> kernel_events 映射
            aggregation_fields: 聚合字段列表
            call_stack_source: 调用栈来源，'args' 或 'tree'
            
        Returns:
            Dict[Union[str, tuple], AggregatedData]: 聚合后的数据
        """
        aggregated_data = defaultdict(lambda: AggregatedData([], [], ""))
        
        for external_id in cpu_events_by_external_id:
            cpu_events = cpu_events_by_external_id[external_id]
            kernel_events = kernel_events_by_external_id.get(external_id, [])
            
            for cpu_event in cpu_events:
                # 检查是否有调用栈信息
                call_stack = cpu_event.get_call_stack(call_stack_source)
                
                # 调试输出：显示前几个事件的处理情况
                if len(aggregated_data) < 5:  # 增加调试事件数量
                    call_stack_args = cpu_event.get_call_stack('args')
                    call_stack_tree = cpu_event.get_call_stack('tree')
                    # print(f"DEBUG _aggregate_with_call_stack: 事件 {cpu_event.name}, external_id={cpu_event.external_id}")
                    # print(f"DEBUG _aggregate_with_call_stack: call_stack_source={call_stack_source}")
                    # print(f"DEBUG _aggregate_with_call_stack: call_stack(from args)长度={len(call_stack_args) if call_stack_args else 0}")
                    # print(f"DEBUG _aggregate_with_call_stack: call_stack(from tree)长度={len(call_stack_tree) if call_stack_tree else 0}")
                    # print(f"DEBUG _aggregate_with_call_stack: current call_stack长度={len(call_stack) if call_stack else 0}")
                    # if call_stack:
                    #     print(f"DEBUG _aggregate_with_call_stack: current call_stack内容:")
                    #     for i, frame in enumerate(call_stack):
                    #         print(f"  [{i}] {frame}")
                
                if call_stack is None:
                    continue  # 跳过没有 call stack 的事件
                
                try:
                    new_key = self._generate_aggregation_key(cpu_event, aggregation_fields, call_stack_source)
                    
                    # 检查是否已有相似的调用栈（使用startswith判断）
                    existing_key = None
                    for existing_key_candidate in aggregated_data.keys():
                        if self._is_similar_call_stack_key(new_key, existing_key_candidate, aggregation_fields):
                            # 保留较长的调用栈
                            if self._should_keep_new_key(new_key, existing_key_candidate, aggregation_fields):
                                existing_key = new_key
                            else:
                                existing_key = existing_key_candidate
                            break
                    
                    if existing_key:
                        # 合并到现有键
                        aggregated_data[existing_key].cpu_events.append(cpu_event)
                        aggregated_data[existing_key].kernel_events.extend(kernel_events)
                    else:
                        # 创建新键
                        aggregated_data[new_key] = AggregatedData(
                            cpu_events=[cpu_event],
                            kernel_events=kernel_events,
                            key=new_key
                        )
                except Exception as e:
                    print(f"警告: 跳过事件 {cpu_event.name}，错误: {e}")
                    continue
        
        print(f"聚合后得到 {len(aggregated_data)} 个不同的键")
        
        return dict(aggregated_data)
    
    def _is_similar_call_stack_key(self, key1: Union[str, tuple], key2: Union[str, tuple], aggregation_fields: List[str]) -> bool:
        """
        判断两个键是否相似（调用栈部分使用startswith判断）
        
        Args:
            key1: 第一个键
            key2: 第二个键
            aggregation_fields: 聚合字段列表
            
        Returns:
            bool: 是否相似
        """
        # 确保键是元组格式
        if not isinstance(key1, tuple):
            key1 = (key1,)
        if not isinstance(key2, tuple):
            key2 = (key2,)
        
        # 检查非调用栈字段是否相同
        for i, field in enumerate(aggregation_fields):
            if field != 'call_stack':
                if i < len(key1) and i < len(key2) and key1[i] != key2[i]:
                    return False
        
        # 检查调用栈字段
        call_stack_idx = aggregation_fields.index('call_stack') if 'call_stack' in aggregation_fields else -1
        if call_stack_idx >= 0 and call_stack_idx < len(key1) and call_stack_idx < len(key2):
            call_stack1 = key1[call_stack_idx]
            call_stack2 = key2[call_stack_idx]
            
            if call_stack1 is None or call_stack2 is None:
                return False
            
            # 将调用栈转换为字符串进行比较
            call_stack1_str = str(call_stack1)
            call_stack2_str = str(call_stack2)
            
            return (call_stack1_str.startswith(call_stack2_str) or 
                    call_stack2_str.startswith(call_stack1_str))
        
        return False
    
    def _should_keep_new_key(self, new_key: Union[str, tuple], existing_key: Union[str, tuple], aggregation_fields: List[str]) -> bool:
        """
        判断是否应该保留新键（保留较长的调用栈）
        
        Args:
            new_key: 新键
            existing_key: 现有键
            aggregation_fields: 聚合字段列表
            
        Returns:
            bool: 是否保留新键
        """
        call_stack_idx = aggregation_fields.index('call_stack') if 'call_stack' in aggregation_fields else -1
        if call_stack_idx < 0:
            return False
        
        # 确保键是元组格式
        if not isinstance(new_key, tuple):
            new_key = (new_key,)
        if not isinstance(existing_key, tuple):
            existing_key = (existing_key,)
        
        if (call_stack_idx < len(new_key) and call_stack_idx < len(existing_key) and
            new_key[call_stack_idx] is not None and existing_key[call_stack_idx] is not None):
            
            new_call_stack = new_key[call_stack_idx]
            existing_call_stack = existing_key[call_stack_idx]
            
            # 获取调用栈长度
            new_len = len(new_call_stack.call_stack)
            existing_len = len(existing_call_stack.call_stack)
            
            return new_len > existing_len
        
        return False
    
    def _normalize_call_stack(self, call_stack: List[str]) -> List[str]:
        """
        标准化 call stack，使用公共工具函数
        
        Args:
            call_stack: 原始 call stack
            
        Returns:
            List[str]: 标准化后的 call stack
        """
        from ..utils import normalize_call_stack
        return normalize_call_stack(call_stack)

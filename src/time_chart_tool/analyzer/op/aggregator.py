"""
数据聚合阶段 (纯函数实现)

目标：降低认知负担，将复杂聚合流程拆分为若干职责单一的纯函数；
并在关键步骤提供详细中文注释以便维护和扩展。
"""

from collections import defaultdict
from typing import Dict, List, Union, Tuple, Optional, Any
import logging

from ...models import ActivityEvent
from ..utils.data_structures import AggregatedData
from ..utils.call_stack_utils import normalize_call_stack

logger = logging.getLogger(__name__)


def _list_to_tuple(obj) -> Any:
    """
    将可能嵌套的列表转换为元组，使其在作为聚合键的一部分时具备可哈希性。

    说明：
    - 部分事件的 `Input Dims`、`dtype` 等属性可能是列表或嵌套结构；
    - 为了将这些属性参与到聚合键的生成，需要统一转换为可哈希的、不可变的类型（元组）。
    """
    if isinstance(obj, list):
        return tuple(_list_to_tuple(item) for item in obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        # 对于其他非基础类型，转换为字符串以保证可哈希性
        return str(obj)


def data_aggregation(cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                           kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                           aggregation_spec: List[str] = ['name'], 
                           coarse_call_stack: bool = False) -> Dict[Union[str, tuple], AggregatedData]:
    """
    数据聚合的入口函数（纯函数）。

    功能概述：
    - 根据 `aggregation_spec` 指定的字段组合，生成聚合键并将相关事件合并到同一分组；
    - 对包含特殊字段（如 `call_stack`、`op_index`）的聚合采用专用流程；
    - 对普通字段（如 `name`、`shape`、`dtype`、`fwd_bwd_type`）采用通用流程。

    参数：
    - `cpu_events_by_external_id`: external_id -> cpu_events 映射
    - `kernel_events_by_external_id`: external_id -> kernel_events 映射
    - `aggregation_spec`: 聚合字段列表（支持：call_stack, name, shape, dtype, op_index, fwd_bwd_type）
    - `coarse_call_stack`: 是否使用粗糙的调用栈匹配（将部分细节归一化）

    返回：
    - `Dict[Union[str, tuple], AggregatedData]`: 聚合结果，键为聚合键，值为聚合后的 CPU 与 Kernel 事件集合。
    """
    print(f"=== Stage 2: 数据聚合 ({aggregation_spec}) ===")
    
    # 验证聚合字段
    _validate_aggregation_fields(aggregation_spec)
    
    if 'call_stack' in aggregation_spec:
        # 包含调用栈的聚合需要特殊处理，因为需要合并相似的调用栈
        print(f"使用调用栈聚合， coarse={coarse_call_stack}")
        return _aggregate_with_call_stack(cpu_events_by_external_id, kernel_events_by_external_id, 
                                        aggregation_spec, coarse_call_stack)
    
    if 'op_index' in aggregation_spec:
        # 包含op_index的聚合需要特殊处理，按时间顺序排列
        return _aggregate_with_op_index(cpu_events_by_external_id, kernel_events_by_external_id, 
                                      aggregation_spec, coarse_call_stack)
    
    # 普通聚合处理
    aggregated_data = defaultdict(lambda: AggregatedData([], [], ""))
    
    for external_id in cpu_events_by_external_id:
        cpu_events = cpu_events_by_external_id[external_id]
        kernel_events = kernel_events_by_external_id.get(external_id, [])
        
        for cpu_event in cpu_events:
            try:
                key = _generate_aggregation_key(cpu_event, aggregation_spec, coarse_call_stack)
                aggregated_data[key].cpu_events.append(cpu_event)
                aggregated_data[key].kernel_events.extend(kernel_events)
                aggregated_data[key].key = key
            except Exception as e:
                logger.warning(f"警告: 跳过事件 {cpu_event.name}，错误: {e}")
                continue
    
    print(f"聚合后得到 {len(aggregated_data)} 个不同的键")
    
    # 调试：统计跳过的事件数量
    total_events = sum(len(cpu_events_by_external_id[ext_id]) for ext_id in cpu_events_by_external_id)
    valid_events = 0
    skipped_events = 0
    for external_id in cpu_events_by_external_id:
        cpu_events = cpu_events_by_external_id[external_id]
        for cpu_event in cpu_events:
            call_stack = cpu_event.call_stack
            if call_stack is None:
                skipped_events += 1
            else:
                valid_events += 1
    
    print(f"DEBUG: 总事件数={total_events}, 有效事件数={valid_events}, 跳过事件数={skipped_events}")
    
    return dict(aggregated_data)


def _generate_aggregation_key(cpu_event: ActivityEvent, aggregation_fields: List[str], 
                             coarse_call_stack: bool = False) -> Union[str, tuple]:
    """
    根据指定字段为单个 CPU 事件生成聚合键（纯函数）。

    设计要点：
    - 将调用栈、名称、输入形状、数据类型、前向/反向类型等信息组合为键；
    - 当只有一个有效字段时直接返回该字段，否则返回由多字段组成的元组；
    - 对 `shape` 等可能为列表/嵌套结构的字段，统一转换为元组以保证可哈希性。
    """
    key_parts = []
    
    for field in aggregation_fields:
        if field == 'call_stack':
            call_stack = cpu_event.call_stack
            
            if call_stack is not None:
                normalized_call_stack_wrapper = normalize_call_stack(call_stack, coarse_call_stack=coarse_call_stack)
                if normalized_call_stack_wrapper.call_stack:
                    key_parts.append(normalized_call_stack_wrapper)
                else:
                    key_parts.append(None)
            else:
                key_parts.append(None)
        elif field == 'name':
            key_parts.append(cpu_event.name)
        elif field == 'shape':
            args = cpu_event.args or {}
            input_dims = args.get('Input Dims', [])
            key_parts.append(_list_to_tuple(input_dims))
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
        elif field == 'fwd_bwd_type':
            # 获取 fwd_bwd_type 属性
            fwd_bwd_type = getattr(cpu_event, 'fwd_bwd_type', 'none')
            key_parts.append(fwd_bwd_type)
        else:
            raise ValueError(f"不支持的聚合字段: {field}")
    
    # 如果只有一个字段且不是None，直接返回该字段
    if len(key_parts) == 1 and key_parts[0] is not None:
        return key_parts[0]
    
    # 否则返回元组
    return tuple(key_parts)


def _validate_aggregation_fields(aggregation_spec: List[str]) -> None:
    """
    验证聚合字段组合（纯函数）。

    校验规则：
    - 仅允许使用集合 {call_stack, name, shape, dtype, op_index, fwd_bwd_type} 中的字段；
    - 如包含未支持字段，抛出错误以提示调用方修正配置。
    """
    valid_fields = {'call_stack', 'name', 'shape', 'dtype', 'op_index', 'fwd_bwd_type'}
    for field in aggregation_spec:
        if field not in valid_fields:
            raise ValueError(f"不支持的聚合字段: {field}。支持的字段: {', '.join(valid_fields)}")


def _aggregate_with_op_index(cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                            kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                            aggregation_fields: List[str], coarse_call_stack: bool = False) -> Dict[Union[str, tuple], AggregatedData]:
    """
    含 `op_index` 的聚合专用流程（纯函数）。

    策略说明：
    - 将所有 CPU 事件按启动时间 `ts` 升序排序，生成全局执行顺序索引；
    - 在聚合键中用实际索引替换 `op_index` 字段，其他字段按常规处理；
    - 将关联的 Kernel 事件一并收集，形成有序的聚合结果。
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
                    normalized_call_stack_wrapper = normalize_call_stack(cpu_event.call_stack, coarse_call_stack=coarse_call_stack)
                    if normalized_call_stack_wrapper.call_stack:
                        key_parts.append(tuple(normalized_call_stack_wrapper.call_stack))
                    else:
                        key_parts.append(None)
                else:
                    key_parts.append(None)
            elif field == 'name':
                key_parts.append(cpu_event.name)
            elif field == 'shape':
                args = cpu_event.args or {}
                input_dims = args.get('Input Dims', [])
                key_parts.append(_list_to_tuple(input_dims))
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


def _aggregate_with_call_stack(cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                              kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                              aggregation_fields: List[str], 
                              coarse_call_stack: bool = False) -> Dict[Union[str, tuple], AggregatedData]:
    """
    含调用栈的聚合专用流程（纯函数）。

    策略说明：
    - 对每个 CPU 事件生成聚合键（包含调用栈）；
    - 若发现已有键的调用栈是新键调用栈的前缀（或相反），则进行“相似键合并”；
    - 合并时保留调用栈更“长”的键，以提高代表性和稳定性；
    - 对于缺失调用栈的事件，记录并跳过。
    """
    aggregated_data = defaultdict(lambda: AggregatedData([], [], ""))
    
    skipped_no_stack = 0
    processed_events = 0
    
    for external_id in cpu_events_by_external_id:
        cpu_events = cpu_events_by_external_id[external_id]
        kernel_events = kernel_events_by_external_id.get(external_id, [])
        
        for cpu_event in cpu_events:
            processed_events += 1
            # 检查是否有调用栈信息
            call_stack = cpu_event.call_stack
            
            if call_stack is None:
                skipped_no_stack += 1
                if skipped_no_stack <= 5: # 仅打印前5个跳过事件的详细信息，避免日志过多
                     logger.debug(f"事件 {cpu_event.name} (ID: {cpu_event.external_id}) 缺少调用栈，跳过。Args: {cpu_event.args}")
                continue  # 跳过没有 call stack 的事件
            
            try:
                new_key = _generate_aggregation_key(cpu_event, aggregation_fields, coarse_call_stack)

                # 查找相似键，并根据调用栈长度选择保留新键或旧键
                existing_key = _choose_key_for_merge(new_key, list(aggregated_data.keys()), aggregation_fields)
                
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
                logger.warning(f"警告: 跳过事件 {cpu_event.name}，错误: {e}")
                continue
    
    print(f"聚合后得到 {len(aggregated_data)} 个不同的键")
    print(f"处理了 {processed_events} 个事件，跳过 {skipped_no_stack} 个无调用栈事件")
    
    return dict(aggregated_data)


def _is_similar_call_stack_key(key1: Union[str, tuple], key2: Union[str, tuple], aggregation_fields: List[str]) -> bool:
    """
    判断两个聚合键是否相似（纯函数）。

    判定规则：
    - 非调用栈字段必须完全相同；
    - 调用栈字段允许“前缀关系”（使用字符串 startswith 判断）。
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


def _should_keep_new_key(new_key: Union[str, tuple], existing_key: Union[str, tuple], aggregation_fields: List[str]) -> bool:
    """
    选择保留新键还是旧键（纯函数）。

    策略说明：
    - 调用栈越长代表信息越完整，优先保留调用栈更长的键；
    - 若任一调用栈为空或位置无效，则默认不保留新键。
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


def _choose_key_for_merge(new_key: Union[str, tuple], existing_keys: List[Union[str, tuple]], aggregation_fields: List[str]) -> Optional[Union[str, tuple]]:
    """
    在现有键集合中查找与新键相似的键，并根据调用栈长度决定保留哪个键（纯函数）。

    返回：
    - 若找到相似键，返回最终用于合并的键（可能是 new_key 或某个 existing_key）；
    - 若未找到相似键，返回 None。
    """
    for candidate in existing_keys:
        if _is_similar_call_stack_key(new_key, candidate, aggregation_fields):
            # 根据调用栈长度选择保留新键或旧键
            return new_key if _should_keep_new_key(new_key, candidate, aggregation_fields) else candidate
    return None

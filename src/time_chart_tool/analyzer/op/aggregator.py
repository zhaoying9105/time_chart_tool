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
                           aggregation_spec: List[str] = ['name']) -> Dict[Union[str, tuple], AggregatedData]:
    """
    数据聚合的入口函数（纯函数）。

    功能概述：
    - 根据 `aggregation_spec` 指定的字段组合，生成聚合键并将相关事件合并到同一分组；
    - 对包含特殊字段（如 `call_stack`）的聚合采用专用流程；
    - 对普通字段（如 `name`、`shape`、`dtype`、`fwd_bwd_type`）采用通用流程。

    参数：
    - `cpu_events_by_external_id`: external_id -> cpu_events 映射
    - `kernel_events_by_external_id`: external_id -> kernel_events 映射
    - `aggregation_spec`: 聚合字段列表（支持：call_stack, name, shape, dtype, fwd_bwd_type, pid, tid）

    返回：
    - `Dict[Union[str, tuple], AggregatedData]`: 聚合结果，键为聚合键，值为聚合后的 CPU 与 Kernel 事件集合。
    """
    print(f"=== Stage 2: 数据聚合 ({aggregation_spec}) ===")
    
    # 验证聚合字段
    _validate_aggregation_fields(aggregation_spec)
    
    if 'call_stack' in aggregation_spec:
        # 包含调用栈的聚合需要特殊处理，因为需要合并相似的调用栈
        print(f"使用调用栈聚合")
        return _aggregate_with_call_stack(cpu_events_by_external_id, kernel_events_by_external_id, 
                                        aggregation_spec)
    
    # 普通聚合处理
    aggregated_data = defaultdict(lambda: AggregatedData([], [], ""))
    
    for external_id in cpu_events_by_external_id:
        cpu_events = cpu_events_by_external_id[external_id]
        kernel_events = kernel_events_by_external_id.get(external_id, [])
        
        for cpu_event in cpu_events:
            try:
                key = _generate_aggregation_key(cpu_event, aggregation_spec)
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


def _generate_aggregation_key(cpu_event: ActivityEvent, aggregation_fields: List[str]) -> Union[str, tuple, int]:
    """
    根据指定字段为单个 CPU 事件生成聚合键（纯函数）。

    设计要点：
    - 将调用栈、名称、输入形状、数据类型、前向/反向类型等信息组合为键；
    - 当只有一个有效字段时直接使用该字段，否则返回由多字段组成的元组；
    - 对 `shape` 等可能为列表/嵌套结构的字段，统一转换为元组以保证可哈希性。
    """
    key_parts = []
    
    for field in aggregation_fields:
        if field == 'call_stack':
            call_stack = cpu_event.call_stack
            
            if call_stack is not None:
                # 不再进行 normalize_call_stack，直接使用 tuple(call_stack)
                # 假设 call_stack 是 list of strings
                key_parts.append(tuple(call_stack))
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
        elif field == 'fwd_bwd_type':
            # 获取 fwd_bwd_type 属性
            fwd_bwd_type = getattr(cpu_event, 'fwd_bwd_type', 'none')
            key_parts.append(fwd_bwd_type)
        elif field == 'pid':
            key_parts.append(getattr(cpu_event, 'pid', 'none'))
        elif field == 'tid':
            key_parts.append(getattr(cpu_event, 'tid', 'none'))
        elif field == 'op_index':
            key_parts.append(getattr(cpu_event, 'op_index', None))
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
    - 仅允许使用集合 {call_stack, name, shape, dtype, fwd_bwd_type, pid, tid, op_index} 中的字段；
    - 如包含未支持字段，抛出错误以提示调用方修正配置。
    - op_index 必须独占，不能与其他字段组合。
    """
    valid_fields = {'call_stack', 'name', 'shape', 'dtype', 'fwd_bwd_type', 'pid', 'tid', 'op_index'}
    
    # 检查字段有效性
    for field in aggregation_spec:
        if field not in valid_fields:
            raise ValueError(f"不支持的聚合字段: {field}。支持的字段: {', '.join(valid_fields)}")
            
    # 检查 op_index 约束
    if 'op_index' in aggregation_spec and len(aggregation_spec) > 1:
        raise ValueError("当使用 op_index 聚合时，不能同时使用其他聚合字段")


def _aggregate_with_call_stack(cpu_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                              kernel_events_by_external_id: Dict[Union[int, str], List[ActivityEvent]], 
                              aggregation_fields: List[str]) -> Dict[Union[str, tuple], AggregatedData]:
    """
    含调用栈的聚合专用流程（纯函数）。

    策略说明：
    1. 先进行精确匹配聚合（完全相同的调用栈视为一组）。
    2. 再对聚合后的结果进行后处理合并：若发现已有键的调用栈是新键调用栈的前缀（或相反），则进行“相似键合并”。
    """
    # 1. 精确聚合
    exact_aggregated_data = defaultdict(lambda: AggregatedData([], [], ""))
    skipped_no_stack = 0
    processed_events = 0

    for external_id in cpu_events_by_external_id:
        cpu_events = cpu_events_by_external_id[external_id]
        kernel_events = kernel_events_by_external_id.get(external_id, [])
        
        for cpu_event in cpu_events:
            processed_events += 1
            call_stack = cpu_event.call_stack
            
            if call_stack is None:
                skipped_no_stack += 1
                if skipped_no_stack <= 5: 
                     logger.debug(f"事件 {cpu_event.name} (ID: {cpu_event.external_id}) 缺少调用栈，跳过。Args: {cpu_event.args}")
                continue 
            
            try:
                # 生成精确的聚合键
                key = _generate_aggregation_key(cpu_event, aggregation_fields)
                exact_aggregated_data[key].cpu_events.append(cpu_event)
                exact_aggregated_data[key].kernel_events.extend(kernel_events)
                exact_aggregated_data[key].key = key
            except Exception as e:
                logger.warning(f"警告: 跳过事件 {cpu_event.name}，错误: {e}")
                continue
    
    print(f"精确聚合后得到 {len(exact_aggregated_data)} 个不同的键")
    print(f"处理了 {processed_events} 个事件，跳过 {skipped_no_stack} 个无调用栈事件")

    # 2. 后处理合并 (相似键合并)
    final_aggregated_data = {}
    
    # 将精确聚合的结果按 key 排序，保证处理顺序的稳定性
    sorted_keys = sorted(exact_aggregated_data.keys(), key=lambda k: str(k))
    
    for key in sorted_keys:
        data = exact_aggregated_data[key]
        
        # 尝试合并到已有的 final_aggregated_data 中
        existing_key = _choose_key_for_merge(key, list(final_aggregated_data.keys()), aggregation_fields)
        
        if existing_key:
            # 合并到现有键
            # 注意：这里我们要决定是用 existing_key 还是 current key 作为合并后的 key
            # _choose_key_for_merge 返回的是最终应该保留的 key
            
            target_key = existing_key
            source_key = key if target_key != key else None # 这里逻辑有点绕，existing_key 返回的是 final_aggregated_data 中的键
            
            # 如果返回的 existing_key 在 final_aggregated_data 中，说明我们要把 data 合并进去
            # 如果 existing_key 就是 key 本身（实际上 _choose_key_for_merge 现在的逻辑是返回 'best key'）
            # 让我们回顾一下 _choose_key_for_merge 的逻辑：
            # 它遍历 existing_keys，如果找到相似的，返回 new_key (if longer) or candidate (if longer)
            # 这里 new_key 是当前的 key, candidate 是 final_aggregated_data 中的某个 key
            
            # 如果返回的是 key (当前键)，说明当前键比已有的相似键更好 -> 替换
            # 如果返回的是 candidate (已有键)，说明已有键更好 -> 合并进去
            
            if target_key == key:
                # 这种情况比较复杂，因为 target_key 不在 final_aggregated_data 中，而是一个新的更优键
                # 这意味着我们需要找到那个"被比较"的 candidate，把它替换掉
                # 但 _choose_key_for_merge 只返回了结果键，没告诉我是哪个 candidate
                # 所以我们需要稍微修改一下逻辑或者重新查找
                
                # 简化逻辑：我们直接遍历 final_aggregated_data 的 keys
                merged = False
                candidates = list(final_aggregated_data.keys())
                for candidate in candidates:
                     if _is_similar_call_stack_key(key, candidate, aggregation_fields):
                        if _should_keep_new_key(key, candidate, aggregation_fields):
                            # 当前 key 更好，替换 candidate
                            # 取出 candidate 的数据
                            candidate_data = final_aggregated_data.pop(candidate)
                            # 合并数据到 key
                            data.cpu_events.extend(candidate_data.cpu_events)
                            data.kernel_events.extend(candidate_data.kernel_events)
                            final_aggregated_data[key] = data
                        else:
                            # candidate 更好，合并到 candidate
                            final_aggregated_data[candidate].cpu_events.extend(data.cpu_events)
                            final_aggregated_data[candidate].kernel_events.extend(data.kernel_events)
                        merged = True
                        break
                
                if not merged:
                    final_aggregated_data[key] = data
            else:
                # 返回的是 candidate，即 target_key 在 final_aggregated_data 中
                final_aggregated_data[target_key].cpu_events.extend(data.cpu_events)
                final_aggregated_data[target_key].kernel_events.extend(data.kernel_events)
        else:
            # 没有找到相似键，直接添加
            final_aggregated_data[key] = data
            
    print(f"合并后得到 {len(final_aggregated_data)} 个最终键")
    return final_aggregated_data


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
        # 注意：这里 call_stack 已经是 tuple of strings 了
        # 为了比较前缀，我们可以直接比较 tuple
        
        # Tuple startswith logic:
        len1 = len(call_stack1)
        len2 = len(call_stack2)
        min_len = min(len1, len2)
        
        return call_stack1[:min_len] == call_stack2[:min_len]
    
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
        new_len = len(new_call_stack)
        existing_len = len(existing_call_stack)
        
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

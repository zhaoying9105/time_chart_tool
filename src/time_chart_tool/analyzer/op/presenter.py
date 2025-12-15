"""
数据展示阶段 (纯函数实现)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

from ..utils.data_structures import AggregatedData
from ..utils.statistics import calculate_kernel_statistics

logger = logging.getLogger(__name__)


def _present_single_file(data: Dict[str,AggregatedData],
                       output_dir: str,
                       show_attributes: List[str],
                       aggregation_spec: List[str] = ['name'],
                       label: Optional[str] = None,
                       per_rank_stats: Optional[Dict[str, Dict[str, int]]] = None) -> List[Path]:
    """
    展示单文件数据
    
    Args:
        data: 聚合数据
        output_dir: 输出目录
        show_attributes: 显示属性列表
        aggregation_spec: 聚合字段列表
        label: 文件标签
        per_rank_stats: 每个rank的统计信息
        
    Returns:
        List[Path]: 生成的文件路径列表
    """
    print("生成单文件展示结果...")
    print(f"输入数据包含 {len(data)} 个聚合项")
    
    # 显示每个 rank 的统计信息（如果有的话）
    if per_rank_stats:
        print("\n=== 每个 Rank 的统计信息 ===")
        total_cpu_count = 0
        total_kernel_count = 0
        
        for rank_name, stats in per_rank_stats.items():
            cpu_count = stats['cpu_count']
            kernel_count = stats['kernel_count']
            total_cpu_count += cpu_count
            total_kernel_count += kernel_count
            print(f"文件: {rank_name}")
            print(f"  CPU操作数: {cpu_count}")
            print(f"  Kernel数: {kernel_count}")
            
        print(f"总计: {len(per_rank_stats)} 个文件")
        print(f"  总CPU操作数: {total_cpu_count}")
        print(f"  总Kernel数: {total_kernel_count}")
    
    # 准备表格数据
    rows = []
    
    # 预先计算 op_total_duration 用于排序
    data_items = []
    for key, agg_data in data.items():
        # 计算kernel统计信息
        kernel_events = agg_data.kernel_events
        
        # 计算op_total_duration
        op_total_duration = 0.0
        if kernel_events:
            # 按 kernel name 分组计算统计信息，或者直接计算总 duration
            # 这里为了简单，直接计算所有 kernel 的 duration 之和
            op_total_duration = sum(e.dur for e in kernel_events if e.dur is not None)
        
        # 获取第一个CPU事件的启动时间作为排序依据（如果是按op_index聚合）
        start_time = float('inf')
        if agg_data.cpu_events:
            if agg_data.cpu_events[0].ts is not None:
                start_time = agg_data.cpu_events[0].ts
        
        data_items.append((key, agg_data, op_total_duration, start_time))
    
    # 排序：如果有op_index，按时间排序；否则按duration排序
    if 'op_index' in aggregation_spec:
        data_items.sort(key=lambda x: x[3])  # 按start_time排序
    else:
        data_items.sort(key=lambda x: x[2], reverse=True)  # 按duration降序排序
    
    print(f"排序后有 {len(data_items)} 个数据项准备生成表格")
    
    for key, agg_data, op_total_duration, _ in data_items:
        cpu_events = agg_data.cpu_events
        if not cpu_events:
            continue
            
        cpu_event = cpu_events[0]
        args = cpu_event.args or {}
        
        row = {}
        
        # 添加聚合键相关列
        if aggregation_spec == ['name']:
            row['name'] = cpu_event.name
        elif 'op_index' in aggregation_spec:
            # 对于op_index聚合，我们需要显示更多信息
            row['op_index'] = key if not isinstance(key, tuple) else key[aggregation_spec.index('op_index')]
            row['name'] = cpu_event.name
        else:
            # 复合键，显示各个部分
            if isinstance(key, tuple):
                for i, field in enumerate(aggregation_spec):
                    if field == 'call_stack':
                        row[field] = str(key[i])
                    else:
                        row[field] = key[i]
            else:
                if aggregation_spec[0] == 'call_stack':
                    row[aggregation_spec[0]] = str(key)
                else:
                    row[aggregation_spec[0]] = key
        
        # 添加其他列
        if 'timestamp' in show_attributes:
            if 'readable_timestamp' in show_attributes and cpu_event.readable_timestamp:
                row['timestamp'] = cpu_event.readable_timestamp
            elif cpu_event.ts is not None:
                row['timestamp'] = f"{cpu_event.ts:.2f}"
            else:
                row['timestamp'] = ""
                
        if 'kernel_timestamp' in show_attributes:
            # 获取kernel的时间戳范围
            if agg_data.kernel_events:
                min_ts = min(e.ts for e in agg_data.kernel_events if e.ts is not None)
                max_ts = max(e.ts + (e.dur if e.dur else 0) for e in agg_data.kernel_events if e.ts is not None)
                row['kernel_timestamp_start'] = f"{min_ts:.2f}"
                row['kernel_timestamp_end'] = f"{max_ts:.2f}"
            else:
                row['kernel_timestamp_start'] = ""
                row['kernel_timestamp_end'] = ""
        
        if 'name' in show_attributes and 'name' not in row:
            row['name'] = cpu_event.name
            
        if 'dtype' in show_attributes:
            row['dtype'] = args.get('Input type', '')
            
        if 'shape' in show_attributes:
            row['shape'] = args.get('Input Dims', [])
            
        if 'fwd_bwd_type' in show_attributes:
            row['fwd_bwd_type'] = getattr(cpu_event, 'fwd_bwd_type', 'none')

        if 'pid' in show_attributes:
            row['pid'] = getattr(cpu_event, 'pid', 'none')
            
        if 'tid' in show_attributes:
            row['tid'] = getattr(cpu_event, 'tid', 'none')
            
        if 'op_index' in show_attributes:
            row['op_index'] = getattr(cpu_event, 'op_index', '')
            
        if 'call_stack' in show_attributes and 'call_stack' not in row:
            # 如果聚合键中不包含call_stack，但要求显示
            call_stack = cpu_event.call_stack
            if call_stack:
                row['call_stack'] = ' -> '.join(call_stack)
            else:
                row['call_stack'] = ""
        
        if 'stream' in show_attributes:
            # 显示kernel的stream信息
            streams = set()
            for k_event in agg_data.kernel_events:
                if k_event.stream is not None:
                    streams.add(k_event.stream)
            row['stream'] = ','.join(map(str, sorted(streams)))
        
        # Kernel 统计信息
        # kernel_events = agg_data.kernel_events
        # kernel_stats = calculate_kernel_statistics(kernel_events)
        
                # 如果显示kernel duration，先计算总耗时
        total_duration = 0.0
        if 'kernel_duration' in show_attributes:
            for aggregated_data in data.values():
                kernel_stats = calculate_kernel_statistics(aggregated_data.kernel_events)
                if kernel_stats:
                    total_duration += sum(stats.mean_duration * stats.count for stats in kernel_stats)

        # if show_kernel_names or show_kernel_duration or show_kernel_timestamp:
        if 'kernel_names' in show_attributes or 'kernel_duration' in show_attributes or 'kernel_timestamp' in show_attributes:
            # 收集 kernel 信息
            kernel_stats = calculate_kernel_statistics(agg_data.kernel_events)
            if kernel_stats:
                row['kernel_count'] = sum(stats.count for stats in kernel_stats)
                if 'kernel_names' in show_attributes:
                    row['kernel_names'] = '\n'.join(stats.kernel_name for stats in kernel_stats)
                if 'kernel_duration' in show_attributes:
                    op_total_duration = sum(stats.mean_duration * stats.count for stats in kernel_stats)
                    row['kernel_mean_duration'] = op_total_duration / sum(stats.count for stats in kernel_stats)
                    row['kernel_min_duration'] = min(stats.min_duration for stats in kernel_stats)
                    row['kernel_max_duration'] = max(stats.max_duration for stats in kernel_stats)
                    row['kernel_total_duration'] = op_total_duration
                    row['kernel_duration_ratio'] = (op_total_duration / total_duration * 100) if total_duration > 0 else 0.0
                if 'kernel_timestamp' in show_attributes:
                    # 收集kernel时间戳信息
                    kernel_timestamps = []
                    for kernel_event in agg_data.kernel_events:
                        if kernel_event.ts is not None:
                            kernel_timestamps.append(kernel_event.ts)
                    if kernel_timestamps:
                        row['kernel_timestamps'] = '\n'.join(map(str, sorted(kernel_timestamps)))
                    else:
                        row['kernel_timestamps'] = ''
            else:
                row['kernel_count'] = 0
                if 'kernel_names' in show_attributes:
                    row['kernel_names'] = ''
                if 'kernel_duration' in show_attributes:
                    row['kernel_mean_duration'] = 0.0
                    row['kernel_min_duration'] = 0.0
                    row['kernel_max_duration'] = 0.0
                    row['kernel_total_duration'] = 0.0
                    row['kernel_duration_ratio'] = 0.0
                if 'kernel_timestamp' in show_attributes:
                    row['kernel_timestamps'] = ''
        
        rows.append(row)
    
    # 生成输出文件
    base_name = _generate_base_name(aggregation_spec, show_attributes, file_labels=None)
    if label:
        base_name = f"{label}_{base_name}"
        
    return _generate_output_files(rows, output_dir, base_name)


def _present_multiple_files(data: Dict[str, Any], 
                          output_dir: str,
                          show_attributes: List[str],
                          aggregation_spec: List[str] = ['name'],
                          compare_attributes: List[str] = [],
                          file_labels: List[str] = None,
                          include_op_patterns: List[str] = None,
                          exclude_op_patterns: List[str] = None,
                          include_kernel_patterns: List[str] = None,
                          exclude_kernel_patterns: List[str] = None) -> List[Path]:
    """
    展示多文件数据
    
    Args:
        data: 比较数据
        output_dir: 输出目录
        show_attributes: 显示属性列表
        aggregation_spec: 聚合字段列表
        compare_attributes: 比较属性列表
        file_labels: 文件标签列表
        include_op_patterns: 包含的操作名称模式列表
        exclude_op_patterns: 排除的操作名称模式列表
        include_kernel_patterns: 包含的kernel名称模式列表
        exclude_kernel_patterns: 排除的kernel名称模式列表
        
    Returns:
        List[Path]: 生成的文件路径列表
    """
    print("生成多文件展示结果...")
    
    # 如果显示kernel duration，先计算每个文件的总耗时
    file_total_durations = {}
    if 'kernel_duration' in show_attributes:
        for key, entry in data.items():
            for label, file_data in entry.items():
                if label not in file_total_durations:
                    file_total_durations[label] = 0.0
                kernel_events = file_data.kernel_events
                kernel_stats = calculate_kernel_statistics(kernel_events)
                if kernel_stats:
                    file_total_durations[label] += sum(stats.mean_duration * stats.count for stats in kernel_stats)
    
    rows = []
    for key, entry in data.items():
        # 初始化行数据
        row = {}
        
        # 添加时间戳列（如果启用，作为第一列）
        if 'timestamp' in show_attributes:
            # 获取第一个文件的第一个CPU事件的启动时间戳作为基准时间戳
            first_timestamp = None
            for label, file_data in entry.items():
                cpu_events = file_data.cpu_events
                if cpu_events:
                    first_cpu_event = cpu_events[0]
                    if 'readable_timestamp' in show_attributes and first_cpu_event.readable_timestamp:
                        first_timestamp = first_cpu_event.readable_timestamp
                        break
                    elif first_cpu_event.ts is not None:
                        first_timestamp = first_cpu_event.ts
                        break
            
            if first_timestamp is not None:
                row['timestamp'] = str(first_timestamp)
            else:
                row['timestamp'] = ""

        # 添加聚合键相关列
        if aggregation_spec == ['name']:
            row['name'] = key
        elif 'op_index' in aggregation_spec:
            row['op_index'] = key if not isinstance(key, tuple) else key[aggregation_spec.index('op_index')]
            # 获取任意一个文件的第一个事件的名称
            for label, file_data in entry.items():
                if file_data.cpu_events:
                    row['name'] = file_data.cpu_events[0].name
                    break
        else:
            if isinstance(key, tuple):
                for i, field in enumerate(aggregation_spec):
                    if field == 'call_stack':
                        row[field] = str(key[i])
                    else:
                        row[field] = key[i]
            else:
                if aggregation_spec[0] == 'call_stack':
                    row[aggregation_spec[0]] = str(key)
                else:
                    row[aggregation_spec[0]] = key
        
        if 'name' in show_attributes and 'name' not in row:
             for label, file_data in entry.items():
                if file_data.cpu_events:
                    row['name'] = file_data.cpu_events[0].name
                    break
        
        # 添加每个文件的数据
        # 确保所有文件都有列，即使数据缺失
        all_labels = set()
        if file_labels:
            all_labels.update(file_labels)
        all_labels.update(entry.keys())
        
        for label in all_labels:
            if label in entry:
                file_data = entry[label]
                cpu_events = file_data.cpu_events
                
                if cpu_events:
                    cpu_event = cpu_events[0]
                    args = cpu_event.args or {}
                    
                    if 'dtype' in show_attributes:
                        row[f'{label}_dtype'] = args.get('Input type', '')
                    if 'shape' in show_attributes:
                        row[f'{label}_shape'] = args.get('Input Dims', [])
                    
                    if 'fwd_bwd_type' in show_attributes:
                        row[f'{label}_fwd_bwd_type'] = getattr(cpu_event, 'fwd_bwd_type', 'none')

                    if 'pid' in show_attributes:
                        row[f'{label}_pid'] = getattr(cpu_event, 'pid', 'none')
                        
                    if 'tid' in show_attributes:
                        row[f'{label}_tid'] = getattr(cpu_event, 'tid', 'none')
                        
                    if 'op_index' in show_attributes:
                        row[f'{label}_op_index'] = getattr(cpu_event, 'op_index', '')
                        
                    if 'call_stack' in show_attributes and 'call_stack' not in row:
                        call_stack = cpu_event.call_stack
                        if call_stack:
                            row[f'{label}_call_stack'] = ' -> '.join(call_stack)
                        else:
                            row[f'{label}_call_stack'] = ""
                            
                    if 'stream' in show_attributes:
                        streams = set()
                        for k_event in file_data.kernel_events:
                            if k_event.stream is not None:
                                streams.add(k_event.stream)
                        row[f'{label}_stream'] = ','.join(map(str, sorted(streams)))
                else:
                    # 填充空值
                    if 'dtype' in show_attributes:
                        row[f'{label}_dtype'] = ''
                    if 'shape' in show_attributes:
                        row[f'{label}_shape'] = ''
                    if 'fwd_bwd_type' in show_attributes:
                        row[f'{label}_fwd_bwd_type'] = ''
                    if 'pid' in show_attributes:
                        row[f'{label}_pid'] = ''
                    if 'tid' in show_attributes:
                        row[f'{label}_tid'] = ''
                    if 'op_index' in show_attributes:
                        row[f'{label}_op_index'] = ''
                    if 'call_stack' in show_attributes and 'call_stack' not in row:
                        row[f'{label}_call_stack'] = ''
                    if 'stream' in show_attributes:
                        row[f'{label}_stream'] = ''

                # Kernel 统计
                kernel_events = file_data.kernel_events
                kernel_stats = calculate_kernel_statistics(kernel_events)
                
                if kernel_stats:
                    row[f'{label}_kernel_count'] = sum(stats.count for stats in kernel_stats)
                    if 'kernel_names' in show_attributes:
                        row[f'{label}_kernel_names'] = '\n'.join(f"{stats.kernel_name} ({stats.count})" for stats in kernel_stats)
                    if 'kernel_duration' in show_attributes:
                        op_total_duration = sum(stats.mean_duration * stats.count for stats in kernel_stats)
                        row[f'{label}_kernel_mean_duration'] = op_total_duration / sum(stats.count for stats in kernel_stats)
                        row[f'{label}_kernel_min_duration'] = min(stats.min_duration for stats in kernel_stats)
                        row[f'{label}_kernel_max_duration'] = max(stats.max_duration for stats in kernel_stats)
                        row[f'{label}_kernel_total_duration'] = op_total_duration
                        row[f'{label}_kernel_duration_ratio'] = (op_total_duration / file_total_durations[label] * 100) if file_total_durations.get(label, 0) > 0 else 0.0
                else:
                    row[f'{label}_kernel_count'] = 0
                    if 'kernel_names' in show_attributes:
                        row[f'{label}_kernel_names'] = ''
                    if 'kernel_duration' in show_attributes:
                        row[f'{label}_kernel_mean_duration'] = 0.0
                        row[f'{label}_kernel_min_duration'] = 0.0
                        row[f'{label}_kernel_max_duration'] = 0.0
                        row[f'{label}_kernel_total_duration'] = 0.0
                        row[f'{label}_kernel_duration_ratio'] = 0.0
            else:
                # 文件缺失该条目
                if 'dtype' in show_attributes:
                    row[f'{label}_dtype'] = 'N/A'
                if 'shape' in show_attributes:
                    row[f'{label}_shape'] = 'N/A'
                if 'fwd_bwd_type' in show_attributes:
                    row[f'{label}_fwd_bwd_type'] = 'N/A'
                if 'pid' in show_attributes:
                    row[f'{label}_pid'] = 'N/A'
                if 'tid' in show_attributes:
                    row[f'{label}_tid'] = 'N/A'
                if 'op_index' in show_attributes:
                    row[f'{label}_op_index'] = 'N/A'
                if 'call_stack' in show_attributes and 'call_stack' not in row:
                    row[f'{label}_call_stack'] = 'N/A'
                if 'stream' in show_attributes:
                    row[f'{label}_stream'] = 'N/A'
                row[f'{label}_kernel_count'] = 0
                if 'kernel_names' in show_attributes:
                    row[f'{label}_kernel_names'] = 'N/A'
                if 'kernel_duration' in show_attributes:
                    row[f'{label}_kernel_mean_duration'] = 0.0
                    row[f'{label}_kernel_min_duration'] = 0.0
                    row[f'{label}_kernel_max_duration'] = 0.0
                    row[f'{label}_kernel_total_duration'] = 0.0
                    row[f'{label}_kernel_duration_ratio'] = 0.0
        
        # 添加比较列
        if 'dtype' in compare_attributes or 'shape' in compare_attributes or 'name' in compare_attributes or 'kernel_duration' in show_attributes:
            file_labels_list = list(entry.keys())
            if len(file_labels_list) >= 2:
                label1, label2 = file_labels_list[0], file_labels_list[1]
                
                if 'dtype' in compare_attributes:
                    dtypes1 = set()
                    dtypes2 = set()
                    for cpu_event in entry[label1].cpu_events:
                        args = cpu_event.args or {}
                        input_types = args.get('Input type', [])
                        if input_types:
                            dtypes1.add(str(input_types))
                    for cpu_event in entry[label2].cpu_events:
                        args = cpu_event.args or {}
                        input_types = args.get('Input type', [])
                        if input_types:
                            dtypes2.add(str(input_types))
                    row['dtype_equal'] = dtypes1 == dtypes2
                
                if 'shape' in compare_attributes:
                    shapes1 = set()
                    shapes2 = set()
                    for cpu_event in entry[label1].cpu_events:
                        args = cpu_event.args or {}
                        input_dims = args.get('Input Dims', [])
                        if input_dims:
                            shapes1.add(str(input_dims))
                    for cpu_event in entry[label2].cpu_events:
                        args = cpu_event.args or {}
                        input_dims = args.get('Input Dims', [])
                        if input_dims:
                            shapes2.add(str(input_dims))
                    row['shape_equal'] = shapes1 == shapes2
                
                if 'name' in compare_attributes:
                    names1 = set()
                    names2 = set()
                    for cpu_event in entry[label1].cpu_events:
                        names1.add(cpu_event.name)
                    for cpu_event in entry[label2].cpu_events:
                        names2.add(cpu_event.name)
                    row['name_equal'] = names1 == names2
                
                # 添加 mean duration ratio 比较
                if 'kernel_duration' in show_attributes:
                    mean_duration1 = row.get(f'{label1}_kernel_mean_duration', 0.0)
                    mean_duration2 = row.get(f'{label2}_kernel_mean_duration', 0.0)
                    
                    if mean_duration1 > 0 and mean_duration2 > 0:
                        ratio = mean_duration2 / mean_duration1
                        row[f'{label2}_vs_{label1}_mean_duration_ratio'] = ratio
                        
                        if ratio < 1:
                            improvement = (1 - ratio) * 100
                            row[f'{label2}_vs_{label1}_performance_improvement'] = f"{improvement:.1f}%"
                        else:
                            degradation = (ratio - 1) * 100
                            row[f'{label2}_vs_{label1}_performance_improvement'] = f"-{degradation:.1f}%"
        
        rows.append(row)
    
    base_name = _generate_base_name(aggregation_spec, show_attributes, file_labels, include_op_patterns, exclude_op_patterns, include_kernel_patterns, exclude_kernel_patterns)
    generated_files = _generate_output_files(rows, output_dir, base_name)
    
    return generated_files


def _generate_base_name(aggregation_spec: List[str], show_attributes: List[str], file_labels: List[str] = None,
                       include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None,
                       include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None) -> str:
    """
    生成基础文件名
    """
    parts = ['comparison']
    
    if file_labels:
        parts.append('_vs_'.join(file_labels))
        
    parts.append(f"by_{'_'.join(aggregation_spec)}")
    
    if 'dtype' in show_attributes:
        parts.append('dtype')
    if 'shape' in show_attributes:
        parts.append('shape')
    if 'kernel_names' in show_attributes:
        parts.append('kernel')
    if 'kernel_duration' in show_attributes:
        parts.append('duration')
    if 'timestamp' in show_attributes:
        parts.append('ts')
    if 'name' in show_attributes:
        parts.append('name')
    if 'call_stack' in show_attributes:
        parts.append('stack')
    if 'stream' in show_attributes:
        parts.append('stream')
        
    if include_op_patterns:
        parts.append(f"inc_op_{len(include_op_patterns)}")
    if exclude_op_patterns:
        parts.append(f"exc_op_{len(exclude_op_patterns)}")
    if include_kernel_patterns:
        parts.append(f"inc_ker_{len(include_kernel_patterns)}")
    if exclude_kernel_patterns:
        parts.append(f"exc_ker_{len(exclude_kernel_patterns)}")
        
    return '_'.join(parts)


def _generate_output_files(rows: List[Dict[str, Any]], output_dir: str, base_name: str) -> List[Path]:
    """
    生成输出文件 (CSV和Excel)
    
    Args:
        rows: 数据行列表
        output_dir: 输出目录
        base_name: 基础文件名
        
    Returns:
        List[Path]: 生成的文件路径列表
    """
    import pandas as pd
    
    if not rows:
        logger.warning("没有数据可供展示")
        return []
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(rows)
    
    files = []
    
    # 保存CSV
    csv_file = output_path / f"{base_name}.csv"
    try:
        df.to_csv(csv_file, index=False)
        files.append(csv_file)
        print(f"生成 CSV 文件: {csv_file}")
    except Exception as e:
        logger.error(f"生成 CSV 文件失败: {e}")
        
    # 保存Excel
    excel_file = output_path / f"{base_name}.xlsx"
    try:
        df.to_excel(excel_file, index=False)
        files.append(excel_file)
        print(f"生成 Excel 文件: {excel_file}")
    except Exception as e:
        logger.error(f"生成 Excel 文件失败: {e}")
        
    return files

"""
PyTorch Profiler 高级分析器
实现功能5和功能6：数据重组、统计分析和Excel输出
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
import statistics
from dataclasses import dataclass

from .models import ActivityEvent, ProfilerData
from .parser import PyTorchProfilerParser


@dataclass
class KernelStatistics:
    """Kernel 统计信息"""
    kernel_name: str
    min_duration: float
    max_duration: float
    mean_duration: float
    variance: float
    count: int
    
    def __str__(self):
        return f"KernelStatistics({self.kernel_name}, count={self.count}, mean={self.mean_duration:.3f}, std={self.variance**0.5:.3f})"


class Analyzer:
    """高级分析器"""
    
    def __init__(self):
        self.parser = PyTorchProfilerParser()
    
    def reorganize_by_external_id(self, data: ProfilerData) -> Dict[Union[int, str], List[ActivityEvent]]:
        """
        功能5.1: 用'External id'重新组织数据
        
        Args:
            data: ProfilerData 对象
            
        Returns:
            Dict[External id, List[ActivityEvent]]: 按 External id 组织的事件映射
        """
        external_id_map = defaultdict(list)
        
        for event in data.events:
            # 只保留有 External id 的事件
            if event.external_id is not None:
                # 只保留 cpu_op 和 kernel 两个类别
                if event.cat in ['cpu_op', 'kernel']:
                    external_id_map[event.external_id].append(event)
        
        return dict(external_id_map)
    
    def get_cpu_op_info(self, event: ActivityEvent) -> Tuple[str, List, List, List]:
        """
        从 cpu_op 事件中提取信息
        
        Args:
            event: cpu_op 事件
            
        Returns:
            Tuple[name, input_strides, input_dims, input_type]
        """
        name = event.name
        args = event.args or {}
        
        input_strides = args.get('Input Strides', [])
        input_dims = args.get('Input Dims', [])
        input_type = args.get('Input type', [])
        
        return name, input_strides, input_dims, input_type
    
    def calculate_kernel_statistics(self, kernel_events: List[ActivityEvent]) -> List[KernelStatistics]:
        """
        计算 kernel 事件的统计信息
        
        Args:
            kernel_events: kernel 事件列表
            
        Returns:
            List[KernelStatistics]: 统计信息列表
        """
        if not kernel_events:
            return []
        
        # 按 kernel name 分组
        kernel_groups = defaultdict(list)
        for event in kernel_events:
            kernel_groups[event.name].append(event)
        
        statistics_list = []
        
        for kernel_name, events in kernel_groups.items():
            if len(events) == 0:
                continue
                
            # 计算持续时间统计
            durations = [event.dur for event in events if event.dur is not None]
            
            if not durations:
                continue
                
            mean_duration = statistics.mean(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            # 计算方差
            if len(durations) > 1:
                variance = statistics.variance(durations)
            else:
                variance = 0.0
            
            stats = KernelStatistics(
                kernel_name=kernel_name,
                min_duration=min_duration,
                max_duration=max_duration,
                mean_duration=mean_duration,
                variance=variance,
                count=len(events)
            )
            
            statistics_list.append(stats)
        
        return statistics_list
    
    def analyze_cpu_op_kernel_mapping(self, data: ProfilerData) -> Dict:
        """
        功能5.2: 分析 cpu_op 和 kernel 的映射关系
        
        Args:
            data: ProfilerData 对象
            
        Returns:
            Dict: 逐级映射的数据结构
        """
        # 首先按 External id 重组数据
        external_id_map = self.reorganize_by_external_id(data)
        
        # 逐级映射: Map[cpu_op_name][cpu_op_input_strides][cpu_op_input_dims][cpu_op_input_type] -> List[kernel_events]
        mapping = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        
        for external_id, events in external_id_map.items():
            cpu_op_events = [e for e in events if e.cat == 'cpu_op']
            kernel_events = [e for e in events if e.cat == 'kernel']
            
            if not cpu_op_events or not kernel_events:
                continue
            
            # 检查是否有多个不同的 kernel name
            kernel_names = set(e.name for e in kernel_events)
            if len(kernel_names) > 1:
                pass
                # print(f"警告: External id {external_id} 对应的 kernel 事件有不同的名称: {kernel_names}")
                # print(f"  cpu_op 事件: {[e.name for e in cpu_op_events]}")
                # print(f"  kernel 事件: {[e.name for e in kernel_events]}")
            
            # 为每个 cpu_op 事件创建映射
            for cpu_op_event in cpu_op_events:
                name, input_strides, input_dims, input_type = self.get_cpu_op_info(cpu_op_event)
                
                # 将 input_strides, input_dims, input_type 转换为可哈希的类型
                def make_hashable(obj):
                    if isinstance(obj, list):
                        return tuple(make_hashable(item) for item in obj)
                    elif isinstance(obj, dict):
                        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
                    else:
                        return obj
                
                input_strides_key = make_hashable(input_strides)
                input_dims_key = make_hashable(input_dims)
                input_type_key = make_hashable(input_type)
                
                mapping[name][input_strides_key][input_dims_key][input_type_key].extend(kernel_events)
        
        return mapping
    
    def generate_excel_from_mapping(self, mapping: Dict, output_file: str = "cpu_op_kernel_analysis.xlsx") -> None:
        """
        功能5.3: 将映射数据生成 Excel 表格
        
        Args:
            mapping: 映射数据
            output_file: 输出文件名
        """
        rows = []
        
        for cpu_op_name, strides_map in mapping.items():
            for input_strides, dims_map in strides_map.items():
                for input_dims, types_map in dims_map.items():
                    for input_type, kernel_events in types_map.items():
                        # 计算 kernel 统计信息
                        kernel_stats = self.calculate_kernel_statistics(kernel_events)
                        
                        for stats in kernel_stats:
                            row = {
                                'cpu_op_name': cpu_op_name,
                                'cpu_op_input_strides': str(input_strides),
                                'cpu_op_input_dims': str(input_dims),
                                'cpu_op_input_type': str(input_type),
                                'kernel_name': stats.kernel_name,
                                'kernel_count': stats.count,
                                'kernel_min_duration': stats.min_duration,
                                'kernel_max_duration': stats.max_duration,
                                'kernel_mean_duration': stats.mean_duration,
                                'kernel_std_duration': stats.variance ** 0.5
                            }
                            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            try:
                df.to_excel(output_file, index=False)
                print(f"Excel 文件已生成: {output_file}")
                print(f"包含 {len(rows)} 行数据")
            except ImportError:
                # 如果没有 openpyxl，保存为 CSV 和 JSON
                csv_file = output_file.replace('.xlsx', '.csv')
                json_file = output_file.replace('.xlsx', '.json')
                
                # 使用安全的 CSV 输出方法
                self._safe_csv_output(df, csv_file)
                print(f"包含 {len(rows)} 行数据")
                
                # 同时保存为 JSON 格式，便于查看
                df.to_json(json_file, orient='records', indent=2, force_ascii=False)
                print(f"JSON 文件已生成: {json_file}")
                
                print("注意: 需要安装 openpyxl 来生成 Excel 文件: pip install openpyxl")
                print("注意: CSV 文件使用制表符分隔或带引号，请用 Excel 或支持这些格式的编辑器打开")
        else:
            print("没有数据可以生成文件")
    
    def analyze_multiple_files(self, file_labels: List[Tuple[str, str]]) -> Dict:
        """
        功能5.4: 分析多个 JSON 文件
        
        Args:
            file_labels: List of (file_path, label) tuples
            
        Returns:
            Dict: 合并后的映射数据
        """
        all_mappings = {}
        
        for file_path, label in file_labels:
            print(f"正在分析文件: {file_path} (标签: {label})")
            
            try:
                data = self.parser.load_json_file(file_path)
                mapping = self.analyze_cpu_op_kernel_mapping(data)
                all_mappings[label] = mapping
                print(f"  完成分析，找到 {len(mapping)} 个 cpu_op")
                
            except Exception as e:
                print(f"  分析文件失败: {e}")
                continue
        
        return all_mappings
    
    def merge_mappings(self, all_mappings: Dict) -> Dict:
        """
        合并多个文件的映射数据
        
        Args:
            all_mappings: Dict[label, mapping]
            
        Returns:
            Dict: 合并后的映射
        """
        merged = defaultdict(list)
        
        for label, mapping in all_mappings.items():
            for cpu_op_name, strides_map in mapping.items():
                for input_strides, dims_map in strides_map.items():
                    for input_dims, types_map in dims_map.items():
                        # 不考虑 input_type，直接合并
                        for input_type, kernel_events in types_map.items():
                            key = (cpu_op_name, input_strides, input_dims)
                            merged[key].append((label, kernel_events))
        
        return dict(merged)

    def _safe_csv_output(self, df: pd.DataFrame, csv_file: str) -> None:
        """
        安全地生成 CSV 文件，处理包含逗号的字段
        
        Args:
            df: pandas DataFrame
            csv_file: CSV 文件路径
        """
        # 方法1: 使用制表符分隔符
        df.to_csv(csv_file, index=False, sep='\t', encoding='utf-8')
        
        # 方法2: 同时生成一个带引号的 CSV 文件
        quoted_csv_file = csv_file.replace('.csv', '_quoted.csv')
        df.to_csv(quoted_csv_file, index=False, quoting=1, encoding='utf-8')  # quoting=1 表示所有字段都用引号包围
        
        print(f"CSV 文件已生成 (制表符分隔): {csv_file}")
        print(f"CSV 文件已生成 (带引号): {quoted_csv_file}")

    def generate_comparison_excel(self, merged_mapping: Dict, output_file: str = "comparison_analysis.xlsx") -> None:
        """
        生成比较分析的 Excel 表格
        
        Args:
            merged_mapping: 合并后的映射数据
            output_file: 输出文件名
        """
        rows = []
        
        for (cpu_op_name, input_strides, input_dims), label_events_list in merged_mapping.items():
            # 收集所有 kernel 统计信息
            all_kernel_stats = {}
            
            for label, kernel_events in label_events_list:
                kernel_stats = self.calculate_kernel_statistics(kernel_events)
                all_kernel_stats[label] = {stats.kernel_name: stats for stats in kernel_stats}
            
            # 获取所有标签
            labels = list(all_kernel_stats.keys())
            
            if len(labels) < 2:
                # 只有一个文件的数据，直接添加
                for label, kernel_stats in all_kernel_stats.items():
                    for kernel_name, stats in kernel_stats.items():
                        row = {
                            'cpu_op_name': cpu_op_name,
                            'cpu_op_input_strides': str(input_strides),
                            'cpu_op_input_dims': str(input_dims),
                            'file_label': label,
                            'kernel_name': kernel_name,
                            'kernel_count': stats.count,
                            'kernel_min_duration': stats.min_duration,
                            'kernel_max_duration': stats.max_duration,
                            'kernel_mean_duration': stats.mean_duration,
                            'kernel_std_duration': stats.variance ** 0.5
                        }
                        rows.append(row)
            else:
                # 有多个文件的数据，进行比较
                # 获取所有kernel名称
                all_kernel_names = set()
                for kernel_stats in all_kernel_stats.values():
                    all_kernel_names.update(kernel_stats.keys())
                
                for kernel_name in all_kernel_names:
                    # 检查是否所有文件都有这个kernel
                    if all(kernel_name in kernel_stats for kernel_stats in all_kernel_stats.values()):
                        # 所有文件都有这个kernel，进行比较
                        base_label = labels[0]
                        base_stats = all_kernel_stats[base_label][kernel_name]
                        base_mean = base_stats.mean_duration
                        
                        row = {
                            'cpu_op_name': cpu_op_name,
                            'cpu_op_input_strides': str(input_strides),
                            'cpu_op_input_dims': str(input_dims),
                            'kernel_name': kernel_name,
                            'comparison_type': 'matched'
                        }
                        
                        # 添加每个文件的数据
                        for label in labels:
                            stats = all_kernel_stats[label][kernel_name]
                            row[f'{label}_count'] = stats.count
                            row[f'{label}_min_duration'] = stats.min_duration
                            row[f'{label}_max_duration'] = stats.max_duration
                            row[f'{label}_mean_duration'] = stats.mean_duration
                            row[f'{label}_std_duration'] = stats.variance ** 0.5
                            
                            # 计算相对于基准的比值
                            if base_mean > 0:
                                row[f'{label}_ratio_to_{base_label}'] = stats.mean_duration / base_mean
                            else:
                                row[f'{label}_ratio_to_{base_label}'] = float('inf') if stats.mean_duration > 0 else 1.0
                        
                        rows.append(row)
                    else:
                        # 不是所有文件都有这个kernel，分别添加
                        for label, kernel_stats in all_kernel_stats.items():
                            if kernel_name in kernel_stats:
                                stats = kernel_stats[kernel_name]
                                row = {
                                    'cpu_op_name': cpu_op_name,
                                    'cpu_op_input_strides': str(input_strides),
                                    'cpu_op_input_dims': str(input_dims),
                                    'kernel_name': kernel_name,
                                    'comparison_type': 'unmatched',
                                    'file_label': label,
                                    'kernel_count': stats.count,
                                    'kernel_min_duration': stats.min_duration,
                                    'kernel_max_duration': stats.max_duration,
                                    'kernel_mean_duration': stats.mean_duration,
                                    'kernel_std_duration': stats.variance ** 0.5
                                }
                                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            try:
                df.to_excel(output_file, index=False)
                print(f"比较分析 Excel 文件已生成: {output_file}")
                print(f"包含 {len(rows)} 行数据")
                
                # 打印统计信息
                if 'comparison_type' in df.columns:
                    matched_count = len(df[df['comparison_type'] == 'matched'])
                    unmatched_count = len(df[df['comparison_type'] == 'unmatched'])
                    print(f"匹配的kernel比较: {matched_count} 行")
                    print(f"不匹配的kernel: {unmatched_count} 行")
                    
            except ImportError:
                # 如果没有 openpyxl，保存为 CSV 和 JSON
                csv_file = output_file.replace('.xlsx', '.csv')
                json_file = output_file.replace('.xlsx', '.json')
                
                # 使用安全的 CSV 输出方法
                self._safe_csv_output(df, csv_file)
                print(f"包含 {len(rows)} 行数据")
                
                # 同时保存为 JSON 格式，便于查看
                df.to_json(json_file, orient='records', indent=2, force_ascii=False)
                print(f"比较分析 JSON 文件已生成: {json_file}")
                
                print("注意: 需要安装 openpyxl 来生成 Excel 文件: pip install openpyxl")
                print("注意: CSV 文件使用制表符分隔或带引号，请用 Excel 或支持这些格式的编辑器打开")
        else:
            print("没有数据可以生成比较分析文件")
    
    def run_complete_analysis(self, file_labels: List[Tuple[str, str]], 
                            output_dir: str = ".", 
                            output_formats: List[str] = None) -> None:
        """
        运行完整的分析流程
        
        Args:
            file_labels: List of (file_path, label) tuples
            output_dir: 输出目录
            output_formats: 输出格式列表，支持 ['json', 'xlsx']
        """
        if output_formats is None:
            output_formats = ['json', 'xlsx']
        
        print("=== 开始完整分析流程 ===")
        
        # 分析多个文件
        all_mappings = self.analyze_multiple_files(file_labels)
        
        if not all_mappings:
            print("没有成功分析任何文件")
            return
        
        # 为每个文件生成单独的分析
        for label, mapping in all_mappings.items():
            base_name = f"{label}_single_file_analysis"
            
            if 'json' in output_formats:
                json_file = f"{output_dir}/{base_name}.json"
                self.save_mapping_to_json(mapping, json_file)
            
            if 'xlsx' in output_formats:
                xlsx_file = f"{output_dir}/{base_name}.xlsx"
                self.generate_excel_from_mapping(mapping, xlsx_file)
        
        # 合并映射并生成比较分析
        if len(all_mappings) > 1:
            merged_mapping = self.merge_mappings(all_mappings)
            base_name = "comparison_analysis"
            
            if 'json' in output_formats:
                json_file = f"{output_dir}/{base_name}.json"
                self.save_comparison_to_json(merged_mapping, json_file)
            
            if 'xlsx' in output_formats:
                xlsx_file = f"{output_dir}/{base_name}.xlsx"
                self.generate_comparison_excel(merged_mapping, xlsx_file)
        
        print("=== 分析流程完成 ===")

    def save_mapping_to_json(self, mapping: Dict, output_file: str) -> None:
        """
        将映射数据保存为 JSON 格式
        
        Args:
            mapping: 映射数据
            output_file: 输出文件路径
        """
        # 转换数据为可序列化的格式
        serializable_mapping = {}
        
        for cpu_op_name, strides_map in mapping.items():
            serializable_mapping[cpu_op_name] = {}
            
            for strides, dims_map in strides_map.items():
                serializable_mapping[cpu_op_name][str(strides)] = {}
                
                for dims, types_map in dims_map.items():
                    serializable_mapping[cpu_op_name][str(strides)][str(dims)] = {}
                    
                    for input_type, kernel_events in types_map.items():
                        # 转换 kernel 事件为可序列化的格式
                        serializable_events = []
                        for event in kernel_events:
                            serializable_event = {
                                'name': event.name,
                                'cat': event.cat,
                                'ph': event.ph,
                                'ts': event.ts,
                                'dur': event.dur,
                                'tid': event.tid,
                                'pid': event.pid,
                                'args': event.args,
                                'external_id': event.external_id
                            }
                            serializable_events.append(serializable_event)
                        
                        serializable_mapping[cpu_op_name][str(strides)][str(dims)][str(input_type)] = serializable_events
        
        # 保存为 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"映射数据已保存为 JSON: {output_file}")
    
    def save_comparison_to_json(self, merged_mapping: Dict, output_file: str) -> None:
        """
        将比较分析数据保存为 JSON 格式
        
        Args:
            merged_mapping: 合并的映射数据
            output_file: 输出文件路径
        """
        # 转换数据为可序列化的格式
        serializable_data = []
        
        for (cpu_op_name, input_strides, input_dims), label_events_list in merged_mapping.items():
            # 收集所有 kernel 统计信息
            all_kernel_stats = {}
            
            for label, kernel_events in label_events_list:
                kernel_stats = self.calculate_kernel_statistics(kernel_events)
                all_kernel_stats[label] = {stats.kernel_name: stats for stats in kernel_stats}
            
            # 获取所有标签
            labels = list(all_kernel_stats.keys())
            
            if len(labels) < 2:
                # 只有一个文件的数据，直接添加
                for label, kernel_stats in all_kernel_stats.items():
                    for kernel_name, stats in kernel_stats.items():
                        serializable_item = {
                            'cpu_op_name': cpu_op_name,
                            'cpu_op_input_strides': str(input_strides),
                            'cpu_op_input_dims': str(input_dims),
                            'file_label': label,
                            'kernel_name': kernel_name,
                            'kernel_count': stats.count,
                            'kernel_min_duration': stats.min_duration,
                            'kernel_max_duration': stats.max_duration,
                            'kernel_mean_duration': stats.mean_duration,
                            'kernel_std_duration': stats.variance ** 0.5
                        }
                        serializable_data.append(serializable_item)
            else:
                # 有多个文件的数据，进行比较
                # 获取所有kernel名称
                all_kernel_names = set()
                for kernel_stats in all_kernel_stats.values():
                    all_kernel_names.update(kernel_stats.keys())
                
                for kernel_name in all_kernel_names:
                    # 检查是否所有文件都有这个kernel
                    if all(kernel_name in kernel_stats for kernel_stats in all_kernel_stats.values()):
                        # 所有文件都有这个kernel，进行比较
                        base_label = labels[0]
                        base_stats = all_kernel_stats[base_label][kernel_name]
                        base_mean = base_stats.mean_duration
                        
                        for label in labels:
                            stats = all_kernel_stats[label][kernel_name]
                            serializable_item = {
                                'cpu_op_name': cpu_op_name,
                                'cpu_op_input_strides': str(input_strides),
                                'cpu_op_input_dims': str(input_dims),
                                'kernel_name': kernel_name,
                                'comparison_type': 'matched',
                                'file_label': label,
                                'kernel_count': stats.count,
                                'kernel_min_duration': stats.min_duration,
                                'kernel_max_duration': stats.max_duration,
                                'kernel_mean_duration': stats.mean_duration,
                                'kernel_std_duration': stats.variance ** 0.5,
                                'ratio_to_base': stats.mean_duration / base_mean if base_mean > 0 else float('inf')
                            }
                            serializable_data.append(serializable_item)
                    else:
                        # 不是所有文件都有这个kernel，分别添加
                        for label, kernel_stats in all_kernel_stats.items():
                            if kernel_name in kernel_stats:
                                stats = kernel_stats[kernel_name]
                                serializable_item = {
                                    'cpu_op_name': cpu_op_name,
                                    'cpu_op_input_strides': str(input_strides),
                                    'cpu_op_input_dims': str(input_dims),
                                    'kernel_name': kernel_name,
                                    'comparison_type': 'unmatched',
                                    'file_label': label,
                                    'kernel_count': stats.count,
                                    'kernel_min_duration': stats.min_duration,
                                    'kernel_max_duration': stats.max_duration,
                                    'kernel_mean_duration': stats.mean_duration,
                                    'kernel_std_duration': stats.variance ** 0.5
                                }
                                serializable_data.append(serializable_item)
        
        # 保存为 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"比较分析数据已保存为 JSON: {output_file}")
        print(f"包含 {len(serializable_data)} 条记录")

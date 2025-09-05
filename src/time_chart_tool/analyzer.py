"""
PyTorch Profiler 高级分析器
实现功能5和功能6：数据重组、统计分析和Excel输出
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import re
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
            Dict: 合并后的映射，key为(cpu_op_name, input_strides, input_dims)，value为Dict[label, Dict[input_type, kernel_events]]
        """
        merged = defaultdict(lambda: defaultdict(dict))
        
        for label, mapping in all_mappings.items():
            for cpu_op_name, strides_map in mapping.items():
                for input_strides, dims_map in strides_map.items():
                    for input_dims, types_map in dims_map.items():
                        for input_type, kernel_events in types_map.items():
                            key = (cpu_op_name, input_strides, input_dims)
                            merged[key][label][input_type] = kernel_events
        
        return dict(merged)

    def generate_comparison_excel(self, merged_mapping: Dict, output_file: str = "comparison_analysis.xlsx") -> None:
        """
        生成比较分析的 Excel 表格
        
        Args:
            merged_mapping: 合并后的映射数据
            output_file: 输出文件名
        """
        rows = []
        
        for (cpu_op_name, input_strides, input_dims), label_types_map in merged_mapping.items():
            # 收集所有标签
            labels = list(label_types_map.keys())
            
            # 基础行数据
            row = {
                'cpu_op_name': cpu_op_name,
                'cpu_op_input_strides': str(input_strides),
                'cpu_op_input_dims': str(input_dims)
            }
            
            # 为每个标签收集数据
            for label in labels:
                types_map = label_types_map[label]
                
                # 收集所有input_type和kernel信息
                input_types = []
                kernel_names = []
                kernel_stats_list = []
                
                for input_type, kernel_events in types_map.items():
                    input_types.append(str(input_type))
                    kernel_stats = self.calculate_kernel_statistics(kernel_events)
                    
                    for stats in kernel_stats:
                        kernel_names.append(stats.kernel_name)
                        kernel_stats_list.append(stats)
                
                # 用||连接多个值
                row[f'{label}_input_types'] = '||'.join(input_types) if input_types else ''
                row[f'{label}_kernel_names'] = '||'.join(kernel_names) if kernel_names else ''
                
                # 计算kernel统计信息
                if kernel_stats_list:
                    # 计算所有kernel的总调用次数和加权平均duration
                    total_count = sum(stats.count for stats in kernel_stats_list)
                    weighted_mean = sum(stats.mean_duration * stats.count for stats in kernel_stats_list) / total_count if total_count > 0 else 0.0
                    
                    row[f'{label}_kernel_count'] = total_count
                    row[f'{label}_kernel_mean_duration'] = weighted_mean
                else:
                    row[f'{label}_kernel_count'] = 0
                    row[f'{label}_kernel_mean_duration'] = 0.0
            
            # 计算相对变化和比较信息（如果有多个标签）
            if len(labels) >= 2:
                base_label = labels[0]
                base_mean = row.get(f'{base_label}_kernel_mean_duration', 0.0)
                
                # 收集所有kernel_names和kernel_count进行比较
                all_kernel_names = []
                all_kernel_counts = []
                
                for label in labels:
                    all_kernel_names.append(set(row.get(f'{label}_kernel_names', '').split('||') if row.get(f'{label}_kernel_names') else []))
                    all_kernel_counts.append(row.get(f'{label}_kernel_count', 0))
                
                # 检查kernel_names是否相等（不考虑顺序）
                kernel_names_equal = len(set(frozenset(names) for names in all_kernel_names)) == 1
                row['kernel_names_equal'] = kernel_names_equal
                
                # 检查kernel_count是否相等
                kernel_count_equal = len(set(all_kernel_counts)) == 1
                row['kernel_count_equal'] = kernel_count_equal
                
                for label in labels[1:]:
                    current_mean = row.get(f'{label}_kernel_mean_duration', 0.0)
                    if base_mean > 0:
                        ratio = current_mean / base_mean
                        row[f'{label}_ratio_to_{base_label}'] = ratio
                    else:
                        row[f'{label}_ratio_to_{base_label}'] = float('inf') if current_mean > 0 else 1.0
            else:
                # 只有一个标签时，设置默认值
                row['kernel_names_equal'] = True
                row['kernel_count_equal'] = True
            
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            try:
                df.to_excel(output_file, index=False)
                print(f"比较分析 Excel 文件已生成: {output_file}")
                print(f"包含 {len(rows)} 行数据")
                
                # 打印统计信息
                print(f"分析了 {len(merged_mapping)} 个不同的 (cpu_op_name, input_strides, input_dims) 组合")
                
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
            
            # 生成包含标签的文件名
            labels = list(all_mappings.keys())
            labels_str = '_'.join(labels)
            base_name = f"comparison_analysis_{labels_str}"
            
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
        
        for key, label_types_map in merged_mapping.items():
            cpu_op_name, input_strides, input_dims = key
            
            # 收集所有标签
            labels = list(label_types_map.keys())
            
            # 基础行数据
            row = {
                'cpu_op_name': cpu_op_name,
                'cpu_op_input_strides': str(input_strides),
                'cpu_op_input_dims': str(input_dims)
            }
            
            # 为每个标签收集数据
            for label in labels:
                types_map = label_types_map[label]
                
                # 收集所有input_type和kernel信息
                input_types = []
                kernel_names = []
                kernel_stats_list = []
                
                for input_type, kernel_events in types_map.items():
                    input_types.append(str(input_type))
                    kernel_stats = self.calculate_kernel_statistics(kernel_events)
                    
                    for stats in kernel_stats:
                        kernel_names.append(stats.kernel_name)
                        kernel_stats_list.append(stats)
                
                # 用||连接多个值
                row[f'{label}_input_types'] = '||'.join(input_types) if input_types else ''
                row[f'{label}_kernel_names'] = '||'.join(kernel_names) if kernel_names else ''
                
                # 计算kernel统计信息
                if kernel_stats_list:
                    # 计算所有kernel的总调用次数和加权平均duration
                    total_count = sum(stats.count for stats in kernel_stats_list)
                    weighted_mean = sum(stats.mean_duration * stats.count for stats in kernel_stats_list) / total_count if total_count > 0 else 0.0
                    
                    row[f'{label}_kernel_count'] = total_count
                    row[f'{label}_kernel_mean_duration'] = weighted_mean
                else:
                    row[f'{label}_kernel_count'] = 0
                    row[f'{label}_kernel_mean_duration'] = 0.0
            
            # 计算相对变化和比较信息（如果有多个标签）
            if len(labels) >= 2:
                base_label = labels[0]
                base_mean = row.get(f'{base_label}_kernel_mean_duration', 0.0)
                
                # 收集所有kernel_names和kernel_count进行比较
                all_kernel_names = []
                all_kernel_counts = []
                
                for label in labels:
                    all_kernel_names.append(set(row.get(f'{label}_kernel_names', '').split('||') if row.get(f'{label}_kernel_names') else []))
                    all_kernel_counts.append(row.get(f'{label}_kernel_count', 0))
                
                # 检查kernel_names是否相等（不考虑顺序）
                kernel_names_equal = len(set(frozenset(names) for names in all_kernel_names)) == 1
                row['kernel_names_equal'] = kernel_names_equal
                
                # 检查kernel_count是否相等
                kernel_count_equal = len(set(all_kernel_counts)) == 1
                row['kernel_count_equal'] = kernel_count_equal
                
                for label in labels[1:]:
                    current_mean = row.get(f'{label}_kernel_mean_duration', 0.0)
                    if base_mean > 0:
                        ratio = current_mean / base_mean
                        row[f'{label}_ratio_to_{base_label}'] = ratio
                    else:
                        row[f'{label}_ratio_to_{base_label}'] = float('inf') if current_mean > 0 else 1.0
            else:
                # 只有一个标签时，设置默认值
                row['kernel_names_equal'] = True
                row['kernel_count_equal'] = True
            
            serializable_data.append(row)
        
        # 保存为 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"比较分析数据已保存为 JSON: {output_file}")
        print(f"包含 {len(serializable_data)} 条记录")

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

    def extract_matmul_dimensions(self, input_dims_str: str) -> Optional[Tuple[int, int, int]]:
        """
        从matmul算子的input_dims中提取m, k, n维度
        
        Args:
            input_dims_str: 格式如 "((2048, 3), (3, 32))" 的字符串
            
        Returns:
            Optional[Tuple[m, k, n]]: 提取的维度，如果解析失败返回None
        """
        try:
            # 使用正则表达式提取数字
            pattern = r'\(\((\d+),\s*(\d+)\)\s*,\s*\((\d+),\s*(\d+)\)\)'
            match = re.match(pattern, input_dims_str)
            
            if match:
                m, k1, k2, n = map(int, match.groups())
                # 验证k1 == k2 (矩阵乘法的要求)
                if k1 == k2:
                    return (m, k1, n)
            
            return None
        except Exception:
            return None

    def analyze_matmul_by_min_dim(self, comparison_data: List[Dict]) -> Dict[int, List]:
        """
        专门分析matmul算子，按最小维度分组
        
        Args:
            comparison_data: 比较分析的数据列表
            
        Returns:
            Dict[int, List]: 按min_dim分组的matmul数据
        """
        matmul_data = {}
        
        # 首先收集所有标签
        labels = []
        for item in comparison_data:
            if item.get('cpu_op_name') == 'aten::mm':
                for key in item.keys():
                    if key.endswith('_input_types'):
                        label = key.replace('_input_types', '')
                        if label not in labels:
                            labels.append(label)
                break
        
        if not labels:
            return {}
        
        # 按min_dim分组收集数据
        min_dim_data = defaultdict(lambda: {label: [] for label in labels})
        
        for item in comparison_data:
            if item.get('cpu_op_name') == 'aten::mm':
                input_dims = item.get('cpu_op_input_dims', '')
                dimensions = self.extract_matmul_dimensions(input_dims)
                
                if dimensions:
                    m, k, n = dimensions
                    min_dim = min(m, k, n)
                    
                    # 为每个标签收集数据
                    for label in labels:
                        input_types = item.get(f'{label}_input_types', '')
                        kernel_count = item.get(f'{label}_kernel_count', 0)
                        kernel_mean_duration = item.get(f'{label}_kernel_mean_duration', 0.0)
                        
                        # 只有当所有字段都有有效数据时才添加
                        if input_types and kernel_count > 0 and kernel_mean_duration > 0:
                            min_dim_data[min_dim][label].append([input_types, kernel_count, kernel_mean_duration])
        
        # 只保留在所有标签中都有数据的min_dim
        filtered_matmul_data = {}
        for min_dim, label_data in min_dim_data.items():
            # 检查是否所有标签都有数据
            all_labels_have_data = all(len(data_list) > 0 for data_list in label_data.values())
            
            if all_labels_have_data:
                # 为每个标签选择第一个数据条目（或者可以取平均值）
                filtered_entries = []
                for label in labels:
                    data_list = label_data[label]
                    if data_list:
                        # 取第一个条目，或者可以计算平均值
                        filtered_entries.extend(data_list[0])
                
                filtered_matmul_data[min_dim] = [filtered_entries]
        
        return filtered_matmul_data

    def generate_matmul_analysis(self, comparison_data: List[Dict], output_dir: str = ".") -> None:
        """
        生成matmul算子的专门分析
        
        Args:
            comparison_data: 比较分析的数据列表
            output_dir: 输出目录
        """
        print("=== 开始matmul算子专门分析 ===")
        
        # 分析matmul数据
        matmul_data = self.analyze_matmul_by_min_dim(comparison_data)
        
        if not matmul_data:
            print("没有找到matmul算子数据")
            return
        
        # 获取标签信息
        labels = []
        for item in comparison_data:
            if item.get('cpu_op_name') == 'aten::mm':
                for key in item.keys():
                    if key.endswith('_input_types'):
                        label = key.replace('_input_types', '')
                        if label not in labels:
                            labels.append(label)
                break
        
        # 生成JSON数据
        json_data = []
        for min_dim, entries in matmul_data.items():
            for entry in entries:
                row = {'mm_min_dim': min_dim}
                
                # 为每个标签添加数据
                for i, label in enumerate(labels):
                    start_idx = i * 3
                    if start_idx + 2 < len(entry):
                        row[f'{label}_input_types'] = entry[start_idx]
                        row[f'{label}_kernel_count'] = entry[start_idx + 1]
                        row[f'{label}_kernel_mean_duration'] = entry[start_idx + 2]
                
                # 添加input shape和input strides信息
                # 从comparison_data中找到对应的原始数据
                for item in comparison_data:
                    if item.get('cpu_op_name') == 'aten::mm':
                        input_dims = item.get('cpu_op_input_dims', '')
                        input_strides = item.get('cpu_op_input_strides', '')
                        dimensions = self.extract_matmul_dimensions(input_dims)
                        
                        if dimensions:
                            m, k, n = dimensions
                            current_min_dim = min(m, k, n)
                            
                            if current_min_dim == min_dim:
                                # 为每个标签添加input shape和input strides
                                for label in labels:
                                    # 检查这个标签是否有数据
                                    has_data = False
                                    for i, label_check in enumerate(labels):
                                        if label_check == label:
                                            start_idx = i * 3
                                            if start_idx + 2 < len(entry):
                                                has_data = True
                                                break
                                    
                                    if has_data:
                                        row[f'{label}_input_shape'] = input_dims
                                        row[f'{label}_input_strides'] = input_strides
                                break
                
                # 计算比率（如果有多个标签）
                if len(labels) >= 2:
                    base_label = labels[0]
                    base_duration = row.get(f'{base_label}_kernel_mean_duration', 0.0)
                    
                    for label in labels[1:]:
                        current_duration = row.get(f'{label}_kernel_mean_duration', 0.0)
                        if base_duration > 0:
                            ratio = current_duration / base_duration
                            row[f'{label}_ratio_to_{base_label}'] = ratio
                        else:
                            row[f'{label}_ratio_to_{base_label}'] = float('inf') if current_duration > 0 else 1.0
                
                json_data.append(row)
        
        # 保存JSON文件
        json_file = f"{output_dir}/matmul_analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Matmul分析JSON文件已生成: {json_file}")
        
        # 生成Excel文件
        if json_data:
            df = pd.DataFrame(json_data)
            xlsx_file = f"{output_dir}/matmul_analysis.xlsx"
            try:
                df.to_excel(xlsx_file, index=False)
                print(f"Matmul分析Excel文件已生成: {xlsx_file}")
            except ImportError:
                csv_file = f"{output_dir}/matmul_analysis.csv"
                self._safe_csv_output(df, csv_file)
                print(f"Matmul分析CSV文件已生成: {csv_file}")
        
        # 生成折线图
        if len(labels) >= 2:
            self.generate_matmul_chart(json_data, labels, output_dir)
        
        print(f"=== Matmul分析完成，共处理 {len(matmul_data)} 个不同的min_dim ===")

    def generate_matmul_chart(self, json_data: List[Dict], labels: List[str], output_dir: str) -> None:
        """
        生成matmul算子的折线图
        
        Args:
            json_data: matmul分析数据
            labels: 标签列表
            output_dir: 输出目录
        """
        print(f"开始生成matmul图表，标签数量: {len(labels)}")
        print(f"JSON数据数量: {len(json_data)}")
        
        if len(labels) < 2:
            print("标签数量少于2，跳过图表生成")
            return
        
        # 按min_dim分组数据
        min_dim_data = defaultdict(list)
        for item in json_data:
            min_dim = item.get('mm_min_dim')
            if min_dim is not None:
                min_dim_data[min_dim].append(item)
        
        print(f"找到 {len(min_dim_data)} 个不同的min_dim")
        
        # 计算每个min_dim的平均比率，只考虑在所有标签中都有数据的点
        chart_data = {}
        base_label = labels[0]
        
        for min_dim, items in min_dim_data.items():
            # 检查这个min_dim是否在所有标签中都有数据
            all_labels_have_data = True
            for label in labels:
                has_data = False
                for item in items:
                    duration_key = f'{label}_kernel_mean_duration'
                    if duration_key in item and item[duration_key] > 0:
                        has_data = True
                        break
                if not has_data:
                    all_labels_have_data = False
                    break
            
            if all_labels_have_data:
                chart_data[min_dim] = {}
                for label in labels[1:]:
                    ratios = []
                    for item in items:
                        ratio_key = f'{label}_ratio_to_{base_label}'
                        if ratio_key in item and item[ratio_key] != float('inf'):
                            ratios.append(item[ratio_key])
                    
                    if ratios:
                        chart_data[min_dim][label] = sum(ratios) / len(ratios)
        
        print(f"在所有标签中都有数据的min_dim数量: {len(chart_data)}")
        
        if not chart_data:
            print("没有找到在所有标签中都有数据的min_dim，跳过图表生成")
            return
        
        try:
            # 生成图表
            plt.figure(figsize=(12, 8))
            
            # 为每个标签绘制一条线
            for label in labels[1:]:
                x_values = []
                y_values = []
                
                for min_dim in sorted(chart_data.keys()):
                    if label in chart_data[min_dim]:
                        x_values.append(min_dim)
                        y_values.append(chart_data[min_dim][label])
                
                if x_values and y_values:
                    plt.plot(x_values, y_values, marker='o', label=f'{label} ratio to {base_label}', linewidth=2, markersize=6)
            
            plt.xlabel('Matmul Min Dimension (m/k/n)', fontsize=12)
            plt.ylabel(f'Performance Ratio ({base_label} = 1.0)', fontsize=12)
            plt.title('Matmul Performance Analysis by Min Dimension\n(Only shapes present in all time charts)', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图表
            chart_file = f"{output_dir}/matmul_performance_chart.jpg"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Matmul性能图表已生成: {chart_file}")
            print(f"图表包含 {len(chart_data)} 个在所有标签中都有数据的min_dim点")
        except Exception as e:
            print(f"生成图表时出错: {e}")
            print("尝试使用Agg后端...")
            try:
                # 重新设置matplotlib后端
                import matplotlib
                matplotlib.use('Agg')
                # 重新导入plt
                import matplotlib.pyplot as plt_new
                
                # 重新生成图表
                plt_new.figure(figsize=(12, 8))
                
                # 为每个标签绘制一条线
                for label in labels[1:]:
                    x_values = []
                    y_values = []
                    
                    for min_dim in sorted(chart_data.keys()):
                        if label in chart_data[min_dim]:
                            x_values.append(min_dim)
                            y_values.append(chart_data[min_dim][label])
                    
                    if x_values and y_values:
                        plt_new.plot(x_values, y_values, marker='o', label=f'{label} ratio to {base_label}', linewidth=2, markersize=6)
                
                plt_new.xlabel('Matmul Min Dimension (m/k/n)', fontsize=12)
                plt_new.ylabel(f'Performance Ratio ({base_label} = 1.0)', fontsize=12)
                plt_new.title('Matmul Performance Analysis by Min Dimension\n(Only shapes present in all time charts)', fontsize=14, fontweight='bold')
                plt_new.legend(fontsize=10)
                plt_new.grid(True, alpha=0.3)
                plt_new.tight_layout()
                
                # 保存图表
                chart_file = f"{output_dir}/matmul_performance_chart.jpg"
                plt_new.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt_new.close()
                
                print(f"Matmul性能图表已生成: {chart_file}")
                print(f"图表包含 {len(chart_data)} 个在所有标签中都有数据的min_dim点")
            except Exception as e2:
                print(f"使用Agg后端仍然失败: {e2}")
                print("无法生成matmul性能图表")

    def run_complete_analysis_with_matmul(self, file_labels: List[Tuple[str, str]], 
                                        output_dir: str = ".", 
                                        output_formats: List[str] = None) -> None:
        """
        运行包含matmul专门分析的完整分析流程
        
        Args:
            file_labels: List of (file_path, label) tuples
            output_dir: 输出目录
            output_formats: 输出格式列表，支持 ['json', 'xlsx']
        """
        if output_formats is None:
            output_formats = ['json', 'xlsx']
        
        print("=== 开始完整分析流程（包含matmul专门分析）===")
        
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
                
                # 读取比较分析数据用于matmul分析
                with open(json_file, 'r', encoding='utf-8') as f:
                    comparison_data = json.load(f)
                
                # 生成matmul专门分析
                self.generate_matmul_analysis(comparison_data, output_dir)
            
            if 'xlsx' in output_formats:
                xlsx_file = f"{output_dir}/{base_name}.xlsx"
                self.generate_comparison_excel(merged_mapping, xlsx_file)
        
        print("=== 分析流程完成 ===")



    def generate_cpu_op_performance_summary(self, data: 'ProfilerData', output_dir: str = ".", label: str = "") -> None:
        """
        生成cpu_op性能统计摘要（基于对应的kernel事件耗时）
        
        Args:
            data: 分析的数据
            output_dir: 输出目录
            label: 文件标签，用于生成带前缀的输出文件名
        """
        print("=== 开始生成cpu_op性能统计摘要（基于kernel耗时） ===")
        
        # 首先获取cpu_op和kernel的映射关系
        mapping = self.analyze_cpu_op_kernel_mapping(data)
        
        # 统计每种cpu_op对应的kernel性能数据
        cpu_op_stats = {}
        total_kernel_duration = 0.0
        
        # 遍历映射数据，统计每种cpu_op对应的kernel耗时
        for cpu_op_name, strides_map in mapping.items():
            if cpu_op_name not in cpu_op_stats:
                cpu_op_stats[cpu_op_name] = {
                    'call_count': 0,
                    'total_kernel_duration': 0.0,
                    'min_kernel_duration': float('inf'),
                    'max_kernel_duration': 0.0,
                    'kernel_count': 0
                }
            
            stats = cpu_op_stats[cpu_op_name]
            
            # 遍历该cpu_op的所有配置
            for strides, dims_map in strides_map.items():
                for dims, types_map in dims_map.items():
                    for input_type, kernel_events in types_map.items():
                        # 统计该配置下的kernel事件
                        for kernel_event in kernel_events:
                            if kernel_event.dur is not None:
                                stats['call_count'] += 1
                                stats['total_kernel_duration'] += kernel_event.dur
                                stats['min_kernel_duration'] = min(stats['min_kernel_duration'], kernel_event.dur)
                                stats['max_kernel_duration'] = max(stats['max_kernel_duration'], kernel_event.dur)
                                stats['kernel_count'] += 1
                                total_kernel_duration += kernel_event.dur
        
        if not cpu_op_stats:
            print("没有找到cpu_op对应的kernel事件")
            return
        
        # 计算平均耗时和比例
        summary_data = []
        for cpu_op_name, stats in cpu_op_stats.items():
            if stats['call_count'] > 0:
                avg_duration = stats['total_kernel_duration'] / stats['call_count']
                percentage = (stats['total_kernel_duration'] / total_kernel_duration * 100) if total_kernel_duration > 0 else 0.0
                
                summary_data.append({
                    'cpu_op_name': cpu_op_name,
                    'call_count': stats['call_count'],
                    'total_kernel_duration': stats['total_kernel_duration'],
                    'avg_kernel_duration': avg_duration,
                    'min_kernel_duration': stats['min_kernel_duration'] if stats['min_kernel_duration'] != float('inf') else 0.0,
                    'max_kernel_duration': stats['max_kernel_duration'],
                    'kernel_count': stats['kernel_count'],
                    'percentage_of_total': percentage
                })
        
        # 按总kernel耗时降序排序
        summary_data.sort(key=lambda x: x['total_kernel_duration'], reverse=True)
        
        # 添加总计行
        if summary_data:
            summary_data.append({
                'cpu_op_name': 'TOTAL',
                'call_count': sum(item['call_count'] for item in summary_data),
                'total_kernel_duration': total_kernel_duration,
                'avg_kernel_duration': total_kernel_duration / sum(item['call_count'] for item in summary_data) if sum(item['call_count'] for item in summary_data) > 0 else 0.0,
                'min_kernel_duration': 0.0,
                'max_kernel_duration': 0.0,
                'kernel_count': sum(item['kernel_count'] for item in summary_data),
                'percentage_of_total': 100.0
            })
        
        # 生成Excel文件
        import pandas as pd
        df = pd.DataFrame(summary_data)
        
        # 使用label前缀生成文件名
        base_name = f"{label}_cpu_op_performance_summary" if label else "cpu_op_performance_summary"
        xlsx_file = f"{output_dir}/{base_name}.xlsx"
        
        try:
            df.to_excel(xlsx_file, index=False)
            print(f"CPU Op性能统计Excel文件已生成: {xlsx_file}")
        except ImportError:
            csv_file = f"{output_dir}/{base_name}.csv"
            self._safe_csv_output(df, csv_file)
            print(f"CPU Op性能统计CSV文件已生成: {csv_file}")
        
        # 生成JSON文件
        json_file = f"{output_dir}/{base_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"CPU Op性能统计JSON文件已生成: {json_file}")
        
        # 打印统计信息
        print(f"\n=== CPU Op性能统计摘要（基于kernel耗时） ===")
        print(f"总共发现 {len(cpu_op_stats)} 种不同的cpu_op")
        print(f"总kernel调用次数: {sum(item['call_count'] for item in summary_data[:-1]) if summary_data else 0}")
        print(f"总kernel耗时: {total_kernel_duration:.2f} 微秒")
        
        # 打印markdown表格形式的summary结果
        self._print_cpu_op_summary_markdown_table(summary_data)
        
        print(f"\n前5个最耗时的cpu_op（基于kernel耗时）:")
        
        for i, item in enumerate(summary_data[:5]):
            if item['cpu_op_name'] != 'TOTAL':
                print(f"  {i+1}. {item['cpu_op_name']}")
                print(f"     kernel调用次数: {item['call_count']}")
                print(f"     总kernel耗时: {item['total_kernel_duration']:.2f} 微秒")
                print(f"     平均kernel耗时: {item['avg_kernel_duration']:.2f} 微秒")
                print(f"     占总kernel耗时比例: {item['percentage_of_total']:.2f}%")
                print()
        
        print(f"=== CPU Op性能统计完成 ===")

    def _normalize_call_stack(self, call_stack: List[str]) -> List[str]:
        """
        标准化 call stack，只保留包含 nn.Module 的有价值部分
        
        Args:
            call_stack: 原始 call stack
            
        Returns:
            List[str]: 标准化后的 call stack，只包含模型相关的层级
        """
        if not call_stack:
            return call_stack
        
        # 找到第一个包含 nn.Module 的层级
        model_start_idx = -1
        for i, frame in enumerate(call_stack):
            if 'nn.Module:' in frame:
                model_start_idx = i
                break
        
        if model_start_idx == -1:
            # 如果没有找到 nn.Module，返回空列表
            return []
        
        # 从第一个 nn.Module 开始，收集所有相关的层级
        # 包括 nn.Module 层级和它们之间的调用层级
        normalized = []
        in_model_context = False
        
        for i, frame in enumerate(call_stack):
            if i >= model_start_idx:
                # 检查是否是 nn.Module 层级
                if 'nn.Module:' in frame:
                    in_model_context = True
                    normalized.append(frame)
                # 检查是否是模型内部的调用（在 nn.Module 之间的层级）
        
        return normalized

    def analyze_cpu_op_by_call_stack(self, data: ProfilerData) -> Dict:
        """
        基于 call stack 分析 cpu_op 事件
        
        Args:
            data: ProfilerData 对象
            
        Returns:
            Dict: 按 call stack 组织的 cpu_op 数据，结构为 call_stack -> op_name -> [event_info]
        """
        # 获取所有包含 call stack 的 cpu_op 事件
        cpu_op_events = []
        for event in data.events:
            if event.cat == 'cpu_op' and event.call_stack is not None:
                cpu_op_events.append(event)
        
        # 按标准化后的 call stack 分组
        call_stack_groups = defaultdict(list)
        for event in cpu_op_events:
            normalized_call_stack = self._normalize_call_stack(event.call_stack)
            if normalized_call_stack:  # 只处理有有效 call stack 的事件
                call_stack_key = tuple(normalized_call_stack)
                call_stack_groups[call_stack_key].append(event)
        
        # 为每个 call stack 收集信息
        call_stack_analysis = {}
        for call_stack_key, events in call_stack_groups.items():
            call_stack_list = list(call_stack_key)
            
            # 按 op_name 分组收集信息
            op_groups = defaultdict(list)
            for event in events:
                name = event.name
                args = event.args or {}
                input_dims = args.get('Input Dims', [])
                input_strides = args.get('Input Strides', [])
                input_type = args.get('Input type', [])
                
                # 获取对应的 kernel 事件
                kernel_names = []
                if event.external_id is not None:
                    kernel_events = data.get_events_by_external_id(event.external_id)
                    for kernel_event in kernel_events:
                        if kernel_event.cat == 'kernel':
                            kernel_names.append(kernel_event.name)
                
                # 收集每个 op 的详细信息
                op_info = {
                    'op_name': name,
                    'input_dims': input_dims,
                    'input_strides': input_strides,
                    'input_type': input_type,
                    'kernel_names': kernel_names,
                    'external_id': event.external_id,
                    'ts': event.ts,
                    'dur': event.dur
                }
                op_groups[name].append(op_info)
            
            call_stack_analysis[call_stack_key] = {
                'call_stack': call_stack_list,
                'ops': dict(op_groups),
                'first_occurrence_time': min(event.ts for event in events)
            }
        
        return call_stack_analysis

    def compare_by_call_stack(self, file_labels: List[Tuple[str, str]], output_dir: str = ".") -> None:
        """
        基于 call stack 对比多个文件
        
        Args:
            file_labels: List of (file_path, label) tuples
            output_dir: 输出目录
        """
        print("=== 开始基于 call stack 的对比分析 ===")
        
        # 分析每个文件的 call stack
        all_call_stack_analyses = {}
        
        for file_path, label in file_labels:
            print(f"正在分析文件: {file_path} (标签: {label})")
            
            try:
                data = self.parser.load_json_file(file_path)
                call_stack_analysis = self.analyze_cpu_op_by_call_stack(data)
                all_call_stack_analyses[label] = call_stack_analysis
                print(f"  完成分析，找到 {len(call_stack_analysis)} 个唯一的 call stack")
                
            except Exception as e:
                print(f"  分析文件失败: {e}")
                continue
        
        if not all_call_stack_analyses:
            print("没有成功分析任何文件")
            return
        
        # 合并所有 call stack 分析
        merged_call_stack_analysis = self.merge_call_stack_analyses(all_call_stack_analyses)
        
        # 提取标签列表
        labels = list(all_call_stack_analyses.keys())
        
        # 生成对比结果
        self.generate_call_stack_comparison_excel(merged_call_stack_analysis, output_dir, labels)
        
        print("=== 基于 call stack 的对比分析完成 ===")

    def analyze_single_file_by_call_stack(self, data: ProfilerData, label: str, output_dir: str, output_formats: List[str] = None) -> None:
        """
        分析单个文件的 call stack
        
        Args:
            data: ProfilerData 对象
            label: 文件标签
            output_dir: 输出目录
            output_formats: 输出格式列表，如 ['json', 'xlsx']
        """
        print(f"=== 开始单文件 call stack 分析 (标签: {label}) ===")
        
        # 分析 call stack
        call_stack_analysis = self.analyze_cpu_op_by_call_stack(data)
        
        if not call_stack_analysis:
            print("没有找到 call stack 数据")
            return
        
        print(f"找到 {len(call_stack_analysis)} 个唯一的 call stack")
        
        # 生成单文件分析结果
        self.generate_single_file_call_stack_analysis(call_stack_analysis, label, output_dir, output_formats)
        
        print(f"=== 单文件 call stack 分析完成 ===")

    def generate_single_file_call_stack_analysis(self, call_stack_analysis: Dict, label: str, output_dir: str, output_formats: List[str] = None) -> None:
        """
        生成单文件的 call stack 分析结果
        
        Args:
            call_stack_analysis: call stack 分析结果
            label: 文件标签
            output_dir: 输出目录
            output_formats: 输出格式列表，如 ['json', 'xlsx']
        """
        # 生成 JSON 格式的多级结构数据
        json_data = []
        excel_rows = []
        
        for call_stack_key, call_stack_data in call_stack_analysis.items():
            call_stack = call_stack_data['call_stack']
            ops = call_stack_data['ops']
            
            # 为每个 op 创建条目
            for op_name in sorted(ops.keys()):
                op_events = ops[op_name]
                
                # 收集该 op 的所有事件信息
                input_dims_list = []
                input_strides_list = []
                input_types_list = []
                kernel_names_list = []
                
                for event_info in op_events:
                    input_dims_list.append(str(event_info['input_dims']))
                    input_strides_list.append(str(event_info['input_strides']))
                    input_types_list.append(str(event_info['input_type']))
                    kernel_names_list.extend(event_info['kernel_names'])
                
                # JSON 数据结构
                json_entry = {
                    'call_stack': ' -> '.join(call_stack),
                    'call_stack_depth': len(call_stack),
                    'op_name': op_name,
                    'input_dims': input_dims_list,
                    'input_strides': input_strides_list,
                    'input_types': input_types_list,
                    'kernel_names': list(set(kernel_names_list)),
                    'event_count': len(op_events)
                }
                
                # Excel 行数据
                excel_row = {
                    'call_stack': ' -> '.join(call_stack),
                    'call_stack_depth': len(call_stack),
                    'op_name': op_name,
                    'input_dims': '||'.join(input_dims_list) if input_dims_list else '',
                    'input_strides': '||'.join(input_strides_list) if input_strides_list else '',
                    'input_types': '||'.join(input_types_list) if input_types_list else '',
                    'kernel_names': '||'.join(set(kernel_names_list)) if kernel_names_list else '',
                    'event_count': len(op_events)
                }
                
                json_data.append(json_entry)
                excel_rows.append(excel_row)
        
        # 生成文件名
        base_name = f"{label}_single_file_call_stack_analysis"
        
        # 默认输出格式
        if output_formats is None:
            output_formats = ['json', 'xlsx']
        
        # 生成 JSON 文件
        if 'json' in output_formats:
            json_file = f"{output_dir}/{base_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"单文件 call stack 分析 JSON 文件已生成: {json_file}")
            print(f"包含 {len(json_data)} 个 call stack-op 组合")
        
        # 生成 Excel 文件
        if 'xlsx' in output_formats and excel_rows:
            df = pd.DataFrame(excel_rows)
            xlsx_file = f"{output_dir}/{base_name}.xlsx"
            try:
                df.to_excel(xlsx_file, index=False)
                print(f"单文件 call stack 分析 Excel 文件已生成: {xlsx_file}")
                print(f"包含 {len(excel_rows)} 行数据")
            except ImportError:
                csv_file = f"{output_dir}/{base_name}.csv"
                self._safe_csv_output(df, csv_file)
                print(f"单文件 call stack 分析 CSV 文件已生成: {csv_file}")
        elif 'xlsx' in output_formats and not excel_rows:
            print("没有数据可以生成单文件 call stack 分析文件")

    def merge_call_stack_analyses(self, all_call_stack_analyses: Dict) -> Dict:
        """
        合并多个文件的 call stack 分析
        
        Args:
            all_call_stack_analyses: Dict[label, call_stack_analysis]
            
        Returns:
            Dict: 合并后的 call stack 分析
        """
        merged = {}
        
        # 收集所有唯一的 call stack
        all_call_stacks = set()
        for label, analysis in all_call_stack_analyses.items():
            all_call_stacks.update(analysis.keys())
        
        # 为每个 call stack 创建合并条目
        for call_stack_key in all_call_stacks:
            call_stack_list = list(call_stack_key)
            merged_entry = {
                'call_stack': call_stack_list,
                'labels': {}
            }
            
            # 收集每个标签的数据
            for label, analysis in all_call_stack_analyses.items():
                if call_stack_key in analysis:
                    merged_entry['labels'][label] = analysis[call_stack_key]
                else:
                    # 如果某个标签没有这个 call stack，创建空条目
                    merged_entry['labels'][label] = {
                        'call_stack': call_stack_list,
                        'ops': {},
                        'first_occurrence_time': None
                    }
            
            # 按第一次出现的时间排序
            first_occurrence_times = []
            for label_data in merged_entry['labels'].values():
                if label_data['first_occurrence_time'] is not None:
                    first_occurrence_times.append(label_data['first_occurrence_time'])
            
            if first_occurrence_times:
                merged_entry['earliest_occurrence_time'] = min(first_occurrence_times)
            else:
                merged_entry['earliest_occurrence_time'] = float('inf')
            
            merged[call_stack_key] = merged_entry
        
        # 按最早出现时间排序
        sorted_merged = dict(sorted(merged.items(), key=lambda x: x[1]['earliest_occurrence_time']))
        
        return sorted_merged

    def generate_call_stack_comparison_excel(self, merged_analysis: Dict, output_dir: str, labels: List[str] = None) -> None:
        """
        生成基于 call stack 的对比 Excel 文件，使用多级结构：call stack -> op -> event_info
        
        Args:
            merged_analysis: 合并的 call stack 分析
            output_dir: 输出目录
            labels: 标签列表，用于生成文件名
        """
        # 生成 JSON 格式的多级结构数据
        json_data = []
        excel_rows = []
        
        for call_stack_key, call_stack_data in merged_analysis.items():
            call_stack = call_stack_data['call_stack']
            labels_data = call_stack_data['labels']
            labels_list = list(labels_data.keys())
            
            # 收集所有 op 名称
            all_ops = set()
            for label_data in labels_data.values():
                all_ops.update(label_data['ops'].keys())
            
            # 为每个 op 创建条目
            for op_name in sorted(all_ops):
                # JSON 数据结构
                json_entry = {
                    'call_stack': ' -> '.join(call_stack),
                    'call_stack_depth': len(call_stack),
                    'op_name': op_name,
                    'labels': {}
                }
                
                # Excel 行数据
                excel_row = {
                    'call_stack': ' -> '.join(call_stack),
                    'call_stack_depth': len(call_stack),
                    'op_name': op_name
                }
                
                # 为每个标签收集该 op 的数据
                for label in labels_list:
                    label_data = labels_data[label]
                    ops = label_data['ops']
                    
                    if op_name in ops:
                        op_events = ops[op_name]
                        # 收集该 op 的所有事件信息
                        input_dims_list = []
                        input_strides_list = []
                        input_types_list = []
                        kernel_names_list = []
                        
                        for event_info in op_events:
                            input_dims_list.append(str(event_info['input_dims']))
                            input_strides_list.append(str(event_info['input_strides']))
                            input_types_list.append(str(event_info['input_type']))
                            kernel_names_list.extend(event_info['kernel_names'])
                        
                        # JSON 数据
                        json_entry['labels'][label] = {
                            'input_dims': input_dims_list,
                            'input_strides': input_strides_list,
                            'input_types': input_types_list,
                            'kernel_names': list(set(kernel_names_list)),
                            'event_count': len(op_events)
                        }
                        
                        # Excel 数据
                        excel_row[f'{label}_input_dims'] = '||'.join(input_dims_list) if input_dims_list else ''
                        excel_row[f'{label}_input_strides'] = '||'.join(input_strides_list) if input_strides_list else ''
                        excel_row[f'{label}_input_types'] = '||'.join(input_types_list) if input_types_list else ''
                        excel_row[f'{label}_kernel_names'] = '||'.join(set(kernel_names_list)) if kernel_names_list else ''
                        excel_row[f'{label}_event_count'] = len(op_events)
                    else:
                        # 该标签没有这个 op
                        json_entry['labels'][label] = {
                            'input_dims': [],
                            'input_strides': [],
                            'input_types': [],
                            'kernel_names': [],
                            'event_count': 0
                        }
                        
                        excel_row[f'{label}_input_dims'] = ''
                        excel_row[f'{label}_input_strides'] = ''
                        excel_row[f'{label}_input_types'] = ''
                        excel_row[f'{label}_kernel_names'] = ''
                        excel_row[f'{label}_event_count'] = 0
                
                json_data.append(json_entry)
                excel_rows.append(excel_row)
        
        # 生成文件名，包含标签信息
        if labels:
            labels_str = '_'.join(labels)
            base_name = f"call_stack_comparison_{labels_str}"
        else:
            base_name = "call_stack_comparison"
        
        # 生成 JSON 文件
        json_file = f"{output_dir}/{base_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Call stack 对比 JSON 文件已生成: {json_file}")
        print(f"包含 {len(json_data)} 个 call stack-op 组合")
        
        # 生成 Excel 文件
        if excel_rows:
            df = pd.DataFrame(excel_rows)
            xlsx_file = f"{output_dir}/{base_name}.xlsx"
            try:
                df.to_excel(xlsx_file, index=False)
                print(f"Call stack 对比 Excel 文件已生成: {xlsx_file}")
                print(f"包含 {len(excel_rows)} 行数据")
            except ImportError:
                csv_file = f"{output_dir}/{base_name}.csv"
                self._safe_csv_output(df, csv_file)
                print(f"Call stack 对比 CSV 文件已生成: {csv_file}")
        else:
            print("没有数据可以生成 call stack 对比文件")

    def _print_cpu_op_summary_markdown_table(self, summary_data: List[Dict]) -> None:
        """
        以markdown表格形式打印cpu_op性能统计摘要
        
        Args:
            summary_data: 统计摘要数据列表
        """
        if not summary_data:
            print("没有数据可显示")
            return
        
        print(f"\n## CPU Op性能统计摘要表格")
        print()
        
        # 打印表头
        print("| CPU Op名称 | 调用次数 | 总耗时(μs) | 平均耗时(μs) | 最小耗时(μs) | 最大耗时(μs) | Kernel数量 | 占比(%) |")
        print("|------------|----------|------------|--------------|--------------|--------------|------------|---------|")
        
        # 打印数据行
        for item in summary_data:
            cpu_op_name = item['cpu_op_name']
            call_count = item['call_count']
            total_duration = item['total_kernel_duration']
            avg_duration = item['avg_kernel_duration']
            min_duration = item['min_kernel_duration']
            max_duration = item['max_kernel_duration']
            kernel_count = item['kernel_count']
            percentage = item['percentage_of_total']
            
            # 格式化数值
            if cpu_op_name == 'TOTAL':
                # 总计行使用粗体
                print(f"| **{cpu_op_name}** | **{call_count}** | **{total_duration:.2f}** | **{avg_duration:.2f}** | **{min_duration:.2f}** | **{max_duration:.2f}** | **{kernel_count}** | **{percentage:.2f}** |")
            else:
                print(f"| {cpu_op_name} | {call_count} | {total_duration:.2f} | {avg_duration:.2f} | {min_duration:.2f} | {max_duration:.2f} | {kernel_count} | {percentage:.2f} |")
        
        print()

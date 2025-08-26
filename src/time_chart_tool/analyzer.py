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
class PythonFunctionNode:
    """Python函数节点"""
    event: ActivityEvent
    children: List['PythonFunctionNode']
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def python_id(self) -> Optional[Union[int, str]]:
        return self.event.python_id
    
    @property
    def python_parent_id(self) -> Optional[Union[int, str]]:
        return self.event.python_parent_id
    
    @property
    def name(self) -> str:
        return self.event.name
    
    @property
    def start_time(self) -> float:
        return self.event.ts
    
    @property
    def end_time(self) -> float:
        return self.event.ts + (self.event.dur or 0)
    
    @property
    def time_range(self) -> Tuple[float, float]:
        return (self.start_time, self.end_time)
    
    def contains_time_range(self, start_time: float, end_time: float) -> bool:
        """检查是否包含指定的时间范围"""
        return self.start_time <= start_time and self.end_time >= end_time
    
    def get_call_stack(self) -> List[str]:
        """获取从根节点到当前节点的调用栈"""
        stack = [self.name]
        current = self
        while current.children:
            # 找到父节点（这里简化处理，实际应该通过python_parent_id查找）
            parent = None
            for child in current.children:
                if child.python_id == current.python_parent_id:
                    parent = child
                    break
            if parent:
                stack.append(parent.name)
                current = parent
            else:
                break
        return list(reversed(stack))


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
    
    def build_python_function_tree(self, data: ProfilerData) -> Dict[int, Dict[Union[int, str], PythonFunctionNode]]:
        """
        构建python_function树，按线程分组
        
        Args:
            data: ProfilerData 对象
            
        Returns:
            Dict[thread_id, Dict[python_id, PythonFunctionNode]]: 按线程分组的python_function节点映射
        """
        # 获取所有python_function事件
        python_events = [event for event in data.events if event.cat == 'python_function']
        
        if not python_events:
            return {}
        
        # 按线程分组
        thread_groups = defaultdict(list)
        for event in python_events:
            thread_groups[event.tid].append(event)
        
        # 为每个线程构建独立的python_function树
        thread_trees = {}
        
        for thread_id, thread_events in thread_groups.items():
            # 创建节点映射
            node_map = {}
            root_nodes = []
            
            # 第一遍：创建所有节点
            for event in thread_events:
                node = PythonFunctionNode(event=event, children=[])
                node_map[event.python_id] = node
                
                # 识别root节点（python_parent_id为null或不存在）
                if event.python_parent_id is None:
                    root_nodes.append(node)
            
            # 第二遍：建立父子关系（只在同一线程内）
            for event in thread_events:
                if event.python_parent_id is not None and event.python_parent_id in node_map:
                    parent_node = node_map[event.python_parent_id]
                    child_node = node_map[event.python_id]
                    parent_node.children.append(child_node)
            
            # 将root_nodes信息添加到节点映射中，便于后续查找
            node_map['_root_nodes'] = root_nodes
            
            thread_trees[thread_id] = node_map
        
        return thread_trees
    
    def find_containing_python_function(self, cpu_op_event: ActivityEvent, 
                                      thread_trees: Dict[int, Dict[Union[int, str], PythonFunctionNode]]) -> Optional[PythonFunctionNode]:
        """
        找到包含cpu_op时间范围的python_function（在同一线程内）
        
        Args:
            cpu_op_event: cpu_op事件
            thread_trees: 按线程分组的python_function树
            
        Returns:
            Optional[PythonFunctionNode]: 包含cpu_op的最小python_function节点
        """
        # 根据cpu_op的线程ID找到对应的python_function树
        thread_id = cpu_op_event.tid
        if thread_id not in thread_trees:
            return None
        
        python_tree = thread_trees[thread_id]
        if not python_tree or '_root_nodes' not in python_tree:
            return None
        
        cpu_start = cpu_op_event.ts
        cpu_end = cpu_op_event.ts + (cpu_op_event.dur or 0)
        
        # 从root节点开始递归查找
        root_nodes = python_tree['_root_nodes']
        best_match = None
        
        for root_node in root_nodes:
            match = self._find_containing_node_recursive(root_node, cpu_start, cpu_end)
            if match:
                # 如果找到匹配的节点，选择时间范围最小的
                if best_match is None or (match.end_time - match.start_time) < (best_match.end_time - best_match.start_time):
                    best_match = match
        
        return best_match
    
    def _find_containing_node_recursive(self, node: PythonFunctionNode, 
                                      cpu_start: float, cpu_end: float) -> Optional[PythonFunctionNode]:
        """
        递归查找包含cpu_op时间范围的python_function节点
        
        Args:
            node: 当前节点
            cpu_start: cpu_op开始时间
            cpu_end: cpu_op结束时间
            
        Returns:
            Optional[PythonFunctionNode]: 包含cpu_op的最小python_function节点
        """
        # 检查当前节点是否包含cpu_op时间范围
        contains = node.contains_time_range(cpu_start, cpu_end)
        
        if not contains:
            return None
        
        # 当前节点包含cpu_op时间范围，检查是否有子节点也包含
        best_match = node  # 当前节点作为候选
        
        # 递归检查所有子节点
        for child in node.children:
            child_match = self._find_containing_node_recursive(child, cpu_start, cpu_end)
            if child_match:
                # 如果子节点也包含，选择时间范围更小的
                if (child_match.end_time - child_match.start_time) < (best_match.end_time - best_match.start_time):
                    best_match = child_match
        
        return best_match
    
    def get_python_call_stack(self, python_node: PythonFunctionNode, 
                             thread_trees: Dict[int, Dict[Union[int, str], PythonFunctionNode]]) -> List[str]:
        """
        获取python_function的调用栈（在同一线程内）
        
        Args:
            python_node: python_function节点
            thread_trees: 按线程分组的python_function树
            
        Returns:
            List[str]: 调用栈（从根到叶子）
        """
        if python_node is None:
            return []
        
        # 找到python_node所在的线程树
        thread_id = python_node.event.tid
        if thread_id not in thread_trees:
            return []
        
        python_tree = thread_trees[thread_id]
        
        # 过滤掉无意义的函数名
        def is_meaningful_function(name: str) -> bool:
            return True
            return not (name.startswith('<built-in method') or name.startswith('torch/nn'))
        
        # 构建调用栈
        stack = []
        current = python_node
        
        # 向上遍历到根节点
        while current:
            if is_meaningful_function(current.name):
                stack.append(current.name)
            
            # 查找父节点（在同一线程内）
            parent = None
            if current.python_parent_id and current.python_parent_id in python_tree:
                parent = python_tree[current.python_parent_id]
            
            current = parent
        
        # 反转栈（从根到叶子）
        return list(reversed(stack))
    
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
        功能5.2: 分析 cpu_op 和 kernel 的映射关系，并添加python_function call stack
        
        Args:
            data: ProfilerData 对象
            
        Returns:
            Dict: 逐级映射的数据结构，包含call stack信息
        """
        # 首先按 External id 重组数据
        external_id_map = self.reorganize_by_external_id(data)
        
        # 构建python_function树（按线程分组）
        thread_trees = self.build_python_function_tree(data)
        
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
                
                # 找到包含此cpu_op的python_function（在同一线程内）
                python_node = self.find_containing_python_function(cpu_op_event, thread_trees)
                call_stack = self.get_python_call_stack(python_node, thread_trees) if python_node else []
                
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
                
                # 添加call stack信息到kernel_events
                for kernel_event in kernel_events:
                    kernel_event.call_stack = call_stack
                
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
                                'kernel_std_duration': stats.variance ** 0.5,
                                'call_stack': ' -> '.join(kernel_events[0].call_stack) if kernel_events and kernel_events[0].call_stack else ''
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
            Dict: 合并后的映射，key为(cpu_op_name, input_strides, input_dims, call_stack)，value为Dict[label, Dict[input_type, kernel_events]]
        """
        merged = defaultdict(lambda: defaultdict(dict))
        
        for label, mapping in all_mappings.items():
            for cpu_op_name, strides_map in mapping.items():
                for input_strides, dims_map in strides_map.items():
                    for input_dims, types_map in dims_map.items():
                        for input_type, kernel_events in types_map.items():
                            # 获取call stack（从第一个kernel_event中获取）
                            call_stack = kernel_events[0].call_stack if kernel_events else []
                            call_stack_key = tuple(call_stack)
                            
                            key = (cpu_op_name, input_strides, input_dims, call_stack_key)
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
        
        for (cpu_op_name, input_strides, input_dims, call_stack), label_types_map in merged_mapping.items():
            # 收集所有标签
            labels = list(label_types_map.keys())
            
            # 基础行数据
            row = {
                'cpu_op_name': cpu_op_name,
                'cpu_op_input_strides': str(input_strides),
                'cpu_op_input_dims': str(input_dims),
                'call_stack': ' -> '.join(call_stack) if call_stack else ''
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
        if len(labels) < 2:
            return
        
        # 按min_dim分组数据
        min_dim_data = defaultdict(list)
        for item in json_data:
            min_dim = item.get('mm_min_dim')
            if min_dim is not None:
                min_dim_data[min_dim].append(item)
        
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

    def analyze_by_call_stack(self, file_labels: List[Tuple[str, str]], 
                            output_dir: str = ".", 
                            output_formats: List[str] = None) -> None:
        """
        基于call stack的比较分析功能
        
        Args:
            file_labels: List of (file_path, label) tuples
            output_dir: 输出目录
            output_formats: 输出格式列表，支持 ['json', 'xlsx']
        """
        if output_formats is None:
            output_formats = ['json', 'xlsx']
        
        print("=== 开始基于call stack的比较分析 ===")
        
        # 分析多个文件
        all_mappings = self.analyze_multiple_files(file_labels)
        
        if not all_mappings:
            print("没有成功分析任何文件")
            return
        
        # 合并映射（包含call stack信息）
        if len(all_mappings) > 1:
            merged_mapping = self.merge_mappings(all_mappings)
            
            # 生成基于call stack的比较分析
            self.generate_call_stack_comparison(merged_mapping, file_labels, output_dir, output_formats)
        
        print("=== 基于call stack的分析完成 ===")

    def generate_call_stack_comparison(self, merged_mapping: Dict, 
                                     file_labels: List[Tuple[str, str]], 
                                     output_dir: str, 
                                     output_formats: List[str]) -> None:
        """
        生成基于call stack的比较分析
        
        Args:
            merged_mapping: 合并的映射数据
            file_labels: 文件标签列表
            output_dir: 输出目录
            output_formats: 输出格式
        """
        print("正在生成基于call stack的比较分析...")
        
        rows = []
        labels = [label for _, label in file_labels]
        
        for (cpu_op_name, input_strides, input_dims, call_stack), label_types_map in merged_mapping.items():
            # 基础行数据
            row = {
                'cpu_op_name': cpu_op_name,
                'cpu_op_input_strides': str(input_strides),
                'cpu_op_input_dims': str(input_dims),
                'call_stack': ' -> '.join(call_stack) if call_stack else ''
            }
            
            # 为每个标签收集数据
            for label in labels:
                if label in label_types_map:
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
                else:
                    # 如果某个标签没有数据，设置默认值
                    row[f'{label}_input_types'] = ''
                    row[f'{label}_kernel_names'] = ''
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
        
        # 生成输出文件
        if rows:
            df = pd.DataFrame(rows)
            base_name = "call_stack_comparison_analysis"
            
            if 'json' in output_formats:
                json_file = f"{output_dir}/{base_name}.json"
                df.to_json(json_file, orient='records', indent=2, force_ascii=False)
                print(f"基于call stack的比较分析JSON文件已生成: {json_file}")
            
            if 'xlsx' in output_formats:
                xlsx_file = f"{output_dir}/{base_name}.xlsx"
                try:
                    df.to_excel(xlsx_file, index=False)
                    print(f"基于call stack的比较分析Excel文件已生成: {xlsx_file}")
                except ImportError:
                    csv_file = f"{output_dir}/{base_name}.csv"
                    self._safe_csv_output(df, csv_file)
                    print(f"基于call stack的比较分析CSV文件已生成: {csv_file}")
            
            print(f"基于call stack的比较分析完成，共处理 {len(rows)} 条记录")
        else:
            print("没有数据可以生成基于call stack的比较分析文件")

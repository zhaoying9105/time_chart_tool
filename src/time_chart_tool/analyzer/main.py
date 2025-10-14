"""
主分析器模块
"""

import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from ..parser import PyTorchProfilerParser
from .stages import DataPostProcessor, DataAggregator, DataComparator, DataPresenter
from .communication import CommunicationAnalyzer
from .utils import process_single_file_parallel, AggregatedData


class Analyzer:
    """重构后的分析器，按照4个stage组织"""
    
    def __init__(self, step_idx: Optional[int] = None, coarse_call_stack: bool = False):
        self.parser = PyTorchProfilerParser(step_idx=step_idx)
        self.coarse_call_stack = coarse_call_stack
        self.postprocessor = DataPostProcessor(coarse_call_stack=coarse_call_stack)
        self.aggregator = DataAggregator(coarse_call_stack=coarse_call_stack)
        self.comparator = DataComparator()
        self.presenter = DataPresenter()
        self.comm_analyzer = CommunicationAnalyzer()
    
    # ==================== Stage 1: 数据后处理 ====================
    
    def stage1_data_postprocessing(self, data, 
                                 include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None,
                                 include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None):
        """Stage 1: 数据后处理"""
        return self.postprocessor.stage1_data_postprocessing(data, 
                                                           include_op_patterns, exclude_op_patterns,
                                                           include_kernel_patterns, exclude_kernel_patterns)
    
    # ==================== Stage 2: 数据聚合 ====================
    
    def stage2_data_aggregation(self, cpu_events_by_external_id, kernel_events_by_external_id, aggregation_spec: str = 'name', call_stack_source: str = 'tree'):
        """Stage 2: 数据聚合"""
        return self.aggregator.stage2_data_aggregation(cpu_events_by_external_id, kernel_events_by_external_id, aggregation_spec, call_stack_source)
    
    # ==================== Stage 3: 数据比较 ====================
    
    def stage3_comparison(self, single_file_data=None, multiple_files_data=None, aggregation_spec: str = 'name'):
        """Stage 3: 数据比较"""
        return self.comparator.stage3_comparison(single_file_data, multiple_files_data, aggregation_spec)
    
    # ==================== Stage 4: 数据展示 ====================
    
    def stage4_presentation(self, comparison_result, **kwargs):
        """Stage 4: 数据展示"""
        return self.presenter.stage4_presentation(comparison_result, **kwargs)
    
    # ==================== 公共接口方法 ====================
    
    def analyze_single_file(self, file_path: str, aggregation_spec: str = 'name', 
                           show_dtype: bool = False, show_shape: bool = False,
                           show_kernel_names: bool = False, show_kernel_duration: bool = False,
                           show_timestamp: bool = False, show_readable_timestamp: bool = False,
                           show_kernel_timestamp: bool = False, show_name: bool = False,
                           output_dir: str = ".", label: str = None, print_markdown: bool = False,
                           call_stack_source: str = 'tree', not_show_fwd_bwd_type: bool = False,
                           step_idx: Optional[int] = None) -> List[Path]:
        """
        分析单个文件
        
        Args:
            file_path: JSON文件路径
            aggregation_spec: 聚合字段组合
            show_dtype: 是否显示数据类型
            show_shape: 是否显示形状信息
            show_kernel_names: 是否显示kernel名称
            show_kernel_duration: 是否显示kernel持续时间
            show_timestamp: 是否显示时间戳
            show_readable_timestamp: 是否显示可读时间戳
            show_kernel_timestamp: 是否显示kernel时间戳
            show_name: 是否显示名称
            output_dir: 输出目录
            label: 文件标签
            print_markdown: 是否打印markdown表格
            call_stack_source: 调用栈来源，'args' 或 'tree'
            not_show_fwd_bwd_type: 是否不显示fwd_bwd_type列
            step_idx: 指定要分析的step索引，如果不指定则分析所有step
            
        Returns:
            List[Path]: 生成的文件路径列表
        """
        print(f"=== 分析单个文件: {file_path} ===")
        
        # 加载数据
        data = self.parser.load_json_file(file_path)
        
        # 如果需要使用调用栈树，先构建调用栈树
        if call_stack_source == 'tree':
            print("构建调用栈树...")
            self.parser.build_call_stack_trees()
        
        # Stage 1: 数据后处理
        cpu_events_by_external_id, kernel_events_by_external_id = self.stage1_data_postprocessing(data)
        
        # Stage 2: 数据聚合
        aggregated_data = self.stage2_data_aggregation(cpu_events_by_external_id, kernel_events_by_external_id, aggregation_spec, call_stack_source)
        
        # Stage 3: 数据比较
        comparison_result = self.stage3_comparison(single_file_data=aggregated_data, aggregation_spec=aggregation_spec)
        
        # Stage 4: 数据展示
        generated_files = self.stage4_presentation(
            comparison_result=comparison_result,
            output_dir=output_dir,
            show_dtype=show_dtype,
            show_shape=show_shape,
            show_kernel_names=show_kernel_names,
            show_kernel_duration=show_kernel_duration,
            show_timestamp=show_timestamp,
            show_readable_timestamp=show_readable_timestamp,
            show_kernel_timestamp=show_kernel_timestamp,
            show_name=show_name,
            aggregation_spec=aggregation_spec,
            label=label,
            print_markdown=print_markdown,
            not_show_fwd_bwd_type=not_show_fwd_bwd_type
        )
        
        return generated_files
    
    def analyze_single_file_with_glob(self, file_paths: List[str], aggregation_spec: str = 'name',
                                     show_dtype: bool = False, show_shape: bool = False,
                                     show_kernel_names: bool = False, show_kernel_duration: bool = False,
                                     show_timestamp: bool = False, show_readable_timestamp: bool = False,
                                     show_kernel_timestamp: bool = False, show_name: bool = False,
                                     output_dir: str = ".", label: str = None, print_markdown: bool = False,
                                     include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None,
                                     include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None,
                                     call_stack_source: str = 'tree', not_show_fwd_bwd_type: bool = False,
                                     step_idx: Optional[int] = None) -> List[Path]:
        """
        分析多个文件（使用glob模式）
        
        Args:
            file_paths: 文件路径列表
            aggregation_spec: 聚合字段组合
            show_dtype: 是否显示数据类型
            show_shape: 是否显示形状信息
            show_kernel_names: 是否显示kernel名称
            show_kernel_duration: 是否显示kernel持续时间
            show_timestamp: 是否显示时间戳
            show_readable_timestamp: 是否显示可读时间戳
            show_kernel_timestamp: 是否显示kernel时间戳
            show_name: 是否显示名称
            output_dir: 输出目录
            label: 文件标签
            print_markdown: 是否打印markdown表格
            call_stack_source: 调用栈来源，'args' 或 'tree'
            not_show_fwd_bwd_type: 是否不显示fwd_bwd_type列
            step_idx: 指定要分析的step索引，如果不指定则分析所有step
            
        Returns:
            List[Path]: 生成的文件路径列表
        """
        print(f"=== 分析多个文件: {len(file_paths)} 个文件 ===")
        
        # 并行处理文件
        aggregated_data_list = self._process_files_parallel(
            file_paths, aggregation_spec, mp.cpu_count(),
            include_op_patterns, exclude_op_patterns,
            include_kernel_patterns, exclude_kernel_patterns,
            call_stack_source, step_idx
        )
        
        # 合并相同标签的文件
        merged_data = self._merge_same_label_files(aggregated_data_list, aggregation_spec)
        
        # 收集统计信息
        per_rank_stats = self.postprocessor._collect_per_rank_statistics(file_paths, aggregated_data_list)
        
        # Stage 3: 数据比较
        comparison_result = self.stage3_comparison(single_file_data=merged_data, aggregation_spec=aggregation_spec)
        
        # Stage 4: 数据展示
        generated_files = self.stage4_presentation(
            comparison_result=comparison_result,
            output_dir=output_dir,
            show_dtype=show_dtype,
            show_shape=show_shape,
            show_kernel_names=show_kernel_names,
            show_kernel_duration=show_kernel_duration,
            show_timestamp=show_timestamp,
            show_readable_timestamp=show_readable_timestamp,
            show_kernel_timestamp=show_kernel_timestamp,
            show_name=show_name,
            aggregation_spec=aggregation_spec,
            label=label,
            print_markdown=print_markdown,
            include_op_patterns=include_op_patterns,
            exclude_op_patterns=exclude_op_patterns,
            include_kernel_patterns=include_kernel_patterns,
            exclude_kernel_patterns=exclude_kernel_patterns,
            per_rank_stats=per_rank_stats,
            not_show_fwd_bwd_type=not_show_fwd_bwd_type
        )
        
        return generated_files
    
    def analyze_multiple_files(self, file_labels: List[Tuple[List[str], str]], aggregation_spec: str = 'name',
                              show_dtype: bool = False, show_shape: bool = False,
                              show_kernel_names: bool = False, show_kernel_duration: bool = False,
                              show_timestamp: bool = False, show_readable_timestamp: bool = False,
                              show_kernel_timestamp: bool = False, show_name: bool = False,
                              special_matmul: bool = False, output_dir: str = ".",
                              compare_dtype: bool = False, compare_shape: bool = False, compare_name: bool = False,
                              print_markdown: bool = False, max_workers: int = None,
                              include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None,
                              include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None,
                              call_stack_source: str = 'tree', not_show_fwd_bwd_type: bool = False) -> List[Path]:
        """
        分析多个文件并对比
        
        Args:
            file_labels: 文件标签列表 [(file_paths, label), ...]
            aggregation_spec: 聚合字段组合
            show_dtype: 是否显示数据类型
            show_shape: 是否显示形状信息
            show_kernel_names: 是否显示kernel名称
            show_kernel_duration: 是否显示kernel持续时间
            show_timestamp: 是否显示时间戳
            show_readable_timestamp: 是否显示可读时间戳
            show_kernel_timestamp: 是否显示kernel时间戳
            show_name: 是否显示名称
            special_matmul: 是否进行特殊的matmul分析
            output_dir: 输出目录
            compare_dtype: 是否比较数据类型
            compare_shape: 是否比较形状
            compare_name: 是否比较名称
            print_markdown: 是否打印markdown表格
            max_workers: 最大工作进程数
            include_op_patterns: 包含的操作名称模式列表
            exclude_op_patterns: 排除的操作名称模式列表
            include_kernel_patterns: 包含的kernel名称模式列表
            exclude_kernel_patterns: 排除的kernel名称模式列表
            call_stack_source: 调用栈来源，'args' 或 'tree'
            not_show_fwd_bwd_type: 是否不显示fwd_bwd_type列
            
        Returns:
            List[Path]: 生成的文件路径列表
        """
        print(f"=== 分析多个文件并对比: {len(file_labels)} 个标签 ===")
        
        # 处理每个标签的文件
        multiple_files_data = {}
        file_labels_list = []
        
        for file_paths, label in file_labels:
            # 并行处理文件
            aggregated_data_list = self._process_files_parallel(file_paths, aggregation_spec, max_workers or mp.cpu_count(), 
                                                              include_op_patterns, exclude_op_patterns,
                                                              include_kernel_patterns, exclude_kernel_patterns, call_stack_source)
            
            # 合并相同标签的文件
            merged_data = self._merge_same_label_files(aggregated_data_list, aggregation_spec)
            
            multiple_files_data[label] = merged_data
            file_labels_list.append(label)
        
        # Stage 3: 数据比较
        comparison_result = self.stage3_comparison(multiple_files_data=multiple_files_data, aggregation_spec=aggregation_spec)
        
        # Stage 4: 数据展示
        generated_files = self.stage4_presentation(
            comparison_result=comparison_result,
            output_dir=output_dir,
            show_dtype=show_dtype,
            show_shape=show_shape,
            show_kernel_names=show_kernel_names,
            show_kernel_duration=show_kernel_duration,
            show_timestamp=show_timestamp,
            show_readable_timestamp=show_readable_timestamp,
            show_kernel_timestamp=show_kernel_timestamp,
            show_name=show_name,
            aggregation_spec=aggregation_spec,
            special_matmul=special_matmul,
            compare_dtype=compare_dtype,
            compare_shape=compare_shape,
            compare_name=compare_name,
            file_labels=file_labels_list,
            print_markdown=print_markdown,
            include_op_patterns=include_op_patterns,
            exclude_op_patterns=exclude_op_patterns,
            include_kernel_patterns=include_kernel_patterns,
            exclude_kernel_patterns=exclude_kernel_patterns,
            not_show_fwd_bwd_type=not_show_fwd_bwd_type
        )
        
        return generated_files
    
    def analyze_communication_performance(self, pod_dir: str, step: Optional[int] = None, comm_idx: Optional[int] = None,
                                         fastest_card_idx: Optional[int] = None, slowest_card_idx: Optional[int] = None,
                                         kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL",
                                         prev_kernel_pattern: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL_BF16_ADD",
                                         output_dir: str = ".",
                                         show_dtype: bool = False, show_shape: bool = False,
                                         show_kernel_names: bool = False, show_kernel_duration: bool = False,
                                         show_timestamp: bool = False, show_readable_timestamp: bool = False,
                                         show_kernel_timestamp: bool = False) -> List[Path]:
        """
        分析通信性能
        
        Args:
            pod_dir: Pod目录路径
            step: 指定要分析的step
            comm_idx: 指定要分析的通信操作索引
            fastest_card_idx: 指定最快卡的索引
            slowest_card_idx: 指定最慢卡的索引
            kernel_prefix: 通信kernel前缀
            prev_kernel_pattern: 上一个通信kernel的匹配模式
            output_dir: 输出目录
            show_dtype: 是否显示数据类型
            show_shape: 是否显示形状信息
            show_kernel_names: 是否显示kernel名称
            show_kernel_duration: 是否显示kernel持续时间
            show_timestamp: 是否显示时间戳
            show_readable_timestamp: 是否显示可读时间戳
            show_kernel_timestamp: 是否显示kernel时间戳
            
        Returns:
            List[Path]: 生成的文件路径列表
        """
        return self.comm_analyzer.analyze_communication_performance(
            pod_dir=pod_dir, step=step, comm_idx=comm_idx,
            fastest_card_idx=fastest_card_idx, slowest_card_idx=slowest_card_idx,
            kernel_prefix=kernel_prefix, prev_kernel_pattern=prev_kernel_pattern,
            output_dir=output_dir, show_dtype=show_dtype, show_shape=show_shape,
            show_kernel_names=show_kernel_names, show_kernel_duration=show_kernel_duration,
            show_timestamp=show_timestamp, show_readable_timestamp=show_readable_timestamp,
            show_kernel_timestamp=show_kernel_timestamp
        )
    
    # ==================== 辅助方法 ====================
    
    def _process_files_parallel(self, file_paths: List[str], aggregation_spec: str, max_workers: int, 
                              include_op_patterns: List[str] = None, exclude_op_patterns: List[str] = None,
                              include_kernel_patterns: List[str] = None, exclude_kernel_patterns: List[str] = None,
                              call_stack_source: str = 'tree', step_idx: Optional[int] = None) -> List[Dict[Union[str, tuple], AggregatedData]]:
        """
        并行处理文件
        
        Args:
            file_paths: 文件路径列表
            aggregation_spec: 聚合字段组合
            max_workers: 最大工作进程数
            include_op_patterns: 包含的操作名称模式列表
            exclude_op_patterns: 排除的操作名称模式列表
            include_kernel_patterns: 包含的kernel名称模式列表
            exclude_kernel_patterns: 排除的kernel名称模式列表
            call_stack_source: 调用栈来源，'args' 或 'tree'
            step_idx: 指定要分析的step索引，如果不指定则分析所有step
            
        Returns:
            List[Dict[Union[str, tuple], AggregatedData]]: 每个文件的聚合数据列表
        """
        print(f"并行处理 {len(file_paths)} 个文件，使用 {max_workers} 个进程")
        
        aggregated_data_list = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_file = {
                executor.submit(process_single_file_parallel, file_path, aggregation_spec,
                              include_op_patterns, exclude_op_patterns, include_kernel_patterns, exclude_kernel_patterns, call_stack_source, step_idx, self.coarse_call_stack): file_path
                for file_path in file_paths
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    _, aggregated_data = future.result()
                    aggregated_data_list.append(aggregated_data)
                except Exception as e:
                    print(f"处理文件 {file_path} 失败: {e}")
                    aggregated_data_list.append({})
        
        return aggregated_data_list
    
    def _merge_same_label_files(self, aggregated_data_list: List[Dict[Union[str, tuple], AggregatedData]], 
                               aggregation_spec: str) -> Dict[Union[str, tuple], AggregatedData]:
        """
        合并相同标签的文件数据
        
        Args:
            aggregated_data_list: 每个文件的聚合数据列表
            aggregation_spec: 聚合字段组合
            
        Returns:
            Dict[Union[str, tuple], AggregatedData]: 合并后的数据
        """
        print("合并相同标签的文件数据")
        
        merged_data = {}
        
        for aggregated_data in aggregated_data_list:
            for key, agg_data in aggregated_data.items():
                if key in merged_data:
                    # 合并数据
                    merged_data[key] = self._merge_aggregated_data_for_key([merged_data[key], agg_data])
                else:
                    merged_data[key] = agg_data
        
        print(f"合并后得到 {len(merged_data)} 个唯一的键")
        return merged_data
    
    def _merge_aggregated_data_for_key(self, aggregated_data_list: List[AggregatedData]) -> AggregatedData:
        """
        合并同一个键的多个聚合数据
        
        Args:
            aggregated_data_list: 聚合数据列表
            
        Returns:
            AggregatedData: 合并后的聚合数据
        """
        if not aggregated_data_list:
            return AggregatedData([], [], "")
        
        if len(aggregated_data_list) == 1:
            return aggregated_data_list[0]
        
        # 合并所有CPU事件和kernel事件
        all_cpu_events = []
        all_kernel_events = []
        
        for agg_data in aggregated_data_list:
            all_cpu_events.extend(agg_data.cpu_events)
            all_kernel_events.extend(agg_data.kernel_events)
        
        # 使用第一个数据的键
        key = aggregated_data_list[0].key
        
        return AggregatedData(
            cpu_events=all_cpu_events,
            kernel_events=all_kernel_events,
            key=key
        )

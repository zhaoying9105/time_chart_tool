"""
对比命令模块
"""

import time
import random
from pathlib import Path
from typing import List, Tuple

from ..validators import validate_aggregation_fields, parse_show_options, parse_compare_options, validate_file, validate_filter_options, parse_filter_patterns
from ..file_utils import parse_file_label
from ...analyzer import Analyzer


class CompareCommand:
    """对比命令处理器"""
    
    def __init__(self):
        # analyzer 将在 run 方法中创建，因为需要根据参数设置
        self.analyzer = None
    
    def run(self, args) -> int:
        """运行多文件对比分析"""
        print(f"=== 多文件对比分析 ===")
        print(f"聚合字段: {args.aggregation}")
        print(f"显示选项: {args.show if args.show else '无'}")
        print(f"打印markdown表格: {args.print_markdown}")
        print(f"特殊matmul: {args.special_matmul}")
        print(f"每个标签最大文件数: {args.max_files_per_label if args.max_files_per_label else '不限制'}")
        print(f"随机种子: {args.random_seed}")
        print(f"输出格式: {args.output_format}")
        print(f"输出目录: {args.output_dir}")
        print(f"包含操作模式: {args.include_op if args.include_op else '无'}")
        print(f"排除操作模式: {args.exclude_op if args.exclude_op else '无'}")
        print(f"包含kernel模式: {args.include_kernel if args.include_kernel else '无'}")
        print(f"排除kernel模式: {args.exclude_kernel if args.exclude_kernel else '无'}")
        if hasattr(args, 'call_stack_source'):
            print(f"调用栈来源: {args.call_stack_source}")
        print()
        
        # 验证聚合字段
        try:
            aggregation_fields = validate_aggregation_fields(args.aggregation)
            print(f"聚合字段验证通过: {aggregation_fields}")
        except ValueError as e:
            print(f"错误: 聚合字段验证失败 - {e}")
            return 1
        
        # 解析show选项
        try:
            show_options = parse_show_options(args.show)
            print(f"显示选项: {show_options}")
        except ValueError as e:
            print(f"错误: 显示选项解析失败 - {e}")
            return 1
        
        # 解析compare选项
        try:
            compare_options = parse_compare_options(args.compare)
            print(f"比较选项: {compare_options}")
        except ValueError as e:
            print(f"错误: 比较选项解析失败 - {e}")
            return 1
        
        # 验证过滤选项
        try:
            validate_filter_options(args.include_op, args.exclude_op, args.include_kernel, args.exclude_kernel)
            print("过滤选项验证通过")
        except ValueError as e:
            print(f"错误: 过滤选项验证失败 - {e}")
            return 1
        
        # 解析过滤模式
        include_op_patterns = parse_filter_patterns(args.include_op) if args.include_op else None
        exclude_op_patterns = parse_filter_patterns(args.exclude_op) if args.exclude_op else None
        include_kernel_patterns = parse_filter_patterns(args.include_kernel) if args.include_kernel else None
        exclude_kernel_patterns = parse_filter_patterns(args.exclude_kernel) if args.exclude_kernel else None
        
        if include_op_patterns:
            print(f"包含操作模式: {include_op_patterns}")
        if exclude_op_patterns:
            print(f"排除操作模式: {exclude_op_patterns}")
        if include_kernel_patterns:
            print(f"包含kernel模式: {include_kernel_patterns}")
        if exclude_kernel_patterns:
            print(f"排除kernel模式: {exclude_kernel_patterns}")
        
        # 检查聚合字段和显示选项是否重复
        show_fields = set()
        for option, enabled in show_options.items():
            if enabled and option in ['dtype', 'shape']:
                show_fields.add(option)
        
        aggregation_fields_set = set(aggregation_fields)
        overlap = show_fields.intersection(aggregation_fields_set)
        if overlap:
            print(f"警告: 聚合字段 {list(overlap)} 与显示选项重复，将跳过重复的显示列")
        
        # 解析文件列表
        file_labels = []
        for file_label in args.files:
            try:
                file_paths, label = parse_file_label(file_label)
                
                # 验证所有文件
                valid_files = []
                for file_path in file_paths:
                    if validate_file(file_path):
                        valid_files.append(file_path)
                    else:
                        print(f"警告: 跳过无效文件 {file_path}")
                
                if not valid_files:
                    print(f"错误: 标签 {label} 没有有效的文件")
                    return 1
                    
                # 随机采样文件（如果需要）
                if args.max_files_per_label and len(valid_files) > args.max_files_per_label:
                    random.seed(args.random_seed)
                    sampled_files = random.sample(valid_files, args.max_files_per_label)
                    print(f"  {label}: 从 {len(valid_files)} 个文件中随机采样 {len(sampled_files)} 个文件")
                    file_labels.append((sampled_files, label))
                else:
                    file_labels.append((valid_files, label))
                
                # 显示文件信息
                final_files = file_labels[-1][0]  # 获取最终使用的文件列表
                if len(final_files) == 1:
                    print(f"  {label}: {final_files[0]}")
                else:
                    print(f"  {label}: {len(final_files)} 个文件")
                    for i, file_path in enumerate(final_files[:3]):  # 只显示前3个
                        print(f"    {i+1}. {file_path}")
                    if len(final_files) > 3:
                        print(f"    ... 还有 {len(final_files) - 3} 个文件")
                        
            except ValueError as e:
                print(f"错误: {e}")
                return 1
        
        if len(file_labels) < 2:
            print("错误: 对比分析需要至少2个文件")
            return 1
        
        # 计算总文件数
        total_files = sum(len(files) for files, _ in file_labels)
        print(f"\n将分析 {len(file_labels)} 个标签，共 {total_files} 个文件")
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            start_time = time.time()
            
            # 创建 Analyzer 实例，传递 coarse_call_stack 参数
            coarse_call_stack = getattr(args, 'coarse_call_stack', False)
            self.analyzer = Analyzer(coarse_call_stack=coarse_call_stack)
            
            # 使用新的分析流程
            generated_files = self.analyzer.analyze_multiple_files(
                file_labels=file_labels,
                aggregation_spec=args.aggregation,
                show_dtype=show_options['dtype'],
                show_shape=show_options['shape'],
                show_kernel_names=show_options['kernel_names'],
                show_kernel_duration=show_options['kernel_duration'],
                show_timestamp=show_options['timestamp'],
                show_readable_timestamp=show_options['readable_timestamp'],
                show_kernel_timestamp=show_options['kernel_timestamp'],
                show_name=show_options['name'],
                show_call_stack=show_options['call_stack'],
                special_matmul=args.special_matmul,
                output_dir=str(output_dir),
                compare_dtype=compare_options['dtype'],
                compare_shape=compare_options['shape'],
                compare_name=compare_options['name'],
                print_markdown=args.print_markdown,
                max_workers=args.max_workers,
                include_op_patterns=include_op_patterns,
                exclude_op_patterns=exclude_op_patterns,
                include_kernel_patterns=include_kernel_patterns,
                exclude_kernel_patterns=exclude_kernel_patterns,
                call_stack_source=args.call_stack_source,
                not_show_fwd_bwd_type=getattr(args, 'not_show_fwd_bwd_type', False),
            )
            
            total_time = time.time() - start_time
            print(f"\n分析完成，总耗时: {total_time:.2f} 秒")
            
            # 显示生成的文件
            print("\n生成的文件:")
            for file_path in generated_files:
                print(f"  {file_path}")
            
            return 0
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            return 1

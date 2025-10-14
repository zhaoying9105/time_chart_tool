"""
分析命令模块
"""

import time
from pathlib import Path
from typing import List

from ..validators import validate_aggregation_fields, parse_show_options, validate_file, \
    validate_filter_options, parse_filter_patterns
from ..file_utils import parse_file_paths
from ...analyzer import Analyzer


class AnalysisCommand:
    """分析命令处理器"""
    
    def __init__(self):
        # analyzer 将在 run 方法中创建，因为需要根据参数设置
        self.analyzer = None
    
    def run(self, args) -> int:
        """运行单个或多个文件分析"""
        print(f"=== 文件分析 ===")
        print(f"文件模式: {args.file}")
        print(f"标签: {args.label}")
        print(f"聚合字段: {args.aggregation}")
        print(f"显示选项: {args.show if args.show else '无'}")
        print(f"打印markdown表格: {args.print_markdown}")
        print(f"输出格式: {args.output_format}")
        print(f"输出目录: {args.output_dir}")
        if hasattr(args, 'max_files') and args.max_files:
            print(f"最大文件数: {args.max_files}")
            print(f"随机种子: {getattr(args, 'random_seed', 42)}")
        if hasattr(args, 'call_stack_source'):
            print(f"调用栈来源: {args.call_stack_source}")
        if hasattr(args, 'step_idx') and args.step_idx is not None:
            print(f"Step索引: {args.step_idx}")
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

        # 验证过滤选项（与 compare 保持一致）
        try:
            validate_filter_options(args.include_op, args.exclude_op, args.include_kernel, args.exclude_kernel)
            print("过滤选项验证通过")
        except ValueError as e:
            print(f"错误: 过滤选项验证失败 - {e}")
            return 1

        # 解析过滤模式
        include_op_patterns = parse_filter_patterns(args.include_op) if getattr(args, 'include_op', None) else None
        exclude_op_patterns = parse_filter_patterns(args.exclude_op) if getattr(args, 'exclude_op', None) else None
        include_kernel_patterns = parse_filter_patterns(args.include_kernel) if getattr(args, 'include_kernel', None) else None
        exclude_kernel_patterns = parse_filter_patterns(args.exclude_kernel) if getattr(args, 'exclude_kernel', None) else None
        
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
        
        # 解析文件路径，支持 glob 模式
        try:
            file_paths = parse_file_paths(args.file)
            if not file_paths:
                print(f"错误: 没有找到匹配的文件: {args.file}")
                return 1
            
            # 应用文件数量限制
            if hasattr(args, 'max_files') and args.max_files and len(file_paths) > args.max_files:
                import random
                random_seed = getattr(args, 'random_seed', 42)
                random.seed(random_seed)
                file_paths = random.sample(file_paths, args.max_files)
                print(f"随机采样 {args.max_files} 个文件 (种子: {random_seed})")
            
            print(f"找到 {len(file_paths)} 个文件:")
            for i, file_path in enumerate(file_paths[:5]):  # 只显示前5个
                print(f"  {i+1}. {file_path}")
            if len(file_paths) > 5:
                print(f"  ... 还有 {len(file_paths) - 5} 个文件")
        except Exception as e:
            print(f"错误: 解析文件路径失败 - {e}")
            return 1
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            start_time = time.time()
            
            # 创建 Analyzer 实例，传递 coarse_call_stack 参数
            coarse_call_stack = getattr(args, 'coarse_call_stack', False)
            self.analyzer = Analyzer(step_idx=getattr(args, 'step_idx', None), coarse_call_stack=coarse_call_stack)
            
            # 使用新的分析流程 - 支持多文件独立解析和聚合
            generated_files = self.analyzer.analyze_single_file_with_glob(
                file_paths=file_paths,
                aggregation_spec=args.aggregation,
                show_dtype=show_options['dtype'],
                show_shape=show_options['shape'],
                show_kernel_names=show_options['kernel_names'],
                show_kernel_duration=show_options['kernel_duration'],
                show_timestamp=show_options['timestamp'],
                show_readable_timestamp=show_options['readable_timestamp'],
                show_kernel_timestamp=show_options['kernel_timestamp'],
                show_name=show_options['name'],
                output_dir=str(output_dir),
                label=args.label,
                print_markdown=args.print_markdown,
                include_op_patterns=include_op_patterns,
                exclude_op_patterns=exclude_op_patterns,
                include_kernel_patterns=include_kernel_patterns,
                exclude_kernel_patterns=exclude_kernel_patterns,
                call_stack_source=args.call_stack_source,
                not_show_fwd_bwd_type=getattr(args, 'not_show_fwd_bwd_type', False),
                step_idx=getattr(args, 'step_idx', None),
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

"""
分析命令模块
"""

import time
from pathlib import Path
from typing import List

from ..validators import validate_aggregation_fields, parse_show_options, validate_file
from ..file_utils import parse_file_paths
from ...analyzer import Analyzer


class AnalysisCommand:
    """分析命令处理器"""
    
    def __init__(self):
        self.analyzer = Analyzer()
    
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
                output_dir=str(output_dir),
                label=args.label,
                print_markdown=args.print_markdown,
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

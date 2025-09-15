#!/usr/bin/env python3
"""
Time Chart Tool 命令行工具 - 重构版本

支持分析多个 timechart JSON 文件，生成单个文件分析和对比分析结果。
支持 JSON 和 XLSX 输出格式。
"""

import argparse
import sys
import time
import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import json

from .analyzer import Analyzer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Time Chart Tool - 分析多个 timechart JSON 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析单个文件 (基于CPU操作，默认方法，不包含kernel信息)
  time-chart-tool analysis file.json --label "baseline" --aggregation on_op_name --output-format json,xlsx
  
  # 分析单个文件 (基于CPU操作，输出包含kernel信息)
  time-chart-tool analysis file.json --label "baseline" --aggregation on_op_name --show-kernel-names --show-kernel-duration --output-format json,xlsx
  
  # 分析单个文件并在stdout中打印markdown表格
  time-chart-tool analysis file.json --label "baseline" --aggregation on_op_name --show-kernel-duration --print-markdown
  
  # 分析单个文件 (基于调用栈，输出包含shape和strides信息)
  time-chart-tool analysis file.json --label "baseline" --aggregation on_call_stack --show-shape --output-format json,xlsx
  
  # 分析单个文件 (基于调用栈，输出包含所有信息)
  time-chart-tool analysis file.json --label "baseline" --aggregation on_call_stack --show-dtype --show-shape --show-kernel-names --show-kernel-duration --output-format json,xlsx
  
  # 分析单个文件 (按CPU操作启动时间排序)
  time-chart-tool analysis file.json --label "baseline" --aggregation on_op_timestamp --show-kernel-duration --output-format json,xlsx
  
  # 分析单个文件 (显示CPU操作启动时间戳)
  time-chart-tool analysis file.json --label "baseline" --aggregation on_op_name --show-timestamp --output-format json,xlsx
  
  # 基于CPU操作对比多个文件 (默认方法，输出不包含kernel信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation on_op_name --output-format json,xlsx
  
  # 基于CPU操作对比多个文件 (输出包含kernel信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation on_op_name --show-kernel-names --show-kernel-duration --output-format json,xlsx
  
  # 基于调用栈对比多个文件 (输出包含shape和strides信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation on_call_stack --show-shape --output-format json,xlsx
  
  # 基于调用栈对比多个文件 (输出包含所有信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation on_call_stack --show-dtype --show-shape --show-kernel-names --show-kernel-duration --output-format json,xlsx
  
  # 按CPU操作启动时间排序对比多个文件
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation on_op_timestamp --show-kernel-duration --output-format json,xlsx
  
  # 对比多个文件 (显示CPU操作启动时间戳)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation on_op_name --show-timestamp --output-format json,xlsx
  
  # 对比多个文件并包含特殊的matmul分析
  time-chart-tool compare file1.json:fp32 file2.json:bf16 --aggregation on_op_name --special-matmul --output-format json,xlsx
  
  # 混合模式：支持单文件、多文件、目录混合使用
  time-chart-tool compare single_file.json:baseline "dir/*.json":test --aggregation on_op_name --output-format json,xlsx
  
  # 多文件模式：同一标签下多个文件自动聚合
  time-chart-tool compare "file1.json,file2.json,file3.json":baseline "file4.json,file5.json":optimized --aggregation on_op_name --output-format json,xlsx
  
  # 目录模式：自动查找目录下所有json文件
  time-chart-tool compare step1_results/:baseline step2_results/:optimized --aggregation on_op_name --output-format json,xlsx
  
  # 控制每个标签的文件数量，确保比较公平性
  time-chart-tool compare "dir1/*.json":baseline "dir2/*.json":optimized --max-files-per-label 10 --random-seed 42 --aggregation on_op_name
  
  # 混合模式：单文件 vs 多文件（限制数量）
  time-chart-tool compare single_file.json:reference "multi_files/*.json":test --max-files-per-label 5 --aggregation on_op_name
  
  # 只输出 JSON 格式
  time-chart-tool analysis file.json --aggregation on_call_stack --output-format json
  time-chart-tool compare file1.json:fp32 file2.json:tf32 --aggregation on_op_name --output-format json
  
  # 只输出 XLSX 格式
  time-chart-tool analysis file.json --aggregation on_op_name --output-format xlsx
  time-chart-tool compare file1.json:baseline file2.json:optimized --aggregation on_call_stack --output-format xlsx
  
  # 按时间排序分析
  time-chart-tool analysis file.json --aggregation on_op_timestamp --output-format json,xlsx
  time-chart-tool compare file1.json:baseline file2.json:optimized --aggregation on_op_timestamp --output-format xlsx
  
  # 显示时间戳分析
  time-chart-tool analysis file.json --aggregation on_op_name --show-timestamp --output-format json,xlsx
  time-chart-tool compare file1.json:baseline file2.json:optimized --aggregation on_op_name --show-timestamp --output-format xlsx
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # analysis 命令 - 分析单个文件 (原来的 single 命令)
    analysis_parser = subparsers.add_parser('analysis', help='分析单个 JSON 文件')
    analysis_parser.add_argument('file', help='要分析的 JSON 文件路径')
    analysis_parser.add_argument('--label', default='single_file', help='文件标签 (默认: single_file)')
    analysis_parser.add_argument('--aggregation', choices=['on_op_name', 'on_op_shape', 'on_call_stack', 'on_op_timestamp'], 
                                default='on_op_name',
                                help='聚合方法: on_op_name (基于操作名) 或 on_op_shape (基于操作形状) 或 on_call_stack (基于调用栈) 或 on_op_timestamp (按CPU操作启动时间排序) (默认: on_op_name)')
    analysis_parser.add_argument('--show-dtype', action='store_true', 
                                help='在输出结果时是否展示 dtype 信息 (默认: False)')
    analysis_parser.add_argument('--show-shape', action='store_true', 
                                help='在输出结果时是否展示 shape 和 strides 信息 (默认: False)')
    analysis_parser.add_argument('--show-kernel-names', action='store_true', 
                                help='在输出结果时是否展示 kernel 名称信息 (默认: False)')
    analysis_parser.add_argument('--show-kernel-duration', action='store_true', 
                                help='在输出结果时是否展示 kernel 持续时间信息 (默认: False)')
    analysis_parser.add_argument('--show-timestamp', action='store_true', 
                                help='在输出结果时是否展示 CPU 操作启动时间戳 (默认: False)')
    analysis_parser.add_argument('--print-markdown', action='store_true', 
                                help='是否在stdout中以markdown格式打印表格 (默认: False)')
    analysis_parser.add_argument('--output-format', default='json,xlsx', 
                                choices=['json', 'xlsx', 'json,xlsx'],
                                help='输出格式 (默认: json,xlsx)')
    analysis_parser.add_argument('--output-dir', default='.', help='输出目录 (默认: 当前目录)')
    
    # all2all 命令 - 分析All-to-All通信性能
    all2all_parser = subparsers.add_parser('all2all', help='分析All-to-All通信性能')
    all2all_parser.add_argument('pod_dir', help='Pod文件夹路径，包含executor_trainer-runner_*_*_*格式的文件夹')
    all2all_parser.add_argument('--step', type=int, help='指定要分析的step，如果不指定则分析所有step')
    all2all_parser.add_argument('--all2all-idx', type=int, help='指定要分析的all2all索引，如果不指定则分析所有all2all操作')
    all2all_parser.add_argument('--output-dir', default='.', help='输出目录 (默认: 当前目录)')
    
    # compare 命令 - 分析多个文件并对比
    compare_parser = subparsers.add_parser('compare', help='分析多个 JSON 文件并对比')
    compare_parser.add_argument('files', nargs='+', 
                               help='文件列表，支持多种格式:\n'
                                    '  单文件: file.json:label\n'
                                    '  多文件: "file1.json,file2.json,file3.json":label\n'
                                    '  目录: dir/:label (自动查找所有*.json文件)\n'
                                    '  通配符: "dir/*.json":label')
    compare_parser.add_argument('--aggregation', choices=['on_op_name', 'on_op_shape', 'on_call_stack', 'on_op_timestamp'], 
                               default='on_op_name',
                               help='聚合方法: on_op_name (基于操作名) 或 on_op_shape (基于操作形状) 或 on_call_stack (基于调用栈) 或 on_op_timestamp (按CPU操作启动时间排序) (默认: on_op_name)')
    compare_parser.add_argument('--show-dtype', action='store_true', 
                               help='在输出结果时是否展示 dtype 信息 (默认: False)')
    compare_parser.add_argument('--show-shape', action='store_true', 
                               help='在输出结果时是否展示 shape 和 strides 信息 (默认: False)')
    compare_parser.add_argument('--show-kernel-names', action='store_true', 
                               help='在输出结果时是否展示 kernel 名称信息 (默认: False)')
    compare_parser.add_argument('--show-kernel-duration', action='store_true', 
                               help='在输出结果时是否展示 kernel 持续时间信息 (默认: False)')
    compare_parser.add_argument('--show-timestamp', action='store_true', 
                               help='在输出结果时是否展示 CPU 操作启动时间戳 (默认: False)')
    compare_parser.add_argument('--print-markdown', action='store_true', 
                               help='是否在stdout中以markdown格式打印表格 (默认: False)')
    compare_parser.add_argument('--special-matmul', action='store_true',
                               help='是否进行特殊的 matmul 分析 (默认: False)')
    compare_parser.add_argument('--compare-dtype', action='store_true',
                               help='是否添加 dtype 比较列 (默认: False)')
    compare_parser.add_argument('--compare-shape', action='store_true',
                               help='是否添加 shape 比较列 (默认: False)')
    compare_parser.add_argument('--max-files-per-label', type=int, default=None,
                               help='每个标签最多使用的文件数量，用于随机采样确保比较公平性 (默认: 不限制)')
    compare_parser.add_argument('--random-seed', type=int, default=42,
                               help='随机采样的种子，确保结果可重现 (默认: 42)')
    compare_parser.add_argument('--output-format', default='json,xlsx', 
                               choices=['json', 'xlsx', 'json,xlsx'],
                               help='输出格式 (默认: json,xlsx)')
    compare_parser.add_argument('--output-dir', default='.', help='输出目录 (默认: 当前目录)')
    compare_parser.add_argument('--max-workers', type=int, default=None,
                               help='并行处理的最大工作进程数，默认为CPU核心数')
    
    return parser.parse_args()


def validate_file(file_path: str) -> bool:
    """验证文件是否存在且为 JSON 格式"""
    path = Path(file_path)
    if not path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return False
    
    if not path.suffix.lower() == '.json':
        print(f"警告: 文件可能不是 JSON 格式: {file_path}")
    
    return True


def parse_file_label(file_label: str) -> Tuple[List[str], str]:
    """解析 file:label 格式的字符串，支持混合模式
    
    支持的模式：
    1. 单文件: file.json:label
    2. 多文件: file1.json,file2.json,file3.json:label  
    3. 目录: dir/:label (自动查找所有*.json文件)
    4. 通配符: "dir/*.json":label
    
    Returns:
        Tuple[List[str], str]: (文件路径列表, 标签)
    """
    if ':' in file_label:
        file_part, label = file_label.rsplit(':', 1)
        file_part = file_part.strip()
        label = label.strip()
        
        # 检查是否是目录
        if os.path.isdir(file_part):
            # 目录模式：查找所有json文件
            json_files = glob.glob(os.path.join(file_part, "*.json"))
            if not json_files:
                raise ValueError(f"目录 {file_part} 中没有找到任何 .json 文件")
            return sorted(json_files), label
        elif ',' in file_part:
            # 多文件模式：逗号分隔
            files = [f.strip() for f in file_part.split(',')]
            return files, label
        elif '*' in file_part or '?' in file_part:
            # 通配符模式
            matched_files = glob.glob(file_part)
            if not matched_files:
                raise ValueError(f"通配符 {file_part} 没有匹配到任何文件")
            return sorted(matched_files), label
        else:
            # 单文件模式
            return [file_part], label
    else:
        # 没有标签，使用文件名作为标签
        return [file_label], Path(file_label).stem


def run_analysis(args):
    """运行单个文件分析"""
    print(f"=== 单个文件分析 ===")
    print(f"文件: {args.file}")
    print(f"标签: {args.label}")
    print(f"聚合方法: {args.aggregation}")
    print(f"展示dtype: {args.show_dtype}")
    print(f"展示shape: {args.show_shape}")
    print(f"展示kernel名称: {args.show_kernel_names}")
    print(f"展示kernel持续时间: {args.show_kernel_duration}")
    print(f"展示时间戳: {args.show_timestamp}")
    print(f"打印markdown表格: {args.print_markdown}")
    print(f"输出格式: {args.output_format}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    if not validate_file(args.file):
        return 1
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建分析器
    analyzer = Analyzer()
    
    try:
        start_time = time.time()
        
        # 使用新的分析流程
        generated_files = analyzer.analyze_single_file(
            file_path=args.file,
            aggregation_type=args.aggregation,
            show_dtype=args.show_dtype,
            show_shape=args.show_shape,
            show_kernel_names=args.show_kernel_names,
            show_kernel_duration=args.show_kernel_duration,
            show_timestamp=args.show_timestamp,
            output_dir=str(output_dir),
            label=args.label,
            print_markdown=args.print_markdown
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


def run_all2all_analysis(args):
    """运行All-to-All通信性能分析"""
    print(f"=== All-to-All通信性能分析 ===")
    print(f"Pod目录: {args.pod_dir}")
    print(f"Step: {args.step if args.step is not None else '所有step'}")
    print(f"All2All索引: {args.all2all_idx if args.all2all_idx is not None else '所有all2all操作'}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 验证pod目录
    pod_path = Path(args.pod_dir)
    if not pod_path.exists():
        print(f"错误: Pod目录不存在: {args.pod_dir}")
        return 1
    
    if not pod_path.is_dir():
        print(f"错误: 路径不是目录: {args.pod_dir}")
        return 1
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建分析器
    analyzer = Analyzer()
    
    try:
        start_time = time.time()
        
        # 运行all2all分析
        generated_files = analyzer.analyze_all2all_performance(
            pod_dir=str(pod_path),
            step=args.step,
            all2all_idx=args.all2all_idx,
            output_dir=str(output_dir)
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


def run_compare_analysis(args):
    """运行多文件对比分析"""
    print(f"=== 多文件对比分析 ===")
    print(f"聚合方法: {args.aggregation}")
    print(f"展示dtype: {args.show_dtype}")
    print(f"展示shape: {args.show_shape}")
    print(f"展示kernel名称: {args.show_kernel_names}")
    print(f"展示kernel持续时间: {args.show_kernel_duration}")
    print(f"打印markdown表格: {args.print_markdown}")
    print(f"特殊matmul: {args.special_matmul}")
    print(f"每个标签最大文件数: {args.max_files_per_label if args.max_files_per_label else '不限制'}")
    print(f"随机种子: {args.random_seed}")
    print(f"输出格式: {args.output_format}")
    print(f"输出目录: {args.output_dir}")
    print()
    
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
                import random
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
    
    # 创建分析器
    analyzer = Analyzer()
    
    try:
        start_time = time.time()
        
        # 使用新的分析流程
        generated_files = analyzer.analyze_multiple_files(
            file_labels=file_labels,
            aggregation_type=args.aggregation,
            show_dtype=args.show_dtype,
            show_shape=args.show_shape,
            show_kernel_names=args.show_kernel_names,
            show_kernel_duration=args.show_kernel_duration,
            show_timestamp=args.show_timestamp,
            special_matmul=args.special_matmul,
            output_dir=str(output_dir),
            compare_dtype=args.compare_dtype,
            compare_shape=args.compare_shape,
            print_markdown=args.print_markdown,
            max_workers=args.max_workers
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


def main():
    """主函数"""
    args = parse_arguments()
    
    if not args.command:
        print("错误: 请指定命令 (analysis, compare, all2all)")
        print("使用 --help 查看帮助信息")
        return 1
    
    if args.command == 'analysis':
        return run_analysis(args)
    elif args.command == 'compare':
        return run_compare_analysis(args)
    elif args.command == 'all2all':
        return run_all2all_analysis(args)
    else:
        print(f"错误: 未知命令: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

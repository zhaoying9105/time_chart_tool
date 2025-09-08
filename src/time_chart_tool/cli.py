#!/usr/bin/env python3
"""
Time Chart Tool 命令行工具 - 重构版本

支持分析多个 timechart JSON 文件，生成单个文件分析和对比分析结果。
支持 JSON 和 XLSX 输出格式。
"""

import argparse
import sys
import time
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
  
  # 分析单个文件 (基于调用栈，输出包含shape和strides信息)
  time-chart-tool analysis file.json --label "baseline" --aggregation on_call_stack --show-shape --output-format json,xlsx
  
  # 分析单个文件 (基于调用栈，输出包含所有信息)
  time-chart-tool analysis file.json --label "baseline" --aggregation on_call_stack --show-dtype --show-shape --show-kernel-names --show-kernel-duration --output-format json,xlsx
  
  # 基于CPU操作对比多个文件 (默认方法，输出不包含kernel信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation on_op_name --output-format json,xlsx
  
  # 基于CPU操作对比多个文件 (输出包含kernel信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation on_op_name --show-kernel-names --show-kernel-duration --output-format json,xlsx
  
  # 基于调用栈对比多个文件 (输出包含shape和strides信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation on_call_stack --show-shape --output-format json,xlsx
  
  # 基于调用栈对比多个文件 (输出包含所有信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation on_call_stack --show-dtype --show-shape --show-kernel-names --show-kernel-duration --output-format json,xlsx
  
  # 对比多个文件并包含特殊的matmul分析
  time-chart-tool compare file1.json:fp32 file2.json:bf16 --aggregation on_op_name --special-matmul --output-format json,xlsx
  
  # 只输出 JSON 格式
  time-chart-tool analysis file.json --aggregation on_call_stack --output-format json
  time-chart-tool compare file1.json:fp32 file2.json:tf32 --aggregation on_op_name --output-format json
  
  # 只输出 XLSX 格式
  time-chart-tool analysis file.json --aggregation on_op_name --output-format xlsx
  time-chart-tool compare file1.json:baseline file2.json:optimized --aggregation on_call_stack --output-format xlsx
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # analysis 命令 - 分析单个文件 (原来的 single 命令)
    analysis_parser = subparsers.add_parser('analysis', help='分析单个 JSON 文件')
    analysis_parser.add_argument('file', help='要分析的 JSON 文件路径')
    analysis_parser.add_argument('--label', default='single_file', help='文件标签 (默认: single_file)')
    analysis_parser.add_argument('--aggregation', choices=['on_op_name', 'on_op_shape', 'on_call_stack'], 
                                default='on_op_name',
                                help='聚合方法: on_op_name (基于操作名) 或 on_op_shape (基于操作形状) 或 on_call_stack (基于调用栈) (默认: on_op_name)')
    analysis_parser.add_argument('--show-dtype', action='store_true', 
                                help='在输出结果时是否展示 dtype 信息 (默认: False)')
    analysis_parser.add_argument('--show-shape', action='store_true', 
                                help='在输出结果时是否展示 shape 和 strides 信息 (默认: False)')
    analysis_parser.add_argument('--show-kernel-names', action='store_true', 
                                help='在输出结果时是否展示 kernel 名称信息 (默认: False)')
    analysis_parser.add_argument('--show-kernel-duration', action='store_true', 
                                help='在输出结果时是否展示 kernel 持续时间信息 (默认: False)')
    analysis_parser.add_argument('--output-format', default='json,xlsx', 
                                choices=['json', 'xlsx', 'json,xlsx'],
                                help='输出格式 (默认: json,xlsx)')
    analysis_parser.add_argument('--output-dir', default='.', help='输出目录 (默认: 当前目录)')
    
    # compare 命令 - 分析多个文件并对比
    compare_parser = subparsers.add_parser('compare', help='分析多个 JSON 文件并对比')
    compare_parser.add_argument('files', nargs='+', 
                               help='文件列表，格式: file.json:label')
    compare_parser.add_argument('--aggregation', choices=['on_op_name', 'on_op_shape', 'on_call_stack'], 
                               default='on_op_name',
                               help='聚合方法: on_op_name (基于操作名) 或 on_op_shape (基于操作形状) 或 on_call_stack (基于调用栈) (默认: on_op_name)')
    compare_parser.add_argument('--show-dtype', action='store_true', 
                               help='在输出结果时是否展示 dtype 信息 (默认: False)')
    compare_parser.add_argument('--show-shape', action='store_true', 
                               help='在输出结果时是否展示 shape 和 strides 信息 (默认: False)')
    compare_parser.add_argument('--show-kernel-names', action='store_true', 
                               help='在输出结果时是否展示 kernel 名称信息 (默认: False)')
    compare_parser.add_argument('--show-kernel-duration', action='store_true', 
                               help='在输出结果时是否展示 kernel 持续时间信息 (默认: False)')
    compare_parser.add_argument('--special-matmul', action='store_true',
                               help='是否进行特殊的 matmul 分析 (默认: False)')
    compare_parser.add_argument('--compare-dtype', action='store_true',
                               help='是否添加 dtype 比较列 (默认: False)')
    compare_parser.add_argument('--compare-shape', action='store_true',
                               help='是否添加 shape 比较列 (默认: False)')
    compare_parser.add_argument('--output-format', default='json,xlsx',
                               choices=['json', 'xlsx', 'json,xlsx'],
                               help='输出格式 (默认: json,xlsx)')
    compare_parser.add_argument('--output-dir', default='.', help='输出目录 (默认: 当前目录)')
    
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


def parse_file_label(file_label: str) -> Tuple[str, str]:
    """解析 file:label 格式的字符串"""
    if ':' in file_label:
        file_path, label = file_label.rsplit(':', 1)
        return file_path.strip(), label.strip()
    else:
        return file_label, Path(file_label).stem


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
            output_dir=str(output_dir),
            label=args.label
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
    print(f"特殊matmul: {args.special_matmul}")
    print(f"输出格式: {args.output_format}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    # 解析文件列表
    file_labels = []
    for file_label in args.files:
        file_path, label = parse_file_label(file_label)
        if not validate_file(file_path):
            return 1
        file_labels.append((file_path, label))
        print(f"  {label}: {file_path}")
    
    if len(file_labels) < 2:
        print("错误: 对比分析需要至少2个文件")
        return 1
    
    print(f"\n将分析 {len(file_labels)} 个文件")
    
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
            special_matmul=args.special_matmul,
            output_dir=str(output_dir),
            compare_dtype=args.compare_dtype,
            compare_shape=args.compare_shape
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
        print("错误: 请指定命令 (analysis, compare)")
        print("使用 --help 查看帮助信息")
        return 1
    
    if args.command == 'analysis':
        return run_analysis(args)
    elif args.command == 'compare':
        return run_compare_analysis(args)
    else:
        print(f"错误: 未知命令: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

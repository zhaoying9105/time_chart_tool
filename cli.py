#!/usr/bin/env python3
"""
PyTorch Profiler Parser Tool 命令行工具

支持分析多个 timechart JSON 文件，生成单个文件分析和对比分析结果。
支持 JSON 和 XLSX 输出格式。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
import json

from time_chart_tool.analyzer import Analyzer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PyTorch Profiler Parser Tool - 分析多个 timechart JSON 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析单个文件
  torch-profiler-parser single file.json --label "baseline" --output-format json,xlsx
  
  # 分析多个文件并对比
  torch-profiler-parser compare file1.json:label1 file2.json:label2 --output-format json,xlsx
  
  # 只输出 JSON 格式
  torch-profiler-parser compare file1.json:fp32 file2.json:tf32 --output-format json
  
  # 只输出 XLSX 格式
  torch-profiler-parser compare file1.json:baseline file2.json:optimized --output-format xlsx
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # single 命令 - 分析单个文件
    single_parser = subparsers.add_parser('single', help='分析单个 JSON 文件')
    single_parser.add_argument('file', help='要分析的 JSON 文件路径')
    single_parser.add_argument('--label', default='single_file', help='文件标签 (默认: single_file)')
    single_parser.add_argument('--output-format', default='json,xlsx', 
                              choices=['json', 'xlsx', 'json,xlsx'],
                              help='输出格式 (默认: json,xlsx)')
    single_parser.add_argument('--output-dir', default='.', help='输出目录 (默认: 当前目录)')
    
    # compare 命令 - 分析多个文件并对比
    compare_parser = subparsers.add_parser('compare', help='分析多个 JSON 文件并对比')
    compare_parser.add_argument('files', nargs='+', 
                               help='文件列表，格式: file.json:label')
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


def run_single_analysis(args):
    """运行单个文件分析"""
    print(f"=== 单个文件分析 ===")
    print(f"文件: {args.file}")
    print(f"标签: {args.label}")
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
        
        # 加载文件
        print(f"正在加载文件: {args.file}")
        data = analyzer.parser.load_json_file(args.file)
        load_time = time.time() - start_time
        print(f"文件加载完成，耗时: {load_time:.2f} 秒")
        
        # 分析 cpu_op 和 kernel 的映射关系
        print("正在分析 cpu_op 和 kernel 的映射关系...")
        mapping = analyzer.analyze_cpu_op_kernel_mapping(data)
        print(f"找到 {len(mapping)} 个 cpu_op 的映射关系")
        
        # 生成输出文件
        base_name = f"{args.label}_single_file_analysis"
        
        if 'json' in args.output_format:
            json_file = output_dir / f"{base_name}.json"
            print(f"正在生成 JSON 文件: {json_file}")
            analyzer.save_mapping_to_json(mapping, str(json_file))
        
        if 'xlsx' in args.output_format:
            xlsx_file = output_dir / f"{base_name}.xlsx"
            print(f"正在生成 XLSX 文件: {xlsx_file}")
            analyzer.generate_excel_from_mapping(mapping, str(xlsx_file))
        
        total_time = time.time() - start_time
        print(f"\n分析完成，总耗时: {total_time:.2f} 秒")
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_compare_analysis(args):
    """运行多文件对比分析"""
    print(f"=== 多文件对比分析 ===")
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
        
        # 运行完整的多文件分析
        print("\n开始分析...")
        analyzer.run_complete_analysis(file_labels, output_dir=str(output_dir), 
                                     output_formats=args.output_format.split(','))
        
        total_time = time.time() - start_time
        print(f"\n分析完成，总耗时: {total_time:.2f} 秒")
        
        # 显示生成的文件
        print("\n生成的文件:")
        for file_path in output_dir.glob("*analysis*"):
            if file_path.suffix in ['.json', '.xlsx']:
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
        print("错误: 请指定命令 (single 或 compare)")
        print("使用 --help 查看帮助信息")
        return 1
    
    if args.command == 'single':
        return run_single_analysis(args)
    elif args.command == 'compare':
        return run_compare_analysis(args)
    else:
        print(f"错误: 未知命令: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

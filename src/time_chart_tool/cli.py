#!/usr/bin/env python3
"""
Time Chart Tool 命令行工具

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
  time-chart-tool single file.json --label "baseline" --method on_cpu_op --output-format json,xlsx
  
  # 分析单个文件 (基于CPU操作，输出包含kernel信息)
  time-chart-tool single file.json --label "baseline" --method on_cpu_op --include-kernel --output-format json,xlsx
  
  # 分析单个文件 (基于调用栈，输出不包含kernel信息)
  time-chart-tool single file.json --label "baseline" --method on_call_stack --output-format json,xlsx
  
  # 分析单个文件 (基于调用栈，输出包含kernel信息)
  time-chart-tool single file.json --label "baseline" --method on_call_stack --include-kernel --output-format json,xlsx
  
  # 基于CPU操作对比多个文件 (默认方法，输出不包含kernel信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --method on_cpu_op --output-format json,xlsx
  
  # 基于CPU操作对比多个文件 (输出包含kernel信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --method on_cpu_op --include-kernel --output-format json,xlsx
  
  # 基于调用栈对比多个文件 (输出不包含kernel信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --method on_call_stack --output-format json,xlsx
  
  # 基于调用栈对比多个文件 (输出包含kernel信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --method on_call_stack --include-kernel --output-format json,xlsx
  
  # 专门分析matmul算子
  time-chart-tool matmul file1.json:fp32 file2.json:bf16 --output-format json,xlsx
  
  # 只输出 JSON 格式
  time-chart-tool single file.json --method on_call_stack --output-format json
  time-chart-tool compare file1.json:fp32 file2.json:tf32 --method on_cpu_op --output-format json
  
  # 只输出 XLSX 格式
  time-chart-tool single file.json --method on_cpu_op --output-format xlsx
  time-chart-tool compare file1.json:baseline file2.json:optimized --method on_call_stack --output-format xlsx
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # single 命令 - 分析单个文件
    single_parser = subparsers.add_parser('single', help='分析单个 JSON 文件')
    single_parser.add_argument('file', help='要分析的 JSON 文件路径')
    single_parser.add_argument('--label', default='single_file', help='文件标签 (默认: single_file)')
    single_parser.add_argument('--method', choices=['on_cpu_op', 'on_call_stack'], 
                              default='on_cpu_op',
                              help='分析方法: on_cpu_op (基于CPU操作) 或 on_call_stack (基于调用栈) (默认: on_cpu_op)')
    single_parser.add_argument('--include-kernel', action='store_true', 
                              help='在输出结果时是否保留 kernel 相关信息 (默认: False)')
    single_parser.add_argument('--output-format', default='json,xlsx', 
                              choices=['json', 'xlsx', 'json,xlsx'],
                              help='输出格式 (默认: json,xlsx)')
    single_parser.add_argument('--output-dir', default='.', help='输出目录 (默认: 当前目录)')
    
    # compare 命令 - 分析多个文件并对比
    compare_parser = subparsers.add_parser('compare', help='分析多个 JSON 文件并对比')
    compare_parser.add_argument('files', nargs='+', 
                               help='文件列表，格式: file.json:label')
    compare_parser.add_argument('--method', choices=['on_cpu_op', 'on_call_stack'], 
                               default='on_cpu_op',
                               help='对比方法: on_cpu_op (基于CPU操作) 或 on_call_stack (基于调用栈) (默认: on_cpu_op)')
    compare_parser.add_argument('--include-kernel', action='store_true', 
                               help='在输出结果时是否保留 kernel 相关信息 (默认: False)')
    compare_parser.add_argument('--output-format', default='json,xlsx',
                               choices=['json', 'xlsx', 'json,xlsx'],
                               help='输出格式 (默认: json,xlsx)')
    compare_parser.add_argument('--output-dir', default='.', help='输出目录 (默认: 当前目录)')
    
    # matmul 命令 - 专门分析matmul算子
    matmul_parser = subparsers.add_parser('matmul', help='专门分析matmul算子 (aten::mm)')
    matmul_parser.add_argument('files', nargs='+', 
                              help='文件列表，格式: file.json:label')
    matmul_parser.add_argument('--output-format', default='json,xlsx',
                              choices=['json', 'xlsx', 'json,xlsx'],
                              help='输出格式 (默认: json,xlsx)')
    matmul_parser.add_argument('--output-dir', default='.', help='输出目录 (默认: 当前目录)')
    
    
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
    print(f"分析方法: {args.method}")
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
        
        # 根据方法选择不同的分析流程
        if args.method == 'on_cpu_op':
            print("\n使用 on_cpu_op 方法进行分析...")
            
            # 根据 --include-kernel 选项决定输出方式
            if args.include_kernel:
                print("输出包含 kernel 信息...")
                # 分析 cpu_op 和 kernel 的映射关系
                print("正在分析 cpu_op 和 kernel 的映射关系...")
                mapping = analyzer.analyze_cpu_op_kernel_mapping(data)
                print(f"找到 {len(mapping)} 个有对应 kernel 的 cpu_op")
                
                # 生成cpu_op性能统计摘要
                print("正在生成cpu_op性能统计摘要...")
                analyzer.generate_cpu_op_performance_summary(data, str(output_dir), args.label)
                
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
            else:
                print("输出不包含 kernel 信息...")
                # 只分析 cpu_op 信息
                print("正在分析 cpu_op 信息...")
                mapping = analyzer.analyze_cpu_op_only(data)
                print(f"找到 {len(mapping)} 个有对应 kernel 的 cpu_op")
                
                # 生成输出文件
                base_name = f"{args.label}_single_file_analysis"
                
                if 'json' in args.output_format:
                    json_file = output_dir / f"{base_name}.json"
                    print(f"正在生成 JSON 文件: {json_file}")
                    analyzer.save_cpu_op_mapping_to_json(mapping, str(json_file))
                
                if 'xlsx' in args.output_format:
                    xlsx_file = output_dir / f"{base_name}.xlsx"
                    print(f"正在生成 XLSX 文件: {xlsx_file}")
                    analyzer.generate_excel_from_cpu_op_mapping(mapping, str(xlsx_file))
                
        elif args.method == 'on_call_stack':
            print("\n使用 on_call_stack 方法进行分析...")
            
            # 分析 call stack
            analyzer.analyze_single_file_by_call_stack(data, args.label, str(output_dir), args.output_format.split(','), args.include_kernel)
            
        else:
            print(f"错误: 未知的分析方法: {args.method}")
            return 1
        
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
    print(f"对比方法: {args.method}")
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
        
        # 根据方法选择不同的分析流程
        if args.method == 'on_cpu_op':
            print("\n使用 on_cpu_op 方法进行分析...")
            analyzer.run_complete_analysis(file_labels, output_dir=str(output_dir), 
                                         output_formats=args.output_format.split(','))
        elif args.method == 'on_call_stack':
            print("\n使用 on_call_stack 方法进行分析...")
            analyzer.compare_by_call_stack(file_labels, str(output_dir), args.include_kernel)
        else:
            print(f"错误: 未知的对比方法: {args.method}")
            return 1
        
        total_time = time.time() - start_time
        print(f"\n分析完成，总耗时: {total_time:.2f} 秒")
        
        # 显示生成的文件
        print("\n生成的文件:")
        for file_path in output_dir.glob("*"):
            if file_path.suffix in ['.json', '.xlsx']:
                print(f"  {file_path}")
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_matmul_analysis(args):
    """运行matmul算子专门分析"""
    print(f"=== Matmul算子专门分析 ===")
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
        print("错误: matmul分析需要至少2个文件进行对比")
        return 1
    
    print(f"\n将分析 {len(file_labels)} 个文件中的matmul算子")
    print("分析内容:")
    print("1. 提取所有 'aten::mm' 算子")
    print("2. 解析输入维度 (m, k, n)")
    print("3. 计算最小维度 min_dim = min(m, k, n)")
    print("4. 按 min_dim 分组统计性能数据")
    print("5. 生成性能对比图表")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建分析器
    analyzer = Analyzer()
    
    try:
        start_time = time.time()
        
        # 运行包含matmul专门分析的完整分析流程
        print("\n开始分析...")
        analyzer.run_complete_analysis_with_matmul(
            file_labels, 
            output_dir=str(output_dir), 
            output_formats=args.output_format.split(',')
        )
        
        total_time = time.time() - start_time
        print(f"\n分析完成，总耗时: {total_time:.2f} 秒")
        
        # 显示生成的文件
        print("\n生成的文件:")
        for file_path in output_dir.glob("*"):
            if file_path.suffix in ['.json', '.xlsx', '.jpg']:
                if 'matmul' in file_path.name or 'comparison' in file_path.name:
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
        print("错误: 请指定命令 (single, compare, matmul)")
        print("使用 --help 查看帮助信息")
        return 1
    
    if args.command == 'single':
        return run_single_analysis(args)
    elif args.command == 'compare':
        return run_compare_analysis(args)
    elif args.command == 'matmul':
        return run_matmul_analysis(args)
    else:
        print(f"错误: 未知命令: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

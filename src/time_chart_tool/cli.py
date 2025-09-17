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
from typing import List, Tuple, Optional, Dict
import json

from .analyzer import Analyzer


def validate_aggregation_fields(aggregation_spec: str) -> List[str]:
    """
    验证聚合字段组合是否合规
    
    Args:
        aggregation_spec: 聚合字段组合字符串
        
    Returns:
        List[str]: 验证后的字段列表
        
    Raises:
        ValueError: 如果字段组合不合法
    """
    if not aggregation_spec or not aggregation_spec.strip():
        raise ValueError("聚合字段不能为空")
    
    # 解析字段组合：逗号分隔的字段
    if ',' in aggregation_spec:
        fields = [field.strip() for field in aggregation_spec.split(',')]
    else:
        fields = [aggregation_spec.strip()]
    
    # 验证字段
    valid_fields = {'call_stack', 'name', 'shape', 'dtype'}
    for field in fields:
        if not field:
            raise ValueError("聚合字段不能为空字符串")
        if field not in valid_fields:
            raise ValueError(f"不支持的聚合字段: {field}。支持的字段: {', '.join(sorted(valid_fields))}")
    
    # 检查字段重复
    if len(fields) != len(set(fields)):
        raise ValueError("聚合字段不能重复")
    
    return fields


def parse_show_options(show_spec: str) -> Dict[str, bool]:
    """
    解析show选项
    
    Args:
        show_spec: show参数字符串，逗号分隔
        
    Returns:
        Dict[str, bool]: 各show选项的开关状态
    """
    valid_show_options = {
        'dtype', 'shape', 'kernel-names', 'kernel-duration', 
        'timestamp', 'readable-timestamp', 'kernel-timestamp'
    }
    
    show_options = {
        'dtype': False,
        'shape': False, 
        'kernel_names': False,
        'kernel_duration': False,
        'timestamp': False,
        'readable_timestamp': False,
        'kernel_timestamp': False
    }
    
    if not show_spec or not show_spec.strip():
        return show_options
    
    # 解析逗号分隔的选项
    show_args = [arg.strip() for arg in show_spec.split(',')]
    
    for arg in show_args:
        if not arg:
            continue
        if arg not in valid_show_options:
            raise ValueError(f"不支持的show选项: {arg}。支持的选项: {', '.join(sorted(valid_show_options))}")
        
        if arg == 'kernel-names':
            show_options['kernel_names'] = True
        elif arg == 'kernel-duration':
            show_options['kernel_duration'] = True
        elif arg == 'readable-timestamp':
            show_options['readable_timestamp'] = True
        elif arg == 'kernel-timestamp':
            show_options['kernel_timestamp'] = True
        else:
            show_options[arg] = True
    
    return show_options


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Time Chart Tool - 分析多个 timechart JSON 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析单个文件 (按操作名聚合，默认方法)
  time-chart-tool analysis file.json --label "baseline" --aggregation name --output-format json,xlsx
  
  # 分析单个文件 (按操作名聚合，显示kernel信息)
  time-chart-tool analysis file.json --label "baseline" --aggregation name --show "kernel-names,kernel-duration" --output-format json,xlsx
  
  # 分析单个文件并在stdout中打印markdown表格
  time-chart-tool analysis file.json --label "baseline" --aggregation name --show "kernel-duration" --print-markdown
  
  # 分析单个文件 (按调用栈和操作名聚合，显示shape信息)
  time-chart-tool analysis file.json --label "baseline" --aggregation "call_stack,name" --show "shape" --output-format json,xlsx
  
  # 分析单个文件 (按调用栈和操作名聚合，显示所有信息)
  time-chart-tool analysis file.json --label "baseline" --aggregation "call_stack,name" --show "dtype,shape,kernel-names,kernel-duration" --output-format json,xlsx
  
  # 分析单个文件 (按操作名和形状聚合)
  time-chart-tool analysis file.json --label "baseline" --aggregation "name,shape" --show "dtype" --output-format json,xlsx
  
  # 分析单个文件 (按调用栈和操作名聚合)
  time-chart-tool analysis file.json --label "baseline" --aggregation "call_stack,name" --show "shape" --output-format json,xlsx
  
  # 分析单个文件 (按操作名、形状和数据类型聚合)
  time-chart-tool analysis file.json --label "baseline" --aggregation "name,shape,dtype" --output-format json,xlsx
  
  # 分析单个文件 (显示CPU操作启动时间戳)
  time-chart-tool analysis file.json --label "baseline" --aggregation name --show "timestamp" --output-format json,xlsx
  
  # 基于操作名对比多个文件 (默认方法)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation name --output-format json,xlsx
  
  # 基于操作名对比多个文件 (显示kernel信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation name --show "kernel-names,kernel-duration" --output-format json,xlsx
  
  # 基于调用栈和操作名对比多个文件 (显示shape信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation "call_stack,name" --show "shape" --output-format json,xlsx
  
  # 基于调用栈和操作名对比多个文件 (显示所有信息)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation "call_stack,name" --show "dtype,shape,kernel-names,kernel-duration" --output-format json,xlsx
  
  # 对比多个文件 (按操作名和形状聚合)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation "name,shape" --show "dtype" --output-format json,xlsx
  
  # 对比多个文件 (按调用栈和操作名聚合)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation "call_stack,name" --show "shape" --output-format json,xlsx
  
  # 对比多个文件 (按操作名、形状和数据类型聚合)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation "name,shape,dtype" --output-format json,xlsx
  
  # 对比多个文件 (显示CPU操作启动时间戳)
  time-chart-tool compare file1.json:label1 file2.json:label2 --aggregation name --show "timestamp" --output-format json,xlsx
  
  # 对比多个文件并包含特殊的matmul分析
  time-chart-tool compare file1.json:fp32 file2.json:bf16 --aggregation name --special-matmul --output-format json,xlsx
  
  # 混合模式：支持单文件、多文件、目录混合使用
  time-chart-tool compare single_file.json:baseline "dir/*.json":test --aggregation name --output-format json,xlsx
  
  # 多文件模式：同一标签下多个文件自动聚合
  time-chart-tool compare "file1.json,file2.json,file3.json":baseline "file4.json,file5.json":optimized --aggregation name --output-format json,xlsx
  
  # 目录模式：自动查找目录下所有json文件
  time-chart-tool compare step1_results/:baseline step2_results/:optimized --aggregation name --output-format json,xlsx
  
  # 控制每个标签的文件数量，确保比较公平性
  time-chart-tool compare "dir1/*.json":baseline "dir2/*.json":optimized --max-files-per-label 10 --random-seed 42 --aggregation name
  
  # 混合模式：单文件 vs 多文件（限制数量）
  time-chart-tool compare single_file.json:reference "multi_files/*.json":test --max-files-per-label 5 --aggregation name
  
  # 高级聚合示例：按操作名和数据类型对比
  time-chart-tool compare file1.json:fp32 file2.json:bf16 --aggregation "name,dtype" --output-format json,xlsx
  
  # 高级聚合示例：按调用栈和形状对比
  time-chart-tool compare file1.json:baseline file2.json:optimized --aggregation "call_stack,shape" --show-dtype --output-format json,xlsx
  
  # 高级聚合示例：四字段组合聚合
  time-chart-tool compare file1.json:baseline file2.json:optimized --aggregation "call_stack,name,shape,dtype" --output-format json,xlsx
  
  # 只输出 JSON 格式
  time-chart-tool analysis file.json --aggregation "call_stack,name" --output-format json
  time-chart-tool compare file1.json:fp32 file2.json:tf32 --aggregation name --output-format json
  
  # 只输出 XLSX 格式
  time-chart-tool analysis file.json --aggregation name --output-format xlsx
  time-chart-tool compare file1.json:baseline file2.json:optimized --aggregation "call_stack,name" --output-format xlsx
  
  # 按时间排序分析
  time-chart-tool analysis file.json --aggregation on_op_timestamp --output-format json,xlsx
  time-chart-tool compare file1.json:baseline file2.json:optimized --aggregation on_op_timestamp --output-format xlsx
  
  # 显示时间戳分析
  time-chart-tool analysis file.json --aggregation on_op_name --show-timestamp --output-format json,xlsx
  time-chart-tool compare file1.json:baseline file2.json:optimized --aggregation on_op_name --show-timestamp --output-format xlsx
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # analysis 命令 - 分析单个或多个 JSON 文件
    analysis_parser = subparsers.add_parser('analysis', help='分析单个或多个 JSON 文件')
    analysis_parser.add_argument('file', help='要分析的 JSON 文件路径，支持 glob 模式 (如: "*.json" 或 "dir/*.json")')
    analysis_parser.add_argument('--label', default='single_file', help='文件标签 (默认: single_file)')
    analysis_parser.add_argument('--aggregation', default='name',
                                help='聚合字段组合，使用逗号分隔的字段组合\n'
                                     '支持的字段: call_stack, name, shape, dtype\n'
                                     '示例: "name" 或 "name,shape" 或 "call_stack,name" 或 "name,shape,dtype"\n'
                                     '(默认: name)')
    analysis_parser.add_argument('--show', type=str, default='',
                                help='显示额外信息，使用逗号分隔的选项:\n'
                                     '  dtype: 显示数据类型信息\n'
                                     '  shape: 显示形状和步长信息\n'
                                     '  kernel-names: 显示kernel名称\n'
                                     '  kernel-duration: 显示kernel持续时间\n'
                                     '  timestamp: 显示时间戳\n'
                                     '  readable-timestamp: 显示可读时间戳\n'
                                     '  kernel-timestamp: 显示kernel时间戳\n'
                                     '示例: --show "dtype,shape,kernel-duration"')
    analysis_parser.add_argument('--print-markdown', action='store_true', 
                                help='是否在stdout中以markdown格式打印表格 (默认: False)')
    analysis_parser.add_argument('--output-format', default='json,xlsx', 
                                choices=['json', 'xlsx', 'json,xlsx'],
                                help='输出格式 (默认: json,xlsx)')
    analysis_parser.add_argument('--output-dir', default='.', help='输出目录 (默认: 当前目录)')
    
    # comm 命令 - 分析通信性能
    comm_parser = subparsers.add_parser('comm', help='分析分布式训练中的通信性能')
    comm_parser.add_argument('pod_dir', help='Pod文件夹路径，包含executor_trainer-runner_*_*_*格式的文件夹')
    comm_parser.add_argument('--step', type=int, help='指定要分析的step，如果不指定则分析所有step')
    comm_parser.add_argument('--comm-idx', type=int, help='指定要分析的通信操作索引，如果不指定则分析所有通信操作')
    comm_parser.add_argument('--fastest-card-idx', type=int, help='指定最快卡的索引，用于深度分析')
    comm_parser.add_argument('--slowest-card-idx', type=int, help='指定最慢卡的索引，用于深度分析')
    comm_parser.add_argument('--kernel-prefix', default='TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL', 
                            help='要检测的通信kernel前缀 (默认: TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL)\n'
                                 '支持的通信kernel前缀:\n'
                                 '  - TCDP_ONESHOT_ALLREDUCELL_SIMPLE\n'
                                 '  - TCDP_RING_ALLGATHER_SIMPLE\n'
                                 '  - TCDP_RING_ALLREDUCELL_SIMPLE\n'
                                 '  - TCDP_RING_ALLREDUCE_SIMPLE\n'
                                 '  - TCDP_RING_REDUCESCATTER_SIMPLE\n'
                                 '  - TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL')
    comm_parser.add_argument('--prev-kernel-pattern', default='TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL_BF16_ADD', 
                            help='上一个通信kernel的匹配模式，用于确定对比区间 (默认: TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL_BF16_ADD)')
    comm_parser.add_argument('--output-dir', default='.', help='输出目录 (默认: 当前目录)')
    comm_parser.add_argument('--show', type=str, default='',
                            help='显示额外信息，使用逗号分隔的选项:\n'
                                 '  dtype: 显示数据类型信息\n'
                                 '  shape: 显示形状和步长信息\n'
                                 '  kernel-names: 显示kernel名称\n'
                                 '  kernel-duration: 显示kernel持续时间\n'
                                 '  timestamp: 显示时间戳\n'
                                 '  readable-timestamp: 显示可读时间戳\n'
                                 '  kernel-timestamp: 显示kernel时间戳\n'
                                 '示例: --show "dtype,shape,kernel-duration"')
    
    # compare 命令 - 分析多个文件并对比
    compare_parser = subparsers.add_parser('compare', help='分析多个 JSON 文件并对比')
    compare_parser.add_argument('files', nargs='+', 
                               help='文件列表，支持多种格式:\n'
                                    '  单文件: file.json:label\n'
                                    '  多文件: "file1.json,file2.json,file3.json":label\n'
                                    '  目录: dir/:label (自动查找所有*.json文件)\n'
                                    '  通配符: "dir/*.json":label')
    compare_parser.add_argument('--aggregation', default='name',
                               help='聚合字段组合，使用逗号分隔的字段组合\n'
                                    '支持的字段: call_stack, name, shape, dtype\n'
                                    '示例: "name" 或 "name,shape" 或 "call_stack,name" 或 "name,shape,dtype"\n'
                                    '(默认: name)')
    compare_parser.add_argument('--show', type=str, default='',
                               help='显示额外信息，使用逗号分隔的选项:\n'
                                    '  dtype: 显示数据类型信息\n'
                                    '  shape: 显示形状和步长信息\n'
                                    '  kernel-names: 显示kernel名称\n'
                                    '  kernel-duration: 显示kernel持续时间\n'
                                    '  timestamp: 显示时间戳\n'
                                    '  readable-timestamp: 显示可读时间戳\n'
                                    '  kernel-timestamp: 显示kernel时间戳\n'
                                    '示例: --show "dtype,shape,kernel-duration"')
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


def parse_file_paths(file_pattern: str) -> List[str]:
    """
    解析文件路径，支持 glob 模式
    
    Args:
        file_pattern: 文件路径模式，支持 glob 通配符
        
    Returns:
        List[str]: 匹配的文件路径列表
    """
    # 检查是否包含 glob 通配符
    if '*' in file_pattern or '?' in file_pattern or '[' in file_pattern:
        # 使用 glob 模式匹配
        matched_files = glob.glob(file_pattern)
        if not matched_files:
            raise ValueError(f"glob 模式 {file_pattern} 没有匹配到任何文件")
        
        # 过滤出 JSON 文件
        json_files = [f for f in matched_files if f.lower().endswith('.json')]
        if not json_files:
            raise ValueError(f"glob 模式 {file_pattern} 没有匹配到任何 JSON 文件")
        
        return sorted(json_files)
    else:
        # 单个文件路径
        if not os.path.exists(file_pattern):
            raise ValueError(f"文件不存在: {file_pattern}")
        
        if not file_pattern.lower().endswith('.json'):
            raise ValueError(f"文件不是 JSON 格式: {file_pattern}")
        
        return [file_pattern]


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
    
    # 创建分析器
    analyzer = Analyzer()
    
    try:
        start_time = time.time()
        
        # 使用新的分析流程 - 支持多文件独立解析和聚合
        generated_files = analyzer.analyze_single_file_with_glob(
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


def run_comm_analysis(args):
    """运行通信性能分析"""
    print(f"=== 通信性能分析 ===")
    print(f"Pod目录: {args.pod_dir}")
    print(f"Step: {args.step if args.step is not None else '所有step'}")
    print(f"通信操作索引: {args.comm_idx if args.comm_idx is not None else '所有通信操作'}")
    if args.fastest_card_idx is not None:
        print(f"指定最快卡索引: {args.fastest_card_idx}")
    if args.slowest_card_idx is not None:
        print(f"指定最慢卡索引: {args.slowest_card_idx}")
    print(f"通信Kernel前缀: {args.kernel_prefix}")
    print(f"上一个通信Kernel模式: {args.prev_kernel_pattern}")
    print(f"输出目录: {args.output_dir}")
    print(f"显示选项: {args.show if args.show else '无'}")
    print()
    
    # 解析show选项
    try:
        show_options = parse_show_options(args.show)
        print(f"显示选项解析: {show_options}")
    except ValueError as e:
        print(f"错误: 显示选项解析失败 - {e}")
        return 1
    
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
        
        # 运行通信性能分析
        generated_files = analyzer.analyze_communication_performance(
            pod_dir=str(pod_path),
            step=args.step,
            comm_idx=args.comm_idx,
            fastest_card_idx=args.fastest_card_idx,
            slowest_card_idx=args.slowest_card_idx,
            kernel_prefix=args.kernel_prefix,
            prev_kernel_pattern=args.prev_kernel_pattern,
            output_dir=str(output_dir),
            show_dtype=show_options['dtype'],
            show_shape=show_options['shape'],
            show_kernel_names=show_options['kernel_names'],
            show_kernel_duration=show_options['kernel_duration'],
            show_timestamp=show_options['timestamp'],
            show_readable_timestamp=show_options['readable_timestamp'],
            show_kernel_timestamp=show_options['kernel_timestamp']
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
    print(f"聚合字段: {args.aggregation}")
    print(f"显示选项: {args.show if args.show else '无'}")
    print(f"打印markdown表格: {args.print_markdown}")
    print(f"特殊matmul: {args.special_matmul}")
    print(f"每个标签最大文件数: {args.max_files_per_label if args.max_files_per_label else '不限制'}")
    print(f"随机种子: {args.random_seed}")
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
            aggregation_spec=args.aggregation,
            show_dtype=show_options['dtype'],
            show_shape=show_options['shape'],
            show_kernel_names=show_options['kernel_names'],
            show_kernel_duration=show_options['kernel_duration'],
            show_timestamp=show_options['timestamp'],
            show_readable_timestamp=show_options['readable_timestamp'],
            show_kernel_timestamp=show_options['kernel_timestamp'],
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
        print("错误: 请指定命令 (analysis, compare, comm)")
        print("使用 --help 查看帮助信息")
        return 1
    
    if args.command == 'analysis':
        return run_analysis(args)
    elif args.command == 'compare':
        return run_compare_analysis(args)
    elif args.command == 'comm':
        return run_comm_analysis(args)
    else:
        print(f"错误: 未知命令: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
CLI主模块
"""

import argparse
import sys
from .commands import AnalysisCommand, CompareCommand, CommCommand


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
                                     '  name: 显示cpu_op名称\n'
                                     '示例: --show "dtype,shape,kernel-duration,name"')
    analysis_parser.add_argument('--print-markdown', action='store_true', 
                                help='是否在stdout中以markdown格式打印表格 (默认: False)')
    analysis_parser.add_argument('--drop', choices=['comm'], default=None,
                                help='丢弃特定类型的 event，目前支持: comm (丢弃包含TCDP的kernel events) (默认: None)')
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
                                    '  name: 显示cpu_op名称\n'
                                    '示例: --show "dtype,shape,kernel-duration,name"')
    compare_parser.add_argument('--print-markdown', action='store_true', 
                               help='是否在stdout中以markdown格式打印表格 (默认: False)')
    compare_parser.add_argument('--special-matmul', action='store_true',
                               help='是否进行特殊的 matmul 分析 (默认: False)')
    compare_parser.add_argument('--compare', type=str, default='',
                               help='比较选项，使用逗号分隔的选项:\n'
                                    '  dtype: 比较数据类型\n'
                                    '  shape: 比较形状\n'
                                    '  name: 比较操作名称\n'
                                    '  kernel_name: 比较kernel名称\n'
                                    '示例: --compare "dtype,shape,name"')
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


def main():
    """主函数"""
    args = parse_arguments()
    
    if not args.command:
        print("错误: 请指定命令 (analysis, compare, comm)")
        print("使用 --help 查看帮助信息")
        return 1
    
    if args.command == 'analysis':
        command = AnalysisCommand()
        return command.run(args)
    elif args.command == 'compare':
        command = CompareCommand()
        return command.run(args)
    elif args.command == 'comm':
        command = CommCommand()
        return command.run(args)
    else:
        print(f"错误: 未知命令: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

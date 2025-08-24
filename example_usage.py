#!/usr/bin/env python3
"""
PyTorch Profiler Parser Tool 使用示例

这个脚本展示了如何使用 torch_profiler_parser_tool 来分析 PyTorch profiler 数据。
"""

import sys
import os
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from torch_profiler_parser_tool import Analyzer

def example_single_file_analysis():
    """示例：单个文件分析"""
    print("=== 单个文件分析示例 ===")
    
    # 创建分析器
    analyzer = Analyzer()
    
    # 检查是否有可用的 JSON 文件
    json_files = list(Path("..").glob("*.json"))
    
    if not json_files:
        print("没有找到 JSON 文件，跳过单个文件分析示例")
        return
    
    # 使用第一个找到的 JSON 文件
    json_file = json_files[0]
    print(f"使用文件: {json_file}")
    
    try:
        # 加载数据
        print("正在加载数据...")
        data = analyzer.parser.load_json_file(str(json_file))
        print(f"加载完成，包含 {len(data.events)} 个事件")
        
        # 分析 cpu_op 和 kernel 的映射关系
        print("正在分析映射关系...")
        mapping = analyzer.analyze_cpu_op_kernel_mapping(data)
        print(f"找到 {len(mapping)} 个 cpu_op 的映射关系")
        
        # 生成 Excel 报告
        output_file = "example_single_analysis.xlsx"
        print(f"正在生成 Excel 报告: {output_file}")
        analyzer.generate_excel_from_mapping(mapping, output_file)
        
        # 生成 JSON 报告
        json_output = "example_single_analysis.json"
        print(f"正在生成 JSON 报告: {json_output}")
        analyzer.save_mapping_to_json(mapping, json_output)
        
        print("单个文件分析完成！")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")

def example_multiple_files_analysis():
    """示例：多文件对比分析"""
    print("\n=== 多文件对比分析示例 ===")
    
    # 检查是否有多个 JSON 文件
    json_files = list(Path("..").glob("*.json"))
    
    if len(json_files) < 2:
        print("没有找到足够的 JSON 文件，跳过多文件分析示例")
        return
    
    # 创建分析器
    analyzer = Analyzer()
    
    # 准备文件列表（使用文件名作为标签）
    file_labels = []
    for i, json_file in enumerate(json_files[:2]):  # 只使用前两个文件
        label = f"file_{i+1}"
        file_labels.append((str(json_file), label))
        print(f"  {label}: {json_file}")
    
    try:
        # 运行完整的多文件分析
        print("开始多文件分析...")
        analyzer.run_complete_analysis(
            file_labels, 
            output_dir="./example_results",
            output_formats=['json', 'xlsx']
        )
        
        print("多文件分析完成！")
        print("生成的文件:")
        for file_path in Path("./example_results").glob("*"):
            if file_path.suffix in ['.json', '.xlsx']:
                print(f"  {file_path}")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")

def example_programmatic_usage():
    """示例：编程接口使用"""
    print("\n=== 编程接口使用示例 ===")
    
    # 创建分析器
    analyzer = Analyzer()
    
    # 检查是否有可用的 JSON 文件
    json_files = list(Path("..").glob("*.json"))
    
    if not json_files:
        print("没有找到 JSON 文件，跳过编程接口示例")
        return
    
    json_file = json_files[0]
    print(f"使用文件: {json_file}")
    
    try:
        # 加载数据
        data = analyzer.parser.load_json_file(str(json_file))
        
        # 按 External id 重新组织数据
        print("按 External id 重新组织数据...")
        external_id_map = analyzer.reorganize_by_external_id(data)
        print(f"找到 {len(external_id_map)} 个有 External id 的事件组")
        
        # 分析映射关系
        mapping = analyzer.analyze_cpu_op_kernel_mapping(data)
        
        # 显示一些统计信息
        total_cpu_ops = len(mapping)
        total_kernels = 0
        
        for cpu_op_name, strides_map in mapping.items():
            for strides, dims_map in strides_map.items():
                for dims, types_map in dims_map.items():
                    for input_type, kernel_events in types_map.items():
                        total_kernels += len(kernel_events)
        
        print(f"统计信息:")
        print(f"  CPU 操作数量: {total_cpu_ops}")
        print(f"  Kernel 事件总数: {total_kernels}")
        
        # 显示前几个 CPU 操作
        print("\n前 5 个 CPU 操作:")
        for i, (cpu_op_name, strides_map) in enumerate(mapping.items()):
            if i >= 5:
                break
            
            kernel_count = 0
            for strides, dims_map in strides_map.items():
                for dims, types_map in dims_map.items():
                    for input_type, kernel_events in types_map.items():
                        kernel_count += len(kernel_events)
            
            print(f"  {cpu_op_name}: {kernel_count} 个 kernel")
        
        print("编程接口使用示例完成！")
        
    except Exception as e:
        print(f"示例过程中出错: {e}")

def main():
    """主函数"""
    print("PyTorch Profiler Parser Tool 使用示例")
    print("=" * 50)
    
    # 创建输出目录
    Path("./example_results").mkdir(exist_ok=True)
    
    # 运行示例
    example_single_file_analysis()
    example_multiple_files_analysis()
    example_programmatic_usage()
    
    print("\n" + "=" * 50)
    print("所有示例完成！")
    print("\n命令行使用示例:")
    print("1. 单个文件分析:")
    print("   python3 -m torch_profiler_parser_tool.cli single ../file.json --label example")
    print("\n2. 多文件对比分析:")
    print("   python3 -m torch_profiler_parser_tool.cli compare ../file1.json:fp32 ../file2.json:tf32")
    print("\n3. 指定输出格式:")
    print("   python3 -m torch_profiler_parser_tool.cli single ../file.json --output-format json")
    print("\n4. 指定输出目录:")
    print("   python3 -m torch_profiler_parser_tool.cli single ../file.json --output-dir ./results")

if __name__ == "__main__":
    main()

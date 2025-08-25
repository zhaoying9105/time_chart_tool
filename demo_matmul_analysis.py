#!/usr/bin/env python3
"""
Matmul算子专门分析演示脚本

这个脚本演示如何使用新增的matmul算子专门分析功能。
"""

import json
import tempfile
import os
from pathlib import Path

# 添加src目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from time_chart_tool.analyzer import Analyzer

def create_demo_data():
    """创建演示数据"""
    # 模拟比较分析数据，包含matmul算子
    demo_comparison_data = [
        {
            "cpu_op_name": "aten::mm",
            "cpu_op_input_strides": "((3, 1), (1, 3))",
            "cpu_op_input_dims": "((2048, 3), (3, 32))",
            "fp32_input_types": "('float', 'float')",
            "fp32_kernel_names": "void MLUUnion1KernelGemmRb<float, float, float, float, float, float>(void*, void*, void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, float, float, bool, bool, bool)",
            "fp32_kernel_count": 15,
            "fp32_kernel_mean_duration": 2.9066666666666667,
            "op_bf16_input_types": "('c10::BFloat16', 'c10::BFloat16')",
            "op_bf16_kernel_names": "void MLUUnion1KernelGemmRb<__bang_bfloat16, __bang_bfloat16, __bang_bfloat16, __bang_bfloat16, __bang_bfloat16, float>(void*, void*, void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, float, float, bool, bool, bool)",
            "op_bf16_kernel_count": 15,
            "op_bf16_kernel_mean_duration": 2.5893333333333333,
            "kernel_names_equal": False,
            "kernel_count_equal": True,
            "op_bf16_ratio_to_fp32": 0.8908256880733945
        },
        {
            "cpu_op_name": "aten::mm",
            "cpu_op_input_strides": "((1, 32), (32, 1))",
            "cpu_op_input_dims": "((1024, 32), (32, 64))",
            "fp32_input_types": "('float', 'float')",
            "fp32_kernel_names": "void MLUUnion1KernelGemmRb<float, float, float, float, float, float>(void*, void*, void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, float, float, bool, bool, bool)",
            "fp32_kernel_count": 20,
            "fp32_kernel_mean_duration": 3.5,
            "op_bf16_input_types": "('c10::BFloat16', 'c10::BFloat16')",
            "op_bf16_kernel_names": "void MLUUnion1KernelGemmRb<__bang_bfloat16, __bang_bfloat16, __bang_bfloat16, __bang_bfloat16, __bang_bfloat16, float>(void*, void*, void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, float, float, bool, bool, bool)",
            "op_bf16_kernel_count": 20,
            "op_bf16_kernel_mean_duration": 3.1,
            "kernel_names_equal": False,
            "kernel_count_equal": True,
            "op_bf16_ratio_to_fp32": 0.8857142857142857
        },
        {
            "cpu_op_name": "aten::mm",
            "cpu_op_input_strides": "((1, 64), (64, 1))",
            "cpu_op_input_dims": "((512, 64), (64, 128))",
            "fp32_input_types": "('float', 'float')",
            "fp32_kernel_names": "void MLUUnion1KernelGemmRb<float, float, float, float, float, float>(void*, void*, void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, float, float, bool, bool, bool)",
            "fp32_kernel_count": 25,
            "fp32_kernel_mean_duration": 4.2,
            "op_bf16_input_types": "('c10::BFloat16', 'c10::BFloat16')",
            "op_bf16_kernel_names": "void MLUUnion1KernelGemmRb<__bang_bfloat16, __bang_bfloat16, __bang_bfloat16, __bang_bfloat16, __bang_bfloat16, float>(void*, void*, void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, float, float, bool, bool, bool)",
            "op_bf16_kernel_count": 25,
            "op_bf16_kernel_mean_duration": 3.8,
            "kernel_names_equal": False,
            "kernel_count_equal": True,
            "op_bf16_ratio_to_fp32": 0.9047619047619048
        },
        {
            "cpu_op_name": "aten::add",
            "cpu_op_input_strides": "((1,), (1,))",
            "cpu_op_input_dims": "((1024,), (1024,))",
            "fp32_input_types": "('float', 'float')",
            "fp32_kernel_count": 10,
            "fp32_kernel_mean_duration": 1.0,
            "op_bf16_input_types": "('c10::BFloat16', 'c10::BFloat16')",
            "op_bf16_kernel_count": 10,
            "op_bf16_kernel_mean_duration": 0.9,
            "op_bf16_ratio_to_fp32": 0.9
        }
    ]
    
    return demo_comparison_data

def main():
    """主函数：演示matmul算子专门分析功能"""
    
    print("=== Matmul算子专门分析演示 ===")
    print()
    
    # 创建分析器实例
    analyzer = Analyzer()
    
    # 创建演示数据
    demo_data = create_demo_data()
    
    print("演示数据包含以下算子：")
    for item in demo_data:
        op_name = item.get('cpu_op_name', 'unknown')
        if op_name == 'aten::mm':
            input_dims = item.get('cpu_op_input_dims', '')
            dimensions = analyzer.extract_matmul_dimensions(input_dims)
            if dimensions:
                m, k, n = dimensions
                min_dim = min(m, k, n)
                print(f"  - {op_name}: 维度 ({m}, {k}, {n}), 最小维度 = {min_dim}")
        else:
            print(f"  - {op_name}")
    
    print()
    
    # 演示维度提取功能
    print("=== 维度提取功能演示 ===")
    for item in demo_data:
        if item.get('cpu_op_name') == 'aten::mm':
            input_dims = item.get('cpu_op_input_dims', '')
            dimensions = analyzer.extract_matmul_dimensions(input_dims)
            if dimensions:
                m, k, n = dimensions
                min_dim = min(m, k, n)
                print(f"输入维度: {input_dims}")
                print(f"解析结果: m={m}, k={k}, n={n}, min_dim={min_dim}")
                print()
    
    # 演示matmul分析功能
    print("=== Matmul分析功能演示 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 生成matmul分析
        analyzer.generate_matmul_analysis(demo_data, temp_dir)
        
        # 读取生成的JSON文件
        json_file = os.path.join(temp_dir, "matmul_analysis.json")
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                matmul_data = json.load(f)
            
            print(f"生成了 {len(matmul_data)} 条matmul分析记录：")
            for i, record in enumerate(matmul_data, 1):
                min_dim = record.get('mm_min_dim')
                fp32_duration = record.get('fp32_kernel_mean_duration')
                bf16_duration = record.get('op_bf16_kernel_mean_duration')
                ratio = record.get('op_bf16_ratio_to_fp32')
                
                print(f"  记录 {i}: min_dim={min_dim}, fp32={fp32_duration:.3f}, bf16={bf16_duration:.3f}, 比率={ratio:.3f}")
        
        # 检查生成的文件
        print(f"\n生成的文件：")
        for file_path in Path(temp_dir).glob("*"):
            if file_path.suffix in ['.json', '.xlsx', '.jpg']:
                print(f"  - {file_path.name}")
    
    print()
    print("=== 演示完成 ===")
    print("这个演示展示了：")
    print("1. 如何从matmul算子的输入维度中提取m, k, n")
    print("2. 如何计算最小维度 min_dim = min(m, k, n)")
    print("3. 如何按min_dim分组统计性能数据")
    print("4. 如何生成JSON、Excel和性能图表")
    print()
    print("在实际使用中，你可以：")
    print("- 使用命令行: time-chart-tool matmul file1.json:fp32 file2.json:bf16")
    print("- 使用编程接口: analyzer.run_complete_analysis_with_matmul(file_labels)")

if __name__ == "__main__":
    main()

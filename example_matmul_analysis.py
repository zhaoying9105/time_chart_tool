#!/usr/bin/env python3
"""
Matmul算子专门分析示例脚本

这个脚本演示如何使用新增的matmul算子专门分析功能。
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from time_chart_tool.analyzer import Analyzer

def main():
    """主函数：演示matmul算子专门分析功能"""
    
    # 创建分析器实例
    analyzer = Analyzer()
    
    # 示例：分析两个文件（假设你有两个包含matmul算子的JSON文件）
    # 这里使用示例文件路径，你需要替换为实际的文件路径
    file_labels = [
        ("fp32_profiler_data.json", "fp32"),
        ("bf16_profiler_data.json", "op_bf16")
    ]
    
    print("=== Matmul算子专门分析示例 ===")
    print("这个功能会：")
    print("1. 从比较分析数据中提取所有 'aten::mm' 算子")
    print("2. 解析每个matmul算子的输入维度 (m, k, n)")
    print("3. 计算最小维度 min_dim = min(m, k, n)")
    print("4. 按 min_dim 分组统计性能数据")
    print("5. 生成 JSON、Excel 和性能图表")
    print()
    
    # 检查文件是否存在
    existing_files = []
    for file_path, label in file_labels:
        if os.path.exists(file_path):
            existing_files.append((file_path, label))
            print(f"找到文件: {file_path} (标签: {label})")
        else:
            print(f"文件不存在: {file_path} (标签: {label})")
    
    if not existing_files:
        print("\n没有找到任何可用的文件。")
        print("请确保你有包含matmul算子数据的JSON文件。")
        print("文件应该包含 'aten::mm' 算子的性能数据。")
        return
    
    print(f"\n将分析 {len(existing_files)} 个文件...")
    
    # 运行包含matmul专门分析的完整分析流程
    try:
        analyzer.run_complete_analysis_with_matmul(
            file_labels=existing_files,
            output_dir=".",
            output_formats=['json', 'xlsx']
        )
        
        print("\n=== 分析完成 ===")
        print("生成的文件：")
        print("- comparison_analysis.json: 完整的比较分析数据")
        print("- comparison_analysis.xlsx: 完整的比较分析Excel表格")
        print("- matmul_analysis.json: matmul算子专门分析数据")
        print("- matmul_analysis.xlsx: matmul算子专门分析Excel表格")
        print("- matmul_performance_chart.jpg: matmul性能折线图")
        
        print("\nmatmul_analysis.json 的数据结构示例：")
        print("""
[
  {
    "mm_min_dim": 3,
    "fp32_input_types": "('float', 'float')",
    "fp32_kernel_count": 15,
    "fp32_kernel_mean_duration": 2.9066666666666667,
    "op_bf16_input_types": "('c10::BFloat16', 'c10::BFloat16')",
    "op_bf16_kernel_count": 15,
    "op_bf16_kernel_mean_duration": 2.5893333333333333,
    "op_bf16_ratio_to_fp32": 0.8908256880733945
  }
]
        """)
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

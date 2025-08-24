#!/usr/bin/env python3
"""
测试修复后的比较分析功能
"""

import os
import sys
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from advanced_analyzer import AdvancedAnalyzer

def test_comparison_analysis():
    """测试比较分析功能"""
    print("=== 测试比较分析功能 ===")
    
    # 检查文件是否存在
    fp32_file = "../fp32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_115_3021.json"
    tf32_file = "tf32_single_file_analysis.json"
    
    if not os.path.exists(fp32_file):
        print(f"错误: 找不到文件 {fp32_file}")
        return
    
    if not os.path.exists(tf32_file):
        print(f"错误: 找不到文件 {tf32_file}")
        return
    
    # 创建分析器
    analyzer = AdvancedAnalyzer()
    
    # 定义文件标签
    file_labels = [
        (fp32_file, "fp32"),
        (tf32_file, "tf32")
    ]
    
    print(f"准备分析 {len(file_labels)} 个文件:")
    for file_path, label in file_labels:
        print(f"  {file_path} -> {label}")
    
    # 运行完整分析
    analyzer.run_complete_analysis(
        file_labels=file_labels,
        single_output="single_file_analysis.xlsx",
        comparison_output="comparison_analysis_fixed.xlsx"
    )
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_comparison_analysis()

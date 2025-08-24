#!/usr/bin/env python3
"""
测试 CSV 输出功能
"""

import sys
import pandas as pd
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from advanced_analyzer import AdvancedAnalyzer


def test_csv_output():
    """测试 CSV 输出功能"""
    print("=== 测试 CSV 输出功能 ===")
    
    # 创建测试数据，包含逗号的字段
    test_data = [
        {
            'cpu_op_name': 'aten::add',
            'cpu_op_input_strides': '((1, 2), (3, 4))',
            'cpu_op_input_dims': '((5, 6), (7, 8))',
            'cpu_op_input_type': '["float", "float"]',
            'kernel_name': 'void MLUOpTensorElementBlock<(cnnlOpTensorParamMode_t)0, float, float, float, float, float>(float const*, float const*, float*, float, float, float, cnnlOpTensorDesc_t, DataInfo)',
            'kernel_count': 4,
            'kernel_min_duration': 1.8,
            'kernel_max_duration': 1.96,
            'kernel_mean_duration': 1.88,
            'kernel_std_duration': 0.092
        },
        {
            'cpu_op_name': 'aten::mul',
            'cpu_op_input_strides': '((0,), (1,))',
            'cpu_op_input_dims': '((2048,), (2048,))',
            'cpu_op_input_type': '["float", "float"]',
            'kernel_name': 'void MLUBlockKernelExpand1toA<int, unsigned int>(void const*, void*, Shapes<unsigned int>)',
            'kernel_count': 2,
            'kernel_min_duration': 1.8,
            'kernel_max_duration': 1.8,
            'kernel_mean_duration': 1.8,
            'kernel_std_duration': 0.0
        }
    ]
    
    df = pd.DataFrame(test_data)
    
    print("测试数据:")
    print(df.to_string())
    print()
    
    # 测试不同的 CSV 输出方法
    analyzer = AdvancedAnalyzer()
    
    # 测试安全的 CSV 输出
    print("测试安全的 CSV 输出:")
    analyzer._safe_csv_output(df, "test_output.csv")
    print()
    
    # 检查生成的文件
    print("生成的文件:")
    for file_path in Path(".").glob("test_output*"):
        print(f"  {file_path}")
        if file_path.suffix == '.csv':
            print("  文件内容预览:")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:3]):  # 显示前3行
                    print(f"    第{i+1}行: {line.strip()}")
            print()
    
    # 测试 JSON 输出
    json_file = "test_output.json"
    df.to_json(json_file, orient='records', indent=2, force_ascii=False)
    print(f"JSON 文件已生成: {json_file}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_csv_output()

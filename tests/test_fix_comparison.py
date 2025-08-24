#!/usr/bin/env python3
"""
测试修复后的比较分析功能 - 版本2
处理已经分析过的JSON文件，实现真正的性能比较
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

def load_analyzed_json(file_path: str) -> list:
    """加载已经分析过的JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def group_by_key(data: list) -> dict:
    """
    按 (cpu_op_name, cpu_op_input_strides, cpu_op_input_dims) 分组
    注意：不考虑 cpu_op_input_type
    """
    grouped = defaultdict(list)
    
    for item in data:
        # 创建分组键，不考虑 cpu_op_input_type
        key = (
            item['cpu_op_name'],
            item['cpu_op_input_strides'],
            item['cpu_op_input_dims']
        )
        grouped[key].append(item)
    
    return dict(grouped)

def calculate_performance_comparison(grouped_data: dict) -> list:
    """
    计算性能比较结果
    对于相同的 (cpu_op_name, cpu_op_input_strides, cpu_op_input_dims)，
    比较不同文件标签的 kernel mean duration
    """
    comparison_results = []
    
    for (cpu_op_name, input_strides, input_dims), items in grouped_data.items():
        # 按文件标签分组
        file_groups = defaultdict(list)
        for item in items:
            file_groups[item['file_label']].append(item)
        
        # 如果只有一个文件标签，跳过比较
        if len(file_groups) < 2:
            continue
        
        # 获取所有文件标签
        file_labels = list(file_groups.keys())
        
        # 为每个kernel_name创建比较行
        kernel_comparisons = defaultdict(dict)
        
        for file_label, file_items in file_groups.items():
            for item in file_items:
                kernel_name = item['kernel_name']
                kernel_comparisons[kernel_name][file_label] = {
                    'mean_duration': item['kernel_mean_duration'],
                    'count': item['kernel_count'],
                    'min_duration': item['kernel_min_duration'],
                    'max_duration': item['kernel_max_duration'],
                    'std_duration': item['kernel_std_duration']
                }
        
        # 生成比较结果
        for kernel_name, file_data in kernel_comparisons.items():
            if len(file_data) < 2:
                continue  # 跳过只有一个文件有数据的kernel
            
            # 选择基准文件（第一个文件）
            base_label = file_labels[0]
            base_data = file_data.get(base_label)
            
            if not base_data:
                continue
            
            base_duration = base_data['mean_duration']
            
            # 创建比较行
            comparison_row = {
                'cpu_op_name': cpu_op_name,
                'cpu_op_input_strides': input_strides,
                'cpu_op_input_dims': input_dims,
                'kernel_name': kernel_name,
                'base_file_label': base_label,
                'base_mean_duration': base_duration,
                'base_count': base_data['count'],
                'base_min_duration': base_data['min_duration'],
                'base_max_duration': base_data['max_duration'],
                'base_std_duration': base_data['std_duration']
            }
            
            # 添加其他文件的比较数据
            for file_label in file_labels[1:]:
                if file_label in file_data:
                    other_data = file_data[file_label]
                    other_duration = other_data['mean_duration']
                    
                    # 计算相对比值（相对于基准文件）
                    if base_duration > 0:
                        ratio = other_duration / base_duration
                        speedup = base_duration / other_duration
                    else:
                        ratio = float('inf') if other_duration > 0 else 1.0
                        speedup = float('inf') if other_duration > 0 else 1.0
                    
                    comparison_row.update({
                        f'{file_label}_mean_duration': other_duration,
                        f'{file_label}_count': other_data['count'],
                        f'{file_label}_min_duration': other_data['min_duration'],
                        f'{file_label}_max_duration': other_data['max_duration'],
                        f'{file_label}_std_duration': other_data['std_duration'],
                        f'{file_label}_ratio': ratio,
                        f'{file_label}_speedup': speedup
                    })
                else:
                    # 如果某个文件没有这个kernel的数据，填充为None
                    comparison_row.update({
                        f'{file_label}_mean_duration': None,
                        f'{file_label}_count': None,
                        f'{file_label}_min_duration': None,
                        f'{file_label}_max_duration': None,
                        f'{file_label}_std_duration': None,
                        f'{file_label}_ratio': None,
                        f'{file_label}_speedup': None
                    })
            
            comparison_results.append(comparison_row)
    
    return comparison_results

def merge_analyzed_data(file_labels: list) -> list:
    """合并多个已分析的文件数据"""
    all_data = []
    
    for file_path, label in file_labels:
        print(f"正在加载文件: {file_path} (标签: {label})")
        
        try:
            data = load_analyzed_json(file_path)
            # 为每个数据项添加文件标签
            for item in data:
                item['file_label'] = label
            all_data.extend(data)
            print(f"  成功加载 {len(data)} 行数据")
            
        except Exception as e:
            print(f"  加载文件失败: {e}")
            continue
    
    return all_data

def generate_comparison_excel(comparison_data: list, output_file: str = "comparison_analysis_fixed.xlsx") -> None:
    """生成比较分析的Excel表格"""
    if not comparison_data:
        print("没有数据可以生成比较分析文件")
        return
    
    df = pd.DataFrame(comparison_data)
    
    # 总是生成 JSON 文件，便于查看
    json_file = output_file.replace('.xlsx', '.json')
    df.to_json(json_file, orient='records', indent=2, force_ascii=False)
    print(f"JSON 文件已生成: {json_file}")
    
    try:
        df.to_excel(output_file, index=False)
        print(f"比较分析 Excel 文件已生成: {output_file}")
        print(f"包含 {len(comparison_data)} 行比较数据")
    except ImportError:
        # 如果没有 openpyxl，保存为 CSV
        csv_file = output_file.replace('.xlsx', '.csv')
        
        # 使用安全的 CSV 输出方法
        df.to_csv(csv_file, index=False, sep='\t', encoding='utf-8')
        quoted_csv_file = csv_file.replace('.csv', '_quoted.csv')
        df.to_csv(quoted_csv_file, index=False, quoting=1, encoding='utf-8')
        
        print(f"CSV 文件已生成 (制表符分隔): {csv_file}")
        print(f"CSV 文件已生成 (带引号): {quoted_csv_file}")
        print(f"包含 {len(comparison_data)} 行比较数据")
        
        print("注意: 需要安装 openpyxl 来生成 Excel 文件: pip install openpyxl")

def test_comparison_analysis():
    """测试比较分析功能"""
    print("=== 测试比较分析功能 - 版本2（性能比较） ===")
    
    # 检查文件是否存在
    fp32_file = "fp32_single_file_analysis.json"
    tf32_file = "tf32_single_file_analysis.json"
    
    if not os.path.exists(fp32_file):
        print(f"错误: 找不到文件 {fp32_file}")
        return
    
    if not os.path.exists(tf32_file):
        print(f"错误: 找不到文件 {tf32_file}")
        return
    
    # 定义文件标签
    file_labels = [
        (fp32_file, "fp32"),
        (tf32_file, "tf32")
    ]
    
    print(f"准备分析 {len(file_labels)} 个文件:")
    for file_path, label in file_labels:
        print(f"  {file_path} -> {label}")
    
    # 合并数据
    merged_data = merge_analyzed_data(file_labels)
    
    if merged_data:
        print(f"\n开始性能比较分析...")
        
        # 按键分组
        grouped_data = group_by_key(merged_data)
        print(f"找到 {len(grouped_data)} 个唯一的 (cpu_op_name, input_strides, input_dims) 组合")
        
        # 计算性能比较
        comparison_data = calculate_performance_comparison(grouped_data)
        print(f"生成了 {len(comparison_data)} 行比较数据")
        
        # 生成比较分析
        generate_comparison_excel(comparison_data, "comparison_analysis_fixed.xlsx")
        
        # 显示一些统计信息
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(f"\n比较分析统计信息:")
            print(f"总比较行数: {len(df)}")
            
            # 显示一些性能比较示例
            print(f"\n性能比较示例（前5行）:")
            if 'tf32_ratio' in df.columns:
                ratio_cols = [col for col in df.columns if 'ratio' in col]
                print(df[['cpu_op_name', 'kernel_name', 'base_file_label'] + ratio_cols].head())
            
            # 统计有比较数据的组合数量
            if 'tf32_ratio' in df.columns:
                valid_comparisons = df['tf32_ratio'].notna().sum()
                print(f"\n有fp32-tf32比较数据的组合: {valid_comparisons}")
    else:
        print("没有成功加载任何数据")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_comparison_analysis()

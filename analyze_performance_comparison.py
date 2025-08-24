#!/usr/bin/env python3
"""
分析性能比较结果
"""

import json
import pandas as pd
import numpy as np

def analyze_performance_comparison(json_file: str = "comparison_analysis_fixed.json"):
    """分析性能比较结果"""
    print("=== 性能比较分析 ===")
    
    # 加载数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    print(f"总比较行数: {len(df)}")
    
    # 基本统计
    if 'tf32_ratio' in df.columns:
        ratios = df['tf32_ratio'].dropna()
        speedups = df['tf32_speedup'].dropna()
        
        print(f"\n=== fp32 vs tf32 性能比较 ===")
        print(f"有效比较数据: {len(ratios)} 行")
        
        # 性能提升统计
        faster_than_fp32 = (speedups > 1.0).sum()
        slower_than_fp32 = (speedups < 1.0).sum()
        same_performance = (speedups == 1.0).sum()
        
        print(f"tf32 比 fp32 快的组合: {faster_than_fp32} ({faster_than_fp32/len(speedups)*100:.1f}%)")
        print(f"tf32 比 fp32 慢的组合: {slower_than_fp32} ({slower_than_fp32/len(speedups)*100:.1f}%)")
        print(f"性能相同的组合: {same_performance} ({same_performance/len(speedups)*100:.1f}%)")
        
        # 统计信息
        print(f"\n性能比值统计:")
        print(f"  平均比值 (tf32/fp32): {ratios.mean():.3f}")
        print(f"  中位数比值: {ratios.median():.3f}")
        print(f"  标准差: {ratios.std():.3f}")
        print(f"  最小值: {ratios.min():.3f}")
        print(f"  最大值: {ratios.max():.3f}")
        
        print(f"\n加速比统计:")
        print(f"  平均加速比 (fp32/tf32): {speedups.mean():.3f}")
        print(f"  中位数加速比: {speedups.median():.3f}")
        print(f"  最大加速比: {speedups.max():.3f}")
        print(f"  最小加速比: {speedups.min():.3f}")
        
        # 按CPU操作类型分组统计
        print(f"\n=== 按CPU操作类型统计 ===")
        op_stats = df.groupby('cpu_op_name').agg({
            'tf32_ratio': ['count', 'mean', 'std'],
            'tf32_speedup': ['mean', 'std']
        }).round(3)
        
        print(op_stats.head(10))
        
        # 找出性能差异最大的组合
        print(f"\n=== 性能差异最大的组合 ===")
        print("tf32 比 fp32 快最多的组合:")
        fastest = df.nlargest(5, 'tf32_speedup')[['cpu_op_name', 'kernel_name', 'base_mean_duration', 'tf32_mean_duration', 'tf32_speedup']]
        print(fastest)
        
        print("\ntf32 比 fp32 慢最多的组合:")
        slowest = df.nsmallest(5, 'tf32_speedup')[['cpu_op_name', 'kernel_name', 'base_mean_duration', 'tf32_mean_duration', 'tf32_speedup']]
        print(slowest)
        
        # 按性能提升程度分组
        print(f"\n=== 性能提升程度分布 ===")
        performance_groups = pd.cut(speedups, bins=[0, 0.8, 0.9, 1.0, 1.1, 1.2, float('inf')], 
                                  labels=['<0.8x', '0.8-0.9x', '0.9-1.0x', '1.0-1.1x', '1.1-1.2x', '>1.2x'])
        group_counts = performance_groups.value_counts().sort_index()
        print(group_counts)
        
    else:
        print("未找到 tf32_ratio 列，请检查数据格式")

if __name__ == "__main__":
    analyze_performance_comparison()

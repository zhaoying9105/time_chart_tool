#!/usr/bin/env python3
"""
PyTorch Profiler Parser Tool 使用示例
"""

import sys
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from parser import PyTorchProfilerParser
from advanced_analyzer import AdvancedAnalyzer


def basic_usage_example():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建解析器
    parser = PyTorchProfilerParser()
    
    # 加载数据（这里使用示例数据）
    # 实际使用时请替换为真实的 JSON 文件路径
    try:
        data = parser.load_json_file("../fp32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_115_3021.json")
        
        # 显示基本信息
        print(f"总事件数: {data.total_events}")
        print(f"Kernel 事件数: {len(data.kernel_events)}")
        print(f"唯一进程数: {len(data.unique_processes)}")
        
        # 搜索示例
        test_id = 233880
        any_id_events = parser.search_by_any_id(test_id)
        print(f"找到 {len(any_id_events)} 个 ID 为 {test_id} 的事件")
        
    except FileNotFoundError:
        print("示例文件不存在，跳过基本使用示例")


def advanced_usage_example():
    """高级使用示例"""
    print("\n=== 高级使用示例 ===")
    
    # 创建高级分析器
    analyzer = AdvancedAnalyzer()
    
    # 加载数据
    try:
        data = analyzer.parser.load_json_file("../fp32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_115_3021.json")
        
        # 功能5.1: 按 External id 重新组织数据
        external_id_map = analyzer.reorganize_by_external_id(data)
        print(f"找到 {len(external_id_map)} 个有 External id 的事件组")
        
        # 功能5.2: 分析 cpu_op 和 kernel 的映射关系
        mapping = analyzer.analyze_cpu_op_kernel_mapping(data)
        print(f"找到 {len(mapping)} 个 cpu_op 的映射关系")
        
        # 功能5.3: 生成分析表格
        analyzer.generate_excel_from_mapping(mapping, "example_analysis.xlsx")
        
        # 多文件分析示例
        file_labels = [
            ("../fp32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_115_3021.json", "fp32"),
            ("../tf32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_42_3021.json", "tf32")
        ]
        
        # 检查文件是否存在
        existing_files = []
        for file_path, label in file_labels:
            try:
                with open(file_path, 'r') as f:
                    existing_files.append((file_path, label))
            except FileNotFoundError:
                print(f"文件不存在: {file_path}")
        
        if len(existing_files) > 1:
            print(f"将分析 {len(existing_files)} 个文件")
            analyzer.run_complete_analysis(existing_files)
        else:
            print("可用文件不足，跳过多文件分析")
            
    except FileNotFoundError:
        print("示例文件不存在，跳过高级使用示例")


def main():
    """主函数"""
    print("PyTorch Profiler Parser Tool 使用示例")
    print("=" * 50)
    
    # 运行基本使用示例
    basic_usage_example()
    
    # 运行高级使用示例
    advanced_usage_example()
    
    print("\n=== 示例完成 ===")
    print("更多详细信息请参考 README.md 文件")


if __name__ == "__main__":
    main()

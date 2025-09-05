#!/usr/bin/env python3
"""
测试 Call Stack Demo 功能
"""

import os
import sys
from pathlib import Path

# 添加项目路径到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from time_chart_tool.analyzer import Analyzer
from call_stack_demo import main as run_demo


def test_call_stack_functionality():
    """测试 call stack 功能"""
    print("=== 开始测试 Call Stack 功能 ===")
    
    # 1. 运行 demo 生成 trace 文件
    print("1. 运行 demo 生成 trace 文件...")
    normal_trace, autocast_trace = run_demo()
    
    if not os.path.exists(normal_trace) or not os.path.exists(autocast_trace):
        print("错误: 无法生成 trace 文件")
        return False
    
    print(f"普通模式 trace: {normal_trace}")
    print(f"Autocast FP16 模式 trace: {autocast_trace}")
    
    # 2. 使用 analyzer 进行 call stack 对比
    print("\n2. 进行 call stack 对比分析...")
    analyzer = Analyzer()
    
    file_labels = [
        (normal_trace, "normal"),
        (autocast_trace, "autocast_fp16")
    ]
    
    output_dir = "call_stack_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 运行 call stack 对比
        analyzer.compare_by_call_stack(file_labels, output_dir)
        
        # 检查输出文件
        expected_files = [
            f"{output_dir}/call_stack_comparison.xlsx",
            f"{output_dir}/call_stack_comparison.csv", 
            f"{output_dir}/call_stack_comparison.json"
        ]
        
        print("\n3. 检查输出文件...")
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✓ 文件已生成: {file_path}")
            else:
                print(f"✗ 文件未生成: {file_path}")
        
        # 4. 验证结果
        print("\n4. 验证结果...")
        json_file = f"{output_dir}/call_stack_comparison.json"
        if os.path.exists(json_file):
            import json
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print(f"找到 {len(results)} 个 call stack")
            
            # 显示前几个 call stack 的信息
            for i, result in enumerate(results[:3]):
                print(f"\nCall Stack {i+1}:")
                print(f"  Call Stack: {result.get('call_stack', 'N/A')}")
                print(f"  Depth: {result.get('call_stack_depth', 'N/A')}")
                print(f"  Normal CPU Op Names: {result.get('normal_cpu_op_names', 'N/A')}")
                print(f"  Autocast CPU Op Names: {result.get('autocast_fp16_cpu_op_names', 'N/A')}")
                print(f"  CPU Op Names Equal: {result.get('cpu_op_names_equal', 'N/A')}")
                print(f"  Kernel Names Equal: {result.get('kernel_names_equal', 'N/A')}")
        
        print("\n=== Call Stack 功能测试完成 ===")
        return True
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parser_call_stack_features():
    """测试 parser 的 call stack 功能"""
    print("\n=== 测试 Parser Call Stack 功能 ===")
    
    from time_chart_tool.parser import PyTorchProfilerParser
    
    # 查找 trace 文件
    profiler_outputs = Path("profiler_outputs")
    if not profiler_outputs.exists():
        print("未找到 profiler 输出目录")
        return False
    
    trace_files = list(profiler_outputs.glob("*_trace.json"))
    if not trace_files:
        print("未找到 trace 文件")
        return False
    
    # 使用第一个 trace 文件进行测试
    trace_file = trace_files[0]
    print(f"使用 trace 文件: {trace_file}")
    
    parser = PyTorchProfilerParser()
    
    try:
        # 加载数据
        data = parser.load_json_file(trace_file)
        print(f"成功加载数据，包含 {data.total_events} 个事件")
        
        # 测试 call stack 相关功能
        print("\n测试 call stack 相关功能:")
        
        # 获取包含 call stack 的 cpu_op 事件
        cpu_op_events = parser.get_cpu_op_events_with_call_stack()
        print(f"包含 call stack 的 cpu_op 事件数: {len(cpu_op_events)}")
        
        # 获取唯一的 call stack
        unique_call_stacks = parser.get_unique_call_stacks()
        print(f"唯一的 call stack 数量: {len(unique_call_stacks)}")
        
        # 显示前几个 call stack
        for i, call_stack in enumerate(unique_call_stacks[:3]):
            print(f"  Call Stack {i+1}: {' -> '.join(call_stack)}")
        
        return True
        
    except Exception as e:
        print(f"Parser 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = True
    
    # 测试 parser 功能
    success &= test_parser_call_stack_features()
    
    # 测试完整的 call stack 功能
    success &= test_call_stack_functionality()
    
    if success:
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 部分测试失败")
        sys.exit(1)

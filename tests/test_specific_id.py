#!/usr/bin/env python3
"""
测试特定的 External id
"""

import sys
import time
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from time_chart_tool.parser import PyTorchProfilerParser


def main():
    """主函数"""
    print("=== 测试特定的 External id ===\n")
    
    # 创建解析器实例
    parser = PyTorchProfilerParser()
    
    # 指定要解析的文件路径
    json_file = "../fp32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_115_3021.json"
    
    # 检查文件是否存在
    if not Path(json_file).exists():
        print(f"错误: 文件不存在: {json_file}")
        return
    
    print(f"正在加载文件: {json_file}")
    start_time = time.time()
    
    try:
        # 加载 JSON 文件
        data = parser.load_json_file(json_file)
        load_time = time.time() - start_time
        print(f"文件加载完成，耗时: {load_time:.2f} 秒\n")
        
        # 测试特定的 External id
        test_ids = [239617, 239618, 239619, 233880]  # 包含一个存在的和一个不存在的
        
        for test_id in test_ids:
            print(f"--- 测试 External id: {test_id} ---")
            
            # 使用解析器搜索
            external_events = parser.search_by_external_id(test_id)
            
            if external_events:
                print(f"找到 {len(external_events)} 个事件")
                parser.print_events_summary(external_events, f"External id {test_id} 事件")
                
                # 显示前3个事件的详细信息
                print(f"External id {test_id} 事件详细信息 (前3个):")
                for i, event in enumerate(external_events[:3]):
                    print(f"  事件 {i+1}:")
                    print(f"    名称: {event.name}")
                    print(f"    类别: {event.cat}")
                    print(f"    进程: {event.pid}, 线程: {event.tid}")
                    print(f"    时间戳: {event.ts}")
                    if event.dur:
                        print(f"    持续时间: {event.dur}")
                    print(f"    参数: {event.args}")
                    print()
            else:
                print(f"未找到 External id 为 {test_id} 的事件")
            
            print()
        
        # 测试 correlation id 搜索（虽然之前显示没有 correlation id）
        print("--- 测试 correlation id 搜索 ---")
        correlation_events = parser.search_by_correlation_id("233880")
        if correlation_events:
            print(f"找到 {len(correlation_events)} 个 correlation id 事件")
        else:
            print("未找到 correlation id 事件")
        
        # 测试同时搜索
        print("\n--- 测试同时搜索 External id 和 correlation id ---")
        external_events, correlation_events = parser.search_by_id("233880")
        print(f"External id 事件数: {len(external_events)}")
        print(f"Correlation id 事件数: {len(correlation_events)}")
        
        # 测试实际存在的 ID
        print("\n--- 测试实际存在的 ID ---")
        external_events, correlation_events = parser.search_by_id(239617)
        print(f"External id 事件数: {len(external_events)}")
        print(f"Correlation id 事件数: {len(correlation_events)}")
        
        print(f"\n=== 测试完成 ===")
        print(f"总处理时间: {time.time() - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

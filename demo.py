#!/usr/bin/env python3
"""
PyTorch Profiler Parser Demo

演示如何使用 PyTorch Profiler Parser 工具库
"""

import sys
import time
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from parser import PyTorchProfilerParser


def main():
    """主函数"""
    print("=== PyTorch Profiler Parser Demo ===\n")
    
    # 创建解析器实例
    parser = PyTorchProfilerParser()
    
    # 指定要解析的文件路径
    json_file = "../fp32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_115_3021.json"
    
    # 检查文件是否存在
    if not Path(json_file).exists():
        print(f"错误: 文件不存在: {json_file}")
        print("请确保文件路径正确")
        return
    
    print(f"正在加载文件: {json_file}")
    print("这可能需要一些时间，因为文件很大...\n")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 加载 JSON 文件
        data = parser.load_json_file(json_file)
        
        # 记录加载时间
        load_time = time.time() - start_time
        print(f"文件加载完成，耗时: {load_time:.2f} 秒\n")
        
        # 功能1: 打印元数据
        print("功能1: 显示元数据")
        parser.print_metadata()
        
        # 功能2: 打印统计信息
        print("功能2: 显示统计信息")
        parser.print_statistics()
        
        # 功能3: 演示检索功能
        print("功能3: 演示检索功能")
        
        # 获取一些示例进程、线程、流 ID
        if data.unique_processes:
            sample_pid = data.unique_processes[0]
            print(f"\n--- 按进程 ID 检索 (PID: {sample_pid}) ---")
            events = parser.search_by_process(sample_pid)
            parser.print_events_summary(events, f"进程 {sample_pid} 事件")
        
        if data.unique_threads:
            sample_tid = data.unique_threads[0]
            print(f"\n--- 按线程 ID 检索 (TID: {sample_tid}) ---")
            events = parser.search_by_thread(sample_tid)
            parser.print_events_summary(events, f"线程 {sample_tid} 事件")
        
        if data.unique_streams:
            sample_stream = data.unique_streams[0]
            print(f"\n--- 按流 ID 检索 (Stream: {sample_stream}) ---")
            events = parser.search_by_stream(sample_stream)
            parser.print_events_summary(events, f"流 {sample_stream} 事件")
        
        # 功能4: 演示增强的 ID 检索功能
        print("\n功能4: 演示增强的 ID 检索功能")
        
        # 测试 ID: 233880
        test_id = 233880
        print(f"\n--- 检索 ID: {test_id} ---")
        
        # 4.1 测试 External id 搜索
        print(f"\n4.1 External id '{test_id}' 匹配的事件:")
        external_events = parser.search_by_external_id(test_id)
        if external_events:
            parser.print_events_summary(external_events, f"External id '{test_id}' 事件")
        else:
            print(f"未找到 External id 为 '{test_id}' 的事件")
        
        # 4.2 测试 Ev Idx 搜索
        print(f"\n4.2 Ev Idx '{test_id}' 匹配的事件:")
        ev_idx_events = parser.search_by_ev_idx(test_id)
        if ev_idx_events:
            parser.print_events_summary(ev_idx_events, f"Ev Idx '{test_id}' 事件")
        else:
            print(f"未找到 Ev Idx 为 '{test_id}' 的事件")
        
        # 4.3 测试 Python id 搜索
        print(f"\n4.3 Python id '{test_id}' 匹配的事件:")
        python_id_events = parser.search_by_python_id(test_id)
        if python_id_events:
            parser.print_events_summary(python_id_events, f"Python id '{test_id}' 事件")
        else:
            print(f"未找到 Python id 为 '{test_id}' 的事件")
        
        # 4.4 测试任意 ID 搜索（最全面的搜索）
        print(f"\n4.4 任意 ID '{test_id}' 匹配的事件（最全面的搜索）:")
        any_id_events = parser.search_by_any_id(test_id)
        if any_id_events:
            parser.print_events_summary(any_id_events, f"任意 ID '{test_id}' 事件")
            
            # 按 ID 类型分组显示
            external_count = len([e for e in any_id_events if e.external_id == test_id])
            correlation_count = len([e for e in any_id_events if e.correlation == test_id])
            ev_idx_count = len([e for e in any_id_events if e.ev_idx == test_id])
            python_id_count = len([e for e in any_id_events if e.python_id == test_id])
            
            print(f"\nID 类型分布:")
            print(f"  - External id: {external_count} 个")
            print(f"  - correlation: {correlation_count} 个")
            print(f"  - Ev Idx: {ev_idx_count} 个")
            print(f"  - Python id: {python_id_count} 个")
        else:
            print(f"未找到任意 ID 为 '{test_id}' 的事件")
        
        # 显示一些匹配事件的详细信息
        if external_events:
            print(f"\nExternal id '{test_id}' 事件详细信息 (前3个):")
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
        
        if any_id_events:
            print(f"\n任意 ID '{test_id}' 事件详细信息 (前3个):")
            for i, event in enumerate(any_id_events[:3]):
                print(f"  事件 {i+1}:")
                print(f"    名称: {event.name}")
                print(f"    类别: {event.cat}")
                print(f"    进程: {event.pid}, 线程: {event.tid}")
                print(f"    时间戳: {event.ts}")
                if event.dur:
                    print(f"    持续时间: {event.dur}")
                
                # 显示相关的 ID 字段
                id_fields = []
                if event.external_id == test_id:
                    id_fields.append(f"External id: {event.external_id}")
                if event.correlation == test_id:
                    id_fields.append(f"correlation: {event.correlation}")
                if event.ev_idx == test_id:
                    id_fields.append(f"Ev Idx: {event.ev_idx}")
                if event.python_id == test_id:
                    id_fields.append(f"Python id: {event.python_id}")
                
                print(f"    匹配的 ID 字段: {', '.join(id_fields)}")
                print(f"    参数: {event.args}")
                print()
        
        # 额外统计信息
        print("\n=== 额外统计信息 ===")
        
        # 统计有 External id 的事件
        external_id_events = [e for e in data.events if e.external_id is not None]
        print(f"包含 External id 的事件数: {len(external_id_events)}")
        
        # 统计有 correlation id 的事件
        correlation_id_events = [e for e in data.events if e.correlation_id is not None]
        print(f"包含 correlation id 的事件数: {len(correlation_id_events)}")
        
        # 显示一些 External id 和 correlation id 的示例值
        if external_id_events:
            sample_external_ids = list(set(e.external_id for e in external_id_events[:10]))
            print(f"External id 示例值: {sample_external_ids}")
        
        if correlation_id_events:
            sample_correlation_ids = list(set(e.correlation_id for e in correlation_id_events[:10]))
            print(f"Correlation id 示例值: {sample_correlation_ids}")
        
        print(f"\n=== Demo 完成 ===")
        print(f"总处理时间: {time.time() - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

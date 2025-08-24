#!/usr/bin/env python3
"""
测试增强的搜索功能
"""

import sys
import time
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from time_chart_tool.parser import PyTorchProfilerParser


def main():
    """主函数"""
    print("=== 测试增强的搜索功能 ===\n")
    
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
        
        # 测试 ID: 233880
        test_id = 233880
        print(f"=== 测试 ID: {test_id} ===\n")
        
        # 1. 测试 External id 搜索
        print("1. 测试 External id 搜索:")
        external_events = parser.search_by_external_id(test_id)
        if external_events:
            print(f"   找到 {len(external_events)} 个 External id 事件")
            for i, event in enumerate(external_events[:3]):
                print(f"   事件 {i+1}: {event.name} ({event.cat})")
        else:
            print("   未找到 External id 事件")
        print()
        
        # 2. 测试 correlation 搜索
        print("2. 测试 correlation 搜索:")
        correlation_events = parser.search_by_correlation(test_id)
        if correlation_events:
            print(f"   找到 {len(correlation_events)} 个 correlation 事件")
            for i, event in enumerate(correlation_events[:3]):
                print(f"   事件 {i+1}: {event.name} ({event.cat})")
        else:
            print("   未找到 correlation 事件")
        print()
        
        # 3. 测试 Ev Idx 搜索
        print("3. 测试 Ev Idx 搜索:")
        ev_idx_events = parser.search_by_ev_idx(test_id)
        if ev_idx_events:
            print(f"   找到 {len(ev_idx_events)} 个 Ev Idx 事件")
            for i, event in enumerate(ev_idx_events[:3]):
                print(f"   事件 {i+1}: {event.name} ({event.cat})")
        else:
            print("   未找到 Ev Idx 事件")
        print()
        
        # 4. 测试 Python id 搜索
        print("4. 测试 Python id 搜索:")
        python_id_events = parser.search_by_python_id(test_id)
        if python_id_events:
            print(f"   找到 {len(python_id_events)} 个 Python id 事件")
            for i, event in enumerate(python_id_events[:3]):
                print(f"   事件 {i+1}: {event.name} ({event.cat})")
        else:
            print("   未找到 Python id 事件")
        print()
        
        # 5. 测试任意 ID 搜索（最全面的搜索）
        print("5. 测试任意 ID 搜索:")
        any_id_events = parser.search_by_any_id(test_id)
        if any_id_events:
            print(f"   找到 {len(any_id_events)} 个任意 ID 事件")
            
            # 按 ID 类型分组显示
            external_count = len([e for e in any_id_events if e.external_id == test_id])
            correlation_count = len([e for e in any_id_events if e.correlation == test_id])
            ev_idx_count = len([e for e in any_id_events if e.ev_idx == test_id])
            python_id_count = len([e for e in any_id_events if e.python_id == test_id])
            
            print(f"   - External id: {external_count} 个")
            print(f"   - correlation: {correlation_count} 个")
            print(f"   - Ev Idx: {ev_idx_count} 个")
            print(f"   - Python id: {python_id_count} 个")
            
            # 显示前5个事件的详细信息
            print(f"\n   前5个事件详细信息:")
            for i, event in enumerate(any_id_events[:5]):
                print(f"   事件 {i+1}:")
                print(f"     名称: {event.name}")
                print(f"     类别: {event.cat}")
                print(f"     进程: {event.pid}, 线程: {event.tid}")
                print(f"     时间戳: {event.ts}")
                if event.dur:
                    print(f"     持续时间: {event.dur}")
                
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
                
                print(f"     匹配的 ID 字段: {', '.join(id_fields)}")
                print()
        else:
            print("   未找到任意 ID 事件")
        
        # 6. 统计信息
        print("6. 统计信息:")
        total_external_id_events = len([e for e in data.events if e.external_id is not None])
        total_correlation_events = len([e for e in data.events if e.correlation is not None])
        total_ev_idx_events = len([e for e in data.events if e.ev_idx is not None])
        total_python_id_events = len([e for e in data.events if e.python_id is not None])
        
        print(f"   - 包含 External id 的事件总数: {total_external_id_events}")
        print(f"   - 包含 correlation 的事件总数: {total_correlation_events}")
        print(f"   - 包含 Ev Idx 的事件总数: {total_ev_idx_events}")
        print(f"   - 包含 Python id 的事件总数: {total_python_id_events}")
        
        print(f"\n=== 测试完成 ===")
        print(f"总处理时间: {time.time() - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

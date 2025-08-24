#!/usr/bin/env python3
"""
PyTorch Profiler Parser 使用示例
"""

import sys
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from parser import PyTorchProfilerParser


def main():
    """简单的使用示例"""
    print("=== PyTorch Profiler Parser 使用示例 ===\n")
    
    # 创建解析器实例
    parser = PyTorchProfilerParser()
    
    # 指定 JSON 文件路径
    json_file = "../fp32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_115_3021.json"
    
    if not Path(json_file).exists():
        print(f"错误: 文件不存在: {json_file}")
        print("请确保文件路径正确")
        return
    
    try:
        # 1. 加载 JSON 文件
        print("1. 加载 JSON 文件...")
        data = parser.load_json_file(json_file)
        print(f"   成功加载，包含 {data.total_events} 个事件\n")
        
        # 2. 显示基本统计信息
        print("2. 基本统计信息:")
        print(f"   - 总事件数: {data.total_events}")
        print(f"   - Kernel 事件数: {len(data.kernel_events)}")
        print(f"   - CUDA 事件数: {len(data.cuda_events)}")
        print(f"   - 唯一进程数: {len(data.unique_processes)}")
        print(f"   - 唯一线程数: {len(data.unique_threads)}")
        print(f"   - 唯一流数: {len(data.unique_streams)}\n")
        
        # 3. 搜索特定 External id
        print("3. 搜索 External id 233880:")
        external_events = parser.search_by_external_id(233880)
        if external_events:
            print(f"   找到 {len(external_events)} 个事件")
            for i, event in enumerate(external_events[:2]):  # 显示前2个
                print(f"   事件 {i+1}: {event.name} ({event.cat})")
        else:
            print("   未找到事件")
        print()
        
        # 4. 按进程搜索
        if data.unique_processes:
            sample_pid = data.unique_processes[0]
            print(f"4. 按进程 ID {sample_pid} 搜索:")
            process_events = parser.search_by_process(sample_pid)
            print(f"   找到 {len(process_events)} 个事件\n")
        
        # 5. 按线程搜索
        if data.unique_threads:
            sample_tid = data.unique_threads[0]
            print(f"5. 按线程 ID {sample_tid} 搜索:")
            thread_events = parser.search_by_thread(sample_tid)
            print(f"   找到 {len(thread_events)} 个事件\n")
        
        # 6. 按流搜索
        if data.unique_streams:
            sample_stream = data.unique_streams[0]
            print(f"6. 按流 ID {sample_stream} 搜索:")
            stream_events = parser.search_by_stream(sample_stream)
            print(f"   找到 {len(stream_events)} 个事件\n")
        
        print("=== 示例完成 ===")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

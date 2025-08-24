#!/usr/bin/env python3
"""
测试实际存在的 External id
"""

import sys
import time
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from time_chart_tool.parser import PyTorchProfilerParser


def main():
    """主函数"""
    print("=== 测试实际存在的 External id ===\n")
    
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
        
        # 获取一些实际存在的 External id
        external_id_events = [e for e in data.events if e.external_id is not None]
        if external_id_events:
            # 获取前10个不同的 External id
            sample_external_ids = list(set(e.external_id for e in external_id_events[:100]))
            print(f"找到 {len(sample_external_ids)} 个不同的 External id 示例")
            
            # 测试前5个
            for i, test_id in enumerate(sample_external_ids[:5]):
                print(f"\n--- 测试 External id: {test_id} ---")
                external_events = parser.search_by_external_id(str(test_id))
                
                if external_events:
                    print(f"找到 {len(external_events)} 个事件")
                    # 显示前3个事件的详细信息
                    for j, event in enumerate(external_events[:3]):
                        print(f"  事件 {j+1}:")
                        print(f"    名称: {event.name}")
                        print(f"    类别: {event.cat}")
                        print(f"    进程: {event.pid}, 线程: {event.tid}")
                        print(f"    时间戳: {event.ts}")
                        if event.dur:
                            print(f"    持续时间: {event.dur}")
                        print(f"    参数: {event.args}")
                        print()
                else:
                    print("未找到事件")
        
        print(f"\n=== 测试完成 ===")
        print(f"总处理时间: {time.time() - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

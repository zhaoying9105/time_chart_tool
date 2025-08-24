#!/usr/bin/env python3
"""
调试 External id 的数据类型
"""

import sys
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from parser import PyTorchProfilerParser


def main():
    """主函数"""
    print("=== 调试 External id 数据类型 ===\n")
    
    # 创建解析器实例
    parser = PyTorchProfilerParser()
    
    # 指定要解析的文件路径
    json_file = "../fp32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_115_3021.json"
    
    # 检查文件是否存在
    if not Path(json_file).exists():
        print(f"错误: 文件不存在: {json_file}")
        return
    
    print(f"正在加载文件: {json_file}")
    
    try:
        # 加载 JSON 文件
        data = parser.load_json_file(json_file)
        print("文件加载完成\n")
        
        # 获取一些有 External id 的事件
        external_id_events = [e for e in data.events if e.external_id is not None]
        print(f"包含 External id 的事件数: {len(external_id_events)}")
        
        if external_id_events:
            # 检查前10个事件的 External id 类型和值
            print("\n前10个事件的 External id 信息:")
            for i, event in enumerate(external_id_events[:10]):
                external_id = event.external_id
                print(f"  事件 {i+1}:")
                print(f"    名称: {event.name}")
                print(f"    类别: {event.cat}")
                print(f"    External id: {external_id} (类型: {type(external_id)})")
                print(f"    External id 在 args 中的值: {event.args.get('External id', 'Not found')}")
                print()
            
            # 获取一些不同的 External id 值
            unique_external_ids = list(set(e.external_id for e in external_id_events[:100]))
            print(f"前100个事件中的不同 External id 值 (前10个):")
            for i, external_id in enumerate(unique_external_ids[:10]):
                print(f"  {i+1}: {external_id} (类型: {type(external_id)})")
            
            # 测试搜索功能
            if unique_external_ids:
                test_id = unique_external_ids[0]
                print(f"\n测试搜索 External id: {test_id} (类型: {type(test_id)})")
                
                # 直接搜索
                events = [e for e in data.events if e.external_id == test_id]
                print(f"直接搜索找到 {len(events)} 个事件")
                
                # 使用解析器搜索
                parser_events = parser.search_by_external_id(test_id)
                print(f"解析器搜索找到 {len(parser_events)} 个事件")
                
                if events:
                    print("第一个事件的详细信息:")
                    event = events[0]
                    print(f"  名称: {event.name}")
                    print(f"  类别: {event.cat}")
                    print(f"  进程: {event.pid}, 线程: {event.tid}")
                    print(f"  External id: {event.external_id}")
                    print(f"  参数: {event.args}")
        
        print(f"\n=== 调试完成 ===")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

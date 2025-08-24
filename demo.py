#!/usr/bin/env python3
"""
PyTorch Profiler 高级分析功能演示
"""

import sys
import time
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from torch_profiler_parser_tool.analyzer import Analyzer


def main():
    """主函数"""
    print("=== PyTorch Profiler 高级分析功能演示 ===\n")
    
    # 创建高级分析器实例
    analyzer = Analyzer()
    
    # 指定要分析的 JSON 文件
    json_file = "../fp32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_115_3021.json"
    
    # 检查文件是否存在
    if not Path(json_file).exists():
        print(f"错误: 文件不存在: {json_file}")
        print("请确保文件路径正确")
        return
    
    print(f"正在加载文件: {json_file}")
    start_time = time.time()
    
    try:
        # 加载 JSON 文件
        data = analyzer.parser.load_json_file(json_file)
        load_time = time.time() - start_time
        print(f"文件加载完成，耗时: {load_time:.2f} 秒\n")
        
        # 功能5.1: 用'External id'重新组织数据
        print("功能5.1: 用'External id'重新组织数据")
        external_id_map = analyzer.reorganize_by_external_id(data)
        print(f"找到 {len(external_id_map)} 个有 External id 的 cpu_op/kernel 事件组")
        
        # 显示一些示例
        sample_count = 0
        for external_id, events in external_id_map.items():
            if sample_count >= 3:  # 只显示前3个示例
                break
            cpu_op_count = len([e for e in events if e.cat == 'cpu_op'])
            kernel_count = len([e for e in events if e.cat == 'kernel'])
            print(f"  External id {external_id}: {cpu_op_count} 个 cpu_op, {kernel_count} 个 kernel")
            sample_count += 1
        
        if len(external_id_map) > 3:
            print(f"  ... 还有 {len(external_id_map) - 3} 个事件组")
        print()
        
        # 功能5.2: 分析 cpu_op 和 kernel 的映射关系
        print("功能5.2: 分析 cpu_op 和 kernel 的映射关系")
        mapping = analyzer.analyze_cpu_op_kernel_mapping(data)
        print(f"找到 {len(mapping)} 个 cpu_op 的映射关系")
        
        # 显示一些示例
        sample_count = 0
        for cpu_op_name, strides_map in mapping.items():
            if sample_count >= 3:  # 只显示前3个示例
                break
            total_kernel_events = 0
            for strides, dims_map in strides_map.items():
                for dims, types_map in dims_map.items():
                    for input_type, kernel_events in types_map.items():
                        total_kernel_events += len(kernel_events)
            
            print(f"  cpu_op '{cpu_op_name}': 对应 {total_kernel_events} 个 kernel 事件")
            sample_count += 1
        
        if len(mapping) > 3:
            print(f"  ... 还有 {len(mapping) - 3} 个 cpu_op")
        print()
        
        # 功能5.3: 生成 Excel 表格
        print("功能5.3: 生成 Excel 表格")
        output_file = "single_file_analysis.xlsx"
        analyzer.generate_excel_from_mapping(mapping, output_file)
        print()
        
        # 功能5.4 和功能6: 多文件分析演示
        print("功能5.4 和功能6: 多文件分析演示")
        
        # 使用两个真实的 JSON 文件
        fp32_file = "../fp32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_115_3021.json"
        tf32_file = "../tf32-trainer-runner_var_log_tiger_trace_ctr_torch_v210_fix_relu01_r9612311_0_42_3021.json"
        
        file_labels = []
        
        # 检查文件是否存在并添加到分析列表
        if Path(fp32_file).exists():
            file_labels.append((fp32_file, "fp32"))
            print(f"  fp32: {fp32_file}")
        else:
            print(f"  警告: fp32 文件不存在: {fp32_file}")
        
        if Path(tf32_file).exists():
            file_labels.append((tf32_file, "tf32"))
            print(f"  tf32: {tf32_file}")
        else:
            print(f"  警告: tf32 文件不存在: {tf32_file}")
        
        if not file_labels:
            print("  没有找到可用的文件，使用单文件演示")
            file_labels = [(json_file, "single_file")]
        
        print(f"将分析 {len(file_labels)} 个文件")
        print()
        
        # 运行完整的多文件分析
        analyzer.run_complete_analysis(file_labels)
        
        print(f"\n=== 演示完成 ===")
        print(f"总处理时间: {time.time() - start_time:.2f} 秒")
        
        # 显示生成的文件
        print("\n生成的文件:")
        for file_path in Path(".").glob("*.xlsx"):
            print(f"  {file_path}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
通信性能分析器
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .excel_generator import (
    _generate_raw_data_excel,
    _generate_statistics_excel,
)


from .deep_analysis import (
    _perform_deep_analysis,
    _auto_select_comm_target
)

from .data_extractor import (
    _extract_communication_data,
)
from .utils import _scan_executor_folders


def analyze_communication_performance(pod_dir: str, step: Optional[int] = None, comm_idx: Optional[int] = None,
                                     fastest_card_idx: Optional[int] = None, slowest_card_idx: Optional[int] = None,
                                     kernel_prefix: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL",
                                     prev_kernel_pattern: str = "TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL_BF16_ADD",
                                     output_dir: str = ".",
                                     show_timestamp: bool = False, 
                                     show_readable_timestamp: bool = False) -> List[Path]:
    """
    分析通信性能
    
    Args:
        pod_dir: Pod目录路径
        step: 指定要分析的step
        comm_idx: 指定要分析的通信操作索引
        fastest_card_idx: 指定最快卡的索引
        slowest_card_idx: 指定最慢卡的索引
        kernel_prefix: 通信kernel前缀
        prev_kernel_pattern: 上一个通信kernel的匹配模式
        output_dir: 输出目录
        show_dtype: 是否显示数据类型
        show_shape: 是否显示形状信息
        show_kernel_names: 是否显示kernel名称
        show_kernel_duration: 是否显示kernel持续时间
        show_timestamp: 是否显示时间戳
        show_readable_timestamp: 是否显示可读时间戳
        show_kernel_timestamp: 是否显示kernel时间戳
        show_call_stack: 是否显示调用栈信息
        
    Returns:
        List[Path]: 生成的文件路径列表
    """
    print("=== 通信性能分析 ===")
    
    # 1. 扫描executor文件夹
    executor_folders = _scan_executor_folders(pod_dir)
    if not executor_folders:
        print("错误: 没有找到executor文件夹")
        return []
    
    print(f"找到 {len(executor_folders)} 个executor文件夹")
    
    generated_files = []
    
    # 2. 确定是否为快速分析模式 (指定了 step 和两个 card_idx)
    is_fast_mode = (step is not None and fastest_card_idx is not None and slowest_card_idx is not None)
    target_indices = [fastest_card_idx, slowest_card_idx] if is_fast_mode else None
    
    if is_fast_mode:
        print(f"检测到已指定最快卡({fastest_card_idx})和最慢卡({slowest_card_idx})，进入快速分析模式")
    
    # 3. 提取通信数据
    comm_data = _extract_communication_data(executor_folders, step, kernel_prefix, target_indices)
    if not comm_data:
        print("错误: 没有找到通信数据")
        return []
    
    # 4. 生成统计报表 (非快速模式)
    if not is_fast_mode:
        # 生成原始数据Excel
        raw_data_file = _generate_raw_data_excel(comm_data, output_dir)
        generated_files.append(raw_data_file)
        
        # 生成统计信息Excel
        stats_file = _generate_statistics_excel(comm_data, output_dir)
        generated_files.append(stats_file)
    
    # 5. 深度分析 (如果指定了 step)
    if step is not None:
    # if False:
        # 确保step在数据中
        if step not in comm_data:
            print(f"警告: 未找到 Step {step} 的数据")
            return generated_files
        
        # 自动选择 comm_idx
        if comm_idx is None:
            comm_idx, auto_fastest, auto_slowest = _auto_select_comm_target(comm_data, step)
            if comm_idx is None:
                print(f"无法自动选择 comm_idx，跳过深度分析")
                return generated_files
            
            # 如果用户没有指定快慢卡，使用自动检测的结果
            if not is_fast_mode:
                print(f"自动选择comm_idx: {comm_idx}")
                if fastest_card_idx is None:
                    fastest_card_idx = auto_fastest
                    print(f"  建议最快卡: {fastest_card_idx}")
                if slowest_card_idx is None:
                    slowest_card_idx = auto_slowest
                    print(f"  建议最慢卡: {slowest_card_idx}")
        
        # 执行深度分析
        deep_analysis_files = _perform_deep_analysis(
            comm_data, executor_folders, step, comm_idx, output_dir, 
            kernel_prefix, prev_kernel_pattern, fastest_card_idx, slowest_card_idx,
            show_timestamp, show_readable_timestamp
        )
        
        if deep_analysis_files:
            if isinstance(deep_analysis_files, list):
                generated_files.extend(deep_analysis_files)
            else:
                generated_files.append(deep_analysis_files)
    
    return generated_files

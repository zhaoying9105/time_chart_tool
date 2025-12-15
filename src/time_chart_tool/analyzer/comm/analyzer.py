"""
通信性能分析器
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .excel_generator import (
    _generate_raw_data_excel,
    _generate_statistics_excel,
)

from .data_extractor import (
    _extract_communication_data,
)
from .utils import _scan_executor_folders


def analyze_communication_performance(pod_dir: str, step: Optional[int] = None, comm_idx: Optional[int] = None,
                                     fastest_card_idx: Optional[int] = None, slowest_card_idx: Optional[int] = None,
                                     kernel_prefix: str = "TCDP_",
                                     prev_kernel_pattern: str = "TCDP__BF16_ADD",
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
    
    return generated_files

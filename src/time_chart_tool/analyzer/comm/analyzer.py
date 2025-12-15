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


def analyze_communication_performance(pod_dir: str, step: Optional[int] = None, kernel_prefix: str = "TCDP_", output_dir: str = ".") -> List[Path]:
    """
    分析通信性能
    
    Args:
        pod_dir: Pod目录路径
        step: 指定要分析的step
        kernel_prefix: 通信kernel前缀
        output_dir: 输出目录
        
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
    
    
    # 2. 提取通信数据
    comm_data = _extract_communication_data(executor_folders, step, kernel_prefix)
    if not comm_data:
        print("错误: 没有找到通信数据")
        return []
    
    # 3. 生成统计报表 (非快速模式)
    raw_data_file = _generate_raw_data_excel(comm_data, output_dir)
    generated_files.append(raw_data_file)
    
    # 生成统计信息Excel
    stats_file = _generate_statistics_excel(comm_data, output_dir)
    generated_files.append(stats_file)
    
    return generated_files

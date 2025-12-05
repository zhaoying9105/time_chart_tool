
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from .utils import _calculate_step_statistics

def _generate_raw_data_excel(all2all_data: Dict[int, Dict[int, List[float]]], output_dir: str) -> Path:
    """生成原始数据Excel文件"""
    return _generate_excel_file(
        all2all_data, output_dir, "communication_raw_data.xlsx", 
        _create_raw_data_sheets
    )

def _generate_statistics_excel(all2all_data: Dict[int, Dict[int, List[float]]], output_dir: str) -> Path:
    """生成统计信息Excel文件"""
    return _generate_excel_file(
        all2all_data, output_dir, "communication_statistics.xlsx",
        _create_statistics_sheets
    )

def _generate_excel_file(data, output_dir: str, filename: str, sheet_creator) -> Path:
    """通用的Excel文件生成方法"""
    if pd is None:
        raise ImportError("pandas is required for Excel output. Please install pandas and openpyxl.")
    
    output_path = Path(output_dir) / filename
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        sheet_creator(data, writer)
    
    print(f"生成文件: {output_path}")
    return output_path

def _create_raw_data_sheets(all2all_data, writer):
    """创建原始数据表格"""
    for step, card_data in all2all_data.items():
        rows = []
        for card_idx, entries in card_data.items():
            for i, entry in enumerate(entries):
                rows.append({
                    'Step': step, 
                    'Card_Index': card_idx, 
                    'Comm_Index': i, 
                    'Name': entry.name,
                    'Timestamp': entry.ts,
                    'Duration_us': entry.dur,
                    'In_Msg_Nelems': entry.in_msg_nelems,
                    'Out_Msg_Nelems': entry.out_msg_nelems,
                    'Group_Size': entry.group_size,
                    'Dtype': entry.dtype,
                    'Process_Group_Ranks': entry.process_group_ranks
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=f'Step_{step}', index=False)

def _create_statistics_sheets(all2all_data, writer):
    """创建统计信息表格"""
    # Step统计
    step_stats = _calculate_step_statistics(all2all_data)
    if step_stats:
        pd.DataFrame(step_stats).to_excel(writer, sheet_name='Step_Statistics', index=False)

"""
通信数据提取模块
"""

import os
import glob
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

@dataclass
class CommunicationData:
    name: str
    ts: float
    dur: float
    in_msg_nelems: int
    out_msg_nelems: int
    group_size: int
    dtype: str
    process_group_ranks: str

def _extract_communication_data(executor_folders: List[str], step: Optional[int] = None, 
                               kernel_prefix: str = "TCDP_",
                               target_card_indices: Optional[List[int]] = None) -> Dict[int, Dict[int, List[CommunicationData]]]:
    """
    提取通信数据
    
    Args:
        executor_folders: executor文件夹列表
        step: 指定要分析的step
        kernel_prefix: 通信kernel前缀
        target_card_indices: 指定要提取的card索引列表，如果为None则提取所有
        
    Returns:
        Dict[int, Dict[int, List[CommunicationData]]]: 通信数据 {step: {card_idx: [CommunicationData]}}
    """
    comm_data = {}
    total_json_files = 0
    
    print(f"开始提取通信数据:")
    print(f"  - Executor文件夹数量: {len(executor_folders)}")
    print(f"  - 目标step: {step}")
    print(f"  - Kernel前缀: {kernel_prefix}")
    if target_card_indices:
        print(f"  - 目标Card索引: {target_card_indices}")
    
    for executor_folder in executor_folders:
        # 如果指定了target_card_indices，我们可以只查找特定模式的文件名，或者在遍历时过滤
        # 这里为了简单，还是遍历文件但过滤
        
        # 查找JSON文件
        json_files = glob.glob(os.path.join(executor_folder, "*.json"))
        total_json_files += len(json_files)
        # print(f"    找到 {len(json_files)} 个JSON文件")
        
        for json_file in json_files:
            step_num, card_idx = _parse_json_filename(os.path.basename(json_file))
            
            if step_num is None or card_idx is None:
                continue
            
            if step is not None and step_num != step:
                continue
            
            if target_card_indices is not None and card_idx not in target_card_indices:
                continue
            
            # 提取通信数据
            print(f"    正在处理 {json_file}")
            comm_entries = _extract_communication_entries(json_file, kernel_prefix)
            
            if comm_entries:
                if step_num not in comm_data:
                    comm_data[step_num] = {}
                comm_data[step_num][card_idx] = comm_entries
    
    return comm_data

def _parse_json_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """
    解析JSON文件名，提取step和card索引
    
    Args:
        filename: JSON文件名
        
    Returns:
        Tuple[Optional[int], Optional[int]]: (step, card_idx)
    """
    # 匹配多种文件名模式
    patterns = [
        r'(\d+)_(\d+)\.json',
        r'step(\d+)\.json',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:
                card_idx = int(match.group(1))
                step = int(match.group(2))
                return step, card_idx
            elif len(match.groups()) == 1:
                # 只有step信息，card_idx设为0
                step = int(match.group(1))
                return step, 0
    
    return None, None

def _extract_communication_entries(json_file_path: str, kernel_prefix: str = "TCDP_") -> List[CommunicationData]:
    """
    从JSON文件中提取指定通信kernel前缀的详细信息
    
    Args:
        json_file_path: JSON文件路径
        kernel_prefix: 要检测的通信kernel前缀
        
    Returns:
        List[CommunicationData]: 通信数据列表，最多6个
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        escaped_kernel_prefix = re.escape(kernel_prefix)
        
        entries = []
        
        # 1. 尝试加载整个JSON
        try:
            data = json.loads(content)
            events = []
            if isinstance(data, dict):
                if 'traceEvents' in data:
                    events = data['traceEvents']
                else:
                    # 可能是其他格式，或者是单个对象
                    pass
            elif isinstance(data, list):
                events = data
            
            # 过滤并提取
            count = 0
            for event in events:
                if count >= 6:
                    break
                    
                if event.get('name', '').startswith(kernel_prefix) and event.get('ph') == 'X':
                    args = event.get('args', {})
                    
                    entry = CommunicationData(
                        name=event.get('name', ''),
                        ts=event.get('ts', 0.0),
                        dur=event.get('dur', 0.0),
                        in_msg_nelems=args.get('In msg nelems', 0),
                        out_msg_nelems=args.get('Out msg nelems', 0),
                        group_size=args.get('Group size', 0),
                        dtype=args.get('dtype', ''),
                        process_group_ranks=args.get('Process Group Ranks', '')
                    )
                    entries.append(entry)
                    count += 1
            
            if entries:
                return entries
                
        except json.JSONDecodeError:
            # 如果直接解析失败（可能是截断的JSON或其他格式），回退到正则
            pass
        return []
        
    except Exception as e:
        print(f"    读取JSON文件失败 {json_file_path}: {e}")
        return []

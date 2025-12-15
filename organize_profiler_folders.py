import os
import re
import json
import shutil
from datetime import datetime
import glob

def get_time_range_from_json(file_path):
    """
    解析 JSON 文件以找到第一个和最后一个 "ts" 时间戳以及 "baseTimeNanoseconds"。
    返回 (start_time_ns, end_time_ns) 或者如果解析失败则返回 None。
    """
    first_ts = None
    last_ts = None
    base_time_ns = None
    
    ts_pattern = re.compile(r'"ts":\s*(\d+(?:\.\d+)?)')
    base_time_pattern = re.compile(r'"baseTimeNanoseconds":\s*(\d+)')

    try:
        with open(file_path, 'r') as f:
            # 逐行读取以避免将巨大文件加载到内存中
            # 读取前几行以找到第一个时间戳
            for line in f:
                ts_match = ts_pattern.search(line)
                if ts_match:
                    first_ts = float(ts_match.group(1))
                    break
            
            if first_ts is None:
                return None
            
            # 移动到末尾以找到最后一个时间戳和 baseTimeNanoseconds
            # 这是一种启发式方法；如果文件很大且是单行 JSON，这可能会很慢或很棘手。
            # 然而，提示暗示了基于行的结构或可管理的大小。
            # 为了在处理大文件时提高效率，我们可以使用 `seek` 读取末尾。
            # 但是 "ts" 可能会分散各处。让我们尝试读取文件。
            # 如果文件是严格的基于行的日志，从末尾读取是好的。
            # 如果是带有换行符的标准 JSON，这种方法也有效。
            
            # 让我们尝试通过高效读取文件来找到最后一个 ts。
            # 既然用户提到“grep 最后几行”，我们可以通过编程尝试这种方法。
            # 如果文件不太大，我们将读取整个文件，或者移动到末尾。
            
            # 重新打开以高效读取或继续读取
            # 如果文件巨大，我们应该实现类似 tail 的读取。
            
            f.seek(0, 2) # 移动到文件末尾
            file_size = f.tell()
            
            # 读取最后 10KB 以找到 baseTimeNanoseconds 和最后一个 ts
            read_size = min(file_size, 50 * 1024) # 50KB 应该足以包含末尾的元数据
            f.seek(file_size - read_size)
            content = f.read()
            
            # 在最后一块内容中找到所有 ts
            ts_matches = list(ts_pattern.finditer(content))
            if ts_matches:
                last_ts = float(ts_matches[-1].group(1))
            else:
                # 如果最后一块没有 ts，我们可能需要读取更多或扫描整个文件。
                # 如果需要，回退到扫描整个文件，但让我们假设用户的提示是有效的。
                # 用户说：“在最后几行找到最后一个 ts”
                # 如果我们错过了，我们可能会默认使用 first_ts（点间隔）或扫描更多。
                pass

            # 如果我们仍然没有 last_ts，让我们从头开始扫描文件（较慢但更安全）
            if last_ts is None:
                f.seek(0)
                for line in f:
                    ts_match = ts_pattern.search(line)
                    if ts_match:
                        last_ts = float(ts_match.group(1))
            
            # 找到 baseTimeNanoseconds
            base_match = base_time_pattern.search(content)
            if base_match:
                base_time_ns = int(base_match.group(1))
            
            if first_ts is not None and last_ts is not None and base_time_ns is not None:
                # 计算纳秒级的绝对时间戳
                # ts 是微秒，baseTimeNanoseconds 是纳秒。
                # 我们需要将 ts 转换为纳秒 (乘以 1000) 然后加上 baseTimeNanoseconds。
                
                start_abs = base_time_ns + int(first_ts * 1000)
                end_abs = base_time_ns + int(last_ts * 1000)
                
                # 转换为秒以获得可读格式
                # 假设结果是纳秒（因为 base 是纳秒）
                return start_abs, end_abs

    except Exception as e:
        print(f"处理 {file_path} 时出错: {e}")
        return None
    
    return None

def format_time(ns_timestamp):
    """将纳秒时间戳转换为可读字符串。"""
    # 将纳秒转换为秒
    seconds = ns_timestamp / 1e9
    try:
        dt = datetime.fromtimestamp(seconds)
        return dt.strftime("%Y-%m-%d_%H-%M-%S")
    except ValueError:
        # 处理潜在的范围超出问题
        return str(ns_timestamp)

import argparse

def main():
    parser = argparse.ArgumentParser(description='Organize profiler folders by time overlap.')
    parser.add_argument('root_dir', help='Root directory containing profiler folders')
    args = parser.parse_args()
    
    root_dir = args.root_dir
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        return
    
    # 找到所有子目录
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    folder_ranges = []

    print(f"找到 {len(subdirs)} 个子目录。")

    for subdir in subdirs:
        dir_path = os.path.join(root_dir, subdir)
        # 找到第一个 json 文件
        json_files = sorted(glob.glob(os.path.join(dir_path, "trace_*.json")))
        
        if not json_files:
            print(f"在 {subdir} 中未找到 JSON 文件，跳过。")
            continue
            
        # 使用第一个 json 文件
        first_json = json_files[0]
        print(f"正在处理 {first_json}...")
        
        time_range = get_time_range_from_json(first_json)
        
        if time_range:
            start, end = time_range
            folder_ranges.append({
                "path": dir_path,
                "name": subdir,
                "start": start,
                "end": end
            })
            print(f"  范围: {start} - {end}")
        else:
            print(f"  无法确定 {subdir} 的时间范围")

    # 按重叠时间分组
    # 按开始时间排序
    folder_ranges.sort(key=lambda x: x["start"])
    
    groups = []
    if folder_ranges:
        current_group = [folder_ranges[0]]
        current_end = folder_ranges[0]["end"]
        
        for i in range(1, len(folder_ranges)):
            item = folder_ranges[i]
            # 检查重叠：项目在当前组结束之前开始
            # 如果需要，我们可以添加一个小缓冲区，但严格重叠是：start < current_end
            if item["start"] < current_end:
                current_group.append(item)
                current_end = max(current_end, item["end"])
            else:
                groups.append(current_group)
                current_group = [item]
                current_end = item["end"]
        groups.append(current_group)

    print(f"识别出 {len(groups)} 个组。")

    # 创建文件夹并移动/组织
    for idx, group in enumerate(groups):
        # 确定组的标签
        # 使用组的最小开始时间格式化
        min_start = min(item["start"] for item in group)
        max_end = max(item["end"] for item in group)
        
        label = format_time(min_start)
        group_dir_name = f"{label}_group_{idx}"
        group_dir_path = os.path.join(root_dir, group_dir_name)
        
        print(f"创建组目录: {group_dir_path}")
        if not os.path.exists(group_dir_path):
            os.mkdir(group_dir_path)
            
        for item in group:
            src_path = item["path"]
            dst_path = os.path.join(group_dir_path, item["name"])
            
            print(f"  移动 {item['name']} 到 {group_dir_name}")
            try:
                shutil.move(src_path, dst_path)
            except Exception as e:
                print(f"  移动 {item['name']} 时出错: {e}")

if __name__ == "__main__":
    main()

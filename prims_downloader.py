#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import requests
import argparse
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Union
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class UrlParser:
    """URL解析工具类"""
    
    @staticmethod
    def extract_project_id_from_url(primus_url: str) -> str:
        """
        从primus URL中提取项目ID
        
        Args:
            primus_url: 完整的primus URL
            
        Returns:
            str: 项目ID
        """
        from urllib.parse import urlparse
        parsed_url = urlparse(primus_url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        # 查找包含项目ID的部分（通常是包含连字符的长字符串）
        pod_name = 'unknown'
        for part in path_parts:
            if '-' in part and len(part) > 10:  # 项目ID通常包含连字符且较长
                pod_name = part
                break
        # 如果没找到，则使用倒数第三个部分（通常是项目ID的位置）
        if pod_name == 'unknown' and len(path_parts) >= 3:
            pod_name = path_parts[-3]
        
        return pod_name

class NodeInfoFetcher:
    """节点信息获取器 - 负责从API获取和过滤节点信息"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
    def get_all_nodes_info(self, primus_url: str, state: str = "") -> List[Dict]:
        """
        获取pod下所有node的信息
        
        Args:
            primus_url: 完整的primus URL，例如：https://primus-ui-cn.byted.org/gaura-yx/nj-ed62daba-2db1-4b59-aaa6-cf939-pj/webapps/primus/
            state: 状态过滤条件
            
        Returns:
            List[Dict]: 所有node信息列表
        """
        # 先尝试trainer-runner，如果获取不到结果则尝试base-runner
        roles_to_try = ['trainer-runner', 'base-runner']
        
        for role in roles_to_try:
            print(f"尝试使用role: {role}")
            all_nodes = self._get_nodes_with_role(primus_url, state, role)
            
            if all_nodes:
                print(f"使用 {role} 成功获取到 {len(all_nodes)} 个节点")
                return all_nodes
            else:
                print(f"使用 {role} 未获取到任何节点")
        
        print("所有role都未获取到节点")
        return []
    
    def _get_nodes_with_role(self, primus_url: str, state: str, role: str) -> List[Dict]:
        """
        使用指定role获取节点信息
        
        Args:
            primus_url: 完整的primus URL
            state: 状态过滤条件
            role: 角色类型 (trainer-runner 或 base-runner)
            
        Returns:
            List[Dict]: 节点信息列表
        """
        all_nodes = []
        page = 1
        rows = 100
        
        while True:
            # 构建URL
            url = f"{primus_url.rstrip('/')}/newStatus.json"
            params = {
                'from': 'page',
                'rows': rows,
                'page': page,
                'role': role,
                'state': 'ALL' if not state else state.upper(),
                'filter': '',
                'sortOrder': 'asc'
            }
            
            try:
                if self.verbose:
                    print(f"[DEBUG] 请求URL: {url}")
                    print(f"[DEBUG] 请求参数: {params}")
                print(f"正在获取第 {page} 页数据 (role: {role})...")
                response = requests.get(url, params=params)
                if self.verbose:
                    print(f"[DEBUG] HTTP状态码: {response.status_code}")
                response.raise_for_status()
                data = response.json()
                if self.verbose:
                    print(f"[DEBUG] 返回数据keys: {list(data.keys())}")
                
                if not data.get('rows') or len(data['rows']) == 0:
                    print(f"第 {page} 页没有数据，停止获取")
                    break
                    
                all_nodes.extend(data['rows'])
                print(f"第 {page} 页获取到 {len(data['rows'])} 个节点")
                
                # 如果当前页的数据少于rows，说明是最后一页
                if len(data['rows']) < rows:
                    if self.verbose:
                        print(f"[DEBUG] 当前页数据少于每页数量，已到最后一页")
                    break
                    
                page += 1
                time.sleep(0.1)  # 避免请求过快
                
            except Exception as e:
                print(f"获取第 {page} 页数据失败: {e}")
                break
                
        print(f"使用 {role} 总共获取到 {len(all_nodes)} 个节点")
        return all_nodes
    
    def filter_nodes(self, nodes: List[Dict], 
                    state: Optional[str] = None,
                    node_id: Optional[str] = None,
                    attempt: Optional[Union[int, str]] = None,
                    launch_time_start: Optional[Union[int, str]] = None,
                    launch_time_end: Optional[Union[int, str]] = None,
                    release_time_start: Optional[Union[int, str]] = None,
                    release_time_end: Optional[Union[int, str]] = None) -> List[Dict]:
        """
        根据条件过滤节点
        
        Args:
            nodes: 节点列表
            state: 状态过滤
            node_id: 节点ID过滤
            attempt: 尝试次数过滤
            launch_time_start: 启动时间开始
            launch_time_end: 启动时间结束
            release_time_start: 释放时间开始
            release_time_end: 释放时间结束
            
        Returns:
            List[Dict]: 过滤后的节点列表
        """
        if self.verbose:
            print(f"[DEBUG] 开始过滤节点, 原始节点数: {len(nodes)}")
            print(f"[DEBUG] 过滤条件: state={state}, node_id={node_id}, attempt={attempt}, launch_time_start={launch_time_start}, launch_time_end={launch_time_end}, release_time_start={release_time_start}, release_time_end={release_time_end}")
        filtered_nodes = []
        
        for idx, node in enumerate(nodes):
            debug_skip = False
            # 状态过滤
            if state and node.get('state', '').upper() != state.upper():
                if self.verbose:
                    print(f"[DEBUG] 节点{idx} state不匹配: {node.get('state', '')} != {state}")
                continue
            
            # 从节点名称中提取node_id和attempt进行过滤
            node_name = node.get('node', '')
            if node_id is not None or attempt is not None:
                # 解析节点名称格式: executor_trainer-runner_${node_id}_${attempt}
                # 或者: executor_base-runner_${node_id}_${attempt}
                node_id_from_name = None
                attempt_from_name = None
                
                # 匹配格式: executor_xxx_${node_id}_${attempt}
                # 支持 trainer-runner 和 base-runner 等格式
                match = re.search(r'executor_[\w-]+_(\d+)_(\d+)', node_name)
                if match:
                    node_id_from_name = int(match.group(1))
                    attempt_from_name = int(match.group(2))
                
                # 节点ID过滤
                if node_id is not None:
                    try:
                        target_node_id = int(node_id)
                        if node_id_from_name is None or node_id_from_name != target_node_id:
                            if self.verbose:
                                print(f"[DEBUG] 节点{idx} node_id不匹配: {node_id_from_name} != {target_node_id} (节点名称: {node_name})")
                            continue
                    except (ValueError, TypeError) as e:
                        if self.verbose:
                            print(f"[DEBUG] 节点{idx} node_id转换失败: {e}")
                        continue
                
                # 尝试次数过滤
                if attempt is not None:
                    try:
                        target_attempt = int(attempt)
                        if attempt_from_name is None or attempt_from_name != target_attempt:
                            if self.verbose:
                                print(f"[DEBUG] 节点{idx} attempt不匹配: {attempt_from_name} != {target_attempt} (节点名称: {node_name})")
                            continue
                    except (ValueError, TypeError) as e:
                        if self.verbose:
                            print(f"[DEBUG] 节点{idx} attempt转换失败: {e}")
                        continue
                
            # 启动时间过滤
            if launch_time_start is not None:
                launch_time = node.get('launchTime')
                if launch_time is None or launch_time < self._parse_time(launch_time_start):
                    if self.verbose:
                        print(f"[DEBUG] 节点{idx} 启动时间不满足 >= {launch_time_start}, 实际: {launch_time}")
                    continue
                    
            if launch_time_end is not None:
                launch_time = node.get('launchTime')
                if launch_time is None or launch_time > self._parse_time(launch_time_end):
                    if self.verbose:
                        print(f"[DEBUG] 节点{idx} 启动时间不满足 <= {launch_time_end}, 实际: {launch_time}")
                    continue
                    
            # 释放时间过滤
            if release_time_start is not None:
                release_time = node.get('releaseTime')
                if release_time is None or release_time < self._parse_time(release_time_start):
                    if self.verbose:
                        print(f"[DEBUG] 节点{idx} 释放时间不满足 >= {release_time_start}, 实际: {release_time}")
                    continue
                    
            if release_time_end is not None:
                release_time = node.get('releaseTime')
                if release_time is None or release_time > self._parse_time(release_time_end):
                    if self.verbose:
                        print(f"[DEBUG] 节点{idx} 释放时间不满足 <= {release_time_end}, 实际: {release_time}")
                    continue
                    
            filtered_nodes.append(node)
            
        print(f"过滤后剩余 {len(filtered_nodes)} 个节点")
        return filtered_nodes
    
    def _parse_time(self, time_value: Union[int, str]) -> int:
        """
        解析时间值，支持时间戳和字符串格式
        
        Args:
            time_value: 时间值
            
        Returns:
            int: 毫秒时间戳
        """
        if self.verbose:
            print(f"[DEBUG] 解析时间: {time_value} (类型: {type(time_value)})")
        if isinstance(time_value, int):
            return time_value
            
        if isinstance(time_value, str):
            # 尝试解析为时间戳
            try:
                val = int(time_value)
                if self.verbose:
                    print(f"[DEBUG] 字符串直接转int成功: {val}")
                return val
            except ValueError:
                # 尝试解析为日期时间字符串
                try:
                    dt = datetime.fromisoformat(time_value.replace('Z', '+00:00'))
                    ts = int(dt.timestamp() * 1000)
                    if self.verbose:
                        print(f"[DEBUG] 字符串转datetime成功: {dt}, 毫秒时间戳: {ts}")
                    return ts
                except ValueError:
                    if self.verbose:
                        print(f"[DEBUG] 无法解析时间值: {time_value}")
                    raise ValueError(f"无法解析时间值: {time_value}")
                    
        if self.verbose:
            print(f"[DEBUG] 不支持的时间格式: {time_value}")
        raise ValueError(f"不支持的时间格式: {time_value}")


class FileDownloader:
    """文件下载器 - 负责下载节点的日志文件"""
    
    def __init__(self, url_prefix: str = "https://mljob-log-proxy.byted.org/yodel-logs/proxy", verbose: bool = False):
        self.url_prefix = url_prefix
        self.verbose = verbose
    

    
    def download_node_logs(self, node: Dict, 
                          primus_url: str,
                          include_pattern: Optional[str] = None,
                          exclude_pattern: Optional[str] = None,
                          output_dir: str = ".",
                          max_files: Optional[int] = None,
                          max_workers: int = 4) -> List[str]:
        """
        直接下载单个节点的日志文件
        
        Args:
            node: 节点信息
            primus_url: 完整的primus URL
            include_pattern: 包含的文件名正则表达式模式
            exclude_pattern: 排除的文件名正则表达式模式
            output_dir: 输出目录
            max_files: 最多下载的文件数量限制
            max_workers: 线程池最大线程数
            
        Returns:
            List[str]: 下载成功的文件列表
        """
        # 从primus_url中提取primus_name
        primus_name = UrlParser.extract_project_id_from_url(primus_url)
        if self.verbose:
            print(f"从URL提取的primus_name: {primus_name}")
        
        # 从node信息中提取必要信息
        node_name = node.get('node', '')
        pod_name = node.get('podName', '')
        if self.verbose:
            print(f"[DEBUG] 处理节点: node_name={node_name}, pod_name={pod_name}")
        
        # 提取IP地址
        ip_match = re.search(r'\[(g340-[^\]]+|n136-[^\]]+)\]', node_name)
        if not ip_match:
            print(f"警告: 无法从节点名称 {node_name} 提取IP地址")
            return []
            
        ip = ip_match.group(1)
        if self.verbose:
            print(f"[DEBUG] 提取到IP: {ip}")
        
        # 构建最终的日志URL（直接访问目标URL，避免重定向）
        # 从logUrl中提取必要参数
        log_url_params = node.get('logUrl', '')
        if log_url_params:
            # 解析logUrl中的参数
            import urllib.parse
            parsed_params = urllib.parse.parse_qs(urllib.parse.urlparse(log_url_params).query)
            
            # 提取必要参数
            hostip = parsed_params.get('hostip', [''])[0]
            pod_name = parsed_params.get('podName', [''])[0]
            username = parsed_params.get('username', [''])[0]
            
            # 构建最终URL
            final_log_url = f"{self.url_prefix}/{hostip}/{pod_name}/{username}?pod_name={pod_name}"
        else:
            # 如果没有logUrl，使用传统方式
            # 从primus_url中提取基础URL
            from urllib.parse import urlparse
            parsed_url = urlparse(primus_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            final_log_url = f"{base_url.rstrip('/')}/{node.get('logUrl', '')}"
        
        if self.verbose:
            print(f"[DEBUG] 构建最终日志URL: {final_log_url}")
        
        # 创建目录名 - 每个node单独一个文件夹，包含更多信息
        node_id = node.get('id', 'unknown')
        launch_time = node.get('launchTime')
        
        # 从node名称中提取executor信息
        node_executor = 'unknown'
        if node_name:
            # 提取node名称中方括号前的部分，例如："executor_trainer-runner_14_18 [...]" -> "executor_trainer-runner_14_18"
            executor_match = re.search(r'^([^\[]+)', node_name.strip())
            if executor_match:
                node_executor = executor_match.group(1).strip()
        
        # 格式化启动时间
        if launch_time:
            try:
                # 将毫秒时间戳转换为datetime
                dt = datetime.fromtimestamp(launch_time / 1000)
                time_str = dt.strftime('%Y%m%d_%H%M%S')
            except:
                time_str = str(launch_time)
        else:
            time_str = 'unknown_time'
        
        # 创建目录结构：output_dir/primus_name/node_executor_time_str/
        node_dir_name = f"{node_executor}_{time_str}"
        full_dir_name = os.path.join(output_dir, primus_name, node_dir_name)
        if self.verbose:
            print(f"[DEBUG] 完整目录名: {full_dir_name}")
            print(f"[DEBUG] 节点信息: id={node_id}, executor={node_executor}, launch_time={launch_time}")
        
        # 直接下载文件
        downloaded_files = self._download_files(
            final_log_url, ip, pod_name, full_dir_name, 
            include_pattern, exclude_pattern, primus_url, max_files, max_workers
        )
        
        return downloaded_files

    def list_node_files(self, node: Dict, primus_url: str) -> List[str]:
        """
        列出单个节点上的所有日志文件，不进行下载

        Args:
            node: 节点信息
            primus_url: 完整的primus URL

        Returns:
            List[str]: 文件名列表
        """
        primus_name = UrlParser.extract_project_id_from_url(primus_url)
        node_name = node.get('node', '')
        
        log_url_params = node.get('logUrl', '')
        if log_url_params:
            import urllib.parse
            parsed_params = urllib.parse.parse_qs(urllib.parse.urlparse(log_url_params).query)
            hostip = parsed_params.get('hostip', [''])[0]
            pod_name = parsed_params.get('podName', [''])[0]
            username = parsed_params.get('username', [''])[0]
            final_log_url = f"{self.url_prefix}/{hostip}/{pod_name}/{username}?pod_name={pod_name}"
        else:
            from urllib.parse import urlparse
            parsed_url = urlparse(primus_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            final_log_url = f"{base_url.rstrip('/')}/{node.get('logUrl', '')}"

        if self.verbose:
            print(f"[DEBUG] Listing files from URL: {final_log_url}")

        try:
            response = requests.get(final_log_url)
            response.raise_for_status()
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            a_tags = soup.find_all('a')
            
            files = []
            for a_tag in a_tags:
                href = a_tag.get('href', '')
                if not href:
                    continue
                
                if '/yodel-logs/proxy/' in href:
                    file_match = re.search(r'file=([^&]+)', href)
                    if file_match:
                        file_path = file_match.group(1)
                        filename = self._extract_filename_from_path(file_path)
                        files.append(filename)
                else:
                    # Fallback for older formats if necessary
                    filename = os.path.basename(href)
                    if filename and filename != '../':
                         files.append(filename)
            return files
        except Exception as e:
            print(f"Error listing files for node {node_name}: {e}")
            return []
    
    def _download_single_file(self, download_url: str, local_file_name: str) -> Optional[str]:
        """下载单个文件的辅助方法"""
        try:
            if self.verbose:
                print(f"[DEBUG] 下载文件: {os.path.basename(local_file_name)}")
                print(f"[DEBUG] 下载URL: {download_url}")
            
            file_response = requests.get(download_url, stream=True)
            file_response.raise_for_status()
            
            with open(local_file_name, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ 成功下载: {local_file_name}")
            return local_file_name
        except Exception as e:
            print(f"下载文件 {local_file_name} 失败: {e}")
            return None

    def _download_files(self, log_url: str, ip: str, pod_name: str, 
                       dir_name: str, include_pattern: Optional[str] = None,
                       exclude_pattern: Optional[str] = None, primus_url: str = "", 
                       max_files: Optional[int] = None, max_workers: int = 4) -> List[str]:
        """
        直接下载文件
        
        Args:
            log_url: 日志URL
            ip: IP地址
            pod_name: pod名称
            dir_name: 目录名
            include_pattern: 包含的文件名正则表达式模式
            exclude_pattern: 排除的文件名正则表达式模式
            primus_url: primus URL
            max_files: 最多下载的文件数量限制
            max_workers: 线程池最大线程数
            
        Returns:
            List[str]: 下载成功的文件列表
        """
        downloaded_files = []
        
        try:
            if self.verbose:
                print(f"[DEBUG] 请求日志页面: {log_url}")
            response = requests.get(log_url)
            if self.verbose:
                print(f"[DEBUG] 日志页面HTTP状态码: {response.status_code}")
            response.raise_for_status()
            html_content = response.text
            if self.verbose:
                print(f"[DEBUG] 日志页面内容长度: {len(html_content)}")
            
            # 解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            a_tags = soup.find_all('a')
            if self.verbose:
                print(f"[DEBUG] 解析到{len(a_tags)}个<a>标签")
            
            # 创建目录
            os.makedirs(dir_name, exist_ok=True)
            
            # 准备下载任务
            download_tasks = []
            
            for idx, a_tag in enumerate(a_tags):
                # 检查是否已达到最大文件数限制
                if max_files is not None and len(download_tasks) >= max_files:
                    if self.verbose:
                        print(f"[DEBUG] 已达到最大文件数限制 {max_files}，停止添加任务")
                    break
                    
                href = a_tag.get('href', '')
                if not href:
                    if self.verbose:
                        print(f"[DEBUG] 第{idx}个<a>标签没有href属性")
                    continue
                
                download_url = None
                filename = None
                
                # 处理新的HTML格式
                if '/yodel-logs/proxy/' in href:
                    file_match = re.search(r'file=([^&]+)', href)
                    if file_match:
                        file_path = file_match.group(1)
                        filename = self._extract_filename_from_path(file_path)
                    else:
                        path_match = re.search(r'/([^/]+)$', href.split('?')[0])
                        if path_match:
                            filename = path_match.group(1)
                            file_path = filename
                        else:
                            if self.verbose:
                                print(f"[DEBUG] 第{idx}个<a>标签无法提取文件名, href={href}")
                            continue
                    
                    escaped_path = file_path.replace("/", "%252F")
                    escaped_path = escaped_path.replace("%2F", "%252F")
                    path_param = file_path.replace("/", "%2F")
                    
                    download_url = f"{self.url_prefix}/{ip}/{pod_name}/default/{escaped_path}?action=download&file={path_param}&pod_name={pod_name}"
                else:
                    file_match = re.search(r'file=([^&]+)', href)
                    if not file_match:
                        if self.verbose:
                            print(f"[DEBUG] 第{idx}个<a>标签未找到file参数, href={href}")
                        continue
                        
                    file_path = file_match.group(1)
                    filename = self._extract_filename_from_path(file_path)
                    
                    escaped_path = file_path.replace("/", "%252F")
                    escaped_path = escaped_path.replace("%2F", "%252F")
                    path_param = file_path.replace("/", "%2F")
                    
                    download_url = f"{self.url_prefix}/{ip}/{pod_name}/default/{escaped_path}?action=download&file={path_param}&pod_name={pod_name}"
                
                # 文件类型过滤
                if not self._should_download_file(filename, include_pattern, exclude_pattern):
                    if self.verbose:
                        print(f"[DEBUG] 跳过文件: {filename} (不符合模式过滤)")
                    continue
                
                local_file_name = f"{dir_name}/{filename}"
                download_tasks.append((download_url, local_file_name))

            # 并行下载
            if download_tasks:
                print(f"开始下载 {len(download_tasks)} 个文件，线程池大小: {max_workers}")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(self._download_single_file, url, path): path 
                        for url, path in download_tasks
                    }
                    
                    for future in as_completed(future_to_file):
                        result = future.result()
                        if result:
                            downloaded_files.append(result)
            
        except Exception as e:
            print(f"处理节点 {dir_name} 失败: {e}")
            
        return downloaded_files
    
    def _should_download_file(self, filename: str, 
                            include_pattern: Optional[str] = None,
                            exclude_pattern: Optional[str] = None) -> bool:
        """
        判断是否应该下载该文件
        
        Args:
            filename: 文件名
            include_pattern: 包含的文件名正则表达式模式
            exclude_pattern: 排除的文件名正则表达式模式
            
        Returns:
            bool: 是否应该下载
        """
        if self.verbose:
            print(f"[DEBUG] 判断文件是否下载: {filename}, include_pattern={include_pattern}, exclude_pattern={exclude_pattern}")
        
        import re
        include_ok = True
        exclude_ok = True

        if include_pattern:
            try:
                include_ok = re.search(include_pattern, filename) is not None
                if self.verbose:
                    print(f"[DEBUG] include_pattern 判断: pattern={include_pattern}, filename={filename}, match={include_ok}")
            except re.error as e:
                print(f"[WARNING] 包含模式正则表达式错误: {e}")
                include_ok = False
        
        if exclude_pattern:
            try:
                excluded = re.search(exclude_pattern, filename) is not None
                exclude_ok = not excluded
                if self.verbose:
                    print(f"[DEBUG] exclude_pattern 判断: pattern={exclude_pattern}, filename={filename}, match={excluded}")
            except re.error as e:
                print(f"[WARNING] 排除模式正则表达式错误: {e}")
                # 排除模式错误时，保守起见不排除
                exclude_ok = True
        
        if self.verbose:
            print(f"[DEBUG] 综合判断: include_ok={include_ok}, exclude_ok={exclude_ok}, result={include_ok and exclude_ok}")
        return include_ok and exclude_ok
    
    def _extract_filename_from_path(self, file_path: str) -> str:
        """
        从文件路径中提取文件名，只保留最后一个%2F后面的部分
        
        Args:
            file_path: 文件路径，可能包含%2F编码的斜杠
            
        Returns:
            str: 提取的文件名
        """
        # 将%2F替换为/，然后取最后一个/后面的部分
        decoded_path = file_path.replace('%2F', '/')
        filename = os.path.basename(decoded_path)
        if self.verbose:
            print(f"[DEBUG] 文件路径: {file_path} -> 提取的文件名: {filename}")
        return filename


class PodLogDownloader:
    """Pod日志下载工具 - 组合节点获取和文件下载功能"""
    
    def __init__(self, url_prefix: str = "https://mljob-log-proxy.byted.org/yodel-logs/proxy", verbose: bool = False):
        self.node_fetcher = NodeInfoFetcher(verbose)
        self.file_downloader = FileDownloader(url_prefix, verbose)
    
    def get_all_nodes_info(self, primus_url: str, state: str = "") -> List[Dict]:
        """获取pod下所有node的信息"""
        return self.node_fetcher.get_all_nodes_info(primus_url, state)
    
    def filter_nodes(self, nodes: List[Dict], **kwargs) -> List[Dict]:
        """过滤节点"""
        return self.node_fetcher.filter_nodes(nodes, **kwargs)
    
    def download_node_logs(self, node: Dict, primus_url: str, **kwargs) -> List[str]:
        """下载单个节点的日志文件"""
        return self.file_downloader.download_node_logs(node, primus_url, **kwargs)
    
    def download_all_filtered_logs(self, primus_url: str, 
                                   state: Optional[str] = None,
                                   node_id: Optional[str] = None,
                                   attempt: Optional[Union[int, str]] = None,
                                   launch_time_start: Optional[Union[int, str]] = None,
                                   launch_time_end: Optional[Union[int, str]] = None,
                                   release_time_start: Optional[Union[int, str]] = None,
                                   release_time_end: Optional[Union[int, str]] = None,
                                   include_pattern: Optional[str] = None,
                                   exclude_pattern: Optional[str] = None,
                                   output_dir: str = ".",
                                   max_files: Optional[int] = None,
                                   max_workers: int = 4) -> List[str]:
        """
        完整的工作流程: 获取节点 -> 过滤 -> 下载
        
        Args:
            primus_url: 完整的primus URL
            state: 状态过滤
            node_id: 节点ID过滤
            attempt: 尝试次数过滤
            launch_time_start: 启动时间开始
            launch_time_end: 启动时间结束
            release_time_start: 释放时间开始
            release_time_end: 释放时间结束
            include_pattern: 包含的文件名正则表达式模式
            exclude_pattern: 排除的文件名正则表达式模式
            output_dir: 输出目录
            max_files: 最多下载的文件数量限制
            max_workers: 线程池最大线程数
            
        Returns:
            List[str]: 所有下载成功的文件列表
        """
        # 1. 获取所有节点信息
        print(f"正在获取primus URL {primus_url} 的所有节点信息...")
        nodes = self.get_all_nodes_info(primus_url, state if state else "")
        
        if not nodes:
            print("没有找到任何节点")
            return []
        
        # 2. 过滤节点
        print("正在根据条件过滤节点...")
        filtered_nodes = self.filter_nodes(
            nodes,
            state=state,
            node_id=node_id,
            attempt=attempt,
            launch_time_start=launch_time_start,
            launch_time_end=launch_time_end,
            release_time_start=release_time_start,
            release_time_end=release_time_end
        )
        
        if not filtered_nodes:
            print("过滤后没有符合条件的节点")
            return []
        
        # 3. 为每个节点下载文件
        print("正在为每个节点下载文件...")
        all_downloaded_files = []
        
        # 使用线程池并发处理多个节点
        print(f"开始处理 {len(filtered_nodes)} 个节点，线程池大小: {max_workers}")
        
        # 定义单个节点处理函数
        def process_node(node_data):
            idx, node = node_data
            print(f"处理节点 {idx}/{len(filtered_nodes)}: {node.get('node', 'unknown')}")
            if self.node_fetcher.verbose:
                print(f"[DEBUG] 节点详细信息: {json.dumps(node, ensure_ascii=False)}")
            return self.download_node_logs(
                node, 
                primus_url,
                include_pattern=include_pattern,
                exclude_pattern=exclude_pattern,
                output_dir=output_dir,
                max_files=max_files,
                max_workers=max_workers
            )

        # 准备任务数据
        node_tasks = list(enumerate(filtered_nodes, 1))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_node = {
                executor.submit(process_node, task): task 
                for task in node_tasks
            }
            
            for future in as_completed(future_to_node):
                try:
                    downloaded_files = future.result()
                    all_downloaded_files.extend(downloaded_files)
                except Exception as e:
                    print(f"处理节点失败: {e}")
        
        return all_downloaded_files


def write_primus_url_file(output_dir: str, primus_url: str) -> None:
    """在输出目录写入 primus_url.txt，记录来源URL"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        url_file_path = os.path.join(output_dir, 'primus_url.txt')
        with open(url_file_path, 'w', encoding='utf-8') as f:
            f.write(f"{primus_url}\n")
    except Exception as e:
        print(f"[WARNING] 写入 primus_url.txt 失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='Pod日志下载工具')
    parser.add_argument('primus_url', help='完整的primus URL，例如：https://primus-ui-cn.byted.org/gaura-yx/nj-ed62daba-2db1-4b59-aaa6-cf939-pj/webapps/primus/')
    parser.add_argument('--mode', choices=['nodes', 'download', 'both'], default='both',
                       help='运行模式: nodes(只获取节点信息), download(只下载), both(获取+下载)')
    parser.add_argument('--nodes-file', help='节点信息JSON文件路径 (download模式时使用)')
    parser.add_argument('--state', default='', help='节点状态过滤 (running, killed, released)')
    parser.add_argument('--node_id', help='节点ID过滤')
    parser.add_argument('--attempt', help='尝试次数过滤')
    parser.add_argument('--launch_time_start', help='启动时间开始')
    parser.add_argument('--launch_time_end', help='启动时间结束')
    parser.add_argument('--release_time_start', help='释放时间开始')
    parser.add_argument('--release_time_end', help='释放时间结束')
    parser.add_argument('--include_pattern', help='包含的文件名正则表达式模式')
    parser.add_argument('--exclude_pattern', help='排除的文件名正则表达式模式')
    parser.add_argument('--output-dir', default='.', help='输出目录 (默认当前目录)')
    parser.add_argument('--max-files', type=int, help='最多下载的文件数量限制')
    parser.add_argument('--max-workers', type=int, default=16, help='线程池最大线程数 (默认4)')
    parser.add_argument('--verbose', action='store_true', help='启用详细日志输出')
    
    args = parser.parse_args()
    
    # 验证参数 - 允许同时使用包含和排除模式
    
    if args.mode == 'download' and not args.nodes_file:
        print("错误: download模式需要指定 --nodes-file 参数")
        return
    
    if args.verbose:
        print(f"[DEBUG] 命令行参数: {args}")
    
    # 在程序开始时记录 primus_url 来源（仅调用一次）
    write_primus_url_file(args.output_dir, args.primus_url)
    
    if args.mode == 'nodes':
        # 只获取节点信息
        fetcher = NodeInfoFetcher(args.verbose)
        nodes = fetcher.get_all_nodes_info(args.primus_url, args.state)
        
        if nodes:
            # 过滤节点
            filtered_nodes = fetcher.filter_nodes(
                nodes,
                state=args.state if args.state else None,
                node_id=args.node_id,
                attempt=args.attempt,
                launch_time_start=args.launch_time_start,
                launch_time_end=args.launch_time_end,
                release_time_start=args.release_time_start,
                release_time_end=args.release_time_end
            )
            
            if filtered_nodes:
                print(f"总共找到 {len(filtered_nodes)} 个符合条件的节点:")
                downloader = FileDownloader(verbose=args.verbose)
                for node in filtered_nodes:
                    print(json.dumps(node, indent=4))
                    files = downloader.list_node_files(node, args.primus_url)
                    if files:
                        print("  可下载的文件:")
                        for f in files:
                            print(f"    - {f}")
                    else:
                        print("  没有找到可下载的文件或无法访问日志。")
            else:
                print("没有找到符合条件的节点")
        
        if nodes:
            # 过滤节点
            filtered_nodes = fetcher.filter_nodes(
                nodes,
                state=args.state if args.state else None,
                node_id=args.node_id,
                attempt=args.attempt,
                launch_time_start=args.launch_time_start,
                launch_time_end=args.launch_time_end,
                release_time_start=args.release_time_start,
                release_time_end=args.release_time_end
            )
            
            # 保存节点信息到文件
            # 从URL中提取pod名称作为文件名
            pod_name = UrlParser.extract_project_id_from_url(args.primus_url)
            output_file = os.path.join(args.output_dir, f"nodes_{pod_name}.json")
            
            # 确保输出目录存在
            os.makedirs(args.output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_nodes, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 节点信息已保存到: {output_file}")
            print(f"共找到 {len(filtered_nodes)} 个符合条件的节点")
        else:
            print("没有找到任何节点")
    
    elif args.mode == 'download':
        # 只下载文件
        if not os.path.exists(args.nodes_file):
            print(f"错误: 节点文件 {args.nodes_file} 不存在")
            return
        
        # 加载节点信息
        with open(args.nodes_file, 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        
        downloader = FileDownloader(verbose=args.verbose)
        all_downloaded_files = []
        
        for i, node in enumerate(nodes, 1):
            print(f"处理节点 {i}/{len(nodes)}: {node.get('node', 'unknown')}")
            downloaded_files = downloader.download_node_logs(
                node, 
                args.primus_url,
                include_pattern=args.include_pattern,
                exclude_pattern=args.exclude_pattern,
                output_dir=args.output_dir,
                max_files=args.max_files,
                max_workers=args.max_workers
            )
            all_downloaded_files.extend(downloaded_files)
        
        print(f"\n✅ 完成! 共下载 {len(all_downloaded_files)} 个文件:")
        for file_path in all_downloaded_files:
            print(f"  - {file_path}")
    
    else:  # both
        # 获取节点信息并下载文件
        downloader = PodLogDownloader(verbose=args.verbose)
        all_downloaded_files = downloader.download_all_filtered_logs(
            args.primus_url,
            state=args.state if args.state else None,
            node_id=args.node_id,
            attempt=args.attempt,
            launch_time_start=args.launch_time_start,
            launch_time_end=args.launch_time_end,
            release_time_start=args.release_time_start,
            release_time_end=args.release_time_end,
            include_pattern=args.include_pattern,
            exclude_pattern=args.exclude_pattern,
            output_dir=args.output_dir,
            max_files=args.max_files,
            max_workers=args.max_workers
        )
        
        print(f"\n✅ 完成! 共下载 {len(all_downloaded_files)} 个文件:")
        for file_path in all_downloaded_files:
            print(f"  - {file_path}")

if __name__ == "__main__":
    main()

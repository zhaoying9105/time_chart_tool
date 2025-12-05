# Time Chart Tool

一个用于解析 PyTorch Profiler 时间图表 JSON 数据的强大分析工具，支持多种分析模式和深度性能对比。

## 工具概述

Time Chart Tool 是一个专门为 PyTorch Profiler 设计的性能分析工具，能够：

- **解析 PyTorch Profiler JSON 数据**：从 PyTorch Profiler 导出的时间图表数据中提取关键信息
- **多维度聚合分析**：按调用栈、操作名、形状、数据类型等维度进行聚合分析
- **通信性能分析**：专门针对分布式训练中的通信操作进行性能分析
- **快慢卡对比分析**：识别性能瓶颈，对比快卡和慢卡之间的差异
- **可视化输出**：生成 Excel 和 JSON 格式的分析报告

## 安装

### 从源码安装

在项目根目录执行：
```bash
pip3 install -e .
```

### 依赖要求

- Python >= 3.7
- pandas >= 1.3.0
- openpyxl >= 3.0.0
- matplotlib == 3.0.0

## 使用方法

### 1. 通过模块调用（推荐）
```bash
python3 -m time_chart_tool --help
python3 -m time_chart_tool compare --help
python3 -m time_chart_tool comm --help
python3 -m time_chart_tool analysis --help
```

### 2. 通过控制台脚本调用
```bash
time-chart-tool --help
time-chart-tool compare --help
time-chart-tool comm --help
time-chart-tool analysis --help
```

## 核心功能

### 1. 对比分析 (compare)

对比多个 JSON 文件的性能数据，支持多种聚合和比较维度。

#### 示例 1：对比多个文件
```bash
time-chart-tool compare file1.json:label1 file2.json:label2  \
  --aggregation 'name,shape,fwd_bwd_type' \
  --show 'dtype,call_stack,kernel-names,timestamp' \
  --compare 'dtype'
```

#### 参数说明
- `--aggregation`: 聚合字段组合，支持：`call_stack`, `name`, `shape`, `dtype`, `fwd_bwd_type`
- `--show`: 显示额外信息，支持：`dtype`, `shape`, `kernel-names`, `kernel-duration`, `timestamp` 等
- `--compare`: 比较选项，支持：`dtype`, `shape`, `name`, `kernel_name`

#### 支持的输入格式
- 单文件：`file.json:label`
- 多文件：`"file1.json,file2.json,file3.json":label`
- 目录：`dir/:label`（自动查找所有 `*.json` 文件）
- 通配符：`"dir/*.json":label`

### 2. 通信性能分析 (comm)

专门分析分布式训练中的通信性能，支持概览分析和深度分析两种模式。

#### 示例 2：通信性能分析
```bash
# 分析所有 step 的通信性能
python3 -m time_chart_tool comm --output-dir ./compiled_step_bf16_all2al '/data03/zhaoying/engine/compile_bf16/nj-d8f89fba-c1a8-4b60-8227-b30f5-pj/' &> log_comm_compiled_douguang_bf16 &

# 分析特定 step 的通信性能
python3 -m time_chart_tool comm --output-dir ./compiled_step_10040_bf16_all2all_deep --step 10040 '/data03/zhaoying/engine/compile_bf16/nj-d8f89fba-c1a8-4b60-8227-b30f5-pj/' &> log_comm_compiled_douguang_bf16_10040 &

#### 参数说明
- `pod_dir`: Pod文件夹路径，包含 `executor_trainer-runner_*_*_*` 格式的文件夹
- `--step`: 指定要分析的 step，如果不指定则分析所有 step
- `--comm-idx`: 指定要分析的通信操作索引，如果不指定则分析所有通信操作
- `--fastest-card-idx`: 指定最快卡的索引，用于深度分析
- `--slowest-card-idx`: 指定最慢卡的索引，用于深度分析
- `--kernel-prefix`: 要检测的通信 kernel 前缀
- `--output-dir`: 输出目录

#### 支持的通信 Kernel 前缀
- `TCDP_ONESHOT_ALLREDUCELL_SIMPLE`
- `TCDP_RING_ALLGATHER_SIMPLE`
- `TCDP_RING_ALLREDUCELL_SIMPLE`
- `TCDP_RING_ALLREDUCE_SIMPLE`
- `TCDP_RING_REDUCESCATTER_SIMPLE`
- `TCDP_TCDPALLCONNECTED_PXMMIXALLTOALLV_ALLTOALL`

### 3. 单文件分析 (analysis)

分析单个或多个 JSON 文件的性能数据。

```bash
time-chart-tool analysis file.json --label "baseline" --aggregation "call_stack,name" --show "dtype,shape,kernel-names,kernel-duration,timestamp"
```

## 工具原理

### 架构设计

Time Chart Tool 采用模块化设计，主要包含以下核心模块：

#### 1. 解析器模块 (parser.py)
- 解析 PyTorch Profiler JSON 数据
- 提取 CPU 操作和 GPU kernel 事件
- 构建调用栈树结构

#### 2. 分析器模块 (analyzer/)
- 执行数据聚合和统计分析
- 支持多种聚合维度
- 生成性能指标

#### 3. 通信分析模块 (comm/)
- 专门处理分布式通信性能分析
- 实现快慢卡对比算法
- 支持深度性能分析

#### 4. 可视化模块 (visualization/)
- 生成对齐矩阵图
- 创建时间线可视化
- 输出统计图表

### 核心算法

#### 1. 调用栈构建算法
- 基于扫描线和线段树的算法生成调用栈
- 支持从 JSON args 字段提取调用栈
- 可选粗糙调用栈模式（去除模块名后缀）

#### 2. 事件对齐算法 (LCS)
- 使用最长公共子序列算法进行事件对齐
- 支持模糊匹配和容错处理
- 生成对齐矩阵用于可视化

#### 3. 通信性能分析算法
- 自动识别通信 kernel 操作
- 计算快慢卡之间的性能差异
- 分析性能瓶颈的根本原因

### 数据处理流程

1. **数据加载**：解析 JSON 文件，提取事件数据
2. **预处理**：过滤、排序、标准化事件数据
3. **聚合分析**：按指定维度聚合数据
4. **对比分析**：比较不同文件或不同卡之间的差异
5. **结果输出**：生成 Excel 和 JSON 格式的报告

## 输出文件

### 对比分析输出
- `multiple_files_comparison.json` - JSON 格式的分析结果
- `multiple_files_comparison.xlsx` - Excel 格式的分析结果

### 通信分析输出
#### 概览分析
- `comm_raw_data.xlsx` - 原始数据（step -> card -> comm_idx -> duration）
- `comm_statistics.xlsx` - 统计分析（最快/最慢 card、统计信息等）

#### 深度分析
- `comm_deep_analysis_step_{step}_idx_{idx}.xlsx` - 深度分析结果
  - `详细对比` 工作表：快慢卡之间的详细对比数据
  - `汇总信息` 工作表：分析汇总信息

## 高级功能

### 1. 调用栈来源选择
- `--call-stack-source args`：从 JSON 的 args 字段中获取调用栈
- `--call-stack-source tree`：使用基于扫描线和线段树的算法生成调用栈

### 2. 粗糙调用栈
- `--coarse-call-stack`：去除模块名后缀（如 `Dense_10` -> `Dense`）

### 3. 操作过滤
- `--include-op`：包含的操作名称模式
- `--exclude-op`：排除的操作名称模式
- `--include-kernel`：包含的 kernel 名称模式
- `--exclude-kernel`：排除的 kernel 名称模式

### 4. 采样控制
- `--max-files`：最多使用的文件数量
- `--max-files-per-label`：每个标签最多使用的文件数量
- `--random-seed`：随机采样的种子

## 使用场景

### 1. 模型性能优化
- 识别计算瓶颈
- 分析内存使用模式
- 优化 kernel 执行效率

### 2. 分布式训练调试
- 分析通信性能瓶颈
- 识别同步问题
- 优化通信模式

### 3. 框架对比
- 比较不同框架的性能差异
- 分析优化策略的效果
- 验证性能改进

## 故障排除

### 常见问题

1. **找不到 executor 文件夹**：检查 pod_dir 路径和 attempt-idx 参数
2. **找不到 JSON 文件**：检查文件命名格式和路径
3. **没有通信数据**：检查 JSON 文件中是否包含目标通信操作
4. **深度分析失败**：确保指定的 step 和 comm_idx 存在数据，且至少有 2 个 card 的数据

### 调试技巧

1. **使用详细日志**：重定向输出到日志文件进行分析
2. **逐步调试**：先使用概览分析，再针对特定问题进行深度分析
3. **参数验证**：确保所有参数格式正确，特别是文件路径和标签格式
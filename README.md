# Time Chart Tool

一个用于解析 PyTorch Profiler 时间图表 JSON 数据的强大分析工具，支持多种分析模式和深度性能对比。

## 工具概述

Time Chart Tool 是一个专门为 PyTorch Profiler 设计的性能分析工具，能够：

- **解析 PyTorch Profiler JSON 数据**：从 PyTorch Profiler 导出的时间图表数据中提取关键信息
- **多维度聚合分析**：按调用栈、操作名、形状、数据类型、OP序号等维度进行聚合分析
- **通信性能分析**：专门针对分布式训练中的通信操作进行性能分析
- **多文件对比分析**：支持单文件、多文件、目录等多种方式的性能数据对比
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

## 使用方法

推荐通过模块调用方式使用：

```bash
# 查看帮助
python3 -m time_chart_tool --help

# 各子命令帮助
python3 -m time_chart_tool analysis --help
python3 -m time_chart_tool compare --help
python3 -m time_chart_tool comm --help
```

也可以通过控制台脚本调用：`time-chart-tool [command] ...`

## 核心功能

### 1. 单文件/多文件分析 (analysis)

分析单个或多个 JSON 文件的性能数据，支持灵活的聚合和展示选项。

**基础用法：**
```bash
python3 -m time_chart_tool analysis [file_path] --label [label] [options]
```

**示例：**

```bash
# 1. 按操作名聚合，展示详细信息
# 相同op name放在一行，展示op的type, call_stack, kernel-names, pid, tid, kernel-duration
python3 -m time_chart_tool analysis file.json --label 'baseline' \
    --aggregation 'name' \
    --show 'dtype,call_stack,kernel-names,pid,tid,kernel-duration'

# 2. 多维度聚合 (操作名 + 调用栈 + 数据类型)
# 相同name, call_stack, dtype放在一行
python3 -m time_chart_tool analysis file.json --label 'baseline' \
    --aggregation 'name,call_stack,dtype' \
    --show 'kernel-names,pid,tid,kernel-duration'

# 3. 按 OP 下发顺序展示 (op_index)
# 注意：使用 op_index 聚合时不能与其他字段组合
python3 -m time_chart_tool analysis file.json --label 'baseline' \
    --aggregation 'op_index' \
    --show 'name,call_stack,dtype,shape,kernel-names,pid,tid'
```

**常用参数：**
- `--aggregation`: 聚合字段组合。支持: `call_stack`, `name`, `shape`, `dtype`, `fwd_bwd_type`, `pid`, `tid`, `op_index` (需单独使用)。
- `--show`: 显示额外信息。支持: `dtype`, `shape`, `kernel-names`, `kernel-duration`, `timestamp`, `pid`, `tid`, `stream` 等。
- `--include-op` / `--exclude-op`: 包含/排除特定操作名称（支持正则）。
- `--include-kernel` / `--exclude-kernel`: 包含/排除特定 kernel 名称（支持正则）。

### 2. 对比分析 (compare)

对比多个 JSON 文件的性能数据，支持多种聚合和比较维度。

**基础用法：**
```bash
python3 -m time_chart_tool compare [files...] [options]
```

**文件输入格式：**
- 单文件: `file.json:label`
- 多文件: `"file1.json,file2.json":label`
- 目录: `dir/:label` (自动查找所有 *.json)
- 通配符: `"dir/*.json":label`

**示例：**

```bash
# 1. 基础对比：按操作名和调用栈聚合，对比数据类型和形状
python3 -m time_chart_tool compare \
    'data/rank_0.json:baseline' \
    'data/rank_1.json:test' \
    --aggregation 'name,call_stack' \
    --show 'call_stack,kernel-names,pid,tid' \
    --compare "dtype,shape"

# 2. 混合输入对比：单文件 vs 目录
python3 -m time_chart_tool compare \
    single_file.json:baseline \
    "dir/*.json":test \
    --aggregation name

# 3. 控制采样数量对比
python3 -m time_chart_tool compare \
    "dir1/*.json":baseline \
    "dir2/*.json":optimized \
    --max-files-per-label 10 \
    --random-seed 42 \
    --aggregation name
```

**常用参数：**
- `--compare`: 指定要比较的字段，如 `dtype`, `shape`, `name`, `kernel_name`。
- `--max-files-per-label`: 限制每个标签参与对比的文件数量，确保公平性。
- `--coarse-call-stack`: 使用粗糙调用栈（去除模块名后缀，如 Dense_10 -> Dense）。

### 3. 通信性能分析 (comm)

专门分析分布式训练中的通信性能。

**基础用法：**
```bash
python3 -m time_chart_tool comm [pod_dir] [options]
```

**示例：**

```bash
# 分析指定 step 的通信性能，指定 kernel 前缀
python3 -m time_chart_tool comm 'data/pod_folder/' \
    --kernel-prefix TCDP_ \
    --step 0 \
    --output-dir ./comm_out
```

**常用参数：**
- `pod_dir`: Pod文件夹路径，通常包含 `executor_trainer-runner_*` 格式的文件夹。
- `--step`: 指定要分析的 step，不指定则分析所有 step。
- `--kernel-prefix`: 通信 kernel 的前缀 (默认: TCDP_)。
- `--output-dir`: 输出目录。

## 输出文件说明

- **Analysis/Compare**: 默认生成 `output.json` 和 `output.xlsx` (或根据输入文件名自动命名)。
- **Comm**: 
  - `comm_raw_data.xlsx`: 原始通信数据。
  - `comm_statistics.xlsx`: 通信统计分析报告。

## 高级特性

### 过滤功能
所有分析命令均支持强大的过滤功能：
- `--include-op "Conv.*"`: 只分析匹配正则表达式的操作。
- `--exclude-kernel "Nccl.*"`: 排除匹配正则表达式的 kernel。

### 粗糙调用栈
使用 `--coarse-call-stack` 参数可以将 `Layer_1`, `Layer_2` 等统一视为 `Layer`，便于在模型结构发生微小变化（如层数改变）时进行对比。

### OP 序号聚合 (op_index)
`analysis` 命令支持 `--aggregation op_index`，这将按照 OP 下发的顺序（Trace 中的顺序）进行展示，非常适合查看执行流。注意此模式下不能与其他聚合字段组合。

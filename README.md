# Time Chart Tool

一个用于解析和分析 PyTorch profiler 时间图表 JSON 数据的工具库。

## 功能特性

- 解析 PyTorch profiler 生成的 JSON 文件
- 分析 cpu_op 和 kernel 的映射关系
- 支持单个文件分析和多文件对比分析
- **CPU Op性能统计摘要**：基于kernel耗时统计每种cpu_op的性能数据
- **专门分析matmul算子 (aten::mm)，按最小维度分组统计**
- ~~基于python_function call stack的比较分析~~ (已废弃)
- 生成 JSON、XLSX 格式的分析报告和性能图表
- 命令行工具支持，便于批量处理
- **终端markdown表格输出**：支持在终端以markdown表格形式显示分析结果

## 安装

### 开发模式安装（推荐用于实验）

```bash
pip3 install -e .
```

### 正式安装

```bash
pip3 install .
```

## 使用方法

### 命令行工具

安装后，您可以使用以下命令：

#### 分析单个文件

```bash
# 基本用法
python3 -m time_chart_tool.cli single file.json --label "baseline"

# 指定输出格式
python3 -m time_chart_tool.cli single file.json --label "baseline" --output-format json,xlsx

# 只输出 JSON 格式
python3 -m time_chart_tool.cli single file.json --label "baseline" --output-format json

# 只输出 XLSX 格式
python3 -m time_chart_tool.cli single file.json --label "baseline" --output-format xlsx

# 指定输出目录
python3 -m time_chart_tool.cli single file.json --label "baseline" --output-dir ./results
```

**单个文件分析功能说明：**
- **CPU Op和Kernel映射分析**：分析每个cpu_op对应的kernel事件，建立映射关系
- **CPU Op性能统计摘要**：基于kernel耗时统计每种cpu_op的性能数据，包括：
  - 调用次数和总耗时
  - 平均、最小、最大kernel耗时
  - 占总kernel耗时的比例
  - 在终端以markdown表格形式显示结果
- **输出文件**：生成带label前缀的分析文件（如`fp32_single_file_analysis.xlsx`）

#### 分析多个文件并对比

```bash
# 基本用法
python3 -m time_chart_tool.cli compare file1.json:label1 file2.json:label2

# 指定输出格式
python3 -m time_chart_tool.cli compare file1.json:fp32 file2.json:tf32 --output-format json,xlsx

# 只输出 JSON 格式
python3 -m time_chart_tool.cli compare file1.json:baseline file2.json:optimized --output-format json

# 只输出 XLSX 格式
python3 -m time_chart_tool.cli compare file1.json:fp32 file2.json:tf32 --output-format xlsx

# 指定输出目录
python3 -m time_chart_tool.cli compare file1.json:fp32 file2.json:tf32 --output-dir ./comparison_results
```

#### 专门分析matmul算子

```bash
# 基本用法 - 专门分析matmul算子并按最小维度分组
python3 -m time_chart_tool.cli matmul file1.json:fp32 file2.json:bf16

# 指定输出格式
python3 -m time_chart_tool.cli matmul file1.json:fp32 file2.json:bf16 --output-format json,xlsx

# 指定输出目录
python3 -m time_chart_tool.cli matmul file1.json:fp32 file2.json:bf16 --output-dir ./matmul_results
```

**Matmul分析功能说明：**
- 自动提取所有 `aten::mm` 算子
- 解析输入维度格式 `((m, k), (k, n))` 并计算最小维度 `min_dim = min(m, k, n)`
- 按 `min_dim` 分组统计性能数据
- **数据完整性检查**：只对比在所有time chart中都存在的shape数据
- 生成性能对比折线图，横轴为 `min_dim`，纵轴为性能比率
- 支持任意数量的标签（如 fp32, bf16, tf32 等）

**Call Stack分析功能说明：**
- ~~构建python_function调用树结构~~ (已废弃)
- ~~通过时间范围匹配cpu_op和python_function~~ (已废弃)
- ~~生成python_function调用栈，过滤无意义的函数名~~ (已废弃)
- ~~基于调用栈进行性能对比分析~~ (已废弃)
- ~~支持多个time chart文件的比较~~ (已废弃)

### 编程接口

```python
from time_chart_tool import Analyzer

# 创建分析器
analyzer = Analyzer()

# 分析单个文件
data = analyzer.parser.load_json_file("file.json")
mapping = analyzer.analyze_cpu_op_kernel_mapping(data)

# 生成 Excel 报告
analyzer.generate_excel_from_mapping(mapping, "analysis.xlsx")

# 分析多个文件
file_labels = [
    ("file1.json", "fp32"),
    ("file2.json", "tf32")
]
analyzer.run_complete_analysis(file_labels, output_dir="./results")

# 专门分析matmul算子
analyzer.run_complete_analysis_with_matmul(file_labels, output_dir="./matmul_results")

# 基于call stack的比较分析 (已废弃)
# analyzer.analyze_by_call_stack(file_labels, output_dir="./callstack_results")
```

## 输出文件说明

### 单个文件分析

- `{label}_single_file_analysis.json`: 包含完整的映射关系数据
- `{label}_single_file_analysis.xlsx`: Excel 格式的分析报告
- `{label}_cpu_op_performance_summary.json`: CPU Op性能统计摘要数据
- `{label}_cpu_op_performance_summary.xlsx`: CPU Op性能统计摘要Excel报告

### 多文件对比分析

- `comparison_analysis.json`: 包含所有文件的对比数据
- `comparison_analysis.xlsx`: Excel 格式的对比报告

### Matmul专门分析

- `matmul_analysis.json`: matmul算子的专门分析数据，按最小维度分组
- `matmul_analysis.xlsx`: matmul算子的Excel分析报告
- `matmul_performance_chart.jpg`: matmul性能对比折线图

### Call Stack比较分析 (已废弃)

- ~~`call_stack_comparison_analysis.json`: 基于call stack的比较分析数据~~ (已废弃)
- ~~`call_stack_comparison_analysis.xlsx`: 基于call stack的Excel比较报告~~ (已废弃)

## 数据格式

### JSON 输出格式

单个文件分析的 JSON 格式：
```json
{
  "cpu_op_name": {
    "input_strides": {
      "input_dims": {
        "input_type": [
          {
            "name": "kernel_name",
            "cat": "kernel",
            "ph": "X",
            "ts": 1234567890,
            "dur": 1000,
            "tid": 1,
            "pid": 1,
            "args": {...},
            "external_id": 123
          }
        ]
      }
    }
  }
}
```

对比分析的 JSON 格式：
```json
[
  {
    "cpu_op_name": "op_name",
    "cpu_op_input_strides": "[1, 2, 3]",
    "cpu_op_input_dims": "[4, 5, 6]",
    "fp32_input_types": "('float', 'float')",
    "fp32_kernel_names": "kernel1||kernel2",
    "fp32_kernel_count": 100,
    "fp32_kernel_mean_duration": 5.0,
    "tf32_input_types": "('float', 'float')",
    "tf32_kernel_names": "kernel1||kernel2",
    "tf32_kernel_count": 100,
    "tf32_kernel_mean_duration": 4.8,
    "kernel_names_equal": true,
    "kernel_count_equal": true,
    "tf32_ratio_to_fp32": 0.96
  }
]
```

Matmul专门分析的 JSON 格式：
```json
[
  {
    "mm_min_dim": 3,
    "fp32_input_types": "('float', 'float')",
    "fp32_kernel_count": 15,
    "fp32_kernel_mean_duration": 2.9066666666666667,
    "op_bf16_input_types": "('c10::BFloat16', 'c10::BFloat16')",
    "op_bf16_kernel_count": 15,
    "op_bf16_kernel_mean_duration": 2.5893333333333333,
    "op_bf16_ratio_to_fp32": 0.8908256880733945
  }
]
```

### Excel 输出格式

#### 单个文件分析
Excel 文件包含以下列：
- cpu_op_name: CPU 操作名称
- cpu_op_input_strides: 输入步长
- cpu_op_input_dims: 输入维度
- cpu_op_input_type: 输入类型
- kernel_name: Kernel 名称
- kernel_count: Kernel 执行次数
- kernel_min_duration: 最小执行时间
- kernel_max_duration: 最大执行时间
- kernel_mean_duration: 平均执行时间
- kernel_std_duration: 执行时间标准差

#### 多文件对比分析
Excel 文件包含以下列：
- cpu_op_name: CPU 操作名称（共享列）
- cpu_op_input_strides: 输入步长（共享列）
- cpu_op_input_dims: 输入维度（共享列）
- {label}_input_types: 各文件的输入类型（用||连接多个值）
- {label}_kernel_names: 各文件的kernel名称（用||连接多个值）
- {label}_kernel_count: 各文件的kernel执行次数
- {label}_kernel_mean_duration: 各文件的平均执行时间
- kernel_names_equal: kernel名称是否相同（不考虑顺序）
- kernel_count_equal: kernel执行次数是否相同
- {label}_ratio_to_{base_label}: 相对于基准文件的性能比值

#### Matmul专门分析
Excel 文件包含以下列：
- mm_min_dim: matmul算子的最小维度 (min(m, k, n))
- {label}_input_types: 各文件的输入类型
- {label}_kernel_count: 各文件的kernel执行次数
- {label}_kernel_mean_duration: 各文件的平均执行时间
- {label}_ratio_to_{base_label}: 相对于基准文件的性能比值

#### Call Stack比较分析 (已废弃)
~~Excel 文件包含以下列：~~
- ~~cpu_op_name: CPU操作名称~~ (已废弃)
- ~~cpu_op_input_strides: 输入步长~~ (已废弃)
- ~~cpu_op_input_dims: 输入维度~~ (已废弃)
- ~~call_stack: Python函数调用栈（用 -> 连接）~~ (已废弃)
- ~~{label}_input_types: 各文件的输入类型~~ (已废弃)
- ~~{label}_kernel_names: 各文件的kernel名称~~ (已废弃)
- ~~{label}_kernel_count: 各文件的kernel执行次数~~ (已废弃)
- ~~{label}_kernel_mean_duration: 各文件的平均执行时间~~ (已废弃)
- ~~kernel_names_equal: kernel名称是否相同~~ (已废弃)
- ~~kernel_count_equal: kernel执行次数是否相同~~ (已废弃)
- ~~{label}_ratio_to_{base_label}: 相对于基准文件的性能比值~~ (已废弃)

## 功能特性详解

### CPU Op性能统计摘要功能

CPU Op性能统计摘要功能基于kernel耗时统计每种cpu_op的性能数据，提供详细的性能分析报告。

#### 功能特点
- **基于kernel耗时统计**：统计每个cpu_op对应的所有kernel事件的耗时
- **多维度性能指标**：包括调用次数、总耗时、平均耗时、最小/最大耗时、占比等
- **终端markdown表格输出**：在终端以美观的markdown表格形式显示结果
- **带label前缀的文件输出**：生成带label前缀的JSON和Excel文件

#### 输出示例

**终端markdown表格输出：**
```
## CPU Op性能统计摘要表格

| CPU Op名称 | 调用次数 | 总耗时(μs) | 平均耗时(μs) | 最小耗时(μs) | 最大耗时(μs) | Kernel数量 | 占比(%) |
|------------|----------|------------|--------------|--------------|--------------|------------|---------|
| aten::add | 150 | 1250.50 | 8.34 | 5.20 | 15.80 | 150 | 25.5 |
| aten::mm | 80 | 980.30 | 12.25 | 8.10 | 20.50 | 80 | 20.0 |
| **TOTAL** | **230** | **2230.80** | **9.70** | **0.00** | **0.00** | **230** | **100.0** |
```

**JSON输出格式：**
```json
[
  {
    "cpu_op_name": "aten::add",
    "call_count": 150,
    "total_kernel_duration": 1250.5,
    "avg_kernel_duration": 8.34,
    "min_kernel_duration": 5.2,
    "max_kernel_duration": 15.8,
    "kernel_count": 150,
    "percentage_of_total": 25.5
  },
  {
    "cpu_op_name": "TOTAL",
    "call_count": 230,
    "total_kernel_duration": 2230.8,
    "avg_kernel_duration": 9.7,
    "min_kernel_duration": 0.0,
    "max_kernel_duration": 0.0,
    "kernel_count": 230,
    "percentage_of_total": 100.0
  }
]
```

#### 使用方法

```bash
# 单个文件分析（自动生成CPU Op性能统计摘要）
python3 -m time_chart_tool.cli single file.json --label "fp32"

# 编程接口使用
from time_chart_tool import Analyzer

analyzer = Analyzer()
data = analyzer.parser.load_json_file("file.json")
analyzer.generate_cpu_op_performance_summary(data, ".", "fp32")
```

### 多文件对比分析增强功能

1. **保留Input Type信息**: 在对比分析中保留并展示Input Type信息
2. **行级展示**: 将同一个(cpu_op_name, input_strides, input_dims)下的信息展示成一行
3. **多值连接**: 使用||连接多个Input Type和kernel_name
4. **标签前缀**: 所有列名都有label前缀（如fp32_, tf32_）
5. **比较列**: 
   - kernel_names_equal: 比较不同time chart的kernel_names是否相同（不考虑顺序）
   - kernel_count_equal: 比较不同time chart的kernel_count是否相同
6. **性能比较**: 只计算mean duration的变化

### Matmul算子专门分析功能

1. **自动识别**: 自动提取所有 `aten::mm` 算子
2. **维度解析**: 解析输入维度格式 `((m, k), (k, n))` 并验证 k1 == k2
3. **最小维度计算**: 计算 `min_dim = min(m, k, n)` 作为分组依据
4. **性能统计**: 按 `min_dim` 分组统计各标签的性能数据
5. **数据完整性检查**: 只对比在所有time chart中都存在的shape数据
6. **可视化图表**: 生成性能对比折线图，横轴为 `min_dim`，纵轴为性能比率
7. **灵活标签**: 支持任意数量的标签（如 fp32, bf16, tf32 等）

### Call Stack比较分析功能 (已废弃)

~~1. **调用树构建**: 基于Python parent id和Python id构建python_function调用树~~ (已废弃)
~~2. **时间范围匹配**: 通过时间范围匹配cpu_op和python_function~~ (已废弃)
~~3. **调用栈生成**: 生成完整的python_function调用栈~~ (已废弃)
~~4. **函数名过滤**: 自动过滤无意义的函数名（如`<built-in method`、`torch/nn`等）~~ (已废弃)
~~5. **性能对比**: 基于调用栈进行多文件性能对比分析~~ (已废弃)
~~6. **聚合分析**: 按(call_stack + cpu_op信息)进行聚合比较~~ (已废弃)

## 使用示例

### 示例1：单个文件分析

```bash
# 分析fp32模型文件
python3 -m time_chart_tool.cli single fp32_model.json --label "fp32" --output-dir ./results
```

**输出结果：**
```
=== 单个文件分析 ===
文件: fp32_model.json
标签: fp32
输出格式: json,xlsx
输出目录: ./results

正在加载文件: fp32_model.json
文件加载完成，耗时: 2.45 秒

正在分析 cpu_op 和 kernel 的映射关系...
找到 15 个 cpu_op 的映射关系

正在生成cpu_op性能统计摘要...
=== 开始生成cpu_op性能统计摘要（基于kernel耗时） ===

## CPU Op性能统计摘要表格

| CPU Op名称 | 调用次数 | 总耗时(μs) | 平均耗时(μs) | 最小耗时(μs) | 最大耗时(μs) | Kernel数量 | 占比(%) |
|------------|----------|------------|--------------|--------------|--------------|------------|---------|
| aten::mm | 120 | 8500.50 | 70.84 | 45.20 | 120.80 | 120 | 45.2 |
| aten::add | 200 | 4200.30 | 21.00 | 12.10 | 35.50 | 200 | 22.3 |
| aten::relu | 150 | 2800.20 | 18.67 | 8.50 | 28.90 | 150 | 14.9 |
| **TOTAL** | **470** | **18800.00** | **40.00** | **0.00** | **0.00** | **470** | **100.0** |

CPU Op性能统计Excel文件已生成: ./results/fp32_cpu_op_performance_summary.xlsx
CPU Op性能统计JSON文件已生成: ./results/fp32_cpu_op_performance_summary.json

前5个最耗时的cpu_op（基于kernel耗时）:
  1. aten::mm
     kernel调用次数: 120
     总kernel耗时: 8500.50 微秒
     平均kernel耗时: 70.84 微秒
     占总kernel耗时比例: 45.20%

分析完成，总耗时: 3.12 秒
```

### 示例2：多文件对比分析

```bash
# 对比fp32和bf16模型
python3 -m time_chart_tool.cli compare fp32.json:fp32 bf16.json:bf16 --output-dir ./comparison
```

**输出结果：**
```
=== 多文件对比分析 ===
输出格式: json,xlsx
输出目录: ./comparison

  fp32: fp32.json
  bf16: bf16.json

将分析 2 个文件

开始分析...
正在分析文件: fp32.json (标签: fp32)
正在分析文件: bf16.json (标签: bf16)
正在生成对比分析报告...

分析完成，总耗时: 8.45 秒

生成的文件:
  ./comparison/comparison_analysis.json
  ./comparison/comparison_analysis.xlsx
```

### 示例3：Matmul专门分析

```bash
# 专门分析matmul算子性能
python3 -m time_chart_tool.cli matmul fp32.json:fp32 bf16.json:bf16 tf32.json:tf32 --output-dir ./matmul_analysis
```

**输出结果：**
```
=== Matmul算子专门分析 ===
输出格式: json,xlsx
输出目录: ./matmul_analysis

  fp32: fp32.json
  bf16: bf16.json
  tf32: tf32.json

将分析 3 个文件

开始分析...
找到 25 个不同的matmul shape组合
正在生成性能对比图表...

分析完成，总耗时: 5.67 秒

生成的文件:
  ./matmul_analysis/matmul_analysis.json
  ./matmul_analysis/matmul_analysis.xlsx
  ./matmul_analysis/matmul_performance_chart.jpg
```

## 依赖要求

- Python >= 3.7
- pandas >= 1.3.0
- openpyxl >= 3.0.0
- matplotlib >= 3.5.0

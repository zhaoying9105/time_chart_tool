# Time Chart Tool

一个用于解析和分析 PyTorch profiler 时间图表 JSON 数据的工具库。

## 功能特性

- 解析 PyTorch profiler 生成的 JSON 文件
- 分析 cpu_op 和 kernel 的映射关系
- 支持单个文件分析和多文件对比分析
- 生成 JSON 和 XLSX 格式的分析报告
- 命令行工具支持，便于批量处理

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
```

## 输出文件说明

### 单个文件分析

- `{label}_single_file_analysis.json`: 包含完整的映射关系数据
- `{label}_single_file_analysis.xlsx`: Excel 格式的分析报告

### 多文件对比分析

- `comparison_analysis.json`: 包含所有文件的对比数据
- `comparison_analysis.xlsx`: Excel 格式的对比报告

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

## 功能5.4特性

### 多文件对比分析增强功能

1. **保留Input Type信息**: 在对比分析中保留并展示Input Type信息
2. **行级展示**: 将同一个(cpu_op_name, input_strides, input_dims)下的信息展示成一行
3. **多值连接**: 使用||连接多个Input Type和kernel_name
4. **标签前缀**: 所有列名都有label前缀（如fp32_, tf32_）
5. **比较列**: 
   - kernel_names_equal: 比较不同time chart的kernel_names是否相同（不考虑顺序）
   - kernel_count_equal: 比较不同time chart的kernel_count是否相同
6. **性能比较**: 只计算mean duration的变化

## 依赖要求

- Python >= 3.7
- pandas >= 1.3.0
- openpyxl >= 3.0.0

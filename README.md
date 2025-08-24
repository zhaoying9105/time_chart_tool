# PyTorch Profiler Parser Tool

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

安装后，您可以使用 `torch-profiler-parser` 命令：

#### 分析单个文件

```bash
# 基本用法
torch-profiler-parser single file.json --label "baseline"

# 指定输出格式
torch-profiler-parser single file.json --label "baseline" --output-format json,xlsx

# 只输出 JSON 格式
torch-profiler-parser single file.json --label "baseline" --output-format json

# 只输出 XLSX 格式
torch-profiler-parser single file.json --label "baseline" --output-format xlsx

# 指定输出目录
torch-profiler-parser single file.json --label "baseline" --output-dir ./results
```

#### 分析多个文件并对比

```bash
# 基本用法
torch-profiler-parser compare file1.json:label1 file2.json:label2

# 指定输出格式
torch-profiler-parser compare file1.json:fp32 file2.json:tf32 --output-format json,xlsx

# 只输出 JSON 格式
torch-profiler-parser compare file1.json:baseline file2.json:optimized --output-format json

# 只输出 XLSX 格式
torch-profiler-parser compare file1.json:fp32 file2.json:tf32 --output-format xlsx

# 指定输出目录
torch-profiler-parser compare file1.json:fp32 file2.json:tf32 --output-dir ./comparison_results
```

### 编程接口

```python
from torch_profiler_parser_tool import Analyzer

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
    "kernel_name": "kernel_name",
    "file_label": "fp32",
    "kernel_count": 100,
    "kernel_min_duration": 1.0,
    "kernel_max_duration": 10.0,
    "kernel_mean_duration": 5.0,
    "kernel_std_duration": 2.0
  }
]
```

### Excel 输出格式

Excel 文件包含以下列：
- cpu_op_name: CPU 操作名称
- cpu_op_input_strides: 输入步长
- cpu_op_input_dims: 输入维度
- kernel_name: Kernel 名称
- kernel_count: Kernel 执行次数
- kernel_min_duration: 最小执行时间
- kernel_max_duration: 最大执行时间
- kernel_mean_duration: 平均执行时间
- kernel_std_duration: 执行时间标准差

对于对比分析，还会包含：
- file_label: 文件标签
- comparison_type: 比较类型（matched/unmatched）
- 各文件的统计数据和比值

## 依赖要求

- Python >= 3.7
- pandas >= 1.3.0
- openpyxl >= 3.0.0

## 开发

### 运行测试

```bash
pytest tests/
```

### 代码格式化

```bash
black .
flake8 .
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！ 

# Matmul算子专门分析功能

## 功能概述

新增了专门针对matmul算子（`aten::mm`）的分析功能，能够：

1. **自动识别matmul算子**：从比较分析数据中提取所有 `aten::mm` 算子
2. **维度解析**：解析输入维度格式 `((m, k), (k, n))` 并验证 k1 == k2
3. **最小维度计算**：计算 `min_dim = min(m, k, n)` 作为分组依据
4. **性能统计**：按 `min_dim` 分组统计各标签的性能数据
5. **可视化图表**：生成性能对比折线图，横轴为 `min_dim`，纵轴为性能比率
6. **灵活标签**：支持任意数量的标签（如 fp32, bf16, tf32 等）

## 实现的功能

### 1. 维度提取功能

```python
def extract_matmul_dimensions(self, input_dims_str: str) -> Optional[Tuple[int, int, int]]:
    """
    从matmul算子的input_dims中提取m, k, n维度
    
    Args:
        input_dims_str: 格式如 "((2048, 3), (3, 32))" 的字符串
        
    Returns:
        Optional[Tuple[m, k, n]]: 提取的维度，如果解析失败返回None
    """
```

**功能特点：**
- 使用正则表达式解析维度格式
- 验证 k1 == k2（矩阵乘法的要求）
- 返回 (m, k, n) 元组或 None

### 2. Matmul专门分析

```python
def analyze_matmul_by_min_dim(self, comparison_data: List[Dict]) -> Dict[int, List]:
    """
    专门分析matmul算子，按最小维度分组
    
    Args:
        comparison_data: 比较分析的数据列表
        
    Returns:
        Dict[int, List]: 按min_dim分组的matmul数据
    """
```

**功能特点：**
- 过滤出所有 `aten::mm` 算子
- 提取维度信息并计算最小维度
- 按最小维度分组统计性能数据

### 3. 分析结果生成

```python
def generate_matmul_analysis(self, comparison_data: List[Dict], output_dir: str = ".") -> None:
    """
    生成matmul算子的专门分析
    
    Args:
        comparison_data: 比较分析的数据列表
        output_dir: 输出目录
    """
```

**生成的文件：**
- `matmul_analysis.json`: matmul算子的专门分析数据
- `matmul_analysis.xlsx`: matmul算子的Excel分析报告
- `matmul_performance_chart.jpg`: matmul性能对比折线图

### 4. 性能图表生成

```python
def generate_matmul_chart(self, json_data: List[Dict], labels: List[str], output_dir: str) -> None:
    """
    生成matmul算子的折线图
    
    Args:
        json_data: matmul分析数据
        labels: 标签列表
        output_dir: 输出目录
    """
```

**图表特点：**
- 横轴：matmul最小维度 (min_dim)
- 纵轴：性能比率（相对于基准标签）
- 多条线：每个非基准标签一条线
- 自动计算每个min_dim的平均比率

## 使用方法

### 命令行使用

```bash
# 基本用法
python3 -m time_chart_tool.cli matmul file1.json:fp32 file2.json:bf16

# 指定输出格式
python3 -m time_chart_tool.cli matmul file1.json:fp32 file2.json:bf16 --output-format json,xlsx

# 指定输出目录
python3 -m time_chart_tool.cli matmul file1.json:fp32 file2.json:bf16 --output-dir ./matmul_results
```

### 编程接口使用

```python
from time_chart_tool import Analyzer

# 创建分析器
analyzer = Analyzer()

# 分析多个文件（包含matmul专门分析）
file_labels = [
    ("file1.json", "fp32"),
    ("file2.json", "bf16")
]
analyzer.run_complete_analysis_with_matmul(file_labels, output_dir="./matmul_results")

# 或者单独使用matmul分析功能
comparison_data = [...]  # 从比较分析中获取的数据
analyzer.generate_matmul_analysis(comparison_data, output_dir="./matmul_results")
```

## 输出数据格式

### JSON格式

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

### Excel格式

Excel文件包含以下列：
- `mm_min_dim`: matmul算子的最小维度 (min(m, k, n))
- `{label}_input_types`: 各文件的输入类型
- `{label}_kernel_count`: 各文件的kernel执行次数
- `{label}_kernel_mean_duration`: 各文件的平均执行时间
- `{label}_ratio_to_{base_label}`: 相对于基准文件的性能比值

## 测试验证

创建了完整的测试套件 `tests/test_matmul_analysis.py`，包含：

1. **维度提取测试**：验证正则表达式解析功能
2. **数据分析测试**：验证matmul数据分组功能
3. **文件生成测试**：验证JSON、Excel、图表生成功能
4. **图表生成测试**：验证性能图表生成功能

运行测试：
```bash
python3 -m pytest tests/test_matmul_analysis.py -v
```

## 演示脚本

创建了演示脚本 `demo_matmul_analysis.py`，展示：

1. 如何从matmul算子的输入维度中提取m, k, n
2. 如何计算最小维度 min_dim = min(m, k, n)
3. 如何按min_dim分组统计性能数据
4. 如何生成JSON、Excel和性能图表

运行演示：
```bash
python3 demo_matmul_analysis.py
```

## 依赖更新

新增了matplotlib依赖用于生成图表：
```
matplotlib>=3.5.0
```

## 文档更新

更新了以下文档：
- `README.md`：添加了matmul专门分析功能的说明
- `example_matmul_analysis.py`：创建了使用示例脚本
- CLI帮助信息：添加了matmul命令的说明

## 总结

这个新功能为time chart tool增加了专门针对matmul算子的分析能力，能够：

1. **自动化分析**：自动识别和解析matmul算子
2. **维度分组**：按最小维度分组，便于性能分析
3. **可视化展示**：生成直观的性能对比图表
4. **灵活使用**：支持命令行和编程接口两种使用方式
5. **完整测试**：包含完整的测试和演示

这个功能特别适用于分析不同精度（fp32、bf16、tf32等）下matmul算子的性能表现，帮助开发者优化模型性能。

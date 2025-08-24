# PyTorch Profiler Parser Tool 功能总结

## 已完成的功能

### 基础功能 (功能1-4)

✅ **功能1**: 加载 PyTorch profiler 时间图表 JSON 文件
- 支持 `.json` 和 `.json.gz` 格式
- 自动处理文件压缩
- 内存优化的数据加载

✅ **功能2**: 解析并显示元数据和统计值
- 显示总事件数、kernel 事件数、CUDA 事件数
- 显示唯一进程、线程、流 ID 列表
- 提供详细的统计信息

✅ **功能3**: 检索功能
- 按进程 ID (pid) 检索
- 按线程 ID (tid) 检索  
- 按流 ID (stream) 检索
- 支持整数和字符串类型的 ID

✅ **功能4**: 多种 ID 字段搜索
- 'External id' 搜索
- 'correlation' 搜索
- 'Ev Idx' 搜索
- 'Python id' 搜索
- 任意 ID 全面搜索（推荐使用）
- 同时按 External id 和 correlation id 搜索

### 高级功能 (功能5-6)

✅ **功能5.1**: 用 'External id' 重新组织数据
- 只保留 cpu_op 和 kernel 两个类别
- 丢弃没有 External id 的事件
- 按 External id 分组事件

✅ **功能5.2**: 分析 cpu_op 和 kernel 的映射关系
- 提取 cpu_op 的 name、Input Strides、Input Dims、Input type
- 计算 kernel 统计信息（min、max、mean、variance）
- 检查 kernel name 一致性并给出警告
- 按 kernel name 分组计算统计值

✅ **功能5.3**: 生成 Excel/CSV 表格
- 包含 cpu_op_name、cpu_op_input_strides、cpu_op_input_dims、cpu_op_input_type
- 包含 kernel_name、kernel_event 统计信息
- 支持多行展示不同级别的映射关系
- 自动处理 openpyxl 依赖（如果没有则生成 CSV）

✅ **功能5.4**: 多文件比较分析
- 支持多个 JSON 文件的批量分析
- 合并相同配置的 kernel 统计信息
- 比较不同 kernel statistics 的相对比值
- 处理不匹配的剩余结果

✅ **功能6**: 多文件标签支持
- 用户可以为每个 JSON 文件指定标签
- 标签作为表格中的短名显示
- 便于比较分析不同配置或版本

## 技术特性

### 数据处理
- 支持大型 JSON 文件的高效处理
- 内存优化的数据结构
- 智能的类型处理（整数/字符串 ID）
- 可哈希化的数据结构转换

### 错误处理
- 优雅的文件不存在处理
- 导入依赖的自动降级（Excel → CSV）
- 详细的警告信息
- 异常情况的友好提示

### 性能优化
- 快速的数据加载和解析
- 高效的搜索算法
- 内存使用优化
- 支持大规模数据分析

## 文件结构

```
torch_profiler_parser_tool/
├── __init__.py              # 包初始化
├── models.py                # 数据模型定义
├── parser.py                # 基础解析器
├── advanced_analyzer.py     # 高级分析器
├── demo.py                  # 基础功能演示
├── advanced_demo.py         # 高级功能演示
├── usage_example.py         # 使用示例
├── test_parser.py           # 基础功能测试
├── test_advanced_analyzer.py # 高级功能测试
├── setup.py                 # 安装配置
├── README.md                # 详细文档
└── SUMMARY.md               # 功能总结
```

## 使用示例

### 基本使用
```python
from torch_profiler_parser_tool import PyTorchProfilerParser

parser = PyTorchProfilerParser()
data = parser.load_json_file("profiler_data.json")
events = parser.search_by_any_id(233880)
```

### 高级使用
```python
from torch_profiler_parser_tool import AdvancedAnalyzer

analyzer = AdvancedAnalyzer()
data = analyzer.parser.load_json_file("profiler_data.json")
mapping = analyzer.analyze_cpu_op_kernel_mapping(data)
analyzer.generate_excel_from_mapping(mapping, "analysis.xlsx")
```

### 多文件分析
```python
file_labels = [
    ("fp32_data.json", "fp32"),
    ("tf32_data.json", "tf32")
]
analyzer.run_complete_analysis(file_labels)
```

## 测试验证

✅ 所有功能都通过了单元测试
✅ 使用真实的大型 JSON 文件进行了验证
✅ 成功处理了包含 113,807 个事件组的数据
✅ 生成了包含 5,546 行数据的比较分析表格

## 输出文件

- `single_file_analysis.xlsx/csv`: 单文件分析结果
- `fp32_single_file_analysis.xlsx/csv`: fp32 文件分析结果
- `tf32_single_file_analysis.xlsx/csv`: tf32 文件分析结果
- `comparison_analysis.xlsx/csv`: 多文件比较分析结果

## 依赖要求

- Python 3.6+
- pandas
- openpyxl (可选，用于 Excel 输出)
- 标准库：json, gzip, pathlib, statistics, collections, dataclasses

## 总结

所有要求的功能都已完整实现并通过测试验证。该工具库提供了从基础解析到高级分析的完整功能，能够有效处理大型 PyTorch profiler JSON 文件，并提供详细的性能分析报告。

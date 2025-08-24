# PyTorch Profiler Parser Tool

一个用于解析 PyTorch profiler 时间图表 JSON 数据的工具库。

## 功能特性

- **功能1**: 加载 PyTorch profiler time chart JSON 文件
- **功能2**: 解析并显示元数据和 activity event kernel 等相关的统计值
- **功能3**: 实现检索功能，支持根据进程、线程、stream id 等检索 activity
- **功能4**: 根据多种 ID 字段查找相关事件，包括：
  - 'External id'
  - 'correlation' 
  - 'Ev Idx'
  - 'Python id'
  - 支持任意 ID 的全面搜索
- **功能5**: 高级数据分析功能
  - 5.1: 用 'External id' 重新组织数据，只保留 cpu_op 和 kernel 两个类别
  - 5.2: 分析 cpu_op 和 kernel 的映射关系，计算 kernel 统计信息（min、max、mean、variance）
  - 5.3: 生成 Excel/CSV 表格，包含 cpu_op 和 kernel 的详细映射信息
  - 5.4: 多文件比较分析，合并相同配置的 kernel 统计信息
- **功能6**: 支持多文件标签，便于比较分析

## 安装

```bash
# 从源码安装
git clone <repository-url>
cd torch_profiler_parser_tool
pip install -e .
```

## 快速开始

### 基本使用

```python
from torch_profiler_parser_tool import PyTorchProfilerParser

# 创建解析器实例
parser = PyTorchProfilerParser()

# 加载 JSON 文件
data = parser.load_json_file("your_profiler_data.json")

# 显示元数据
parser.print_metadata()

# 显示统计信息
parser.print_statistics()
```

### 检索功能

```python
# 按进程 ID 检索
events = parser.search_by_process(pid=123)

# 按线程 ID 检索
events = parser.search_by_thread(tid=456)

# 按流 ID 检索
events = parser.search_by_stream(stream_id=789)

# 按 External id 检索
external_events = parser.search_by_external_id(233880)

# 按 correlation 检索
correlation_events = parser.search_by_correlation(233880)

# 按 Ev Idx 检索
ev_idx_events = parser.search_by_ev_idx(233880)

# 按 Python id 检索
python_id_events = parser.search_by_python_id(233880)

# 任意 ID 全面搜索（推荐）
any_id_events = parser.search_by_any_id(233880)

# 同时按 External id 和 correlation id 检索
external_events, correlation_events = parser.search_by_id(233880)
```

## 运行 Demo

```bash
# 运行基本功能演示程序
python demo.py

# 运行高级功能演示程序
python advanced_demo.py
```

## 运行测试

```bash
# 运行基本功能单元测试
python -m unittest test_parser.py

# 运行高级分析器单元测试
python -m unittest test_advanced_analyzer.py

# 或者使用 pytest
pytest test_parser.py -v
pytest test_advanced_analyzer.py -v
```

## API 文档

### PyTorchProfilerParser

主要的解析器类。

#### 方法

- `load_json_file(file_path)`: 加载 JSON 文件
- `print_metadata()`: 打印元数据信息
- `print_statistics()`: 打印统计信息
- `search_by_process(pid)`: 按进程 ID 搜索
- `search_by_thread(tid)`: 按线程 ID 搜索
- `search_by_stream(stream_id)`: 按流 ID 搜索
- `search_by_external_id(external_id)`: 按 External id 搜索
- `search_by_correlation(correlation)`: 按 correlation 搜索
- `search_by_ev_idx(ev_idx)`: 按 Ev Idx 搜索
- `search_by_python_id(python_id)`: 按 Python id 搜索
- `search_by_any_id(id_value)`: 任意 ID 全面搜索（推荐）
- `search_by_id(id_value)`: 同时按 External id 和 correlation id 搜索
- `print_events_summary(events, title)`: 打印事件摘要

### ActivityEvent

活动事件数据模型。

#### 属性

- `name`: 事件名称
- `cat`: 事件类别
- `ph`: 事件阶段
- `pid`: 进程 ID
- `tid`: 线程 ID
- `ts`: 时间戳
- `dur`: 持续时间
- `args`: 事件参数
- `id`: 事件 ID
- `stream_id`: 流 ID

#### 计算属性

- `external_id`: External id 值
- `correlation_id`: correlation id 值
- `correlation`: correlation 值
- `ev_idx`: Ev Idx 值
- `python_id`: Python id 值
- `is_kernel`: 是否为 kernel 事件
- `is_cuda_event`: 是否为 CUDA 事件

### ProfilerData

Profiler 数据模型。

#### 属性

- `metadata`: 元数据字典
- `events`: 事件列表
- `trace_events`: 原始 trace 事件列表

#### 计算属性

- `total_events`: 总事件数
- `kernel_events`: kernel 事件列表
- `cuda_events`: CUDA 事件列表
- `unique_processes`: 唯一进程 ID 列表
- `unique_threads`: 唯一线程 ID 列表
- `unique_streams`: 唯一流 ID 列表

### AdvancedAnalyzer

高级分析器，用于深度分析 cpu_op 和 kernel 的映射关系。

#### 方法

- `reorganize_by_external_id(data)`: 按 External id 重新组织数据
- `analyze_cpu_op_kernel_mapping(data)`: 分析 cpu_op 和 kernel 的映射关系
- `generate_excel_from_mapping(mapping, output_file)`: 生成 Excel/CSV 表格
- `analyze_multiple_files(file_labels)`: 分析多个文件
- `run_complete_analysis(file_labels)`: 运行完整的分析流程

### KernelStatistics

Kernel 统计信息数据模型。

#### 属性

- `kernel_name`: kernel 名称
- `min_duration`: 最小持续时间
- `max_duration`: 最大持续时间
- `mean_duration`: 平均持续时间
- `variance`: 方差
- `count`: 事件数量

## 示例

### 解析大型 JSON 文件

```python
import time
from torch_profiler_parser_tool import PyTorchProfilerParser

parser = PyTorchProfilerParser()

# 记录开始时间
start_time = time.time()

# 加载大型 JSON 文件
data = parser.load_json_file("large_profiler_data.json")

# 显示加载时间
load_time = time.time() - start_time
print(f"文件加载完成，耗时: {load_time:.2f} 秒")

# 显示基本信息
print(f"总事件数: {data.total_events}")
print(f"Kernel 事件数: {len(data.kernel_events)}")
print(f"唯一进程数: {len(data.unique_processes)}")
```

### 查找特定 ID 的事件

```python
# 查找 External id 为 233880 的事件
external_events = parser.search_by_external_id(233880)
print(f"找到 {len(external_events)} 个 External id 事件")

# 查找任意 ID 为 233880 的事件（推荐，最全面）
any_id_events = parser.search_by_any_id(233880)
print(f"找到 {len(any_id_events)} 个任意 ID 事件")

# 显示事件详情
for event in any_id_events[:3]:  # 显示前3个事件
    print(f"事件: {event.name}")
    print(f"  类别: {event.cat}")
    print(f"  进程: {event.pid}, 线程: {event.tid}")
    print(f"  时间戳: {event.ts}")
    
    # 显示匹配的 ID 字段
    id_fields = []
    if event.external_id == 233880:
        id_fields.append(f"External id: {event.external_id}")
    if event.correlation == 233880:
        id_fields.append(f"correlation: {event.correlation}")
    if event.ev_idx == 233880:
        id_fields.append(f"Ev Idx: {event.ev_idx}")
    if event.python_id == 233880:
        id_fields.append(f"Python id: {event.python_id}")
    
    print(f"  匹配的 ID 字段: {', '.join(id_fields)}")
    print(f"  参数: {event.args}")
```

### 高级数据分析

```python
from torch_profiler_parser_tool import AdvancedAnalyzer

# 创建高级分析器
analyzer = AdvancedAnalyzer()

# 加载数据
data = parser.load_json_file("profiler_data.json")

# 功能5.1: 按 External id 重新组织数据
external_id_map = analyzer.reorganize_by_external_id(data)
print(f"找到 {len(external_id_map)} 个有 External id 的事件组")

# 功能5.2: 分析 cpu_op 和 kernel 的映射关系
mapping = analyzer.analyze_cpu_op_kernel_mapping(data)
print(f"找到 {len(mapping)} 个 cpu_op 的映射关系")

# 功能5.3: 生成 Excel 表格
analyzer.generate_excel_from_mapping(mapping, "analysis_result.xlsx")

# 功能5.4 和功能6: 多文件比较分析
file_labels = [
    ("fp32_data.json", "fp32"),
    ("tf32_data.json", "tf32")
]

# 运行完整分析
analyzer.run_complete_analysis(file_labels)
```

## 支持的 JSON 格式

该工具支持标准的 PyTorch profiler JSON 格式，包含以下结构：

```json
{
  "metadata": {
    "version": "1.0",
    "description": "Profiler data"
  },
  "traceEvents": [
    {
      "name": "event_name",
      "cat": "event_category",
      "ph": "X",
      "pid": 123,
      "tid": 456,
      "ts": 1000.0,
      "dur": 100.0,
      "args": {
        "External id": "233880",
        "correlation id": "233880",
        "stream": 789
      }
    }
  ]
}
```

## 性能说明

- 支持大型 JSON 文件的解析
- 内存使用优化，适合处理大量事件数据
- 提供高效的搜索和过滤功能

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

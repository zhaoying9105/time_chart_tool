# 重命名总结

## 重命名概述

已将 `torch_profiler_parser_tool` 全面重命名为 `time_chart_tool`。

## 重命名内容

### 1. 目录结构
- 旧目录：`torch_profiler_parser_tool/`
- 新目录：`time_chart_tool/`

### 2. 包名
- 旧包名：`torch_profiler_parser_tool`
- 新包名：`time_chart_tool`

### 3. 安装包名
- 旧安装名：`torch-profiler-parser`
- 新安装名：`time-chart-tool`

### 4. 命令行工具
- 旧命令：`torch-profiler-parser`
- 新命令：`time-chart-tool`

## 修改的文件

### 核心配置文件
1. **setup.py**
   - 包名：`torch-profiler-parser` → `time-chart-tool`
   - 入口点：`torch-profiler-parser` → `time-chart-tool`

2. **src/ 目录结构**
   - 旧：`src/torch_profiler_parser_tool/`
   - 新：`src/time_chart_tool/`

### 代码文件
1. **__init__.py**
   - 文档字符串更新

2. **cli.py**
   - 命令行描述更新
   - 示例命令更新

3. **README.md**
   - 标题更新
   - 所有命令示例更新
   - 导入示例更新

4. **example_usage.py**
   - 导入语句更新
   - 命令行示例更新

### 测试文件
1. **tests/test_analyzer.py**
   - 导入语句更新

2. **tests/test_parser.py**
   - 导入语句更新

3. **tests/test_real_id.py**
   - 导入语句更新

4. **tests/test_specific_id.py**
   - 导入语句更新

5. **tests/test_enhanced_search.py**
   - 导入语句更新

## 使用方法

### 安装
```bash
# 开发模式安装
pip3 install -e .

# 从 whl 文件安装
pip3 install dist/time_chart_tool-1.0.0-py3-none-any.whl
```

### 命令行使用
```bash
# 模块方式（推荐）
python3 -m time_chart_tool.cli --help

# 单个文件分析
python3 -m time_chart_tool.cli single file.json --label "baseline"

# 多文件对比分析
python3 -m time_chart_tool.cli compare file1.json:fp32 file2.json:tf32
```

### 编程接口
```python
from time_chart_tool import Analyzer

analyzer = Analyzer()
# ... 使用分析器
```

## 生成的文件

### whl 文件
- 新文件名：`time_chart_tool-1.0.0-py3-none-any.whl`
- 大小：13.8 KB

### 输出文件
- 单个文件分析：`{label}_single_file_analysis.json/xlsx`
- 对比分析：`comparison_analysis.json/xlsx`

## 兼容性

- ✅ 所有功能保持不变
- ✅ API 接口保持不变
- ✅ 输出格式保持不变
- ✅ 命令行参数保持不变

## 测试验证

1. **导入测试**：`python3 -c "import time_chart_tool; print('Import successful')"`
2. **命令行测试**：`python3 -m time_chart_tool.cli --help`
3. **功能测试**：所有原有功能正常工作

## 注意事项

1. 如果之前安装了旧版本，需要先卸载：
   ```bash
   pip3 uninstall torch-profiler-parser
   ```

2. 新的命令行工具可能需要添加到 PATH 中，或者使用模块方式调用：
   ```bash
   python3 -m time_chart_tool.cli --help
   ```

3. 所有文档和示例都已更新为新的包名

## 版本信息

- 重命名版本：1.0.0
- 重命名日期：2024年8月24日
- 重命名状态：完成 ✅

# -*- coding: utf-8 -*-
"""
Time Chart Tool

一个用于解析 PyTorch profiler 时间图表 JSON 数据的工具库。
"""

# 设置全局 print flush=True
import functools
import builtins

original_print = builtins.print
builtins.print = functools.partial(original_print, flush=True)

from .parser import PyTorchProfilerParser
from .models import ActivityEvent, ProfilerData
from .analyzer import Analyzer, KernelStatistics
from .cli import main

__version__ = "1.0.6"
__all__ = ["PyTorchProfilerParser", "ActivityEvent", "ProfilerData", "Analyzer", "KernelStatistics", "main"]

# 支持 python3 -m time_chart_tool 调用
if __name__ == "__main__":
    main()

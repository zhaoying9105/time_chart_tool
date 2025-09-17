# -*- coding: utf-8 -*-
"""
Time Chart Tool

一个用于解析 PyTorch profiler 时间图表 JSON 数据的工具库。
"""

from .parser import PyTorchProfilerParser
from .models import ActivityEvent, ProfilerData
from .analyzer import Analyzer, KernelStatistics

__version__ = "1.0.3"
__all__ = ["PyTorchProfilerParser", "ActivityEvent", "ProfilerData", "Analyzer", "KernelStatistics"]

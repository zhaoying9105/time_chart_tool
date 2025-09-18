# -*- coding: utf-8 -*-
"""
CLI模块 - 命令行接口
"""

from .main import main
from .commands import AnalysisCommand, CompareCommand, CommCommand

__all__ = ['main', 'AnalysisCommand', 'CompareCommand', 'CommCommand']

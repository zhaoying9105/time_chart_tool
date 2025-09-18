#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="time_chart_tool",
    version="1.0.4",
    description="PyTorch Profiler 高级分析工具",
    author="Time Chart Tool Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "openpyxl>=3.0.0",
        "matplotlib>=3.5.0",
    ],
    entry_points={
        "console_scripts": [
            "time-chart-tool=time_chart_tool.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
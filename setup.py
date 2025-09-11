"""
PyTorch Profiler Parser Tool Setup
"""

from setuptools import setup, find_packages

setup(
    name="time-chart-tool",
    version="1.0.2",
    author="PyTorch Profiler Parser Team",
    description="一个用于解析 PyTorch profiler 时间图表 JSON 数据的工具库",
    long_description="",
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "openpyxl>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "time-chart-tool=time_chart_tool.cli:main",
        ],
    },
)

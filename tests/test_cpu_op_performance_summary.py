#!/usr/bin/env python3
"""
测试cpu_op性能统计功能
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch

from src.time_chart_tool.analyzer import Analyzer
from src.time_chart_tool.models import ActivityEvent, ProfilerData


class TestCpuOpPerformanceSummary(unittest.TestCase):
    """测试cpu_op性能统计功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.analyzer = Analyzer()
        
        # 创建测试数据，包含cpu_op和对应的kernel事件
        self.test_events = [
            # cpu_op事件
            ActivityEvent(
                name="aten::mm",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1000,
                dur=50,  # cpu_op本身的耗时
                args={"External id": 1}
            ),
            ActivityEvent(
                name="aten::mm",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1200,
                dur=60,  # cpu_op本身的耗时
                args={"External id": 2}
            ),
            ActivityEvent(
                name="aten::add",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1400,
                dur=30,  # cpu_op本身的耗时
                args={"External id": 3}
            ),
            ActivityEvent(
                name="aten::add",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1500,
                dur=35,  # cpu_op本身的耗时
                args={"External id": 4}
            ),
            ActivityEvent(
                name="aten::conv2d",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1600,
                dur=80,  # cpu_op本身的耗时
                args={"External id": 5}
            ),
            # 对应的kernel事件（这些才是我们要统计的耗时）
            ActivityEvent(
                name="kernel1",
                cat="kernel",
                ph="X",
                pid=1,
                tid=2,
                ts=1100,
                dur=100,  # kernel耗时
                args={"External id": 1}
            ),
            ActivityEvent(
                name="kernel2",
                cat="kernel",
                ph="X",
                pid=1,
                tid=2,
                ts=1300,
                dur=150,  # kernel耗时
                args={"External id": 2}
            ),
            ActivityEvent(
                name="kernel3",
                cat="kernel",
                ph="X",
                pid=1,
                tid=2,
                ts=1450,
                dur=50,  # kernel耗时
                args={"External id": 3}
            ),
            ActivityEvent(
                name="kernel4",
                cat="kernel",
                ph="X",
                pid=1,
                tid=2,
                ts=1550,
                dur=60,  # kernel耗时
                args={"External id": 4}
            ),
            ActivityEvent(
                name="kernel5",
                cat="kernel",
                ph="X",
                pid=1,
                tid=2,
                ts=1650,
                dur=200,  # kernel耗时
                args={"External id": 5}
            ),
            # 其他类型的事件（应该被忽略）
            ActivityEvent(
                name="python_function",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=900,
                dur=300,
                args={}
            )
        ]
        
        self.test_data = ProfilerData(
            events=self.test_events,
            metadata={},
            trace_events=[]
        )
    
    def test_generate_cpu_op_performance_summary(self):
        """测试生成cpu_op性能统计摘要（基于kernel耗时）"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 运行性能统计
            self.analyzer.generate_cpu_op_performance_summary(self.test_data, temp_dir)
            
            # 验证输出文件
            xlsx_file = os.path.join(temp_dir, "cpu_op_performance_summary.xlsx")
            json_file = os.path.join(temp_dir, "cpu_op_performance_summary.json")
            
            assert os.path.exists(xlsx_file), "Excel文件应该存在"
            assert os.path.exists(json_file), "JSON文件应该存在"
            
            # 读取JSON文件并验证数据
            with open(json_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            # 验证数据结构
            assert len(summary_data) == 4, "应该有4行数据（3种cpu_op + 1个总计）"
            
            # 验证每种cpu_op的统计
            aten_mm_data = None
            aten_add_data = None
            aten_conv2d_data = None
            total_data = None
            
            for item in summary_data:
                if item['cpu_op_name'] == 'aten::mm':
                    aten_mm_data = item
                elif item['cpu_op_name'] == 'aten::add':
                    aten_add_data = item
                elif item['cpu_op_name'] == 'aten::conv2d':
                    aten_conv2d_data = item
                elif item['cpu_op_name'] == 'TOTAL':
                    total_data = item
            
            # 验证aten::mm的统计（基于kernel耗时）
            assert aten_mm_data is not None, "应该包含aten::mm的统计"
            assert aten_mm_data['call_count'] == 2, "aten::mm应该对应2个kernel调用"
            assert aten_mm_data['total_kernel_duration'] == 250, "aten::mm总kernel耗时应该是250"
            assert aten_mm_data['avg_kernel_duration'] == 125, "aten::mm平均kernel耗时应该是125"
            assert aten_mm_data['min_kernel_duration'] == 100, "aten::mm最小kernel耗时应该是100"
            assert aten_mm_data['max_kernel_duration'] == 150, "aten::mm最大kernel耗时应该是150"
            assert aten_mm_data['kernel_count'] == 2, "aten::mm应该有2个kernel事件"
            
            # 验证aten::add的统计（基于kernel耗时）
            assert aten_add_data is not None, "应该包含aten::add的统计"
            assert aten_add_data['call_count'] == 2, "aten::add应该对应2个kernel调用"
            assert aten_add_data['total_kernel_duration'] == 110, "aten::add总kernel耗时应该是110"
            assert aten_add_data['avg_kernel_duration'] == 55, "aten::add平均kernel耗时应该是55"
            assert aten_add_data['kernel_count'] == 2, "aten::add应该有2个kernel事件"
            
            # 验证aten::conv2d的统计（基于kernel耗时）
            assert aten_conv2d_data is not None, "应该包含aten::conv2d的统计"
            assert aten_conv2d_data['call_count'] == 1, "aten::conv2d应该对应1个kernel调用"
            assert aten_conv2d_data['total_kernel_duration'] == 200, "aten::conv2d总kernel耗时应该是200"
            assert aten_conv2d_data['avg_kernel_duration'] == 200, "aten::conv2d平均kernel耗时应该是200"
            assert aten_conv2d_data['kernel_count'] == 1, "aten::conv2d应该有1个kernel事件"
            
            # 验证总计
            assert total_data is not None, "应该包含总计行"
            assert total_data['call_count'] == 5, "总kernel调用次数应该是5"
            assert total_data['total_kernel_duration'] == 560, "总kernel耗时应该是560"
            assert total_data['percentage_of_total'] == 100.0, "总计行的比例应该是100%"
            assert total_data['kernel_count'] == 5, "总kernel事件数应该是5"
            
            # 验证比例计算
            total_duration = 560
            expected_mm_percentage = (250 / total_duration) * 100
            expected_add_percentage = (110 / total_duration) * 100
            expected_conv2d_percentage = (200 / total_duration) * 100
            
            assert abs(aten_mm_data['percentage_of_total'] - expected_mm_percentage) < 0.01, "aten::mm的比例计算错误"
            assert abs(aten_add_data['percentage_of_total'] - expected_add_percentage) < 0.01, "aten::add的比例计算错误"
            assert abs(aten_conv2d_data['percentage_of_total'] - expected_conv2d_percentage) < 0.01, "aten::conv2d的比例计算错误"
            
            # 验证排序（应该按总kernel耗时降序）
            assert summary_data[0]['cpu_op_name'] == 'aten::mm', "第一个应该是aten::mm（kernel耗时最多）"
            assert summary_data[1]['cpu_op_name'] == 'aten::conv2d', "第二个应该是aten::conv2d"
            assert summary_data[2]['cpu_op_name'] == 'aten::add', "第三个应该是aten::add"
            assert summary_data[3]['cpu_op_name'] == 'TOTAL', "最后一个应该是TOTAL"
    
    def test_generate_cpu_op_performance_summary_no_cpu_op(self):
        """测试没有cpu_op事件的情况"""
        # 创建只包含其他类型事件的测试数据
        other_events = [
            ActivityEvent(
                name="kernel1",
                cat="kernel",
                ph="X",
                pid=1,
                tid=2,
                ts=1100,
                dur=80,
                args={}
            ),
            ActivityEvent(
                name="python_function",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=900,
                dur=300,
                args={}
            )
        ]
        
        test_data_no_cpu_op = ProfilerData(
            events=other_events,
            metadata={},
            trace_events=[]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 运行性能统计
            self.analyzer.generate_cpu_op_performance_summary(test_data_no_cpu_op, temp_dir)
            
            # 验证没有生成文件
            xlsx_file = os.path.join(temp_dir, "cpu_op_performance_summary.xlsx")
            json_file = os.path.join(temp_dir, "cpu_op_performance_summary.json")
            
            assert not os.path.exists(xlsx_file), "不应该生成Excel文件"
            assert not os.path.exists(json_file), "不应该生成JSON文件"
    
    def test_generate_cpu_op_performance_summary_empty_data(self):
        """测试空数据的情况"""
        empty_data = ProfilerData(
            events=[],
            metadata={},
            trace_events=[]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 运行性能统计
            self.analyzer.generate_cpu_op_performance_summary(empty_data, temp_dir)
            
            # 验证没有生成文件
            xlsx_file = os.path.join(temp_dir, "cpu_op_performance_summary.xlsx")
            json_file = os.path.join(temp_dir, "cpu_op_performance_summary.json")
            
            assert not os.path.exists(xlsx_file), "不应该生成Excel文件"
            assert not os.path.exists(json_file), "不应该生成JSON文件"


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
高级分析器测试脚本
"""

import unittest
import tempfile
import os
import json
from pathlib import Path

# 添加当前目录到 Python 路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

from time_chart_tool.analyzer import Analyzer, KernelStatistics
from time_chart_tool.models import ActivityEvent, ProfilerData


class TestAnalyzer(unittest.TestCase):
    """测试高级分析器"""
    
    def setUp(self):
        """设置测试环境"""
        self.analyzer = Analyzer()
        
        # 创建测试数据
        self.test_events = [
            # cpu_op 事件
            ActivityEvent(
                name="aten::add",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1000.0,
                dur=50.0,
                args={
                    "External id": 1001,
                    "Input Strides": [[1, 2], [3, 4]],
                    "Input Dims": [[5, 6], [7, 8]],
                    "Input type": ["float", "float"]
                }
            ),
            ActivityEvent(
                name="aten::mul",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1100.0,
                dur=60.0,
                args={
                    "External id": 1002,
                    "Input Strides": [[1, 2]],
                    "Input Dims": [[5, 6]],
                    "Input type": ["float"]
                }
            ),
            # kernel 事件
            ActivityEvent(
                name="add_kernel",
                cat="kernel",
                ph="X",
                pid=1,
                tid=1,
                ts=1050.0,
                dur=30.0,
                args={"External id": 1001}
            ),
            ActivityEvent(
                name="add_kernel",
                cat="kernel",
                ph="X",
                pid=1,
                tid=1,
                ts=1080.0,
                dur=35.0,
                args={"External id": 1001}
            ),
            ActivityEvent(
                name="mul_kernel",
                cat="kernel",
                ph="X",
                pid=1,
                tid=1,
                ts=1160.0,
                dur=40.0,
                args={"External id": 1002}
            ),
            # 没有 External id 的事件（应该被过滤掉）
            ActivityEvent(
                name="other_event",
                cat="other",
                ph="X",
                pid=1,
                tid=1,
                ts=1200.0,
                dur=10.0,
                args={}
            )
        ]
        
        self.test_data = ProfilerData(
            metadata={"version": "1.0"},
            events=self.test_events,
            trace_events=[]
        )
    
    def test_reorganize_by_external_id(self):
        """测试按 External id 重组数据"""
        external_id_map = self.analyzer.reorganize_by_external_id(self.test_data)
        
        # 应该只有 2 个 External id
        self.assertEqual(len(external_id_map), 2)
        
        # 检查 1001
        self.assertIn(1001, external_id_map)
        events_1001 = external_id_map[1001]
        self.assertEqual(len(events_1001), 3)  # 1个cpu_op + 2个kernel
        
        # 检查 1002
        self.assertIn(1002, external_id_map)
        events_1002 = external_id_map[1002]
        self.assertEqual(len(events_1002), 2)  # 1个cpu_op + 1个kernel
    
    def test_get_cpu_op_info(self):
        """测试从 cpu_op 事件中提取信息"""
        cpu_op_event = self.test_events[0]  # aten::add 事件
        
        name, input_strides, input_dims, input_type = self.analyzer.get_cpu_op_info(cpu_op_event)
        
        self.assertEqual(name, "aten::add")
        self.assertEqual(input_strides, [[1, 2], [3, 4]])
        self.assertEqual(input_dims, [[5, 6], [7, 8]])
        self.assertEqual(input_type, ["float", "float"])
    
    def test_calculate_kernel_statistics(self):
        """测试计算 kernel 统计信息"""
        kernel_events = [
            event for event in self.test_events 
            if event.cat == "kernel" and event.external_id == 1001
        ]
        
        stats_list = self.analyzer.calculate_kernel_statistics(kernel_events)
        
        self.assertEqual(len(stats_list), 1)  # 只有一种 kernel
        
        stats = stats_list[0]
        self.assertEqual(stats.kernel_name, "add_kernel")
        self.assertEqual(stats.count, 2)
        self.assertEqual(stats.min_duration, 30.0)
        self.assertEqual(stats.max_duration, 35.0)
        self.assertEqual(stats.mean_duration, 32.5)
    
    def test_analyze_cpu_op_kernel_mapping(self):
        """测试分析 cpu_op 和 kernel 的映射关系"""
        mapping = self.analyzer.analyze_cpu_op_kernel_mapping(self.test_data)
        
        # 应该找到 2 个 cpu_op
        self.assertEqual(len(mapping), 2)
        
        # 检查 aten::add
        self.assertIn("aten::add", mapping)
        add_mapping = mapping["aten::add"]
        
        # 检查映射结构
        input_strides_key = tuple(tuple(s) for s in [[1, 2], [3, 4]])
        input_dims_key = tuple(tuple(d) for d in [[5, 6], [7, 8]])
        input_type_key = tuple(["float", "float"])
        
        self.assertIn(input_strides_key, add_mapping)
        self.assertIn(input_dims_key, add_mapping[input_strides_key])
        self.assertIn(input_type_key, add_mapping[input_strides_key][input_dims_key])
        
        kernel_events = add_mapping[input_strides_key][input_dims_key][input_type_key]
        self.assertEqual(len(kernel_events), 2)  # 2个 add_kernel 事件
    
    def test_generate_excel_from_mapping(self):
        """测试生成 Excel 表格"""
        mapping = self.analyzer.analyze_cpu_op_kernel_mapping(self.test_data)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_file = f.name
        
        try:
            # 生成 Excel 文件
            self.analyzer.generate_excel_from_mapping(mapping, temp_file)
            
            # 检查文件是否存在
            self.assertTrue(os.path.exists(temp_file))
            
            # 检查文件大小
            file_size = os.path.getsize(temp_file)
            self.assertGreater(file_size, 0)
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_merge_mappings(self):
        """测试合并映射"""
        # 创建两个模拟的映射
        mapping1 = {
            "aten::add": {
                tuple([tuple([1, 2])]): {
                    tuple([tuple([5, 6])]): {
                        tuple(["float"]): [self.test_events[2]]  # add_kernel
                    }
                }
            }
        }
        
        mapping2 = {
            "aten::add": {
                tuple([tuple([1, 2])]): {
                    tuple([tuple([5, 6])]): {
                        tuple(["float"]): [self.test_events[4]]  # mul_kernel
                    }
                }
            }
        }
        
        all_mappings = {
            "file1": mapping1,
            "file2": mapping2
        }
        
        merged = self.analyzer.merge_mappings(all_mappings)
        
        # 应该有一个合并的键
        self.assertEqual(len(merged), 1)
        
        # 检查合并的结果
        key = ("aten::add", tuple([tuple([1, 2])]), tuple([tuple([5, 6])]))
        self.assertIn(key, merged)
        
        label_events_list = merged[key]
        self.assertEqual(len(label_events_list), 2)  # 两个文件的映射
    
    def test_kernel_statistics_str(self):
        """测试 KernelStatistics 的字符串表示"""
        stats = KernelStatistics(
            kernel_name="test_kernel",
            min_duration=10.0,
            max_duration=20.0,
            mean_duration=15.0,
            variance=25.0,
            count=5
        )
        
        str_repr = str(stats)
        self.assertIn("test_kernel", str_repr)
        self.assertIn("count=5", str_repr)
        self.assertIn("mean=15.000", str_repr)


if __name__ == '__main__':
    unittest.main()

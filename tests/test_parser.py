"""
PyTorch Profiler Parser 单元测试
"""

import unittest
import json
import tempfile
import os
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from time_chart_tool.parser import PyTorchProfilerParser
from time_chart_tool.models import ActivityEvent, ProfilerData


class TestActivityEvent(unittest.TestCase):
    """测试 ActivityEvent 类"""
    
    def test_activity_event_creation(self):
        """测试创建 ActivityEvent"""
        event = ActivityEvent(
            name="test_event",
            cat="test_category",
            ph="X",
            pid=123,
            tid=456,
            ts=1000.0,
            dur=100.0,
            args={"key": "value"},
            id="test_id",
            stream_id=789
        )
        
        self.assertEqual(event.name, "test_event")
        self.assertEqual(event.cat, "test_category")
        self.assertEqual(event.ph, "X")
        self.assertEqual(event.pid, 123)
        self.assertEqual(event.tid, 456)
        self.assertEqual(event.ts, 1000.0)
        self.assertEqual(event.dur, 100.0)
        self.assertEqual(event.args, {"key": "value"})
        self.assertEqual(event.id, "test_id")
        self.assertEqual(event.stream_id, 789)
    
    def test_activity_event_properties(self):
        """测试 ActivityEvent 属性"""
        # 测试 External id
        event1 = ActivityEvent(
            name="test", cat="test", ph="X", pid=1, tid=1, ts=0.0,
            args={"External id": "233880"}
        )
        self.assertEqual(event1.external_id, "233880")
        
        # 测试 correlation id
        event2 = ActivityEvent(
            name="test", cat="test", ph="X", pid=1, tid=1, ts=0.0,
            args={"correlation id": "233880"}
        )
        self.assertEqual(event2.correlation_id, "233880")
        
        # 测试 kernel 判断
        event3 = ActivityEvent(
            name="cuda_kernel", cat="kernel", ph="X", pid=1, tid=1, ts=0.0
        )
        self.assertTrue(event3.is_kernel)
        
        # 测试 CUDA 事件判断
        event4 = ActivityEvent(
            name="cuda_event", cat="cuda", ph="X", pid=1, tid=1, ts=0.0
        )
        self.assertTrue(event4.is_cuda_event)


class TestProfilerData(unittest.TestCase):
    """测试 ProfilerData 类"""
    
    def setUp(self):
        """设置测试数据"""
        self.events = [
            ActivityEvent(name="event1", cat="cpu", ph="X", pid=1, tid=1, ts=0.0),
            ActivityEvent(name="event2", cat="kernel", ph="X", pid=1, tid=2, ts=100.0),
            ActivityEvent(name="event3", cat="cuda", ph="X", pid=2, tid=1, ts=200.0),
            ActivityEvent(name="event4", cat="cpu", ph="X", pid=1, tid=1, ts=300.0, 
                         args={"External id": "233880"}),
            ActivityEvent(name="event5", cat="kernel", ph="X", pid=2, tid=2, ts=400.0,
                         args={"correlation id": "233880"}),
        ]
        
        self.data = ProfilerData(
            metadata={"version": "1.0"},
            events=self.events,
            trace_events=[]
        )
    
    def test_profiler_data_properties(self):
        """测试 ProfilerData 属性"""
        self.assertEqual(self.data.total_events, 5)
        self.assertEqual(len(self.data.kernel_events), 2)
        self.assertEqual(len(self.data.cuda_events), 1)
        self.assertEqual(self.data.unique_processes, [1, 2])
        self.assertEqual(self.data.unique_threads, [1, 2])
    
    def test_search_methods(self):
        """测试搜索方法"""
        # 测试按进程搜索
        events_pid1 = self.data.get_events_by_process(1)
        self.assertEqual(len(events_pid1), 3)
        
        # 测试按线程搜索
        events_tid1 = self.data.get_events_by_thread(1)
        self.assertEqual(len(events_tid1), 3)
        
        # 测试按 External id 搜索
        external_events = self.data.get_events_by_external_id("233880")
        self.assertEqual(len(external_events), 1)
        self.assertEqual(external_events[0].name, "event4")
        
        # 测试按 correlation id 搜索
        correlation_events = self.data.get_events_by_correlation_id("233880")
        self.assertEqual(len(correlation_events), 1)
        self.assertEqual(correlation_events[0].name, "event5")


class TestPyTorchProfilerParser(unittest.TestCase):
    """测试 PyTorchProfilerParser 类"""
    
    def setUp(self):
        """设置测试环境"""
        self.parser = PyTorchProfilerParser()
        
        # 创建测试 JSON 数据
        self.test_data = {
            "metadata": {
                "version": "1.0",
                "description": "Test profiler data"
            },
            "traceEvents": [
                {
                    "name": "cpu_event",
                    "cat": "cpu",
                    "ph": "X",
                    "pid": 1,
                    "tid": 1,
                    "ts": 0.0,
                    "dur": 100.0,
                    "args": {"key": "value"}
                },
                {
                    "name": "kernel_event",
                    "cat": "kernel",
                    "ph": "X",
                    "pid": 1,
                    "tid": 2,
                    "ts": 100.0,
                    "dur": 50.0,
                    "args": {"External id": "233880"}
                },
                {
                    "name": "cuda_event",
                    "cat": "cuda",
                    "ph": "X",
                    "pid": 2,
                    "tid": 1,
                    "ts": 200.0,
                    "dur": 75.0,
                    "args": {"correlation id": "233880"}
                }
            ]
        }
    
    def test_load_json_file(self):
        """测试加载 JSON 文件"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_file = f.name
        
        try:
            # 测试加载
            data = self.parser.load_json_file(temp_file)
            
            # 验证数据
            self.assertIsNotNone(data)
            self.assertEqual(data.total_events, 3)
            self.assertEqual(len(data.kernel_events), 1)
            self.assertEqual(len(data.cuda_events), 1)
            self.assertEqual(data.metadata["version"], "1.0")
            
        finally:
            # 清理临时文件
            os.unlink(temp_file)
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with self.assertRaises(FileNotFoundError):
            self.parser.load_json_file("nonexistent_file.json")
    
    def test_load_invalid_json(self):
        """测试加载无效的 JSON 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.parser.load_json_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_search_functionality(self):
        """测试搜索功能"""
        # 先加载数据
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_file = f.name
        
        try:
            self.parser.load_json_file(temp_file)
            
            # 测试按进程搜索
            events = self.parser.search_by_process(1)
            self.assertEqual(len(events), 2)
            
            # 测试按线程搜索
            events = self.parser.search_by_thread(1)
            self.assertEqual(len(events), 2)
            
            # 测试按 External id 搜索
            events = self.parser.search_by_external_id("233880")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].name, "kernel_event")
            
            # 测试按 correlation id 搜索
            events = self.parser.search_by_correlation_id("233880")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].name, "cuda_event")
            
            # 测试按 ID 搜索（返回两个列表）
            external_events, correlation_events = self.parser.search_by_id("233880")
            self.assertEqual(len(external_events), 1)
            self.assertEqual(len(correlation_events), 1)
            
        finally:
            os.unlink(temp_file)
    
    def test_print_methods(self):
        """测试打印方法"""
        # 先加载数据
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_file = f.name
        
        try:
            self.parser.load_json_file(temp_file)
            
            # 测试打印元数据（不会抛出异常）
            self.parser.print_metadata()
            
            # 测试打印统计信息（不会抛出异常）
            self.parser.print_statistics()
            
            # 测试打印事件摘要
            events = self.parser.search_by_process(1)
            self.parser.print_events_summary(events, "进程 1 事件")
            
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()

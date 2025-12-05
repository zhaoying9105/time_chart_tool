
import unittest
from time_chart_tool.analyzer.op.comparator import stage3_comparison
from time_chart_tool.analyzer.utils.data_structures import AggregatedData
from time_chart_tool.models import ActivityEvent

class TestComparatorSkipSingle(unittest.TestCase):
    def setUp(self):
        self.agg_data1 = AggregatedData(
            cpu_events=[ActivityEvent(name="op1", ts=100, dur=10, args={}, cat="cpu_op", ph="X", pid=1, tid=1)],
            kernel_events=[],
            key="op1"
        )
        self.multiple_files_data_single = {
            "file1": {"op1": self.agg_data1}
        }
        self.multiple_files_data_empty = {}

    def test_skip_single_file(self):
        # 当只有一个文件时，stage3应返回该文件的结构化数据而非执行比较
        result = stage3_comparison(multiple_files_data=self.multiple_files_data_single)
        self.assertIn("op1", result)
        self.assertIn("file1", result["op1"])
        self.assertEqual(result["op1"]["file1"], self.agg_data1)

    def test_skip_empty_files(self):
        # 当没有文件时，应返回空字典
        result = stage3_comparison(multiple_files_data=self.multiple_files_data_empty)
        self.assertEqual(result, {})

if __name__ == '__main__':
    unittest.main()

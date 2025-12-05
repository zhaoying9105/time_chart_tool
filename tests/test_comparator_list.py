
import unittest
from time_chart_tool.analyzer.op.comparator import stage3_comparison
from time_chart_tool.analyzer.utils.data_structures import AggregatedData
from time_chart_tool.models import ActivityEvent

class TestComparatorListParams(unittest.TestCase):
    def setUp(self):
        # 构造 ActivityEvent 时补充必要的 cat, ph, pid, tid 参数
        self.agg_data1 = AggregatedData(
            cpu_events=[ActivityEvent(name="op1", ts=100, dur=10, args={}, cat="cpu_op", ph="X", pid=1, tid=1)],
            kernel_events=[],
            key="op1"
        )
        self.agg_data2 = AggregatedData(
            cpu_events=[ActivityEvent(name="op1", ts=200, dur=12, args={}, cat="cpu_op", ph="X", pid=1, tid=1)],
            kernel_events=[],
            key="op1"
        )
        
        self.multiple_files_data = {
            "file1": {"op1": self.agg_data1},
            "file2": {"op1": self.agg_data2}
        }

    def test_comparison_multiple_files(self):
        spec = ['name', 'dtype']
        result = stage3_comparison(multiple_files_data=self.multiple_files_data, aggregation_spec=spec)
        
        self.assertIn("op1", result)
        self.assertIn("file1", result["op1"])
        self.assertIn("file2", result["op1"])
        self.assertEqual(result["op1"]["file1"], self.agg_data1)
        self.assertEqual(result["op1"]["file2"], self.agg_data2)

    def test_comparison_single_file(self):
        # 单文件模式直接返回原数据
        result = stage3_comparison(single_file_data={"op1": self.agg_data1})
        self.assertIn("single_file", result)
        self.assertEqual(result["single_file"]["op1"], self.agg_data1)

if __name__ == '__main__':
    unittest.main()

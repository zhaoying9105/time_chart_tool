import unittest
from time_chart_tool.analyzer.op.comparator import compare
from time_chart_tool.analyzer.utils import AggregatedData
from time_chart_tool.models import ActivityEvent

class TestComparator(unittest.TestCase):
    def setUp(self):
        self.event = ActivityEvent(
            name="op1", cat="cpu_op", ph="X", ts=100, dur=50, pid=1, tid=1, args={}
        )
        self.aggregated_data = {
            "op1": AggregatedData(
                cpu_events=[self.event],
                kernel_events=[],
                key="op1"
            )
        }

    def test_compare_single_file(self):
        result = compare(single_file_data=self.aggregated_data)
        self.assertIn("single_file", result)
        self.assertEqual(result["single_file"], self.aggregated_data)

    def test_compare_multiple_files(self):
        multi_file_data = {
            "file1": self.aggregated_data,
            "file2": self.aggregated_data
        }
        result = compare(multiple_files_data=multi_file_data)
        # The result structure depends on implementation, but it should not be empty
        self.assertTrue(len(result) > 0)
        # Assuming it merges data
        self.assertIn("op1", result)

if __name__ == '__main__':
    unittest.main()

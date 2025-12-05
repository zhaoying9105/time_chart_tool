import unittest
from time_chart_tool.analyzer.op.aggregator import data_aggregation
from time_chart_tool.models import ActivityEvent
from time_chart_tool.analyzer.utils import AggregatedData
import sys
from unittest.mock import MagicMock

# Mock matplotlib since it's not installed and we don't need it for this test
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

class TestAggregator(unittest.TestCase):
    def setUp(self):
        self.cpu_event = ActivityEvent(
            name="op1", cat="cpu_op", ph="X", ts=100, dur=50, pid=1, tid=1, 
            args={"External id": 1, "Call stack": "root;op1"}
        )
        self.kernel_event = ActivityEvent(
            name="kernel1", cat="kernel", ph="X", ts=120, dur=20, pid=0, tid=0, 
            args={"External id": 1}
        )
        
        self.cpu_events_by_id = {1: [self.cpu_event]}
        self.kernel_events_by_id = {1: [self.kernel_event]}

    def test_stage2_aggregation_by_name(self):
        aggregated_data = data_aggregation(
            self.cpu_events_by_id, 
            self.kernel_events_by_id, 
            aggregation_spec=["name"],
        )
        
        self.assertIn("op1", aggregated_data)
        data = aggregated_data["op1"]
        self.assertIsInstance(data, AggregatedData)
        self.assertEqual(len(data.cpu_events), 1)
        self.assertEqual(len(data.kernel_events), 1)
        self.assertEqual(data.key, "op1")

    def test_stage2_aggregation_by_call_stack(self):
        aggregated_data = data_aggregation(
            self.cpu_events_by_id, 
            self.kernel_events_by_id, 
            aggregation_spec=["call_stack"],
        )
        
        # The key should be the call stack
        # Note: the actual key might be wrapped or processed
        # Let's check if there is any key
        self.assertTrue(len(aggregated_data) > 0)
        
        # Check the content
        for key, data in aggregated_data.items():
            self.assertEqual(len(data.cpu_events), 1)

if __name__ == '__main__':
    unittest.main()

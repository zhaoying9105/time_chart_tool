
import unittest
import os
from time_chart_tool.analyzer.op.aggregator import stage2_data_aggregation, _generate_aggregation_key
from time_chart_tool.models import ActivityEvent

class TestAggregatorListParams(unittest.TestCase):
    def setUp(self):
        self.cpu_events = {
            "ext1": [
                ActivityEvent(name="op1", ts=100, dur=10, args={"Input type": "float", "Input Dims": [1, 2]}, cat="cpu_op", ph="X", pid=1, tid=1),
                ActivityEvent(name="op1", ts=200, dur=10, args={"Input type": "float", "Input Dims": [1, 2]}, cat="cpu_op", ph="X", pid=1, tid=1)
            ],
            "ext2": [
                ActivityEvent(name="op2", ts=150, dur=20, args={"Input type": "int", "Input Dims": [3, 4]}, cat="cpu_op", ph="X", pid=1, tid=1)
            ]
        }
        self.kernel_events = {}

    def test_aggregation_spec_name(self):
        spec = ['name']
        result = stage2_data_aggregation(self.cpu_events, self.kernel_events, aggregation_spec=spec)
        self.assertIn("op1", result)
        self.assertIn("op2", result)
        self.assertEqual(len(result["op1"].cpu_events), 2)

    def test_aggregation_spec_dtype(self):
        spec = ['dtype']
        result = stage2_data_aggregation(self.cpu_events, self.kernel_events, aggregation_spec=spec)
        self.assertIn("float", result)
        self.assertIn("int", result)

    def test_aggregation_spec_multiple(self):
        spec = ['name', 'dtype']
        result = stage2_data_aggregation(self.cpu_events, self.kernel_events, aggregation_spec=spec)
        self.assertIn(("op1", "float"), result)
        self.assertIn(("op2", "int"), result)

    def test_invalid_spec(self):
        spec = ['invalid']
        with self.assertRaises(ValueError):
            stage2_data_aggregation(self.cpu_events, self.kernel_events, aggregation_spec=spec)

if __name__ == '__main__':
    unittest.main()

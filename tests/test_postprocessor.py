import unittest
from time_chart_tool.analyzer.op.postprocessor import postprocessing
from time_chart_tool.models import ActivityEvent, ProfilerData
from time_chart_tool.analyzer.utils import normalize_call_stack

class TestPostProcessor(unittest.TestCase):
    def setUp(self):
        self.events = [
            ActivityEvent(
                name="op1", cat="cpu_op", ph="X", ts=100, dur=50, pid=1, tid=1, 
                args={"External id": 1, "Call stack": "root;op1"}
            ),
            ActivityEvent(
                name="kernel1", cat="kernel", ph="X", ts=120, dur=20, pid=0, tid=0, 
                args={"External id": 1}
            ),
            ActivityEvent(
                name="op2", cat="cpu_op", ph="X", ts=200, dur=50, pid=1, tid=1, 
                args={"External id": 2, "Call stack": "root;op2"}
            )
        ]
        self.profiler_data = ProfilerData(events=self.events, metadata={})

    def test_postprocessing(self):
        cpu_events_by_id, kernel_events_by_id = postprocessing(self.profiler_data)
        
        self.assertIn(1, cpu_events_by_id)
        self.assertIn(1, kernel_events_by_id)
        # ID 2 should be filtered out because it has no corresponding kernel event
        # while ID 1 does.
        self.assertNotIn(2, cpu_events_by_id)
        
        self.assertEqual(len(cpu_events_by_id[1]), 1)
        self.assertEqual(len(kernel_events_by_id[1]), 1)
        self.assertEqual(cpu_events_by_id[1][0].name, "op1")
        self.assertEqual(kernel_events_by_id[1][0].name, "kernel1")

    def test_postprocessing_filtering(self):
        # Add a kernel event for op2 so it isn't filtered out by the "missing kernel" check
        events_with_kernel2 = self.events + [
            ActivityEvent(
                name="kernel2", cat="kernel", ph="X", ts=220, dur=20, pid=0, tid=0, 
                args={"External id": 2}
            )
        ]
        profiler_data = ProfilerData(events=events_with_kernel2, metadata={})

        cpu_events_by_id, _ = postprocessing(
            profiler_data,
            include_op_patterns=["op1"]
        )
        self.assertIn(1, cpu_events_by_id)
        self.assertNotIn(2, cpu_events_by_id)

    def test_normalize_call_stack(self):
        stack = ["root", "op1", "op2"]
        normalized = normalize_call_stack(stack)
        # normalize_call_stack returns a CallStackWrapper
        self.assertEqual(normalized.call_stack, ["root", "op1", "op2"])
        
        stack_with_addr = ["root", "op1 object at 0x123", "op2"]
        normalized = normalize_call_stack(stack_with_addr)
        self.assertEqual(normalized.call_stack, ["root", "op1", "op2"])
        
        self.assertEqual(normalize_call_stack(None).call_stack, [])

if __name__ == '__main__':
    unittest.main()

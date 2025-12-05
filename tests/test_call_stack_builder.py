import unittest
from time_chart_tool.call_stack_builder import build_call_stacks, CallStackNode
from time_chart_tool.utils.tree_utils import extract_and_attach_call_stacks
from time_chart_tool.models import ActivityEvent
from time_chart_tool.utils.event_utils import normalize_timestamps

class TestCallStackBuilder(unittest.TestCase):
    def setUp(self):
        self.events = [
            ActivityEvent(name="root", cat="cpu_op", ph="X", ts=100, dur=100, pid=1, tid=1, args={}),
            ActivityEvent(name="child1", cat="cpu_op", ph="X", ts=110, dur=30, pid=1, tid=1, args={}),
            ActivityEvent(name="child2", cat="cpu_op", ph="X", ts=150, dur=40, pid=1, tid=1, args={})
        ]

    def test_build_call_stacks(self):
        stacks = build_call_stacks(self.events)
        self.assertIn((1, 1), stacks)
        virtual_root = stacks[(1, 1)]
        
        self.assertIsInstance(virtual_root, CallStackNode)
        self.assertEqual(virtual_root.event.name, "ROOT")
        
        # The virtual root should have one child: "root"
        self.assertEqual(len(virtual_root.children), 1)
        root_node = virtual_root.children[0]
        self.assertEqual(root_node.event.name, "root")
        
        # "root" should have 2 children: child1 and child2
        self.assertEqual(len(root_node.children), 2)
        self.assertEqual(root_node.children[0].event.name, "child1")
        self.assertEqual(root_node.children[1].event.name, "child2")

    def test_extract_and_attach_call_stacks(self):
        stacks = build_call_stacks(self.events)
        extract_and_attach_call_stacks(self.events, stacks)
        
        # Check if call_stack_from_tree is attached
        # Note: extract_and_attach_call_stacks modifies events in place
        
        # Need to find the events again or rely on the fact they are the same objects
        root_event = self.events[0]
        child1_event = self.events[1]
        
        self.assertIsNotNone(root_event.call_stack_from_tree)
        # The call stack for "root" (child of ROOT) should be ["root"]
        self.assertEqual(root_event.call_stack_from_tree, ["root"])
        
        self.assertIsNotNone(child1_event.call_stack_from_tree)
        # The call stack for "child1" should be ["root", "child1"]
        self.assertEqual(child1_event.call_stack_from_tree, ["root", "child1"])

    def test_normalize_timestamps(self):
        normalized_events = normalize_timestamps(self.events)
        self.assertEqual(normalized_events[0].ts, 0)
        self.assertEqual(normalized_events[1].ts, 10)
        self.assertEqual(normalized_events[2].ts, 50)

if __name__ == '__main__':
    unittest.main()

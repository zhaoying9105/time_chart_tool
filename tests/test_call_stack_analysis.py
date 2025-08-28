#!/usr/bin/env python3
"""
测试call stack分析功能
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

# 添加src目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from time_chart_tool.analyzer import Analyzer, PythonFunctionNode
from time_chart_tool.models import ActivityEvent


class TestCallStackAnalysis:
    """测试call stack分析功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.analyzer = Analyzer()
        
        # 创建测试数据（单线程）
        self.test_events = [
            # python_function事件（线程1）
            ActivityEvent(
                name="main",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,  # 线程1
                ts=1000,
                dur=150,  # 修改为150，确保包含CPU OP的完整时间范围
                args={"Python id": 1, "Python parent id": None}
            ),
            ActivityEvent(
                name="train",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,  # 线程1
                ts=1050,
                dur=80,
                args={"Python id": 2, "Python parent id": 1}
            ),
            ActivityEvent(
                name="forward",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,  # 线程1
                ts=1070,
                dur=60,
                args={"Python id": 3, "Python parent id": 2}
            ),
            # cpu_op事件（线程1）
            ActivityEvent(
                name="aten::mm",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,  # 线程1
                ts=1080,
                dur=40,
                args={
                    "External id": 100,
                    "Input Strides": [(3, 1), (1, 3)],
                    "Input Dims": [(2048, 3), (3, 32)],
                    "Input type": ["float", "float"]
                }
            ),
            # kernel事件（可能在不同线程）
            ActivityEvent(
                name="void MLUUnion1KernelGemmRb",
                cat="kernel",
                ph="X",
                pid=1,
                tid=2,  # 线程2（跨线程）
                ts=1085,
                dur=30,
                args={"External id": 100}
            ),
        ]
        
        # 创建多线程测试数据
        self.multi_thread_events = [
            # 线程1的python_function事件
            ActivityEvent(
                name="main_thread1",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1000,
                dur=150,
                args={"Python id": 1, "Python parent id": None}
            ),
            ActivityEvent(
                name="func1_thread1",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1050,
                dur=80,
                args={"Python id": 2, "Python parent id": 1}
            ),
            # 线程2的python_function事件
            ActivityEvent(
                name="main_thread2",
                cat="python_function",
                ph="X",
                pid=1,
                tid=2,
                ts=2000,
                dur=150,
                args={"Python id": 3, "Python parent id": None}
            ),
            ActivityEvent(
                name="func1_thread2",
                cat="python_function",
                ph="X",
                pid=1,
                tid=2,
                ts=2050,
                dur=80,
                args={"Python id": 4, "Python parent id": 3}
            ),
            # cpu_op事件（线程1）
            ActivityEvent(
                name="aten::mm_thread1",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1080,
                dur=40,
                args={
                    "External id": 100,
                    "Input Strides": [(3, 1), (1, 3)],
                    "Input Dims": [(2048, 3), (3, 32)],
                    "Input type": ["float", "float"]
                }
            ),
            # cpu_op事件（线程2）
            ActivityEvent(
                name="aten::mm_thread2",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=2,
                ts=2080,
                dur=40,
                args={
                    "External id": 101,
                    "Input Strides": [(3, 1), (1, 3)],
                    "Input Dims": [(2048, 3), (3, 32)],
                    "Input type": ["float", "float"]
                }
            ),
            # kernel事件（可能在不同线程）
            ActivityEvent(
                name="void MLUUnion1KernelGemmRb",
                cat="kernel",
                ph="X",
                pid=1,
                tid=3,  # 线程3（跨线程）
                ts=1085,
                dur=30,
                args={"External id": 100}
            ),
            ActivityEvent(
                name="void MLUUnion1KernelGemmRb",
                cat="kernel",
                ph="X",
                pid=1,
                tid=3,  # 线程3（跨线程）
                ts=2085,
                dur=30,
                args={"External id": 101}
            ),
        ]
    
    def test_build_python_function_tree(self):
        """测试python_function树构建功能（按线程分组）"""
        from time_chart_tool.models import ProfilerData
        
        # 创建ProfilerData对象
        data = ProfilerData(
            metadata={},
            events=self.test_events,
            trace_events=[]
        )
        
        # 构建python_function树（按线程分组）
        thread_trees = self.analyzer.build_python_function_tree(data)
        
        # 验证线程分组
        assert len(thread_trees) == 1  # 只有一个线程（tid=1）
        assert 1 in thread_trees
        
        # 获取线程1的python_function树
        python_tree = thread_trees[1]
        
        # 验证树结构
        assert len(python_tree) == 4  # 3个python_function节点 + 1个root_nodes信息
        
        # 验证root节点识别
        root_nodes = python_tree['_root_nodes']
        assert len(root_nodes) == 1
        root_node = root_nodes[0]
        assert root_node.name == "main"
        assert root_node.python_parent_id is None
        assert len(root_node.children) == 1
        
        # 验证子节点
        child_node = root_node.children[0]
        assert child_node.name == "train"
        assert child_node.python_parent_id == 1
        assert len(child_node.children) == 1
        
        # 验证孙节点
        grandchild_node = child_node.children[0]
        assert grandchild_node.name == "forward"
        assert grandchild_node.python_parent_id == 2
        assert len(grandchild_node.children) == 0
    
    def test_find_containing_python_function(self):
        """测试查找包含cpu_op的python_function（在同一线程内）"""
        from time_chart_tool.models import ProfilerData
        
        # 创建ProfilerData对象
        data = ProfilerData(
            metadata={},
            events=self.test_events,
            trace_events=[]
        )
        
        # 构建python_function树（按线程分组）
        thread_trees = self.analyzer.build_python_function_tree(data)
        
        # 查找包含cpu_op的python_function
        cpu_op_event = self.test_events[3]  # aten::mm事件
        python_node = self.analyzer.find_containing_python_function(cpu_op_event, thread_trees)
        
        # 验证结果
        assert python_node is not None
        assert python_node.name == "forward"  # 应该找到最内层的函数
        
        # 验证递归查找逻辑：cpu_op时间范围应该在forward函数内
        cpu_start = cpu_op_event.ts
        cpu_end = cpu_op_event.ts + (cpu_op_event.dur or 0)
        assert python_node.contains_time_range(cpu_start, cpu_end)
        
        # 验证这是最小的包含节点
        for child in python_node.children:
            assert not child.contains_time_range(cpu_start, cpu_end)
    
    def test_recursive_search_logic(self):
        """测试递归查找逻辑（按线程分组）"""
        from time_chart_tool.models import ProfilerData
        
        # 创建更复杂的测试数据，包含多个嵌套层级
        complex_events = [
            # 根节点
            ActivityEvent(
                name="main",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1000,
                dur=200,
                args={"Python id": 1, "Python parent id": None}
            ),
            # 第一层子节点
            ActivityEvent(
                name="level1",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1050,
                dur=150,
                args={"Python id": 2, "Python parent id": 1}
            ),
            # 第二层子节点
            ActivityEvent(
                name="level2",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1070,
                dur=110,
                args={"Python id": 3, "Python parent id": 2}
            ),
            # 第三层子节点（最内层）
            ActivityEvent(
                name="level3",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1080,
                dur=90,
                args={"Python id": 4, "Python parent id": 3}
            ),
            # cpu_op事件（在level3的时间范围内）
            ActivityEvent(
                name="aten::mm",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1085,
                dur=40,
                args={
                    "External id": 100,
                    "Input Strides": [(3, 1), (1, 3)],
                    "Input Dims": [(2048, 3), (3, 32)],
                    "Input type": ["float", "float"]
                }
            ),
        ]
        
        # 创建ProfilerData对象
        data = ProfilerData(
            metadata={},
            events=complex_events,
            trace_events=[]
        )
        
        # 构建python_function树（按线程分组）
        thread_trees = self.analyzer.build_python_function_tree(data)
        
        # 验证线程分组
        assert len(thread_trees) == 1
        assert 1 in thread_trees
        
        # 获取线程1的python_function树
        python_tree = thread_trees[1]
        
        # 查找包含cpu_op的python_function
        cpu_op_event = complex_events[4]  # aten::mm事件
        python_node = self.analyzer.find_containing_python_function(cpu_op_event, thread_trees)
        
        # 验证结果：应该找到最内层的level3函数
        assert python_node is not None
        assert python_node.name == "level3"
        
        # 验证递归查找逻辑：cpu_op时间范围应该在level3函数内
        cpu_start = cpu_op_event.ts
        cpu_end = cpu_op_event.ts + (cpu_op_event.dur or 0)
        assert python_node.contains_time_range(cpu_start, cpu_end)
        
        # 验证这是最小的包含节点（level3没有子节点）
        assert len(python_node.children) == 0
    
    def test_get_python_call_stack(self):
        """测试获取python_function调用栈（按线程分组）"""
        from time_chart_tool.models import ProfilerData
        
        # 创建ProfilerData对象
        data = ProfilerData(
            metadata={},
            events=self.test_events,
            trace_events=[]
        )
        
        # 构建python_function树（按线程分组）
        thread_trees = self.analyzer.build_python_function_tree(data)
        
        # 获取线程1的python_function树
        python_tree = thread_trees[1]
        
        # 获取forward节点
        forward_node = python_tree[3]  # Python id = 3
        
        # 获取调用栈
        call_stack = self.analyzer.get_python_call_stack(forward_node, thread_trees)
        
        # 验证调用栈
        assert len(call_stack) == 3
        assert call_stack[0] == "main"  # 根节点
        assert call_stack[1] == "train"  # 中间节点
        assert call_stack[2] == "forward"  # 当前节点
        
        # 验证调用栈不包含无意义的函数名
        call_stack_str = ' -> '.join(call_stack)
        assert "<built-in method" not in call_stack_str
        assert "torch/nn" not in call_stack_str
    
    def test_filter_meaningful_functions(self):
        """测试过滤无意义的函数名（按线程分组）"""
        from time_chart_tool.models import ProfilerData
        
        # 创建包含无意义函数名的测试数据
        test_events = [
            ActivityEvent(
                name="main",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1000,
                dur=100,
                args={"Python id": 1, "Python parent id": None}
            ),
            ActivityEvent(
                name="<built-in method>",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1050,
                dur=50,
                args={"Python id": 2, "Python parent id": 1}
            ),
            ActivityEvent(
                name="torch/nn/module",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1070,
                dur=30,
                args={"Python id": 3, "Python parent id": 2}
            ),
            ActivityEvent(
                name="my_function",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1080,
                dur=20,
                args={"Python id": 4, "Python parent id": 3}
            ),
        ]
        
        # 创建ProfilerData对象
        data = ProfilerData(
            metadata={},
            events=test_events,
            trace_events=[]
        )
        
        # 构建python_function树（按线程分组）
        thread_trees = self.analyzer.build_python_function_tree(data)
        
        # 获取线程1的python_function树
        python_tree = thread_trees[1]
        
        # 获取my_function节点
        my_function_node = python_tree[4]  # Python id = 4
        
        # 获取调用栈
        call_stack = self.analyzer.get_python_call_stack(my_function_node, thread_trees)
        
        # 验证调用栈过滤了无意义的函数名
        assert len(call_stack) == 2  # 现在过滤掉无意义的函数名
        assert call_stack[0] == "main"  # 根节点
        assert call_stack[1] == "my_function"  # 当前节点
        
        # 验证无意义的函数名被过滤掉了
        call_stack_str = ' -> '.join(call_stack)
        assert "<built-in method>" not in call_stack_str, "built-in method应该被过滤"
        assert "torch/nn" not in call_stack_str, "torch/nn应该被过滤"

    def test_multi_thread_isolation(self):
        """测试多线程隔离功能"""
        from time_chart_tool.models import ProfilerData
        
        # 创建ProfilerData对象
        data = ProfilerData(
            metadata={},
            events=self.multi_thread_events,
            trace_events=[]
        )
        
        # 构建python_function树（按线程分组）
        thread_trees = self.analyzer.build_python_function_tree(data)
        
        # 验证线程分组
        assert len(thread_trees) == 2  # 有两个线程（tid=1, tid=2）
        assert 1 in thread_trees
        assert 2 in thread_trees
        
        # 验证线程1的树结构
        tree1 = thread_trees[1]
        root_nodes1 = tree1['_root_nodes']
        assert len(root_nodes1) == 1
        assert root_nodes1[0].name == "main_thread1"
        assert len(root_nodes1[0].children) == 1
        assert root_nodes1[0].children[0].name == "func1_thread1"
        
        # 验证线程2的树结构
        tree2 = thread_trees[2]
        root_nodes2 = tree2['_root_nodes']
        assert len(root_nodes2) == 1
        assert root_nodes2[0].name == "main_thread2"
        assert len(root_nodes2[0].children) == 1
        assert root_nodes2[0].children[0].name == "func1_thread2"
        
        # 验证线程隔离：线程1的cpu_op只能找到线程1的python_function
        cpu_op_thread1 = self.multi_thread_events[4]  # aten::mm_thread1 (tid=1)
        python_node1 = self.analyzer.find_containing_python_function(cpu_op_thread1, thread_trees)
        assert python_node1 is not None
        assert python_node1.name == "func1_thread1"
        
        # 验证线程隔离：线程2的cpu_op只能找到线程2的python_function
        cpu_op_thread2 = self.multi_thread_events[5]  # aten::mm_thread2 (tid=2)
        python_node2 = self.analyzer.find_containing_python_function(cpu_op_thread2, thread_trees)
        assert python_node2 is not None
        assert python_node2.name == "func1_thread2"
        
        # 验证跨线程查找失败
        # 如果尝试用线程1的cpu_op在线程2的树中查找，应该返回None
        # 但这是通过线程ID自动隔离的，所以不需要额外测试

    def test_call_stack_comparison_analysis(self):
        """测试call stack比较分析功能"""
        from time_chart_tool.models import ProfilerData
        import tempfile
        import os
        
        # 创建测试数据（两个不同的time chart）
        # Time Chart 1: 线程1
        events1 = [
            # python_function事件
            ActivityEvent(
                name="main",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1000,
                dur=150,
                args={"Python id": 1, "Python parent id": None}
            ),
            ActivityEvent(
                name="train",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1050,
                dur=80,
                args={"Python id": 2, "Python parent id": 1}
            ),
            # cpu_op事件
            ActivityEvent(
                name="aten::mm",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1080,
                dur=40,
                args={
                    "External id": 100,
                    "Input Strides": [(3, 1), (1, 3)],
                    "Input Dims": [(2048, 3), (3, 32)],
                    "Input type": ["float", "float"]
                }
            ),
            # kernel事件
            ActivityEvent(
                name="void MLUUnion1KernelGemmRb",
                cat="kernel",
                ph="X",
                pid=1,
                tid=2,
                ts=1085,
                dur=30,
                args={"External id": 100}
            ),
        ]
        
        # Time Chart 2: 线程2，相同的call stack但不同的input type
        events2 = [
            # python_function事件
            ActivityEvent(
                name="main",
                cat="python_function",
                ph="X",
                pid=1,
                tid=2,
                ts=2000,
                dur=150,
                args={"Python id": 3, "Python parent id": None}
            ),
            ActivityEvent(
                name="train",
                cat="python_function",
                ph="X",
                pid=1,
                tid=2,
                ts=2050,
                dur=80,
                args={"Python id": 4, "Python parent id": 3}
            ),
            # cpu_op事件（相同的call stack，但不同的input type）
            ActivityEvent(
                name="aten::mm",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=2,
                ts=2080,
                dur=40,
                args={
                    "External id": 101,
                    "Input Strides": [(3, 1), (1, 3)],
                    "Input Dims": [(2048, 3), (3, 32)],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"]  # 不同的input type
                }
            ),
            # kernel事件
            ActivityEvent(
                name="void MLUUnion1KernelGemmRb",
                cat="kernel",
                ph="X",
                pid=1,
                tid=3,
                ts=2085,
                dur=30,
                args={"External id": 101}
            ),
        ]
        
        # 创建临时文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建两个JSON文件
            file1 = os.path.join(temp_dir, "chart1.json")
            file2 = os.path.join(temp_dir, "chart2.json")
            
            # 创建ProfilerData对象并保存为JSON
            data1 = ProfilerData(metadata={}, events=events1, trace_events=[])
            data2 = ProfilerData(metadata={}, events=events2, trace_events=[])
            
            # 模拟JSON文件（这里我们直接使用数据对象进行测试）
            file_labels = [
                (file1, "fp32"),
                (file2, "bf16")
            ]
            
            # 模拟analyze_multiple_files的结果
            all_mappings = {}
            
            # 处理第一个文件
            mapping1 = self.analyzer.analyze_cpu_op_kernel_mapping(data1)
            all_mappings["fp32"] = mapping1
            
            # 处理第二个文件
            mapping2 = self.analyzer.analyze_cpu_op_kernel_mapping(data2)
            all_mappings["bf16"] = mapping2
            
            # 测试call stack比较分析
            self.analyzer.generate_call_stack_comparison(all_mappings, file_labels, temp_dir, ['json'])
            
            # 验证输出文件
            output_file = os.path.join(temp_dir, "call_stack_comparison_analysis.json")
            assert os.path.exists(output_file), "输出文件应该存在"
            
            # 读取并验证JSON内容
            import json
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据结构
            assert len(data) == 1, "应该有一个call stack记录"
            
            row = data[0]
            assert 'call_stack' in row, "应该包含call_stack字段"
            assert 'fp32_input_types' in row, "应该包含fp32_input_types字段"
            assert 'bf16_input_types' in row, "应该包含bf16_input_types字段"
            assert 'input_types_equal' in row, "应该包含input_types_equal字段"
            
            # 验证call stack内容
            expected_call_stack = "main -> train"
            assert row['call_stack'] == expected_call_stack, f"call stack应该是 {expected_call_stack}"
            
            # 验证input types不一致
            assert row['input_types_equal'] == False, "input types应该不一致"
            assert "('float', 'float')" in row['fp32_input_types'], "fp32应该有float类型"
            assert "('c10::BFloat16', 'c10::BFloat16')" in row['bf16_input_types'], "bf16应该有BFloat16类型"
            
            # 验证pid和tid信息
            assert 'fp32_pids' in row, "应该包含fp32_pids字段"
            assert 'fp32_tids' in row, "应该包含fp32_tids字段"
            assert 'bf16_pids' in row, "应该包含bf16_pids字段"
            assert 'bf16_tids' in row, "应该包含bf16_tids字段"
            
            assert "1" in row['fp32_pids'], "fp32的pid应该是1"
            assert "2" in row['fp32_tids'], "fp32的tid应该是2（kernel的tid）"
            assert "1" in row['bf16_pids'], "bf16的pid应该是1"
            assert "3" in row['bf16_tids'], "bf16的tid应该是3（kernel的tid）"

    def test_call_stack_grouping_logic(self):
        """测试call stack分组逻辑"""
        from time_chart_tool.models import ProfilerData
        
        # 创建测试数据：相同的call stack，不同的cpu_op属性
        events = [
            # python_function事件
            ActivityEvent(
                name="main",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1000,
                dur=150,
                args={"Python id": 1, "Python parent id": None}
            ),
            ActivityEvent(
                name="train",
                cat="python_function",
                ph="X",
                pid=1,
                tid=1,
                ts=1050,
                dur=80,
                args={"Python id": 2, "Python parent id": 1}
            ),
            # cpu_op事件1：aten::mm
            ActivityEvent(
                name="aten::mm",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1080,
                dur=40,
                args={
                    "External id": 100,
                    "Input Strides": [(3, 1), (1, 3)],
                    "Input Dims": [(2048, 3), (3, 32)],
                    "Input type": ["float", "float"]
                }
            ),
            # cpu_op事件2：aten::add（相同的call stack，不同的cpu_op）
            ActivityEvent(
                name="aten::add",
                cat="cpu_op",
                ph="X",
                pid=1,
                tid=1,
                ts=1090,  # 修改时间，确保在train函数内
                dur=30,
                args={
                    "External id": 101,
                    "Input Strides": [(1, 1), (1, 1)],
                    "Input Dims": [(1024, 1), (1, 1024)],
                    "Input type": ["float", "float"]
                }
            ),
            # kernel事件1
            ActivityEvent(
                name="void MLUUnion1KernelGemmRb",
                cat="kernel",
                ph="X",
                pid=1,
                tid=2,
                ts=1085,
                dur=30,
                args={"External id": 100}
            ),
            # kernel事件2
            ActivityEvent(
                name="void MLUUnion1KernelAdd",
                cat="kernel",
                ph="X",
                pid=1,
                tid=2,
                ts=1095,  # 修改时间，确保在cpu_op之后
                dur=20,
                args={"External id": 101}
            ),
        ]
        
        # 创建ProfilerData对象
        data = ProfilerData(
            metadata={},
            events=events,
            trace_events=[]
        )
        
        # 分析数据
        mapping = self.analyzer.analyze_cpu_op_kernel_mapping(data)
        
        # 验证mapping结构
        assert "aten::mm" in mapping
        assert "aten::add" in mapping
        
        # 验证call stack信息
        mm_kernel_events = None
        add_kernel_events = None
        
        for cpu_op_name, strides_map in mapping.items():
            for input_strides, dims_map in strides_map.items():
                for input_dims, types_map in dims_map.items():
                    for input_type, kernel_events in types_map.items():
                        if cpu_op_name == "aten::mm":
                            mm_kernel_events = kernel_events
                        elif cpu_op_name == "aten::add":
                            add_kernel_events = kernel_events
        
        # 验证两个cpu_op有相同的call stack
        assert mm_kernel_events is not None
        assert add_kernel_events is not None
        assert mm_kernel_events[0].call_stack == add_kernel_events[0].call_stack
        
        # 测试call stack比较分析
        all_mappings = {"test": mapping}
        file_labels = [("test.json", "test")]
        
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self.analyzer.generate_call_stack_comparison(all_mappings, file_labels, temp_dir, ['json'])
            
            # 验证输出文件
            output_file = os.path.join(temp_dir, "call_stack_comparison_analysis.json")
            assert os.path.exists(output_file)
            
            # 读取并验证JSON内容
            import json
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 应该只有一个call stack记录，包含两个cpu_op
            assert len(data) == 1, "应该只有一个call stack记录"
            
            row = data[0]
            assert 'call_stack' in row
            assert 'test_cpu_op_names' in row
            assert 'test_input_types' in row
            
            # 验证包含两个cpu_op
            cpu_op_names = row['test_cpu_op_names'].split('||')
            assert "aten::mm" in cpu_op_names
            assert "aten::add" in cpu_op_names
            assert len(cpu_op_names) == 2


if __name__ == "__main__":
    pytest.main([__file__])

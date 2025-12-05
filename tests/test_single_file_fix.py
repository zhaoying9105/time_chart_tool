
import unittest
from unittest.mock import MagicMock
from time_chart_tool.analyzer.op.comparator import stage3_comparison
from time_chart_tool.analyzer.op.presenter import stage4_presentation
from time_chart_tool.analyzer.utils.data_structures import AggregatedData
from time_chart_tool.models import ActivityEvent

class TestSingleFileInGlobMode(unittest.TestCase):
    def setUp(self):
        self.agg_data1 = AggregatedData(
            cpu_events=[ActivityEvent(name="op1", ts=100, dur=10, args={}, cat="cpu_op", ph="X", pid=1, tid=1)],
            kernel_events=[],
            key="op1"
        )
        # 模拟 glob 模式下解析到的单个文件数据
        self.multiple_files_data_single = {
            "file1": {"op1": self.agg_data1}
        }

    def test_stage3_returns_single_file_structure(self):
        # 测试 stage3 在接收到单个文件数据时，返回 {'single_file': ...} 结构
        result = stage3_comparison(multiple_files_data=self.multiple_files_data_single)
        
        self.assertIn("single_file", result)
        self.assertEqual(result["single_file"], {"op1": self.agg_data1})
        
    def test_stage4_handles_single_file_structure(self):
        # 测试 stage4 在接收到 {'single_file': ...} 结构时，正确调用单文件展示逻辑
        comparison_result = {"single_file": {"op1": self.agg_data1}}
        
        # 我们不能真正生成文件，所以这里主要验证代码路径是否通畅
        # 并且没有报错
        # 为了避免文件生成，我们可以 mock _generate_output_files 或者提供临时目录
        # 这里我们简单地运行，捕获可能的异常，因为 stage4 是纯函数，除了文件写入没有副作用
        
        # 注意：stage4_presentation 内部会调用 _present_single_file
        # 我们只要确保它不返回空列表（如果有数据的话），或者不抛出异常
        
        try:
            # 使用临时目录作为输出
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                generated_files = stage4_presentation(
                    comparison_result=comparison_result,
                    output_dir=tmpdir,
                    show_attributes=['name'],
                    aggregation_spec=['name'],
                    label='test_label'
                )
                # 应该生成文件
                self.assertTrue(len(generated_files) > 0)
        except Exception as e:
            self.fail(f"stage4_presentation failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()

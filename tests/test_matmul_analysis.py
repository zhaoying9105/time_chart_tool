#!/usr/bin/env python3
"""
测试matmul算子专门分析功能
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

# 添加src目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from time_chart_tool.analyzer import Analyzer


class TestMatmulAnalysis:
    """测试matmul分析功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.analyzer = Analyzer()
        
        # 创建测试数据
        self.test_comparison_data = [
            {
                "cpu_op_name": "aten::mm",
                "cpu_op_input_strides": "((3, 1), (1, 3))",
                "cpu_op_input_dims": "((2048, 3), (3, 32))",
                "fp32_input_types": "('float', 'float')",
                "fp32_kernel_names": "void MLUUnion1KernelGemmRb<float, float, float, float, float, float>(void*, void*, void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, float, float, bool, bool, bool)",
                "fp32_kernel_count": 15,
                "fp32_kernel_mean_duration": 2.9066666666666667,
                "op_bf16_input_types": "('c10::BFloat16', 'c10::BFloat16')",
                "op_bf16_kernel_names": "void MLUUnion1KernelGemmRb<__bang_bfloat16, __bang_bfloat16, __bang_bfloat16, __bang_bfloat16, __bang_bfloat16, float>(void*, void*, void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, float, float, bool, bool, bool)",
                "op_bf16_kernel_count": 15,
                "op_bf16_kernel_mean_duration": 2.5893333333333333,
                "kernel_names_equal": False,
                "kernel_count_equal": True,
                "op_bf16_ratio_to_fp32": 0.8908256880733945
            },
            {
                "cpu_op_name": "aten::mm",
                "cpu_op_input_strides": "((1, 32), (32, 1))",
                "cpu_op_input_dims": "((1024, 32), (32, 64))",
                "fp32_input_types": "('float', 'float')",
                "fp32_kernel_names": "void MLUUnion1KernelGemmRb<float, float, float, float, float, float>(void*, void*, void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, float, float, bool, bool, bool)",
                "fp32_kernel_count": 20,
                "fp32_kernel_mean_duration": 3.5,
                "op_bf16_input_types": "('c10::BFloat16', 'c10::BFloat16')",
                "op_bf16_kernel_names": "void MLUUnion1KernelGemmRb<__bang_bfloat16, __bang_bfloat16, __bang_bfloat16, __bang_bfloat16, __bang_bfloat16, float>(void*, void*, void*, void*, float, float, int, int, int, int, int, int, int, int, int, int, float, float, bool, bool, bool)",
                "op_bf16_kernel_count": 20,
                "op_bf16_kernel_mean_duration": 3.1,
                "kernel_names_equal": False,
                "kernel_count_equal": True,
                "op_bf16_ratio_to_fp32": 0.8857142857142857
            },
            {
                "cpu_op_name": "aten::add",
                "cpu_op_input_strides": "((1,), (1,))",
                "cpu_op_input_dims": "((1024,), (1024,))",
                "fp32_input_types": "('float', 'float')",
                "fp32_kernel_count": 10,
                "fp32_kernel_mean_duration": 1.0,
                "op_bf16_input_types": "('c10::BFloat16', 'c10::BFloat16')",
                "op_bf16_kernel_count": 10,
                "op_bf16_kernel_mean_duration": 0.9,
                "op_bf16_ratio_to_fp32": 0.9
            }
        ]
    
    def test_extract_matmul_dimensions(self):
        """测试matmul维度提取功能"""
        # 测试正常情况
        input_dims = "((2048, 3), (3, 32))"
        result = self.analyzer.extract_matmul_dimensions(input_dims)
        assert result == (2048, 3, 32)
        
        # 测试另一个正常情况
        input_dims = "((1024, 32), (32, 64))"
        result = self.analyzer.extract_matmul_dimensions(input_dims)
        assert result == (1024, 32, 64)
        
        # 测试无效格式
        input_dims = "((1024, 32), (33, 64))"  # k1 != k2
        result = self.analyzer.extract_matmul_dimensions(input_dims)
        assert result is None
        
        # 测试非matmul格式
        input_dims = "((1024,), (1024,))"
        result = self.analyzer.extract_matmul_dimensions(input_dims)
        assert result is None
    
    def test_analyze_matmul_by_min_dim(self):
        """测试matmul按最小维度分组分析"""
        matmul_data = self.analyzer.analyze_matmul_by_min_dim(self.test_comparison_data)
        
        # 应该有两个matmul条目（min_dim=3和min_dim=32，因为这两个在所有标签中都有数据）
        assert len(matmul_data) == 2
        
        # 检查min_dim=3的数据
        assert 3 in matmul_data
        assert len(matmul_data[3]) == 1
        
        # 检查min_dim=32的数据
        assert 32 in matmul_data
        assert len(matmul_data[32]) == 1
        
        # 验证数据内容
        entry_3 = matmul_data[3][0]
        assert len(entry_3) == 6  # 2个标签 * 3个字段
        
        # 验证fp32数据
        assert entry_3[0] == "('float', 'float')"  # fp32_input_types
        assert entry_3[1] == 15  # fp32_kernel_count
        assert entry_3[2] == 2.9066666666666667  # fp32_kernel_mean_duration
    
    def test_generate_matmul_analysis(self):
        """测试matmul分析生成功能"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 生成matmul分析
            self.analyzer.generate_matmul_analysis(self.test_comparison_data, temp_dir)
            
            # 检查生成的文件
            json_file = os.path.join(temp_dir, "matmul_analysis.json")
            xlsx_file = os.path.join(temp_dir, "matmul_analysis.xlsx")
            chart_file = os.path.join(temp_dir, "matmul_performance_chart.jpg")
            
            # 检查JSON文件
            assert os.path.exists(json_file)
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证JSON数据结构
            assert len(data) == 2  # 两个matmul条目
            
            # 检查第一个条目
            first_entry = data[0]
            assert 'mm_min_dim' in first_entry
            assert 'fp32_input_types' in first_entry
            assert 'fp32_kernel_count' in first_entry
            assert 'fp32_kernel_mean_duration' in first_entry
            assert 'op_bf16_input_types' in first_entry
            assert 'op_bf16_kernel_count' in first_entry
            assert 'op_bf16_kernel_mean_duration' in first_entry
            assert 'op_bf16_ratio_to_fp32' in first_entry
            
            # 检查Excel文件
            if os.path.exists(xlsx_file):
                # 如果openpyxl可用，Excel文件应该存在
                assert True
            else:
                # 否则应该有CSV文件
                csv_file = os.path.join(temp_dir, "matmul_analysis.csv")
                assert os.path.exists(csv_file)
            
            # 检查图表文件
            assert os.path.exists(chart_file)
    
    def test_generate_matmul_chart(self):
        """测试matmul图表生成功能"""
        # 准备测试数据
        json_data = [
            {
                "mm_min_dim": 3,
                "fp32_kernel_mean_duration": 2.9,
                "op_bf16_kernel_mean_duration": 2.6,
                "op_bf16_ratio_to_fp32": 0.89
            },
            {
                "mm_min_dim": 32,
                "fp32_kernel_mean_duration": 3.5,
                "op_bf16_kernel_mean_duration": 3.1,
                "op_bf16_ratio_to_fp32": 0.89
            }
        ]
        
        labels = ["fp32", "op_bf16"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 生成图表
            self.analyzer.generate_matmul_chart(json_data, labels, temp_dir)
            
            # 检查图表文件
            chart_file = os.path.join(temp_dir, "matmul_performance_chart.jpg")
            assert os.path.exists(chart_file)
            
            # 检查文件大小（确保不是空文件）
            assert os.path.getsize(chart_file) > 0

    def test_analyze_matmul_data_completeness(self):
        """测试matmul数据完整性检查逻辑"""
        # 创建包含缺失数据的测试数据
        incomplete_data = [
            {
                "cpu_op_name": "aten::mm",
                "cpu_op_input_strides": "((3, 1), (1, 3))",
                "cpu_op_input_dims": "((2048, 3), (3, 32))",
                "fp32_input_types": "('float', 'float')",
                "fp32_kernel_count": 15,
                "fp32_kernel_mean_duration": 2.9066666666666667,
                "op_bf16_input_types": "('c10::BFloat16', 'c10::BFloat16')",
                "op_bf16_kernel_count": 15,
                "op_bf16_kernel_mean_duration": 2.5893333333333333,
            },
            {
                "cpu_op_name": "aten::mm",
                "cpu_op_input_strides": "((1, 32), (32, 1))",
                "cpu_op_input_dims": "((1024, 32), (32, 64))",
                "fp32_input_types": "('float', 'float')",
                "fp32_kernel_count": 20,
                "fp32_kernel_mean_duration": 3.5,
                # 注意：这个条目缺少op_bf16的数据
            },
            {
                "cpu_op_name": "aten::mm",
                "cpu_op_input_strides": "((1, 64), (64, 1))",
                "cpu_op_input_dims": "((512, 64), (64, 128))",
                "fp32_input_types": "('float', 'float')",
                "fp32_kernel_count": 25,
                "fp32_kernel_mean_duration": 4.2,
                "op_bf16_input_types": "('c10::BFloat16', 'c10::BFloat16')",
                "op_bf16_kernel_count": 25,
                "op_bf16_kernel_mean_duration": 3.8,
            }
        ]
        
        matmul_data = self.analyzer.analyze_matmul_by_min_dim(incomplete_data)
        
        # 应该只有两个条目：min_dim=3（完整数据）和min_dim=64（完整数据）
        # min_dim=32应该被过滤掉，因为缺少op_bf16数据
        assert len(matmul_data) == 2
        
        # 检查min_dim=3的数据（完整）
        assert 3 in matmul_data
        
        # 检查min_dim=64的数据（完整）
        assert 64 in matmul_data
        
        # 检查min_dim=32的数据（不完整，应该被过滤）
        assert 32 not in matmul_data


if __name__ == "__main__":
    pytest.main([__file__])

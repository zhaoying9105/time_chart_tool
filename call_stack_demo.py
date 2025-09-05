#!/usr/bin/env python3
"""
Call Stack Demo - 演示 call stack 功能的简单模型和 profiler 测试
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import json
import os
from pathlib import Path


class CustomLinear(nn.Module):
    """自定义线性层，用于产生 call stack"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # 这里会产生 call stack
        x = self.linear(x)
        x = self.activation(x)
        return x


class CustomBlock(nn.Module):
    """自定义块，包含多个自定义层"""
    
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layer1 = CustomLinear(in_features, hidden_features)
        self.layer2 = CustomLinear(hidden_features, hidden_features)
        self.layer3 = CustomLinear(hidden_features, out_features)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x


class DemoModel(nn.Module):
    """演示模型，包含嵌套的自定义模块"""
    
    def __init__(self, input_size=128, hidden_size=256, output_size=64):
        super().__init__()
        self.input_projection = CustomLinear(input_size, hidden_size)
        self.block1 = CustomBlock(hidden_size, hidden_size, hidden_size)
        self.block2 = CustomBlock(hidden_size, hidden_size, hidden_size)
        self.output_projection = CustomLinear(hidden_size, output_size)
        self.final_activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.output_projection(x)
        x = self.final_activation(x)
        return x


def create_profiler_config(with_stack=True):
    """创建 profiler 配置"""
    config = {
        'activities': [torch.profiler.ProfilerActivity.CPU],
        'schedule': torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        'record_shapes': True,
        'profile_memory': True,
        'with_flops': True,
        'with_modules': True,
    }
    
    if with_stack:
        config['with_stack'] = True
        config['experimental_config'] = torch.profiler._ExperimentalConfig(verbose=True)
    
    return config


def train_step(model, optimizer, criterion, data, target, use_autocast=False):
    """执行一个训练步骤"""
    optimizer.zero_grad()
    
    if use_autocast:
        with torch.autocast(device_type='cpu', dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target)
    else:
        output = model(data)
        loss = criterion(output, target)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def run_profiler_demo(model_name, use_autocast=False, output_dir="profiler_outputs"):
    """运行 profiler 演示"""
    print(f"=== 运行 {model_name} 演示 ===")
    print(f"使用 autocast: {use_autocast}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型和数据
    model = DemoModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 创建随机数据
    batch_size = 32
    input_size = 128
    output_size = 64
    
    data = torch.randn(batch_size, input_size)
    target = torch.randn(batch_size, output_size)
    
    # 创建 profiler 配置
    profiler_config = create_profiler_config(with_stack=True)
    
    # 运行 profiler
    with torch.profiler.profile(**profiler_config) as prof:
        for step in range(3):
            loss = train_step(model, optimizer, criterion, data, target, use_autocast)
            print(f"Step {step + 1}, Loss: {loss:.4f}")
            prof.step()
    
    # 导出 trace
    trace_file = os.path.join(output_dir, f"{model_name}_trace.json")
    prof.export_chrome_trace(trace_file)
    print(f"Trace 文件已保存: {trace_file}")
    
    return trace_file


def main():
    """主函数"""
    print("=== Call Stack Demo 开始 ===")
    
    # 运行普通模式
    normal_trace = run_profiler_demo("normal", use_autocast=False)
    
    # 运行 autocast fp16 模式
    autocast_trace = run_profiler_demo("autocast_fp16", use_autocast=True)
    
    print("=== Call Stack Demo 完成 ===")
    print(f"普通模式 trace: {normal_trace}")
    print(f"Autocast FP16 模式 trace: {autocast_trace}")
    
    return normal_trace, autocast_trace


if __name__ == "__main__":
    main()

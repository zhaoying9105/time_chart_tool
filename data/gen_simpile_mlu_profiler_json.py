import torch
import torch.nn as nn
import torch.optim as optim
import torch_mlu
import torch.distributed as dist
import os
import argparse

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # 深度可分离卷积 (覆盖 Group Conv)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pointwise = nn.Conv2d(32, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # LayerNorm
        self.ln = nn.LayerNorm([64, 16, 16])
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 2: Depthwise Separable Conv -> BN -> ReLU
        identity = x
        x = self.conv2(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Element-wise Add (Residual connection needs channel match, here we just add a resized identity for demo or skip)
        # Since channels changed 32->64, we can't direct add. Let's just proceed.
        
        # LayerNorm
        x = self.ln(x)
        
        # Flatten -> FC -> Dropout -> FC
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def train_step(model, criterion, optimizer, inputs, labels):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss

def run_communication_ops(rank, world_size, device):
    """运行通信算子：allreduce, all2all, broadcast"""
    tensor_size = 1024 * 1024
    
    # 1. AllReduce
    tensor = torch.ones(tensor_size, device=device) * (rank + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # print(f"Rank {rank} AllReduce result (first element): {tensor[0]}")
    
    # 2. Broadcast
    if rank == 0:
        tensor = torch.ones(tensor_size, device=device) * 100
    else:
        tensor = torch.zeros(tensor_size, device=device)
    dist.broadcast(tensor, src=0)
    # print(f"Rank {rank} Broadcast result (first element): {tensor[0]}")

    # 3. AllToAll
    # AllToAll 需要每个 rank 发送 world_size 个 tensor，接收 world_size 个 tensor
    # 这里简单起见，我们把 tensor 分割成 world_size 份
    input_tensor = torch.arange(world_size * 10, dtype=torch.float32, device=device) + rank * world_size * 10
    output_tensor = torch.zeros(world_size * 10, dtype=torch.float32, device=device)
    
    # scatter list (send) and gather list (receive) are not needed for all_to_all_single if tensors are contiguous and same size
    # But standard all_to_all takes lists
    
    # For simplicity using all_to_all_single if available or constructing lists
    # input_split = list(input_tensor.chunk(world_size))
    # output_split = list(output_tensor.chunk(world_size)) # This creates new tensors, not views usually writable
    
    # Let's use list of tensors for standard all_to_all
    send_list = [torch.ones(100, device=device) * (rank + i) for i in range(world_size)]
    recv_list = [torch.zeros(100, device=device) for _ in range(world_size)]
    
    dist.all_to_all(recv_list, send_list)
    # print(f"Rank {rank} AllToAll result (first element of from rank 0): {recv_list[0][0]}")


def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # 初始化分布式环境
    # 优先从环境变量获取，如果没有则尝试从 args 获取 (兼容 torch.distributed.launch 和 torchrun)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    else:
        # 默认单机单卡，或者非分布式环境
        print("Not running in distributed mode. Please use torchrun or launch script.")
        rank = 0
        world_size = 1
        local_rank = 0
        # 为了演示代码也能跑，我们初始化一个伪分布式或者直接退出
        # return

    if world_size > 1:
        dist.init_process_group(backend='cncl')
        torch.mlu.set_device(local_rank) # MLU 通常也兼容 mlu 语义，或者用 torch.mlu.set_device
        device = torch.device(f'mlu:{local_rank}')
    else:
        if torch.mlu.is_available():
            device = torch.device('mlu')
        else:
            device = torch.device('cpu')

    print(f"Rank {rank}/{world_size}, Local Rank {local_rank}, Device: {device}")

    model = ComplexModel().to(device)
    
    # 分布式模型 (DDP)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 模拟输入数据 (Batch Size 8, 3 Channels, 32x32 Image)
    inputs = torch.randn(8, 3, 32, 32).to(device)
    labels = torch.randint(0, 10, (8,)).to(device)

    # 配置 Profiler Activities
    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.profiler.ProfilerActivity, 'MLU'):
        activities.append(torch.profiler.ProfilerActivity.MLU)
    
    # 输出文件路径
    output_json = f"./mlu_profiler_trace_rank_{rank}.json"
    
    print("Starting profiling...")

    # 使用 torch.profiler.profile
    # 只运行 1 个 step，所以 wait=0, warmup=0, active=1
    
    # 自定义 trace handler 以确保每个 rank 保存不同的文件，并且避免重复保存错误
    def trace_handler(p):
        output = output_json
        p.export_chrome_trace(output)
        print(f"Rank {rank}: Profiler trace saved to {os.path.abspath(output)}")

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True
    ) as p:
        # 运行 1 个 step
        # 在 step 内部加入通信算子
        
        # 1. 前向传播 + 反向传播 + 优化
        loss = train_step(model, criterion, optimizer, inputs, labels)
        print(f"Rank {rank} Loss: {loss.item()}")
        
        # 2. 显式运行通信算子 (为了 Profiler 能抓到显式的通信 Op)
        if world_size > 1:
            run_communication_ops(rank, world_size, device)
            
        p.step()

    # 导出 Chrome Trace JSON (on_trace_ready 已经处理了保存，这里不需要再显式调用 export_chrome_trace，否则会报错 RuntimeError: Trace is already saved)
    # p.export_chrome_trace(output_json)

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

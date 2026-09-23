import torch
import time
import numpy as np
from model import LumbarCNNMamba


def main():
    # 自动检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型并切换到评估模式（极其重要：关闭 Dropout，固定 BN）
    model = LumbarCNNMamba(input_channels=4, d_model=128, dropout_rate=0.1).to(device).float()
    model.eval()

    # 模拟边缘计算的实时环境：每次只接收并处理 1 个窗口的数据 (Batch Size = 1)
    # 输入形状：[1, 30, 4] -> 对应 [batch, window_size, channels]
    dummy_input = torch.randn(1, 30, 4).to(device, dtype=torch.float32)

    print(f"当前测试设备: {device}")
    print("正在预热模型 (消除 CUDA 初始化和显存分配的冷启动延迟)...")

    # 1. 预热 (Warm-up)
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    # 2. 正式测速
    num_iterations = 1000
    latencies = []

    print(f"开始进行 {num_iterations} 次连续前向推理测速...")
    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == 'cuda':
                # CUDA 测速必须使用专门的 Event，否则异步机制会导致时间测量偏小
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()  # 阻塞等待 GPU 执行完毕
                curr_time = starter.elapsed_time(ender)  # 单位：毫秒
            else:
                start_time = time.perf_counter()
                _ = model(dummy_input)
                curr_time = (time.perf_counter() - start_time) * 1000  # 单位：毫秒

            latencies.append(curr_time)

    # 3. 统计结果
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p99_latency = np.percentile(latencies, 99)

    print("\n" + "=" * 50)
    print(" 🚀 CNN-Mamba 实时推理延迟报告 (单次前向传播)")
    print("=" * 50)
    print(f"  - 输入张量: [1, 30, 4] (模拟单次 300ms 窗口传入)")
    print(f"  - 平均延迟 (Average): {avg_latency:.4f} 毫秒")
    print(f"  - 延迟波动 (Std Dev): {std_latency:.4f} 毫秒")
    print(f"  - P99 延迟 (最差情况):{p99_latency:.4f} 毫秒")
    print("-" * 50)

    # 4. 换算 1 秒钟连续运动的解算时间
    # 假设控制系统的更新频率为 100 Hz（即 1 秒钟进行 100 次实时预测）
    time_per_second_motion = avg_latency * 100
    print(f"  💡 估算完成 1 秒钟运动 (100 次预测) 的总耗时: {time_per_second_motion:.2f} 毫秒")
    print("=" * 50)


if __name__ == "__main__":
    main()
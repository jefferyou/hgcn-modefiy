import torch
import sys


def test_cuda():
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"当前设备数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
            print(f"设备 {i} 显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")

        # 创建一个测试张量并移动到GPU
        print("\n进行简单的GPU测试...")
        x = torch.rand(1000, 1000)
        gpu_x = x.cuda()
        result = gpu_x @ gpu_x.t()
        print(f"矩阵乘法结果形状: {result.shape}")
        print("GPU测试成功!")
    else:
        print("CUDA不可用，请检查PyTorch安装或GPU驱动")


if __name__ == "__main__":
    test_cuda()
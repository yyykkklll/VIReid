import torch
import sys

def check_installation():
    print("="*40)
    print("Checking Environment Configuration...")
    print("="*40)

    # 1. 检查 PyTorch 和 CUDA
    print(f"[PyTorch] Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"[CUDA] Available: Yes")
        print(f"[CUDA] Device Name: {torch.cuda.get_device_name(0)}")
        print(f"[CUDA] Version: {torch.version.cuda}")
    else:
        print("[CUDA] Available: NO (Mamba usually requires CUDA)")
        # Mamba 官方主要支持 CUDA，如果没有 GPU 可能会报错或非常慢
    
    print("-" * 20)

    # 2. 检查 Causal Conv1d (Mamba 的核心依赖)
    try:
        import causal_conv1d
        print(f"[Causal Conv1d] Import: Success")
        print(f"[Causal Conv1d] Version: {causal_conv1d.__version__}")
    except ImportError:
        print("[Causal Conv1d] Import: FAILED")
        print("  -> 请尝试: pip install causal-conv1d>=1.2.0")
    except Exception as e:
        print(f"[Causal Conv1d] Error: {e}")

    print("-" * 20)

    # 3. 检查 Mamba SSM 并运行简单测试
    try:
        from mamba_ssm import Mamba
        print("[Mamba SSM] Import: Success")
        
        # 只有在 CUDA 可用时才进行前向测试
        if cuda_available:
            print("Running simple Mamba forward pass test...")
            device = "cuda"
            batch, length, dim = 2, 64, 16
            
            # 定义一个极简的 Mamba 模块
            model = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            ).to(device)
            
            x = torch.randn(batch, length, dim).to(device)
            y = model(x)
            
            assert y.shape == x.shape
            print(f"[Test] Forward pass successful! Output shape: {y.shape}")
        else:
            print("[Test] Skipped forward pass (CUDA not available)")
            
    except ImportError:
        print("[Mamba SSM] Import: FAILED")
        print("  -> 请尝试: pip install mamba-ssm")
    except Exception as e:
        print(f"[Mamba SSM] Error: {e}")
        print("  -> 常见原因: PyTorch 版本与 CUDA 版本不匹配，或者未安装 causal-conv1d")

    print("="*40)

if __name__ == "__main__":
    check_installation()


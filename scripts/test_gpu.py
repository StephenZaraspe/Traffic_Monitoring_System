"""
GPU Verification Script
Run this first to ensure CUDA is working
"""

import torch
import sys

def check_gpu():
    print("="*60)
    print("GPU CONFIGURATION CHECK")
    print("="*60)
    
    # Python version
    print(f"\nPython version: {sys.version}")
    
    # PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Unknown'}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        # GPU details
        gpu_count = torch.cuda.device_count()
        print(f"\nNumber of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Test GPU computation
        print("\nTesting GPU computation...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU computation successful!")
        
        print("\n" + "="*60)
        print("✓ SYSTEM READY FOR TRAINING")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ WARNING: CUDA NOT AVAILABLE")
        print("Training will run on CPU (very slow)")
        print("="*60)
        print("\nTroubleshooting:")
        print("1. Verify NVIDIA GPU is installed")
        print("2. Install CUDA 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive")
        print("3. Reinstall PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == '__main__':
    check_gpu()
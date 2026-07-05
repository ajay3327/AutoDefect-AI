"""
PyTorch Verification Script.
"""
import sys

def verify_pytorch():
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch is not installed!")
        sys.exit(1)

    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics is not installed!")
        sys.exit(1)

if __name__ == "__main__":
    verify_pytorch()

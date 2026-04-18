import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")


if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

if not torch.cuda.is_available():
    import os
    print("\n--- Diagnostic Check ---")
    print(f"Is CUDA_PATH set in Environment? {os.environ.get('CUDA_PATH')}")
    # This checks if the version string has '+cu', which means it's the GPU version
    if "+cu" not in torch.__version__:
        print("ERROR: You have the CPU-only version of PyTorch installed.")
    else:
        print("ERROR: You have the GPU version, but it can't talk to your driver.")
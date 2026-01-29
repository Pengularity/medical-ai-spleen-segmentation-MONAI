import torch
import monai

print("--- System diagnostic ---")
print(f"PyTorch version: {torch.__version__}")
print(f"MONAI version: {monai.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING : CUDA is not detected. Check your NVIDIA drivers.")
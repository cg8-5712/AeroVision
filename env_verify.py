import torch
import timm
from ultralytics import YOLO

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"timm: {timm.__version__}")

# 测试模型加载
model = timm.create_model("convnext_base", pretrained=True)
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(f"ConvNeXt output shape: {y.shape}")
print("Environment OK!")
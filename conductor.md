# QuanPhotos AI 模型训练路线图

> 本文档是自训练航空照片识别系统的完整实施指南

---

## 目录

1. [项目目标](#项目目标)
2. [环境配置](#环境配置)
3. [数据规范](#数据规范)
4. [目录结构](#目录结构)
5. [训练阶段](#训练阶段)
6. [评估指标](#评估指标)
7. [常见问题](#常见问题)

---

## 项目目标

构建一个航空照片视觉识别系统，实现以下能力：

| 任务 | 输入 | 输出 | 优先级 |
|------|------|------|--------|
| 机型分类 | 飞机图片 | Boeing 737-800 等 | P0 |
| 航司识别 | 飞机图片 | China Eastern 等 | P1 |
| 清晰度评估 | 飞机图片 | 0-1 分数 | P1 |
| 注册号识别 | 飞机图片 | B-1234 等字符串 | P2 |
| 置信度输出 | 所有预测 | 可信度分数 | P2 |

---

## 环境配置

### 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|----------|----------|
| GPU | RTX 3060 12GB | RTX 4090 24GB |
| 内存 | 16GB | 32GB+ |
| 硬盘 | 100GB SSD | 500GB NVMe |
| CUDA | 11.8+ | 12.1+ |

### 软件环境

```bash
# 1. 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装训练相关包
pip install timm==1.0.3           # 预训练模型库
pip install ultralytics==8.1.0    # YOLOv8
pip install albumentations==1.4.0 # 数据增强
pip install wandb==0.16.0         # 实验追踪
pip install tensorboard==2.15.0   # 可视化

# 4. 安装 OCR 相关（阶段 6 使用）
pip install paddlepaddle-gpu
pip install paddleocr
```

### 验证安装

```python
# verify_env.py
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
```

---

## 数据规范

### 数据来源

| 来源 | 网址 | 特点 |
|------|------|------|
| JetPhotos | jetphotos.com | 高质量、有机型标注 |
| Planespotters | planespotters.net | 注册号数据丰富 |
| Airliners.net | airliners.net | 数据量大 |
| Flickr | flickr.com | 需筛选 |

### 数据采集规范

```
每个机型目标数据量：
├── 训练集: 300-500 张
├── 验证集: 50-100 张
└── 测试集: 50-100 张

初期建议机型（10类）:
├── Boeing: 737-800, 747-400, 777-300ER, 787-9
├── Airbus: A320, A330-300, A350-900, A380
└── Others: ARJ21, C919
```

### 标注格式

#### 机型+航司标注 (CSV)

```csv
# labels/aircraft_labels.csv
filename,type_id,type_name,airline_id,airline_name,quality_score
A320_0001.jpg,0,A320,5,China Eastern,1.0
B737_0002.jpg,1,B737-800,3,Air China,0.85
A380_0003.jpg,7,A380,12,Emirates,1.0
B787_0004.jpg,4,B787-9,,,,0.7
```

#### 类别映射 (JSON)

```json
// labels/type_classes.json
{
  "classes": [
    "A320",
    "B737-800",
    "B747-400",
    "B777-300ER",
    "B787-9",
    "A330-300",
    "A350-900",
    "A380",
    "ARJ21",
    "C919"
  ],
  "num_classes": 10
}
```

```json
// labels/airline_classes.json
{
  "classes": [
    "Air China",
    "China Eastern",
    "China Southern",
    "Hainan Airlines",
    "Xiamen Airlines",
    "Sichuan Airlines",
    "Spring Airlines",
    "Juneyao Airlines",
    "Emirates",
    "Singapore Airlines",
    "Cathay Pacific",
    "Unknown"
  ],
  "num_classes": 12
}
```

#### 注册号标注 (YOLO 格式 + OCR)

```
# labels/registration/B-1234_bbox.txt
# class x_center y_center width height
0 0.85 0.65 0.12 0.04

# labels/registration/B-1234_ocr.txt
B-1234
```

### 数据质量要求

| 检查项 | 要求 | 检查方法 |
|--------|------|----------|
| 分辨率 | >= 640x480 | 脚本批量检查 |
| 飞机占比 | >= 30% 画面 | YOLO 检测后计算 |
| 清晰度 | 无明显模糊 | Laplacian 方差 |
| 遮挡 | 主体遮挡 < 20% | 人工抽检 |
| 标注准确 | 机型正确 | 人工抽检 10% |

---

## 目录结构

```
training/
├── configs/                    # 配置文件
│   ├── base.yaml
│   ├── stage2_type.yaml
│   ├── stage3_multi.yaml
│   └── stage5_hybrid.yaml
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始图片
│   │   └── jetphotos/
│   ├── processed/              # 处理后数据
│   │   ├── aircraft_crop/      # 裁剪后的飞机图
│   │   │   ├── train/
│   │   │   │   ├── A320/
│   │   │   │   ├── B737-800/
│   │   │   │   └── ...
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── registration_crop/  # 注册号区域裁剪
│   └── labels/                 # 标注文件
│       ├── aircraft_labels.csv
│       ├── type_classes.json
│       └── airline_classes.json
│
├── src/                        # 源代码
│   ├── data/
│   │   ├── dataset.py          # Dataset 类
│   │   ├── transforms.py       # 数据增强
│   │   └── crop_aircraft.py    # 飞机裁剪脚本
│   ├── models/
│   │   ├── convnext.py         # ConvNeXt 模型
│   │   ├── hybrid.py           # 混合模型
│   │   └── heads.py            # 各任务 Head
│   ├── trainers/
│   │   ├── base_trainer.py     # 基础训练器
│   │   └── multi_task.py       # 多任务训练器
│   └── utils/
│       ├── metrics.py          # 评估指标
│       ├── checkpoint.py       # 模型保存加载
│       └── visualize.py        # 可视化
│
├── scripts/                    # 运行脚本
│   ├── prepare_data.py         # 数据准备
│   ├── train.py                # 训练入口
│   ├── evaluate.py             # 评估脚本
│   └── export.py               # 模型导出
│
├── checkpoints/                # 模型检查点
│   ├── stage2/
│   ├── stage3/
│   └── ...
│
├── logs/                       # 日志
│   ├── tensorboard/
│   └── wandb/
│
└── notebooks/                  # 实验笔记
    ├── 01_data_exploration.ipynb
    └── 02_model_analysis.ipynb
```

---

## 训练阶段

### 总体路线图

```
阶段 0 → 阶段 1 → 阶段 2 → 阶段 3 → 阶段 4 → 阶段 5 → 阶段 6 → 阶段 7
 环境     数据     单任务    多Head    清晰度    Hybrid    OCR      联合
 1天     2-4天    2-3天     2天      1-2天    2-3天    2-4天     2天
```

**铁律：任何阶段没"过关"，不准跳到下一阶段**

---

### 阶段 0：环境 & 基础认知（1 天）

#### 目标
- 环境能跑通
- 理解 backbone / head / loss 概念

#### 操作步骤

```bash
# 1. 安装依赖（见环境配置章节）
pip install torch torchvision timm ultralytics

# 2. 运行验证脚本
python verify_env.py
```

```python
# 3. 跑一个最简单的 forward
import timm
import torch

model = timm.create_model("convnext_base", pretrained=True)
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(f"Output shape: {y.shape}")  # [1, 1000]
```

#### 过关标准
- [ ] 程序能跑，无报错
- [ ] 能说清楚：backbone 提取特征，head 输出预测，loss 计算误差

#### 禁止事项
- ❌ 看 Swin
- ❌ 想 Hybrid
- ❌ 写多任务

---

### 阶段 1：数据准备 & 飞机裁剪（2-4 天）

> **80% 的失败都死在数据上**

#### 目标
获得一个"只包含飞机的图片数据集"

#### 操作步骤

**1.1 收集原始数据**

```bash
# 创建目录结构
mkdir -p data/raw/jetphotos
mkdir -p data/processed/aircraft_crop/{train,val,test}

# 目标：5-10 个机型，每类 200-500 张
```

**1.2 使用 YOLOv8 检测飞机**

```python
# src/data/crop_aircraft.py
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import os

def crop_aircraft(input_dir: str, output_dir: str, conf_threshold: float = 0.5):
    """
    使用 YOLOv8 检测并裁剪飞机

    Args:
        input_dir: 原始图片目录
        output_dir: 输出目录
        conf_threshold: 置信度阈值
    """
    model = YOLO("yolov8n.pt")

    # COCO 数据集中飞机的类别 ID 是 4
    AIRPLANE_CLASS = 4

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in input_path.glob("*.jpg"):
        results = model(str(img_file), verbose=False)

        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                if int(box.cls) == AIRPLANE_CLASS and float(box.conf) >= conf_threshold:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 扩展边界框 10%
                    img = Image.open(img_file)
                    w, h = img.size
                    pad_x = int((x2 - x1) * 0.1)
                    pad_y = int((y2 - y1) * 0.1)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w, x2 + pad_x)
                    y2 = min(h, y2 + pad_y)

                    # 裁剪并保存
                    crop = img.crop((x1, y1, x2, y2))
                    crop_name = f"{img_file.stem}_crop{i}.jpg"
                    crop.save(output_path / crop_name)

        print(f"Processed: {img_file.name}")

if __name__ == "__main__":
    crop_aircraft(
        input_dir="data/raw/jetphotos",
        output_dir="data/processed/aircraft_crop/unsorted"
    )
```

**1.3 整理并分类**

```python
# scripts/organize_data.py
import shutil
from pathlib import Path

def organize_by_type(source_dir: str, target_dir: str, labels_csv: str):
    """根据标注文件整理数据到对应类别文件夹"""
    import pandas as pd

    df = pd.read_csv(labels_csv)
    source = Path(source_dir)
    target = Path(target_dir)

    for _, row in df.iterrows():
        filename = row["filename"]
        type_name = row["type_name"]

        src_file = source / filename
        if src_file.exists():
            dst_dir = target / type_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, dst_dir / filename)

    print("Data organized!")
```

**1.4 数据质量检查**

```python
# scripts/check_data_quality.py
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

def check_quality(data_dir: str):
    """检查数据质量"""
    data_path = Path(data_dir)
    stats = {"total": 0, "small": 0, "blurry": 0}

    for img_file in data_path.rglob("*.jpg"):
        stats["total"] += 1

        img = Image.open(img_file)
        w, h = img.size

        # 检查分辨率
        if w < 640 or h < 480:
            stats["small"] += 1
            print(f"Small image: {img_file} ({w}x{h})")

        # 检查清晰度 (Laplacian 方差)
        img_cv = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(img_cv, cv2.CV_64F).var()
        if laplacian_var < 100:
            stats["blurry"] += 1
            print(f"Blurry image: {img_file} (var={laplacian_var:.1f})")

    print(f"\n=== Quality Report ===")
    print(f"Total images: {stats['total']}")
    print(f"Small images: {stats['small']} ({100*stats['small']/stats['total']:.1f}%)")
    print(f"Blurry images: {stats['blurry']} ({100*stats['blurry']/stats['total']:.1f}%)")

if __name__ == "__main__":
    check_quality("data/processed/aircraft_crop")
```

#### 过关标准
- [ ] 有 `data/processed/aircraft_crop/` 文件夹
- [ ] 每张图基本只有飞机
- [ ] 肉眼抽查 50 张，90% 以上"干净"
- [ ] 每个类别至少 200 张图片

#### 禁止事项
- ❌ 直接用原图训练
- ❌ 上来就加航司 / OCR

---

### 阶段 2：单模型 + 单任务（ConvNeXt × 机型分类）（2-3 天）

> **这是整个系统的地基**

#### 目标
让模型学会："这是一架什么机型"

#### 操作步骤

**2.1 配置文件**

```yaml
# configs/stage2_type.yaml
model:
  name: convnext_base
  pretrained: true
  num_classes: 10

data:
  train_dir: data/processed/aircraft_crop/train
  val_dir: data/processed/aircraft_crop/val
  image_size: 224
  batch_size: 32
  num_workers: 4

training:
  epochs: 30
  lr: 1e-4
  weight_decay: 1e-5
  scheduler: cosine

augmentation:
  horizontal_flip: true
  rotation: 15
  color_jitter: 0.2
```

**2.2 Dataset**

```python
# src/data/dataset.py
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class AircraftTypeDataset(Dataset):
    """阶段 2：单任务机型分类数据集"""

    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.transform = transform

        # 获取类别（子文件夹名）
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 收集所有图片
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(image_size: int = 224, is_train: bool = True):
    """获取数据变换"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
```

**2.3 Model**

```python
# src/models/convnext.py
import timm
import torch.nn as nn

class AircraftTypeClassifier(nn.Module):
    """阶段 2：ConvNeXt 机型分类器"""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            "convnext_base",
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
```

**2.4 Trainer**

```python
# src/trainers/base_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class BaseTrainer:
    """阶段 2：基础训练器"""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["epochs"]
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*correct/total:.2f}%"
            })

        return total_loss / len(self.train_loader), correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / len(self.val_loader), correct / total

    def train(self, epochs: int, save_dir: str):
        best_acc = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                }, f"{save_dir}/best_model.pth")
                print(f"Saved best model with acc: {val_acc:.4f}")
```

**2.5 训练脚本**

```python
# scripts/train_stage2.py
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.dataset import AircraftTypeDataset, get_transforms
from src.models.convnext import AircraftTypeClassifier
from src.trainers.base_trainer import BaseTrainer

def main():
    # 加载配置
    with open("configs/stage2_type.yaml") as f:
        config = yaml.safe_load(f)

    # 创建数据集
    train_dataset = AircraftTypeDataset(
        config["data"]["train_dir"],
        transform=get_transforms(config["data"]["image_size"], is_train=True)
    )
    val_dataset = AircraftTypeDataset(
        config["data"]["val_dir"],
        transform=get_transforms(config["data"]["image_size"], is_train=False)
    )

    print(f"Classes: {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )

    # 创建模型
    model = AircraftTypeClassifier(
        num_classes=len(train_dataset.classes),
        pretrained=config["model"]["pretrained"]
    )

    # 训练
    trainer = BaseTrainer(model, train_loader, val_loader, config)
    trainer.train(
        epochs=config["training"]["epochs"],
        save_dir="checkpoints/stage2"
    )

if __name__ == "__main__":
    main()
```

#### 过关标准
- [ ] loss 稳定下降（前 5 epoch 降 50%+）
- [ ] val acc 明显 > 随机（10 类应 > 30%）
- [ ] 能保存 / 加载模型并推理

#### 禁止事项
- ❌ 多任务
- ❌ Hybrid
- ❌ Swin

---

### 阶段 3：单模型 + 多 Head（机型 + 航司）（2 天）

> **第一次接触"多任务"，但不杂交模型**

#### 目标
一个模型，同时输出：机型 + 航司

#### 关键变更

**3.1 多任务模型**

```python
# src/models/multi_head.py
import timm
import torch
import torch.nn as nn

class MultiHeadClassifier(nn.Module):
    """阶段 3：多 Head 分类器"""

    def __init__(self, num_types: int, num_airlines: int, pretrained: bool = True):
        super().__init__()

        # 共享 backbone
        self.backbone = timm.create_model(
            "convnext_base",
            pretrained=pretrained,
            num_classes=0  # 移除原分类头
        )

        # 获取特征维度
        self.feature_dim = self.backbone.num_features  # 1024 for convnext_base

        # 多任务 Head
        self.type_head = nn.Linear(self.feature_dim, num_types)
        self.airline_head = nn.Linear(self.feature_dim, num_airlines)

    def forward(self, x):
        features = self.backbone(x)  # [B, 1024]

        type_logits = self.type_head(features)      # [B, num_types]
        airline_logits = self.airline_head(features) # [B, num_airlines]

        return {
            "type": type_logits,
            "airline": airline_logits
        }
```

**3.2 支持缺失标签的 Dataset**

```python
# src/data/dataset.py (添加)
class MultiTaskDataset(Dataset):
    """阶段 3：多任务数据集，支持缺失标签"""

    def __init__(self, csv_path: str, image_dir: str, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # 加载类别映射
        # ... (加载 type_classes.json 和 airline_classes.json)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 加载图片
        image = Image.open(self.image_dir / row["filename"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 构造标签和 mask
        labels = {
            "type": row["type_id"],
            "airline": row["airline_id"] if pd.notna(row["airline_id"]) else -1
        }

        task_mask = {
            "type": True,  # 机型总是有标签
            "airline": pd.notna(row["airline_id"])  # 航司可能缺失
        }

        return {
            "image": image,
            "labels": labels,
            "task_mask": task_mask
        }
```

**3.3 多任务训练器**

```python
# src/trainers/multi_task.py
class MultiTaskTrainer:
    """阶段 3：多任务训练器"""

    def __init__(self, model, train_loader, val_loader, config):
        # ... 初始化代码
        self.type_criterion = nn.CrossEntropyLoss()
        self.airline_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def compute_loss(self, outputs, batch):
        loss = 0

        # 机型 loss（总是计算）
        type_loss = self.type_criterion(outputs["type"], batch["labels"]["type"])
        loss += type_loss

        # 航司 loss（只对有标签的样本计算）
        airline_labels = batch["labels"]["airline"]
        if (airline_labels >= 0).any():
            airline_loss = self.airline_criterion(outputs["airline"], airline_labels)
            loss += airline_loss

        return loss
```

#### 过关标准
- [ ] 缺航司标签不会报错
- [ ] 两个任务都能正常训练
- [ ] 两个任务的 acc 都在合理范围

#### 禁止事项
- ❌ OCR
- ❌ 清晰度
- ❌ Swin

---

### 阶段 4：加清晰度 Head（1-2 天）

> **"免费但很有价值"的任务**

#### 目标
输出一个 0-1 的清晰度分数

#### 关键变更

**4.1 模型增加 Quality Head**

```python
# 在 MultiHeadClassifier 中添加
self.quality_head = nn.Sequential(
    nn.Linear(self.feature_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

# forward 中添加
quality_score = self.quality_head(features).squeeze(-1)  # [B]
```

**4.2 生成清晰度伪标签**

```python
# scripts/generate_quality_labels.py
import cv2
from PIL import Image, ImageFilter

def generate_quality_variants(image_path: str, output_dir: str):
    """生成不同清晰度的图片变体"""
    img = Image.open(image_path)
    stem = Path(image_path).stem

    # 原图 -> 1.0
    img.save(f"{output_dir}/{stem}_q100.jpg")

    # 轻模糊 -> 0.7
    img_blur1 = img.filter(ImageFilter.GaussianBlur(radius=1))
    img_blur1.save(f"{output_dir}/{stem}_q70.jpg")

    # 中模糊 -> 0.5
    img_blur2 = img.filter(ImageFilter.GaussianBlur(radius=2))
    img_blur2.save(f"{output_dir}/{stem}_q50.jpg")

    # 重模糊 -> 0.3
    img_blur3 = img.filter(ImageFilter.GaussianBlur(radius=4))
    img_blur3.save(f"{output_dir}/{stem}_q30.jpg")
```

**4.3 清晰度 Loss**

```python
# MSE Loss 用于回归
quality_loss = F.mse_loss(outputs["quality"], batch["labels"]["quality"])
```

#### 过关标准
- [ ] 清晰图分数 > 0.8
- [ ] 模糊图分数 < 0.5
- [ ] 分数连续、稳定，无跳变

#### 禁止事项
- ❌ 用主观打分
- ❌ 上 OCR

---

### 阶段 5：模型杂交 1.0（ConvNeXt + Swin 冻结）（2-3 天）

> **第一次真正"杂交"**

#### 目标
引入 Swin 的全局信息，不破坏原系统

#### 关键变更

**5.1 Hybrid 模型**

```python
# src/models/hybrid.py
import timm
import torch
import torch.nn as nn

class HybridClassifier(nn.Module):
    """阶段 5：ConvNeXt + Swin 混合模型"""

    def __init__(self, num_types: int, num_airlines: int):
        super().__init__()

        # ConvNeXt（可训练）
        self.convnext = timm.create_model(
            "convnext_base",
            pretrained=True,
            num_classes=0
        )

        # Swin（冻结）
        self.swin = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )
        self._freeze_swin()

        # 融合后的特征维度
        conv_dim = self.convnext.num_features  # 1024
        swin_dim = self.swin.num_features       # 1024
        fused_dim = conv_dim + swin_dim         # 2048

        # 多任务 Head（接融合特征）
        self.type_head = nn.Linear(fused_dim, num_types)
        self.airline_head = nn.Linear(fused_dim, num_airlines)
        self.quality_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def _freeze_swin(self):
        """冻结 Swin 所有参数"""
        for param in self.swin.parameters():
            param.requires_grad = False
        self.swin.eval()

    def forward(self, x):
        # 提取特征
        conv_feat = self.convnext(x)  # [B, 1024]

        with torch.no_grad():
            swin_feat = self.swin(x)  # [B, 1024]

        # 特征级融合（concat）
        fused = torch.cat([conv_feat, swin_feat], dim=1)  # [B, 2048]

        return {
            "type": self.type_head(fused),
            "airline": self.airline_head(fused),
            "quality": self.quality_head(fused).squeeze(-1)
        }

    def train(self, mode=True):
        """重写 train 方法，确保 Swin 始终在 eval 模式"""
        super().train(mode)
        self.swin.eval()  # Swin 始终 eval
        return self
```

#### 过关标准
- [ ] loss 不炸（不出现 NaN）
- [ ] 性能 >= 不用 Swin 的版本
- [ ] 显存可控（< 20GB）

#### 禁止事项
- ❌ Feature map 级融合
- ❌ 解冻 Swin
- ❌ 加 attention 模块

---

### 阶段 6：注册号 OCR（独立模块）（2-4 天）

> **永远不要一开始就端到端**

#### 目标
从图片中读出注册号字符串

#### 操作步骤

**6.1 注册号区域检测（微调 YOLOv8）**

```python
# 训练一个检测注册号区域的 YOLO
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data/registration_detection.yaml",
    epochs=50,
    imgsz=640
)
```

**6.2 OCR 识别**

```python
# src/services/registration_ocr.py
from paddleocr import PaddleOCR

class RegistrationOCR:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",  # 注册号主要是字母数字
            show_log=False
        )

    def recognize(self, image_path: str):
        """识别注册号"""
        result = self.ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            return None, 0.0

        # 取置信度最高的结果
        best_result = max(result[0], key=lambda x: x[1][1])
        text = best_result[1][0]
        confidence = best_result[1][1]

        # 后处理：注册号格式校验
        text = self._post_process(text)

        return text, confidence

    def _post_process(self, text: str) -> str:
        """后处理：清理和格式化"""
        import re
        # 移除空格，转大写
        text = text.replace(" ", "").upper()
        # 只保留字母数字和连字符
        text = re.sub(r"[^A-Z0-9\-]", "", text)
        return text
```

**6.3 完整 Pipeline**

```python
# src/services/registration_pipeline.py
class RegistrationPipeline:
    def __init__(self):
        self.detector = YOLO("checkpoints/reg_detector/best.pt")
        self.ocr = RegistrationOCR()

    def process(self, image_path: str):
        # 1. 检测注册号区域
        results = self.detector(image_path)

        if len(results[0].boxes) == 0:
            return {"detected": False, "value": None, "confidence": 0}

        # 2. 裁剪区域
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        img = Image.open(image_path)
        crop = img.crop((x1, y1, x2, y2))
        crop_path = "/tmp/reg_crop.jpg"
        crop.save(crop_path)

        # 3. OCR 识别
        text, confidence = self.ocr.recognize(crop_path)

        return {
            "detected": True,
            "value": text,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        }
```

#### 过关标准
- [ ] 清晰图能正确识别（准确率 > 80%）
- [ ] 有字符级 confidence
- [ ] 模块独立，可单独测试

#### 禁止事项
- ❌ 自己写 OCR
- ❌ 和 backbone 强耦合

---

### 阶段 7：全系统联合 & 可信度（2 天）

#### 目标
输出结构化结果 + 可信度

#### 关键变更

**7.1 置信度校准（Temperature Scaling）**

```python
# src/utils/calibration.py
import torch
import torch.nn as nn

class TemperatureScaling(nn.Module):
    """温度缩放校准"""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, logits, labels, lr=0.01, epochs=100):
        """在验证集上学习温度参数"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=epochs)

        def eval():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
```

**7.2 综合置信度计算**

```python
# src/services/confidence.py
def compute_final_confidence(predictions: dict) -> dict:
    """计算最终置信度"""

    # 各任务置信度
    type_conf = predictions["type"]["confidence"]
    airline_conf = predictions["airline"]["confidence"]
    reg_conf = predictions["registration"]["confidence"]
    quality = predictions["quality"]["score"]

    # 综合置信度 = 任务置信度 × 清晰度
    results = {
        "type": {
            "value": predictions["type"]["value"],
            "confidence": type_conf,
            "final_confidence": type_conf * quality
        },
        "airline": {
            "value": predictions["airline"]["value"],
            "confidence": airline_conf,
            "final_confidence": airline_conf * quality
        },
        "registration": {
            "value": predictions["registration"]["value"],
            "confidence": reg_conf,
            "final_confidence": reg_conf * quality
        },
        "quality": quality,
        "overall_confidence": min(type_conf, airline_conf) * quality
    }

    # 是否可信（阈值判断）
    results["is_reliable"] = results["overall_confidence"] > 0.7

    return results
```

**7.3 完整推理 Pipeline**

```python
# src/services/inference.py
class AircraftAnalyzer:
    """完整的飞机分析 Pipeline"""

    def __init__(self, model_path: str, reg_pipeline: RegistrationPipeline):
        self.model = HybridClassifier.load(model_path)
        self.reg_pipeline = reg_pipeline
        self.temp_scaling = TemperatureScaling.load("checkpoints/calibration.pth")

    @torch.no_grad()
    def analyze(self, image_path: str) -> dict:
        # 1. 主模型推理
        image = self._load_image(image_path)
        outputs = self.model(image)

        # 2. 置信度校准
        type_logits = self.temp_scaling(outputs["type"])
        type_probs = F.softmax(type_logits, dim=1)
        type_conf, type_pred = type_probs.max(dim=1)

        # 3. 注册号识别
        reg_result = self.reg_pipeline.process(image_path)

        # 4. 汇总结果
        predictions = {
            "type": {
                "value": self.type_classes[type_pred.item()],
                "confidence": type_conf.item()
            },
            "airline": {...},
            "registration": reg_result,
            "quality": {"score": outputs["quality"].item()}
        }

        # 5. 计算综合置信度
        return compute_final_confidence(predictions)
```

#### 过关标准
- [ ] 低清晰度 → 自动低置信
- [ ] 可设置阈值进行拒识
- [ ] 输出结构化 JSON

---

## 评估指标

### 分类任务

| 指标 | 计算方式 | 目标值 |
|------|----------|--------|
| Top-1 Accuracy | 预测正确数 / 总数 | > 85% |
| Top-5 Accuracy | Top5 包含正确类 / 总数 | > 95% |
| Macro F1 | 各类 F1 平均 | > 0.80 |
| Per-class Accuracy | 每个类单独计算 | 各类 > 70% |

### 回归任务（清晰度）

| 指标 | 目标值 |
|------|--------|
| MAE | < 0.1 |
| RMSE | < 0.15 |
| Correlation | > 0.9 |

### OCR 任务

| 指标 | 目标值 |
|------|--------|
| 检测率 | > 90% |
| 完全正确率 | > 80% |
| 字符准确率 | > 95% |

---

## 常见问题

### Q1: loss 不降？

检查顺序：
1. 学习率是否太大/太小？
2. 数据加载是否正确？（打印几个样本看看）
3. 标签是否正确？
4. 模型是否在 train() 模式？

### Q2: 显存不够？

解决方案：
1. 减小 batch_size
2. 使用 gradient accumulation
3. 使用混合精度训练 `torch.cuda.amp`
4. 用更小的模型（convnext_small）

### Q3: 过拟合？

检查顺序：
1. 数据量是否太少？
2. 增加数据增强
3. 增加 dropout
4. 使用 early stopping
5. 减小模型复杂度

### Q4: 多任务冲突？

解决方案：
1. 检查各任务 loss 的量级是否一致
2. 使用 loss 加权
3. 使用 gradient normalization
4. 分阶段训练

---

## 三条铁律

1. **任何新东西，只加一个** —— 不要同时改多个变量
2. **任何阶段，都要能单独跑** —— 保持模块独立
3. **模型复杂度 < 系统稳定性** —— 简单能跑比复杂不跑强

---

## 下一步

当你完成所有阶段后，可以开始将模型集成到 QuanPhotos AI Service 中：

1. 导出模型为 ONNX/TorchScript
2. 集成到 FastAPI 服务
3. 添加批量推理支持
4. 部署到生产环境

---

*最后更新：2024-12*

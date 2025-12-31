# QuanPhotos AI Service

## 项目概述

QuanPhotos 是一个类似 JetPhotos 的航空摄影社区平台，本仓库为 AI 审核微服务。

### 核心功能

为上传的航空照片提供自动化审核：
- 图片质量评估
- 飞机识别与机型分类
- 注册号识别与清晰度检测
- 主体遮挡检测
- 违规内容检测

## 技术栈

- **语言**: Python 3.11+
- **Web 框架**: FastAPI
- **AI/ML**:
  - 现有 API：Claude Vision / GPT-4V / 通义千问 VL
  - 自训练模型：PyTorch + timm + YOLOv8 + PaddleOCR
- **图像处理**: Pillow, OpenCV, albumentations
- **部署**: Docker, Uvicorn

## 项目结构

```
QuanPhotos-ai/
├── app/                        # FastAPI 服务
│   ├── main.py                 # 应用入口
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── review.py       # 审核接口
│   │   │   └── health.py       # 健康检查
│   │   └── deps.py             # 依赖注入
│   ├── core/
│   │   ├── config.py           # 配置管理
│   │   └── logging.py          # 日志配置
│   ├── schemas/
│   │   ├── request.py          # 请求模型
│   │   └── response.py         # 响应模型
│   ├── services/
│   │   ├── review_service.py   # 审核主服务
│   │   ├── quality/            # 质量评估
│   │   ├── aircraft/           # 飞机识别
│   │   ├── registration/       # 注册号识别
│   │   ├── occlusion/          # 遮挡检测
│   │   └── violation/          # 违规检测
│   └── utils/
│       ├── image.py            # 图像处理工具
│       └── ai_client.py        # AI API 客户端
│
├── training/                   # 自训练模型（详见 conductor.md）
│   ├── configs/                # 训练配置
│   │   ├── stage2_type.yaml
│   │   ├── stage3_multi.yaml
│   │   └── stage5_hybrid.yaml
│   ├── data/                   # 数据目录
│   │   ├── raw/                # 原始图片
│   │   ├── processed/          # 处理后数据
│   │   │   └── aircraft_crop/  # 裁剪后的飞机图
│   │   └── labels/             # 标注文件
│   │       ├── aircraft_labels.csv
│   │       ├── type_classes.json
│   │       ├── airline_classes.json
│   │       └── registration/   # 注册号 bbox (YOLO 格式)
│   ├── src/                    # 训练源代码
│   │   ├── data/               # Dataset 类
│   │   ├── models/             # 模型定义
│   │   ├── trainers/           # 训练器
│   │   └── utils/              # 工具函数
│   ├── scripts/                # 训练脚本
│   ├── checkpoints/            # 模型检查点
│   └── logs/                   # 训练日志
│
├── annotation_server/          # 标注服务（可选）
│   ├── main.py                 # FastAPI 标注后端
│   └── static/
│       └── index.html          # Vue 3 标注前端
│
├── tests/
├── scripts/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── conductor.md                # 模型训练路线图
```

## 代码规范

### Python 风格

- 遵循 PEP 8
- 使用 type hints 类型注解
- 使用 Pydantic 进行数据验证
- 异步优先（async/await）
- 使用 Black 格式化代码
- 使用 Ruff 或 Flake8 进行代码检查

### 命名规范

- 文件名：小写下划线 `review_service.py`
- 类名：PascalCase `ReviewService`
- 函数名：小写下划线 `analyze_image`
- 常量：大写下划线 `MAX_IMAGE_SIZE`

### 项目规范

- 配置通过环境变量管理
- 敏感信息不硬编码
- 日志结构化输出
- 错误统一封装返回

## API 设计

### 审核接口

```
POST /api/v1/review
```

**请求**:
```json
{
  "image_url": "string",
  "image_base64": "string",
  "review_types": ["quality", "aircraft", "registration", "occlusion", "violation"]
}
```

**响应**:
```json
{
  "success": true,
  "review_id": "uuid",
  "results": {
    "overall_pass": false,
    "quality": {
      "pass": true,
      "score": 0.85,
      "details": {
        "sharpness": 0.90,
        "exposure": 0.80,
        "composition": 0.85
      }
    },
    "aircraft": {
      "pass": true,
      "is_aircraft": true,
      "confidence": 0.98,
      "aircraft_type": "Boeing 737-800",
      "airline": "China Eastern"
    },
    "registration": {
      "pass": false,
      "detected": true,
      "value": "B-1234",
      "confidence": 0.75,
      "clarity_score": 0.60,
      "reason": "注册号清晰度不足"
    },
    "occlusion": {
      "pass": true,
      "occlusion_percentage": 0.05,
      "details": "轻微翼尖遮挡"
    },
    "violation": {
      "pass": true,
      "has_watermark": false,
      "has_sensitive_content": false
    }
  },
  "fail_reasons": ["注册号清晰度不足"]
}
```

### 健康检查

```
GET /api/v1/health
```

## 审核模块详情

### 1. 图片质量评估 (Quality)

评估维度：
- **清晰度 (Sharpness)**: 对焦是否准确，是否有运动模糊
- **曝光 (Exposure)**: 曝光是否正确，高光/阴影细节
- **构图 (Composition)**: 主体位置，画面平衡
- **噪点 (Noise)**: ISO 噪点程度
- **色彩 (Color)**: 白平衡、色彩还原

### 2. 飞机识别 (Aircraft)

- 是否包含飞机
- 飞机类型识别（客机、货机、公务机等）
- 机型识别（Boeing 737、Airbus A320 等）
- 航空公司涂装识别

### 3. 注册号识别 (Registration)

- OCR 识别注册号
- 清晰度评分
- 遮挡/部分可见检测
- 注册号格式验证

### 4. 遮挡检测 (Occlusion)

- 飞机主体遮挡比例
- 遮挡物类型（建筑、车辆、其他飞机、围栏等）
- 关键部位遮挡（机头、机尾、发动机、注册号位置）

### 5. 违规检测 (Violation)

- 水印检测
- 其他网站 logo 检测
- 敏感/不当内容检测
- 过度后期处理检测

## AI 方案

### 第一阶段：调用现有 API

优先使用成熟的视觉 AI API：
- **Claude Vision** (Anthropic)
- **GPT-4V** (OpenAI)
- **通义千问 VL** (阿里云)
- **Gemini Pro Vision** (Google)

通过 Prompt Engineering 实现审核功能。

### 第二阶段：自训练模型

> 详细训练路线图见 `conductor.md`

针对特定场景训练专用模型：

| 任务 | 模型架构 | 优先级 |
|------|----------|--------|
| 机型分类 | ConvNeXt + Swin Hybrid | P0 |
| 航司识别 | 多 Head 共享 backbone | P1 |
| 清晰度评估 | 回归 Head (0-1) | P1 |
| 注册号检测 | YOLOv8 微调 | P2 |
| 注册号 OCR | PaddleOCR | P2 |

**训练阶段**：
1. 阶段 0：环境配置
2. 阶段 1：数据准备 & 飞机裁剪
3. 阶段 2：单模型单任务（机型分类）
4. 阶段 3：单模型多 Head（机型 + 航司）
5. 阶段 4：加清晰度 Head
6. 阶段 5：模型杂交（ConvNeXt + Swin）
7. 阶段 6：注册号 OCR（独立模块）
8. 阶段 7：全系统联合 & 置信度校准

## 数据标注格式

### 主标注文件 (CSV)

```csv
filename,type_id,type_name,airline_id,airline_name,registration,quality
IMG_0001.jpg,0,A320,1,China Eastern,B-1234,1.0
IMG_0002.jpg,1,B737-800,0,Air China,B-5678,0.9
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `filename` | string | ✅ | 图片文件名 |
| `type_id` | int | ❌ | 机型 ID（自动生成） |
| `type_name` | string | ✅ | 机型名称 |
| `airline_id` | int | ❌ | 航司 ID（自动生成） |
| `airline_name` | string | ❌ | 航司名称 |
| `registration` | string | ❌ | 注册号文字 |
| `quality` | float | ✅ | 图片质量 0.0-1.0 |

### 注册号边界框 (YOLO 格式)

```
# data/labels/registration/IMG_0001.txt
# 格式: class x_center y_center width height (归一化 0-1)
0 0.85 0.65 0.12 0.04
```

## 配置管理

环境变量：

```env
# 服务配置
PORT=8000
LOG_LEVEL=INFO

# AI API Keys
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
DASHSCOPE_API_KEY=

# 审核阈值
QUALITY_THRESHOLD=0.70
REGISTRATION_CLARITY_THRESHOLD=0.80
OCCLUSION_THRESHOLD=0.20

# 后端服务
BACKEND_CALLBACK_URL=http://backend:8080/api/v1/review/callback

# 训练配置（可选）
WANDB_API_KEY=
CUDA_VISIBLE_DEVICES=0
```

## 国际化

- 审核结果描述支持多语言
- 失败原因说明支持中文、英文
- 通过 `Accept-Language` 头或参数指定语言

## 性能考虑

- 图片预处理：压缩、裁剪超大图片
- 并发控制：限制同时处理的审核请求
- 缓存：相同图片短期内不重复审核
- 异步处理：大批量审核使用队列

## 相关文档

- `conductor.md` - 模型训练路线图（详细训练流程、代码示例）
- `annotation_server/` - 自研标注服务（可选）

## 相关仓库

- 后端：QuanPhotos-backend
- 前端：QuanPhotos-web

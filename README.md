# QuanPhotos AI Service

航空摄影社区平台 AI 审核微服务

## 简介

QuanPhotos 是一个专业的航空摄影社区平台，本仓库为 AI 审核微服务，负责对上传的航空照片进行自动化质量审核和内容识别。

## 功能特性

- **图片质量评估**：清晰度、曝光、构图、噪点
- **飞机识别**：检测是否包含飞机、机型识别、航司识别
- **注册号识别**：OCR 识别、清晰度评估
- **遮挡检测**：主体遮挡比例、关键部位遮挡
- **违规检测**：水印、敏感内容、过度后期

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| 框架 | FastAPI |
| AI API | Claude Vision / GPT-4V / 通义千问 VL |
| 图像处理 | Pillow, OpenCV |
| 部署 | Docker, Uvicorn |

## 项目结构

```
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes/
│   │       ├── review.py
│   │       └── health.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── schemas/
│   │   ├── request.py
│   │   └── response.py
│   ├── services/
│   │   ├── review_service.py
│   │   ├── quality/
│   │   ├── aircraft/
│   │   ├── registration/
│   │   ├── occlusion/
│   │   └── violation/
│   └── utils/
│       ├── image.py
│       └── ai_client.py
├── tests/
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 快速开始

### 环境要求

- Python 3.11+
- Docker & Docker Compose（可选）

### 本地开发

1. 克隆仓库

```bash
git clone https://github.com/yourname/QuanPhotos-ai.git
cd QuanPhotos-ai
```

2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填写 AI API Key 等配置
```

5. 启动服务

```bash
uvicorn app.main:app --reload --port 8000
```

服务运行在 `http://localhost:8000`

### Docker 部署

```bash
docker-compose up -d
```

## 配置说明

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `PORT` | 服务端口 | 8000 |
| `LOG_LEVEL` | 日志级别 | INFO |
| `ANTHROPIC_API_KEY` | Claude API Key | - |
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `DASHSCOPE_API_KEY` | 通义千问 API Key | - |
| `QUALITY_THRESHOLD` | 质量评分阈值 | 70 |
| `REGISTRATION_CLARITY_THRESHOLD` | 注册号清晰度阈值 | 80 |
| `OCCLUSION_THRESHOLD` | 遮挡比例阈值 (%) | 20 |

## API 文档

启动服务后访问：

```
http://localhost:8000/docs
```

### 主要接口

#### 照片审核

```
POST /api/v1/review
```

**请求示例**

```json
{
  "image_url": "https://example.com/photo.jpg",
  "review_types": ["quality", "aircraft", "registration", "occlusion", "violation"]
}
```

**响应示例**

```json
{
  "success": true,
  "review_id": "550e8400-e29b-41d4-a716-446655440000",
  "results": {
    "overall_pass": true,
    "quality": {
      "pass": true,
      "score": 85,
      "details": {
        "sharpness": 90,
        "exposure": 80,
        "composition": 85
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
      "pass": true,
      "detected": true,
      "value": "B-1234",
      "confidence": 0.95,
      "clarity_score": 88
    },
    "occlusion": {
      "pass": true,
      "occlusion_percentage": 5
    },
    "violation": {
      "pass": true,
      "has_watermark": false,
      "has_sensitive_content": false
    }
  },
  "fail_reasons": []
}
```

#### 健康检查

```
GET /api/v1/health
```

## 审核模块

### 质量评估 (Quality)

| 指标 | 说明 | 权重 |
|------|------|------|
| sharpness | 清晰度/对焦 | 30% |
| exposure | 曝光正确性 | 25% |
| composition | 构图质量 | 20% |
| noise | 噪点程度 | 15% |
| color | 色彩还原 | 10% |

### 飞机识别 (Aircraft)

- 是否包含飞机
- 飞机类别（客机/货机/公务机/军机等）
- 机型识别（Boeing 737、Airbus A320 等）
- 航空公司涂装识别

### 注册号识别 (Registration)

- OCR 文字识别
- 清晰度评分 (0-100)
- 格式验证（各国注册号格式）
- 部分遮挡检测

### 遮挡检测 (Occlusion)

- 主体遮挡比例
- 关键部位检测：机头、机尾、发动机、注册号区域
- 遮挡物类型识别

### 违规检测 (Violation)

- 水印/Logo 检测
- 敏感内容检测
- 过度后期处理检测

## AI 方案演进

**第一阶段 - 现有 API**

调用成熟视觉 AI API，通过 Prompt Engineering 实现审核：
- Claude Vision (Anthropic)
- GPT-4V (OpenAI)
- 通义千问 VL (阿里云)

**第二阶段 - 自训练模型**

针对特定场景训练专用模型：
- 飞机机型分类器
- 注册号 OCR 模型
- 航司涂装识别模型

## 开发指南

```bash
# 运行测试
pytest

# 代码格式化
black .

# 代码检查
ruff check .

# 类型检查
mypy app/
```

## 相关项目

| 项目 | 说明 |
|------|------|
| [QuanPhotos-backend](https://github.com/yourname/QuanPhotos-backend) | 后端 API 服务 |
| [QuanPhotos-web](https://github.com/yourname/QuanPhotos-web) | 前端 Web 应用 |

## License

MIT License
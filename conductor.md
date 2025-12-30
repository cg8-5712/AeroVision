# QuanPhotos AI 模型训练路线图

> 本文档是自训练航空照片识别系统的完整实施指南

---

## 目录

1. [项目目标](#项目目标)
2. [环境配置](#环境配置)
3. [数据规范](#数据规范)
4. [人工数据标注](#人工数据标注) ⭐ 新增
5. [目录结构](#目录结构)
6. [训练阶段](#训练阶段)
7. [评估指标](#评估指标)
8. [常见问题](#常见问题)

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

#### 主标注文件 (CSV)

```csv
# labels/aircraft_labels.csv
filename,type_id,type_name,airline_id,airline_name,registration,quality
IMG_0001.jpg,0,A320,1,China Eastern,B-1234,1.0
IMG_0002.jpg,1,B737-800,0,Air China,B-5678,0.9
IMG_0003.jpg,7,A380,8,Emirates,A6-EDA,1.0
IMG_0004.jpg,4,B787-9,3,Hainan Airlines,,0.7
IMG_0005.jpg,,,,,B-9999,0.3
```

#### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `filename` | string | ✅ | 图片文件名 |
| `type_id` | int | ❌ | 机型ID（自动生成） |
| `type_name` | string | ✅ | 机型名称，如 `A320`、`Unknown` |
| `airline_id` | int | ❌ | 航司ID（自动生成） |
| `airline_name` | string | ❌ | 航司名称，如 `China Eastern` |
| `registration` | string | ❌ | 注册号文字，如 `B-1234`，不可见则留空 |
| `quality` | float | ✅ | 图片质量 0.0-1.0（1.0=清晰，0.5=一般，0.3=模糊） |

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

#### 注册号标注 (YOLO 格式)

注册号位置用 YOLO 格式存储，文件名与图片对应：

```
图片: data/processed/aircraft_crop/unsorted/IMG_0001.jpg
标注: data/labels/registration/IMG_0001.txt

# IMG_0001.txt 内容 (YOLO格式: class x_center y_center width height)
0 0.85 0.65 0.12 0.04
```

**注意：**
- 文件名与图片同名，只是扩展名从 `.jpg` 改为 `.txt`
- 注册号**文字**存在 CSV 的 `registration` 列，不需要单独的 `_ocr.txt`
- 如果图片中注册号不可见，则不创建对应的 `.txt` 文件
- 一张图可以有多个注册号框（多行）

```
# 示例：IMG_0005.txt (机身有两处注册号)
0 0.25 0.55 0.10 0.03
0 0.82 0.48 0.08 0.025
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

## 人工数据标注

> **数据标注质量直接决定模型上限，这是最值得投入时间的环节**

### 标注方案选择

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Label Studio** | 开箱即用、功能完整、支持团队 | 导出需转换、定制性一般 | 快速启动、小团队 |
| **自研 Web 服务** | 完全定制、格式直出、可集成业务 | 需要开发时间 | 长期项目、特殊需求 |

---

### 方案一：Label Studio（推荐快速启动）

#### 1. 安装与启动

```bash
pip install label-studio
label-studio start --port 8080
```

浏览器打开 `http://localhost:8080`

#### 2. 创建项目配置

新建项目时，粘贴以下 XML 配置：

```xml
<View>
  <Image name="image" value="$image" zoom="true"/>

  <!-- 机型分类 -->
  <Header value="机型 Aircraft Type"/>
  <Choices name="type_name" toName="image" choice="single" required="true">
    <Choice value="A320"/><Choice value="A321"/>
    <Choice value="A330-300"/><Choice value="A350-900"/><Choice value="A380"/>
    <Choice value="B737-800"/><Choice value="B747-400"/>
    <Choice value="B777-300ER"/><Choice value="B787-9"/>
    <Choice value="ARJ21"/><Choice value="C919"/>
    <Choice value="Unknown"/>
  </Choices>

  <!-- 航司分类 -->
  <Header value="航空公司 Airline"/>
  <Choices name="airline_name" toName="image" choice="single" required="true">
    <Choice value="Air China"/><Choice value="China Eastern"/>
    <Choice value="China Southern"/><Choice value="Hainan Airlines"/>
    <Choice value="Xiamen Airlines"/><Choice value="Spring Airlines"/>
    <Choice value="Cathay Pacific"/><Choice value="Singapore Airlines"/>
    <Choice value="Emirates"/><Choice value="Other"/><Choice value="Unknown"/>
  </Choices>

  <!-- 图片质量 (滑块 0-1) -->
  <Header value="图片质量 Quality (0=模糊, 1=清晰)"/>
  <Rating name="quality" toName="image" maxRating="10" icon="star" size="medium"/>

  <!-- 注册号边界框 -->
  <Header value="注册号区域 Registration Area"/>
  <RectangleLabels name="reg_bbox" toName="image">
    <Label value="registration" background="#FF0000"/>
  </RectangleLabels>

  <!-- 注册号文字 -->
  <Header value="注册号 Registration Number"/>
  <TextArea name="registration" toName="image" placeholder="B-1234" maxSubmissions="1"/>
</View>
```

#### 3. 导入图片

```bash
# 方式1: 直接上传文件夹
# 方式2: 生成导入 JSON
python -c "
import json
from pathlib import Path
images = [{'image': f'/data/local-files/?d=images/{p.name}'}
          for p in Path('data/processed/aircraft_crop/unsorted').glob('*.jpg')]
print(json.dumps(images, indent=2))
" > import.json
```

#### 4. 导出并转换格式

```python
# scripts/convert_labelstudio.py
"""Label Studio 导出转换为训练格式"""
import json
import pandas as pd
from pathlib import Path

def convert_labelstudio_export(export_json: str, output_dir: str):
    with open(export_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_path = Path(output_dir)
    (output_path / "registration").mkdir(parents=True, exist_ok=True)

    records = []

    for item in data:
        filename = Path(item['data']['image']).name
        results = item.get('annotations', [{}])[0].get('result', [])

        record = {'filename': filename, 'type_name': '', 'airline_name': '',
                  'registration': '', 'quality': 1.0}
        bboxes = []

        for r in results:
            if r['type'] == 'choices':
                if r['from_name'] == 'type_name':
                    record['type_name'] = r['value']['choices'][0]
                elif r['from_name'] == 'airline_name':
                    record['airline_name'] = r['value']['choices'][0]
            elif r['type'] == 'rating':
                record['quality'] = r['value']['rating'] / 10.0  # 转为 0-1
            elif r['type'] == 'textarea' and r['from_name'] == 'registration':
                text = r['value']['text'][0] if r['value']['text'] else ''
                record['registration'] = text.upper().replace(' ', '')
            elif r['type'] == 'rectanglelabels':
                x, y = r['value']['x'] / 100, r['value']['y'] / 100
                w, h = r['value']['width'] / 100, r['value']['height'] / 100
                bboxes.append(f"0 {x + w/2:.6f} {y + h/2:.6f} {w:.6f} {h:.6f}")

        records.append(record)

        # 保存 YOLO 格式 bbox
        if bboxes:
            txt_path = output_path / "registration" / (Path(filename).stem + ".txt")
            txt_path.write_text('\n'.join(bboxes))

    # 生成 ID 映射
    df = pd.DataFrame(records)
    types = sorted(df['type_name'].dropna().unique())
    airlines = sorted(df['airline_name'].dropna().unique())
    df['type_id'] = df['type_name'].map({t: i for i, t in enumerate(types)})
    df['airline_id'] = df['airline_name'].map({a: i for i, a in enumerate(airlines)})

    # 保存
    df[['filename','type_id','type_name','airline_id','airline_name','registration','quality']].to_csv(
        output_path / "aircraft_labels.csv", index=False)

    print(f"✅ 转换完成: {len(records)} 条记录")

if __name__ == "__main__":
    convert_labelstudio_export("export.json", "data/labels")
```

---

### 方案二：自研 Web 标注服务

适合需要深度定制、长期使用的场景。

#### 1. 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      前端 (Vue/React)                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │图片显示  │ │分类选择  │ │画框工具  │ │文字输入  │           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
└─────────────────────────────────────────────────────────────┘
                            │ REST API
┌─────────────────────────────────────────────────────────────┐
│                    后端 (FastAPI/Flask)                      │
│  GET /api/images/next     获取下一张待标注图片                │
│  POST /api/annotations    保存标注结果                       │
│  GET /api/progress        获取标注进度                       │
│  GET /api/export          导出标注数据                       │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    存储 (SQLite/PostgreSQL)                  │
│  images 表: id, filename, status                            │
│  annotations 表: image_id, type, airline, reg, quality, bbox│
└─────────────────────────────────────────────────────────────┘
```

#### 2. 后端核心代码

```python
# annotation_server/main.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import sqlite3
import json

app = FastAPI(title="Aircraft Annotation Service")

# 配置
IMAGE_DIR = Path("data/processed/aircraft_crop/unsorted")
DB_PATH = "data/labels/annotations.db"

# 数据模型
class Annotation(BaseModel):
    filename: str
    type_name: str
    airline_name: str = ""
    registration: str = ""
    quality: float = 1.0
    bbox: list = []  # [[x_center, y_center, width, height], ...]

# 初始化数据库
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS annotations (
        filename TEXT PRIMARY KEY,
        type_name TEXT,
        airline_name TEXT,
        registration TEXT,
        quality REAL,
        bbox TEXT,
        annotated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

# 静态文件服务（图片）
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")

@app.get("/api/images/next")
def get_next_image():
    """获取下一张待标注图片"""
    conn = sqlite3.connect(DB_PATH)
    annotated = set(row[0] for row in conn.execute("SELECT filename FROM annotations"))
    conn.close()

    for img in sorted(IMAGE_DIR.glob("*.jpg")):
        if img.name not in annotated:
            return {
                "filename": img.name,
                "url": f"/images/{img.name}",
                "total": len(list(IMAGE_DIR.glob("*.jpg"))),
                "done": len(annotated)
            }
    return {"filename": None, "message": "All images annotated!"}

@app.post("/api/annotations")
def save_annotation(ann: Annotation):
    """保存标注"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''INSERT OR REPLACE INTO annotations
        (filename, type_name, airline_name, registration, quality, bbox)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (ann.filename, ann.type_name, ann.airline_name,
         ann.registration, ann.quality, json.dumps(ann.bbox)))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/api/progress")
def get_progress():
    """获取标注进度"""
    conn = sqlite3.connect(DB_PATH)
    done = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    conn.close()
    total = len(list(IMAGE_DIR.glob("*.jpg")))
    return {"done": done, "total": total, "percent": f"{100*done/total:.1f}%"}

@app.get("/api/export")
def export_annotations():
    """导出为 CSV + YOLO 格式"""
    import pandas as pd

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM annotations", conn)
    conn.close()

    # 生成 type_id 和 airline_id
    types = sorted(df['type_name'].dropna().unique())
    airlines = sorted(df['airline_name'].dropna().unique())
    df['type_id'] = df['type_name'].map({t: i for i, t in enumerate(types)})
    df['airline_id'] = df['airline_name'].map({a: i for i, a in enumerate(airlines)})

    # 保存 CSV
    output_dir = Path("data/labels")
    df[['filename','type_id','type_name','airline_id','airline_name','registration','quality']].to_csv(
        output_dir / "aircraft_labels.csv", index=False)

    # 保存 YOLO bbox
    bbox_dir = output_dir / "registration"
    bbox_dir.mkdir(exist_ok=True)
    for _, row in df.iterrows():
        bboxes = json.loads(row['bbox']) if row['bbox'] else []
        if bboxes:
            txt_path = bbox_dir / (Path(row['filename']).stem + ".txt")
            lines = [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in bboxes]
            txt_path.write_text('\n'.join(lines))

    return {"status": "exported", "count": len(df)}

@app.get("/api/config")
def get_config():
    """返回标注选项配置"""
    return {
        "types": ["A320", "A321", "A330-300", "A350-900", "A380",
                  "B737-800", "B747-400", "B777-300ER", "B787-9",
                  "ARJ21", "C919", "Unknown"],
        "airlines": ["Air China", "China Eastern", "China Southern",
                     "Hainan Airlines", "Xiamen Airlines", "Spring Airlines",
                     "Cathay Pacific", "Singapore Airlines", "Emirates",
                     "Other", "Unknown"]
    }

# 启动: uvicorn annotation_server.main:app --reload --port 8000
```

#### 3. 前端核心代码（Vue 3 示例）

```html
<!-- annotation_web/index.html -->
<!DOCTYPE html>
<html>
<head>
  <title>Aircraft Annotation Tool</title>
  <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
  <style>
    body { font-family: Arial; margin: 20px; }
    .container { display: flex; gap: 20px; }
    .image-panel { flex: 2; position: relative; }
    .image-panel img { max-width: 100%; cursor: crosshair; }
    .control-panel { flex: 1; }
    .bbox { position: absolute; border: 2px solid red; background: rgba(255,0,0,0.1); }
    select, input, button { width: 100%; padding: 8px; margin: 5px 0; }
    button { background: #4CAF50; color: white; border: none; cursor: pointer; }
    .progress { background: #e0e0e0; height: 20px; margin: 10px 0; }
    .progress-bar { background: #4CAF50; height: 100%; }
    .quality-slider { width: 100%; }
  </style>
</head>
<body>
  <div id="app">
    <h2>Aircraft Annotation Tool</h2>
    <div class="progress">
      <div class="progress-bar" :style="{width: progress.percent}"></div>
    </div>
    <p>进度: {{ progress.done }} / {{ progress.total }} ({{ progress.percent }})</p>

    <div class="container">
      <!-- 图片面板 -->
      <div class="image-panel" @mousedown="startDraw" @mousemove="drawing" @mouseup="endDraw">
        <img :src="imageUrl" ref="img" v-if="currentImage">
        <div v-for="(box, i) in bboxes" :key="i" class="bbox"
             :style="{left: box.x+'%', top: box.y+'%', width: box.w+'%', height: box.h+'%'}">
          <span style="color:red;font-size:12px">{{ i+1 }}</span>
        </div>
      </div>

      <!-- 控制面板 -->
      <div class="control-panel">
        <label>机型 Type</label>
        <select v-model="annotation.type_name">
          <option v-for="t in config.types" :value="t">{{ t }}</option>
        </select>

        <label>航司 Airline</label>
        <select v-model="annotation.airline_name">
          <option v-for="a in config.airlines" :value="a">{{ a }}</option>
        </select>

        <label>注册号 Registration</label>
        <input v-model="annotation.registration" placeholder="B-1234">

        <label>质量 Quality: {{ annotation.quality.toFixed(1) }}</label>
        <input type="range" class="quality-slider" v-model.number="annotation.quality"
               min="0" max="1" step="0.1">

        <p>边界框: {{ bboxes.length }} 个 (在图上拖拽绘制)</p>
        <button @click="clearBoxes">清除边界框</button>

        <hr>
        <button @click="submit" :disabled="!annotation.type_name">
          提交并下一张 (Enter)
        </button>
        <button @click="skip" style="background:#ff9800">跳过 (S)</button>
      </div>
    </div>
  </div>

  <script>
    const { createApp, ref, reactive, onMounted } = Vue

    createApp({
      setup() {
        const config = ref({ types: [], airlines: [] })
        const progress = ref({ done: 0, total: 0, percent: '0%' })
        const currentImage = ref(null)
        const imageUrl = ref('')
        const annotation = reactive({
          type_name: '', airline_name: '', registration: '', quality: 1.0
        })
        const bboxes = ref([])
        const isDrawing = ref(false)
        const startPos = ref({ x: 0, y: 0 })

        // 加载配置
        async function loadConfig() {
          const res = await fetch('/api/config')
          config.value = await res.json()
        }

        // 加载下一张图片
        async function loadNext() {
          const res = await fetch('/api/images/next')
          const data = await res.json()
          if (data.filename) {
            currentImage.value = data.filename
            imageUrl.value = data.url
            progress.value = { done: data.done, total: data.total,
                              percent: (100 * data.done / data.total).toFixed(1) + '%' }
            // 重置表单
            annotation.type_name = ''
            annotation.airline_name = ''
            annotation.registration = ''
            annotation.quality = 1.0
            bboxes.value = []
          } else {
            alert('所有图片已标注完成!')
          }
        }

        // 绘制边界框
        function startDraw(e) {
          const rect = e.target.getBoundingClientRect()
          startPos.value = {
            x: (e.clientX - rect.left) / rect.width * 100,
            y: (e.clientY - rect.top) / rect.height * 100
          }
          isDrawing.value = true
        }

        function drawing(e) {
          // 可以添加实时预览
        }

        function endDraw(e) {
          if (!isDrawing.value) return
          isDrawing.value = false

          const rect = e.target.getBoundingClientRect()
          const endX = (e.clientX - rect.left) / rect.width * 100
          const endY = (e.clientY - rect.top) / rect.height * 100

          const x = Math.min(startPos.value.x, endX)
          const y = Math.min(startPos.value.y, endY)
          const w = Math.abs(endX - startPos.value.x)
          const h = Math.abs(endY - startPos.value.y)

          if (w > 1 && h > 1) {  // 最小尺寸
            bboxes.value.push({ x, y, w, h })
          }
        }

        function clearBoxes() {
          bboxes.value = []
        }

        // 提交标注
        async function submit() {
          if (!annotation.type_name) {
            alert('请选择机型')
            return
          }

          // 转换 bbox 为 YOLO 格式 (中心点 + 宽高，归一化)
          const yoloBboxes = bboxes.value.map(b => [
            (b.x + b.w / 2) / 100,
            (b.y + b.h / 2) / 100,
            b.w / 100,
            b.h / 100
          ])

          await fetch('/api/annotations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              filename: currentImage.value,
              type_name: annotation.type_name,
              airline_name: annotation.airline_name,
              registration: annotation.registration.toUpperCase().replace(/\s/g, ''),
              quality: annotation.quality,
              bbox: yoloBboxes
            })
          })

          loadNext()
        }

        function skip() {
          loadNext()
        }

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') submit()
          if (e.key === 's' || e.key === 'S') skip()
        })

        onMounted(() => {
          loadConfig()
          loadNext()
        })

        return { config, progress, currentImage, imageUrl, annotation,
                 bboxes, startDraw, drawing, endDraw, clearBoxes, submit, skip }
      }
    }).mount('#app')
  </script>
</body>
</html>
```

#### 4. 启动服务

```bash
# 目录结构
annotation_server/
├── main.py              # FastAPI 后端
└── static/
    └── index.html       # 前端页面

# 启动
cd annotation_server
uvicorn main:app --reload --port 8000

# 访问 http://localhost:8000/static/index.html
```

---

### 标注界面效果

```
┌────────────────────────────────────────────────────────────────┐
│  Aircraft Annotation Tool                                       │
│  ████████████████░░░░░░░░░░░░░░  进度: 156 / 500 (31.2%)       │
├────────────────────────────────┬───────────────────────────────┤
│                                │  机型 Type                     │
│    ┌──────────────────┐        │  [A320           ▼]           │
│    │                  │        │                               │
│    │   ✈️ 飞机图片     │        │  航司 Airline                 │
│    │                  │        │  [China Eastern  ▼]           │
│    │    ┌────────┐    │        │                               │
│    │    │B-1234  │ ←框│        │  注册号 Registration           │
│    │    └────────┘    │        │  [B-1234        ]             │
│    │                  │        │                               │
│    └──────────────────┘        │  质量 Quality: 0.9             │
│                                │  ○────────●──○  [==========]  │
│                                │                               │
│                                │  边界框: 1 个                  │
│                                │  [清除边界框]                  │
│                                │  ─────────────                │
│                                │  [提交并下一张 (Enter)]        │
│                                │  [跳过 (S)]                    │
└────────────────────────────────┴───────────────────────────────┘
```

---

### 标注流程总览

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. 数据收集 │ → │  2. 预处理   │ → │  3. 人工标注 │ → │  4. 质量审核 │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                              ↓                  ↓
                                      ┌─────────────┐    ┌─────────────┐
                                      │  5. 导出格式 │ ← │  6. 修正返工 │
                                      └─────────────┘    └─────────────┘
```

### 阶段 1：数据收集与预处理

#### 1.1 图片下载与命名

```python
# scripts/download_images.py
import requests
from pathlib import Path
import hashlib

def download_and_rename(url: str, save_dir: str, prefix: str = ""):
    """下载图片并用 hash 重命名避免重复"""
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        # 用内容 hash 作为文件名
        content_hash = hashlib.md5(response.content).hexdigest()[:12]
        filename = f"{prefix}_{content_hash}.jpg" if prefix else f"{content_hash}.jpg"

        save_path = Path(save_dir) / filename
        save_path.write_bytes(response.content)
        return str(save_path)
    return None
```

#### 1.2 创建待标注清单

```python
# scripts/create_annotation_list.py
import pandas as pd
from pathlib import Path

def create_annotation_csv(image_dir: str, output_csv: str):
    """创建待标注的 CSV 文件"""
    image_dir = Path(image_dir)

    records = []
    for img_path in sorted(image_dir.glob("*.jpg")):
        records.append({
            "filename": img_path.name,
            "type_name": "",        # 待填写
            "airline_name": "",     # 待填写
            "registration": "",     # 待填写
            "quality": "",          # 待填写: good/medium/poor
            "annotator": "",        # 标注人
            "verified": False,      # 是否已审核
            "notes": ""             # 备注
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")  # utf-8-sig 支持 Excel 直接打开
    print(f"Created annotation file: {output_csv} ({len(records)} images)")

if __name__ == "__main__":
    create_annotation_csv(
        image_dir="data/processed/aircraft_crop/unsorted",
        output_csv="data/labels/annotation_todo.csv"
    )
```

### 阶段 2：机型/航司分类标注

#### 2.1 标注指南

```markdown
## 机型标注规范

### 命名规则
- 使用 ICAO 型号代码简写
- 同系列不同型号分开标注
- 示例：
  - ✅ B737-800（具体型号）
  - ❌ B737（太笼统）
  - ✅ A320-200
  - ✅ A320neo（新发动机型号单独分类）

### 常见易混淆机型
| 机型 A | 机型 B | 区分方法 |
|--------|--------|----------|
| A320 | A321 | A321 更长，看舱门数量 |
| B737-800 | B737-900 | 900 更长，看紧急出口位置 |
| A330-200 | A330-300 | 300 更长 |
| B777-200 | B777-300 | 300 明显更长 |
| A350-900 | A350-1000 | 1000 更长，看起落架舱门 |

### 标注原则
1. **不确定就标"Unknown"** - 宁缺毋滥
2. **参考注册号查询** - 用 flightradar24.com 或 planespotters.net
3. **看翼尖小翼形状** - 不同机型小翼设计不同
4. **看发动机数量和位置** - 双发/四发，翼下/尾部
```

#### 2.2 标注工具：Excel 批量标注

```
推荐工作流：

1. 打开 annotation_todo.csv
2. 使用图片查看器（如 IrfanView）批量预览图片
3. 在 Excel 中逐行填写：
   - type_name: 机型名称
   - airline_name: 航空公司
   - registration: 注册号（如果可见）
   - quality: good / medium / poor
4. 每 100 张保存一次
5. 填写 annotator 字段（你的名字）
```

#### 2.3 辅助标注脚本（图片+表格联动）

```python
# scripts/annotation_helper.py
"""
简易标注辅助工具：显示图片 + 输入标签
"""
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

class AnnotationHelper:
    def __init__(self, csv_path: str, image_dir: str):
        self.csv_path = csv_path
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(csv_path)
        self.current_idx = self._find_first_unlabeled()

        # 预定义选项
        self.type_options = [
            "A320", "A321", "A330-300", "A350-900", "A380",
            "B737-800", "B747-400", "B777-300ER", "B787-9",
            "ARJ21", "C919", "Unknown"
        ]
        self.airline_options = [
            "Air China", "China Eastern", "China Southern",
            "Hainan Airlines", "Xiamen Airlines", "Spring Airlines",
            "Cathay Pacific", "Singapore Airlines", "Emirates",
            "Unknown"
        ]

    def _find_first_unlabeled(self) -> int:
        """找到第一个未标注的图片"""
        for idx, row in self.df.iterrows():
            if pd.isna(row["type_name"]) or row["type_name"] == "":
                return idx
        return len(self.df)

    def show_current(self):
        """显示当前图片"""
        if self.current_idx >= len(self.df):
            print("All images annotated!")
            return

        row = self.df.iloc[self.current_idx]
        img_path = self.image_dir / row["filename"]

        plt.figure(figsize=(12, 8))
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"[{self.current_idx + 1}/{len(self.df)}] {row['filename']}")
        plt.axis("off")
        plt.show()

        print(f"\n当前进度: {self.current_idx + 1}/{len(self.df)}")
        print(f"文件名: {row['filename']}")

    def annotate(self, type_name: str, airline_name: str,
                 registration: str = "", quality: str = "good"):
        """标注当前图片"""
        self.df.loc[self.current_idx, "type_name"] = type_name
        self.df.loc[self.current_idx, "airline_name"] = airline_name
        self.df.loc[self.current_idx, "registration"] = registration
        self.df.loc[self.current_idx, "quality"] = quality
        self.df.loc[self.current_idx, "annotator"] = "manual"

        self.current_idx += 1
        self.save()
        print(f"✓ 已标注，进度: {self.current_idx}/{len(self.df)}")

    def skip(self):
        """跳过当前图片"""
        self.df.loc[self.current_idx, "notes"] = "skipped"
        self.current_idx += 1
        self.save()

    def save(self):
        """保存标注结果"""
        self.df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")

    def print_options(self):
        """打印可选项"""
        print("\n=== 机型选项 ===")
        for i, t in enumerate(self.type_options):
            print(f"  {i}: {t}")
        print("\n=== 航司选项 ===")
        for i, a in enumerate(self.airline_options):
            print(f"  {i}: {a}")

# 使用示例
if __name__ == "__main__":
    helper = AnnotationHelper(
        csv_path="data/labels/annotation_todo.csv",
        image_dir="data/processed/aircraft_crop/unsorted"
    )

    # 交互式标注
    while True:
        helper.show_current()
        helper.print_options()

        cmd = input("\n输入 (type_idx airline_idx [reg] [quality]) 或 's'跳过, 'q'退出: ")
        if cmd == 'q':
            break
        if cmd == 's':
            helper.skip()
            continue

        parts = cmd.split()
        if len(parts) >= 2:
            type_name = helper.type_options[int(parts[0])]
            airline_name = helper.airline_options[int(parts[1])]
            reg = parts[2] if len(parts) > 2 else ""
            quality = parts[3] if len(parts) > 3 else "good"
            helper.annotate(type_name, airline_name, reg, quality)
```

### 阶段 3：注册号区域标注（目标检测）

#### 3.1 使用 Label Studio

```bash
# 安装 Label Studio
pip install label-studio

# 启动
label-studio start
```

**配置标注模板**：

```xml
<!-- Label Studio 配置：注册号区域检测 -->
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="registration" background="#FF0000"/>
    <Label value="airline_logo" background="#00FF00"/>
    <Label value="aircraft_number" background="#0000FF"/>
  </RectangleLabels>
</View>
```

#### 3.2 使用 LabelImg（更轻量）

```bash
# 安装
pip install labelImg

# 启动
labelImg data/processed/aircraft_crop/unsorted data/labels/classes.txt
```

**classes.txt 内容**：
```
registration
airline_logo
```

#### 3.3 标注导出转换

```python
# scripts/convert_labelimg_to_csv.py
"""将 LabelImg 的 YOLO 格式转换为训练用的 CSV"""
import pandas as pd
from pathlib import Path

def convert_yolo_to_csv(label_dir: str, image_dir: str, output_csv: str):
    """转换 YOLO 格式标注到 CSV"""
    records = []
    label_path = Path(label_dir)

    for txt_file in label_path.glob("*.txt"):
        image_name = txt_file.stem + ".jpg"

        with open(txt_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = parts
                    records.append({
                        "filename": image_name,
                        "class_id": int(class_id),
                        "x_center": float(x_center),
                        "y_center": float(y_center),
                        "width": float(width),
                        "height": float(height)
                    })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Converted {len(records)} annotations to {output_csv}")
```

### 阶段 4：标注质量审核

#### 4.1 审核流程

```
审核检查清单：
□ 机型标注是否准确（抽查 20%）
□ 航司标注是否匹配涂装
□ 注册号是否正确（可通过网站验证）
□ 边界框是否紧密包围目标
□ 是否有遗漏标注
□ 是否有重复图片
```

#### 4.2 自动化审核脚本

```python
# scripts/verify_annotations.py
"""标注质量自动检查"""
import pandas as pd
from pathlib import Path
from collections import Counter

def verify_annotations(csv_path: str, image_dir: str):
    """检查标注质量"""
    df = pd.read_csv(csv_path)
    issues = []

    # 1. 检查空值
    empty_type = df[df["type_name"].isna() | (df["type_name"] == "")]
    if len(empty_type) > 0:
        issues.append(f"⚠️ {len(empty_type)} 张图片缺少机型标注")

    # 2. 检查图片是否存在
    image_path = Path(image_dir)
    for filename in df["filename"]:
        if not (image_path / filename).exists():
            issues.append(f"❌ 图片不存在: {filename}")

    # 3. 检查类别分布是否均衡
    type_counts = Counter(df["type_name"].dropna())
    print("\n=== 机型分布 ===")
    for type_name, count in type_counts.most_common():
        bar = "█" * (count // 10)
        print(f"  {type_name:15} {count:4} {bar}")

    # 4. 检查是否有异常值
    valid_types = ["A320", "A321", "A330-300", "A350-900", "A380",
                   "B737-800", "B747-400", "B777-300ER", "B787-9",
                   "ARJ21", "C919", "Unknown"]
    invalid_types = df[~df["type_name"].isin(valid_types + ["", None])]["type_name"].unique()
    if len(invalid_types) > 0:
        issues.append(f"⚠️ 发现非标准机型名称: {list(invalid_types)}")

    # 5. 检查重复图片
    duplicates = df[df.duplicated(subset=["filename"], keep=False)]
    if len(duplicates) > 0:
        issues.append(f"❌ 发现 {len(duplicates)} 条重复记录")

    # 打印问题
    print("\n=== 审核结果 ===")
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ 标注质量检查通过")

    return len(issues) == 0

if __name__ == "__main__":
    verify_annotations(
        csv_path="data/labels/annotation_todo.csv",
        image_dir="data/processed/aircraft_crop/unsorted"
    )
```

#### 4.3 交叉验证（多人标注）

```python
# scripts/cross_validate.py
"""比较多个标注者的结果"""
import pandas as pd

def compare_annotations(csv1: str, csv2: str):
    """比较两个标注者的结果"""
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # 合并
    merged = df1.merge(df2, on="filename", suffixes=("_a", "_b"))

    # 计算一致性
    type_agree = (merged["type_name_a"] == merged["type_name_b"]).mean()
    airline_agree = (merged["airline_name_a"] == merged["airline_name_b"]).mean()

    print(f"机型标注一致性: {type_agree:.1%}")
    print(f"航司标注一致性: {airline_agree:.1%}")

    # 找出不一致的样本
    disagreements = merged[merged["type_name_a"] != merged["type_name_b"]]
    print(f"\n不一致样本数: {len(disagreements)}")

    if len(disagreements) > 0:
        print("\n需要复核的图片:")
        for _, row in disagreements.head(10).iterrows():
            print(f"  {row['filename']}: {row['type_name_a']} vs {row['type_name_b']}")
```

### 阶段 5：标注数据导出

#### 5.1 导出为训练格式

```python
# scripts/export_annotations.py
"""将标注导出为训练所需的最终格式"""
import pandas as pd
import json
from pathlib import Path
import shutil

def export_for_training(
    annotation_csv: str,
    image_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """导出标注数据为训练格式"""
    df = pd.read_csv(annotation_csv)

    # 过滤掉未标注和质量差的
    df = df[df["type_name"].notna() & (df["type_name"] != "")]
    df = df[df["quality"] != "poor"]

    print(f"有效样本数: {len(df)}")

    # 创建类别映射
    types = sorted(df["type_name"].unique())
    airlines = sorted(df["airline_name"].dropna().unique())

    type_to_id = {t: i for i, t in enumerate(types)}
    airline_to_id = {a: i for i, a in enumerate(airlines)}

    # 添加 ID 列
    df["type_id"] = df["type_name"].map(type_to_id)
    df["airline_id"] = df["airline_name"].map(airline_to_id)

    # 随机打乱并划分
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": df[:train_end],
        "val": df[train_end:val_end],
        "test": df[val_end:]
    }

    # 创建输出目录
    output_path = Path(output_dir)
    image_path = Path(image_dir)

    for split_name, split_df in splits.items():
        # 创建目录结构
        for type_name in types:
            (output_path / split_name / type_name).mkdir(parents=True, exist_ok=True)

        # 复制图片到对应目录
        for _, row in split_df.iterrows():
            src = image_path / row["filename"]
            dst = output_path / split_name / row["type_name"] / row["filename"]
            if src.exists():
                shutil.copy(src, dst)

        print(f"{split_name}: {len(split_df)} images")

    # 保存类别映射
    labels_dir = output_path / "labels"
    labels_dir.mkdir(exist_ok=True)

    with open(labels_dir / "type_classes.json", "w") as f:
        json.dump({"classes": types, "num_classes": len(types)}, f, indent=2)

    with open(labels_dir / "airline_classes.json", "w") as f:
        json.dump({"classes": airlines, "num_classes": len(airlines)}, f, indent=2)

    # 保存完整标注 CSV
    df.to_csv(labels_dir / "aircraft_labels.csv", index=False)

    print(f"\n✅ 导出完成: {output_path}")
    print(f"   机型类别: {len(types)}")
    print(f"   航司类别: {len(airlines)}")

if __name__ == "__main__":
    export_for_training(
        annotation_csv="data/labels/annotation_todo.csv",
        image_dir="data/processed/aircraft_crop/unsorted",
        output_dir="data/processed/aircraft_crop"
    )
```

### 标注效率建议

| 任务类型 | 单人效率 | 建议策略 |
|----------|----------|----------|
| 机型分类 | 200-300 张/小时 | 用快捷键，预览+输入 |
| 航司分类 | 300-400 张/小时 | 按涂装颜色分组 |
| 注册号检测 | 80-120 张/小时 | 用 Label Studio 模板 |
| 注册号 OCR | 150-200 张/小时 | 先检测后转录 |

### 标注分工建议

```
小团队（1-2人）标注 5000 张图：
├── 第 1 天：数据收集 + 预处理（500 张/人）
├── 第 2-4 天：机型+航司标注（500 张/人/天）
├── 第 5 天：交叉审核 + 修正
├── 第 6 天：注册号区域标注
└── 第 7 天：导出 + 验证

预计总耗时：1 周
```

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

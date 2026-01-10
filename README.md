# 🎓 深度学习课程项目报告

基于 YOLOv8/YOLOv12 的教室场景人脸检测、跟踪与识别系统，支持模型微调、TensorRT 加速部署及 Web 可视化展示。

---

## 📁 项目结构

```
deeplearning_fdu/
├── face_detection/          # 人脸检测核心模块
├── model_finetune/          # 模型微调脚本（公开）
├── label/                   # 数据标注与训练（私有）
├── cuda_tensorrt/           # CUDA/TensorRT 加速部署
├── frontend/                # Web 前端展示
├── models/                  # 预训练模型 (yolov8n-face, yolov12l-face)
└── 教室学生机位/             # 视频数据源
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装 Python 依赖
pip install ultralytics opencv-python insightface

# 前端依赖
pip install uvicorn fastapi ffmpeg

# TensorRT 部署需要额外安装：cuda, cudnn, tensorrt, opencv
```

### 2. 数据准备

将视频文件放置到 `教室学生机位/` 目录下。

---

## 🔍 人脸检测

### 基线模型测试

使用预训练的 YOLOv8n-face 和 YOLOv12l-face 模型进行检测：

```bash
mkdir -p test_data/face_detection/

# YOLOv8n-face
python face_detection/yolo_face_detector.py \
    --input "教室学生机位/深度学习应用++2025-11-29第1-3节+[30657+-+524563]教室流.mp4" \
    --output "test_data/face_detection/classroom_yolov8n.mp4" \
    --model yolov8n-face \
    --conf 0.3 \
    --start-time 6000 \
    --max-frames 5000

# YOLOv12l-face
python face_detection/yolo_face_detector.py \
    --input "教室学生机位/深度学习应用++2025-11-29第1-3节+[30657+-+524563]教室流.mp4" \
    --output "test_data/face_detection/classroom_yolov12l.mp4" \
    --model yolov12l-face \
    --conf 0.3 \
    --start-time 6000 \
    --max-frames 5000
```

### 微调模型测试

使用微调后的模型进行检测：

```bash
# YOLOv8n 微调模型
python face_detection/yolo_face_detector.py \
    --input "教室学生机位/深度学习应用++2025-11-29第1-3节+[30657+-+524563]教室流.mp4" \
    --output "test_data/face_detection/classroom_yolov8n_finetune.mp4" \
    --model-path label/runs/yolov8n_face_finetune/weights/best.pt \
    --conf 0.3 \
    --start-time 6000 \
    --max-frames 100

# YOLOv12l 微调模型
python face_detection/yolo_face_detector.py \
    --input "教室学生机位/深度学习应用++2025-11-29第1-3节+[30657+-+524563]教室流.mp4" \
    --output "test_data/face_detection/classroom_yolov12l_finetune.mp4" \
    --model-path label/runs/yolov12l_face_finetune6/weights/best.pt \
    --conf 0.3 \
    --start-time 6000 \
    --max-frames 100
```

---

## 👥 人脸跟踪与保存

启用跟踪功能并按间隔保存检测到的人脸：

```bash
python face_detection/yolo_face_detector.py \
    --input "教室学生机位/深度学习应用++2025-11-15第1-3节+[30657+-+524561]教室流.mp4" \
    --output "test_data/tracked_faces.mp4" \
    --model-path label/runs/yolov12l_face_finetune6/weights/best.pt \
    --conf 0.3 \
    --start-time 6000 \
    --max-frames 5000 \
    --track \
    --save-faces \
    --save-interval-sec 3
```

---

## 🎯 人脸识别（与照片库匹配）

结合同学照片库进行人脸识别：

```bash
python face_detection/yolo_face_detector.py \
    --input "教室学生机位/深度学习应用++2025-11-29第1-3节+[30657+-+524563]教室流.mp4" \
    --output "test_data/recognized_faces.mp4" \
    --model-path label/runs/yolov12l_face_finetune6/weights/best.pt \
    --conf 0.3 \
    --start-time 6000 \
    --max-frames 500 \
    --photo-folder classmate_photo_processed/ \
    --similarity-threshold 0.61
```

---

## 🛠️ 模型微调

### 工作流程

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1. 提取帧 | `model_finetune/extract_frames.py` | 从视频中提取训练图片 |
| 2. 数据标注 | `model_finetune/label.py` | 标注人脸边界框 |
| 3. 模型训练 | `model_finetune/model_train.py` | 微调 YOLO 模型 |

> 💡 **提示**：`model_finetune/` 为公开脚本，`label/` 目录包含私有训练数据，不公开。

---

## ⚡ TensorRT 加速部署

### 环境配置

1. 安装依赖：CUDA、cuDNN、TensorRT、OpenCV
2. 修改 `cuda_tensorrt/deploy-yolo/Makefile` 中的路径配置

### 运行

```bash
cd cuda_tensorrt/deploy-yolo

# Python 测试
python src/python/test_model.py

# 编译并运行 TensorRT 推理
make clean && mkdir -p bin && make run
```

---

## 🌐 Web 前端展示

### 启动服务

```bash
# 终端 1：启动后端 API 服务
cd frontend && uvicorn api_server:app --host 0.0.0.0 --port 8000

# 终端 2：启动前端静态服务
cd frontend && python -m http.server 8001
```

### 访问

- 前端界面：http://localhost:8001
- API 文档：http://localhost:8000/docs

### 输出目录

- `frontend/uploads/` - 上传的文件
- `frontend/outputs/` - 处理结果

---

## 📊 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入视频路径 | - |
| `--output` | 输出视频路径 | - |
| `--model` | 预训练模型名称 | `yolov8n-face` |
| `--model-path` | 自定义模型路径 | - |
| `--conf` | 置信度阈值 | 0.3 |
| `--start-time` | 起始时间（秒） | 0 |
| `--max-frames` | 最大处理帧数 | -1 (全部) |
| `--track` | 启用目标跟踪 | False |
| `--save-faces` | 保存检测到的人脸 | False |
| `--save-interval-sec` | 保存间隔（秒） | 1 |
| `--photo-folder` | 照片库路径 | - |
| `--similarity-threshold` | 相似度阈值 | 0.61 |

---

## 📝 License

本项目仅用于课堂大作业。


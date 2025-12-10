# YOLOv8 人脸检测系统

基于 Ultralytics YOLOv8 实现的高效人脸检测系统，支持实时视频检测、批量图像处理、模型训练等功能。

## 🌟 项目特点

- ⚡ **高效检测**: 基于最新的YOLOv8架构，检测速度快、精度高
- 🎯 **专门优化**: 针对人脸检测任务进行专门优化和调参
- 📱 **多种输入**: 支持摄像头实时检测、视频文件、图片文件和批量处理
- 🔧 **易于定制**: 提供完整的训练流程，支持自定义数据集
- 📊 **多种格式**: 支持WIDER FACE、COCO、Pascal VOC等多种数据格式
- 🚀 **模型导出**: 支持ONNX、TensorRT等多种推理格式
- 📈 **可视化**: 提供训练过程可视化和结果分析

## 🏗️ 项目结构

```
yolov8_face_detection/
├── README.md                    # 项目说明文档
├── requirements.txt             # 项目依赖包
├── face_detector.py            # 主要检测脚本
├── train_face_model.py         # 模型训练脚本
├── configs/                    # 配置文件目录
│   ├── face_dataset.yaml      # 数据集配置
│   └── yolov8_face.yaml       # 模型配置
├── scripts/                    # 实用脚本
│   └── download_pretrained.py # 预训练模型下载
├── utils/                      # 工具函数
│   └── data_preprocessor.py   # 数据预处理工具
├── data/                       # 数据存放目录
├── models/                     # 模型文件目录
└── runs/                       # 训练和检测结果
```

## 🔧 环境设置

### 1. 创建虚拟环境 (推荐)

```bash
# 使用conda创建环境
conda create -n yolov8_face python=3.8
conda activate yolov8_face

# 或使用venv
python -m venv yolov8_face
source yolov8_face/bin/activate  # Linux/Mac
# yolov8_face\Scripts\activate   # Windows
```

### 2. 安装依赖包

```bash
cd yolov8_face_detection
pip install -r requirements.txt
```

### 3. 下载预训练模型

```bash
# 下载YOLOv8n和YOLOv8s预训练模型
python scripts/download_pretrained.py --model-sizes n s

# 或手动下载到models目录
mkdir models
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

## 🚀 快速开始

### 1. 实时人脸检测 (摄像头)

```bash
# 使用默认摄像头
python face_detector.py --source 0

# 使用指定摄像头
python face_detector.py --source 1 --model models/yolov8n.pt
```

### 2. 视频文件检测

```bash
# 检测视频文件
python face_detector.py --source video.mp4 --save-video output.mp4

# 不显示窗口，仅保存结果
python face_detector.py --source video.mp4 --save-video output.mp4 --no-show
```

### 3. 图片检测

```bash
# 单张图片检测
python face_detector.py --source image.jpg --output results/

# 批量图片检测
python face_detector.py --source images/ --output results/
```

### 4. 参数说明

```bash
python face_detector.py --help

# 主要参数:
# --model: 模型文件路径 (默认: yolov8n.pt)
# --source: 输入源 (摄像头ID/视频文件/图片文件/目录)
# --output: 输出目录 (默认: runs/detect)
# --conf: 置信度阈值 (默认: 0.5)
# --device: 运行设备 (auto/cpu/0,1,2...)
# --save-video: 保存检测视频的路径
# --no-show: 不显示检测窗口
```

## 📚 数据准备

### 支持的数据格式

项目支持多种人脸数据集格式，提供自动转换工具:

#### 1. WIDER FACE 格式

```bash
python utils/data_preprocessor.py \
    --format wider \
    --input-dir /path/to/WIDER_FACE \
    --output-dir data/processed \
    --annotation-file /path/to/wider_face_train_bbx_gt.txt \
    --split-data
```

#### 2. COCO 格式

```bash
python utils/data_preprocessor.py \
    --format coco \
    --input-dir /path/to/coco \
    --output-dir data/processed \
    --annotation-file annotations/instances_train2017.json \
    --image-dir images/train2017 \
    --split-data
```

#### 3. Pascal VOC 格式

```bash
python utils/data_preprocessor.py \
    --format voc \
    --input-dir /path/to/VOC2012 \
    --output-dir data/processed \
    --annotation-file Annotations \
    --image-dir JPEGImages \
    --split-data
```

### 数据集目录结构

处理后的数据应该具有以下结构:

```
data/
├── train/
│   ├── images/          # 训练图片
│   └── labels/          # YOLO格式标注 (.txt)
├── val/
│   ├── images/          # 验证图片  
│   └── labels/          # YOLO格式标注
└── test/ (可选)
    ├── images/          # 测试图片
    └── labels/          # YOLO格式标注
```

## 🎯 模型训练

### 1. 准备配置文件

编辑 `configs/face_dataset.yaml`，设置正确的数据路径:

```yaml
path: /path/to/your/data
train: train/images
val: val/images
nc: 1
names:
  0: face
```

### 2. 开始训练

```bash
# 基础训练 (使用YOLOv8n)
python train_face_model.py \
    --train-dir data/train \
    --val-dir data/val \
    --model-size n \
    --epochs 100 \
    --batch-size 16

# 高精度训练 (使用YOLOv8s)
python train_face_model.py \
    --train-dir data/train \
    --val-dir data/val \
    --model-size s \
    --epochs 200 \
    --batch-size 8 \
    --imgsz 640
```

### 3. 训练参数说明

```bash
# 必需参数:
# --train-dir: 训练数据目录
# --val-dir: 验证数据目录

# 可选参数:
# --model-size: 模型大小 (n/s/m/l/x)
# --epochs: 训练轮数 (默认100)
# --batch-size: 批次大小 (默认16)  
# --imgsz: 输入图像尺寸 (默认640)
# --lr0: 初始学习率 (默认0.01)
# --device: 训练设备 (默认auto)
# --save-dir: 保存目录 (默认runs/train)
# --resume: 从断点继续训练
# --no-pretrained: 不使用预训练权重
```

### 4. 监控训练过程

训练过程中会自动保存:
- 训练日志和损失曲线
- 验证指标 (mAP, 精度, 召回率)
- 最佳模型权重 (`best.pt`)
- 最后一轮权重 (`last.pt`)

```bash
# 查看TensorBoard (如果安装了)
tensorboard --logdir runs/train
```

## 📊 模型评估

### 验证模型性能

```bash
# 在验证集上评估模型
python train_face_model.py \
    --train-dir data/train \
    --val-dir data/val \
    --validate-only \
    --model-path runs/train/exp/weights/best.pt
```

### 评估指标说明

- **mAP@0.5**: IoU阈值为0.5时的平均精度
- **mAP@0.5:0.95**: IoU阈值从0.5到0.95的平均精度  
- **Precision**: 精度 (检测到的人脸中真正是人脸的比例)
- **Recall**: 召回率 (所有人脸中被检测到的比例)
- **F1-Score**: 精度和召回率的调和平均数

## 🔄 模型导出

### 支持的导出格式

```bash
# 导出为ONNX格式
python -c \"
from ultralytics import YOLO
model = YOLO('runs/train/exp/weights/best.pt')
model.export(format='onnx', imgsz=640)
\"

# 导出为TensorRT格式 (需要NVIDIA GPU)
python -c \"
from ultralytics import YOLO
model = YOLO('runs/train/exp/weights/best.pt')  
model.export(format='engine', imgsz=640)
\"
```

### 使用导出的模型

```bash
# 使用ONNX模型检测
python face_detector.py --model best.onnx --source test.jpg

# 使用TensorRT模型检测  
python face_detector.py --model best.engine --source 0
```

## ⚙️ 高级功能

### 1. 自定义检测参数

```python
from face_detector import YOLOv8FaceDetector

# 创建检测器实例
detector = YOLOv8FaceDetector(
    model_path='models/best.pt',
    conf_threshold=0.6,  # 提高置信度阈值
    device='cuda:0'      # 指定GPU设备
)

# 检测图片
import cv2
image = cv2.imread('test.jpg')
faces, vis_image = detector.detect_faces(image)

print(f\"检测到 {len(faces)} 个人脸\")
for i, face in enumerate(faces):
    bbox = face['bbox']
    conf = face['confidence']
    print(f\"人脸{i+1}: 坐标{bbox}, 置信度{conf:.2f}\")
```

### 2. 批量处理和性能优化

```python
# 批量检测多个图片
detector.batch_detect(
    image_dir='input_images/',
    output_dir='output_results/'
)

# 实时视频检测优化
detector.detect_video(
    source=0,              # 摄像头
    save_path='output.mp4', # 保存视频
    show=True              # 实时显示
)
```

### 3. 结果后处理

```python
# 自定义结果过滤
def filter_faces(faces, min_size=20):
    \"\"\"过滤太小的人脸\"\"\"
    filtered = []
    for face in faces:
        bbox = face['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width >= min_size and height >= min_size:
            filtered.append(face)
    return filtered

faces, _ = detector.detect_faces(image)
large_faces = filter_faces(faces, min_size=50)
```

## 🔍 故障排除

### 常见问题和解决方案

#### 1. CUDA内存不足

```bash
# 减小批次大小
python train_face_model.py --batch-size 8

# 减小输入图像尺寸  
python train_face_model.py --imgsz 416

# 使用混合精度训练
python train_face_model.py --amp
```

#### 2. 训练过慢

```bash
# 增加数据加载线程数
python train_face_model.py --workers 8

# 使用更小的模型
python train_face_model.py --model-size n

# 减少训练轮数进行测试
python train_face_model.py --epochs 10
```

#### 3. 检测精度不够

```bash
# 降低置信度阈值
python face_detector.py --conf 0.3

# 使用更大的模型
python face_detector.py --model yolov8l.pt

# 使用自训练的模型
python face_detector.py --model runs/train/exp/weights/best.pt
```

#### 4. 依赖包安装问题

```bash
# 更新pip和setuptools
pip install --upgrade pip setuptools

# 使用清华源安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 分步安装核心包
pip install torch torchvision ultralytics opencv-python
```

## 📈 性能基准

### 不同模型规模的性能对比

| 模型 | 参数量 | 模型大小 | mAP@0.5 | 推理速度 (FPS) |
|------|--------|----------|---------|----------------|
| YOLOv8n | 3.2M | 6.2MB | 95.1% | 120+ |
| YOLOv8s | 11.2M | 21.5MB | 96.3% | 80+ |  
| YOLOv8m | 25.9M | 49.7MB | 97.2% | 50+ |
| YOLOv8l | 43.7M | 83.7MB | 97.8% | 35+ |
| YOLOv8x | 68.2M | 130.5MB | 98.1% | 25+ |

*性能数据基于WIDER FACE验证集，使用RTX 3080测试*

### 推理速度优化建议

1. **硬件加速**: 使用NVIDIA GPU和TensorRT
2. **模型量化**: INT8量化可提升2-3倍速度
3. **输入尺寸**: 较小的输入尺寸可显著提升速度
4. **批量处理**: 批量推理比单张推理效率更高

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 开发环境设置

```bash
# 克隆项目
git clone <repository-url>
cd yolov8_face_detection

# 安装开发依赖
pip install -r requirements.txt
pip install pre-commit black flake8

# 设置pre-commit钩子
pre-commit install
```

### 提交规范

- 代码风格: 使用Black进行格式化
- 代码质量: 通过flake8检查
- 测试覆盖: 为新功能添加单元测试
- 文档更新: 更新相关文档和示例

## 📝 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8官方实现
- [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) - 人脸检测数据集
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📞 联系方式

如有问题或建议，请通过以下方式联系:

- 提交Issue: [项目Issues页面]
- 邮箱: [your.email@example.com]
- 微信群: [扫码加入技术讨论群]

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

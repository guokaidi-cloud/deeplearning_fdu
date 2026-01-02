# YOLO 模型 TensorRT 部署技术报告

## 一、项目概述

### 1.1 项目目标

本项目实现了基于 NVIDIA TensorRT 的 YOLO 目标检测模型高性能部署，支持 FP32、FP16、INT8 三种精度模式，并使用 CUDA 实现 GPU 加速的前后处理。

### 1.2 项目结构

```
deploy-yolo/
├── src/
│   ├── cpp/                    # C++ 源代码
│   │   ├── main.cpp            # 主程序入口
│   │   ├── trt_worker.cpp      # 推理工作器
│   │   ├── trt_model.cpp       # TensorRT 模型基类
│   │   ├── trt_detector.cpp    # YOLO 检测器实现
│   │   ├── trt_preprocess.cpp  # CPU 预处理
│   │   ├── trt_preprocess.cu   # GPU 预处理（CUDA）
│   │   ├── trt_calibrator.cpp  # INT8 量化校准器
│   │   ├── trt_logger.cpp      # 日志系统
│   │   ├── trt_timer.cpp       # 计时器
│   │   └── utils.cpp           # 工具函数
│   └── python/
│       └── test_model.py       # Python 版本对比测试
├── include/                    # 头文件
├── models/
│   ├── onnx/                   # ONNX 模型文件
│   └── engine/                 # TensorRT 引擎文件（自动生成）
├── data/
│   ├── source/                 # 输入图片
│   └── result/                 # 输出结果
├── calibration/                # INT8 量化校准数据
└── Makefile                    # 编译脚本
```

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                           main.cpp                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                       Worker                             │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │                    Detector                      │    │    │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │    │    │
│  │  │  │Preprocess│→│TRT Infer│→│  Postprocess    │  │    │    │
│  │  │  │  (GPU)  │  │  (GPU)  │  │ Decode + NMS   │  │    │    │
│  │  │  └─────────┘  └─────────┘  └─────────────────┘  │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 类继承关系

```
Model (基类)
├── Detector (YOLO 检测器)
└── Classifier (分类器，可扩展)
```

### 2.3 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| Worker | trt_worker.cpp | 推理工作器，管理模型生命周期 |
| Model | trt_model.cpp | TensorRT 模型基类，提供通用功能 |
| Detector | trt_detector.cpp | YOLO 检测器，实现检测特定的前后处理 |
| Preprocess | trt_preprocess.cu | CUDA 预处理 kernel |
| Calibrator | trt_calibrator.cpp | INT8 量化校准器 |

---

## 三、工作流程详解

### 3.1 完整推理流程

```
                 ┌──────────────────────────────────────────┐
                 │              初始化阶段                   │
                 └──────────────────────────────────────────┘
                                    │
                 ┌──────────────────▼──────────────────────┐
                 │  检查 TensorRT Engine 是否存在           │
                 └──────────────────┬──────────────────────┘
                        ┌───────────┴───────────┐
                        │                       │
                   存在 ▼                  不存在 ▼
            ┌───────────────────┐    ┌───────────────────┐
            │   load_engine()   │    │   build_engine()  │
            │   反序列化 Engine  │    │   ONNX → Engine   │
            └─────────┬─────────┘    └─────────┬─────────┘
                      │                        │
                      └───────────┬────────────┘
                                  ▼
                 ┌──────────────────────────────────────────┐
                 │              setup()                      │
                 │  - 创建 Runtime, Engine, Context         │
                 │  - 分配 Host/Device 内存                  │
                 │  - 设置 Bindings                          │
                 └──────────────────────────────────────────┘
                                  │
                 ┌────────────────▼────────────────────────┐
                 │              推理阶段                    │
                 └─────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Preprocess   │      │   TRT Inference │      │   Postprocess   │
│  (GPU/CPU)    │  →   │      (GPU)      │  →   │   (GPU/CPU)     │
│               │      │                 │      │                 │
│ - LetterBox   │      │ - enqueueV2()   │      │ - Decode        │
│ - BGR2RGB     │      │ - CUDA Stream   │      │ - NMS           │
│ - Normalize   │      │                 │      │ - Draw          │
└───────────────┘      └─────────────────┘      └─────────────────┘
```

### 3.2 Engine 构建流程（build_engine）

当 TensorRT Engine 文件不存在时，需要从 ONNX 模型构建：

```cpp
// 1. 创建 Builder, Network, Config, Parser
auto builder = createInferBuilder(*m_logger);
auto network = builder->createNetworkV2(1);
auto config = builder->createBuilderConfig();
auto parser = createParser(*network, *m_logger);

// 2. 解析 ONNX 模型
parser->parseFromFile(m_onnxPath.c_str(), 1);

// 3. 设置精度模式
if (m_params->prec == model::FP16) {
    config->setFlag(BuilderFlag::kFP16);
} else if (m_params->prec == model::INT8) {
    config->setFlag(BuilderFlag::kINT8);
    config->setInt8Calibrator(calibrator.get());
}

// 4. 构建并序列化 Engine
auto engine = builder->buildEngineWithConfig(*network, *config);
auto plan = builder->buildSerializedNetwork(*network, *config);

// 5. 保存到文件
save_plan(*plan);
```

**精度模式说明：**

| 精度 | 说明 | 速度 | 精度损失 |
|------|------|------|----------|
| FP32 | 单精度浮点 | 基准 | 无 |
| FP16 | 半精度浮点 | 约 2x | 极小 |
| INT8 | 8位整数 | 约 4x | 需要校准 |

### 3.3 GPU 预处理流程（LetterBox）

预处理在 GPU 上使用 CUDA Kernel 完成，与 Python 版本完全一致：

```
输入图像 (1920x1080, BGR, uint8)
         │
         ▼
┌─────────────────────────────────────────┐
│  Step 1: 计算缩放比例                    │
│  scale = min(640/1080, 640/1920)        │
│        = min(0.593, 0.333) = 0.333      │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Step 2: 缩放图像                        │
│  new_size = (640, 360)                  │
│  使用双线性插值                          │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Step 3: 填充到目标尺寸                  │
│  创建 640x640 画布                       │
│  填充值 = 114 (灰色)                     │
│  图像放置在左上角                        │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Step 4: 格式转换                        │
│  BGR → RGB                               │
│  HWC → CHW                               │
│  uint8 → float32 (除以 255)              │
└─────────────────────────────────────────┘
         │
         ▼
输出张量 (1x3x640x640, float32)
```

**CUDA Kernel 核心代码：**

```cuda
__global__ void letterbox_BGR2RGB_kernel(
    float* tar, uint8_t* src, 
    int tarW, int tarH, int srcW, int srcH,
    int newW, int newH, float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= tarW || y >= tarH) return;
    
    int tarIdx = y * tarW + x;
    int tarArea = tarW * tarH;
    
    // 填充区域
    if (x >= newW || y >= newH) {
        float fill = 114.0f / 255.0f;
        tar[tarIdx + tarArea * 0] = fill;  // R
        tar[tarIdx + tarArea * 1] = fill;  // G
        tar[tarIdx + tarArea * 2] = fill;  // B
        return;
    }
    
    // 双线性插值 + BGR2RGB + Normalize
    // ... (省略插值计算)
}
```

### 3.4 TensorRT 推理

```cpp
bool Model::enqueue_bindings() {
    m_timer->start_gpu();
    
    // 异步推理
    m_context->enqueueV2((void**)m_bindings, m_stream, nullptr);
    
    m_timer->stop_gpu();
    m_timer->duration_gpu("trt-inference(GPU)");
    return true;
}
```

**关键概念：**

- **Bindings**: 输入/输出张量的 GPU 内存指针数组
- **Stream**: CUDA 流，用于异步执行
- **enqueueV2**: TensorRT 异步推理 API

### 3.5 GPU 后处理流程

后处理分为两个阶段：

**阶段 1: GPU Decode（并行）**

```
模型输出 [1, 8400，10]
         │
         ▼
┌─────────────────────────────────────────┐
│  GPU Kernel: 每个线程处理一个候选框      │
│  - 解码坐标: cx, cy, w, h → x0, y0, x1, y1 │
│  - 置信度过滤: conf > 0.25               │
│  - LetterBox 逆变换: coords / scale      │
│  - 原子操作写入结果                      │
└─────────────────────────────────────────┘
         │
         ▼
有效候选框 (~100-500个)
```

**阶段 2: CPU NMS（串行）**

```
有效候选框
         │
         ▼
┌─────────────────────────────────────────┐
│  Step 1: 按置信度排序                    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Step 2: NMS 非极大值抑制                │
│  - 遍历所有框                            │
│  - 同类别 IoU > 0.45 的框标记移除        │
└─────────────────────────────────────────┘
         │
         ▼
最终检测结果 (~10-50个)
```

---

## 四、内存管理

### 4.1 内存分配策略

```
Host Memory (CPU)                Device Memory (GPU)
┌──────────────────┐             ┌──────────────────┐
│ m_inputMemory[0] │ ──────────► │ m_inputMemory[1] │
│   (Pinned)       │   cudaMemcpy│     (Device)     │
└──────────────────┘             └──────────────────┘
                                          │
                                          ▼
                                 ┌──────────────────┐
                                 │   TRT Inference  │
                                 └──────────────────┘
                                          │
                                          ▼
┌──────────────────┐             ┌──────────────────┐
│m_outputMemory[0] │ ◄────────── │m_outputMemory[1] │
│   (Pinned)       │   cudaMemcpy│     (Device)     │
└──────────────────┘             └──────────────────┘
```

**Pinned Memory 优势：**
- 使用 `cudaMallocHost` 分配
- 避免 CPU 分页，加速 Host-Device 数据传输
- DMA 直接传输，无需 CPU 中转

### 4.2 Bindings 设置

```cpp
// m_bindings 是指向 GPU 内存的指针数组
m_bindings[0] = m_inputMemory[1];   // 输入
m_bindings[1] = m_outputMemory[1];  // 输出
```

---

## 五、精度模式

### 5.1 FP32 模式

```cpp
// 默认模式，无需额外配置
params.prec = model::precision::FP32;
```

### 5.2 FP16 模式

```cpp
params.prec = model::precision::FP16;

// 构建时设置
if (builder->platformHasFastFp16() && m_params->prec == model::FP16) {
    config->setFlag(BuilderFlag::kFP16);
}
```

### 5.3 INT8 模式

INT8 量化需要校准数据：

```cpp
params.prec = model::precision::INT8;

// 创建校准器
calibrator = new Int8EntropyCalibrator(
    64,                                  // batch size
    "calibration/calibration_list.txt",  // 校准图片列表
    "calibration/calibration_table.txt", // 校准表输出
    3 * 640 * 640,                       // 输入大小
    640, 640                             // 图片尺寸
);

config->setFlag(BuilderFlag::kINT8);
config->setInt8Calibrator(calibrator.get());
```

---

## 六、性能对比

### 6.1 Python vs C++ TensorRT

| 阶段 | Python (ONNX Runtime) | C++ (TensorRT FP16) |
|------|----------------------|---------------------|
| 预处理 | ~5 ms (CPU) | ~0.5 ms (GPU) |
| 推理 | ~15 ms | ~5 ms |
| 后处理 | ~2 ms | ~1 ms |
| **总计** | **~22 ms** | **~6.5 ms** |
| **FPS** | ~45 | ~154 |

### 6.2 不同精度模式对比

| 精度 | 推理时间 | 相对速度 | 精度损失 |
|------|----------|----------|----------|
| FP32 | ~8 ms | 1.0x | 无 |
| FP16 | ~5 ms | 1.6x | <0.1% |
| INT8 | ~3 ms | 2.7x | <1% (需校准) |

---

## 七、使用方法

### 7.1 编译

```bash
cd deploy-yolo
make clean && make -j4
```

### 7.2 运行

```bash
make run
```

### 7.3 配置参数

```cpp
// main.cpp
auto params = model::Params();

params.img = {640, 640, 3};              // 输入尺寸
params.task = model::task_type::DETECTION; // 任务类型
params.dev = model::device::GPU;          // 运行设备
params.prec = model::precision::FP16;     // 精度模式

auto worker = thread::create_worker(onnxPath, level, params);
worker->inference("data/source/image.jpg");
```

---

## 八、关键技术总结

### 8.1 优化策略

| 优化点 | 技术实现 | 效果 |
|--------|----------|------|
| 模型加速 | TensorRT 优化 | 推理速度提升 3-5x |
| 精度优化 | FP16/INT8 量化 | 进一步提升 1.5-3x |
| 预处理加速 | CUDA Kernel | 预处理速度提升 10x |
| 后处理加速 | GPU Decode | 解码速度提升 5x |
| 内存优化 | Pinned Memory | 数据传输速度提升 2x |
| 异步执行 | CUDA Stream | 隐藏传输延迟 |

### 8.2 架构设计亮点

1. **模块化设计**: Model 基类提供通用功能，Detector 子类实现检测特定逻辑
2. **设备无关**: 支持 CPU/GPU 两种模式，通过配置切换
3. **精度可选**: 支持 FP32/FP16/INT8 三种精度
4. **Engine 缓存**: 首次构建后自动缓存，后续直接加载

---

## 九、文件说明

| 文件 | 功能 |
|------|------|
| main.cpp | 程序入口，配置参数并启动推理 |
| trt_worker.cpp | 推理工作器，管理 Detector/Classifier |
| trt_model.cpp | TensorRT 模型基类，Engine 构建/加载 |
| trt_detector.cpp | YOLO 检测器，前后处理实现 |
| trt_preprocess.cu | CUDA 预处理 Kernel (LetterBox) |
| trt_calibrator.cpp | INT8 量化校准器 |
| trt_logger.cpp | 多级日志系统 |
| trt_timer.cpp | CPU/GPU 计时器 |
| test_model.py | Python 版本对比测试 |



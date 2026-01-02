#!/usr/bin/env python3
"""
使用 ONNX Runtime 直接测试模型（处理转置问题）
"""

import onnxruntime as ort
import cv2
import numpy as np
import time
from pathlib import Path

# 获取脚本所在目录（相对路径基于脚本位置，而非运行时工作目录）
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # 从 src/python 向上两级到项目根目录

# 模型路径
onnx_path = str(PROJECT_ROOT / "models/onnx/yolov8m-liyin.onnx")

# 测试图片列表（与 C++ 版本一致）
image_paths = [
    str(PROJECT_ROOT / "data/source/frame_000015.jpg"),
    str(PROJECT_ROOT / "data/source/frame_000020.jpg"),
    str(PROJECT_ROOT / "data/source/frame_000025.jpg"),
    str(PROJECT_ROOT / "data/source/frame_000030.jpg"),
    str(PROJECT_ROOT / "data/source/2_frame_000010.jpg"),
    str(PROJECT_ROOT / "data/source/2_frame_000015.jpg"),
    str(PROJECT_ROOT / "data/source/2_frame_000020.jpg"),
    str(PROJECT_ROOT / "data/source/2_frame_000025.jpg"),
]

# 类别名称
class_names = [
    "Using Computer",        # 0
    "Listening Attentively", # 1
    "Taking Notes",          # 2
    "Using Phone",           # 3
    "Bowing the Head",       # 4
    "Lying on the Desk"      # 5
]

# 每个类别的颜色 (BGR)
class_colors = [
    (255, 0, 0),     # 0: 蓝色 - Using Computer
    (0, 255, 0),     # 1: 绿色 - Listening Attentively
    (0, 0, 255),     # 2: 红色 - Taking Notes
    (255, 255, 0),   # 3: 青色 - Using Phone
    (255, 0, 255),   # 4: 紫色 - Bowing the Head
    (0, 255, 255)    # 5: 黄色 - Lying on the Desk
]

def preprocess(image, input_size=640):
    """预处理：resize + normalize"""
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h))
    
    # Pad to input_size x input_size
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # BGR -> RGB, HWC -> CHW, normalize
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    
    return blob, scale, (0, 0)  # pad_x, pad_y

def iou_calc(box1, box2):
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def nms(boxes, nms_threshold=0.45):
    """
    非极大值抑制 (NMS) - 与 C++ 版本一致的实现
    
    Args:
        boxes: [[x1, y1, x2, y2, conf, class_id], ...]
        nms_threshold: IoU 阈值
    
    Returns:
        过滤后的 boxes
    """
    if len(boxes) == 0:
        return []
    
    # 按置信度从高到低排序
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    # 标记是否移除
    removed = [False] * len(boxes)
    final_boxes = []
    
    for i in range(len(boxes)):
        if removed[i]:
            continue
        
        final_boxes.append(boxes[i])
        
        for j in range(i + 1, len(boxes)):
            if removed[j]:
                continue
            
            # 同一类别才做 NMS
            if boxes[i][5] == boxes[j][5]:
                if iou_calc(boxes[i], boxes[j]) > nms_threshold:
                    removed[j] = True
    
    return final_boxes


def postprocess(output, conf_threshold=0.25, nms_threshold=0.45, scale=1.0):
    """后处理：解码 + NMS（与 C++ 版本一致）"""
    # output shape: [1, 10, 8400] 或 [1, 8400, 10]
    print(f"模型输出维度: {output.shape}")
    
    # 检测是否需要转置
    if output.shape[1] < output.shape[2]:
        # 格式: [1, channels, boxes] -> 转置为 [1, boxes, channels]
        output = output.transpose(0, 2, 1)
        print(f"转置后维度: {output.shape}")
    
    output = output[0]  # 去掉 batch 维度 -> [8400, 10]
    
    # Step 1: Decode - 解码所有置信度高于阈值的框
    boxes = []
    for i in range(output.shape[0]):
        cx, cy, w, h = output[i, :4]
        class_scores = output[i, 4:]
        
        class_id = np.argmax(class_scores)
        conf = class_scores[class_id]
        
        if conf > conf_threshold:
            x1 = (cx - w / 2) / scale
            y1 = (cy - h / 2) / scale
            x2 = (cx + w / 2) / scale
            y2 = (cy + h / 2) / scale
            boxes.append([x1, y1, x2, y2, conf, class_id])
    
    print(f"Decode 后 bbox 数量: {len(boxes)}")
    
    # Step 2: NMS - 非极大值抑制
    final_boxes = nms(boxes, nms_threshold)
    print(f"NMS 后 bbox 数量: {len(final_boxes)}")
    
    return final_boxes

# 配置选项
USE_FP16 = True  # 是否使用 FP16
USE_GPU = True   # 是否使用 GPU

# 加载模型
print(f"加载模型: {onnx_path}")

# 选择 Execution Provider
providers = []
if USE_GPU:
    # 优先使用 TensorRT (支持 FP16)
    if USE_FP16:
        trt_options = {
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': str(PROJECT_ROOT / 'models/trt_cache')
        }
        providers.append(('TensorrtExecutionProvider', trt_options))
    # 备选 CUDA
    providers.append('CUDAExecutionProvider')

providers.append('CPUExecutionProvider')

print(f"Providers: {providers}")
session = ort.InferenceSession(onnx_path, providers=providers)
print(f"实际使用: {session.get_providers()}")

# 获取输入输出信息
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name
print(f"输入: {input_name}, shape={input_shape}")
print(f"输出: {output_name}")

# 遍历所有测试图片
for image_path in image_paths:
    # 加载图片
    print(f"\n测试图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图片 {image_path}")
        continue

    # 预处理
    t0 = time.time()
    blob, scale, pad = preprocess(image)
    preprocess_time = (time.time() - t0) * 1000
    print(f"预处理后维度: {blob.shape}, scale={scale:.4f}")

    # 推理
    t1 = time.time()
    outputs = session.run([output_name], {input_name: blob})
    inference_time = (time.time() - t1) * 1000
    output = outputs[0]

    # 后处理（与 C++ 版本一致：conf=0.25, nms=0.45）
    t2 = time.time()
    boxes = postprocess(output, conf_threshold=0.25, nms_threshold=0.45, scale=scale)
    postprocess_time = (time.time() - t2) * 1000

    # 打印耗时
    total_time = preprocess_time + inference_time + postprocess_time
    print(f"\n耗时统计:")
    print(f"  预处理: {preprocess_time:.2f} ms")
    print(f"  推理:   {inference_time:.2f} ms")
    print(f"  后处理: {postprocess_time:.2f} ms")
    print(f"  总计:   {total_time:.2f} ms ({1000/total_time:.1f} FPS)")

    print(f"\n检测到 {len(boxes)} 个物体:")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls_id = box
        cls_id = int(cls_id)
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class{cls_id}"
        color = class_colors[cls_id] if cls_id < len(class_colors) else (128, 128, 128)
        print(f"  物体 {i+1}: {cls_name}, 置信度={conf:.4f}, 坐标=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        # 绘制框
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        # 绘制标签背景
        label = f"{cls_name}: {conf:.2f}"
        font_scale = 1.0
        font_thick = 2
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        cv2.rectangle(image, (int(x1), int(y1)-th-10), (int(x1)+tw, int(y1)), color, -1)
        cv2.putText(image, label, (int(x1), int(y1)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)

    # 保存结果（与 C++ 版本格式一致：原文件名-detect-python.jpg）
    input_filename = Path(image_path).stem  # 获取原文件名（不含扩展名）
    output_path = str(PROJECT_ROOT / f"data/result/{input_filename}-detect-python.jpg")
    cv2.imwrite(output_path, image)
    print(f"\n结果已保存到: {output_path}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 人脸检测器
基于 Ultralytics YOLOv8 实现的实时人脸检测系统
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
import argparse
import time


class YOLOFaceDetector:
    
    def __init__(self, model_path='models/yolov8n-face.pt', conf_threshold=0.5, device='auto'):
        """
        初始化人脸检测器
        
        Args:
            model_path (str): 模型文件路径
            conf_threshold (float): 置信度阈值
            device (str): 运行设备 ('cpu', 'cuda', 'auto')
        """
        self.conf_threshold = conf_threshold
        self.device = device
        
        # 加载模型
        try:
            self.model = YOLO(model_path)
            print(f"✅ 成功加载模型: {model_path}")
            print(f"🔧 使用设备: {self.model.device}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def detect_faces(self, image, visualize=True):
        """
        检测图片中的人脸
        
        Args:
            image: 输入图像 (numpy array 或 PIL Image)
            visualize (bool): 是否可视化检测结果
            
        Returns:
            tuple: (检测结果, 可视化图像)
        """
        # 运行推理 - 专门的人脸检测模型通常只检测人脸
        results = self.model(image, conf=self.conf_threshold)
        
        faces = []
        vis_image = image.copy() if isinstance(image, np.ndarray) else np.array(image)
        
        # 解析检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    face_info = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence)
                    }
                    faces.append(face_info)
                    
                    if visualize:
                        # 绘制边界框
                        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # 添加置信度标签
                        label = f'Face: {confidence:.2f}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(vis_image, (int(x1), int(y1) - label_size[1] - 10), 
                                    (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                        cv2.putText(vis_image, label, (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return faces, vis_image
    
    def detect_video(self, source=0, save_path=None, show=True):
        """
        实时视频人脸检测
        
        Args:
            source: 视频源 (0为摄像头, 也可以是视频文件路径)
            save_path (str): 保存结果视频的路径
            show (bool): 是否显示检测结果
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"❌ 无法打开视频源: {source}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 设置视频写入器
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        print(f"🎥 开始检测 - 按 'q' 退出")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测人脸
                faces, vis_frame = self.detect_faces(frame, visualize=True)
                
                # 添加FPS信息
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps_current = frame_count / elapsed_time
                
                cv2.putText(vis_frame, f'FPS: {fps_current:.1f} | Faces: {len(faces)}', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # 保存结果
                if writer:
                    writer.write(vis_frame)
                
                # 显示结果
                if show:
                    cv2.imshow('YOLOv8 人脸检测', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\n⏹️  检测已停止")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
    
    def batch_detect(self, image_dir, output_dir):
        """
        批量检测图片中的人脸
        
        Args:
            image_dir (str): 输入图片目录
            output_dir (str): 输出结果目录
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ 在 {image_dir} 中未找到图片文件")
            return
        
        print(f"📂 找到 {len(image_files)} 个图片文件")
        
        for i, image_file in enumerate(image_files):
            print(f"🔍 处理 ({i+1}/{len(image_files)}): {image_file.name}")
            
            # 读取图片
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"⚠️  无法读取图片: {image_file}")
                continue
            
            # 检测人脸
            faces, vis_image = self.detect_faces(image, visualize=True)
            
            # 保存结果
            output_file = output_dir / f"detected_{image_file.name}"
            cv2.imwrite(str(output_file), vis_image)
            
            print(f"   ✅ 检测到 {len(faces)} 个人脸，结果已保存到 {output_file}")
        
        print(f"🎉 批量检测完成！结果保存在 {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8 人脸检测器')
    parser.add_argument('--model', type=str, default='models/yolov8n-face.pt',
                       help='人脸检测模型文件路径')
    parser.add_argument('--source', type=str, default='0', 
                       help='输入源 (摄像头ID/视频文件/图片文件/图片目录)')
    parser.add_argument('--output', type=str, default='runs/detect', 
                       help='输出目录')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='auto', 
                       help='运行设备')
    parser.add_argument('--save-video', type=str, 
                       help='保存检测结果视频的路径')
    parser.add_argument('--no-show', action='store_true', 
                       help='不显示检测结果窗口')
    
    args = parser.parse_args()
    
    # 初始化检测器
    detector = YOLOv8FaceDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # 判断输入类型并处理
    source = args.source
    
    # 如果是摄像头
    if source.isdigit():
        detector.detect_video(
            source=int(source),
            save_path=args.save_video,
            show=not args.no_show
        )
    
    # 如果是文件或目录
    else:
        source_path = Path(source)
        
        if not source_path.exists():
            print(f"❌ 路径不存在: {source}")
            return
        
        # 视频文件
        if source_path.is_file() and source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            detector.detect_video(
                source=str(source_path),
                save_path=args.save_video,
                show=not args.no_show
            )
        
        # 图片文件
        elif source_path.is_file() and source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            image = cv2.imread(str(source_path))
            faces, vis_image = detector.detect_faces(image, visualize=True)
            
            # 保存结果
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"detected_{source_path.name}"
            cv2.imwrite(str(output_file), vis_image)
            
            print(f"✅ 检测到 {len(faces)} 个人脸")
            print(f"📁 结果已保存到: {output_file}")
            
            # 显示结果
            if not args.no_show:
                cv2.imshow('检测结果', vis_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        # 图片目录
        elif source_path.is_dir():
            detector.batch_detect(str(source_path), args.output)
        
        else:
            print(f"❌ 不支持的文件类型: {source}")


if __name__ == '__main__':
    main()

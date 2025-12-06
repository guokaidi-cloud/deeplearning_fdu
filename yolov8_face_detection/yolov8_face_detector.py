#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门的YOLOv8人脸检测器
使用专业的yolov8-face模型进行高精度人脸检测
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
import argparse
import time
import sys
import os

# 添加当前目录到路径，以便导入其他脚本
sys.path.append(str(Path(__file__).parent))

from face_detector import YOLOv8FaceDetector


def check_and_download_model(model_path, model_name='yolov8n-face'):
    """
    检查模型是否存在，如果不存在则自动下载
    
    Args:
        model_path (Path): 模型文件路径
        model_name (str): 模型名称
        
    Returns:
        bool: 模型是否可用
    """
    if model_path.exists():
        print(f"✅ 找到人脸检测模型: {model_path}")
        return True
    
    print(f"⚠️  未找到人脸检测模型: {model_path}")
    print(f"🔄 开始自动下载 {model_name} 模型...")
    
    try:
        # 导入下载脚本
        from scripts.download_face_models import download_file
        
        # 人脸检测模型下载链接
        face_model_urls = {
            'yolov8n-face': 'https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt',
            'yolov8s-face': 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt',
        }
        
        if model_name not in face_model_urls:
            print(f"❌ 不支持的模型: {model_name}")
            return False
        
        # 创建模型目录
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 下载模型
        download_file(face_model_urls[model_name], str(model_path))
        
        if model_path.exists() and model_path.stat().st_size > 1024 * 1024:
            print(f"✅ 模型下载成功: {model_path}")
            return True
        else:
            print(f"❌ 模型下载失败或文件损坏")
            return False
            
    except Exception as e:
        print(f"❌ 自动下载失败: {e}")
        print(f"💡 请手动下载模型到: {model_path}")
        if model_name == 'yolov8n-face':
            print(f"   下载链接: https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt")
        else:
            print(f"   下载链接: https://huggingface.co/Bingsu/adetailer/tree/main")
        return False


class YOLOv8SpecializedFaceDetector(YOLOv8FaceDetector):
    """专门的YOLOv8人脸检测器，使用优化的人脸检测模型"""
    
    def __init__(self, model_name='yolov8n-face', conf_threshold=0.3, device='auto', 
                 models_dir='models'):
        """
        初始化专门的人脸检测器
        
        Args:
            model_name (str): 模型名称 ('yolov8n-face', 'yolov8s-face')
            conf_threshold (float): 置信度阈值
            device (str): 运行设备
            models_dir (str): 模型目录
        """
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        
        # 构造模型路径
        model_path = self.models_dir / f"{model_name}.pt"
        
        # 检查并下载模型
        if not check_and_download_model(model_path, model_name):
            raise RuntimeError(f"无法获取人脸检测模型: {model_name}")
        
        # 使用父类初始化
        super().__init__(
            model_path=str(model_path),
            conf_threshold=conf_threshold,
            device=device
        )
        
        print(f"🎯 专业人脸检测器已就绪")
        print(f"📦 模型: {model_name}")
        print(f"🎚️  置信度阈值: {conf_threshold}")
    
    def detect_faces(self, image, visualize=True):
        """
        检测图片中的人脸（优化版本）
        
        Args:
            image: 输入图像
            visualize (bool): 是否可视化检测结果
            
        Returns:
            tuple: (检测结果, 可视化图像)
        """
        # 运行推理
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
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
                        # 绘制边界框（使用更显眼的颜色）
                        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                        
                        # 添加置信度标签
                        label = f'Face: {confidence:.3f}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(vis_image, (int(x1), int(y1) - label_size[1] - 10), 
                                    (int(x1) + label_size[0], int(y1)), (0, 255, 255), -1)
                        cv2.putText(vis_image, label, (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return faces, vis_image


def process_video_with_yolov8(detector, video_path, output_path=None, show_video=False, 
                              max_frames=None, start_time=None, end_time=None):
    """
    使用YOLOv8处理视频文件进行人脸检测
    """
    print(f"🎥 开始处理视频: {video_path}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 视频信息: {width}x{height}, {fps}FPS, 总帧数={total_frames}, 时长={duration:.1f}秒")
    
    # 解析时间参数（简化版本，与opencv_face_detector保持一致）
    start_frame = 0
    end_frame = total_frames
    
    if start_time:
        try:
            if ':' in start_time:
                parts = start_time.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(float, parts)
                    start_seconds = hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:  # MM:SS
                    minutes, seconds = map(float, parts)
                    start_seconds = minutes * 60 + seconds
            else:
                start_seconds = float(start_time)
            
            start_frame = int(start_seconds * fps)
            start_frame = max(0, min(start_frame, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"⏩ 跳转到开始时间: {start_seconds:.1f}秒 (第{start_frame}帧)")
        except:
            print(f"⚠️  无效的开始时间格式: {start_time}")
    
    if end_time:
        try:
            if ':' in end_time:
                parts = end_time.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(float, parts)
                    end_seconds = hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:  # MM:SS
                    minutes, seconds = map(float, parts)
                    end_seconds = minutes * 60 + seconds
            else:
                end_seconds = float(end_time)
                
            end_frame = int(end_seconds * fps)
            end_frame = max(start_frame, min(end_frame, total_frames))
            print(f"⏹️  结束时间: {end_seconds:.1f}秒 (第{end_frame}帧)")
        except:
            print(f"⚠️  无效的结束时间格式: {end_time}")
    
    # 计算处理帧数
    process_frames = end_frame - start_frame
    if max_frames:
        process_frames = min(process_frames, max_frames)
    
    # 设置视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"📁 输出视频: {output_path}")
    
    # 处理统计
    processed_frames = 0
    total_faces = 0
    process_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 获取当前帧位置
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time_sec = current_frame / fps if fps > 0 else 0
            
            # 检查处理限制
            if processed_frames >= process_frames:
                break
                
            if max_frames and processed_frames >= max_frames:
                print(f"⏹️  已达到最大处理帧数: {max_frames}")
                break
            
            # 检测人脸
            faces, vis_frame = detector.detect_faces(frame, visualize=True)
            total_faces += len(faces)
            
            # 添加统计信息
            elapsed_time = time.time() - process_start_time
            current_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
            
            hours = int(current_time_sec // 3600)
            minutes = int((current_time_sec % 3600) // 60)
            seconds = int(current_time_sec % 60)
            
            stats_text = [
                f'Time: {hours:02d}:{minutes:02d}:{seconds:02d} (Frame: {current_frame})',
                f'Progress: {processed_frames+1}/{process_frames}',
                f'Current Faces: {len(faces)}',
                f'Total Faces: {total_faces}',
                f'Processing FPS: {current_fps:.1f}'
            ]
            
            for i, text in enumerate(stats_text):
                y_pos = 30 + i * 25
                cv2.putText(vis_frame, text, (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 保存帧
            if writer:
                writer.write(vis_frame)
            
            # 显示视频
            if show_video:
                cv2.imshow('YOLOv8人脸检测', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("👤 用户按'q'键退出")
                    break
            
            processed_frames += 1
            
            # 定期输出进度
            if processed_frames % 100 == 0:
                progress = (processed_frames / process_frames) * 100
                current_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                print(f"📈 处理进度: {processed_frames}/{process_frames} "
                      f"({progress:.1f}%) | 时间: {current_time_str} | 检测到人脸: {total_faces}")
    
    except KeyboardInterrupt:
        print("\n⏹️  用户中断处理")
    
    finally:
        # 清理资源
        cap.release()
        if writer:
            writer.release()
        if show_video:
            cv2.destroyAllWindows()
    
    # 输出统计信息
    elapsed_time = time.time() - process_start_time
    avg_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n📊 处理完成统计:")
    print(f"   ⏱️  处理时间: {elapsed_time:.1f}秒")
    print(f"   📹 处理帧数: {processed_frames}")
    print(f"   🎯 检测人脸: {total_faces}")
    print(f"   ⚡ 平均FPS: {avg_fps:.1f}")
    print(f"   📏 平均每帧人脸数: {total_faces/processed_frames:.1f}" if processed_frames > 0 else "")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8专业人脸检测器')
    parser.add_argument('--input', type=str, required=True,
                       help='输入视频文件路径')
    parser.add_argument('--output', type=str, 
                       help='输出视频文件路径')
    parser.add_argument('--show', action='store_true',
                       help='显示检测过程')
    parser.add_argument('--max-frames', type=int,
                       help='最大处理帧数 (用于测试)')
    parser.add_argument('--start-time', type=str,
                       help='开始时间 (秒数或 HH:MM:SS 格式)')
    parser.add_argument('--end-time', type=str, 
                       help='结束时间 (秒数或 HH:MM:SS 格式)')
    parser.add_argument('--model', type=str, default='yolov8n-face',
                       choices=['yolov8n-face', 'yolov8s-face'],
                       help='人脸检测模型名称')
    parser.add_argument('--conf', type=float, default=0.3, 
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='auto', 
                       help='运行设备')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='模型存放目录')
    
    args = parser.parse_args()
    
    try:
        # 初始化专业人脸检测器
        print(f"🚀 初始化YOLOv8专业人脸检测器...")
        detector = YOLOv8SpecializedFaceDetector(
            model_name=args.model,
            conf_threshold=args.conf,
            device=args.device,
            models_dir=args.models_dir
        )
        
        # 检查输入文件
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ 输入文件不存在: {args.input}")
            return
        
        # 设置输出路径
        output_path = args.output
        if not output_path:
            output_path = input_path.parent / f"yolov8_detected_{input_path.name}"
        
        # 处理视频文件
        if input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # 创建专用的视频处理方法
            process_video_with_yolov8(
                detector=detector,
                video_path=input_path,
                output_path=output_path,
                show_video=args.show,
                max_frames=args.max_frames,
                start_time=args.start_time,
                end_time=args.end_time
            )
        
        # 处理图片文件
        elif input_path.is_file() and input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            image = cv2.imread(str(input_path))
            faces, vis_image = detector.detect_faces(image, visualize=True)
            
            # 保存结果
            if not output_path:
                output_path = input_path.parent / f"yolov8_detected_{input_path.name}"
            
            cv2.imwrite(str(output_path), vis_image)
            
            print(f"✅ 检测到 {len(faces)} 个人脸")
            for i, face in enumerate(faces):
                bbox = face['bbox']
                conf = face['confidence']
                print(f"   人脸{i+1}: 坐标{bbox}, 置信度{conf:.3f}")
            
            print(f"📁 结果已保存到: {output_path}")
            
            # 显示结果
            if args.show:
                cv2.imshow('YOLOv8专业人脸检测', vis_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        else:
            print(f"❌ 不支持的文件类型: {input_path}")
    
    except Exception as e:
        print(f"❌ 检测过程中出现错误: {e}")
        print("💡 建议:")
        print("   1. 检查输入文件是否存在")
        print("   2. 确认网络连接正常（用于下载模型）")
        print("   3. 检查设备和CUDA环境")


if __name__ == '__main__':
    main()

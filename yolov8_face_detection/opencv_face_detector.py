#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 专业人脸检测器
使用 Haar Cascade 分类器进行高精度人脸检测
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import time


class OpenCVFaceDetector:
    """OpenCV 人脸检测器类"""
    
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        初始化 OpenCV 人脸检测器
        
        Args:
            scale_factor (float): 图像缩放因子
            min_neighbors (int): 最少邻居数量
            min_size (tuple): 最小人脸尺寸
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # 加载人脸检测器
        try:
            # 正面人脸检测器
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # 侧面人脸检测器 (备用)
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            
            print("✅ OpenCV 人脸检测器加载成功")
            print(f"🔧 参数: scale_factor={scale_factor}, min_neighbors={min_neighbors}, min_size={min_size}")
            
        except Exception as e:
            print(f"❌ 人脸检测器加载失败: {e}")
            raise
    
    def detect_faces(self, image, detect_profile=True):
        """
        检测图像中的人脸
        
        Args:
            image: 输入图像
            detect_profile (bool): 是否检测侧面人脸
            
        Returns:
            list: 检测到的人脸边界框列表 [(x, y, w, h), ...]
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 直方图均衡化，增强对比度
        gray = cv2.equalizeHist(gray)
        
        # 检测正面人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 检测侧面人脸 (可选)
        if detect_profile:
            profile_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 合并检测结果
            if len(profile_faces) > 0:
                if len(faces) > 0:
                    faces = np.vstack((faces, profile_faces))
                else:
                    faces = profile_faces
        
        return faces
    
    def _parse_time(self, time_input):
        """
        解析时间输入，支持秒数或 HH:MM:SS 格式
        
        Args:
            time_input: 时间输入 (float, int, 或 "HH:MM:SS" 字符串)
            
        Returns:
            float: 时间（秒）
        """
        if time_input is None:
            return None
            
        if isinstance(time_input, (int, float)):
            return float(time_input)
            
        if isinstance(time_input, str):
            # 解析 HH:MM:SS 或 MM:SS 格式
            parts = time_input.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds
            else:  # 只有秒数
                return float(parts[0])
        
        return None
    
    def process_video(self, video_path, output_path=None, show_video=False, max_frames=None, 
                     start_time=None, end_time=None):
        """
        处理视频文件进行人脸检测
        
        Args:
            video_path (str): 输入视频路径
            output_path (str): 输出视频路径
            show_video (bool): 是否显示视频窗口
            max_frames (int): 最大处理帧数 (None表示处理全部)
            start_time (float): 开始时间(秒) 或 时间字符串 "HH:MM:SS"
            end_time (float): 结束时间(秒) 或 时间字符串 "HH:MM:SS"
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
        
        # 解析开始和结束时间
        start_seconds = self._parse_time(start_time)
        end_seconds = self._parse_time(end_time)
        
        # 计算开始和结束帧
        start_frame = 0
        end_frame = total_frames
        
        if start_seconds is not None:
            start_frame = int(start_seconds * fps)
            start_frame = max(0, min(start_frame, total_frames - 1))
            print(f"⏩ 跳转到开始时间: {start_seconds:.1f}秒 (第{start_frame}帧)")
            
        if end_seconds is not None:
            end_frame = int(end_seconds * fps)
            end_frame = max(start_frame, min(end_frame, total_frames))
            print(f"⏹️  结束时间: {end_seconds:.1f}秒 (第{end_frame}帧)")
        
        # 跳转到开始位置
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            actual_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            actual_time = actual_pos / fps if fps > 0 else 0
            print(f"✅ 实际跳转到: 第{actual_pos:.0f}帧, {actual_time:.1f}秒")
        
        # 计算实际处理的帧数范围
        process_frames = end_frame - start_frame
        if max_frames:
            process_frames = min(process_frames, max_frames)
            
        print(f"🎯 将处理 {process_frames} 帧 (从第{start_frame}帧到第{start_frame + process_frames}帧)")
        
        # 设置视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"📁 输出视频: {output_path}")
        
        # 处理统计
        frame_count = 0
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
                
                # 检查是否超出结束时间
                if end_seconds is not None and current_time_sec > end_seconds:
                    print(f"⏹️  已达到结束时间: {end_seconds:.1f}秒")
                    break
                
                # 检查最大帧数限制
                if max_frames and processed_frames >= max_frames:
                    print(f"⏹️  已达到最大处理帧数: {max_frames}")
                    break
                
                # 检查是否超出计划处理的帧数
                if processed_frames >= process_frames:
                    print(f"⏹️  已完成计划处理的帧数: {process_frames}")
                    break
                
                # 检测人脸
                faces = self.detect_faces(frame, detect_profile=True)
                total_faces += len(faces)
                
                # 绘制检测结果
                result_frame = frame.copy()
                for (x, y, w, h) in faces:
                    # 绘制人脸框
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 添加置信度标签
                    confidence = 0.95  # OpenCV检测器没有置信度，设置固定值
                    label = f'Face: {confidence:.2f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # 绘制标签背景
                    cv2.rectangle(result_frame, (x, y - label_size[1] - 10), 
                                (x + label_size[0], y), (0, 255, 0), -1)
                    
                    # 绘制标签文字
                    cv2.putText(result_frame, label, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # 添加统计信息
                elapsed_time = time.time() - process_start_time
                current_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
                
                # 计算当前时间戳
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
                    cv2.putText(result_frame, text, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # 保存帧
                if writer:
                    writer.write(result_frame)
                
                # 显示视频
                if show_video:
                    cv2.imshow('人脸检测', result_frame)
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
    parser = argparse.ArgumentParser(description='OpenCV 专业人脸检测器')
    parser.add_argument('--input', type=str, required=True,
                       help='输入视频文件路径')
    parser.add_argument('--output', type=str, 
                       help='输出视频文件路径')
    parser.add_argument('--show', action='store_true',
                       help='显示检测过程')
    parser.add_argument('--max-frames', type=int,
                       help='最大处理帧数 (用于测试)')
    parser.add_argument('--scale-factor', type=float, default=1.1,
                       help='检测缩放因子')
    parser.add_argument('--min-neighbors', type=int, default=5,
                       help='最小邻居数')
    parser.add_argument('--min-size', type=int, nargs=2, default=[30, 30],
                       help='最小人脸尺寸 [width height]')
    parser.add_argument('--start-time', type=str,
                       help='开始时间 (秒数或 HH:MM:SS 格式)')
    parser.add_argument('--end-time', type=str, 
                       help='结束时间 (秒数或 HH:MM:SS 格式)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    # 设置输出路径
    output_path = args.output
    if not output_path:
        output_path = input_path.parent / f"opencv_detected_{input_path.name}"
    
    try:
        # 创建检测器
        detector = OpenCVFaceDetector(
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=tuple(args.min_size)
        )
        
        # 处理视频
        detector.process_video(
            video_path=input_path,
            output_path=output_path,
            show_video=args.show,
            max_frames=args.max_frames,
            start_time=args.start_time,
            end_time=args.end_time
        )
        
        print(f"🎉 处理完成! 结果保存在: {output_path}")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")


if __name__ == '__main__':
    main()

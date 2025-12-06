#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 人脸检测演示脚本
快速演示人脸检测功能
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from face_detector import YOLOv8FaceDetector


def create_demo_image():
    """创建一个演示图像"""
    # 创建一个简单的演示图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # 添加文字说明
    text = "YOLOv8 Face Detection Demo"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 0)
    thickness = 2
    
    # 计算文字位置
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    
    # 添加提示
    instructions = [
        "Instructions:",
        "1. Press SPACE to start webcam detection", 
        "2. Press 'q' to quit",
        "3. Press 's' to save current frame"
    ]
    
    for i, instruction in enumerate(instructions):
        y_pos = text_y + 80 + i * 30
        cv2.putText(img, instruction, (50, y_pos), font, 0.6, color, 1)
    
    return img


def demo_webcam(detector):
    """演示摄像头检测"""
    print("🎥 启动摄像头演示...")
    print("📝 按空格键开始检测，按'q'退出，按's'保存当前帧")
    
    # 显示演示图像
    demo_img = create_demo_image()
    cv2.imshow('YOLOv8 人脸检测演示', demo_img)
    
    # 等待用户按键
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格键开始检测
            break
        elif key == ord('q'):  # 退出
            cv2.destroyAllWindows()
            return
    
    # 启动摄像头检测
    detector.detect_video(source=0, show=True)


def demo_image(detector, image_path=None):
    """演示图像检测"""
    if image_path and Path(image_path).exists():
        print(f"🖼️  检测图像: {image_path}")
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print("❌ 无法读取图像文件")
            return
        
        # 检测人脸
        faces, vis_image = detector.detect_faces(image, visualize=True)
        
        # 显示结果
        print(f"✅ 检测到 {len(faces)} 个人脸")
        for i, face in enumerate(faces):
            bbox = face['bbox']
            conf = face['confidence']
            print(f"   人脸 {i+1}: 坐标 {bbox}, 置信度 {conf:.2f}")
        
        # 显示图像
        cv2.imshow('检测结果', vis_image)
        print("按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        output_path = Path('demo_result.jpg')
        cv2.imwrite(str(output_path), vis_image)
        print(f"📁 结果已保存到: {output_path}")
    
    else:
        print("❌ 图像文件不存在或未指定")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8 人脸检测演示')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='模型文件路径')
    parser.add_argument('--image', type=str,
                       help='演示图像路径')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='auto',
                       help='运行设备')
    parser.add_argument('--webcam-only', action='store_true',
                       help='仅演示摄像头检测')
    
    args = parser.parse_args()
    
    print("🚀 YOLOv8 人脸检测系统演示")
    print("=" * 50)
    
    try:
        # 初始化检测器
        print(f"📦 加载模型: {args.model}")
        detector = YOLOv8FaceDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            device=args.device
        )
        print("✅ 模型加载成功!")
        
        if args.webcam_only:
            # 仅演示摄像头
            demo_webcam(detector)
        elif args.image:
            # 演示图像检测
            demo_image(detector, args.image)
        else:
            # 交互式演示菜单
            while True:
                print("\n📋 演示选项:")
                print("1. 摄像头实时检测")
                print("2. 图像文件检测") 
                print("3. 退出演示")
                
                choice = input("请选择 (1-3): ").strip()
                
                if choice == '1':
                    demo_webcam(detector)
                elif choice == '2':
                    img_path = input("请输入图像文件路径: ").strip()
                    demo_image(detector, img_path)
                elif choice == '3':
                    break
                else:
                    print("❌ 无效选择，请重新输入")
        
        print("\n🎉 演示结束，感谢使用!")
        
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

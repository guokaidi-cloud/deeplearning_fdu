#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载专门的YOLOv8人脸检测模型
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import argparse


def download_file(url, filename, chunk_size=8192):
    """
    下载文件并显示进度条
    
    Args:
        url (str): 下载链接
        filename (str): 保存文件名
        chunk_size (int): 块大小
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=f"下载 {Path(filename).name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ 下载完成: {filename}")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='下载YOLOv8人脸检测模型')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='模型保存目录')
    parser.add_argument('--model', type=str, default='yolov8n-face', 
                       choices=['yolov8n-face', 'yolov8s-face', 'yolov8m-face'],
                       help='要下载的人脸检测模型')
    
    args = parser.parse_args()
    
    # 创建模型目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLOv8人脸检测模型下载链接
    # 这些是一些流行的YOLOv8人脸检测模型
    face_model_urls = {
        'yolov8n-face': 'https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt',
        'yolov8s-face': 'https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8s-face.pt',
        'yolov8m-face': 'https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8m-face.pt',
    }
    
    # 备用下载源
    backup_urls = {
        'yolov8n-face': [
            'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt',
            'https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt'
        ],
        'yolov8s-face': [
            'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt',
        ]
    }
    
    model_name = args.model
    model_file = save_dir / f"{model_name}.pt"
    
    # 检查文件是否已存在
    if model_file.exists():
        print(f"⚠️  模型已存在，跳过: {model_file}")
        return
    
    print(f"🚀 开始下载 {model_name} 人脸检测模型...")
    
    # 尝试主要下载源
    success = False
    if model_name in face_model_urls:
        try:
            print(f"📥 从主要源下载 {model_name}...")
            download_file(face_model_urls[model_name], str(model_file))
            success = True
        except Exception as e:
            print(f"❌ 主要源下载失败: {e}")
    
    # 如果主要源失败，尝试备用源
    if not success and model_name in backup_urls:
        for i, backup_url in enumerate(backup_urls[model_name]):
            try:
                print(f"📥 从备用源 {i+1} 下载 {model_name}...")
                download_file(backup_url, str(model_file))
                success = True
                break
            except Exception as e:
                print(f"❌ 备用源 {i+1} 下载失败: {e}")
    
    if success:
        print(f"🎉 {model_name} 模型下载完成!")
        print(f"📁 保存位置: {model_file}")
        
        # 验证模型文件
        if model_file.stat().st_size > 1024 * 1024:  # 至少1MB
            print(f"✅ 模型文件大小: {model_file.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"⚠️  警告: 模型文件似乎太小: {model_file.stat().st_size} bytes")
    else:
        print(f"❌ 所有下载源都失败了，无法下载 {model_name}")
        print("💡 建议:")
        print("   1. 检查网络连接")
        print("   2. 手动从 GitHub 或 Hugging Face 下载模型")
        print("   3. 使用 OpenCV 人脸检测作为备选方案")


if __name__ == '__main__':
    main()

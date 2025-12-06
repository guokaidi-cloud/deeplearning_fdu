#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载预训练模型脚本
自动下载YOLOv8预训练权重文件
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
    parser = argparse.ArgumentParser(description='下载YOLOv8预训练模型')
    parser.add_argument('--model-sizes', nargs='+', default=['n', 's'], 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='要下载的模型大小')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='模型保存目录')
    
    args = parser.parse_args()
    
    # 创建模型目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLOv8模型下载链接
    base_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    model_urls = {
        'n': f"{base_url}yolov8n.pt",
        's': f"{base_url}yolov8s.pt", 
        'm': f"{base_url}yolov8m.pt",
        'l': f"{base_url}yolov8l.pt",
        'x': f"{base_url}yolov8x.pt"
    }
    
    print("🚀 开始下载YOLOv8预训练模型...")
    
    for size in args.model_sizes:
        if size in model_urls:
            model_file = save_dir / f"yolov8{size}.pt"
            
            # 检查文件是否已存在
            if model_file.exists():
                print(f"⚠️  模型已存在，跳过: {model_file}")
                continue
            
            try:
                print(f"📥 下载 YOLOv8{size.upper()} 模型...")
                download_file(model_urls[size], str(model_file))
            except Exception as e:
                print(f"❌ 下载 YOLOv8{size.upper()} 失败: {e}")
    
    print("🎉 预训练模型下载完成!")


if __name__ == '__main__':
    main()

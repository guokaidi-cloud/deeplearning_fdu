#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理和格式转换工具
支持多种人脸数据集格式转换为YOLO格式
"""

import os
import cv2
import json
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import argparse
from tqdm import tqdm
import shutil
import random


class DataProcessor:
    """数据处理器基类"""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建YOLO格式目录结构
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_bbox(self, bbox, img_width, img_height):
        """
        将边界框坐标标准化为YOLO格式 (x_center, y_center, width, height)
        
        Args:
            bbox: 边界框 [x1, y1, x2, y2] 或 [x1, y1, width, height]
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            tuple: YOLO格式的归一化坐标 (x_center, y_center, width, height)
        """
        x1, y1, w_or_x2, h_or_y2 = bbox
        
        # 判断输入格式并转换为 [x1, y1, x2, y2]
        if w_or_x2 > img_width or h_or_y2 > img_height:
            # 很可能是 [x1, y1, width, height] 格式
            x2 = x1 + w_or_x2
            y2 = y1 + h_or_y2
        else:
            # 可能是 [x1, y1, x2, y2] 格式
            x2, y2 = w_or_x2, h_or_y2
        
        # 确保坐标在有效范围内
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # 计算中心点和尺寸
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        
        # 归一化
        norm_x = center_x / img_width
        norm_y = center_y / img_height
        norm_width = width / img_width
        norm_height = height / img_height
        
        return norm_x, norm_y, norm_width, norm_height
    
    def save_yolo_annotation(self, image_name, bboxes, class_id=0):
        """
        保存YOLO格式的标注文件
        
        Args:
            image_name (str): 图像文件名
            bboxes (list): 边界框列表，每个元素为 (x_center, y_center, width, height)
            class_id (int): 类别ID
        """
        annotation_file = self.labels_dir / f"{Path(image_name).stem}.txt"
        
        with open(annotation_file, 'w') as f:
            for bbox in bboxes:
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


class WIDERFaceProcessor(DataProcessor):
    """WIDER FACE数据集处理器"""
    
    def process(self, annotation_file, split_name='train'):
        """
        处理WIDER FACE数据集
        
        Args:
            annotation_file (str): 标注文件路径
            split_name (str): 数据集分割名称
        """
        print(f"🔄 处理WIDER FACE数据集: {split_name}")
        
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        processed_count = 0
        
        with tqdm(total=len(lines), desc=f"处理{split_name}集") as pbar:
            while i < len(lines):
                # 读取图像文件名
                img_name = lines[i].strip()
                if not img_name.endswith(('.jpg', '.png', '.jpeg')):
                    i += 1
                    pbar.update(1)
                    continue
                
                i += 1
                
                # 读取人脸数量
                if i >= len(lines):
                    break
                
                try:
                    face_count = int(lines[i].strip())
                except:
                    i += 1
                    pbar.update(1)
                    continue
                
                i += 1
                
                # 读取图像
                img_path = self.input_dir / 'images' / img_name
                if not img_path.exists():
                    # 尝试其他可能的路径
                    img_path = self.input_dir / img_name
                
                if not img_path.exists():
                    # 跳过不存在的图像
                    i += face_count
                    pbar.update(1)
                    continue
                
                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        i += face_count
                        pbar.update(1)
                        continue
                    
                    img_height, img_width = image.shape[:2]
                except:
                    i += face_count
                    pbar.update(1)
                    continue
                
                # 读取边界框
                bboxes = []
                for j in range(face_count):
                    if i >= len(lines):
                        break
                    
                    parts = lines[i].strip().split()
                    if len(parts) >= 4:
                        try:
                            x, y, w, h = map(float, parts[:4])
                            # 转换为YOLO格式
                            norm_bbox = self.normalize_bbox([x, y, w, h], img_width, img_height)
                            bboxes.append(norm_bbox)
                        except:
                            pass
                    
                    i += 1
                    pbar.update(1)
                
                # 保存处理后的数据
                if bboxes:
                    # 复制图像
                    output_img_path = self.images_dir / f"{split_name}_{Path(img_name).name}"
                    shutil.copy2(img_path, output_img_path)
                    
                    # 保存标注
                    self.save_yolo_annotation(output_img_path.name, bboxes, class_id=0)
                    processed_count += 1
        
        print(f"✅ 处理完成: {processed_count} 张图像")


class COCOProcessor(DataProcessor):
    """COCO格式数据集处理器"""
    
    def process(self, annotation_file, image_dir=None):
        """
        处理COCO格式数据集
        
        Args:
            annotation_file (str): COCO标注JSON文件路径
            image_dir (str): 图像目录路径
        """
        print("🔄 处理COCO格式数据集")
        
        if image_dir is None:
            image_dir = self.input_dir / 'images'
        else:
            image_dir = Path(image_dir)
        
        # 读取COCO标注文件
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # 创建图像ID到文件名的映射
        image_info = {img['id']: img for img in coco_data['images']}
        
        # 创建类别映射 (假设人脸类别ID为1或者'person')
        face_categories = []
        for cat in coco_data['categories']:
            if 'face' in cat['name'].lower() or 'person' in cat['name'].lower():
                face_categories.append(cat['id'])
        
        if not face_categories:
            print("⚠️  未找到人脸相关类别，使用所有类别")
            face_categories = [cat['id'] for cat in coco_data['categories']]
        
        # 处理标注
        processed_images = set()
        
        for annotation in tqdm(coco_data['annotations'], desc="处理标注"):
            if annotation['category_id'] not in face_categories:
                continue
            
            image_id = annotation['image_id']
            if image_id not in image_info:
                continue
            
            image_data = image_info[image_id]
            img_name = image_data['file_name']
            img_path = image_dir / img_name
            
            if not img_path.exists():
                continue
            
            # 获取图像尺寸
            img_width = image_data['width']
            img_height = image_data['height']
            
            # 处理边界框
            bbox = annotation['bbox']  # [x, y, width, height]
            norm_bbox = self.normalize_bbox(bbox, img_width, img_height)
            
            # 复制图像 (只复制一次)
            if image_id not in processed_images:
                output_img_path = self.images_dir / img_name
                shutil.copy2(img_path, output_img_path)
                processed_images.add(image_id)
            
            # 保存标注 (追加模式)
            annotation_file = self.labels_dir / f"{Path(img_name).stem}.txt"
            with open(annotation_file, 'a') as f:
                x_center, y_center, width, height = norm_bbox
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"✅ 处理完成: {len(processed_images)} 张图像")


class PascalVOCProcessor(DataProcessor):
    """Pascal VOC格式数据集处理器"""
    
    def process(self, annotations_dir, images_dir=None):
        """
        处理Pascal VOC格式数据集
        
        Args:
            annotations_dir (str): XML标注文件目录
            images_dir (str): 图像文件目录
        """
        print("🔄 处理Pascal VOC格式数据集")
        
        annotations_dir = Path(annotations_dir)
        if images_dir is None:
            images_dir = self.input_dir / 'images'
        else:
            images_dir = Path(images_dir)
        
        xml_files = list(annotations_dir.glob('*.xml'))
        processed_count = 0
        
        for xml_file in tqdm(xml_files, desc="处理XML文件"):
            try:
                # 解析XML
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 获取图像信息
                filename = root.find('filename').text
                img_path = images_dir / filename
                
                if not img_path.exists():
                    continue
                
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
                
                # 处理标注对象
                bboxes = []
                for obj in root.findall('object'):
                    class_name = obj.find('name').text.lower()
                    
                    # 只处理人脸相关类别
                    if 'face' not in class_name and 'person' not in class_name and 'head' not in class_name:
                        continue
                    
                    # 获取边界框
                    bbox_elem = obj.find('bndbox')
                    xmin = float(bbox_elem.find('xmin').text)
                    ymin = float(bbox_elem.find('ymin').text)
                    xmax = float(bbox_elem.find('xmax').text)
                    ymax = float(bbox_elem.find('ymax').text)
                    
                    # 转换为YOLO格式
                    norm_bbox = self.normalize_bbox([xmin, ymin, xmax, ymax], img_width, img_height)
                    bboxes.append(norm_bbox)
                
                # 保存处理后的数据
                if bboxes:
                    # 复制图像
                    output_img_path = self.images_dir / filename
                    shutil.copy2(img_path, output_img_path)
                    
                    # 保存标注
                    self.save_yolo_annotation(filename, bboxes, class_id=0)
                    processed_count += 1
                    
            except Exception as e:
                print(f"⚠️  处理文件 {xml_file} 时出错: {e}")
                continue
        
        print(f"✅ 处理完成: {processed_count} 张图像")


class DataSplitter:
    """数据集分割工具"""
    
    @staticmethod
    def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        """
        将数据集分割为训练集、验证集和测试集
        
        Args:
            data_dir (str): 数据目录路径
            train_ratio (float): 训练集比例
            val_ratio (float): 验证集比例  
            test_ratio (float): 测试集比例
            seed (int): 随机种子
        """
        data_dir = Path(data_dir)
        images_dir = data_dir / 'images'
        labels_dir = data_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print("❌ 数据目录结构不正确")
            return
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_dir.glob(ext))
        
        if not image_files:
            print("❌ 未找到图像文件")
            return
        
        # 过滤有对应标注文件的图像
        valid_images = []
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                valid_images.append(img_file)
        
        print(f"📊 找到 {len(valid_images)} 个有效的图像-标注对")
        
        # 随机打乱
        random.seed(seed)
        random.shuffle(valid_images)
        
        # 计算分割点
        total_count = len(valid_images)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        
        # 分割数据
        train_files = valid_images[:train_count]
        val_files = valid_images[train_count:train_count + val_count]
        test_files = valid_images[train_count + val_count:]
        
        print(f"📈 数据分割: 训练集={len(train_files)}, 验证集={len(val_files)}, 测试集={len(test_files)}")
        
        # 创建分割后的目录结构
        for split_name, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
            if not file_list:
                continue
                
            split_images_dir = data_dir / split_name / 'images'
            split_labels_dir = data_dir / split_name / 'labels'
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            for img_file in tqdm(file_list, desc=f"复制{split_name}集"):
                # 复制图像
                shutil.copy2(img_file, split_images_dir / img_file.name)
                
                # 复制标注
                label_file = labels_dir / f"{img_file.stem}.txt"
                shutil.copy2(label_file, split_labels_dir / f"{img_file.stem}.txt")
        
        print("✅ 数据集分割完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='人脸数据集预处理工具')
    parser.add_argument('--format', type=str, required=True,
                       choices=['wider', 'coco', 'voc'],
                       help='输入数据集格式')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='输入数据目录')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='输出数据目录')
    parser.add_argument('--annotation-file', type=str,
                       help='标注文件路径 (WIDER/COCO格式需要)')
    parser.add_argument('--image-dir', type=str,
                       help='图像目录路径 (可选)')
    parser.add_argument('--split-data', action='store_true',
                       help='是否分割数据集')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='测试集比例')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.format in ['wider', 'coco'] and not args.annotation_file:
        print("❌ WIDER FACE和COCO格式需要提供标注文件路径")
        return
    
    # 创建处理器
    if args.format == 'wider':
        processor = WIDERFaceProcessor(args.input_dir, args.output_dir)
        processor.process(args.annotation_file)
    elif args.format == 'coco':
        processor = COCOProcessor(args.input_dir, args.output_dir)
        processor.process(args.annotation_file, args.image_dir)
    elif args.format == 'voc':
        processor = PascalVOCProcessor(args.input_dir, args.output_dir)
        annotations_dir = args.annotation_file or (Path(args.input_dir) / 'annotations')
        processor.process(annotations_dir, args.image_dir)
    
    # 数据集分割
    if args.split_data:
        DataSplitter.split_dataset(
            data_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    
    print("🎉 数据预处理完成!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 人脸检测模型训练脚本
用于训练自定义的人脸检测模型
"""

import argparse
import os
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd


class YOLOv8FaceTrainer:
    """YOLOv8 人脸检测模型训练器"""
    
    def __init__(self, model_size='n', pretrained=True):
        """
        初始化训练器
        
        Args:
            model_size (str): 模型大小 ('n', 's', 'm', 'l', 'x')
            pretrained (bool): 是否使用预训练权重
        """
        self.model_size = model_size
        self.pretrained = pretrained
        
        # 模型文件映射
        self.model_files = {
            'n': 'yolov8n.pt',
            's': 'yolov8s.pt', 
            'm': 'yolov8m.pt',
            'l': 'yolov8l.pt',
            'x': 'yolov8x.pt'
        }
        
        # 初始化模型
        model_file = self.model_files.get(model_size, 'yolov8n.pt')
        self.model = YOLO(model_file if pretrained else f'yolov8{model_size}.yaml')
        
        print(f"🚀 初始化 YOLOv8{model_size.upper()} 模型")
        print(f"📦 预训练权重: {'✅' if pretrained else '❌'}")
    
    def create_dataset_config(self, train_dir, val_dir, test_dir=None, class_names=['face']):
        """
        创建数据集配置文件
        
        Args:
            train_dir (str): 训练数据目录
            val_dir (str): 验证数据目录  
            test_dir (str): 测试数据目录(可选)
            class_names (list): 类别名称列表
            
        Returns:
            str: 配置文件路径
        """
        config = {
            'path': str(Path(train_dir).parent.absolute()),
            'train': str(Path(train_dir).relative_to(Path(train_dir).parent)),
            'val': str(Path(val_dir).relative_to(Path(train_dir).parent)),
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        if test_dir:
            config['test'] = str(Path(test_dir).relative_to(Path(train_dir).parent))
        
        # 保存配置文件
        config_path = Path('configs/face_dataset.yaml')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"📝 数据集配置已保存: {config_path}")
        return str(config_path)
    
    def train(self, data_config, epochs=100, imgsz=640, batch_size=16, lr0=0.01, 
              save_dir='runs/train', device='auto', workers=8, patience=50,
              resume=False, pretrained=True):
        """
        训练模型
        
        Args:
            data_config (str): 数据集配置文件路径
            epochs (int): 训练轮数
            imgsz (int): 输入图像尺寸
            batch_size (int): 批次大小
            lr0 (float): 初始学习率
            save_dir (str): 保存目录
            device (str): 训练设备
            workers (int): 数据加载线程数
            patience (int): 早停耐心值
            resume (bool): 是否从断点继续训练
            pretrained (bool): 是否使用预训练权重
        """
        print("🔥 开始训练...")
        
        try:
            # 训练参数
            train_args = {
                'data': data_config,
                'epochs': epochs,
                'imgsz': imgsz,
                'batch': batch_size,
                'lr0': lr0,
                'project': save_dir,
                'device': device,
                'workers': workers,
                'patience': patience,
                'save_period': 10,  # 每10个epoch保存一次
                'val': True,
                'plots': True,
                'verbose': True
            }
            
            if resume:
                train_args['resume'] = True
            
            # 开始训练
            results = self.model.train(**train_args)
            
            print("✅ 训练完成!")
            print(f"📁 模型保存位置: {results.save_dir}")
            
            # 返回训练结果
            return results
            
        except Exception as e:
            print(f"❌ 训练出错: {e}")
            raise
    
    def validate(self, data_config, model_path=None, imgsz=640, batch_size=32, device='auto'):
        """
        验证模型性能
        
        Args:
            data_config (str): 数据集配置文件
            model_path (str): 模型文件路径 (可选)
            imgsz (int): 输入图像尺寸
            batch_size (int): 批次大小
            device (str): 验证设备
        """
        print("📊 开始验证...")
        
        # 如果提供了模型路径，加载指定模型
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
        
        # 运行验证
        results = model.val(
            data=data_config,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            plots=True,
            verbose=True
        )
        
        print("✅ 验证完成!")
        return results
    
    def export_model(self, model_path, formats=['onnx'], imgsz=640):
        """
        导出模型到不同格式
        
        Args:
            model_path (str): 训练好的模型路径
            formats (list): 导出格式列表
            imgsz (int): 输入图像尺寸
        """
        print(f"📤 导出模型格式: {formats}")
        
        model = YOLO(model_path)
        
        for fmt in formats:
            try:
                export_path = model.export(
                    format=fmt,
                    imgsz=imgsz,
                    optimize=True,
                    int8=False,
                    device='cpu'
                )
                print(f"✅ {fmt.upper()} 格式导出成功: {export_path}")
            except Exception as e:
                print(f"❌ {fmt.upper()} 格式导出失败: {e}")
    
    def plot_training_results(self, results_dir):
        """
        绘制训练结果图表
        
        Args:
            results_dir (str): 训练结果目录
        """
        results_dir = Path(results_dir)
        csv_file = results_dir / 'results.csv'
        
        if not csv_file.exists():
            print(f"❌ 找不到结果文件: {csv_file}")
            return
        
        # 读取训练结果
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()  # 去除列名空格
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLOv8 人脸检测训练结果', fontsize=16)
        
        # 损失函数图
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='训练Box损失', color='blue')
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='验证Box损失', color='red')
        axes[0, 0].set_title('Box损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 类别损失图
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='训练分类损失', color='green')
        axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='验证分类损失', color='orange')
        axes[0, 1].set_title('分类损失')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # mAP指标图
        if 'metrics/mAP50(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='purple')
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='brown')
            axes[1, 0].set_title('mAP指标')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 精度和召回率图
        if 'metrics/precision(B)' in df.columns:
            axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='精度', color='red')
            axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='召回率', color='blue')
            axes[1, 1].set_title('精度和召回率')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = results_dir / 'training_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 训练结果图表已保存: {plot_path}")
        
        # 显示图表
        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8 人脸检测模型训练')
    parser.add_argument('--train-dir', type=str, required=True,
                       help='训练数据目录路径')
    parser.add_argument('--val-dir', type=str, required=True,
                       help='验证数据目录路径')
    parser.add_argument('--test-dir', type=str,
                       help='测试数据目录路径(可选)')
    parser.add_argument('--model-size', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='初始学习率')
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备')
    parser.add_argument('--workers', type=int, default=8,
                       help='数据加载线程数')
    parser.add_argument('--patience', type=int, default=50,
                       help='早停耐心值')
    parser.add_argument('--save-dir', type=str, default='runs/train',
                       help='保存目录')
    parser.add_argument('--resume', action='store_true',
                       help='从断点继续训练')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='不使用预训练权重')
    parser.add_argument('--export-formats', nargs='+', 
                       default=['onnx'], 
                       choices=['onnx', 'tflite', 'coreml', 'engine', 'pb'],
                       help='导出格式')
    parser.add_argument('--validate-only', action='store_true',
                       help='仅进行验证，不训练')
    parser.add_argument('--model-path', type=str,
                       help='验证时使用的模型路径')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not Path(args.train_dir).exists():
        print(f"❌ 训练数据目录不存在: {args.train_dir}")
        return
    
    if not Path(args.val_dir).exists():
        print(f"❌ 验证数据目录不存在: {args.val_dir}")
        return
    
    # 初始化训练器
    trainer = YOLOv8FaceTrainer(
        model_size=args.model_size,
        pretrained=not args.no_pretrained
    )
    
    # 创建数据集配置
    data_config = trainer.create_dataset_config(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        class_names=['face']
    )
    
    if args.validate_only:
        # 仅验证模式
        trainer.validate(
            data_config=data_config,
            model_path=args.model_path,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            device=args.device
        )
    else:
        # 训练模式
        results = trainer.train(
            data_config=data_config,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            lr0=args.lr0,
            save_dir=args.save_dir,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            resume=args.resume,
            pretrained=not args.no_pretrained
        )
        
        # 绘制训练结果
        if results:
            trainer.plot_training_results(results.save_dir)
            
            # 导出模型
            best_model = results.save_dir / 'weights' / 'best.pt'
            if best_model.exists():
                trainer.export_model(
                    model_path=str(best_model),
                    formats=args.export_formats,
                    imgsz=args.imgsz
                )
        
        print("🎉 训练流程完成!")


if __name__ == '__main__':
    main()

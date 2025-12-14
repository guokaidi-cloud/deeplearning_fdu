#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 InsightFace 的人脸识别系统

功能:
1) 多类别训练：目录下的子目录是标签，子目录下的图片是数据
2) 单图匹配：输入一张图片，找出最像哪个人
3) 批量分类：批量处理多张图片

安装依赖:
    pip install insightface onnxruntime-gpu  # GPU版本
    # 或
    pip install insightface onnxruntime      # CPU版本
"""

import os
import glob
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from PIL import Image
import cv2

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("警告: insightface 未安装，请运行: pip install insightface onnxruntime")


@dataclass
class RecognitionResult:
    """识别结果"""
    name: str
    similarity: float
    all_similarities: Optional[Dict[str, float]] = None


class InsightFaceRecognizer:
    """基于 InsightFace 的人脸识别器"""
    
    def __init__(self, model_name: str = 'buffalo_l', ctx_id: int = 0):
        """
        初始化 InsightFace 模型
        
        Args:
            model_name: 模型名称，可选：
                - 'buffalo_l' (推荐，精度高)
                - 'buffalo_s' (更快，精度略低)
                - 'buffalo_sc' (最快，适合边缘设备)
            ctx_id: GPU ID，-1 表示使用 CPU
        """
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("insightface 未安装，请运行: pip install insightface onnxruntime")
        
        print(f"正在加载 InsightFace 模型: {model_name}...")
        print(f"使用设备: {'GPU ' + str(ctx_id) if ctx_id >= 0 else 'CPU'}")
        
        # 初始化人脸分析器
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 
                      else ['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        # 存储训练数据
        self.class_embeddings: Dict[str, np.ndarray] = {}  # {类别名: 平均特征向量}
        self.class_all_embeddings: Dict[str, List[np.ndarray]] = {}  # {类别名: [所有特征向量]}
        self.class_image_counts: Dict[str, int] = {}  # {类别名: 图片数量}
        
        print("InsightFace 模型加载完成！")
    
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        从图片中提取人脸特征向量
        
        Args:
            image_path: 图片路径
        
        Returns:
            np.ndarray: 512维特征向量，如果未检测到人脸则返回None
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # 检测人脸并提取特征
        faces = self.app.get(img)
        
        if len(faces) == 0:
            return None
        
        # 如果有多个人脸，选择最大的（通常是最近的）
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        
        return faces[0].embedding
    
    def extract_embedding_from_array(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        从 numpy 数组图片中提取人脸特征向量
        
        Args:
            img_bgr: BGR格式的图片数组
        
        Returns:
            np.ndarray: 512维特征向量
        """
        faces = self.app.get(img_bgr)
        
        if len(faces) == 0:
            return None
        
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        
        return faces[0].embedding
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个特征向量的余弦相似度
        
        Args:
            emb1: 特征向量1
            emb2: 特征向量2
        
        Returns:
            float: 相似度分数 (0-1)
        """
        # 归一化
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        # 余弦相似度
        sim = np.dot(emb1, emb2)
        # 映射到 0-1 范围
        return (sim + 1) / 2
    
    def train_from_directory(self, train_dir: str, aggregation: str = 'mean', 
                             max_images_per_class: Optional[int] = None) -> Dict[str, int]:
        """
        多类别训练：目录下的子目录是标签，子目录下的图片是数据
        
        目录结构示例:
            train_dir/
            ├── 张三/           <- 子目录名是标签
            │   ├── photo1.jpg
            │   ├── photo2.jpg
            │   └── ...
            ├── 李四/
            │   ├── photo1.jpg
            │   └── ...
            └── 王五/
                └── ...
        
        Args:
            train_dir: 训练目录路径
            aggregation: 特征聚合方式 ('mean' 或 'all')
                - 'mean': 计算平均特征（推荐，速度快）
                - 'all': 保留所有特征（匹配时取最高相似度，更准确但更慢）
            max_images_per_class: 每个类别最多使用的图片数量
        
        Returns:
            dict: {类别名: 成功提取的图片数量}
        """
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"训练目录不存在: {train_dir}")
        
        # 获取所有子目录作为类别
        subdirs = [d for d in os.listdir(train_dir) 
                   if os.path.isdir(os.path.join(train_dir, d))]
        
        if len(subdirs) == 0:
            raise ValueError(f"训练目录下没有找到子目录（类别）: {train_dir}")
        
        print(f"\n找到 {len(subdirs)} 个类别")
        print("-" * 70)
        
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        
        self.class_embeddings = {}
        self.class_all_embeddings = {}
        self.class_image_counts = {}
        
        for class_name in tqdm(subdirs, desc="训练进度"):
            class_dir = os.path.join(train_dir, class_name)
            
            # 获取该类别下的所有图片
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(class_dir, ext)))
            
            image_paths = sorted(list(set(image_paths)))
            
            if len(image_paths) == 0:
                print(f"\n警告: 类别 '{class_name}' 下没有找到图片，跳过")
                continue
            
            # 限制每个类别的图片数量
            if max_images_per_class and len(image_paths) > max_images_per_class:
                image_paths = image_paths[:max_images_per_class]
            
            # 提取该类别所有图片的特征
            embeddings = []
            for img_path in image_paths:
                try:
                    emb = self.extract_embedding(img_path)
                    if emb is not None:
                        embeddings.append(emb)
                except Exception as e:
                    print(f"\n警告: 处理失败 {img_path}: {str(e)}")
                    continue
            
            if len(embeddings) == 0:
                print(f"\n警告: 类别 '{class_name}' 没有成功提取任何人脸特征，跳过")
                continue
            
            # 保存特征
            self.class_all_embeddings[class_name] = embeddings
            self.class_image_counts[class_name] = len(embeddings)
            
            # 聚合特征
            if aggregation == 'mean':
                avg_emb = np.mean(embeddings, axis=0)
                avg_emb = avg_emb / np.linalg.norm(avg_emb)  # 归一化
                self.class_embeddings[class_name] = avg_emb
        
        # 打印训练结果
        print("\n" + "=" * 70)
        print("训练完成！")
        print("=" * 70)
        print(f"总类别数: {len(self.class_embeddings)}")
        total_images = 0
        for class_name, count in self.class_image_counts.items():
            print(f"  - {class_name}: {count} 张图片")
            total_images += count
        print(f"总图片数: {total_images}")
        
        return self.class_image_counts
    
    def classify_image(self, test_image_path: str, top_k: int = 5, 
                       use_all_embeddings: bool = False) -> List[Tuple[str, float]]:
        """
        对测试图片进行分类，找出最像哪个人
        
        Args:
            test_image_path: 测试图片路径
            top_k: 返回前k个最相似的类别
            use_all_embeddings: 是否使用所有特征进行匹配（更准确但更慢）
        
        Returns:
            list: [(类别名, 相似度), ...] 按相似度降序排列
        """
        if len(self.class_embeddings) == 0 and len(self.class_all_embeddings) == 0:
            raise ValueError("请先调用 train_from_directory 进行训练！")
        
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"测试图片不存在: {test_image_path}")
        
        # 提取测试图片特征
        test_emb = self.extract_embedding(test_image_path)
        
        if test_emb is None:
            print(f"警告: 未在测试图片中检测到人脸: {test_image_path}")
            return []
        
        # 计算与所有类别的相似度
        results = []
        
        if use_all_embeddings and self.class_all_embeddings:
            # 使用所有特征，取最高相似度
            for class_name, embeddings in self.class_all_embeddings.items():
                max_sim = max(self.compute_similarity(test_emb, emb) for emb in embeddings)
                results.append((class_name, max_sim))
        else:
            # 使用平均特征
            for class_name, class_emb in self.class_embeddings.items():
                sim = self.compute_similarity(test_emb, class_emb)
                results.append((class_name, sim))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def classify_image_from_array(self, img_bgr: np.ndarray, top_k: int = 5,
                                  use_all_embeddings: bool = False) -> List[Tuple[str, float]]:
        """
        对 numpy 数组图片进行分类
        
        Args:
            img_bgr: BGR格式的图片数组
            top_k: 返回前k个最相似的类别
            use_all_embeddings: 是否使用所有特征进行匹配
        
        Returns:
            list: [(类别名, 相似度), ...]
        """
        if len(self.class_embeddings) == 0:
            raise ValueError("请先调用 train_from_directory 进行训练！")
        
        test_emb = self.extract_embedding_from_array(img_bgr)
        
        if test_emb is None:
            return []
        
        results = []
        
        if use_all_embeddings and self.class_all_embeddings:
            for class_name, embeddings in self.class_all_embeddings.items():
                max_sim = max(self.compute_similarity(test_emb, emb) for emb in embeddings)
                results.append((class_name, max_sim))
        else:
            for class_name, class_emb in self.class_embeddings.items():
                sim = self.compute_similarity(test_emb, class_emb)
                results.append((class_name, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def classify_batch(self, test_image_paths: List[str], top_k: int = 1,
                       use_all_embeddings: bool = False) -> List[dict]:
        """
        批量分类多张测试图片
        
        Args:
            test_image_paths: 测试图片路径列表
            top_k: 每张图片返回前k个最相似的类别
            use_all_embeddings: 是否使用所有特征进行匹配
        
        Returns:
            list: [{'image_path': 路径, 'predictions': [(类别名, 相似度), ...]}, ...]
        """
        results = []
        
        for img_path in tqdm(test_image_paths, desc="分类进度"):
            try:
                predictions = self.classify_image(
                    img_path, top_k=top_k, use_all_embeddings=use_all_embeddings
                )
                results.append({
                    'image_path': img_path,
                    'predictions': predictions
                })
            except Exception as e:
                print(f"\n处理失败 {img_path}: {str(e)}")
                results.append({
                    'image_path': img_path,
                    'predictions': []
                })
        
        return results
    
    def save_embeddings(self, save_path: str) -> None:
        """
        保存训练好的特征到文件
        
        Args:
            save_path: 保存路径 (.npz)
        """
        if len(self.class_embeddings) == 0:
            raise ValueError("没有训练数据可保存")
        
        # 准备保存数据
        data = {
            'class_names': list(self.class_embeddings.keys()),
            'embeddings': np.array([self.class_embeddings[name] for name in self.class_embeddings.keys()]),
            'image_counts': np.array([self.class_image_counts[name] for name in self.class_embeddings.keys()])
        }
        
        np.savez(save_path, **data)
        print(f"✅ 特征已保存到: {save_path}")
    
    def load_embeddings(self, load_path: str) -> None:
        """
        从文件加载训练好的特征
        
        Args:
            load_path: 文件路径 (.npz)
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"文件不存在: {load_path}")
        
        data = np.load(load_path, allow_pickle=True)
        class_names = data['class_names']
        embeddings = data['embeddings']
        image_counts = data['image_counts']
        
        self.class_embeddings = {name: emb for name, emb in zip(class_names, embeddings)}
        self.class_image_counts = {name: int(count) for name, count in zip(class_names, image_counts)}
        
        print(f"✅ 已加载 {len(self.class_embeddings)} 个类别的特征")


def demo_multi_class_classification():
    """多类别训练 + 分类匹配 示例"""
    print("=" * 70)
    print("InsightFace 多类别人脸识别")
    print("=" * 70)
    
    # 创建识别器
    recognizer = InsightFaceRecognizer(
        model_name='buffalo_l',  # 可选: buffalo_l, buffalo_s, buffalo_sc
        ctx_id=0  # GPU ID，-1 表示 CPU
    )
    
    # ========== 配置参数 ==========
    # 训练目录（子目录是标签，子目录下的图片是数据）
    train_directory = "classmate_photo_processed/"
    
    # 测试图片路径
    test_image_path = "frame_261436_id_0028_n_0006.jpg"
    
    # 显示前几个最相似的人
    top_k = 5
    
    print(f"\n训练目录: {train_directory}")
    print(f"测试图片: {test_image_path}")
    print("-" * 70)
    
    try:
        # ========== 步骤1: 多类别训练 ==========
        print("\n【步骤1】多类别训练...")
        class_counts = recognizer.train_from_directory(
            train_dir=train_directory,
            aggregation='mean',
            max_images_per_class=50
        )
        
        # 可选：保存特征以便下次快速加载
        # recognizer.save_embeddings("face_embeddings.npz")
        
        # ========== 步骤2: 分类测试图片 ==========
        print(f"\n【步骤2】正在识别测试图片: {test_image_path}")
        predictions = recognizer.classify_image(
            test_image_path=test_image_path,
            top_k=top_k,
            use_all_embeddings=False  # True 更准确但更慢
        )
        
        if len(predictions) == 0:
            print("\n❌ 未能在测试图片中检测到人脸！")
            return
        
        # 显示结果
        print("\n" + "=" * 70)
        print("🎯 识别结果（按相似度降序）")
        print("=" * 70)
        
        for i, (class_name, similarity) in enumerate(predictions, 1):
            bar_len = int(similarity * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"{i}. {class_name}")
            print(f"   相似度: {similarity:.4f} [{bar}]")
            print()
        
        # 最终预测
        best_class, best_score = predictions[0]
        print("=" * 70)
        print(f"🏆 最终预测: {best_class}")
        print(f"   置信度: {best_score:.4f}")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("使用说明:")
    print("=" * 70)
    print("""
    1. 准备训练数据（目录结构）：
       train_dir/
       ├── 张三/        <- 子目录名 = 人名
       │   ├── img1.jpg
       │   └── img2.jpg
       ├── 李四/
       │   └── ...
       └── 王五/
           └── ...
    
    2. API 使用方法：
       recognizer = InsightFaceRecognizer()
       recognizer.train_from_directory("train_dir/")
       result = recognizer.classify_image("test.jpg", top_k=5)
    
    3. 保存/加载特征（加速后续使用）：
       recognizer.save_embeddings("embeddings.npz")
       recognizer.load_embeddings("embeddings.npz")
    
    4. 模型选择：
       - buffalo_l: 精度最高（推荐）
       - buffalo_s: 速度和精度平衡
       - buffalo_sc: 最快，适合边缘设备
    """)


if __name__ == "__main__":
    demo_multi_class_classification()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 InsightFace 的人脸匹配器 - 使用相似度匹配找到最像的人
专门为人脸识别优化，比 CLIP 更准确
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2


import numpy as np

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️  insightface 库未安装，InsightFace人脸匹配功能不可用")
    print("   安装命令: pip install insightface onnxruntime-gpu")

# 检测 GPU 是否可用
def _check_gpu_available():
    """检查 ONNX Runtime GPU 是否可用"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            return True
        return False
    except Exception:
        return False

GPU_AVAILABLE = _check_gpu_available()


@dataclass
class MatchResult:
    """匹配结果"""
    name: str                    # 匹配的人名
    similarity: float            # 相似度分数 (0-1)
    all_similarities: Optional[Dict[str, float]] = None  # 所有人的相似度


class InsightFaceMatcher:
    """
    基于 InsightFace 的人脸匹配器
    
    使用方式:
        1. 初始化时加载照片库: matcher = InsightFaceMatcher(photo_folder="photos/")
        2. 或手动加载: matcher.load_photo_database("photos/")
        3. 匹配人脸: result = matcher.match(face_image)
    """

    def __init__(
        self,
        photo_folder: Optional[str] = None,
        threshold: float = 0.2,
        model_name: str = "buffalo_l",
        ctx_id: int = 0,
        use_gpu: bool = True,
    ):
        """
        初始化 InsightFace 人脸匹配器 (GPU 加速版本)
        
        Args:
            photo_folder: 人脸照片库文件夹路径
                - 方式1: 文件夹下直接放图片，文件名（不含扩展名）作为人名
                - 方式2: 文件夹下有子文件夹，子文件夹名作为人名，里面放该人的多张照片
            threshold: 相似度阈值，低于此值返回"未知人员"（默认0.2，范围0-1）
            model_name: InsightFace 模型名称 (buffalo_l, buffalo_s, buffalo_sc)
            ctx_id: GPU ID (0, 1, 2...)，-1 表示 CPU
            use_gpu: 是否使用 GPU 加速（默认 True）
        """
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("insightface 库未安装，无法使用 InsightFace 人脸匹配")
        
        self.threshold = 0
        self.model_name = model_name
        
        # 确定使用 GPU 还是 CPU
        if use_gpu and GPU_AVAILABLE:
            self.ctx_id = ctx_id if ctx_id >= 0 else 0
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': self.ctx_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB 显存限制
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }),
                'CPUExecutionProvider'
            ]
            device_str = f"🚀 GPU {self.ctx_id}"
        else:
            self.ctx_id = -1
            providers = ['CPUExecutionProvider']
            device_str = "💻 CPU"
            if use_gpu and not GPU_AVAILABLE:
                print("⚠️  GPU 不可用，回退到 CPU 模式")
                print("   安装 GPU 支持: pip install onnxruntime-gpu")
        
        # 初始化人脸分析器
        print(f"🔄 加载 InsightFace 模型: {model_name} (设备: {device_str})")
        self.app = FaceAnalysis(
            name=model_name,
            providers=providers
        )
        self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))
        print(f"✅ InsightFace 模型加载完成 ({device_str})")

        # 人脸特征数据库: {人名: 特征向量}
        self.face_database: Dict[str, np.ndarray] = {}
        # 每个人的所有特征（用于多图匹配）
        self.face_all_embeddings: Dict[str, List[np.ndarray]] = {}
        
        # 如果提供了照片文件夹，自动加载
        if photo_folder:
            self.load_photo_database(photo_folder)

    @property
    def num_people(self) -> int:
        """返回数据库中的人数"""
        return len(self.face_database)

    def _match_single_face(self, face_crop: np.ndarray) -> MatchResult:
        """
        匹配单个人脸（内部方法，用于多线程）
        
        Args:
            face_crop: 裁剪的人脸图像 (BGR)
            
        Returns:
            MatchResult: 匹配结果
        """
        if face_crop is None or face_crop.size == 0:
            return MatchResult(name="未知人员", similarity=0.0, all_similarities=None)
        
        # 提取特征
        query_emb = self._extract_embedding_from_crop(face_crop)
        
        if query_emb is None:
            return MatchResult(name="未知人员", similarity=0.0, all_similarities=None)
        
        # 与数据库匹配，找最优
        all_similarities = {}
        for name, db_emb in self.face_database.items():
            sim = self._compute_similarity(query_emb, db_emb)
            all_similarities[name] = sim
        
        best_name = max(all_similarities, key=all_similarities.get)
        best_sim = all_similarities[best_name]
        
        # 置信度阈值判断
        if best_sim < self.threshold:
            return MatchResult(
                name="未知人员",
                similarity=best_sim,
                all_similarities=all_similarities
            )
        
        return MatchResult(
            name=best_name,
            similarity=best_sim,
            all_similarities=all_similarities
        )

    def match_all_faces_in_image(self, full_image: np.ndarray, yolo_bboxes: list, num_threads: int = 4) -> list:
        """
        批量匹配多个 YOLO 检测的人脸（优化版本：保留人脸对齐 + 快速匹配）
        
        直接使用 YOLO 的 bbox 裁剪人脸，提取特征后与数据库匹配
        
        Args:
            full_image: 完整的 BGR 图像
            yolo_bboxes: YOLO 检测到的边界框列表 [[x1, y1, x2, y2], ...]
            num_threads: 并行线程数（默认 4）
            
        Returns:
            list: [MatchResult, ...] 与 yolo_bboxes 一一对应
        """
        from concurrent.futures import ThreadPoolExecutor
        
        if len(self.face_database) == 0:
            return [MatchResult(name="未知人员", similarity=0.0, all_similarities=None) 
                    for _ in yolo_bboxes]
        
        if len(yolo_bboxes) == 0:
            return []
        
        h, w = full_image.shape[:2]
        
        # 预计算数据库特征矩阵（只需计算一次）
        if not hasattr(self, '_db_matrix') or self._db_matrix is None:
            self._build_db_matrix()
        
        # 预先裁剪所有人脸
        face_crops = []
        valid_indices = []
        
        for idx, yolo_bbox in enumerate(yolo_bboxes):
            x1, y1, x2, y2 = yolo_bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            if x2 > x1 + 20 and y2 > y1 + 20:  # 最小尺寸检查
                face_crop = full_image[y1:y2, x1:x2].copy()
                face_crops.append(face_crop)
                valid_indices.append(idx)
        
        # 初始化结果
        results = [MatchResult(name="未知人员", similarity=0.0, all_similarities=None) 
                   for _ in yolo_bboxes]
        
        if not face_crops:
            return results
        
        # 使用线程池并行提取特征（保留人脸对齐）
        def _extract_and_match(face_crop):
            emb = self._extract_embedding_from_crop(face_crop)
            if emb is not None:
                return self._match_embedding_fast(emb)
            return MatchResult(name="未知人员", similarity=0.0, all_similarities=None)
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=min(num_threads, len(face_crops))) as executor:
            match_results = list(executor.map(_extract_and_match, face_crops))
        
        # 写回结果
        for idx, result in zip(valid_indices, match_results):
            results[idx] = result
        
        return results
    
    def _build_db_matrix(self):
        """构建数据库特征矩阵，用于快速批量匹配"""
        if len(self.face_database) == 0:
            self._db_matrix = None
            self._db_names = []
            return
        
        self._db_names = list(self.face_database.keys())
        embeddings = [self.face_database[name] for name in self._db_names]
        self._db_matrix = np.array(embeddings)
        # 归一化
        norms = np.linalg.norm(self._db_matrix, axis=1, keepdims=True)
        self._db_matrix = self._db_matrix / norms
    
    def _match_embedding_fast(self, embedding: np.ndarray) -> MatchResult:
        """
        快速匹配单个特征向量（使用预计算的矩阵，向量化计算）
        """
        if self._db_matrix is None or len(self._db_names) == 0:
            return MatchResult(name="未知人员", similarity=0.0, all_similarities=None)
        
        # 归一化查询向量
        emb_norm = embedding / np.linalg.norm(embedding)
        
        # 批量计算余弦相似度（向量化，比循环快）
        similarities = np.dot(self._db_matrix, emb_norm)
        similarities = (similarities + 1) / 2  # 映射到 0-1
        
        best_idx = np.argmax(similarities)
        best_name = self._db_names[best_idx]
        best_sim = float(similarities[best_idx])
        
        if best_sim < self.threshold:
            return MatchResult(name="未知人员", similarity=best_sim, all_similarities=None)
        
        return MatchResult(name=best_name, similarity=best_sim, all_similarities=None)
    
    def _extract_embedding_from_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        从裁剪的人脸图像中提取特征
        
        优先使用 InsightFace 进行关键点检测和对齐（精度高），
        如果检测失败则回退到直接提取特征（保证有输出）
        
        Args:
            face_crop: 裁剪的人脸图像 (BGR)
            
        Returns:
            512维特征向量，如果失败则返回 None
        """
        if face_crop is None or face_crop.size == 0:
            return None
        
        h, w = face_crop.shape[:2]
        
        # # 方案1：添加边距后再检测（处理人脸太靠边的情况）
        # # 边距大小根据人脸尺寸动态调整，并使用边缘复制填充
        pad = max(40, int(min(h, w) * 0.1))
        padded = cv2.copyMakeBorder(face_crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        faces = self.app.get(padded)
        
        if len(faces) > 0:
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            return faces[0].embedding
        
        # 方案2：如果还是检测不到，尝试直接提取特征（精度较低，但保证有输出）
        try:
            rec_model = self.app.models.get('recognition')
            if rec_model is not None:
                face_resized = cv2.resize(face_crop, (112, 112))
                embedding = rec_model.get_feat([face_resized])
                return embedding[0]
        except Exception:
            pass
        
        return None

    def _extract_embedding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        从 BGR 图像提取人脸特征向量（用于完整图像）
        
        Args:
            image_bgr: BGR 格式的图像 (OpenCV 格式)
        
        Returns:
            512维特征向量，如果未检测到人脸则返回 None
        """
        faces = self.app.get(image_bgr)
        
        if len(faces) == 0:
            return None
        
        # 如果有多个人脸，选择最大的
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
        
        return faces[0].embedding

    def _extract_embedding_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """从图片文件提取特征"""
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return None
        return self._extract_embedding(img)

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算两个特征向量的余弦相似度，返回 0-1 范围"""
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        sim = np.dot(emb1, emb2)
        return (sim + 1) / 2  # 映射到 0-1

    def load_photo_database(self, photo_folder: str) -> int:
        """
        加载人脸照片库
        
        支持两种目录结构:
        
        结构1 - 每人一张照片:
            photo_folder/
            ├── 张三.jpg    <- 文件名作为人名
            ├── 李四.png
            └── 王五.jpeg
        
        结构2 - 每人多张照片:
            photo_folder/
            ├── 张三/        <- 文件夹名作为人名
            │   ├── img1.jpg
            │   └── img2.jpg
            ├── 李四/
            │   └── photo.jpg
            └── 王五.png     <- 也可以混合使用
        
        Args:
            photo_folder: 照片文件夹路径
        
        Returns:
            int: 成功加载的人数
        """
        import cv2
        
        folder = Path(photo_folder)
        if not folder.exists():
            print(f"❌ 照片文件夹不存在: {photo_folder}")
            return 0
        
        self.face_database.clear()
        self.face_all_embeddings.clear()
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # 检查是否有子目录（多张照片模式）
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        
        if subdirs:
            # 模式2: 子目录模式
            print(f"📂 检测到子目录模式，每个子目录代表一个人")
            for subdir in subdirs:
                person_name = subdir.name
                embeddings = []
                
                for img_file in subdir.iterdir():
                    if img_file.suffix.lower() not in image_extensions:
                        continue
                    
                    emb = self._extract_embedding_from_file(str(img_file))
                    if emb is not None:
                        embeddings.append(emb)
                
                if embeddings:
                    # 计算平均特征
                    avg_emb = np.mean(embeddings, axis=0)
                    avg_emb = avg_emb / np.linalg.norm(avg_emb)
                    self.face_database[person_name] = avg_emb
                    self.face_all_embeddings[person_name] = embeddings
                    print(f"   ✅ {person_name}: {len(embeddings)} 张照片")
                else:
                    print(f"   ⚠️  {person_name}: 未提取到人脸特征")
        
        # 也处理根目录下的图片（模式1或混合模式）
        root_images = [f for f in folder.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions]
        
        if root_images:
            print(f"📷 处理根目录下的 {len(root_images)} 张照片")
            for img_file in root_images:
                person_name = img_file.stem  # 文件名（不含扩展名）作为人名
                
                emb = self._extract_embedding_from_file(str(img_file))
                if emb is not None:
                    self.face_database[person_name] = emb
                    self.face_all_embeddings[person_name] = [emb]
                    print(f"   ✅ {person_name}")
                else:
                    print(f"   ⚠️  {person_name}: 未检测到人脸")
        
        print(f"\n📊 照片库加载完成: {len(self.face_database)} 人")
        return len(self.face_database)

    def match(self, face_image: np.ndarray) -> MatchResult:
        """
        匹配人脸，找出最相似的人
        
        Args:
            face_image: 人脸图像 (BGR 格式，OpenCV)，已裁剪的人脸区域
        
        Returns:
            MatchResult: 匹配结果
        """
        if len(self.face_database) == 0:
            return MatchResult(name="未知人员", similarity=0.0, all_similarities=None)
        
        # 从裁剪的人脸中提取特征
        query_emb = self._extract_embedding_from_crop(face_image)
        
        if query_emb is None:
            return MatchResult(name="未知人员", similarity=0.0, all_similarities=None)
        
        # 与数据库匹配，找最优
        all_similarities = {}
        for name, db_emb in self.face_database.items():
            sim = self._compute_similarity(query_emb, db_emb)
            all_similarities[name] = sim
        
        best_name = max(all_similarities, key=all_similarities.get)
        best_sim = all_similarities[best_name]
        
        # 置信度阈值判断
        if best_sim < self.threshold:
            return MatchResult(
                name="未知人员",
                similarity=best_sim,
                all_similarities=all_similarities
            )
        
        return MatchResult(
            name=best_name,
            similarity=best_sim,
            all_similarities=all_similarities
        )

    def match_embedding(self, embedding: np.ndarray) -> MatchResult:
        """
        使用预先提取的特征向量进行匹配
        
        Args:
            embedding: 512维人脸特征向量
        
        Returns:
            MatchResult: 匹配结果
        """
        if len(self.face_database) == 0:
            return MatchResult(name="未知人员", similarity=0.0, all_similarities=None)
        
        all_similarities = {}
        for name, db_emb in self.face_database.items():
            sim = self._compute_similarity(embedding, db_emb)
            all_similarities[name] = sim
        
        best_name = max(all_similarities, key=all_similarities.get)
        best_sim = all_similarities[best_name]
        
        # 置信度阈值判断
        if best_sim < self.threshold:
            return MatchResult(
                name="未知人员",
                similarity=best_sim,
                all_similarities=all_similarities
            )
        
        return MatchResult(
            name=best_name,
            similarity=best_sim,
            all_similarities=all_similarities
        )

    def add_person(self, name: str, images: List[np.ndarray]) -> bool:
        """
        动态添加新人到数据库
        
        Args:
            name: 人名
            images: BGR 格式的人脸图像列表
        
        Returns:
            bool: 是否成功
        """
        embeddings = []
        for img in images:
            emb = self._extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            print(f"⚠️  无法为 {name} 提取任何人脸特征")
            return False
        
        avg_emb = np.mean(embeddings, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        
        self.face_database[name] = avg_emb
        self.face_all_embeddings[name] = embeddings
        
        print(f"✅ 已添加 {name}，共 {len(embeddings)} 张照片")
        return True

    def save_database(self, save_path: str) -> None:
        """保存人脸数据库到文件"""
        if len(self.face_database) == 0:
            raise ValueError("没有数据可保存")
        
        data = {
            'names': list(self.face_database.keys()),
            'embeddings': np.array([self.face_database[name] for name in self.face_database.keys()])
        }
        np.savez(save_path, **data)
        print(f"✅ 人脸数据库已保存到: {save_path}")

    def load_database(self, load_path: str) -> int:
        """从文件加载人脸数据库"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"文件不存在: {load_path}")
        
        data = np.load(load_path, allow_pickle=True)
        names = data['names']
        embeddings = data['embeddings']
        
        self.face_database = {name: emb for name, emb in zip(names, embeddings)}
        self.face_all_embeddings = {name: [emb] for name, emb in zip(names, embeddings)}
        
        print(f"✅ 已加载 {len(self.face_database)} 人的人脸数据")
        return len(self.face_database)

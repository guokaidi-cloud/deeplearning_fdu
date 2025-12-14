#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 CLIP 的人脸匹配器 - 使用相似度匹配找到最像的人
无需预训练 SVM，直接用 CLIP 特征的余弦相似度进行匹配
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  clip 库未安装，CLIP人脸匹配功能不可用")
    print("   安装命令: pip install git+https://github.com/openai/CLIP.git")


@dataclass
class MatchResult:
    """匹配结果"""
    name: str                    # 匹配的人名
    similarity: float            # 相似度分数 (0-1)
    all_similarities: Optional[Dict[str, float]] = None  # 所有人的相似度


class ClipFaceMatcher:
    """
    基于 CLIP 的人脸匹配器
    
    使用方式:
        1. 初始化时加载照片库: matcher = ClipFaceMatcher(photo_folder="photos/")
        2. 或手动加载: matcher.load_photo_database("photos/")
        3. 匹配人脸: result = matcher.match(face_image)
    """

    def __init__(
        self,
        photo_folder: Optional[str] = None,
        threshold: float = 0.65,
        clip_model_name: str = "ViT-B/32",
        device: str = "auto",
    ):
        """
        初始化 CLIP 人脸匹配器
        
        Args:
            photo_folder: 人脸照片库文件夹路径（文件名作为人名）
            threshold: 相似度阈值，低于此值返回"未知人员"
            clip_model_name: CLIP 模型名称 (ViT-B/32, ViT-B/16, ViT-L/14)
            device: 运行设备 (auto/cuda/cpu)
        """
        if not CLIP_AVAILABLE:
            raise RuntimeError("clip 库未安装，无法使用 CLIP 人脸匹配")
        
        self.threshold = threshold
        self.clip_model_name = clip_model_name
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 加载 CLIP 模型
        print(f"🔄 加载 CLIP 模型: {clip_model_name} (设备: {self.device})")
        self.clip_model, self.preprocess = clip.load(self.clip_model_name, device=self.device)
        self.clip_model.eval()
        print(f"✅ CLIP 模型加载完成")

        # 人脸特征数据库: {人名: 特征向量}
        self.face_database: Dict[str, np.ndarray] = {}
        
        # 如果提供了照片文件夹，自动加载
        if photo_folder:
            self.load_photo_database(photo_folder)

    def _extract_embedding(self, image: Image.Image) -> np.ndarray:
        """从 PIL 图像提取 CLIP 特征向量"""
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.clip_model.encode_image(tensor)
            feat = F.normalize(feat, dim=1)  # 归一化
        return feat.cpu().numpy()[0]

    def _extract_embedding_from_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        """从 BGR 图像（OpenCV 格式）提取 CLIP 特征向量"""
        # BGR -> RGB -> PIL Image
        image = Image.fromarray(image_bgr[:, :, ::-1])
        return self._extract_embedding(image)

    def load_photo_database(self, photo_folder: str, verbose: bool = True) -> int:
        """
        加载人脸照片库
        
        Args:
            photo_folder: 照片文件夹路径，文件名（不含扩展名）作为人名
            verbose: 是否输出详细信息
            
        Returns:
            加载的人数
        """
        photo_folder = Path(photo_folder)
        if not photo_folder.exists():
            print(f"❌ 照片文件夹不存在: {photo_folder}")
            return 0

        self.face_database.clear()
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        if verbose:
            print(f"📂 加载人脸照片库: {photo_folder}")
        
        for img_file in sorted(photo_folder.iterdir()):
            if img_file.suffix.lower() not in valid_extensions:
                continue
            
            name = img_file.stem  # 文件名作为人名
            try:
                image = Image.open(img_file).convert("RGB")
                embedding = self._extract_embedding(image)
                self.face_database[name] = embedding
                if verbose:
                    print(f"   ✅ {name}")
            except Exception as e:
                if verbose:
                    print(f"   ❌ {name}: {e}")
        
        print(f"📊 人脸库加载完成，共 {len(self.face_database)} 人\n")
        return len(self.face_database)

    def match(self, face_image: np.ndarray, return_all: bool = False) -> MatchResult:
        """
        匹配人脸，返回最相似的人
        
        Args:
            face_image: 人脸图像 (BGR 格式, OpenCV)
            return_all: 是否返回所有人的相似度
            
        Returns:
            MatchResult: 匹配结果，包含人名和相似度
        """
        if not self.face_database:
            return MatchResult(name="未知人员", similarity=0.0)
        
        # 提取人脸特征
        try:
            face_embedding = self._extract_embedding_from_bgr(face_image)
        except Exception as e:
            print(f"⚠️  特征提取失败: {e}")
            return MatchResult(name="未知人员", similarity=0.0)
        
        # 计算与所有人的相似度
        similarities: Dict[str, float] = {}
        for name, db_embedding in self.face_database.items():
            # 余弦相似度
            sim = float(np.dot(face_embedding, db_embedding))
            similarities[name] = sim
        
        # 找到最相似的人
        best_name = max(similarities, key=similarities.get)
        best_similarity = similarities[best_name]
        
        # 阈值判断
        if self.threshold is not None and best_similarity < self.threshold:
            result_name = "未知人员"
        else:
            result_name = best_name
        
        return MatchResult(
            name=result_name,
            similarity=best_similarity,
            all_similarities=similarities if return_all else None
        )

    def match_pil(self, face_image: Image.Image, return_all: bool = False) -> MatchResult:
        """
        匹配人脸 (PIL 图像版本)
        
        Args:
            face_image: 人脸图像 (PIL Image)
            return_all: 是否返回所有人的相似度
            
        Returns:
            MatchResult: 匹配结果
        """
        if not self.face_database:
            return MatchResult(name="未知人员", similarity=0.0)
        
        try:
            face_embedding = self._extract_embedding(face_image)
        except Exception as e:
            print(f"⚠️  特征提取失败: {e}")
            return MatchResult(name="未知人员", similarity=0.0)
        
        similarities: Dict[str, float] = {}
        for name, db_embedding in self.face_database.items():
            sim = float(np.dot(face_embedding, db_embedding))
            similarities[name] = sim
        
        best_name = max(similarities, key=similarities.get)
        best_similarity = similarities[best_name]
        
        if self.threshold is not None and best_similarity < self.threshold:
            result_name = "未知人员"
        else:
            result_name = best_name
        
        return MatchResult(
            name=result_name,
            similarity=best_similarity,
            all_similarities=similarities if return_all else None
        )

    @property
    def num_people(self) -> int:
        """返回数据库中的人数"""
        return len(self.face_database)

    @property
    def names(self) -> List[str]:
        """返回数据库中所有人名"""
        return list(self.face_database.keys())


# 测试用例
if __name__ == "__main__":
    import cv2
    
    print("=" * 60)
    print("CLIP 人脸匹配器测试")
    print("=" * 60)
    
    # 测试照片库路径（请修改为您的路径）
    photo_folder = "../classmate_photo_processed"
    
    if not Path(photo_folder).exists():
        print(f"❌ 请修改 photo_folder 路径: {photo_folder}")
    else:
        # 初始化匹配器
        matcher = ClipFaceMatcher(
            photo_folder=photo_folder,
            threshold=0.65,
            clip_model_name="ViT-B/32"
        )
        
        print(f"\n已加载 {matcher.num_people} 人")
        print(f"人员列表: {matcher.names[:5]}...")  # 只显示前5个

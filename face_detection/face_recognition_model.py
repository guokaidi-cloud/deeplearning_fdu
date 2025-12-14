#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight face recognition trainer and matcher.

功能:
1) 从目录结构加载人脸数据，目录名即标签。
2) 使用 face_recognition 提取128维特征，训练SVM分类器。
3) 支持保存/加载模型，并对单张人脸编码进行预测。

建议数据组织:
dataset_root/
  person_a/xxx.jpg
  person_b/yyy.png
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:  # pragma: no cover
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None


@dataclass
class PredictionResult:
    name: str
    probability: float
    raw_probabilities: Optional[Dict[str, float]] = None


class FaceRecognitionModel:
    """
    基于face_recognition特征 + SVM 分类的轻量人脸识别模型。

    - 训练: 调用 train_from_folder(...)
    - 推理: 调用 predict_embedding(...) 或 predict_image(...)
    """

    def __init__(self, model_path: str = "models/face_id_svm.joblib", threshold: float = 0.55):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.pipeline: Optional[SVC] = None
        self.label_encoder: Optional[LabelEncoder] = None

    # ----------------------------
    # 数据与特征提取
    # ----------------------------
    def _iter_image_files(self, root_dir: Path) -> List[Tuple[Path, str]]:
        pairs: List[Tuple[Path, str]] = []
        for person_dir in sorted(root_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            label = person_dir.name
            for img_file in person_dir.glob("*"):
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                pairs.append((img_file, label))
        return pairs

    def _extract_encoding(self, image_path: Path) -> Optional[np.ndarray]:
        if not FACE_RECOGNITION_AVAILABLE:
            raise RuntimeError("face_recognition 未安装，无法提取特征")
        image = face_recognition.load_image_file(str(image_path))
        encodings = face_recognition.face_encodings(image)
        return encodings[0] if encodings else None

    def _build_dataset(self, root_dir: Path, max_images_per_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        samples: List[np.ndarray] = []
        labels: List[str] = []
        for img_path, label in self._iter_image_files(root_dir):
            if max_images_per_class is not None:
                current = labels.count(label)
                if current >= max_images_per_class:
                    continue
            encoding = self._extract_encoding(img_path)
            if encoding is None:
                continue
            samples.append(encoding)
            labels.append(label)
        if not samples:
            raise RuntimeError(f"未在 {root_dir} 中提取到任何人脸特征")
        return np.vstack(samples), np.array(labels)

    # ----------------------------
    # 训练与保存
    # ----------------------------
    def train_from_folder(
        self,
        dataset_root: str,
        test_size: float = 0.2,
        random_state: int = 42,
        max_images_per_class: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        使用目录结构的数据训练 SVM 分类器。
        返回简单的验证指标（accuracy）。
        """
        root_dir = Path(dataset_root)
        if not root_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {root_dir}")

        x, y_str = self._build_dataset(root_dir, max_images_per_class)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_str)

        # 当某些类别样本只有1张时，分层划分会报错；不足则全量训练、不做验证集
        class_counts = np.bincount(y)
        need_full_train = (
            class_counts.min() < 2
            or len(class_counts) < 2
            or len(y) < 4
            or int(len(y) * test_size) < len(class_counts)
        )

        if need_full_train:
            print("⚠️ 数据量/类别过少，跳过验证集划分，使用全量数据训练。")
            x_train, y_train = x, y
            x_val = y_val = None
        else:
            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=test_size, random_state=random_state, stratify=y
            )

        self.pipeline = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, class_weight="balanced"),
        )
        self.pipeline.fit(x_train, y_train)

        acc = None
        if x_val is not None and y_val is not None:
            y_pred = self.pipeline.predict(x_val)
            acc = accuracy_score(y_val, y_pred)
            print("Validation accuracy:", f"{acc:.4f}")
            try:
                report = classification_report(
                    y_val, y_pred, target_names=self.label_encoder.classes_, digits=3
                )
                print(report)
            except Exception:
                pass
        else:
            print("ℹ️ 未生成验证集（类别样本过少），未计算验证精度。")

        self.save(self.model_path)

        return {
            "val_accuracy": float(acc) if acc is not None else None,
            "num_classes": len(self.label_encoder.classes_),
            "samples": len(y),
        }

    def save(self, model_path: Optional[str] = None) -> None:
        if self.pipeline is None or self.label_encoder is None:
            raise RuntimeError("模型未训练，无法保存")
        path = Path(model_path) if model_path else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "classes": self.label_encoder.classes_.tolist(),
            "threshold": self.threshold,
        }
        joblib.dump(payload, path)
        print(f"✅ 模型已保存: {path}")

    def load(self, model_path: Optional[str] = None) -> None:
        path = Path(model_path) if model_path else self.model_path
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        payload = joblib.load(path)
        self.pipeline = payload["pipeline"]
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(payload["classes"])
        self.threshold = float(payload.get("threshold", self.threshold))
        print(f"✅ 加载模型成功: {path}，类别数={len(self.label_encoder.classes_)}")

    # ----------------------------
    # 推理
    # ----------------------------
    def predict_embedding(self, encoding: np.ndarray) -> PredictionResult:
        if self.pipeline is None or self.label_encoder is None:
            raise RuntimeError("模型未加载或未训练")
        probs = self.pipeline.predict_proba([encoding])[0]
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        best_name = self.label_encoder.inverse_transform([best_idx])[0]

        # 当 threshold 为 None 时，不做阈值过滤，直接返回最优类别
        if self.threshold is not None and best_prob < self.threshold:
            best_name = "未知人员"
        prob_dict = {name: float(p) for name, p in zip(self.label_encoder.classes_, probs)}
        return PredictionResult(name=best_name, probability=best_prob, raw_probabilities=prob_dict)

    def predict_image(self, image_bgr: np.ndarray) -> PredictionResult:
        if not FACE_RECOGNITION_AVAILABLE:
            raise RuntimeError("face_recognition 未安装，无法推理")
        rgb = image_bgr[:, :, ::-1]
        encodings = face_recognition.face_encodings(rgb)
        if not encodings:
            return PredictionResult(name="未知人员", probability=0.0, raw_probabilities=None)
        return self.predict_embedding(encodings[0])


def quick_train(
    dataset_root: str = "~/workspace/deeplearning_fdu/classmate_photo_processed",
    model_path: str = "models/face_id_svm.joblib",
) -> None:
    """
    方便的命令式入口，用一行代码完成训练并保存模型。
    """
    dataset_root = os.path.expanduser(dataset_root)
    model = FaceRecognitionModel(model_path=model_path)
    stats = model.train_from_folder(dataset_root)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    quick_train()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 CLIP+SVM 身份识别模型
数据组织：dataset_root/人物姓名/*.jpg|*.png
默认数据路径：~/workspace/deeplearning_fdu/classmate_photo_processed
默认模型路径：models/clip_face_id_svm.joblib
"""

import argparse
import os
from pathlib import Path

from clip_face_recognition_model import ClipFaceRecognitionModel


def parse_args():
    parser = argparse.ArgumentParser(description="训练 CLIP+SVM 身份识别模型")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="~/workspace/deeplearning_fdu/classmate_photo_processed",
        help="数据集根目录，子目录名为标签",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/clip_face_id_svm.joblib",
        help="输出模型路径",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default="ViT-B/32",
        help="CLIP backbone，如 ViT-B/32 或 ViT-B/16",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="推理时的概率阈值，低于此值视为未知",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="验证集比例，类别样本过少时会自动跳过验证",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="每类最多使用的图片数（可用于均衡或加速）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = os.path.expanduser(args.dataset_root)

    model = ClipFaceRecognitionModel(
        model_path=args.model_path,
        threshold=args.threshold,
        clip_model_name=args.clip_model_name,
    )

    stats = model.train_from_folder(
        dataset_root=dataset_root,
        test_size=args.test_size,
        max_images_per_class=args.max_images_per_class,
    )

    print("\n训练完成，统计信息：", stats)
    print(f"模型已保存到: {Path(args.model_path).absolute()}")


if __name__ == "__main__":
    main()

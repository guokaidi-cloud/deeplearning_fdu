#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 人脸检测器 + InsightFace 人脸识别
使用 YOLO 检测人脸，InsightFace 匹配识别最相似的人
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import time
import sys
from collections import defaultdict
from typing import Optional

# PIL 用于中文字体支持
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

sys.path.append(str(Path(__file__).parent))

from face_detector import YOLOFaceDetector

# 导入 InsightFace 匹配器
try:
    from insightface_matcher import InsightFaceMatcher, INSIGHTFACE_AVAILABLE
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from insightface_matcher import InsightFaceMatcher, INSIGHTFACE_AVAILABLE
    except ImportError:
        INSIGHTFACE_AVAILABLE = False
        InsightFaceMatcher = None

# ======================== 配置 ========================
CHINESE_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

DEFAULT_SIMILARITY_THRESHOLD = 0.15

# 加载中文字体（用于独立函数）
def _load_chinese_font(size=20):
    """加载中文字体"""
    if not PIL_AVAILABLE:
        return None
    for font_path in CHINESE_FONT_PATHS:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return None

def draw_text_pil(image, text, position, font_color=(255, 255, 255), bg_color=(0, 128, 0), font_size=20):
    """
    使用 PIL 在图像上绘制中文文本
    
    Args:
        image: OpenCV图像 (BGR)
        text: 要绘制的文本
        position: 文本位置 (x, y) - 左上角
        font_color: 字体颜色 (R, G, B)
        bg_color: 背景颜色 (R, G, B)
        font_size: 字体大小
    
    Returns:
        处理后的图像
    """
    if not PIL_AVAILABLE:
        # 回退到 OpenCV
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color[::-1], 1)
        return image
    
    font = _load_chinese_font(font_size)
    if font is None:
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color[::-1], 1)
        return image
    
    # 转换为 PIL 图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    x, y = position
    
    # 获取文本边界框
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 绘制背景矩形
    padding = 3
    draw.rectangle(
        [(x, y), (x + text_width + padding * 2, y + text_height + padding * 2)],
        fill=bg_color
    )
    
    # 绘制文本
    draw.text((x + padding, y + padding), text, font=font, fill=font_color)
    
    # 转回 OpenCV 格式
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def get_chinese_font(font_size=20):
    """
    获取可用的中文字体
    
    Args:
        font_size (int): 字体大小
        
    Returns:
        ImageFont: 字体对象，如果没有找到中文字体则返回默认字体
    """
    if not PIL_AVAILABLE:
        return None
    
    for font_path in CHINESE_FONT_PATHS:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, font_size)
            except IOError:
                continue
    
    print("⚠️  未找到中文字体，将使用默认字体（中文可能显示为方块）")
    return ImageFont.load_default()


def check_and_download_model(model_path, model_name='yolov8n-face'):
    """
    检查模型是否存在，如果不存在则自动下载
    
    Args:
        model_path (Path): 模型文件路径
        model_name (str): 模型名称
        
    Returns:
        bool: 模型是否可用
    """
    if model_path.exists():
        print(f"✅ 找到人脸检测模型: {model_path}")
        return True
    
    print(f"⚠️  未找到人脸检测模型: {model_path}")
    print(f"🔄 开始自动下载 {model_name} 模型...")
    
    try:
        # 导入下载脚本
        from scripts.download_face_models import download_file
        
        # 人脸检测模型下载链接
        face_model_urls = {
            'yolov8n-face': 'https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt',
            'yolov12l-face': 'https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov12l-face.pt',
        }
        
        if model_name not in face_model_urls:
            print(f"❌ 不支持的模型: {model_name}")
            return False
        
        # 创建模型目录
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 下载模型
        download_file(face_model_urls[model_name], str(model_path))
        
        if model_path.exists() and model_path.stat().st_size > 1024 * 1024:
            print(f"✅ 模型下载成功: {model_path}")
            return True
        else:
            print(f"❌ 模型下载失败或文件损坏")
            return False
            
    except Exception as e:
        print(f"❌ 自动下载失败: {e}")
        print(f"💡 请手动下载模型到: {model_path}")
        if model_name == 'yolov8n-face':
            print(f"   下载链接: https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt")
        else:
            print(f"   下载链接: https://huggingface.co/Bingsu/adetailer/tree/main")
        return False


class YOLOSpecializedFaceDetector(YOLOFaceDetector):
    """
    专门的YOLOv8人脸检测器
    使用优化的人脸检测模型，支持基于 InsightFace 的人脸匹配识别和 ByteTrack 跟踪
    """
    
    def __init__(self, model_name='yolov8n-face', conf_threshold=0.3, device='auto', 
                 models_dir='models', model_path=None,
                 # 人脸识别参数
                 photo_folder=None, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, 
                 insightface_model_name='buffalo_sc',
                 # 跟踪参数
                 enable_tracking=False, tracker_type='bytetrack', track_buffer=30):
        """
        初始化专门的人脸检测器
        
        Args:
            model_name (str): 模型名称 ('yolov8n-face', 'yolov8s-face', 'yolov12l-face')
            conf_threshold (float): 置信度阈值
            device (str): 运行设备 (auto/cuda/cpu)
            models_dir (str): 模型目录
            model_path (str|None): 自定义模型路径（优先于 model_name/models_dir）
            photo_folder (str): 人脸照片库文件夹路径（用于人脸识别）
            similarity_threshold (float): 相似度阈值，低于此值为"未知人员"
            insightface_model_name (str): InsightFace模型名称 (buffalo_l, buffalo_s, buffalo_sc)
            enable_tracking (bool): 是否启用跟踪
            tracker_type (str): 跟踪器类型 ('bytetrack' 或 'botsort')
            track_buffer (int): 跟踪缓冲帧数（轨迹最大丢失帧数）
        """
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.chinese_font = None
        self.enable_tracking = enable_tracking
        self.tracker_type = tracker_type
        self.track_buffer = track_buffer
        
        # InsightFace 人脸匹配器
        self.face_matcher = None
        
        # 构造模型路径，优先使用自定义路径
        if model_path:
            model_path = Path(model_path)
            custom_model = True
        else:
            model_path = self.models_dir / f"{model_name}.pt"
            custom_model = False
        
        # 检查并下载/验证模型
        if not model_path.exists():
            if custom_model:
                raise RuntimeError(f"自定义模型文件不存在: {model_path}")
            if not check_and_download_model(model_path, model_name):
                raise RuntimeError(f"无法获取人脸检测模型: {model_name}")
        
        # 使用父类初始化
        super().__init__(
            model_path=str(model_path),
            conf_threshold=conf_threshold,
            device=device
        )
        
        # 加载中文字体
        if PIL_AVAILABLE:
            self.chinese_font = get_chinese_font(font_size=20)
        
        # 初始化 InsightFace 人脸匹配器
        if photo_folder:
            if INSIGHTFACE_AVAILABLE:
                try:
                    # 解析 device 参数
                    if device == 'auto':
                        import torch
                        ctx_id = 0 if torch.cuda.is_available() else -1
                    elif device == 'cuda':
                        ctx_id = 0
                    else:
                        ctx_id = -1
                    
                    self.face_matcher = InsightFaceMatcher(
                        photo_folder=photo_folder,
                        threshold=similarity_threshold,
                        model_name=insightface_model_name,
                        ctx_id=ctx_id
                    )
                    print(f"👥 InsightFace人脸匹配: 已加载 {self.face_matcher.num_people} 人")
                except Exception as e:
                    print(f"⚠️ 初始化InsightFace人脸匹配器失败: {e}")
                    import traceback
                    traceback.print_exc()
                    self.face_matcher = None
            else:
                print("⚠️ InsightFace 库未安装，人脸识别功能不可用")
                print("   安装命令: pip install insightface onnxruntime-gpu")
        
        self.model_path = model_path  # 记录实际使用的模型路径
        
        # 打印初始化信息
        print(f"🎯 专业人脸检测器已就绪")
        if custom_model:
            print(f"📦 模型: 自定义 -> {self.model_path}")
        else:
            print(f"📦 模型: {model_name} -> {self.model_path}")
        print(f"🎚️  置信度阈值: {conf_threshold}")
        if enable_tracking:
            print(f"🔄 跟踪器: {tracker_type.upper()} (buffer={track_buffer})")
        else:
            print(f"🔄 跟踪: 已禁用")
    
    def load_photo_database(self, photo_folder):
        """
        加载人脸照片库
        
        Args:
            photo_folder (str): 照片文件夹路径
            
        Returns:
            int: 加载的人数
        """
        if not INSIGHTFACE_AVAILABLE:
            print("❌ InsightFace 库未安装，无法加载人脸库")
            return 0
        
        if self.face_matcher is None:
            self.face_matcher = InsightFaceMatcher(threshold=DEFAULT_SIMILARITY_THRESHOLD)
        
        return self.face_matcher.load_photo_database(photo_folder)
    
    def recognize_face_with_bbox(self, full_image, bbox):
        """
        使用 InsightFace 匹配识别人脸
        
        直接使用 YOLO bbox 裁剪人脸区域进行识别
        
        Args:
            full_image: 完整图像 (BGR格式)
            bbox: YOLO检测到的边界框 [x1, y1, x2, y2]
            
        Returns:
            tuple: (姓名, 相似度)
        """
        if self.face_matcher is None:
            return "未知人员", None
        
        try:
            # 直接使用批量匹配方法（单个 bbox）
            results = self.face_matcher.match_all_faces_in_image(full_image, [bbox])
            if results:
                result = results[0]
                print(f"👥 InsightFace 匹配结果: {result.name} ({result.similarity:.2f})")
                return result.name, result.similarity
            return "未知人员", None
        except Exception as e:
            print(f"⚠️  人脸识别失败: {e}")
            return "未知人员", None
    
    def draw_chinese_text(self, image, text, position, font_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """
        在图像上绘制中文文本
        
        Args:
            image: OpenCV图像 (BGR)
            text: 要绘制的文本
            position: 文本位置 (x, y)
            font_color: 字体颜色 (R, G, B)
            bg_color: 背景颜色 (R, G, B)
            
        Returns:
            处理后的图像
        """
        if not PIL_AVAILABLE or self.chinese_font is None:
            # 如果PIL不可用，使用OpenCV绘制（中文会显示为方块）
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, font_color[::-1], 2)
            return image
        
        # 转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        x, y = position
        
        # 获取文本边界框
        bbox = draw.textbbox((0, 0), text, font=self.chinese_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 绘制背景矩形
        padding = 5
        draw.rectangle(
            [(x, y), (x + text_width + padding * 2, y + text_height + padding * 2)],
            fill=bg_color
        )
        
        # 绘制文本
        draw.text((x + padding, y + padding), text, font=self.chinese_font, fill=font_color)
        
        # 转回OpenCV格式
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def detect_and_track(self, image, recognize=True, persist=True):
        """
        使用YOLO内置的ByteTrack/BotSORT进行检测和跟踪
        
        Args:
            image: 输入图像
            recognize (bool): 是否进行人脸识别（使用 InsightFace 匹配）
            persist (bool): 是否持久化跟踪ID（跨帧保持ID）
            
        Returns:
            list: 跟踪结果列表，包含track_id
        """
        if isinstance(image, np.ndarray):
            original_image = image.copy()
            original_shape = image.shape[:2]
        else:
            original_image = np.array(image)
            original_shape = original_image.shape[:2]
        
        # 使用YOLO的track方法进行跟踪
        results = self.model.track(
            original_image, 
            conf=self.conf_threshold,
            persist=persist,
            tracker=f"{self.tracker_type}.yaml",
            verbose=False
        )
        
        tracked_faces = []
        
        # 先收集所有检测框
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0].cpu().numpy())
                    
                    x1_int = max(0, int(x1))
                    y1_int = max(0, int(y1))
                    x2_int = min(original_shape[1], int(x2))
                    y2_int = min(original_shape[0], int(y2))
                    
                    face_info = {
                        'bbox': [x1_int, y1_int, x2_int, y2_int],
                        'confidence': confidence,
                        'track_id': track_id,
                        'name': "未知人员",
                        'similarity': None
                    }
                    tracked_faces.append(face_info)
        
        # 批量进行人脸识别（每帧只调用一次 InsightFace）
        if recognize and self.face_matcher is not None and len(tracked_faces) > 0:
            # 过滤出有效的人脸框
            valid_indices = []
            valid_bboxes = []
            for i, face in enumerate(tracked_faces):
                x1, y1, x2, y2 = face['bbox']
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    valid_indices.append(i)
                    valid_bboxes.append(face['bbox'])
            
            if valid_bboxes:
                try:
                    # 批量匹配所有人脸
                    match_results = self.face_matcher.match_all_faces_in_image(
                        original_image, valid_bboxes
                    )
                    # 将结果写回
                    for idx, match_result in zip(valid_indices, match_results):
                        tracked_faces[idx]['name'] = match_result.name
                        tracked_faces[idx]['similarity'] = match_result.similarity
                        print(f"👥 InsightFace 匹配结果: {match_result.name} ({match_result.similarity:.2f})")
                except Exception as e:
                    print(f"⚠️  批量人脸识别失败: {e}")
        
        return tracked_faces
    
    def detect_faces(self, image, visualize=True, recognize=True):
        """
        检测图片中的人脸（支持 InsightFace 人脸匹配识别）
        
        Args:
            image: 输入图像
            visualize (bool): 是否可视化检测结果
            recognize (bool): 是否进行人脸识别（使用 InsightFace 匹配）
            
        Returns:
            tuple: (检测结果, 可视化图像)
        """
        # 保存原始图像尺寸
        if isinstance(image, np.ndarray):
            original_image = image.copy()
            original_shape = image.shape[:2]  # (height, width)
        else:
            original_image = np.array(image)
            original_shape = original_image.shape[:2]
        
        # 运行推理
        results = self.model(original_image, conf=self.conf_threshold, verbose=False)
        
        faces = []
        # 确保使用原始分辨率的图像进行可视化
        vis_image = original_image.copy()
        
        # 先收集所有检测框
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    x1_int = max(0, int(x1))
                    y1_int = max(0, int(y1))
                    x2_int = min(original_shape[1], int(x2))
                    y2_int = min(original_shape[0], int(y2))
                    
                    face_info = {
                        'bbox': [x1_int, y1_int, x2_int, y2_int],
                        'confidence': float(confidence),
                        'name': "未知人员",
                        'similarity': None
                    }
                    faces.append(face_info)
        
        # 批量进行人脸识别（每帧只调用一次 InsightFace）
        if recognize and self.face_matcher is not None and len(faces) > 0:
            valid_indices = []
            valid_bboxes = []
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = face['bbox']
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    valid_indices.append(i)
                    valid_bboxes.append(face['bbox'])
            
            if valid_bboxes:
                try:
                    match_results = self.face_matcher.match_all_faces_in_image(
                        original_image, valid_bboxes
                    )
                    for idx, match_result in zip(valid_indices, match_results):
                        faces[idx]['name'] = match_result.name
                        faces[idx]['similarity'] = match_result.similarity
                except Exception as e:
                    print(f"⚠️  批量人脸识别失败: {e}")
        
        # 可视化
        if visualize:
            for face_info in faces:
                x1_int, y1_int, x2_int, y2_int = face_info['bbox']
                confidence = face_info['confidence']
                name = face_info['name']
                similarity = face_info['similarity']
                is_known = name != "未知人员"
                
                # 根据是否识别成功选择颜色
                box_color = (0, 255, 0) if is_known else (0, 255, 255)  # 绿色=已识别, 黄色=未识别
                
                # 绘制边界框
                cv2.rectangle(vis_image, (x1_int, y1_int), (x2_int, y2_int), box_color, 2)
                
                # 构建标签文本
                if is_known and similarity is not None:
                    label = f'{name} ({similarity:.2f})'
                elif is_known:
                    label = f'{name}'
                else:
                    label = f'Face: {confidence:.3f}'
                
                # 绘制标签（支持中文）
                label_y = max(0, y1_int - 28)
                vis_image = self.draw_chinese_text(
                    vis_image, 
                    label, 
                    (x1_int, label_y),
                    font_color=(0, 0, 0),
                    bg_color=box_color
                )
        
        return faces, vis_image


def process_video_with_yolov8(detector, video_path, output_path=None, show_video=False, 
                              max_frames=None, start_time=None, end_time=None, save_faces=True,
                              save_interval_sec=5.0, enable_recognition=True, enable_tracking=True):
    """
    使用YOLOv8处理视频文件进行人脸检测、识别和跟踪
    
    Args:
        detector: YOLOv8人脸检测器实例
        video_path: 视频文件路径
        output_path: 输出视频路径
        show_video: 是否显示视频
        max_frames: 最大处理帧数
        start_time: 开始时间
        end_time: 结束时间
        save_faces (bool): 是否保存裁剪的人脸到data目录
        save_interval_sec (float): 保存人脸的时间间隔（秒），用于降频保存
        enable_recognition (bool): 是否启用人脸识别（使用 InsightFace 匹配）
        enable_tracking (bool): 是否启用跟踪 (ByteTrack/BotSORT)
    """
    print(f"🎥 开始处理视频: {video_path}")
    if enable_recognition and hasattr(detector, 'face_matcher') and detector.face_matcher:
        recognizer_name = type(detector.face_matcher).__name__
        print(f"👥 人脸匹配: 已启用 ({recognizer_name})，数据库中有 {detector.face_matcher.num_people} 人")
    
    # 检查跟踪功能
    tracking_enabled = enable_tracking and hasattr(detector, 'enable_tracking') and detector.enable_tracking
    if tracking_enabled:
        tracker_type = getattr(detector, 'tracker_type', 'bytetrack')
        print(f"🔄 跟踪: 已启用 ({tracker_type.upper()})")
    else:
        print(f"🔄 跟踪: 已禁用")
    
    # 创建data目录用于保存人脸
    if save_faces:
        data_dir = Path(video_path).parent / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 人脸将保存到: {data_dir}")

    # 仅检测与保存，不做跟踪
    
    # 打开视频文件
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 视频信息: {width}x{height}, {fps}FPS, 总帧数={total_frames}, 时长={duration:.1f}秒")
    
    # 解析时间参数（简化版本，与opencv_face_detector保持一致）
    start_frame = 0
    end_frame = total_frames
    
    if start_time:
        try:
            if ':' in start_time:
                parts = start_time.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(float, parts)
                    start_seconds = hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:  # MM:SS
                    minutes, seconds = map(float, parts)
                    start_seconds = minutes * 60 + seconds
            else:
                start_seconds = float(start_time)
            
            start_frame = int(start_seconds * fps)
            start_frame = max(0, min(start_frame, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"⏩ 跳转到开始时间: {start_seconds:.1f}秒 (第{start_frame}帧)")
        except:
            print(f"⚠️  无效的开始时间格式: {start_time}")
    
    if end_time:
        try:
            if ':' in end_time:
                parts = end_time.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(float, parts)
                    end_seconds = hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:  # MM:SS
                    minutes, seconds = map(float, parts)
                    end_seconds = minutes * 60 + seconds
            else:
                end_seconds = float(end_time)
                
            end_frame = int(end_seconds * fps)
            end_frame = max(start_frame, min(end_frame, total_frames))
            print(f"⏹️  结束时间: {end_seconds:.1f}秒 (第{end_frame}帧)")
        except:
            print(f"⚠️  无效的结束时间格式: {end_time}")
    
    # 计算处理帧数
    process_frames = end_frame - start_frame
    if max_frames:
        process_frames = min(process_frames, max_frames)
    
    # 设置视频写入器
    def _create_writer(path: Path, fps: int, size):
        """尝试多种编码，提升浏览器可播放性，并给出日志"""
        width, height = size
        codec_candidates = [
            ("avc1", "H.264 (浏览器兼容性好，需系统支持)"),
            ("mp4v", "MPEG-4 Part 2 (兼容性一般)"),
            ("XVID", "XVID (备用)"),
        ]
        for fourcc_tag, desc in codec_candidates:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"✅ 使用编码 {fourcc_tag} - {desc}")
                return writer, fourcc_tag
            else:
                print(f"⚠️ 创建写入器失败，尝试下一个编码: {fourcc_tag}")
        return None, None

    writer = None
    if output_path:
        writer, used_codec = _create_writer(output_path, fps, (width, height))
        if writer is None:
            print("❌ 无法创建任何可用的视频写入器，停止处理")
            return
        print(f"📁 输出视频: {output_path}")
        print(f"📐 输出分辨率: {width}x{height}, FPS: {fps}, 编码: {used_codec}")
    
    # 处理统计
    processed_frames = 0
    total_faces = 0
    process_start_time = time.time()
    last_save_time = -1e9  # 控制保存频率的时间戳
    track_save_counts = defaultdict(int)  # 用于按track_id保存计数
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 获取当前帧位置
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time_sec = current_frame / fps if fps > 0 else 0
            
            # 检查处理限制
            if processed_frames >= process_frames:
                break
                
            if max_frames and processed_frames >= max_frames:
                print(f"⏹️  已达到最大处理帧数: {max_frames}")
                break
            
            # 检测人脸（根据是否启用跟踪选择不同方法）
            if tracking_enabled:
                # 使用YOLO内置的ByteTrack/BotSORT跟踪
                faces = detector.detect_and_track(frame, recognize=enable_recognition, persist=True)
            else:
                # 仅检测，不跟踪
                faces, _ = detector.detect_faces(frame, visualize=False, recognize=enable_recognition)
            
            total_faces += len(faces)
            
            # 统计识别结果
            recognized_names = [f['name'] for f in faces if f.get('name') and f['name'] != "未知人员"]
            
            # 自定义可视化（支持跟踪ID显示）
            vis_frame = frame.copy()
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                confidence = face['confidence']
                name = face.get('name', '未知人员')
                track_id = face.get('track_id', None)
                is_known = name != "未知人员"
                
                # 根据是否识别成功选择颜色
                if tracking_enabled and track_id is not None:
                    # 跟踪模式：使用track_id生成颜色
                    color_hash = hash(str(track_id)) % 0xFFFFFF
                    box_color = ((color_hash >> 16) & 0xFF, (color_hash >> 8) & 0xFF, color_hash & 0xFF)
                    # 确保颜色足够亮
                    box_color = tuple(max(c, 50) for c in box_color)
                else:
                    box_color = (0, 255, 0) if is_known else (0, 255, 255)  # 绿色=已识别, 黄色=未识别
                
                # 绘制边界框
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)
                
                # 构建标签文本，优先显示姓名
                if tracking_enabled and track_id is not None:
                    if is_known:
                        label = f'{name} | ID:{track_id} ({confidence:.2f})'
                    else:
                        label = f'ID:{track_id} ({confidence:.2f})'
                else:
                    if is_known:
                        label = f'{name} ({confidence:.2f})'
                    else:
                        label = f'Face ({confidence:.2f})'
                
                # 绘制标签（支持中文）
                label_y = max(0, y1 - 28)
                vis_frame = draw_text_pil(
                    vis_frame, label, (x1, label_y),
                    font_color=(0, 0, 0), 
                    bg_color=box_color,
                    font_size=18
                )

            # 确保vis_frame的分辨率与原始frame一致
            if vis_frame.shape[:2] != frame.shape[:2]:
                print(f"⚠️  警告: vis_frame分辨率 {vis_frame.shape[:2]} 与原始frame分辨率 {frame.shape[:2]} 不一致，使用原始frame")
                vis_frame = frame.copy()
                # 重新绘制检测框
                for face in faces:
                    x1, y1, x2, y2 = face['bbox']
                    confidence = face['confidence']
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                    label = f'Face: {confidence:.3f}'
                    label_y = max(0, int(y1) - 28)
                    vis_frame = draw_text_pil(
                        vis_frame, label, (int(x1), label_y),
                        font_color=(0, 0, 0),
                        bg_color=(0, 255, 255),
                        font_size=16
                    )
            
            # 保存裁剪的人脸
            if save_faces:
                # 按时间间隔降频保存；若本帧未到达保存间隔则跳过保存
                allow_save = (current_time_sec - last_save_time) >= save_interval_sec
                if allow_save:
                    last_save_time = current_time_sec
                    for face_idx, face in enumerate(faces):
                        x1, y1, x2, y2 = face['bbox']
                        confidence = face.get('confidence', 0.0)
                        track_id = face.get('track_id', None)
                        name = face.get('name', '未知人员')
                        
                        # 确保坐标在图像范围内
                        x1 = max(0, int(x1))
                        y1 = max(0, int(y1))
                        x2 = min(frame.shape[1], int(x2))
                        y2 = min(frame.shape[0], int(y2))
                        
                        # 裁剪人脸区域
                        face_crop = frame[y1:y2, x1:x2]
                        
                        # 只保存有效的人脸（尺寸不能太小）
                        if face_crop.shape[0] > 20 and face_crop.shape[1] > 20:
                            # 根据是否有track_id决定保存路径
                            if track_id is not None:
                                # 有track_id：按ID分目录保存
                                if name != "未知人员":
                                    id_dir = data_dir / f"id_{int(track_id):04d}_{name}"
                                else:
                                    id_dir = data_dir / f"id_{int(track_id):04d}"
                                id_dir.mkdir(parents=True, exist_ok=True)
                                track_save_counts[track_id] += 1
                                face_filename = f"frame_{current_frame:06d}_id_{int(track_id):04d}_n_{track_save_counts[track_id]:04d}.jpg"
                                face_path = id_dir / face_filename
                            else:
                                # 无track_id：按帧号和人脸索引保存
                                face_filename = f"frame_{current_frame:06d}_face_{face_idx:02d}_conf_{confidence:.3f}.jpg"
                                face_path = data_dir / face_filename
                            cv2.imwrite(str(face_path), face_crop)
            
            # 添加统计信息
            elapsed_time = time.time() - process_start_time
            current_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
            
            hours = int(current_time_sec // 3600)
            minutes = int((current_time_sec % 3600) // 60)
            seconds = int(current_time_sec % 60)
            
            stats_text = [
                f'Time: {hours:02d}:{minutes:02d}:{seconds:02d} (Frame: {current_frame})',
                f'Progress: {processed_frames+1}/{process_frames}',
                f'Current Faces: {len(faces)}',
                f'Recognized: {len(recognized_names)}',
                f'Processing FPS: {current_fps:.1f}'
            ]
            
            # 如果有识别到的人，显示姓名
            if recognized_names:
                names_str = ', '.join(recognized_names[:3])  # 最多显示3个名字
                if len(recognized_names) > 3:
                    names_str += f'... (+{len(recognized_names)-3})'
                stats_text.append(f'Names: {names_str}')
            
            for i, text in enumerate(stats_text):
                y_pos = 10 + i * 28
                vis_frame = draw_text_pil(
                    vis_frame, text, (10, y_pos),
                    font_color=(255, 255, 0),
                    bg_color=(0, 0, 0),
                    font_size=18
                )
            
            # 保存帧
            if writer:
                # 确保vis_frame的分辨率与VideoWriter设置的分辨率一致
                if vis_frame.shape[:2] != (height, width):
                    vis_frame = cv2.resize(vis_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                writer.write(vis_frame)
            
            # 显示视频
            if show_video:
                cv2.imshow('YOLOv8人脸检测', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("👤 用户按'q'键退出")
                    break
            
            processed_frames += 1
            
            # 定期输出进度
            if processed_frames % 100 == 0:
                progress = (processed_frames / process_frames) * 100
                current_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                print(f"📈 处理进度: {processed_frames}/{process_frames} "
                      f"({progress:.1f}%) | 时间: {current_time_str} | 检测到人脸: {total_faces}")
    
    except KeyboardInterrupt:
        print("\n⏹️  用户中断处理")
    
    finally:
        # 清理资源
        cap.release()
        if writer:
            writer.release()
        if show_video:
            cv2.destroyAllWindows()
    
    # 输出统计信息
    elapsed_time = time.time() - process_start_time
    avg_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n📊 处理完成统计:")
    print(f"   ⏱️  处理时间: {elapsed_time:.1f}秒")
    print(f"   📹 处理帧数: {processed_frames}")
    print(f"   🎯 检测人脸: {total_faces}")
    print(f"   ⚡ 平均FPS: {avg_fps:.1f}")
    print(f"   📏 平均每帧人脸数: {total_faces/processed_frames:.1f}" if processed_frames > 0 else "")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8专业人脸检测器（使用 InsightFace 进行人脸匹配识别）')
    parser.add_argument('--input', type=str, required=True,
                       help='输入视频或图片文件路径')
    parser.add_argument('--output', type=str, 
                       help='输出视频文件路径')
    parser.add_argument('--show', action='store_true',
                       help='显示检测过程')
    parser.add_argument('--max-frames', type=int,
                       help='最大处理帧数 (用于测试)')
    parser.add_argument('--start-time', type=str,
                       help='开始时间 (秒数或 HH:MM:SS 格式)')
    parser.add_argument('--end-time', type=str, 
                       help='结束时间 (秒数或 HH:MM:SS 格式)')
    parser.add_argument('--model', type=str, default='yolov8n-face',
                       choices=['yolov8n-face', 'yolov12l-face'],
                       help='人脸检测模型名称')
    parser.add_argument('--model-path', type=str, default=None,
                       help='自定义模型文件路径（优先使用该路径）')
    parser.add_argument('--conf', type=float, default=0.3, 
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='auto', 
                       help='运行设备 (auto/cuda/cpu)')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='模型存放目录')
    parser.add_argument('--save-faces', action='store_true', default=False,
                       help='保存裁剪的人脸到data目录')
    parser.add_argument('--no-save-faces', dest='save_faces', action='store_false',
                       help='不保存裁剪的人脸')
    parser.add_argument('--save-interval-sec', type=float, default=5.0,
                       help='保存人脸的时间间隔（秒），默认5秒')
    
    # InsightFace 人脸匹配参数
    parser.add_argument('--photo-folder', type=str, default=None,
                       help='人脸照片库文件夹路径（用于人脸识别匹配）')
    parser.add_argument('--similarity-threshold', type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                       help=f'相似度阈值，低于此值为未知人员，默认{DEFAULT_SIMILARITY_THRESHOLD}')
    parser.add_argument('--insightface-model', type=str, default='buffalo_sc',
                       help='InsightFace模型名称: buffalo_l(推荐) 或 buffalo_s(更快) 或 buffalo_sc(最快)')
    parser.add_argument('--no-recognition', action='store_true',
                       help='禁用人脸识别功能')
    
    # 跟踪相关参数
    parser.add_argument('--track', action='store_true', default=False,
                       help='启用跟踪功能 (ByteTrack/BotSORT)')
    parser.add_argument('--no-track', dest='track', action='store_false',
                       help='禁用跟踪功能')
    parser.add_argument('--tracker', type=str, default='bytetrack',
                       choices=['bytetrack', 'botsort'],
                       help='跟踪器类型: bytetrack(快速) 或 botsort(更精确)')
    parser.add_argument('--track-buffer', type=int, default=30,
                       help='跟踪缓冲帧数(轨迹最大丢失帧数），默认30')
    
    args = parser.parse_args()
    
    try:
        # 初始化专业人脸检测器
        print(f"🚀 初始化YOLOv8人脸检测器...")
        detector = YOLOSpecializedFaceDetector(
            model_name=args.model,
            conf_threshold=args.conf,
            device=args.device,
            models_dir=args.models_dir,
            model_path=args.model_path,
            photo_folder=args.photo_folder,
            similarity_threshold=args.similarity_threshold,
            insightface_model_name=args.insightface_model,
            enable_tracking=args.track,
            tracker_type=args.tracker,
            track_buffer=args.track_buffer
        )
        
        enable_recognition = not args.no_recognition and detector.face_matcher is not None
        enable_tracking = args.track
        
        # 检查输入文件
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ 输入文件不存在: {args.input}")
            return
        
        # 设置输出路径
        output_path = args.output
        if not output_path:
            output_path = input_path.parent / f"yolov8_detected_{input_path.name}"
        
        # 处理视频文件
        if input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # 创建专用的视频处理方法
            process_video_with_yolov8(
                detector=detector,
                video_path=input_path,
                output_path=output_path,
                show_video=args.show,
                max_frames=args.max_frames,
                start_time=args.start_time,
                end_time=args.end_time,
                save_faces=args.save_faces,
                save_interval_sec=args.save_interval_sec,
                enable_recognition=enable_recognition,
                enable_tracking=enable_tracking
            )
        
        # 处理图片文件
        elif input_path.is_file() and input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            image = cv2.imread(str(input_path))
            faces, vis_image = detector.detect_faces(image, visualize=True, recognize=enable_recognition)
            
            # 保存裁剪的人脸
            if args.save_faces and len(faces) > 0:
                data_dir = input_path.parent / 'data'
                data_dir.mkdir(parents=True, exist_ok=True)
                print(f"📁 人脸将保存到: {data_dir}")
                
                for face_idx, face in enumerate(faces):
                    x1, y1, x2, y2 = face['bbox']
                    confidence = face['confidence']
                    
                    # 确保坐标在图像范围内
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(image.shape[1], int(x2))
                    y2 = min(image.shape[0], int(y2))
                    
                    # 裁剪人脸区域
                    face_crop = image[y1:y2, x1:x2]
                    
                    # 只保存有效的人脸（尺寸不能太小）
                    if face_crop.shape[0] > 20 and face_crop.shape[1] > 20:
                        face_filename = f"{input_path.stem}_face_{face_idx:02d}_conf_{confidence:.3f}.jpg"
                        face_path = data_dir / face_filename
                        cv2.imwrite(str(face_path), face_crop)
                        print(f"   💾 保存人脸: {face_filename}")
            
            # 保存结果
            if not output_path:
                output_path = input_path.parent / f"yolov8_detected_{input_path.name}"
            
            cv2.imwrite(str(output_path), vis_image)
            
            print(f"✅ 检测到 {len(faces)} 个人脸")
            recognized_count = 0
            for i, face in enumerate(faces):
                bbox = face['bbox']
                conf = face['confidence']
                name = face.get('name', '未知人员')
                if name != '未知人员':
                    recognized_count += 1
                    print(f"   人脸{i+1}: {name}, 坐标{bbox}, 置信度{conf:.3f}")
                else:
                    print(f"   人脸{i+1}: 未知人员, 坐标{bbox}, 置信度{conf:.3f}")
            
            if enable_recognition:
                print(f"📊 识别结果: {recognized_count}/{len(faces)} 人被识别")
            
            print(f"📁 结果已保存到: {output_path}")
            
            # 显示结果
            if args.show:
                cv2.imshow('YOLOv8专业人脸检测', vis_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        else:
            print(f"❌ 不支持的文件类型: {input_path}")
    
    except Exception as e:
        print(f"❌ 检测过程中出现错误: {e}")
        print("💡 建议:")
        print("   1. 检查输入文件是否存在")
        print("   2. 确认网络连接正常（用于下载模型）")
        print("   3. 检查设备和CUDA环境")


if __name__ == '__main__':
    main()

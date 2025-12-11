#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 人体检测 + DeepSORT 跟踪
支持视频/摄像头检测，按跟踪ID分文件夹保存人体裁剪。
"""

import argparse
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class YOLOHumanTracker:
    def __init__(
        self,
        model_path="yolo8x.pt",
        conf_threshold=0.4,
        device="auto",
        tracker_max_age=30,
        tracker_n_init=3,
        tracker_max_cosine_distance=0.2,
        enable_track=True,
    ):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = YOLO(model_path)
        self.enable_track = enable_track
        # DeepSORT
        if enable_track:
            use_gpu = torch.cuda.is_available() if device == "auto" else str(device).startswith("cuda")
            self.tracker = DeepSort(
                max_age=tracker_max_age,
                n_init=tracker_n_init,
                max_cosine_distance=tracker_max_cosine_distance,
                embedder_gpu=use_gpu,
            )
            print(f"✅ YOLO 模型加载: {model_path} | 设备: {self.model.device}")
            print(f"✅ DeepSORT 初始化 (gpu={use_gpu}, max_age={tracker_max_age}, n_init={tracker_n_init}, cosine={tracker_max_cosine_distance})")
        else:
            self.tracker = None
            print(f"✅ YOLO 模型加载: {model_path} | 设备: {self.model.device}")
            print("ℹ️ 已关闭 DeepSORT，使用纯检测模式。")

    def _filter_person_boxes(self, result):
        """从YOLO结果中过滤出person类别的框"""
        boxes = []
        if result.boxes is None:
            return boxes
        for box in result.boxes:
            cls = int(box.cls[0].item())
            if cls != 0:  # COCO: person 类别ID为0
                continue
            conf = float(box.conf[0].item())
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf,
                }
            )
        return boxes

    def detect_and_track_video(
        self,
        source,
        output_path=None,
        show=True,
        save_crops=False,
        crops_dir="data/human_crops",
        save_interval_sec=0.0,
        start_time_sec=0.0,
        max_frames=None,
    ):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ 无法打开视频源: {source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

        # 跳转到开始时间
        if start_time_sec > 0 and fps > 0:
            start_frame_idx = int(start_time_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

        writer = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"📁 保存视频到: {output_path}")

        crops_dir = Path(crops_dir)
        if save_crops:
            crops_dir.mkdir(parents=True, exist_ok=True)
            print(f"📂 人体裁剪保存路径: {crops_dir}")

        last_save_time = -1e9
        track_save_counts = defaultdict(int)
        processed = 0
        t0 = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if max_frames and processed >= max_frames:
                    break

                # YOLO 推理
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                person_boxes = self._filter_person_boxes(results[0])

                if self.enable_track and self.tracker is not None:
                    # 送入跟踪器
                    detections = [([b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]], b["conf"], "person") for b in person_boxes]
                    tracks = self.tracker.update_tracks(detections, frame=frame)

                    # 绘制 + 保存
                    for track in tracks:
                        if not track.is_confirmed() or track.time_since_update > 0:
                            continue
                        l, t, r, b = map(int, track.to_ltrb())
                        track_id = track.track_id
                        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID {track_id}", (l, t - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                        # 保存裁剪
                        if save_crops:
                            now_sec = current_frame_idx / fps
                            if now_sec - last_save_time >= save_interval_sec:
                                last_save_time = now_sec
                                crop = frame[max(0, t): max(0, b), max(0, l): max(0, r)]
                                if crop.shape[0] > 10 and crop.shape[1] > 10:
                                    id_dir = crops_dir / f"id_{int(track_id):04d}"
                                    id_dir.mkdir(parents=True, exist_ok=True)
                                    track_save_counts[track_id] += 1
                                    fname = f"frame_{current_frame_idx:06d}_id_{int(track_id):04d}_n_{track_save_counts[track_id]:04d}.jpg"
                                    cv2.imwrite(str(id_dir / fname), crop)
                    tracked_count = len([t for t in tracks if t.is_confirmed()])
                else:
                    # 纯检测模式，不跟踪
                    tracks = []
                    for det_idx, det in enumerate(person_boxes):
                        x1, y1, x2, y2 = det["bbox"]
                        conf = det["conf"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                        # 保存裁剪
                        if save_crops:
                            now_sec = current_frame_idx / fps
                            if now_sec - last_save_time >= save_interval_sec:
                                last_save_time = now_sec
                                crop = frame[max(0, y1): max(0, y2), max(0, x1): max(0, x2)]
                                if crop.shape[0] > 10 and crop.shape[1] > 10:
                                    det_dir = crops_dir / "det_only"
                                    det_dir.mkdir(parents=True, exist_ok=True)
                                    fname = f"frame_{current_frame_idx:06d}_det_{det_idx:02d}_conf_{conf:.3f}.jpg"
                                    cv2.imwrite(str(det_dir / fname), crop)
                    tracked_count = len(person_boxes)

                # 叠加信息
                processed += 1
                elapsed = time.time() - t0
                fps_now = processed / elapsed if elapsed > 0 else 0
                info = f"Frames: {processed} | Tracks: {tracked_count} | FPS: {fps_now:.1f}"
                cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

                if writer:
                    writer.write(frame)
                if show:
                    cv2.imshow("YOLO Human + DeepSORT", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            print("⏹️  手动中断")
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        print(f"✅ 处理完成，帧数: {processed}, 平均FPS: {processed / (time.time() - t0):.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 人体检测 + DeepSORT 跟踪")
    parser.add_argument("--source", type=str, default="0", help="输入源，0为摄像头，或视频文件路径")
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="YOLO模型路径，支持COCO模型（默认yolo11x.pt）")
    parser.add_argument("--conf", type=float, default=0.4, help="置信度阈值")
    parser.add_argument("--device", type=str, default="auto", help="运行设备")
    parser.add_argument("--save-vid", action="store_true", help="保存带框视频")
    parser.add_argument("--output", type=str, help="输出视频路径")
    parser.add_argument("--show", action="store_true", help="显示实时画面")
    parser.add_argument("--save-crops", action="store_true", help="保存人体裁剪")
    parser.add_argument("--crops-dir", type=str, default="data/human_crops", help="裁剪保存目录")
    parser.add_argument("--save-interval-sec", type=float, default=0.0, help="裁剪保存时间间隔（秒），默认每帧")
    parser.add_argument("--start-time", type=float, default=0.0, help="从视频第多少秒开始处理（检测、跟踪和裁剪）")
    parser.add_argument("--max-frames", type=int, help="最多处理的帧数")
    parser.add_argument("--tracker-max-age", type=int, default=30, help="DeepSORT: 轨迹最大丢失帧数")
    parser.add_argument("--tracker-n-init", type=int, default=3, help="DeepSORT: 轨迹确认帧数")
    parser.add_argument("--tracker-max-cosine-distance", type=float, default=0.2, help="DeepSORT: 余弦距离阈值")
    parser.add_argument("--no-track", action="store_true", help="关闭DeepSORT，仅使用YOLO检测")
    return parser.parse_args()


def main():
    args = parse_args()
    tracker = YOLOHumanTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
        tracker_max_age=args.tracker_max_age,
        tracker_n_init=args.tracker_n_init,
        tracker_max_cosine_distance=args.tracker_max_cosine_distance,
        enable_track=not args.no_track,
    )

    source = args.source
    output_path = None
    if args.save_vid:
        if args.output:
            output_path = Path(args.output)
        else:
            # 默认输出到输入同目录
            if source.isdigit():
                output_path = Path("human_detect_output.mp4")
            else:
                src_path = Path(source)
                output_path = src_path.with_name(f"tracked_{src_path.name}")

    if source.isdigit():
        tracker.detect_and_track_video(
            source=int(source),
            output_path=output_path,
            show=args.show,
            save_crops=args.save_crops,
            crops_dir=args.crops_dir,
            save_interval_sec=args.save_interval_sec,
            max_frames=args.max_frames,
            start_time_sec=args.start_time,
        )
    else:
        src_path = Path(source)
        if not src_path.exists():
            print(f"❌ 输入不存在: {source}")
            return
        if src_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
            tracker.detect_and_track_video(
                source=str(src_path),
                output_path=output_path,
                show=args.show,
                save_crops=args.save_crops,
                crops_dir=args.crops_dir,
                save_interval_sec=args.save_interval_sec,
                max_frames=args.max_frames,
                start_time_sec=args.start_time,
            )
        else:
            # 单张图片仅做检测，不做跟踪
            img = cv2.imread(str(src_path))
            if img is None:
                print(f"❌ 无法读取图片: {source}")
                return
            results = tracker.model(img, conf=tracker.conf_threshold, verbose=False)
            boxes = tracker._filter_person_boxes(results[0])
            vis = img.copy()
            for b in boxes:
                x1, y1, x2, y2 = b["bbox"]
                conf = b["conf"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            out_img = src_path.with_name(f"detected_{src_path.name}")
            cv2.imwrite(str(out_img), vis)
            print(f"✅ 检测到 {len(boxes)} 人体，已保存到 {out_img}")
            if args.show:
                cv2.imshow("YOLO Human Detection", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


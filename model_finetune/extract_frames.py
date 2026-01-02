#!/usr/bin/env python3
"""
按给定时间间隔从视频提取帧并保存成图片。

示例：
    python extract_frames.py --video input.mp4 --out frames --interval 2.0
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按时间间隔截取视频帧")
    parser.add_argument(
        "--video",
        required=True,
        help="视频文件路径",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="输出图片目录，不存在会自动创建",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="抽帧间隔（分钟），默认 1 分钟一帧",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="起始时间（秒），默认从视频开头开始",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="输出文件名前缀，默认 frame",
    )
    return parser.parse_args()


def save_frame(frame, output_dir: Path, prefix: str, index: int) -> None:
    output_path = output_dir / f"{prefix}_{index:06d}.jpg"
    cv2.imwrite(str(output_path), frame)


def next_target_time(current: float, interval: float, next_target: float) -> float:
    """避免跳帧时漏掉目标时间点，保证 next_target 始终在当前时间之后。"""
    while next_target <= current:
        next_target += interval
    return next_target


def get_duration_seconds(cap: cv2.VideoCapture) -> Optional[float]:
    """获取视频总时长（秒）；若无法获取则返回 None。兼容无 CAP_PROP_DURATION 的构建。"""
    duration_prop = getattr(cv2, "CAP_PROP_DURATION", None)
    if duration_prop is not None:
        duration_ms = cap.get(duration_prop)
        if duration_ms and duration_ms > 0:
            return duration_ms / 1000.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps and total_frames and fps > 0 and total_frames > 0:
        return total_frames / fps

    return None


def extract_frames(
    video_path: Path, output_dir: Path, interval: float, prefix: str, start: float
) -> None:
    if interval <= 0:
        raise ValueError("interval 必须大于 0")
    if start < 0:
        raise ValueError("start 必须大于等于 0")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    duration = get_duration_seconds(cap)  # 可能为 None 表示未知

    target_time = start  # 秒
    saved = 0

    while True:
        # 若已知视频总时长且超出则提前结束
        if duration is not None and target_time > duration:
            break

        cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break

        save_frame(frame, output_dir, prefix, saved)
        saved += 1
        target_time += interval  # 直接按间隔跳转，无需逐帧读取

    cap.release()
    print(f"完成，已保存 {saved} 张图片到 {output_dir}")


def main() -> None:
    args = parse_args()
    # interval 输入为分钟，此处统一转换为秒
    interval_seconds = args.interval * 60.0
    extract_frames(Path(args.video), Path(args.out), interval_seconds, args.prefix, args.start)


if __name__ == "__main__":
    main()


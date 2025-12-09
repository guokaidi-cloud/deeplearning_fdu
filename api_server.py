#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简易后端 API：接收前端上传的视频并调用现有检测脚本

依赖:
  pip install fastapi uvicorn python-multipart

启动:
  uvicorn api_server:app --host 0.0.0.0 --port 8000

前端:
  将 frontend/index.html 的 API_ENDPOINT 改为 "http://127.0.0.1:8000/api/detect"
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Optional
import uuid


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


app = FastAPI(title="DL Detection API", version="0.1.0")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# 允许跨域，方便前端在不同端口访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _transcode_to_h264(src: Path) -> Optional[Path]:
    """
    使用系统 ffmpeg 转码为 H.264，提升浏览器兼容性。
    仅当 ffmpeg 可用且转码成功时返回新路径，否则返回 None。
    """
    if shutil.which("ffmpeg") is None:
        return None

    dst = src.with_name(f"{src.stem}_h264{src.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "copy",
        str(dst),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception as exc:
        print(f"[transcode] exception: {exc}")
        return None

    if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        print(f"[transcode] success -> {dst}")
        return dst

    print(f"[transcode] failed code={proc.returncode}, tail={proc.stdout[-500:]}")
    return None


def _safe_name(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", ".", " ")).strip() or "video"


@app.post("/api/detect")
async def detect(
    request: Request,
    video: UploadFile = File(...),
    start_time: str = Form("0"),
    model: str = Form("yolov8"),
    conf: float = Form(0.25),
    max_frames: Optional[int] = Form(None),
):
    ts = int(time.time())
    src_name = _safe_name(video.filename)
    input_path = UPLOAD_DIR / f"{ts}_{src_name}"
    output_path = OUTPUT_DIR / f"{input_path.stem}_result.mp4"

    # 保存上传视频
    try:
        with input_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存上传文件失败: {e}")

    # 选择脚本与参数
    cmd = [sys.executable]
    if model == "opencv":
        script = BASE_DIR / "yolov8_face_detection" / "opencv_face_detector.py"
        cmd += [
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--start-time",
            str(start_time),
        ]
        if max_frames is not None:
            cmd += ["--max-frames", str(max_frames)]
    elif model in ("yolov8", "yolov12"):
        script = BASE_DIR / "yolov8_face_detection" / "yolov8_face_detector.py"
        mapped_model = "yolov12l-face" if model == "yolov12" else "yolov8n-face"
        cmd += [
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--start-time",
            str(start_time),
            "--model",
            mapped_model,
            "--conf",
            str(conf),
        ]
        if max_frames is not None:
            cmd += ["--max-frames", str(max_frames)]
    else:
        raise HTTPException(status_code=400, detail="暂不支持的模型类型")

    # 运行外部脚本
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动检测失败: {e}")

    if proc.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "message": "检测脚本执行失败",
                "log": proc.stdout[-4000:],  # 限制输出长度
            },
        )

    if not output_path.exists():
        return JSONResponse(
            status_code=500,
            content={
                "message": "检测完成但未找到输出文件",
                "log": proc.stdout[-4000:],
            },
        )

    # 可选：将结果转码为 H.264，提升浏览器兼容性
    transcoded_path = _transcode_to_h264(output_path)

    # 返回可访问的结果地址（绝对地址，避免端口/域名不一致）
    base = str(request.base_url).rstrip("/")
    final_path = transcoded_path or output_path
    output_url = f"{base}/outputs/{final_path.name}"
    original_url = f"{base}/outputs/{output_path.name}"

    return {
        "outputVideoUrl": output_url,
        "originalVideoUrl": original_url,
        "message": "检测成功",
        "log": proc.stdout[-2000:],
        "transcoded": bool(transcoded_path),
    }


from pathlib import Path

from ultralytics import YOLO


def main():
    """
    使用 data_split_output2 数据集微调 yolov12l-face.pt。
    - 默认使用 GPU 0,如需 CPU 可将 device 改为 "cpu"。
    - 根据显存情况调整 batch/epochs/imgsz。
    """
    root = Path(__file__).resolve().parent
    model_path = root / "model" / "yolov8n-face.pt"
    data_yaml = root / "data_split_output2" / "dataset.yaml"

    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=2,
        device=0,
        workers=4,
        project=str(root / "runs"),
        name="yolov8n_face_finetune",
        pretrained=True,
    )


if __name__ == "__main__":
    main()
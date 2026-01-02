from ultralytics import YOLO
import pynvml


model = YOLO("best.pt")
model.export(format='onnx')
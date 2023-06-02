import os

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x.yaml") # build a new model from scratch

results = model.train(data="data.yaml", epochs=100, batch=20)
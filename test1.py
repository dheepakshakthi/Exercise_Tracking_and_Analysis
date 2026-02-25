from ultralytics import YOLO

model = YOLO('yolov8m-pose.pt')
results = model(source="data/c1.mp4", show=True, conf=0.4, save=True)
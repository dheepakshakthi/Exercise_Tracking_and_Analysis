from ultralytics import YOLO

model = YOLO('yolov8l-pose.pt')
results = model(source="c1.mp4", show=True, conf=0.33, save=True)
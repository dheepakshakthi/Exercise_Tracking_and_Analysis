import cv2
import mediapipe as mp
import urllib.request
import os

# Download the pose landmarker model if not present
MODEL_PATH = 'models/pose_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print('Downloading pose landmarker model...')
    url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
    urllib.request.urlretrieve(url, MODEL_PATH)
    print('Download complete.')

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Connections between landmarks for drawing skeleton
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
)

cap = cv2.VideoCapture("data/c1.mp4")  # Use 0 for webcam

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert BGR to RGB and wrap in MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Get timestamp and run detection
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw landmarks and skeleton
        if result.pose_landmarks:
            h, w = frame.shape[:2]
            for landmarks in result.pose_landmarks:
                # Draw connections
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        x1 = int(landmarks[start_idx].x * w)
                        y1 = int(landmarks[start_idx].y * h)
                        x2 = int(landmarks[end_idx].x * w)
                        y2 = int(landmarks[end_idx].y * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw landmark points
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        cv2.imshow('MediaPipe Pose Tracking', frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()

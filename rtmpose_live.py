# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rtmlib",
#   "onnxruntime",
#   "opencv-python",
# ]
# ///

import time
import threading
import cv2
from rtmlib import Body, draw_skeleton

# ── Config ─────────────────────────────────────────────────────────────────────
CAMERA_ID    = 0            # webcam index (0 = default)
MODE         = 'lightweight' # 'lightweight' | 'balanced' | 'performance'
SCORE_THR    = 0.3
# Inference resolution — smaller = faster. Keypoints are mapped back to display size.
INFER_WIDTH  = 640
INFER_HEIGHT = 480
# ───────────────────────────────────────────────────────────────────────────────


class CameraThread:
    """Continuously grabs frames in a background thread so inference never waits on I/O."""
    def __init__(self, camera_id: int):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_id}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise latency
        self.frame = None
        self.lock  = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return (self.frame is not None), (
                self.frame.copy() if self.frame is not None else None
            )

    def release(self):
        self._stop.set()
        self._thread.join()
        self.cap.release()


# ── RTMO: single-stage model (one ONNX instead of detector + pose) ─────────────
# Replaces the two-stage YOLOX → RTMPose pipeline with a single RTMO model.
body = Body(
    pose='rtmo',            # triggers single-stage RTMO pipeline
    mode=MODE,
    to_openpose=False,
    backend='onnxruntime',
    device='cpu',
)

cam = CameraThread(CAMERA_ID)
print("RTMPose live inference started. Press 'q' or Esc to quit.")

prev_time = time.time()

while True:
    ok, frame = cam.read()
    if not ok or frame is None:
        time.sleep(0.005)
        continue

    # Downscale for inference (major speed-up for the detector)
    small = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))

    # Run inference on the small frame
    keypoints, scores = body(small)

    # Scale keypoints back to display resolution
    disp_h, disp_w = frame.shape[:2]
    if keypoints is not None and len(keypoints):
        keypoints = keypoints.copy().astype(float)
        keypoints[..., 0] *= disp_w / INFER_WIDTH
        keypoints[..., 1] *= disp_h / INFER_HEIGHT

    # Draw skeleton on full-res frame
    annotated = draw_skeleton(frame, keypoints, scores,
                               openpose_skeleton=False, kpt_thr=SCORE_THR)

    # FPS overlay
    now = time.time()
    fps = 1.0 / (now - prev_time + 1e-9)
    prev_time = now
    cv2.putText(annotated, f'FPS: {fps:.1f}  RTMO-{MODE}  {INFER_WIDTH}x{INFER_HEIGHT}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('RTMPose Live', annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cam.release()
cv2.destroyAllWindows()
print("Stopped.")

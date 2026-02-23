# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rtmlib",
#   "onnxruntime-gpu",
#   "opencv-python",
# ]
# ///

import argparse
import time
import threading
import cv2
from rtmlib import Body, draw_skeleton

# ── Config ─────────────────────────────────────────────────────────────────────
MODE      = 'lightweight'  # 'lightweight' | 'balanced' | 'performance'
SCORE_THR = 0.3
INFER_WIDTH  = 640
INFER_HEIGHT = 480
# ───────────────────────────────────────────────────────────────────────────────


class CameraThread:
    """Continuously grabs frames in a background thread so inference never waits on I/O."""
    def __init__(self, camera_id: int):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_id}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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


def run_on_video(source: str, body: Body, save_path: str | None, full_res: bool = True):
    """Inference on a video file — plain cap.read() loop, optional save.

    full_res=True  : feed the original frame to the model (two-stage pipeline,
                     matches test3.py — best accuracy).
    full_res=False : downscale to INFER_WIDTH×INFER_HEIGHT first (RTMO fast mode).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source!r}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"[RTMPose] Saving output to: {save_path}")

    print(f"[RTMPose] Video: {source}  ({width}x{height} @ {fps:.1f} fps,  {total} frames)")
    cv2.namedWindow("RTMPose Video", cv2.WINDOW_NORMAL)

    frame_idx  = 0
    prev_time  = time.time()
    fps_smooth = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if full_res:
            # Full resolution → two-stage model crops the person internally
            keypoints, scores = body(frame)
        else:
            # Downscale → RTMO processes smaller frame, scale kpts back up
            small = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
            keypoints, scores = body(small)
            if keypoints is not None and len(keypoints):
                keypoints = keypoints.copy().astype(float)
                keypoints[..., 0] *= width  / INFER_WIDTH
                keypoints[..., 1] *= height / INFER_HEIGHT

        annotated = draw_skeleton(frame, keypoints, scores,
                                  openpose_skeleton=False, kpt_thr=SCORE_THR)

        now = time.time()
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(now - prev_time, 1e-9))
        prev_time  = now
        frame_idx += 1

        progress = f"{frame_idx}/{total}" if total > 0 else str(frame_idx)
        cv2.putText(annotated,
                    f"FPS: {fps_smooth:.1f}  RTMO-{MODE}  frame {progress}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if writer:
            writer.write(annotated)

        cv2.imshow("RTMPose Video", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[RTMPose] Done. Processed {frame_idx} frames.")


def run_on_webcam(camera_id: int, body: Body):
    """Live webcam inference using a background camera thread."""
    cam = CameraThread(camera_id)

    import onnxruntime as _ort
    _providers  = _ort.get_available_providers()
    _device_tag = "GPU (CUDA)" if "CUDAExecutionProvider" in _providers else "CPU"
    print(f"[RTMPose] Live inference on {_device_tag}.")
    print(f"[RTMPose] Available ORT providers: {_providers}")
    print("[RTMPose] Press 'q' or Esc to quit.")

    cv2.namedWindow("RTMPose Live", cv2.WINDOW_NORMAL)
    prev_time  = time.time()
    fps_smooth = 0.0

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            disp_h, disp_w = frame.shape[:2]
            small = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
            keypoints, scores = body(small)

            if keypoints is not None and len(keypoints):
                keypoints = keypoints.copy().astype(float)
                keypoints[..., 0] *= disp_w / INFER_WIDTH
                keypoints[..., 1] *= disp_h / INFER_HEIGHT

            annotated = draw_skeleton(frame, keypoints, scores,
                                      openpose_skeleton=False, kpt_thr=SCORE_THR)

            now = time.time()
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(now - prev_time, 1e-9))
            prev_time  = now

            cv2.putText(annotated,
                        f"FPS: {fps_smooth:.1f}  RTMO-{MODE}  {INFER_WIDTH}x{INFER_HEIGHT}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("RTMPose Live", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[RTMPose] Stopped.")


# ── Entry point ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RTMPose live / video inference")
parser.add_argument("--source", default="0",
                    help="Camera index (default: 0) or path to a video file")
parser.add_argument("--mode",   default=MODE,
                    choices=["lightweight", "balanced", "performance"],
                    help="Model size (default: lightweight)")
parser.add_argument("--save",   default=None,
                    help="Save annotated output to this .mp4 path (video mode only)")
parser.add_argument("--rtmo",   action="store_true",
                    help="Use single-stage RTMO model (faster but less accurate). "
                         "Default for webcam. NOT default for video files.")
parser.add_argument("--device", default="cuda",
                help="onnxruntime device: cuda or cpu (default: cuda)")
args = parser.parse_args()

MODE = args.mode

# Decide: webcam or video file
try:
    source = int(args.source)    # camera index
    use_rtmo = True              # RTMO is always used for live (speed matters)
except ValueError:
    source = args.source         # video file path
    use_rtmo = args.rtmo         # two-stage by default for video (accuracy matters)

if use_rtmo:
    body = Body(
        pose='rtmo',
        mode=MODE,
        to_openpose=False,
        backend='onnxruntime',
        device=args.device,
    )
    print(f"[RTMPose] Model: RTMO single-stage ({MODE})")
else:
    body = Body(
        mode=MODE,
        to_openpose=False,
        backend='onnxruntime',
        device=args.device,
    )
    print(f"[RTMPose] Model: two-stage RTMDet + RTMPose ({MODE}) – full resolution")

if isinstance(source, int):
    run_on_webcam(source, body)
else:
    run_on_video(source, body, save_path=args.save, full_res=not use_rtmo)


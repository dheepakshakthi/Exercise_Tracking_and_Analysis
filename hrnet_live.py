"""
hrnet_live.py – Live webcam pose inference using HRNet-Pose (Qualcomm AI Hub ONNX).

Model  : job_jpx1yqw1g_optimized_onnx/model.onnx
Input  : [1, 3, 256, 192]  – float32, ImageNet-normalised RGB
Output : [1, 17, 64, 48]   – heatmaps for 17 COCO keypoints

Features
--------
• Letterboxed pre-processing preserves aspect ratio
• Sub-pixel heatmap decode (argmax + 0.5 offset)
• Per-limb group colour-coded skeleton
• Real-time joint-angle overlay (knee / hip / shoulder / elbow)
  → useful for physiotherapy exercise monitoring
• Background camera thread → no dropped frames during inference
• GPU (CUDA) used automatically when available

Usage
-----
    python hrnet_live.py                  # default webcam
    python hrnet_live.py --source 1       # second webcam
    python hrnet_live.py --source video.mp4
    python hrnet_live.py --score-thr 0.4  # stricter confidence
"""

import argparse
import time
import math
import cv2
import numpy as np
import onnxruntime as ort

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH  = "job_jpx1yqw1g_optimized_onnx/model.onnx"
MODEL_W, MODEL_H = 192, 256          # model input width, height (pixels)
HEATMAP_W,  HEATMAP_H  = 48, 64     # heatmap output size
SCORE_THR_DEFAULT      = 0.2        # minimum heatmap confidence to show a keypoint

# ImageNet normalisation
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# ───────────────────────────────────────────────────────────────────────────────

# ── COCO-17 metadata ───────────────────────────────────────────────────────────
KEYPOINT_NAMES = [
    "nose",         # 0
    "left_eye",     # 1
    "right_eye",    # 2
    "left_ear",     # 3
    "right_ear",    # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Skeleton: list of (kpt_idx_a, kpt_idx_b) connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                    # shoulder bar
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12),                          # torso sides
    (11, 12),                                  # hip bar
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

# BGR colours per limb (matches SKELETON order)
_C = dict(
    head = (255, 220,  60),
    torso= ( 60, 220, 100),
    l_arm= (255, 100,  50),
    r_arm= ( 50, 100, 255),
    l_leg= (200,  60, 255),
    r_leg= ( 50, 220, 255),
)
SKELETON_COLORS = [
    _C["head"], _C["head"], _C["head"], _C["head"],
    _C["torso"],
    _C["l_arm"], _C["l_arm"],
    _C["r_arm"], _C["r_arm"],
    _C["torso"], _C["torso"], _C["torso"],
    _C["l_leg"], _C["l_leg"],
    _C["r_leg"], _C["r_leg"],
]

KPT_COLORS = [
    _C["head"],  _C["head"],  _C["head"],  _C["head"],  _C["head"],
    _C["l_arm"], _C["r_arm"],
    _C["l_arm"], _C["r_arm"],
    _C["l_arm"], _C["r_arm"],
    _C["l_leg"], _C["r_leg"],
    _C["l_leg"], _C["r_leg"],
    _C["l_leg"], _C["r_leg"],
]

# Joint angles to compute: (angle_label, vertex_idx, point_a_idx, point_b_idx)
#   angle = angle at `vertex` between rays vertex→a and vertex→b
JOINT_ANGLES = [
    ("L Shoulder", 5,  7, 11),
    ("R Shoulder", 6,  8, 12),
    ("L Elbow",    7,  5,  9),
    ("R Elbow",    8,  6, 10),
    ("L Hip",     11,  5, 13),
    ("R Hip",     12,  6, 14),
    ("L Knee",    13, 11, 15),
    ("R Knee",    14, 12, 16),
]
# ───────────────────────────────────────────────────────────────────────────────


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    BGR uint8 → [1, 3, 256, 192] float32 ImageNet-normalised tensor.

    Uses a plain resize (no letterbox).  HRNet is trained on directly-resized
    person crops; letterboxing introduces grey borders that distort the heatmaps
    near the frame edges and reduce keypoint confidence.
    """
    resized = cv2.resize(img_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return rgb.transpose(2, 0, 1)[np.newaxis]   # HWC → NCHW


# ── Post-processing ────────────────────────────────────────────────────────────
def decode_heatmaps(heatmaps: np.ndarray):
    """
    Decode [1, 17, 64, 48] heatmaps → keypoint coordinates in *model* pixel space.

    Returns
    -------
    kpts   : (17, 2) float32 – (x, y) in MODEL_W × MODEL_H coords
    scores : (17,)   float32 – peak heatmap confidence per keypoint
    """
    hm = heatmaps[0]                       # (17, 64, 48)
    _, hm_h, hm_w = hm.shape
    stride_x = MODEL_W / hm_w              # 192 / 48 = 4.0
    stride_y = MODEL_H / hm_h             # 256 / 64 = 4.0

    flat    = hm.reshape(len(hm), -1)
    scores  = flat.max(axis=1)
    argmax  = flat.argmax(axis=1)
    row_idx = argmax // hm_w
    col_idx = argmax  % hm_w

    # +0.5 centres the sample at the heatmap pixel
    x = (col_idx + 0.5) * stride_x
    y = (row_idx + 0.5) * stride_y
    return np.stack([x, y], axis=1).astype(np.float32), scores.astype(np.float32)


# ── Geometry helpers ───────────────────────────────────────────────────────────
def angle_at_vertex(v, a, b) -> float:
    """
    Compute the interior angle (°) at point *v* between rays v→a and v→b.

    Parameters
    ----------
    v, a, b : array-like (x, y) – keypoint coordinates
    """
    va = np.array(a, dtype=float) - np.array(v, dtype=float)
    vb = np.array(b, dtype=float) - np.array(v, dtype=float)
    norm_va = np.linalg.norm(va)
    norm_vb = np.linalg.norm(vb)
    if norm_va < 1e-6 or norm_vb < 1e-6:
        return 0.0
    cos_theta = np.dot(va, vb) / (norm_va * norm_vb)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


# ── Drawing ────────────────────────────────────────────────────────────────────
def draw_pose(frame: np.ndarray,
              kpts: np.ndarray, scores: np.ndarray,
              score_thr: float) -> np.ndarray:
    """
    Project keypoints from model space back to original frame resolution and draw:
      • colour-coded skeleton limbs
      • filled keypoint circles
      • joint-angle labels at the vertex keypoints

    Keypoints are in model pixel space (0..MODEL_W, 0..MODEL_H); we scale
    them back using independent x/y ratios (direct resize, no letterbox).

    Returns a copy of *frame* with annotations.
    """
    disp   = frame.copy()
    fh, fw = frame.shape[:2]
    sx, sy = fw / MODEL_W, fh / MODEL_H   # independent x/y scale factors

    def to_frame(xy):
        """Model coords → original frame pixel coords."""
        x = int(np.clip(xy[0] * sx, 0, fw - 1))
        y = int(np.clip(xy[1] * sy, 0, fh - 1))
        return x, y

    # --- skeleton limbs ---
    for idx, (i, j) in enumerate(SKELETON):
        if scores[i] >= score_thr and scores[j] >= score_thr:
            cv2.line(disp, to_frame(kpts[i]), to_frame(kpts[j]),
                     SKELETON_COLORS[idx], 2, cv2.LINE_AA)

    # --- keypoint circles ---
    for idx, (pt, sc) in enumerate(zip(kpts, scores)):
        if sc >= score_thr:
            cx, cy = to_frame(pt)
            cv2.circle(disp, (cx, cy), 5, KPT_COLORS[idx], -1, cv2.LINE_AA)
            cv2.circle(disp, (cx, cy), 6, (0, 0, 0),       1,  cv2.LINE_AA)

    # --- joint angle labels ---
    for label, v_idx, a_idx, b_idx in JOINT_ANGLES:
        if (scores[v_idx] >= score_thr and
                scores[a_idx] >= score_thr and
                scores[b_idx] >= score_thr):
            ang = angle_at_vertex(kpts[v_idx], kpts[a_idx], kpts[b_idx])
            cx, cy = to_frame(kpts[v_idx])
            # OpenCV bitmap fonts do NOT support the Unicode degree symbol (°);
            # use the ASCII literal "deg" instead to avoid rendering failures.
            text = f"{ang:.0f} deg"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(disp, (cx + 7, cy - th - 3), (cx + 9 + tw, cy + 3),
                          (0, 0, 0), -1)
            cv2.putText(disp, text, (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return disp


def draw_angle_panel(frame: np.ndarray,
                     kpts: np.ndarray, scores: np.ndarray,
                     score_thr: float) -> np.ndarray:
    """
    Draw a semi-transparent panel in the top-right corner listing all joint angles.
    Useful for physiotherapy monitoring at a glance.
    """
    lines = []
    for label, v_idx, a_idx, b_idx in JOINT_ANGLES:
        if (scores[v_idx] >= score_thr and
                scores[a_idx] >= score_thr and
                scores[b_idx] >= score_thr):
            ang = angle_at_vertex(kpts[v_idx], kpts[a_idx], kpts[b_idx])
            lines.append(f"{label:<12}: {ang:>5.1f} deg")
        else:
            lines.append(f"{label:<12}:  ---")

    if not lines:
        return frame

    fh, fw = frame.shape[:2]
    panel_w, panel_h = 240, 18 * len(lines) + 16
    x0 = fw - panel_w - 8
    y0 = 40

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0 + 8, y0 + 14 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 255, 200), 1, cv2.LINE_AA)
    return frame


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HRNet live pose inference")
    parser.add_argument("--source",    default=0,
                        help="Camera index or video file path (default: 0)")
    parser.add_argument("--model",     default=MODEL_PATH,
                        help="Path to HRNet ONNX model")
    parser.add_argument("--score-thr", default=SCORE_THR_DEFAULT, type=float,
                        help="Minimum heatmap confidence (default: 0.3)")
    parser.add_argument("--no-angles", action="store_true",
                        help="Disable joint-angle overlay")
    args = parser.parse_args()

    # Parse source: int for webcam index, str for file path
    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"[HRNet] Loading model: {args.model}")
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, providers=providers)
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"[HRNet] Provider  : {sess.get_providers()[0]}")
    print(f"[HRNet] Input     : {in_name}  {sess.get_inputs()[0].shape}")
    print(f"[HRNet] Output    : {out_name} {sess.get_outputs()[0].shape}")

    # Warm-up inference session (first call is slow due to JIT init)
    dummy = np.zeros((1, 3, MODEL_H, MODEL_W), dtype=np.float32)
    sess.run([out_name], {in_name: dummy})
    print("[HRNet] Warm-up done.")

    # ── Open camera / video (plain cap.read() – no threading) ─────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    print(f"[HRNet] Source    : {source}")
    print("[HRNet] Press 'q' or Esc to quit, 'a' to toggle angle panel.")

    cv2.namedWindow("HRNet Pose - Physiotherapy Monitor", cv2.WINDOW_NORMAL)

    show_panel = True
    prev_time  = time.time()
    fps_smooth = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

            # ── Pre-process ───────────────────────────────────────────────────
        tensor = preprocess(frame)

        heatmaps = sess.run([out_name], {in_name: tensor})[0]

        kpts, scores = decode_heatmaps(heatmaps)

        annotated = draw_pose(frame, kpts, scores, args.score_thr)

        if show_panel and not args.no_angles:
            annotated = draw_angle_panel(annotated, kpts, scores, args.score_thr)

        now = time.time()
        instant_fps = 1.0 / max(now - prev_time, 1e-9)
        fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps
        prev_time = now

        provider_tag = "GPU" if "CUDA" in sess.get_providers()[0] else "CPU"
        cv2.putText(annotated,
                    f"FPS: {fps_smooth:.1f}  HRNet-Pose [{provider_tag}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("HRNet Pose - Physiotherapy Monitor", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):      # q / Esc → quit
            break
        elif key == ord("a"):          # 'a' → toggle angle panel
            show_panel = not show_panel

    cap.release()
    cv2.destroyAllWindows()
    print("[HRNet] Stopped.")


if __name__ == "__main__":
    main()

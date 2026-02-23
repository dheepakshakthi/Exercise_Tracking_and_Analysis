import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
from rtmlib import Body

# -------------------- KEYPOINT DEFINITIONS --------------------

KP = {
    "NOSE": 0,
    "LEFT_EYE": 1,
    "RIGHT_EYE": 2,
    "LEFT_EAR": 3,
    "RIGHT_EAR": 4,
    "LEFT_SHOULDER": 5,
    "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW": 7,
    "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9,
    "RIGHT_WRIST": 10,
    "LEFT_HIP": 11,
    "RIGHT_HIP": 12,
    "LEFT_KNEE": 13,
    "RIGHT_KNEE": 14,
    "LEFT_ANKLE": 15,
    "RIGHT_ANKLE": 16,
}

KP_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
]

TARGETS = {
    "hip_flexion": (100, 120),
    "knee_flexion": (90, 120),
    "ankle_dorsiflexion": (15, 25),
    "trunk_forward_lean": (20, 45),
    "pelvic_tilt": (0, 10),
}

# -------------------- GEOMETRY HELPERS --------------------

def vector_angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    ab_len = math.hypot(*ab)
    cb_len = math.hypot(*cb)
    if ab_len == 0 or cb_len == 0:
        return None
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    cos_val = max(-1.0, min(1.0, dot / (ab_len * cb_len)))
    return math.degrees(math.acos(cos_val))


def flexion_from_angle(a):
    return None if a is None else max(0.0, 180.0 - a)


def average_ignore_none(a, b):
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return 0.5 * (a + b)


def status_color(v, lo, hi):
    if v is None:
        return (0, 165, 255)
    return (0, 200, 0) if lo <= v <= hi else (0, 0, 255)


def get_point(kps, name):
    i = KP[name]
    if i >= len(kps):
        return None
    return float(kps[i][0]), float(kps[i][1])

# -------------------- METRICS --------------------

def compute_metrics(kps):
    ls, rs = get_point(kps, "LEFT_SHOULDER"), get_point(kps, "RIGHT_SHOULDER")
    lh, rh = get_point(kps, "LEFT_HIP"), get_point(kps, "RIGHT_HIP")
    lk, rk = get_point(kps, "LEFT_KNEE"), get_point(kps, "RIGHT_KNEE")
    la, ra = get_point(kps, "LEFT_ANKLE"), get_point(kps, "RIGHT_ANKLE")

    hip = average_ignore_none(
        flexion_from_angle(vector_angle(ls, lh, lk)) if ls and lh and lk else None,
        flexion_from_angle(vector_angle(rs, rh, rk)) if rs and rh and rk else None,
    )

    knee = average_ignore_none(
        flexion_from_angle(vector_angle(lh, lk, la)) if lh and lk and la else None,
        flexion_from_angle(vector_angle(rh, rk, ra)) if rh and rk and ra else None,
    )

    return {
        "hip_flexion": hip,
        "knee_flexion": knee,
    }

# -------------------- DRAWING --------------------

def draw_wireframe(frame, kps):
    for a, b in KP_CONNECTIONS:
        pa, pb = get_point(kps, a), get_point(kps, b)
        if pa and pb:
            cv2.line(frame, tuple(map(int, pa)), tuple(map(int, pb)), (180, 180, 180), 2)
    for n in KP:
        p = get_point(kps, n)
        if p:
            cv2.circle(frame, tuple(map(int, p)), 4, (255, 255, 255), -1)

def overlay_metrics(frame, metrics):
    y = 30
    for k, (lo, hi) in TARGETS.items():
        v = metrics.get(k)
        color = status_color(v, lo, hi)
        text = f"{k.replace('_',' ')}: {'--' if v is None else f'{v:.1f}°'} [{lo}-{hi}]"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 26

# -------------------- MAIN --------------------

def main():
    parser = argparse.ArgumentParser("Squat Angle Analysis (RTMPose)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--mode", choices=["lightweight", "balanced", "performance"], default="balanced")
    parser.add_argument("--device", default="cuda", help="onnxruntime device: cuda or cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 25,
        (w, h),
    )

    # Use keyword to avoid misrouting positional arg as detector URL
    body = Body(pose="rtmo", mode=args.mode, backend="onnxruntime", device=args.device, to_openpose=False)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        resized = cv2.resize(frame, (640, 480))
        keypoints, scores = body(resized)

        if keypoints is not None and len(keypoints):
            kps = keypoints[0].astype(float)
            kps[:, 0] *= w / 640
            kps[:, 1] *= h / 480

            metrics = compute_metrics(kps)
            draw_wireframe(frame, kps)
            overlay_metrics(frame, metrics)

        writer.write(frame)
        cv2.imshow("Squat RTMPose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
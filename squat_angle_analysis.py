import argparse
import math
import os
import urllib.request

import cv2
import mediapipe as mp

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = "pose_landmarker.task"

# MediaPipe Pose landmark indices (tasks API does not expose solutions enum)
POSE_LANDMARK = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

POSE_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
]


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading MediaPipe pose model ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def vector_angle(a, b, c):
    """Return the angle ABC in degrees (0-180)."""
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    ab_len = math.hypot(*ab)
    cb_len = math.hypot(*cb)
    if ab_len == 0 or cb_len == 0:
        return None
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    cos_val = max(-1.0, min(1.0, dot / (ab_len * cb_len)))
    return math.degrees(math.acos(cos_val))


def flexion_from_angle(raw_angle):
    if raw_angle is None:
        return None
    return max(0.0, 180.0 - raw_angle)


def trunk_forward_angle(hip_mid, shoulder_mid):
    vec = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
    length = math.hypot(*vec)
    if length == 0:
        return None
    vec_norm = (vec[0] / length, vec[1] / length)
    vertical = (0.0, -1.0)
    dot = vec_norm[0] * vertical[0] + vec_norm[1] * vertical[1]
    cos_val = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(cos_val))


def pelvic_tilt_angle(left_hip, right_hip):
    dx = right_hip[0] - left_hip[0]
    dy = right_hip[1] - left_hip[1]
    if dx == 0 and dy == 0:
        return None
    return abs(math.degrees(math.atan2(dy, dx)))


def landmark_point(landmarks, idx):
    lm = landmarks[idx]
    return (lm.x, lm.y)


def compute_metrics(landmarks):
    idx = POSE_LANDMARK
    ls = landmark_point(landmarks, idx["LEFT_SHOULDER"])
    rs = landmark_point(landmarks, idx["RIGHT_SHOULDER"])
    lh = landmark_point(landmarks, idx["LEFT_HIP"])
    rh = landmark_point(landmarks, idx["RIGHT_HIP"])
    lk = landmark_point(landmarks, idx["LEFT_KNEE"])
    rk = landmark_point(landmarks, idx["RIGHT_KNEE"])
    la = landmark_point(landmarks, idx["LEFT_ANKLE"])
    ra = landmark_point(landmarks, idx["RIGHT_ANKLE"])
    lfoot = landmark_point(landmarks, idx["LEFT_FOOT_INDEX"])
    rfoot = landmark_point(landmarks, idx["RIGHT_FOOT_INDEX"])

    mid_shoulder = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5)
    mid_hip = ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)

    hip_left_raw = vector_angle(ls, lh, lk)
    hip_right_raw = vector_angle(rs, rh, rk)
    hip_flex = average_ignore_none(
        flexion_from_angle(hip_left_raw),
        flexion_from_angle(hip_right_raw),
    )

    knee_left_raw = vector_angle(lh, lk, la)
    knee_right_raw = vector_angle(rh, rk, ra)
    knee_flex = average_ignore_none(
        flexion_from_angle(knee_left_raw),
        flexion_from_angle(knee_right_raw),
    )

    ankle_left_raw = vector_angle(lk, la, lfoot)
    ankle_right_raw = vector_angle(rk, ra, rfoot)
    ankle_dorsi = average_ignore_none(
        flexion_from_angle(ankle_left_raw),
        flexion_from_angle(ankle_right_raw),
    )

    trunk = trunk_forward_angle(mid_hip, mid_shoulder)
    pelvic = pelvic_tilt_angle(lh, rh)

    return {
        "hip_flexion": hip_flex,
        "knee_flexion": knee_flex,
        "ankle_dorsiflexion": ankle_dorsi,
        "trunk_forward_lean": trunk,
        "pelvic_tilt": pelvic,
    }


def average_ignore_none(a, b):
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return 0.5 * (a + b)


def status_color(value, lower, upper):
    if value is None:
        return (0, 165, 255)
    if lower <= value <= upper:
        return (0, 200, 0)
    return (0, 0, 255)


def landmark_xy(landmarks, name, width, height):
    idx = POSE_LANDMARK[name]
    lm = landmarks[idx]
    return int(lm.x * width), int(lm.y * height)


def draw_wireframe(frame, landmarks):
    h, w = frame.shape[:2]
    neutral = (180, 180, 180)
    for a, b in POSE_CONNECTIONS:
        p1 = landmark_xy(landmarks, a, w, h)
        p2 = landmark_xy(landmarks, b, w, h)
        cv2.line(frame, p1, p2, neutral, 2, cv2.LINE_AA)
    for name in POSE_LANDMARK:
        x, y = landmark_xy(landmarks, name, w, h)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)


def annotate_joint_angles(frame, landmarks, metrics):
    h, w = frame.shape[:2]
    positions = {
        "hip_flexion": landmark_xy(landmarks, "LEFT_HIP", w, h),
        "knee_flexion": landmark_xy(landmarks, "LEFT_KNEE", w, h),
        "ankle_dorsiflexion": landmark_xy(landmarks, "LEFT_ANKLE", w, h),
        "trunk_forward_lean": landmark_xy(landmarks, "LEFT_SHOULDER", w, h),
        "pelvic_tilt": landmark_xy(landmarks, "RIGHT_HIP", w, h),
    }

    bounds = {
        "hip_flexion": (100, 120),
        "knee_flexion": (90, 120),
        "ankle_dorsiflexion": (15, 25),
        "trunk_forward_lean": (20, 45),
        "pelvic_tilt": (0, 10),
    }

    for key, (x, y) in positions.items():
        val = metrics.get(key)
        low, high = bounds[key]
        color = status_color(val, low, high)
        label = "--" if val is None else f"{val:.0f}°"
        cv2.putText(frame, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def overlay_metrics(frame, metrics):
    items = [
        ("Hip flexion", metrics["hip_flexion"], (100, 120)),
        ("Knee flexion", metrics["knee_flexion"], (90, 120)),
        ("Ankle dorsiflex", metrics["ankle_dorsiflexion"], (15, 25)),
        ("Trunk lean", metrics["trunk_forward_lean"], (20, 45)),
        ("Pelvic tilt", metrics["pelvic_tilt"], (0, 10)),
    ]
    y = 30
    for label, value, bounds in items:
        color = status_color(value, bounds[0], bounds[1])
        val_text = "--" if value is None else f"{value:5.1f} deg"
        rng_text = f"[{bounds[0]}-{bounds[1]}]"
        text = f"{label}: {val_text} {rng_text}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 26


def main():
    parser = argparse.ArgumentParser(description="Squat angle checker (MediaPipe)")
    parser.add_argument("--video", default="c1.mp4", help="Path to input video")
    parser.add_argument("--mirror", action="store_true", help="Flip video horizontally")
    parser.add_argument("--output", default=None, help="Path to save annotated video (mp4)")
    args = parser.parse_args()

    ensure_model()

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        output_segmentation_masks=False,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                metrics = compute_metrics(landmarks)
                draw_wireframe(frame, landmarks)
                annotate_joint_angles(frame, landmarks, metrics)
                overlay_metrics(frame, metrics)
            else:
                cv2.putText(frame, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if writer:
                writer.write(frame)
            cv2.imshow("Squat Angle Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

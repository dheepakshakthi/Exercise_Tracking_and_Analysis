# realtime_classifier.py - Real-time exercise classification using trained ML models
#
# Usage:
#   python realtime_classifier.py [--source 0]              # Use webcam (default)
#   python realtime_classifier.py --source video.mp4        # Use video file
#   python realtime_classifier.py --model xgboost           # Choose model (default: xgboost)
#   python realtime_classifier.py --model random_forest
#
# Controls:
#   'q' or Esc - Quit
#   'r' - Reset statistics

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
from rtmlib import Body, draw_skeleton

# ── Config ─────────────────────────────────────────────────────────────────────
MODE = "balanced"  # RTMPose mode
SCORE_THR = 0.3  # Confidence threshold

# Smoothing parameters
KP_ALPHA = 0.5  # Keypoint EMA weight
ANG_ALPHA = 0.5  # Angle EMA weight

# Classification parameters
WINDOW_SIZE = 10  # Number of frames to average for classification
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to show prediction
# ───────────────────────────────────────────────────────────────────────────────

# COCO-17 keypoint indices
KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Joint angles to compute
JOINT_ANGLES = [
    ("left_knee_angle", "left_hip", "left_knee", "left_ankle"),
    ("right_knee_angle", "right_hip", "right_knee", "right_ankle"),
    ("left_hip_angle", "left_shoulder", "left_hip", "left_knee"),
    ("right_hip_angle", "right_shoulder", "right_hip", "right_knee"),
    ("left_elbow_angle", "left_shoulder", "left_elbow", "left_wrist"),
    ("right_elbow_angle", "right_shoulder", "right_elbow", "right_wrist"),
    ("left_shoulder_angle", "left_elbow", "left_shoulder", "left_hip"),
    ("right_shoulder_angle", "right_elbow", "right_shoulder", "right_hip"),
]

# Feature names (must match training)
FEATURE_NAMES = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
]


# ── Helper Functions ───────────────────────────────────────────────────────────


def _angle_at_vertex(
    kps: np.ndarray,
    scores: np.ndarray,
    a: int,
    b: int,
    c: int,
    min_score: float = SCORE_THR,
) -> float | None:
    """Calculate angle at vertex B formed by points A-B-C."""
    if scores[a] < min_score or scores[b] < min_score or scores[c] < min_score:
        return None
    v1 = kps[a] - kps[b]
    v2 = kps[c] - kps[b]
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-6:
        return None
    cos_a = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return round(math.degrees(math.acos(cos_a)), 2)


def compute_angles(kps: np.ndarray, scores: np.ndarray) -> dict:
    """Compute all joint angles for detected person."""
    return {
        name: _angle_at_vertex(kps, scores, KP[a], KP[b], KP[c])
        for name, a, b, c in JOINT_ANGLES
    }


class ExerciseClassifier:
    """Real-time exercise classifier with smoothing."""

    def __init__(self, model_path: Path, window_size: int = WINDOW_SIZE):
        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        self.window_size = window_size
        self.angle_buffer = deque(maxlen=window_size)
        self.smoothed_kps = None
        self.smoothed_angles = {}

        # Statistics
        self.total_frames = 0
        self.correct_frames = 0
        self.incorrect_frames = 0

    def reset_stats(self):
        """Reset classification statistics."""
        self.total_frames = 0
        self.correct_frames = 0
        self.incorrect_frames = 0

    def update(self, keypoints: np.ndarray, scores: np.ndarray) -> dict | None:
        """
        Process new frame and return classification result.

        Returns:
            dict with keys: 'prediction', 'confidence', 'angles', 'smoothed_kps'
            or None if no valid detection
        """
        if keypoints is None or len(keypoints) == 0:
            return None

        # Use first detected person
        kps = keypoints[0]
        sc = scores[0]

        # Compute angles
        angles = compute_angles(kps, sc)

        # Smooth keypoints
        if self.smoothed_kps is None:
            self.smoothed_kps = kps.copy()
        else:
            confident = sc >= SCORE_THR
            self.smoothed_kps[confident] = (
                KP_ALPHA * kps[confident]
                + (1.0 - KP_ALPHA) * self.smoothed_kps[confident]
            )

        # Smooth angles
        for name, val in angles.items():
            prev = self.smoothed_angles.get(name)
            if val is None:
                pass  # Keep previous
            elif prev is None:
                self.smoothed_angles[name] = val
            else:
                self.smoothed_angles[name] = round(
                    ANG_ALPHA * val + (1.0 - ANG_ALPHA) * prev, 2
                )

        # Add to buffer
        self.angle_buffer.append(self.smoothed_angles.copy())

        # Need enough frames for prediction
        if len(self.angle_buffer) < self.window_size:
            return {
                "prediction": None,
                "confidence": 0.0,
                "angles": self.smoothed_angles,
                "smoothed_kps": self.smoothed_kps,
                "smoothed_scores": sc,
            }

        # Prepare features for prediction
        feature_vector = self._prepare_features()

        if feature_vector is None:
            return {
                "prediction": None,
                "confidence": 0.0,
                "angles": self.smoothed_angles,
                "smoothed_kps": self.smoothed_kps,
                "smoothed_scores": sc,
            }

        # Make prediction
        prediction = self.model.predict([feature_vector])[0]

        # Get prediction probability
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([feature_vector])[0]
            confidence = float(proba[prediction])
        else:
            confidence = 1.0

        # Update statistics
        self.total_frames += 1
        if prediction == 1:
            self.correct_frames += 1
        else:
            self.incorrect_frames += 1

        return {
            "prediction": int(prediction),
            "confidence": confidence,
            "angles": self.smoothed_angles,
            "smoothed_kps": self.smoothed_kps,
            "smoothed_scores": sc,
        }

    def _prepare_features(self) -> np.ndarray | None:
        """Prepare feature vector from buffered angles."""
        # Average angles over window
        avg_angles = {}
        for feature_name in FEATURE_NAMES:
            values = [frame.get(feature_name) for frame in self.angle_buffer]
            # Filter out None values
            valid_values = [v for v in values if v is not None]
            if len(valid_values) == 0:
                return None  # Not enough valid data
            avg_angles[feature_name] = np.mean(valid_values)

        # Create feature vector in correct order
        feature_vector = [avg_angles[name] for name in FEATURE_NAMES]
        return np.array(feature_vector)


def draw_classification_overlay(
    frame: np.ndarray,
    result: dict | None,
    fps: float,
    classifier: ExerciseClassifier,
    model_name: str,
) -> None:
    """Draw classification results and statistics on frame."""
    h, w = frame.shape[:2]

    # Draw FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Draw model name
    cv2.putText(
        frame,
        f"Model: {model_name}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if result is None:
        cv2.putText(
            frame,
            "No person detected",
            (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    prediction = result.get("prediction")
    confidence = result.get("confidence", 0.0)

    # Draw classification result
    if prediction is not None:
        if confidence >= CONFIDENCE_THRESHOLD:
            if prediction == 1:
                status = "CORRECT ✓"
                color = (0, 255, 0)  # Green
            else:
                status = "INCORRECT ✗"
                color = (0, 0, 255)  # Red

            # Large status text
            text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 60

            # Background rectangle
            padding = 10
            cv2.rectangle(
                frame,
                (text_x - padding, text_y - text_size[1] - padding),
                (text_x + text_size[0] + padding, text_y + padding),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                status,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                color,
                3,
                cv2.LINE_AA,
            )

            # Confidence
            cv2.putText(
                frame,
                f"Confidence: {confidence:.1%}",
                (text_x, text_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Analyzing...",
                ((w - 150) // 2, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # Draw statistics panel
    panel_x = 10
    panel_y = h - 150
    cv2.rectangle(
        frame, (panel_x - 5, panel_y - 25), (panel_x + 250, h - 10), (0, 0, 0), -1
    )

    cv2.putText(
        frame,
        "Statistics:",
        (panel_x, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    total = classifier.total_frames
    correct = classifier.correct_frames
    incorrect = classifier.incorrect_frames

    cv2.putText(
        frame,
        f"Total: {total}",
        (panel_x, panel_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Correct: {correct} ({100 * correct / max(total, 1):.1f}%)",
        (panel_x, panel_y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Incorrect: {incorrect} ({100 * incorrect / max(total, 1):.1f}%)",
        (panel_x, panel_y + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    # Draw angle values
    if result.get("angles"):
        angles = result["angles"]
        kps = result.get("smoothed_kps")
        scores = result.get("smoothed_scores")

        if kps is not None and scores is not None:
            angle_label_kp = {
                "left_knee_angle": "left_knee",
                "right_knee_angle": "right_knee",
                "left_hip_angle": "left_hip",
                "right_hip_angle": "right_hip",
                "left_elbow_angle": "left_elbow",
                "right_elbow_angle": "right_elbow",
                "left_shoulder_angle": "left_shoulder",
                "right_shoulder_angle": "right_shoulder",
            }

            for angle_key, kp_name in angle_label_kp.items():
                val = angles.get(angle_key)
                if val is None:
                    continue
                kp_idx = KP[kp_name]
                if scores[kp_idx] < SCORE_THR:
                    continue
                x, y = kps[kp_idx].astype(int)
                cv2.putText(
                    frame,
                    f"{val:.0f}°",
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

    # Instructions
    cv2.putText(
        frame,
        "Press 'q' to quit, 'r' to reset stats",
        (w - 400, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(description="Real-time exercise classification")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: 0 for webcam, or path to video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "random_forest"],
        help="Model to use for classification",
    )
    parser.add_argument("--mode", type=str, default=MODE, help="RTMPose mode")

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    if args.model == "xgboost":
        model_path = script_dir / "xgboost_model.pkl"
        model_name = "XGBoost"
    else:
        model_path = script_dir / "random_forest_model.pkl"
        model_name = "Random Forest"

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please run train_models.py first to train the models.")
        return

    # Initialize classifier
    classifier = ExerciseClassifier(model_path)

    # Initialize pose detector
    print(f"Initializing RTMPose ({args.mode} mode)...")
    body = Body(mode=args.mode, to_openpose=False, backend="onnxruntime", device="cuda")

    # Open video source
    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open video source: {source}")
        return

    print(f"\n{'=' * 60}")
    print(f"Real-time Exercise Classification")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Source: {'Webcam' if source == 0 else source}")
    print(f"Mode: {args.mode}")
    print(f"\nControls:")
    print(f"  'q' or Esc - Quit")
    print(f"  'r' - Reset statistics")
    print(f"{'=' * 60}\n")

    cv2.namedWindow("Exercise Classifier", cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps_smooth = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame")
                break

            # Detect pose
            keypoints, scores = body(frame)

            if keypoints is None or len(keypoints) == 0:
                keypoints = np.empty((0, 17, 2), dtype=float)
                scores = np.empty((0, 17), dtype=float)

            # Draw skeleton
            annotated = draw_skeleton(
                frame, keypoints, scores, openpose_skeleton=False, kpt_thr=SCORE_THR
            )

            # Classify exercise
            result = classifier.update(keypoints, scores)

            # Calculate FPS
            now = time.time()
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(now - prev_time, 1e-9))
            prev_time = now

            # Draw overlay
            draw_classification_overlay(
                annotated, result, fps_smooth, classifier, model_name
            )

            # Display
            cv2.imshow("Exercise Classifier", annotated)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Esc
                print("\nQuitting...")
                break
            elif key == ord("r"):  # Reset statistics
                classifier.reset_stats()
                print("\nStatistics reset")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Print final statistics
        print(f"\n{'=' * 60}")
        print("Final Statistics:")
        print(f"{'=' * 60}")
        print(f"Total frames: {classifier.total_frames}")
        print(
            f"Correct: {classifier.correct_frames} ({100 * classifier.correct_frames / max(classifier.total_frames, 1):.1f}%)"
        )
        print(
            f"Incorrect: {classifier.incorrect_frames} ({100 * classifier.incorrect_frames / max(classifier.total_frames, 1):.1f}%)"
        )
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

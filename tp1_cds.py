# tp1_cds.py  –  Pose tracking + joint-angle collection for physiotherapy data
#
# Usage:
#   python tp1_cds.py  <video_file>  [--mode lightweight|balanced|performance]
#                                    [--save output.mp4]
#
# Output:
#   tp1_<video_stem>_data.json   (written to cwd)
#
# Each frame record stores angles (degrees) at 8 joints per tracked person.
# None is stored when a joint's keypoints are not confident enough.

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
from rtmlib import Body, draw_skeleton

# ── Config ─────────────────────────────────────────────────────────────────────
MODE         = 'balanced'   # default; overridden by --mode
SCORE_THR    = 0.3

# Smoothing: higher → smoother but more lag (0 = no smoothing, 0.7 = heavy)
KP_ALPHA   = 0.5   # keypoint EMA weight for new detection
ANG_ALPHA  = 0.5   # angle EMA weight for new value

# Tracker patience: frames to keep a track alive without a new detection
TRACK_PATIENCE = 5
# ───────────────────────────────────────────────────────────────────────────────

# COCO-17 keypoint indices produced by RTMPose
KP = {
    'nose':            0,
    'left_eye':        1,  'right_eye':        2,
    'left_ear':        3,  'right_ear':         4,
    'left_shoulder':   5,  'right_shoulder':    6,
    'left_elbow':      7,  'right_elbow':       8,
    'left_wrist':      9,  'right_wrist':      10,
    'left_hip':       11,  'right_hip':        12,
    'left_knee':      13,  'right_knee':       14,
    'left_ankle':     15,  'right_ankle':      16,
}

# Each tuple: (output_key, point_A, vertex_B, point_C)  →  angle at B
JOINT_ANGLES = [
    # Lower body
    ('left_knee_angle',       'left_hip',       'left_knee',      'left_ankle'),
    ('right_knee_angle',      'right_hip',      'right_knee',     'right_ankle'),
    ('left_hip_angle',        'left_shoulder',  'left_hip',       'left_knee'),
    ('right_hip_angle',       'right_shoulder', 'right_hip',      'right_knee'),
    # Upper body
    ('left_elbow_angle',      'left_shoulder',  'left_elbow',     'left_wrist'),
    ('right_elbow_angle',     'right_shoulder', 'right_elbow',    'right_wrist'),
    ('left_shoulder_angle',   'left_elbow',     'left_shoulder',  'left_hip'),
    ('right_shoulder_angle',  'right_elbow',    'right_shoulder', 'right_hip'),
]


# ── Math helpers ───────────────────────────────────────────────────────────────

def _angle_at_vertex(kps: np.ndarray, scores: np.ndarray,
                     a: int, b: int, c: int,
                     min_score: float = SCORE_THR) -> float | None:
    """Angle (degrees) at keypoint B, formed by the rays B→A and B→C.
    Returns None when any of the three keypoints is below confidence threshold."""
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
    """Return dict of all joint angles for one detected person."""
    return {
        name: _angle_at_vertex(kps, scores, KP[a], KP[b], KP[c])
        for name, a, b, c in JOINT_ANGLES
    }


# ── Robust centroid tracker with EMA smoothing + patience ─────────────────────

class Track:
    """State for a single tracked person."""
    __slots__ = ('id', 'smoothed_kps', 'smoothed_angles', 'centroid', 'missed')

    def __init__(self, tid: int, kps: np.ndarray, angles: dict):
        self.id              = tid
        self.smoothed_kps    = kps.copy()
        self.smoothed_angles = {k: v for k, v in angles.items()}
        self.centroid        = _visible_centroid(kps, np.ones(len(kps)))
        self.missed          = 0   # consecutive frames without a detection match


def _visible_centroid(kps: np.ndarray, scores: np.ndarray) -> np.ndarray:
    visible = kps[scores >= SCORE_THR]
    return visible.mean(axis=0) if len(visible) else kps.mean(axis=0)


class SmoothTracker:
    """
    Centroid-based nearest-neighbour tracker with:
      - EMA smoothing on keypoint positions  (reduces jitter)
      - EMA smoothing on computed angles     (reduces angle flicker)
      - Patience: keeps a track alive for TRACK_PATIENCE frames after it goes
        missing so a brief detection dropout doesn't create a new ID
    """

    def __init__(self, max_dist: float = 120.0):
        self.next_id  = 0
        self.tracks:  dict[int, Track] = {}
        self.max_dist = max_dist

    def update(self, keypoints: np.ndarray, scores: np.ndarray
               ) -> list[tuple[int, np.ndarray, np.ndarray, dict]]:
        """
        Match detections to existing tracks, apply EMA, handle patience.

        Returns list of (track_id, smoothed_kps, scores, smoothed_angles)
        for every *active* track (including coasted ones).
        """
        n = len(keypoints)

        # ── 1. Compute angles for each raw detection ───────────────────────────
        raw_angles = []
        for i in range(n):
            raw_angles.append(compute_angles(keypoints[i], scores[i]))

        # ── 2. Match detections → tracks ───────────────────────────────────────
        matched_det  = {}    # track_id → detection index
        unmatched    = list(range(n))

        if self.tracks and n > 0:
            track_ids   = list(self.tracks.keys())
            track_cents = np.array([self.tracks[tid].centroid for tid in track_ids])
            det_cents   = np.array([_visible_centroid(keypoints[i], scores[i])
                                    for i in range(n)])
            used_det    = set()

            for j, tid in enumerate(track_ids):
                dists = np.linalg.norm(det_cents - track_cents[j], axis=1)
                order = np.argsort(dists)
                for best in order:
                    if dists[best] < self.max_dist and best not in used_det:
                        matched_det[tid] = best
                        used_det.add(best)
                        break

            unmatched = [i for i in range(n) if i not in used_det]

        # ── 3. Update matched tracks with EMA ──────────────────────────────────
        for tid, det_i in matched_det.items():
            tr = self.tracks[tid]
            kps = keypoints[det_i]
            sc  = scores[det_i]

            # EMA on keypoints (only confident ones)
            confident = sc >= SCORE_THR
            tr.smoothed_kps[confident] = (
                KP_ALPHA * kps[confident] +
                (1.0 - KP_ALPHA) * tr.smoothed_kps[confident]
            )
            tr.centroid = _visible_centroid(tr.smoothed_kps, sc)
            tr.missed   = 0

            # EMA on angles
            for name, val in raw_angles[det_i].items():
                prev = tr.smoothed_angles.get(name)
                if val is None:
                    pass   # keep previous smoothed value
                elif prev is None:
                    tr.smoothed_angles[name] = val
                else:
                    tr.smoothed_angles[name] = round(
                        ANG_ALPHA * val + (1.0 - ANG_ALPHA) * prev, 2)

        # ── 4. Age unmatched tracks; remove if patience exhausted ──────────────
        for tid in list(self.tracks):
            if tid not in matched_det:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > TRACK_PATIENCE:
                    del self.tracks[tid]

        # ── 5. Create new tracks for unmatched detections ──────────────────────
        for det_i in unmatched:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid, keypoints[det_i], raw_angles[det_i])

        # ── 6. Return all live tracks ──────────────────────────────────────────
        result = []
        for tid, tr in self.tracks.items():
            # Find matching detection's scores for display (fallback: all zeros)
            if tid in matched_det:
                sc = scores[matched_det[tid]]
            else:
                sc = np.zeros(len(tr.smoothed_kps), dtype=float)
            result.append((tid, tr.smoothed_kps, sc, tr.smoothed_angles))
        return result


# ── Annotation helper ──────────────────────────────────────────────────────────

# Maps each angle key to the keypoint at whose pixel position the label goes
_ANGLE_LABEL_KP = {
    'left_knee_angle':      'left_knee',
    'right_knee_angle':     'right_knee',
    'left_hip_angle':       'left_hip',
    'right_hip_angle':      'right_hip',
    'left_elbow_angle':     'left_elbow',
    'right_elbow_angle':    'right_elbow',
    'left_shoulder_angle':  'left_shoulder',
    'right_shoulder_angle': 'right_shoulder',
}


def overlay_angles(frame: np.ndarray, kps: np.ndarray, scores: np.ndarray,
                   angles: dict, person_id: int) -> None:
    """Draws angle values and person ID onto the frame in-place."""
    nose_xy = kps[KP['nose']].astype(int)
    cv2.putText(frame, f"P{person_id}",
                (nose_xy[0], nose_xy[1] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    for angle_key, kp_name in _ANGLE_LABEL_KP.items():
        val = angles.get(angle_key)
        if val is None:
            continue
        kp_idx = KP[kp_name]
        if scores[kp_idx] < SCORE_THR:
            continue
        x, y = kps[kp_idx].astype(int)
        cv2.putText(frame, f"{val:.0f}\u00b0",
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(source: str, mode: str, save_path: str | None, json_path: str) -> None:
    # Two-stage RTMDet + RTMPose at full resolution — same as rtmpose_live.py
    # default for video.  Gives the most stable, accurate detections.
    body = Body(
        mode=mode,
        to_openpose=False,
        backend='onnxruntime',
        device='cuda',
    )
    print(f"[tp1_cds] Model: two-stage RTMDet + RTMPose ({mode}) – full resolution")

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
        print(f"[tp1_cds] Annotated video → {save_path}")

    print(f"[tp1_cds] Source  : {source}  ({width}x{height} @ {fps:.1f} fps, {total} frames)")
    print(f"[tp1_cds] JSON    : {json_path}")
    print("[tp1_cds] Press 'q' or Esc to stop early.")

    tracker = SmoothTracker(max_dist=120.0)
    tracks: dict[str, list] = {}          # str(person_id) → list of frame records

    cv2.namedWindow("tp1_cds", cv2.WINDOW_NORMAL)
    frame_idx  = 0
    prev_time  = time.time()
    fps_smooth = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── Full-resolution inference (two-stage model crops person internally) ─
        keypoints, scores = body(frame)

        if keypoints is None or len(keypoints) == 0:
            keypoints = np.empty((0, 17, 2), dtype=float)
            scores    = np.empty((0, 17),    dtype=float)

        # ── draw_skeleton uses the raw detections (not smoothed) ──────────────
        annotated = draw_skeleton(frame, keypoints, scores,
                                  openpose_skeleton=False, kpt_thr=SCORE_THR)

        # ── Track + smooth + record ────────────────────────────────────────────
        track_results = tracker.update(keypoints, scores)
        for pid, smooth_kps, sc, smooth_angles in track_results:
            key = str(pid)
            if key not in tracks:
                tracks[key] = []
            tracks[key].append({
                "frame":       frame_idx,
                "timestamp_s": round(frame_idx / fps, 4),
                "angles":      {k: v for k, v in smooth_angles.items()},
            })
            overlay_angles(annotated, smooth_kps, sc, smooth_angles, pid)

        # ── HUD ───────────────────────────────────────────────────────────────
        now = time.time()
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(now - prev_time, 1e-9))
        prev_time  = now
        frame_idx += 1

        progress = f"{frame_idx}/{total}" if total > 0 else str(frame_idx)
        cv2.putText(annotated,
                    f"FPS:{fps_smooth:.1f}  frame {progress}  tracks:{len(tracker.tracks)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        if writer:
            writer.write(annotated)

        cv2.imshow("tp1_cds", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            print("[tp1_cds] Interrupted by user.")
            break

    # ── Teardown ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Write JSON ─────────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "source":        source,
            "fps":           fps,
            "total_frames":  total,
            "frames_processed": frame_idx,
            "width":         width,
            "height":        height,
            "mode":          mode,
            "angle_joints":  [j[0] for j in JOINT_ANGLES],
            "note": (
                "Angles are in degrees. "
                "None means the joint was not visible / below confidence threshold."
            ),
        },
        "tracks": tracks,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    total_records = sum(len(v) for v in tracks.values())
    print(f"[tp1_cds] Done. {frame_idx} frames, {len(tracks)} track(s), "
          f"{total_records} frame-records saved → {json_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="tp1_cds: RTMPose tracking + joint-angle collection"
    )
    parser.add_argument("source",
                        help="Path to input video file")
    parser.add_argument("--mode", default=MODE,
                        choices=["lightweight", "balanced", "performance"],
                        help=f"RTMO model size (default: {MODE})")
    parser.add_argument("--save", default=None,
                        help="Save annotated output to this .mp4 path")
    args = parser.parse_args()

    stem      = os.path.splitext(os.path.basename(args.source))[0]
    json_path = f"tp1_{stem}_data.json"

    run(args.source, args.mode, args.save, json_path)

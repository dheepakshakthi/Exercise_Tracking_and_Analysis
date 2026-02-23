# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rtmlib",
#   "onnxruntime",
#   "opencv-python",
# ]
# ///

import cv2
from rtmlib import Body, draw_skeleton

VIDEO_PATH = 'c1.mp4'
OUTPUT_PATH = 'c1_rtmpose_output.mp4'

# Initialize RTMPose Body tracker
# mode: 'lightweight' (rtmpose-s), 'balanced' (rtmpose-m), 'performance' (rtmpose-x)
body = Body(
    mode='balanced',
    to_openpose=False,
    backend='onnxruntime',
    device='cuda',
)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"Processing {total} frames from '{VIDEO_PATH}' ...")
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run RTMPose inference
    keypoints, scores = body(frame)

    # Draw skeleton on frame
    annotated = draw_skeleton(frame, keypoints, scores, openpose_skeleton=False, kpt_thr=0.3)

    out.write(annotated)

    # Show live preview (press 'q' to quit early)
    cv2.imshow('RTMPose Body Tracking', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Interrupted by user.")
        break

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"  {frame_idx}/{total} frames processed")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nDone! Output saved to '{OUTPUT_PATH}'")

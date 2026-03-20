# Real-Time Exercise Classification

This script provides real-time feedback on exercise form using machine learning models trained on joint angle data.

## Overview

The `realtime_classifier.py` script uses your webcam or a video file to:
1. Detect your body pose in real-time using RTMPose
2. Calculate joint angles (shoulders, elbows, hips, knees)
3. Classify whether you're performing the exercise correctly using trained ML models
4. Display instant visual feedback

## Features

- **Real-time pose detection** using state-of-the-art RTMPose model
- **Exercise classification** using Random Forest or XGBoost models
- **Smooth predictions** with temporal smoothing to reduce jitter
- **Visual feedback** with clear CORRECT/INCORRECT indicators
- **Live statistics** showing accuracy over time
- **Joint angle overlays** displaying angles at each joint
- **FPS counter** for performance monitoring

## Requirements

### Python Packages
```bash
# Core dependencies
pip install opencv-python numpy pandas scikit-learn xgboost joblib

# Pose detection (RTMPose)
pip install rtmlib
```

### Trained Models
You must first train the models using your exercise data:
```bash
python train_models.py
```

This will create:
- `xgboost_model.pkl` - XGBoost classifier (default, recommended)
- `random_forest_model.pkl` - Random Forest classifier
- `feature_info.json` - Feature metadata

## Usage

### Basic Usage (Webcam)
```bash
# Use XGBoost model with webcam
python realtime_classifier.py

# Use Random Forest model with webcam
python realtime_classifier.py --model random_forest
```

### Video File Input
```bash
# Process a video file
python realtime_classifier.py --source path/to/video.mp4

# With specific model
python realtime_classifier.py --source video.mp4 --model xgboost
```

### Advanced Options
```bash
# Specify RTMPose mode (lightweight/balanced/performance)
python realtime_classifier.py --mode performance

# Full example
python realtime_classifier.py --source 0 --model xgboost --mode balanced
```

## How It Works

### 1. Pose Detection
- Uses RTMPose (RTMDet + RTMPose) for accurate human pose estimation
- Detects 17 COCO keypoints per person
- Applies confidence thresholding (default: 0.3)

### 2. Joint Angle Calculation
Computes 8 joint angles:
- **Lower body:** left/right knee angles, left/right hip angles
- **Upper body:** left/right elbow angles, left/right shoulder angles

### 3. Temporal Smoothing
- **Keypoint smoothing:** Exponential Moving Average (EMA) on keypoint positions
- **Angle smoothing:** EMA on computed angles
- **Prediction smoothing:** Averages angles over a 10-frame window

### 4. Classification
- Extracts smoothed angle features
- Feeds features to trained ML model
- Returns prediction: CORRECT (1) or INCORRECT (0)
- Displays confidence score

## Controls

| Key | Action |
|-----|--------|
| `q` or `Esc` | Quit the application |
| `r` | Reset statistics counter |

## Display Elements

### Main Window
```
┌─────────────────────────────────────┐
│ FPS: 30.5                           │
│ Model: XGBoost                      │
│                                     │
│         CORRECT ✓                   │
│      Confidence: 95.2%              │
│                                     │
│  [Skeleton with angle overlays]    │
│                                     │
│ Statistics:                         │
│   Total: 1234                       │
│   Correct: 1180 (95.6%)            │
│   Incorrect: 54 (4.4%)             │
│                                     │
│ Press 'q' to quit, 'r' to reset    │
└─────────────────────────────────────┘
```

### Color Coding
- **Green (CORRECT ✓):** Exercise form is correct
- **Red (INCORRECT ✗):** Exercise form needs correction
- **Yellow (Analyzing...):** Gathering data or low confidence

### Angle Overlays
- Yellow text next to each joint showing current angle in degrees
- Example: "165°" at knee joint

## Configuration

### Adjustable Parameters (in script)

```python
# Classification sensitivity
WINDOW_SIZE = 10              # Frames to average (higher = smoother)
CONFIDENCE_THRESHOLD = 0.6    # Min confidence to show prediction

# Pose detection
SCORE_THR = 0.3               # Keypoint confidence threshold

# Smoothing
KP_ALPHA = 0.5                # Keypoint EMA (0-1, higher = less smooth)
ANG_ALPHA = 0.5               # Angle EMA (0-1, higher = less smooth)
```

## Troubleshooting

### "Model file not found"
**Solution:** Run `train_models.py` first to generate the model files.

### Low FPS / Laggy performance
**Solutions:**
- Use `--mode lightweight` for faster processing
- Ensure CUDA is available for GPU acceleration
- Close other resource-intensive applications
- Reduce video resolution

### "No person detected"
**Solutions:**
- Ensure good lighting
- Stand within camera view
- Move further from camera to fit full body
- Check if camera is working

### Jittery predictions
**Solutions:**
- Increase `WINDOW_SIZE` (e.g., 15 or 20)
- Increase `ANG_ALPHA` and `KP_ALPHA` (e.g., 0.7)

### Incorrect classifications
**Solutions:**
- Retrain models with more diverse data
- Ensure training data quality is good
- Check if you're performing the same exercise as trained on
- Verify camera angle matches training data

## Model Comparison

Based on training results:

| Model | Accuracy | Speed | Memory |
|-------|----------|-------|--------|
| **XGBoost** | 96.6% | Fast | Low |
| **Random Forest** | 95.8% | Fast | Medium |

**Recommendation:** Use XGBoost (default) for slightly better accuracy.

## Performance Tips

1. **GPU Acceleration:** Ensure CUDA is properly configured for RTMPose
2. **Lighting:** Use well-lit environment for better pose detection
3. **Camera Position:** Position camera to capture full body
4. **Background:** Plain background improves detection accuracy
5. **Movement:** Perform exercises at moderate speed for better tracking

## Data Flow

```
Camera/Video
    ↓
RTMPose Detection
    ↓
Keypoint Extraction
    ↓
Joint Angle Calculation
    ↓
Temporal Smoothing
    ↓
Feature Preparation
    ↓
ML Model Classification
    ↓
Visual Feedback Display
```

## Output Statistics

The script tracks:
- **Total frames processed**
- **Correct form count & percentage**
- **Incorrect form count & percentage**
- **Real-time FPS**

Statistics are displayed both during runtime and as a final summary when you quit.

## Future Enhancements

Potential improvements:
- Multi-person tracking and classification
- Save session recordings with classifications
- Audio feedback (e.g., beep on incorrect form)
- Rep counting functionality
- Exercise type detection (classify which exercise)
- Form correction suggestions (which angle to adjust)

## Credits

- **Pose Detection:** RTMPose/RTMLib
- **ML Models:** Scikit-learn, XGBoost
- **Computer Vision:** OpenCV

## License

See main project README for license information.
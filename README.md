# Physiotherapy Exercise Classifier

A comprehensive machine learning system for tracking, analyzing, and classifying physiotherapy exercises in real-time using pose estimation and ML models.

## Project Overview

This project implements an intelligent physiotherapy exercise monitoring system that uses computer vision and machine learning to:
- Track body poses during exercises using RTMPose
- Extract joint angles and movement patterns
- Train machine learning models to classify correct vs. incorrect exercise execution
- Provide real-time feedback on exercise correctness during live video or webcam input

## System Architecture

The project follows a three-stage pipeline:

```
Data Collection → Model Training → Real-time Classification
      ↓                ↓                    ↓
   tp1_cds.py    train_models.py    realtime_classifier.py
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone or download the project:
```bash
cd PhysiotherapyProject
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Overview

### Stage 1: Data Collection & Pose Tracking (`tp1_cds.py`)

This script captures video of exercises and extracts joint angle data using RTMPose.

**Purpose:**
- Record pose tracking data from exercise videos
- Calculate joint angles at 8 key joints (knees, hips, elbows, shoulders)
- Generate JSON files containing timestamped angle measurements for each frame

**Output:**
- JSON files stored in `tracked_data/<exercise_name>/` directory
- Each file contains frame-by-frame angle measurements for the exercise

**Usage:**
```bash
# Find first video in data/bicep_curl/ directory
python tp1_cds.py --exercise bicep_curl

# Use specific video file
python tp1_cds.py --exercise squats --video video.mp4

# Specify RTMPose mode (balanced/lite/heavy)
python tp1_cds.py --exercise bicep_curl --mode balanced

# Save annotated video output
python tp1_cds.py --exercise bicep_curl --save-video
```

**Key Features:**
- Automatic pose detection and tracking
- Joint angle computation (8 angles per person)
- EMA smoothing for stable angle values
- Confidence scoring for keypoint detection
- Optional video annotation showing skeleton and angles

---

### Stage 2: Model Training (`train_models.py`)

This script trains machine learning models to classify exercise correctness.

**Purpose:**
- Load collected exercise data from JSON files
- Extract features from angle measurements
- Train two classification models:
  - Random Forest Classifier
  - XGBoost Classifier
- Evaluate model performance and save trained models

**Output:**
- Trained model files (.pkl) in `tracking_models/<exercise_name>/`
- Feature importance rankings
- Training metrics and confusion matrices
- Feature info JSON with model metadata

**Usage:**
```bash
python train_models.py
```

**Data Organization:**
The script automatically discovers exercises in `tracked_data/` with the following structure:
```
tracked_data/
├── bicep_curl/
│   ├── video1_correct.json
│   ├── video2_incorrect.json
│   └── ...
└── squats/
    ├── squat1_c1.json
    ├── squat2_incorrect.json
    └── ...
```

**Naming Convention:**
- Files containing "correct" or "c1" → labeled as 1 (correct exercise)
- All other files → labeled as 0 (incorrect exercise)

**Training Process:**
1. Discover all exercise folders in tracked_data
2. Extract angle features from JSON files
3. Handle missing values via median imputation
4. Split data into 80% training / 20% testing sets
5. Train Random Forest and XGBoost models
6. Generate performance reports with accuracy and confusion matrices
7. Save models for real-time classification

---

### Stage 3: Real-time Classification (`realtime_classifier.py`)

This script provides live exercise classification and feedback.

**Purpose:**
- Load trained ML models for specific exercises
- Capture video from webcam or video file
- Extract poses and angles in real-time
- Classify exercise correctness frame-by-frame
- Display visual feedback and statistics

**Output:**
- Real-time video overlay with:
  - Skeleton visualization
  - Joint angles
  - Classification results (CORRECT/INCORRECT)
  - Confidence scores
  - Running statistics

**Usage:**
```bash
# Use webcam input
python realtime_classifier.py --exercise bicep_curl

# Use video file
python realtime_classifier.py --exercise squats --source video.mp4

# Specify model (random_forest or xgboost)
python realtime_classifier.py --exercise bicep_curl --model xgboost

# List available exercises
python realtime_classifier.py --list-exercises
```

**Controls:**
- `q` or `Esc` - Quit application
- `r` - Reset statistics counter

**Features:**
- Real-time pose detection and tracking
- Windowed classification (averages predictions over frames)
- Confidence thresholding for reliable predictions
- Live statistics tracking:
  - Total frames analyzed
  - Correct/Incorrect counts
  - Accuracy percentage

---

## Project Structure

```
PhysiotherapyProject/
├── tp1_cds.py                      # Stage 1: Data collection & tracking
├── train_models.py                 # Stage 2: Model training
├── realtime_classifier.py          # Stage 3: Real-time classification
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/                           # Input video files
│   ├── bicep_curl/
│   ├── squats/
│   └── ...
│
├── tracked_data/                   # Output from Stage 1
│   ├── bicep_curl/
│   │   ├── video1_tracked.json
│   │   └── ...
│   └── squats/
│       └── ...
│
└── tracking_models/                # Output from Stage 2
    ├── bicep_curl/
    │   ├── random_forest_model.pkl
    │   ├── xgboost_model.pkl
    │   └── feature_info.json
    └── squats/
        └── ...
```

## Features Extracted

The system tracks 8 joint angles per person:

1. **Left Knee Angle** - Angle at left knee joint
2. **Right Knee Angle** - Angle at right knee joint
3. **Left Hip Angle** - Angle at left hip joint
4. **Right Hip Angle** - Angle at right hip joint
5. **Left Elbow Angle** - Angle at left elbow joint
6. **Right Elbow Angle** - Angle at right elbow joint
7. **Left Shoulder Angle** - Angle at left shoulder joint
8. **Right Shoulder Angle** - Angle at right shoulder joint

These angles are computed using COCO-17 pose keypoints from RTMPose.

## Configuration Parameters

### RTMPose Settings
- **MODE**: Pose detection model mode (`balanced`, `lite`, or `heavy`)
- **SCORE_THR**: Keypoint confidence threshold (default: 0.3)

### Smoothing Parameters
- **KP_ALPHA**: EMA weight for keypoint smoothing (0-1, higher = smoother)
- **ANG_ALPHA**: EMA weight for angle smoothing (0-1, higher = smoother)

### Classification Settings
- **WINDOW_SIZE**: Number of frames to average for predictions (default: 10)
- **CONFIDENCE_THRESHOLD**: Minimum confidence for displaying predictions (default: 0.6)

## Model Performance

The system trains two complementary models:

### Random Forest Classifier
- Interpretable decision trees
- Feature importance analysis
- Good for capturing non-linear patterns
- Robust to outliers

### XGBoost Classifier
- Gradient boosting ensemble
- Often achieves higher accuracy
- Fast inference
- Excellent generalization

Both models are trained and evaluated, with detailed metrics provided.

## Dependencies

Core dependencies managed via `requirements.txt`:
- **opencv-python**: Video processing and display
- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **scikit-learn**: ML utilities and Random Forest
- **xgboost**: XGBoost classifier
- **joblib**: Model serialization
- **rtmlib**: RTMPose pose estimation

## Workflow Example

### Complete Workflow

1. **Collect Training Data:**
   ```bash
   # Record videos in data/bicep_curl/
   python tp1_cds.py --exercise bicep_curl --save-video
   ```
   → Creates `tracked_data/bicep_curl/*.json`

2. **Train Models:**
   ```bash
   python train_models.py
   ```
   → Creates `tracking_models/bicep_curl/{random_forest,xgboost}_model.pkl`

3. **Test in Real-time:**
   ```bash
   python realtime_classifier.py --exercise bicep_curl
   ```
   → Shows live classification results with feedback

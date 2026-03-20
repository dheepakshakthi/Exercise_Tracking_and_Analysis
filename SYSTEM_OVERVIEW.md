# Physiotherapy Exercise Classification System - Overview

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Components](#components)
3. [Data Pipeline](#data-pipeline)
4. [Technologies](#technologies)
5. [File Structure](#file-structure)
6. [Workflow](#workflow)
7. [Performance Metrics](#performance-metrics)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Exercise Videos    →    Pose Tracking    →    Angle Data     │
│  (c1, w1, w2)           (tp1_cds.py)         (JSON files)      │
│                                                                 │
│       ↓                                                         │
│                                                                 │
│  Feature Extraction  →  Model Training  →  Saved Models        │
│  (train_models.py)      (RF + XGBoost)     (.pkl files)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   REAL-TIME PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Live Camera/Video  →  Pose Detection  →  Angle Calculation   │
│                        (RTMPose)            (realtime)          │
│                                                                 │
│       ↓                                                         │
│                                                                 │
│  Temporal Smoothing  →  Classification  →  Visual Feedback     │
│  (EMA filtering)        (ML models)         (on screen)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Data Collection (`tp1_cds.py`)

**Purpose:** Track body pose in videos and collect joint angle data

**Key Features:**
- Two-stage pose detection (RTMDet + RTMPose)
- 17 COCO keypoint detection
- 8 joint angle calculation (knees, hips, elbows, shoulders)
- Centroid-based person tracking
- Exponential Moving Average (EMA) smoothing
- JSON output format

**Input:** Exercise videos (MP4, AVI, etc.)

**Output:** JSON files with frame-by-frame angle data
- `tp1_c1_data.json` - Correct exercise form
- `tp1_w1_data.json` - Incorrect form (variant 1)
- `tp1_w2_data.json` - Incorrect form (variant 2)

**Usage:**
```bash
python tp1_cds.py video.mp4 --mode balanced --save output.mp4
```

### 2. Model Training (`train_models.py`)

**Purpose:** Train machine learning models to classify exercise form

**Process:**
1. **Load Data:** Read JSON files from all exercise variants
2. **Feature Extraction:** Extract 8 joint angles per frame
3. **Preprocessing:** Handle missing values, normalize data
4. **Train-Test Split:** 80/20 split with stratification
5. **Model Training:** Train Random Forest and XGBoost classifiers
6. **Evaluation:** Generate performance metrics
7. **Save Models:** Export trained models to disk

**Models Trained:**
- **Random Forest:** 100 trees, max depth 10
  - Accuracy: 95.8%
  - Precision (Incorrect): 99%
  - Recall (Correct): 98%
  
- **XGBoost:** 100 estimators, learning rate 0.1
  - Accuracy: 96.6%
  - Precision (Incorrect): 97%
  - Recall (Correct): 97%

**Feature Importance:**
1. Left shoulder angle (highest)
2. Right shoulder angle
3. Right knee angle
4. Right/Left hip angles
5. Left knee angle
6. Elbow angles (lower importance)

**Output Files:**
- `random_forest_model.pkl`
- `xgboost_model.pkl`
- `feature_info.json`

### 3. Real-Time Classification (`realtime_classifier.py`)

**Purpose:** Provide live exercise form feedback

**Key Components:**

#### A. Pose Detection
- RTMPose model for keypoint detection
- Confidence threshold filtering
- Multi-person detection (uses first person)

#### B. Angle Calculation
- Same 8 angles as training data
- Real-time computation from keypoints

#### C. Temporal Smoothing
- **Keypoint EMA:** Smooth jittery pose detections
- **Angle EMA:** Smooth calculated angles
- **Window Averaging:** Average over 10 frames for stability

#### D. Classification
- Feature vector preparation
- ML model inference
- Confidence scoring
- Prediction smoothing

#### E. Visual Feedback
- Skeleton overlay with pose
- Joint angle annotations
- Classification result (CORRECT/INCORRECT)
- Confidence percentage
- Real-time statistics
- FPS counter

**Usage:**
```bash
# Webcam
python realtime_classifier.py

# Video file
python realtime_classifier.py --source video.mp4

# Specific model
python realtime_classifier.py --model random_forest
```

---

## Data Pipeline

### Training Data Flow

```
Video Frame
    ↓
RTMDet (person detection)
    ↓
RTMPose (keypoint estimation)
    ↓
17 COCO Keypoints + Confidence Scores
    ↓
Geometric Angle Calculation
    ↓
8 Joint Angles (degrees)
    ↓
EMA Smoothing (temporal)
    ↓
JSON Record {frame, timestamp, angles}
    ↓
Accumulated Dataset
```

### Classification Data Flow

```
Live Frame
    ↓
RTMPose Detection
    ↓
Keypoints → Angles
    ↓
EMA Smoothing
    ↓
10-Frame Buffer
    ↓
Average Angles
    ↓
Feature Vector [8 angles]
    ↓
ML Model
    ↓
Prediction + Confidence
    ↓
Visual Display
```

---

## Technologies

### Core Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| **RTMLib** | Pose estimation | Latest |
| **OpenCV** | Video processing & display | 4.x |
| **NumPy** | Numerical computations | Latest |
| **Pandas** | Data manipulation | Latest |
| **Scikit-learn** | Random Forest model | 1.x |
| **XGBoost** | Gradient boosting | Latest |
| **Joblib** | Model serialization | Latest |

### Model Architecture

#### RTMPose
- **Detector:** RTMDet (two-stage)
- **Pose:** RTMPose
- **Keypoints:** 17 COCO format
- **Backend:** ONNX Runtime
- **Device:** CUDA (GPU) / CPU

#### Random Forest
- **Type:** Ensemble classifier
- **Trees:** 100
- **Max Depth:** 10
- **Features:** 8 joint angles

#### XGBoost
- **Type:** Gradient boosting
- **Estimators:** 100
- **Max Depth:** 6
- **Learning Rate:** 0.1

---

## File Structure

```
PhysiotherapyProject/
│
├── Data Collection
│   ├── tp1_cds.py              # Pose tracking & angle collection
│   ├── tp1_c1_data.json        # Correct exercise data (981 frames)
│   ├── tp1_w1_data.json        # Incorrect data #1 (612 frames)
│   └── tp1_w2_data.json        # Incorrect data #2 (445 frames)
│
├── Model Training
│   ├── train_models.py         # Training script
│   ├── train_info_1.txt        # Training results/logs
│   ├── random_forest_model.pkl # Trained RF model
│   ├── xgboost_model.pkl       # Trained XGB model
│   └── feature_info.json       # Feature metadata
│
├── Real-Time Classification
│   └── realtime_classifier.py  # Live classification app
│
├── Documentation
│   ├── SYSTEM_OVERVIEW.md      # This file
│   ├── README_REALTIME.md      # Detailed real-time docs
│   └── QUICKSTART.md           # Quick start guide
│
└── Supporting Files
    ├── data/                   # Exercise videos
    ├── models/                 # ONNX models
    └── outputs/                # Generated outputs
```

---

## Workflow

### Phase 1: Data Collection

1. **Record exercise videos**
   - Correct form (c1)
   - Incorrect variants (w1, w2)

2. **Process videos**
   ```bash
   python tp1_cds.py data/c1.mp4
   python tp1_cds.py data/w1.mp4
   python tp1_cds.py data/w2.mp4
   ```

3. **Verify JSON outputs**
   - Check frame counts
   - Inspect angle values
   - Ensure data quality

### Phase 2: Model Training

1. **Train models**
   ```bash
   python train_models.py
   ```

2. **Review results**
   - Check accuracy metrics
   - Examine confusion matrix
   - Analyze feature importance

3. **Select best model**
   - XGBoost: 96.6% (recommended)
   - Random Forest: 95.8%

### Phase 3: Real-Time Deployment

1. **Test with webcam**
   ```bash
   python realtime_classifier.py
   ```

2. **Validate with test videos**
   ```bash
   python realtime_classifier.py --source test_video.mp4
   ```

3. **Deploy for use**
   - Live feedback during exercise
   - Track performance statistics
   - Adjust form based on feedback

---

## Performance Metrics

### Training Dataset
- **Total Frames:** 2,038
- **Correct:** 981 (48.1%)
- **Incorrect:** 1,057 (51.9%)
- **Train/Test Split:** 1,630 / 408

### Model Performance

#### XGBoost (Recommended)
```
Accuracy: 96.57%

Classification Report:
              precision    recall  f1-score   support
   Incorrect       0.97      0.96      0.97       212
     Correct       0.96      0.97      0.96       196

Confusion Matrix:
[[204   8]
 [  6 190]]
```

#### Random Forest
```
Accuracy: 95.83%

Classification Report:
              precision    recall  f1-score   support
   Incorrect       0.99      0.93      0.96       212
     Correct       0.93      0.98      0.96       196

Confusion Matrix:
[[198  14]
 [  3 193]]
```

### Real-Time Performance
- **FPS:** 25-30 (webcam, balanced mode)
- **Latency:** ~100-150ms
- **Smoothing Window:** 10 frames (~0.33s at 30fps)
- **Classification Confidence:** 85-99% typical

---

## Key Design Decisions

### 1. Two-Stage Detection
**Why:** More accurate than single-stage; crops improve pose quality

### 2. EMA Smoothing
**Why:** Reduces jitter without significant lag; preserves real-time feel

### 3. Window Averaging
**Why:** Stabilizes predictions; prevents flickering between classes

### 4. 8 Joint Angles
**Why:** Captures essential body mechanics; computationally efficient

### 5. Binary Classification
**Why:** Clear feedback (correct/incorrect); easy to interpret

### 6. XGBoost as Default
**Why:** Best accuracy (96.6%); fast inference; robust

---

## Limitations & Considerations

### Current Limitations
1. **Single exercise type:** Trained for one specific exercise
2. **Single person tracking:** Uses only first detected person
3. **2D analysis:** No depth information (monocular camera)
4. **Camera angle dependency:** Performance varies with viewpoint
5. **Lighting sensitivity:** Poor lighting affects pose detection

### Best Practices
1. **Training data quality:** Ensure diverse, high-quality samples
2. **Balanced dataset:** Equal correct/incorrect examples
3. **Camera positioning:** Consistent angle and distance
4. **Lighting conditions:** Well-lit, even lighting
5. **Movement speed:** Moderate pace for better tracking

---

## Future Enhancements

### Short-term
- [ ] Multi-person tracking
- [ ] Exercise rep counting
- [ ] Audio feedback
- [ ] Session recording & playback
- [ ] Progress tracking over time

### Medium-term
- [ ] Multi-exercise classification
- [ ] Form correction suggestions
- [ ] 3D pose estimation
- [ ] Mobile app deployment
- [ ] Cloud-based processing

### Long-term
- [ ] Personalized recommendations
- [ ] Injury risk prediction
- [ ] Virtual physiotherapist
- [ ] Integration with fitness apps
- [ ] Research publication

---

## Conclusion

This system demonstrates a complete end-to-end pipeline for exercise form classification:

✅ **Data Collection:** Automated angle tracking from videos
✅ **Model Training:** High-accuracy ML classifiers (96.6%)
✅ **Real-Time Feedback:** Live exercise form validation
✅ **User-Friendly:** Simple interface, clear feedback
✅ **Extensible:** Easy to add new exercises or features

**Impact:** Enables automated, objective exercise form assessment for physiotherapy and fitness applications.

---

**Version:** 1.0  
**Last Updated:** 2024  
**Author:** Dheepak  
**Status:** Production Ready
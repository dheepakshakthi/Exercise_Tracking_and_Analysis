# Physiotherapy Exercise Classification System

A real-time machine learning system that monitors and provides instant feedback on exercise form using computer vision and pose estimation. Perfect for physiotherapy, fitness training, and rehabilitation applications.

## 🎯 Overview

This project uses deep learning for human pose estimation combined with classical machine learning models to classify whether an exercise is being performed with correct form or not. It provides real-time visual feedback, tracks statistics, and helps users maintain proper exercise technique.

**Key Statistics:**
- 🎯 **Model Accuracy:** 96.6% (XGBoost)
- 📹 **Real-time FPS:** 25-30 fps
- 📊 **Training Data:** 2,038 frames from 3 exercise variants
- 🏋️ **Features Tracked:** 8 joint angles (knees, hips, elbows, shoulders)

## ✨ Features

- **Real-time Pose Detection** - Uses RTMPose for accurate body keypoint detection
- **Exercise Classification** - ML models classify form as CORRECT or INCORRECT
- **Live Visual Feedback** - Skeleton overlay with angle annotations
- **Performance Statistics** - Track accuracy percentage during exercise sessions
- **Multi-model Support** - Choose between XGBoost or Random Forest models
- **Flexible Input** - Works with webcam or video files
- **Temporal Smoothing** - Reduces jitter and provides stable predictions
- **Confidence Scoring** - Displays confidence levels for predictions

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│ Exercise Videos → Pose Tracking → Joint Angles → ML Models     │
│ (c1, w1, w2)     (RTMPose)       (8 angles)    (RF/XGBoost)    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   REAL-TIME PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│ Live Camera → Pose Detection → Angle Calculation → Classification
│              (RTMPose)         (Real-time)        (ML Model)     │
│                                        ↓                          │
│                                 Visual Feedback                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- CUDA 11.0+ (GPU recommended for real-time performance)
- Webcam or video file
- 4GB RAM minimum

### Python Dependencies
```bash
opencv-python>=4.5.0
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
xgboost>=1.4.0
joblib>=1.0.0
rtmlib>=0.1.0
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd PhysiotherapyProject

# Install dependencies
pip install opencv-python numpy pandas scikit-learn xgboost joblib rtmlib

# Verify models exist
ls random_forest_model.pkl xgboost_model.pkl
```

### 2. Run Real-Time Classifier

**Using Webcam (Default):**
```bash
python realtime_classifier.py
```

**Using Video File:**
```bash
python realtime_classifier.py --source path/to/video.mp4
```

**Using Random Forest Model:**
```bash
python realtime_classifier.py --model random_forest
```

### 3. Controls

| Key | Action |
|-----|--------|
| `q` or `Esc` | Quit application |
| `r` | Reset statistics counter |

## 📊 Performance Metrics

### Model Comparison

| Metric | XGBoost | Random Forest |
|--------|---------|---------------|
| **Accuracy** | 96.6% | 95.8% |
| **Precision (Incorrect)** | 97% | 99% |
| **Recall (Correct)** | 97% | 98% |
| **Inference Speed** | Fast | Fast |
| **Model Size** | Small | Medium |

**Recommendation:** Use XGBoost (default) for best overall performance.

### Training Dataset

- **Total Frames:** 2,038
- **Correct Exercise:** 981 frames (48.1%)
- **Incorrect Variations:** 1,057 frames (51.9%)
  - Variant 1 (w1): 612 frames
  - Variant 2 (w2): 445 frames
- **Train/Test Split:** 80/20 (1,630 training, 408 testing)

## 📁 Project Structure

```
PhysiotherapyProject/
│
├── 📄 README.md                          # This file
├── 📄 QUICKSTART.md                      # 5-minute quick start guide
├── 📄 README_REALTIME.md                 # Detailed real-time documentation
├── 📄 SYSTEM_OVERVIEW.md                 # Technical architecture details
│
├── 🔧 Core Scripts
│   ├── realtime_classifier.py            # Real-time exercise classification
│   ├── tp1_cds.py                        # Pose tracking & data collection
│   └── train_models.py                   # Model training script
│
├── 📦 Trained Models
│   ├── xgboost_model.pkl                 # XGBoost classifier (96.6% accuracy)
│   ├── random_forest_model.pkl           # Random Forest classifier (95.8%)
│   └── feature_info.json                 # Feature metadata
│
├── 📊 Training Data
│   ├── tp1_c1_data.json                  # Correct exercise data (981 frames)
│   ├── tp1_w1_data.json                  # Incorrect variant 1 (612 frames)
│   └── tp1_w2_data.json                  # Incorrect variant 2 (445 frames)
│
└── 📁 Supporting Directories
    ├── data/                             # Exercise video files
    ├── models/                           # ONNX pose estimation models
    └── outputs/                          # Generated outputs and logs
```

## 🔄 Complete Workflow

### Step 1: Data Collection

Collect video recordings of the exercise in correct and incorrect forms:

```bash
# For correct form
python tp1_cds.py data/correct_exercise.mp4 --mode balanced

# For incorrect variants
python tp1_cds.py data/incorrect_variant1.mp4 --mode balanced
python tp1_cds.py data/incorrect_variant2.mp4 --mode balanced
```

This generates JSON files with frame-by-frame joint angles.

### Step 2: Train Models

```bash
python train_models.py
```

This will:
1. Load the three JSON data files
2. Extract features (8 joint angles per frame)
3. Train Random Forest and XGBoost classifiers
4. Evaluate models and display performance metrics
5. Save trained models to disk

### Step 3: Real-Time Classification

```bash
# Use trained models with live webcam
python realtime_classifier.py

# Or process a video file
python realtime_classifier.py --source test_video.mp4
```

## 💡 How It Works

### 1. Pose Detection
- Uses RTMPose (two-stage RTMDet + RTMPose)
- Detects 17 COCO-format keypoints per person
- Applies confidence thresholding (default: 0.3)

### 2. Joint Angle Calculation
Calculates 8 key angles:
```
Lower Body:
- Left knee angle (left_hip → left_knee → left_ankle)
- Right knee angle (right_hip → right_knee → right_ankle)
- Left hip angle (left_shoulder → left_hip → left_knee)
- Right hip angle (right_shoulder → right_hip → right_knee)

Upper Body:
- Left elbow angle (left_shoulder → left_elbow → left_wrist)
- Right elbow angle (right_shoulder → right_elbow → right_wrist)
- Left shoulder angle (left_elbow → left_shoulder → left_hip)
- Right shoulder angle (right_elbow → right_shoulder → right_hip)
```

### 3. Temporal Smoothing
- **Keypoint Smoothing:** Exponential Moving Average (EMA) on keypoint positions
- **Angle Smoothing:** EMA on computed angles
- **Prediction Smoothing:** Averages angles over a 10-frame window
- Reduces jitter while maintaining real-time responsiveness

### 4. Classification
- Prepares feature vector from smoothed angles
- Feeds to trained ML model
- Returns prediction with confidence score
- Updates running statistics

### 5. Visual Feedback
- Displays skeleton with pose
- Shows joint angles as text overlays
- Large CORRECT (green) or INCORRECT (red) indicator
- Confidence percentage
- Real-time performance statistics

## 🎓 Feature Importance

Based on trained models, these features (joint angles) are most important for classification:

1. **Left Shoulder Angle** (26.7% importance)
2. **Right Shoulder Angle** (13.6%)
3. **Right Knee Angle** (12.6%)
4. **Right Hip Angle** (12.5%)
5. **Left Hip Angle** (10.8%)
6. **Left Knee Angle** (9.2%)
7. **Left Elbow Angle** (7.9%)
8. **Right Elbow Angle** (7.0%)

Shoulder angles are the most discriminative features for this exercise.

## ⚙️ Configuration

### Real-Time Classifier Settings

Edit `realtime_classifier.py` to adjust:

```python
# Classification window (higher = smoother predictions)
WINDOW_SIZE = 10  # frames to average

# Minimum confidence to display result
CONFIDENCE_THRESHOLD = 0.6

# Pose detection confidence threshold
SCORE_THR = 0.3

# Smoothing (0-1, higher = less smoothing)
KP_ALPHA = 0.5    # Keypoint smoothing
ANG_ALPHA = 0.5   # Angle smoothing
```

### RTMPose Modes

- **`lightweight`** - Fast, lower accuracy (15-30 ms/frame)
- **`balanced`** - Default, good balance (30-50 ms/frame)
- **`performance`** - Highest accuracy (50-100 ms/frame)

```bash
python realtime_classifier.py --mode lightweight
```

## 🐛 Troubleshooting

### "Model file not found"
**Solution:** Ensure trained models exist and train if needed:
```bash
python train_models.py
```

### "Cannot open video source" / Webcam not detected
**Solutions:**
- Check if webcam is connected
- Try different camera index: `--source 1` or `--source 2`
- Check permissions (grant camera access)
- Restart the application

### "No person detected"
**Solutions:**
- Ensure you're in the camera frame
- Improve lighting conditions
- Move 2-3 meters from camera
- Use plain background
- Check camera focus

### Low FPS / Laggy performance
**Solutions:**
```bash
# Use lightweight mode
python realtime_classifier.py --mode lightweight

# Reduce window size in code
WINDOW_SIZE = 5  # instead of 10

# Use GPU if available
```

### Jittery predictions
**Solutions:**
- Increase `WINDOW_SIZE` to 15 or 20
- Increase `ANG_ALPHA` to 0.7
- Reduce camera movement
- Improve lighting

### Incorrect classifications consistently
**Solutions:**
- Verify exercise matches training data
- Check camera angle and position
- Ensure proper lighting
- Retrain models with more diverse data
- Record new training videos

## 📚 Detailed Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[README_REALTIME.md](README_REALTIME.md)** - Complete real-time documentation
- **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** - Technical architecture details

## 🔬 Technologies Used

### Computer Vision
- **RTMPose** - State-of-the-art pose estimation
- **OpenCV** - Video processing and visualization

### Machine Learning
- **Scikit-learn** - Random Forest implementation
- **XGBoost** - Gradient boosting classifier
- **Joblib** - Model persistence

### Data Processing
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis

## 🎯 Use Cases

1. **Physiotherapy** - Monitor patient form during rehabilitation
2. **Personal Training** - Real-time exercise form feedback
3. **Fitness Apps** - Integration into workout applications
4. **Sports Training** - Form analysis for athletic training
5. **Home Workouts** - Self-guided exercise form verification

## 📈 Performance Optimization Tips

1. **GPU Acceleration** - Ensure CUDA properly configured
2. **Lighting** - Use well-lit environment (500+ lux)
3. **Camera Position** - Consistent angle and 2-3 meter distance
4. **Background** - Plain or non-distracting background
5. **Movement Speed** - Perform exercises at moderate speed
6. **Model Selection** - Use XGBoost for best accuracy
7. **Window Size** - Balance smoothness vs responsiveness


## 📊 Project Statistics

- **Lines of Code:** ~2000+
- **Models Trained:** 2 (Random Forest, XGBoost)
- **Training Samples:** 2,038 frames
- **Accuracy:** 96.6%
- **Real-time FPS:** 25-30
- **Supported Angles:** 8 joint angles
- **Keypoints Detected:** 17 COCO format

---

**Version:** 1.0  
**Last Updated:** 2024  
**Status:** Production Ready ✅

**Happy Exercising!** 💪

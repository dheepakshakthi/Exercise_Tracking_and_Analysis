# Quick Start Guide - Real-Time Exercise Classifier

Get started with real-time exercise form feedback in under 5 minutes!

## Prerequisites

Make sure you have Python 3.8+ installed and a webcam connected.

## Step 1: Install Dependencies

```bash
pip install opencv-python numpy pandas scikit-learn xgboost joblib rtmlib
```

## Step 2: Verify Models Exist

Check that these files are present in your project directory:
- ✅ `xgboost_model.pkl`
- ✅ `random_forest_model.pkl`

If missing, run:
```bash
python train_models.py
```

## Step 3: Run the Classifier

### Using Webcam (Default)
```bash
python realtime_classifier.py
```

### Using a Video File
```bash
python realtime_classifier.py --source your_video.mp4
```

## Controls

| Key | Action |
|-----|--------|
| **q** or **Esc** | Quit |
| **r** | Reset statistics |

## What You'll See

```
┌─────────────────────────────────┐
│ FPS: 28.3                       │
│ Model: XGBoost                  │
│                                 │
│      CORRECT ✓                  │
│   Confidence: 94.5%             │
│                                 │
│   [Your skeleton with angles]  │
│                                 │
│ Statistics:                     │
│   Total: 245                    │
│   Correct: 231 (94.3%)         │
│   Incorrect: 14 (5.7%)         │
└─────────────────────────────────┘
```

### Color Guide
- 🟢 **Green (CORRECT ✓)** - Good form!
- 🔴 **Red (INCORRECT ✗)** - Adjust your form
- 🟡 **Yellow (Analyzing...)** - Building confidence

## Quick Tips

1. **Position yourself** so your full body is visible
2. **Good lighting** helps detection accuracy
3. **Stand 2-3 meters** from the camera
4. **Plain background** works best
5. **Perform the exercise** at moderate speed

## Common Issues

### ❌ "Model file not found"
**Fix:** Run `python train_models.py` first

### ❌ "Cannot open video source"
**Fix:** 
- Check if your webcam is connected
- Try `--source 0` or `--source 1`
- Grant camera permissions

### ❌ "No person detected"
**Fix:**
- Ensure you're in frame
- Improve lighting
- Move further from camera

### ⚠️ Low FPS / Laggy
**Fix:**
```bash
# Use lightweight mode
python realtime_classifier.py --mode lightweight
```

### ⚠️ Jittery predictions
**Fix:** Edit `realtime_classifier.py` and increase:
```python
WINDOW_SIZE = 15  # Change from 10 to 15
```

## Command Options

```bash
# Use Random Forest instead of XGBoost
python realtime_classifier.py --model random_forest

# Use high-performance mode (requires GPU)
python realtime_classifier.py --mode performance

# Process a video file with RF model
python realtime_classifier.py --source video.mp4 --model random_forest
```

## Model Accuracy

From your training results:
- **XGBoost:** 96.6% accuracy (recommended)
- **Random Forest:** 95.8% accuracy

## Next Steps

Once comfortable with the basics:
1. Read `README_REALTIME.md` for detailed documentation
2. Adjust parameters in the script for your needs
3. Retrain models with more exercise data for better accuracy

## Need Help?

- Check `README_REALTIME.md` for detailed troubleshooting
- Review training results in `train_info_1.txt`
- Ensure your exercise matches the training data (tp1_c1_data.json was correct form)

---

**Ready?** Run `python realtime_classifier.py` and start exercising! 💪
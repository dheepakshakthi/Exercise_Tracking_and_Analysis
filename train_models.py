import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_json_data(filepath):
    """Load JSON file and return the data."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_features_from_json(json_data, label):
    """
    Extract features from JSON data.

    Args:
        json_data: Loaded JSON data
        label: 1 for correct exercise, 0 for incorrect

    Returns:
        DataFrame with features and labels
    """
    features_list = []

    # Get the track data (assuming track "0" contains the main data)
    tracks = json_data.get("tracks", {})
    track_data = tracks.get("0", [])

    # Define angle features
    angle_features = [
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_shoulder_angle",
        "right_shoulder_angle",
    ]

    for frame_data in track_data:
        angles = frame_data.get("angles", {})

        # Extract angle values, replace None with NaN
        feature_row = {}
        for angle_name in angle_features:
            feature_row[angle_name] = angles.get(angle_name, np.nan)

        feature_row["label"] = label
        feature_row["frame"] = frame_data.get("frame", 0)
        feature_row["timestamp_s"] = frame_data.get("timestamp_s", 0.0)

        features_list.append(feature_row)

    return pd.DataFrame(features_list)


def discover_exercise_folders(tracked_data_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover exercise folders and their JSON files.

    Args:
        tracked_data_dir: Path to the tracked_data directory

    Returns:
        Dictionary mapping exercise names to lists of JSON file paths
    """
    exercises = {}

    if not tracked_data_dir.exists():
        print(f"Warning: tracked_data directory not found at {tracked_data_dir}")
        return exercises

    for exercise_folder in tracked_data_dir.iterdir():
        if exercise_folder.is_dir():
            json_files = list(exercise_folder.glob("*.json"))
            if json_files:
                exercises[exercise_folder.name] = json_files
                print(
                    f"   Found exercise '{exercise_folder.name}' with {len(json_files)} files"
                )

    return exercises


def load_exercise_data(json_files: List[Path]) -> pd.DataFrame:
    """
    Load and extract features from multiple JSON files for an exercise.

    Args:
        json_files: List of paths to JSON files

    Returns:
        Combined DataFrame with all features
    """
    dataframes = []

    for idx, filepath in enumerate(json_files):
        try:
            json_data = load_json_data(filepath)

            # Infer label from filename (assuming naming convention: *_correct.json, *_incorrect.json, etc.)
            filename = filepath.stem.lower()
            if "correct" in filename or "c1" in filename:
                label = 1
            else:
                label = 0

            df = extract_features_from_json(json_data, label)
            if len(df) > 0:
                dataframes.append(df)
                print(
                    f"      Loaded {filepath.name}: {len(df)} frames (label: {label})"
                )
        except Exception as e:
            print(f"      Warning: Failed to load {filepath.name}: {e}")

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)


def train_models_for_exercise(
    exercise_name: str,
    df_all: pd.DataFrame,
    feature_cols: List[str],
    models_dir: Path,
) -> Dict:
    """
    Train models for a specific exercise and save them.

    Args:
        exercise_name: Name of the exercise
        df_all: DataFrame with all training data
        feature_cols: List of feature column names
        models_dir: Directory to save models

    Returns:
        Dictionary with training results
    """
    print(f"\n{'=' * 60}")
    print(f"Training Models for Exercise: {exercise_name}")
    print(f"{'=' * 60}")

    # Create exercise-specific directory
    exercise_dir = models_dir / exercise_name
    exercise_dir.mkdir(parents=True, exist_ok=True)

    # Preprocessing
    print("\n1. Preprocessing data...")
    print(f"   - Total frames: {len(df_all)}")
    print(f"   - Missing values before: {df_all[feature_cols].isna().sum().sum()}")

    # Handle missing values
    df_all[feature_cols] = df_all[feature_cols].fillna(df_all[feature_cols].median())

    print(f"   - Missing values after: {df_all[feature_cols].isna().sum().sum()}")

    # Prepare features and labels
    X = df_all[feature_cols].values
    y = df_all["label"].values

    # Check if we have enough samples
    if len(y) < 10:
        print(f"   ✗ Not enough samples for training ({len(y)} samples)")
        return {"error": "Insufficient data"}

    # Check class distribution
    correct_count = np.sum(y)
    incorrect_count = len(y) - correct_count

    print(f"   - Correct exercises: {correct_count}")
    print(f"   - Incorrect exercises: {incorrect_count}")
    print("   ✓ Preprocessing complete")

    # Split data
    print(f"\n2. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Testing samples: {len(X_test)}")
    print(
        f"   - Training correct/incorrect: {np.sum(y_train)}/{len(y_train) - np.sum(y_train)}"
    )
    print(
        f"   - Testing correct/incorrect: {np.sum(y_test)}/{len(y_test) - np.sum(y_test)}"
    )

    results = {
        "exercise": exercise_name,
        "total_samples": len(df_all),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    # Train Random Forest
    print(f"\n3. Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    print(f"   ✓ Random Forest Accuracy: {rf_accuracy:.4f}")
    print("\n   Classification Report:")
    print(
        classification_report(
            y_test, y_pred_rf, target_names=["Incorrect", "Correct"], prefix="      "
        )
    )

    # Feature importance
    feature_importance_rf = pd.DataFrame(
        {"feature": feature_cols, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\n   Top 5 Features (Random Forest):")
    for idx, row in feature_importance_rf.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")

    # Save Random Forest model
    rf_model_path = exercise_dir / "random_forest_model.pkl"
    joblib.dump(rf_model, rf_model_path)
    print(f"\n   ✓ Model saved: {rf_model_path}")

    results["random_forest_accuracy"] = float(rf_accuracy)
    results["random_forest_confusion_matrix"] = confusion_matrix(
        y_test, y_pred_rf
    ).tolist()

    # Train XGBoost
    print(f"\n4. Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

    print(f"   ✓ XGBoost Accuracy: {xgb_accuracy:.4f}")
    print("\n   Classification Report:")
    print(
        classification_report(
            y_test, y_pred_xgb, target_names=["Incorrect", "Correct"], prefix="      "
        )
    )

    # Feature importance
    feature_importance_xgb = pd.DataFrame(
        {"feature": feature_cols, "importance": xgb_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\n   Top 5 Features (XGBoost):")
    for idx, row in feature_importance_xgb.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")

    # Save XGBoost model
    xgb_model_path = exercise_dir / "xgboost_model.pkl"
    joblib.dump(xgb_model, xgb_model_path)
    print(f"\n   ✓ Model saved: {xgb_model_path}")

    results["xgboost_accuracy"] = float(xgb_accuracy)
    results["xgboost_confusion_matrix"] = confusion_matrix(y_test, y_pred_xgb).tolist()

    # Save feature info
    feature_info = {
        "exercise": exercise_name,
        "feature_names": feature_cols,
        "model_type": "binary_classification",
        "classes": ["incorrect", "correct"],
        "random_forest_accuracy": float(rf_accuracy),
        "xgboost_accuracy": float(xgb_accuracy),
    }

    feature_info_path = exercise_dir / "feature_info.json"
    with open(feature_info_path, "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"\n   ✓ Feature info saved: {feature_info_path}")

    # Model comparison
    print(f"\n   {'=' * 50}")
    print(f"   Model Comparison for {exercise_name}:")
    print(f"   Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"   XGBoost Accuracy: {xgb_accuracy:.4f}")
    if rf_accuracy > xgb_accuracy:
        print(f"   → Random Forest performs better (+{rf_accuracy - xgb_accuracy:.4f})")
    else:
        print(f"   → XGBoost performs better (+{xgb_accuracy - rf_accuracy:.4f})")
    print(f"   {'=' * 50}")

    return results


def main():
    print("=" * 60)
    print("Exercise Classification Model Training System")
    print("=" * 60)

    # Define directories
    base_dir = Path(__file__).parent
    tracked_data_dir = base_dir / "tracked_data"
    models_dir = base_dir / "tracking_models"

    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)

    # Define feature columns
    feature_cols = [
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_shoulder_angle",
        "right_shoulder_angle",
    ]

    print("\n1. Discovering exercise folders...")
    exercises = discover_exercise_folders(tracked_data_dir)

    if not exercises:
        print("   ✗ No exercises found in tracked_data folder")
        return

    print(f"\n✓ Found {len(exercises)} exercise(s)")

    # Train models for each exercise
    all_results = []
    for exercise_name, json_files in exercises.items():
        print(f"\n2. Loading data for '{exercise_name}'...")
        df_all = load_exercise_data(json_files)

        if df_all.empty:
            print(f"   ✗ No valid data for exercise '{exercise_name}'")
            continue

        # Train models
        results = train_models_for_exercise(
            exercise_name, df_all, feature_cols, models_dir
        )
        all_results.append(results)

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    if all_results:
        print(f"\nSuccessfully trained models for {len(all_results)} exercise(s):")
        for result in all_results:
            if "error" not in result:
                exercise = result["exercise"]
                rf_acc = result.get("random_forest_accuracy", "N/A")
                xgb_acc = result.get("xgboost_accuracy", "N/A")
                print(
                    f"  • {exercise}: RF={rf_acc:.4f}, XGB={xgb_acc:.4f}"
                    if isinstance(rf_acc, float)
                    else f"  • {exercise}: Error"
                )

        print(f"\nModels saved in: {models_dir}")

        # Save summary report
        summary_path = models_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Summary report saved: {summary_path}")
    else:
        print("✗ No models were successfully trained")


if __name__ == "__main__":
    main()

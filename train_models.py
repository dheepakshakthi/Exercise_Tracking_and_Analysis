import json
from pathlib import Path

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


def main():
    print("=" * 60)
    print("Exercise Classification Model Training")
    print("=" * 60)

    # Define file paths
    data_dir = Path(__file__).parent
    c1_file = data_dir / "tp1_c1_data.json"
    w1_file = data_dir / "tp1_w1_data.json"
    w2_file = data_dir / "tp1_w2_data.json"

    # Check if files exist
    for filepath in [c1_file, w1_file, w2_file]:
        if not filepath.exists():
            print(f"Error: File not found - {filepath}")
            return

    print("\n1. Loading data...")
    # Load JSON data
    c1_data = load_json_data(c1_file)
    w1_data = load_json_data(w1_file)
    w2_data = load_json_data(w2_file)
    print("   ✓ Data loaded successfully")

    print("\n2. Extracting features...")
    # Extract features with labels
    df_c1 = extract_features_from_json(c1_data, label=1)  # Correct exercise
    df_w1 = extract_features_from_json(w1_data, label=0)  # Incorrect exercise
    df_w2 = extract_features_from_json(w2_data, label=0)  # Incorrect exercise

    print(f"   - Correct exercise frames: {len(df_c1)}")
    print(f"   - Incorrect exercise (w1) frames: {len(df_w1)}")
    print(f"   - Incorrect exercise (w2) frames: {len(df_w2)}")

    # Combine all data
    df_all = pd.concat([df_c1, df_w1, df_w2], ignore_index=True)
    print(f"   ✓ Total frames: {len(df_all)}")

    print("\n3. Preprocessing data...")
    # Define feature columns (only angle features)
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

    # Handle missing values
    print(f"   - Missing values before: {df_all[feature_cols].isna().sum().sum()}")

    # Strategy: Fill missing values with the median of each column
    df_all[feature_cols] = df_all[feature_cols].fillna(df_all[feature_cols].median())

    print(f"   - Missing values after: {df_all[feature_cols].isna().sum().sum()}")
    print("   ✓ Preprocessing complete")

    # Prepare features and labels
    X = df_all[feature_cols].values
    y = df_all["label"].values

    print(f"\n4. Splitting data (80% train, 20% test)...")
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Testing samples: {len(X_test)}")
    print(
        f"   - Training correct/incorrect ratio: {np.sum(y_train)}/{len(y_train) - np.sum(y_train)}"
    )
    print(
        f"   - Testing correct/incorrect ratio: {np.sum(y_test)}/{len(y_test) - np.sum(y_test)}"
    )

    # Train Random Forest
    print("\n" + "=" * 60)
    print("Training Random Forest Classifier")
    print("=" * 60)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    rf_model.fit(X_train, y_train)

    # Evaluate Random Forest
    y_pred_rf = rf_model.predict(X_test)

    print("\nRandom Forest Results:")
    print("-" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print("\nClassification Report:")
    print(
        classification_report(y_test, y_pred_rf, target_names=["Incorrect", "Correct"])
    )
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    # Feature importance
    feature_importance_rf = pd.DataFrame(
        {"feature": feature_cols, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nFeature Importance (Random Forest):")
    print(feature_importance_rf.to_string(index=False))

    # Save Random Forest model
    rf_model_path = data_dir / "random_forest_model.pkl"
    joblib.dump(rf_model, rf_model_path)
    print(f"\n✓ Random Forest model saved to: {rf_model_path}")

    # Train XGBoost
    print("\n" + "=" * 60)
    print("Training XGBoost Classifier")
    print("=" * 60)

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    xgb_model.fit(X_train, y_train)

    # Evaluate XGBoost
    y_pred_xgb = xgb_model.predict(X_test)

    print("\nXGBoost Results:")
    print("-" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print("\nClassification Report:")
    print(
        classification_report(y_test, y_pred_xgb, target_names=["Incorrect", "Correct"])
    )
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))

    # Feature importance
    feature_importance_xgb = pd.DataFrame(
        {"feature": feature_cols, "importance": xgb_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nFeature Importance (XGBoost):")
    print(feature_importance_xgb.to_string(index=False))

    # Save XGBoost model
    xgb_model_path = data_dir / "xgboost_model.pkl"
    joblib.dump(xgb_model, xgb_model_path)
    print(f"\n✓ XGBoost model saved to: {xgb_model_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModels saved:")
    print(f"  1. Random Forest: {rf_model_path}")
    print(f"  2. XGBoost: {xgb_model_path}")
    print(f"\nModel Comparison:")
    print(f"  Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"  XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")

    # Save feature names for later use
    feature_info = {
        "feature_names": feature_cols,
        "model_type": "binary_classification",
        "classes": ["incorrect", "correct"],
    }

    feature_info_path = data_dir / "feature_info.json"
    with open(feature_info_path, "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"\n✓ Feature info saved to: {feature_info_path}")


if __name__ == "__main__":
    main()

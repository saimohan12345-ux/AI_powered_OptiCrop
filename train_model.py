#!/usr/bin/env python3
"""
OptiCrop — Model Training Script
=====================================
Trains a RandomForestClassifier on the crop recommendation dataset.
Also trains a LogisticRegression for comparison.
Saves model artifacts (model, encoder, scaler) and performance metrics.

Usage:
    python train_model.py
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import (
    DATASET_PATH, MODEL_DIR, MODEL_PATH,
    METRICS_PATH, FEATURE_COLS,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_and_save() -> dict:
    """
    Full training pipeline:
      1. Load dataset
      2. Encode labels + scale features
      3. Train RandomForest (primary) + LogisticRegression (comparison)
      4. Evaluate, compute metrics
      5. Persist artifacts

    Returns:
        dict: Saved metrics dictionary
    """
    # ── 1. Load ────────────────────────────────────────────────────────────────
    logger.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    logger.info("Dataset shape: %s | Crops: %d", df.shape, df["label"].nunique())

    X_raw = df[FEATURE_COLS].values
    y_raw = df["label"].values

    # ── 2. Encode + Scale ──────────────────────────────────────────────────────
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    logger.info("Label classes: %s", list(encoder.classes_))

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ── 3. Train / Test Split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ── 4a. Primary Model: Random Forest ──────────────────────────────────────
    logger.info("Training RandomForestClassifier …")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred       = rf.predict(X_test)
    rf_proba      = rf.predict_proba(X_test)
    rf_accuracy   = accuracy_score(y_test, rf_pred)
    rf_cv_scores  = cross_val_score(rf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    logger.info(
        "RF  Accuracy : %.4f | CV: %.4f ± %.4f",
        rf_accuracy, rf_cv_scores.mean(), rf_cv_scores.std(),
    )

    # ── 4b. Comparison Model: Logistic Regression ─────────────────────────────
    logger.info("Training LogisticRegression (comparison) …")
    lr = LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")
    lr.fit(X_train, y_train)
    lr_pred     = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    logger.info("LR  Accuracy : %.4f", lr_accuracy)

    # ── 5. Metrics ─────────────────────────────────────────────────────────────
    cr  = classification_report(y_test, rf_pred, target_names=encoder.classes_, output_dict=True)
    cm  = confusion_matrix(y_test, rf_pred)
    fi  = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))

    # Per-crop accuracy
    crop_accuracy = {}
    for idx, cls in enumerate(encoder.classes_):
        mask = y_test == idx
        if mask.sum() > 0:
            crop_accuracy[cls] = float(accuracy_score(y_test[mask], rf_pred[mask]))

    metrics = {
        "rf_accuracy":      float(rf_accuracy),
        "rf_cv_mean":       float(rf_cv_scores.mean()),
        "rf_cv_std":        float(rf_cv_scores.std()),
        "lr_accuracy":      float(lr_accuracy),
        "total_samples":    int(len(X)),
        "train_samples":    int(len(X_train)),
        "test_samples":     int(len(X_test)),
        "n_features":       int(len(FEATURE_COLS)),
        "n_classes":        int(len(encoder.classes_)),
        "classes":          encoder.classes_.tolist(),
        "feature_cols":     FEATURE_COLS,
        "feature_importance": fi,
        "crop_accuracy":    crop_accuracy,
        "classification_report": cr,
        "confusion_matrix": cm.tolist(),
    }

    # ── 6. Save Artifacts ─────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model":        rf,
        "encoder":      encoder,
        "scaler":       scaler,
        "feature_cols": FEATURE_COLS,
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    logger.info("Model artifact saved → %s", MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved      → %s", METRICS_PATH)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    bar = "=" * 60
    print(f"\n{bar}")
    print("  [OK] OptiCrop -- Model Training Complete")
    print(bar)
    print(f"  Algorithm        : RandomForestClassifier (200 trees)")
    print(f"  Test Accuracy    : {rf_accuracy * 100:.2f}%")
    print(f"  CV Accuracy      : {rf_cv_scores.mean() * 100:.2f}% +/- {rf_cv_scores.std() * 100:.2f}%")
    print(f"  LR Accuracy      : {lr_accuracy * 100:.2f}%  (comparison)")
    print(f"  Dataset Size     : {len(X)} samples | {len(encoder.classes_)} crops")
    print(f"  Model saved to   : {MODEL_PATH}")
    print(f"{bar}\n")

    return metrics


if __name__ == "__main__":
    train_and_save()

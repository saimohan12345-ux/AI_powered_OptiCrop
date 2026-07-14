#!/usr/bin/env python3
"""
OptiCrop — Configuration Module
Centralises all paths, constants, and environment settings.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "data.csv"
MODEL_DIR    = BASE_DIR / "model"
MODEL_PATH   = MODEL_DIR / "trained_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

# ─── Features ─────────────────────────────────────────────────────────────────
# Order MUST match the training column order
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FEATURE_LABELS = {
    "N":           "Nitrogen (N)",
    "P":           "Phosphorous (P)",
    "K":           "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity":    "Humidity (%)",
    "ph":          "pH Level",
    "rainfall":    "Rainfall (mm)",
}

FEATURE_RANGES = {
    "N":           {"min": 0,   "max": 140, "step": 1,   "default": 50},
    "P":           {"min": 5,   "max": 145, "step": 1,   "default": 53},
    "K":           {"min": 5,   "max": 205, "step": 1,   "default": 48},
    "temperature": {"min": 8,   "max": 44,  "step": 0.1, "default": 25},
    "humidity":    {"min": 14,  "max": 100, "step": 0.1, "default": 71},
    "ph":          {"min": 3.5, "max": 10,  "step": 0.1, "default": 6.5},
    "rainfall":    {"min": 20,  "max": 299, "step": 0.1, "default": 103},
}

# ─── Flask ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "opcrop-dev-secret-change-in-prod")
DEBUG      = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
HOST       = os.environ.get("HOST", "0.0.0.0")
PORT       = int(os.environ.get("PORT", 5000))

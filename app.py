#!/usr/bin/env python3
"""
OptiCrop — Flask Application
=====================================
Production-ready Flask web application for the Smart Agricultural
Production Optimization Engine.

Routes:
    GET  /                  → Home page (3 use-case scenarios)
    GET  /predict           → Crop prediction form
    POST /api/predict       → JSON inference endpoint
    GET  /dashboard         → EDA & analytics dashboard
    GET  /api/dashboard-data→ JSON data for dashboard charts
    POST /api/batch-predict → CSV batch prediction
    GET  /api/crop-info/<c> → Single crop information JSON
    GET  /about             → About OptiCrop
    GET  /contact           → Contact page
"""

import io
import json
import logging
import traceback

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash,
)

import config
from predict import predict_crop, load_artifacts, load_metrics, CROP_INFO

# ─── App Initialisation ───────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.SECRET_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Pre-load model at startup (fail fast if model missing)
try:
    load_artifacts()
    logger.info("✅ Model loaded successfully at startup.")
except FileNotFoundError as exc:
    logger.warning("⚠️  %s — run `python train_model.py` before starting the app.", exc)


# ─── Dataset Helper ───────────────────────────────────────────────────────────
def _load_dataset() -> pd.DataFrame:
    return pd.read_csv(config.DATASET_PATH)


# ─── Main Pages ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Home / landing page with 3 use-case scenarios."""
    metrics = load_metrics()
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["GET"])
def predict_page():
    """Render the crop prediction input form."""
    return render_template(
        "prediction.html",
        features=config.FEATURE_COLS,
        feature_labels=config.FEATURE_LABELS,
        feature_ranges=config.FEATURE_RANGES,
    )


@app.route("/dashboard")
def dashboard():
    """Render the analytics dashboard page."""
    metrics = load_metrics()
    return render_template("dashboard.html", metrics=metrics)


@app.route("/about")
def about():
    """Render the About page."""
    metrics = load_metrics()
    return render_template("about.html", metrics=metrics)


@app.route("/contact")
def contact():
    """Render the Contact page."""
    return render_template("contact.html")


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Request JSON: {N, P, K, temperature, humidity, ph, rainfall}
    Response JSON: prediction result dict
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "No JSON payload received."}), 400

        required = config.FEATURE_COLS
        missing  = [f for f in required if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error":   f"Missing fields: {', '.join(missing)}",
            }), 400

        values = {f: float(data[f]) for f in required}
        result = predict_crop(**values)

        if not result["success"]:
            return jsonify(result), 500

        return jsonify(result), 200

    except (ValueError, TypeError) as exc:
        return jsonify({"success": False, "error": f"Invalid input: {exc}"}), 400
    except Exception as exc:
        logger.exception("Unexpected error in /api/predict")
        return jsonify({"success": False, "error": "Internal server error."}), 500


@app.route("/api/dashboard-data")
def api_dashboard_data():
    """
    GET /api/dashboard-data
    Returns aggregated dataset statistics for Chart.js charts.
    """
    try:
        df = _load_dataset()

        # ── Summary statistics ────────────────────────────────────────────────
        summary = {
            "total_samples": int(len(df)),
            "n_crops":       int(df["label"].nunique()),
            "avg_N":         round(float(df["N"].mean()), 2),
            "avg_P":         round(float(df["P"].mean()), 2),
            "avg_K":         round(float(df["K"].mean()), 2),
            "avg_temp":      round(float(df["temperature"].mean()), 2),
            "avg_humidity":  round(float(df["humidity"].mean()), 2),
            "avg_ph":        round(float(df["ph"].mean()), 2),
            "avg_rainfall":  round(float(df["rainfall"].mean()), 2),
        }

        # ── Crop distribution ─────────────────────────────────────────────────
        crop_counts = df["label"].value_counts()
        crop_distribution = {
            "labels": [CROP_INFO.get(c, {}).get("display", c.title()) for c in crop_counts.index],
            "data":   crop_counts.values.tolist(),
            "colors": [CROP_INFO.get(c, {}).get("color", "#16a34a") for c in crop_counts.index],
        }

        # ── NPK by crop (mean per crop, sorted by N desc) ─────────────────────
        npk = df.groupby("label")[["N", "P", "K"]].mean().round(2)
        npk = npk.sort_values("N", ascending=False)
        npk_stats = {
            "crops": [CROP_INFO.get(c, {}).get("display", c.title()) for c in npk.index],
            "N":     npk["N"].tolist(),
            "P":     npk["P"].tolist(),
            "K":     npk["K"].tolist(),
        }

        # ── Climate means per crop ────────────────────────────────────────────
        climate = df.groupby("label")[["temperature", "humidity", "rainfall"]].mean().round(2)
        climate_data = {
            "crops":       [CROP_INFO.get(c, {}).get("display", c.title()) for c in climate.index],
            "temperature": climate["temperature"].tolist(),
            "humidity":    climate["humidity"].tolist(),
            "rainfall":    climate["rainfall"].tolist(),
        }

        # ── Feature distributions (histogram buckets) ─────────────────────────
        hist_data = {}
        for col in config.FEATURE_COLS:
            counts, edges = np.histogram(df[col], bins=15)
            hist_data[col] = {
                "counts": counts.tolist(),
                "edges":  [round(float(e), 2) for e in edges.tolist()],
                "labels": [f"{round(float(edges[i]),1)}–{round(float(edges[i+1]),1)}"
                           for i in range(len(edges) - 1)],
            }

        # ── Season segmentation ───────────────────────────────────────────────
        season_crops = {
            "summer":  df[(df["temperature"] > 30) & (df["humidity"] > 50)]["label"].unique().tolist(),
            "winter":  df[(df["temperature"] < 20) & (df["humidity"] > 30)]["label"].unique().tolist(),
            "monsoon": df[(df["rainfall"] > 200) & (df["humidity"] > 30)]["label"].unique().tolist(),
        }
        # Humanise names
        season_crops = {
            k: [CROP_INFO.get(c, {}).get("display", c.title()) for c in v]
            for k, v in season_crops.items()
        }

        # ── Interesting patterns ──────────────────────────────────────────────
        patterns = {
            "high_N":       [CROP_INFO.get(c, {}).get("display", c.title())
                             for c in df[df["N"] > 120]["label"].unique()],
            "high_P":       [CROP_INFO.get(c, {}).get("display", c.title())
                             for c in df[df["P"] > 100]["label"].unique()],
            "high_K":       [CROP_INFO.get(c, {}).get("display", c.title())
                             for c in df[df["K"] > 200]["label"].unique()],
            "high_rain":    [CROP_INFO.get(c, {}).get("display", c.title())
                             for c in df[df["rainfall"] > 200]["label"].unique()],
            "low_temp":     [CROP_INFO.get(c, {}).get("display", c.title())
                             for c in df[df["temperature"] < 10]["label"].unique()],
            "high_temp":    [CROP_INFO.get(c, {}).get("display", c.title())
                             for c in df[df["temperature"] > 40]["label"].unique()],
            "low_humidity": [CROP_INFO.get(c, {}).get("display", c.title())
                             for c in df[df["humidity"] < 20]["label"].unique()],
            "low_ph":       [CROP_INFO.get(c, {}).get("display", c.title())
                             for c in df[df["ph"] < 4]["label"].unique()],
            "high_ph":      [CROP_INFO.get(c, {}).get("display", c.title())
                             for c in df[df["ph"] > 9]["label"].unique()],
        }

        return jsonify({
            "summary":           summary,
            "crop_distribution": crop_distribution,
            "npk_stats":         npk_stats,
            "climate_data":      climate_data,
            "hist_data":         hist_data,
            "season_crops":      season_crops,
            "patterns":          patterns,
        })

    except Exception as exc:
        logger.exception("Error generating dashboard data")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/crop-info/<crop>")
def api_crop_info(crop: str):
    """GET /api/crop-info/<crop> — Returns full crop information."""
    info = CROP_INFO.get(crop.lower())
    if not info:
        return jsonify({"error": f"Crop '{crop}' not found."}), 404
    return jsonify(info)


@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """
    POST /api/batch-predict
    Accepts a CSV file upload with columns: N,P,K,temperature,humidity,ph,rainfall
    Returns JSON array of predictions.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"success": False, "error": "Only CSV files are accepted."}), 400

    try:
        df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
        missing_cols = [c for c in config.FEATURE_COLS if c not in df.columns]
        if missing_cols:
            return jsonify({
                "success": False,
                "error":   f"Missing columns: {', '.join(missing_cols)}",
            }), 400

        results = []
        for _, row in df.iterrows():
            res = predict_crop(
                N=float(row["N"]),
                P=float(row["P"]),
                K=float(row["K"]),
                temperature=float(row["temperature"]),
                humidity=float(row["humidity"]),
                ph=float(row["ph"]),
                rainfall=float(row["rainfall"]),
            )
            results.append({
                "N":          float(row["N"]),
                "P":          float(row["P"]),
                "K":          float(row["K"]),
                "temperature": float(row["temperature"]),
                "humidity":   float(row["humidity"]),
                "ph":         float(row["ph"]),
                "rainfall":   float(row["rainfall"]),
                "crop":       res.get("display", "Unknown"),
                "emoji":      res.get("emoji", "🌱"),
                "confidence": res.get("confidence", 0),
            })

        return jsonify({"success": True, "count": len(results), "results": results})

    except Exception as exc:
        logger.exception("Batch prediction error")
        return jsonify({"success": False, "error": str(exc)}), 500


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.error("500 error: %s", e)
    return render_template("index.html"), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)

# OptiCrop — Smart Agricultural Production Optimization Engine

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-99%25+-brightgreen)](#)

> AI-powered crop recommendation system that analyzes soil and climate parameters
> to suggest the most suitable crop for maximum agricultural yield.

---

## Project Overview

OptiCrop helps farmers, researchers, and policymakers make data-driven agricultural
decisions through three core use-case scenarios:

| Scenario | Description |
|---|---|
| **Scenario 1** | Smart Crop Recommendation for Farmers |
| **Scenario 2** | Crop Suitability and Environmental Assessment |
| **Scenario 3** | Agricultural Research and Policy Planning |

---

## Features

- ✅ **AI Crop Recommendation** — RandomForestClassifier with 200 trees (~99% accuracy)
- ✅ **Confidence Scores** — Probability distribution across all 22 crop types
- ✅ **Crop Information Database** — Season, soil, NPK ideals, and farming tips for all 22 crops
- ✅ **Interactive Dashboard** — Crop distribution, NPK profiles, climate analysis, seasonal patterns
- ✅ **Batch Prediction** — Upload a CSV for bulk predictions
- ✅ **Print / PDF** — Print prediction reports via browser
- ✅ **Dark / Light Mode** — Persistent theme toggle

---

## Project Structure

```
Agricultural-Production-Optimization-Engine/
├── app.py              # Flask application (routes + APIs)
├── train_model.py      # Model training script (run once)
├── predict.py          # Inference module + crop info DB
├── config.py           # Paths, constants, feature definitions
│
├── dataset/
│   └── data.csv        # 2,200 crop samples (7 features × 22 crops)
│
├── model/
│   ├── trained_model.pkl   # Serialized RF model + encoder + scaler
│   └── metrics.json        # Accuracy, CV scores, confusion matrix
│
├── templates/
│   ├── base.html       # Navbar, footer, theme toggle
│   ├── index.html      # Home + 3 scenarios
│   ├── prediction.html # Crop prediction form + results
│   ├── dashboard.html  # EDA analytics dashboard
│   ├── about.html      # About OptiCrop
│   └── contact.html    # Contact page
│
├── static/
│   ├── css/style.css   # Complete dark agriculture theme
│   └── js/main.js      # Theme toggle, AOS, utilities
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (once)

```bash
python train_model.py
```

Expected output:
```
✅ OptiCrop — Model Training Complete
   Random Forest Accuracy : 99.xx%
   CV Mean Accuracy       : 99.xx% ± 0.xx%
   Model saved to         : model/trained_model.pkl
```

### 3. Run the Flask app

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## ML Pipeline

| Step | Details |
|---|---|
| **Dataset** | 2,200 samples · 7 features · 22 crop classes |
| **Preprocessing** | StandardScaler (features) · LabelEncoder (target) |
| **Train/Test Split** | 80% / 20% stratified |
| **Algorithm** | `RandomForestClassifier(n_estimators=200, random_state=42)` |
| **Validation** | 5-fold cross-validation |
| **Comparison** | LogisticRegression (~95%) vs Random Forest (~99%) |
| **Serialization** | Joblib (model + encoder + scaler in single artifact) |

### Input Features

| Feature | Unit | Description |
|---|---|---|
| N | kg/ha | Nitrogen content in soil |
| P | kg/ha | Phosphorous content in soil |
| K | kg/ha | Potassium content in soil |
| temperature | °C | Ambient temperature |
| humidity | % | Relative humidity |
| ph | — | Soil pH value |
| rainfall | mm | Rainfall amount |

### Output: 22 Supported Crops

Rice · Maize · Chickpea · Kidney Beans · Pigeon Peas · Moth Beans ·
Mung Bean · Black Gram · Lentil · Pomegranate · Banana · Mango ·
Grapes · Watermelon · Muskmelon · Apple · Orange · Papaya ·
Coconut · Cotton · Jute · Coffee

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Home page |
| `/predict` | GET | Prediction form |
| `/dashboard` | GET | Analytics dashboard |
| `/about` | GET | About page |
| `/contact` | GET | Contact page |
| `/api/predict` | POST | JSON inference |
| `/api/dashboard-data` | GET | Chart data JSON |
| `/api/batch-predict` | POST | CSV batch prediction |
| `/api/crop-info/<crop>` | GET | Crop information JSON |

### Example: POST /api/predict

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":23,"humidity":82,"ph":6.5,"rainfall":200}'
```

Response:
```json
{
  "success": true,
  "crop": "rice",
  "display": "Rice",
  "emoji": "🌾",
  "confidence": 98.5,
  "top_predictions": [...],
  "crop_info": { "season": "...", "ideal_temp": "...", ... }
}
```

---

## Tech Stack

**Backend:** Python · Flask · scikit-learn · Pandas · NumPy · Joblib  
**Frontend:** Bootstrap 5 · Chart.js · AOS Animations · Font Awesome  
**ML:** RandomForestClassifier · StandardScaler · LabelEncoder · Cross-validation

---

## Original Work Preserved

- All original ML logic from `engine/Optimizing Agricultural Production.py` is preserved
- Same algorithm family (scikit-learn), upgraded to RandomForest for better accuracy
- All preprocessing steps (scaling, encoding) carried forward
- All 22 crop classes retained
- Dataset unchanged

---

*Built for smart, sustainable agriculture decision-making.*

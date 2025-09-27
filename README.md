# CardioMetrix

**Calibrated XGBoost screening for Diabetes & Hypertension** — with fairness checks, SHAP explainability, and a polished Flask + Bootstrap dashboard.

![CardioMetrix Dashboard](assets/dashboard.png)

> **Disclaimer**  
> This project is for **educational decision support** only. It is **not** a medical device and **not** a diagnosis. Always use clinical judgment and confirm with appropriate tests.

---

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [Live Demo Screenshots](#live-demo-screenshots)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Data Sources & Harmonization](#data-sources--harmonization)
- [Label Policy](#label-policy)
- [Modeling Choices](#modeling-choices)
- [Training & Evaluation](#training--evaluation)
- [Fairness & Responsible AI](#fairness--responsible-ai)
- [Explainability (SHAP)](#explainability-shap)
- [API (Flask)](#api-flask)
- [UI (Bootstrap + Chartjs)](#ui-bootstrap--chartjs)
- [Reproducibility](#reproducibility)
- [Roadmap](#roadmap)
- [FAQ](#faq)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Motivation

Preventive care wins when it’s **early, explainable, and responsible**. CardioMetrix is a compact, end-to-end pipeline that:

- Trains **tabular-first models** (XGBoost) for **diabetes** and **hypertension** screening.  
- Ships **calibrated probabilities** and **transparent decisions** (threshold policy).  
- Surfaces **patient-level explanations** (SHAP) and **fairness slices** (sex/age).  
- Provides a clean **Flask API** and a **Bootstrap dashboard** for demo or offline use.

This repo is resume-ready and production-minded: data is harmonized, labels are documented, splits are persisted, artifacts are versioned, and UI is polished.

---

## Features

- 🧠 **Primary model**: **XGBoost** for tabular data; **Logistic Regression** as a baseline reference.  
- 🧼 **Leakage-aware features**:  
  - Diabetes model **excludes fasting glucose** (kept only for labeling).  
  - Hypertension model may use **SBP/DBP** (clinical signal).  
- 🎯 **Calibration**: Isotonic calibration on the validation set; decision thresholds chosen on validation.  
- ⚖️ **Fairness**: Slice metrics by **sex** and **age bands**; optional per-sex thresholds with a recall floor.  
- 🔎 **Explainability**: SHAP (TreeExplainer) top factors per prediction.  
- 📦 **Artifacts**: Preprocessors, models, calibrators, thresholds saved to `registry/` for serving.  
- 🖥️ **Dashboard**: Flask + Bootstrap + Chart.js donuts, with error handling and input validation.

---

## Live Demo Screenshots

**Dashboard**  
![Dashboard](assets/dashboard.png)

**Male example (expected low risk)**  
![Prediction — Male](assets/example_male.png)

**Female example (expected higher risk)**  
![Prediction — Female](assets/example_female.png)

> Put your actual screenshots at:
> - `assets/dashboard.png`
> - `assets/example_male.png`
> - `assets/example_female.png`

Suggested test inputs for those screenshots:

- **Male (low risk)**: Age 35, Sex Male, BMI 24.5, SBP 118, DBP 76, Pedigree 0.20  
- **Female (higher risk)**: Age 58, Sex Female, BMI 33.0, SBP 142, DBP 88, Pregnancies 2, Pedigree 0.60

---

## Project Structure

```text
cardiometrix/
├─ app/
│  ├─ app_flask.py           # Flask server (API + dashboard routes)
│  ├─ predictor.py           # Loads artifacts, validates inputs, predicts, SHAP top factors
│  ├─ templates/
│  │  ├─ base.html
│  │  └─ index.html          # Dashboard UI
│  └─ static/
│     ├─ css/app.css
│     └─ js/app.js
├─ data/
│  ├─ raw/                   # Downloaded CSVs (Pima, UCI Heart, CKD)
│  └─ processed/
│     ├─ harmonized.csv
│     └─ harmonized_labeled{__train,__val,__test}.csv
├─ docs/
│  ├─ PRD.md                 # Product Requirements (objectives, metrics, ethics)
│  ├─ DATA_DICTIONARY.csv
│  ├─ LABEL_POLICY.md
│  └─ SPLIT_STRATEGY.md
├─ indices/
│  ├─ train_idx.csv
│  ├─ val_idx.csv
│  └─ test_idx.csv
├─ notebooks/                # Training pipeline (Week 1–2)
├─ registry/                 # Production artifacts (for serving)
│  ├─ prod_preprocessor__{diabetes,hypertension}.joblib
│  ├─ prod_xgb__{diabetes,hypertension}.joblib
│  ├─ prod_calibrator__{diabetes,hypertension}.joblib
│  ├─ prod_thresholds.json
│  └─ feature_spec_per_target.joblib
└─ README.md

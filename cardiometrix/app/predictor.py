import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import joblib

# Optional (SHAP): if not installed, we degrade gracefully
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ---------- Paths ----------
BASE = Path(__file__).resolve().parents[1]
REG  = BASE / "registry"

# ---------- Load artifacts (once) ----------
# Fitted preprocessors
PRE_D = joblib.load(REG / "prod_preprocessor__diabetes.joblib")
PRE_H = joblib.load(REG / "prod_preprocessor__hypertension.joblib")

# XGB models
XGB_D = joblib.load(REG / "prod_xgb__diabetes.joblib")
XGB_H = joblib.load(REG / "prod_xgb__hypertension.joblib")

# Isotonic calibrators
ISO_D = joblib.load(REG / "prod_calibrator__diabetes.joblib")
ISO_H = joblib.load(REG / "prod_calibrator__hypertension.joblib")

# Threshold policy
with open(REG / "prod_thresholds.json", "r") as f:
    THRESH = json.load(f)

# ---------- Feature specs (must match training) ----------
SPEC = joblib.load(REG / "feature_spec_per_target.joblib")
COLS_D = SPEC["diabetes"]["NUM"] + SPEC["diabetes"]["CAT"]
COLS_H = SPEC["hypertension"]["NUM"] + SPEC["hypertension"]["CAT"]

# ---------- Utilities ----------
def _as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _norm_sex(x):
    if x is None: return np.nan
    s = str(x).strip().upper()
    if s in ["M","MALE","1"]: return "M"
    if s in ["F","FEMALE","0"]: return "F"
    return np.nan

def _default_source(x):
    s = "external"
    if x is None or str(x).strip() == "":
        return s
    return str(x)

def _build_row(payload: Dict[str, Any], target: str) -> Dict[str, Any]:
    # Minimal schema: we only request what each target needs (others are ignored)
    if target == "diabetes":
        row = {
            "age": _as_float(payload.get("age")),
            "bmi": _as_float(payload.get("bmi")),
            "dbp": _as_float(payload.get("dbp")),
            "extra__pregnancies": _as_float(payload.get("extra__pregnancies")),
            "extra__diabetespedigreefunction": _as_float(payload.get("extra__diabetespedigreefunction")),
            "sex": _norm_sex(payload.get("sex")),
            "source_dataset": _default_source(payload.get("source_dataset")),
        }
        # Fill any missing keys expected by preprocessor
        for k in COLS_D:
            if k not in row:
                row[k] = np.nan
        return row

    elif target == "hypertension":
        row = {
            "age": _as_float(payload.get("age")),
            "bmi": _as_float(payload.get("bmi")),
            "sbp": _as_float(payload.get("sbp")),
            "dbp": _as_float(payload.get("dbp")),
            "sex": _norm_sex(payload.get("sex")),
            "source_dataset": _default_source(payload.get("source_dataset")),
        }
        for k in COLS_H:
            if k not in row:
                row[k] = np.nan
        return row

    else:
        raise ValueError("Unknown target: " + target)

def _prep_matrix(row: Dict[str, Any], target: str) -> np.ndarray:
    import pandas as pd
    if target == "diabetes":
        df = pd.DataFrame([row], columns=COLS_D)
        X = PRE_D.transform(df)
    else:
        df = pd.DataFrame([row], columns=COLS_H)
        X = PRE_H.transform(df)
    return X

def _calibrated_prob(model, calibrator, X: np.ndarray) -> float:
    # XGBoost predict_proba → isotonic predict → scalar
    p_raw = model.predict_proba(X)[:, 1]
    p_cal = calibrator.predict(p_raw)
    return float(np.clip(p_cal[0], 1e-7, 1 - 1e-7))

def _feat_names(pre, num: List[str], cat: List[str]) -> List[str]:
    # Build post-transform feature names for SHAP mapping
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(cat))
    return list(num) + cat_names

# Lazily build SHAP explainers (small inputs → fast)
_EXPLAINERS = {"diabetes": None, "hypertension": None}
_FEATNAMES  = {"diabetes": None, "hypertension": None}

def _get_explainer(target: str):
    if not HAS_SHAP:
        return None, None
    if target == "diabetes":
        if _EXPLAINERS["diabetes"] is None:
            _EXPLAINERS["diabetes"] = shap.TreeExplainer(XGB_D)
            _FEATNAMES["diabetes"]   = _feat_names(PRE_D, SPEC["diabetes"]["NUM"], SPEC["diabetes"]["CAT"])
        return _EXPLAINERS["diabetes"], _FEATNAMES["diabetes"]
    else:
        if _EXPLAINERS["hypertension"] is None:
            _EXPLAINERS["hypertension"] = shap.TreeExplainer(XGB_H)
            _FEATNAMES["hypertension"]  = _feat_names(PRE_H, SPEC["hypertension"]["NUM"], SPEC["hypertension"]["CAT"])
        return _EXPLAINERS["hypertension"], _FEATNAMES["hypertension"]

def _top_factors(target: str, X: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
    expl, names = _get_explainer(target)
    if expl is None:
        return []
    vals = expl.shap_values(X)  # shape (1, n_features)
    if isinstance(vals, list):  # safety for some xgb versions
        vals = vals[0]
    vals = np.abs(vals).reshape(-1)
    idx = np.argsort(-vals)[:k]
    out = []
    for i in idx:
        out.append((names[i], float(vals[i])))
    return out

def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run both targets (diabetes, hypertension) on one payload.
    Expects keys like: age, bmi, sbp, dbp, sex, extra__pregnancies, extra__diabetespedigreefunction, source_dataset(optional).
    Missing values allowed; preprocessors will impute.
    """
    results = {}

    # ---- Diabetes ----
    row_d = _build_row(payload, "diabetes")
    X_d   = _prep_matrix(row_d, "diabetes")
    p_d   = _calibrated_prob(XGB_D, ISO_D, X_d)
    thr_d = float(THRESH["diabetes"]["threshold_global"])
    y_d   = int(p_d >= thr_d)
    results["diabetes"] = {
        "prob": p_d,
        "threshold": thr_d,
        "decision": y_d,
        "top_factors": _top_factors("diabetes", X_d, k=5)
    }

    # ---- Hypertension ----
    row_h = _build_row(payload, "hypertension")
    X_h   = _prep_matrix(row_h, "hypertension")
    p_h   = _calibrated_prob(XGB_H, ISO_H, X_h)
    thr_h = float(THRESH["hypertension"]["threshold_global"])
    y_h   = int(p_h >= thr_h)
    results["hypertension"] = {
        "prob": p_h,
        "threshold": thr_h,
        "decision": y_h,
        "top_factors": _top_factors("hypertension", X_h, k=5)
    }

    # Echo back sanitized/normalized inputs for traceability
    results["input_used"] = {
        "diabetes_cols": COLS_D,
        "hypertension_cols": COLS_H
    }
    return results

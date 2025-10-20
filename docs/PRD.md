# CardioMetrix — PRD (Week 1)

## Objective
Early triage for:
- **Diabetes** (binary) — risk-factor model (no fasting glucose used by the model).
- **Hypertension** (binary) — may use SBP/DBP (clinical signal).

Outputs: calibrated probabilities, thresholded decisions, explanations, fairness slices.

## Success Metrics
- AUROC (per label) ≥ 0.85 (stretch), strong PR-AUC.
- Calibration (ECE/Brier) acceptable.
- Honest generalization via stratified splits; document fairness gaps.

## Ethics & Guardrails
- Decision support only, not medical advice.
- No leakage (esp. diabetes: exclude fasting_glucose as a feature).
- Group-aware thresholds for diabetes (by sex) with recall floor; fallback to global.

## Data
Public tabular datasets (no PII). Reproducible pipeline: raw → harmonized → labeled → splits.
# LABEL_POLICY.md

## Diabetes (label_diabetes)
1 if (HbA1c ≥ 6.5%) OR (fasting_glucose ≥ 126 mg/dL) OR dataset diagnosis flag == 1; else 0.
*Model features exclude fasting_glucose to avoid leakage.*

## Hypertension (label_hypertension)
1 if (SBP ≥ 140) OR (DBP ≥ 90); else 0. (Standard adult threshold, documented.)

## ASCVD proxy (optional educational regression target)
Sigmoid combination of age/sbp/tc/hdl → calibrated; not a clinical calculator.

## Hygiene
- Drop physiologically impossible values.
- Persist indices; no patient overlap across splits.
# Model Registry

Tracks trained model versions, hyperparameters, and performance metrics.

**Val set:** WC 2022 (64 matches)  
**Test set:** WC 2018 (64 matches)  
**Outcome encoding:** 0 = Away Win, 1 = Draw, 2 = Home Win

---

## Trained Models

| Model File | Date Trained | Type | Val Log-Loss | Val Accuracy | Val Brier | Notes |
|---|---|---|---|---|---|---|
| `outcome_lr.pkl` | 2026-05-20 | Logistic Regression | 1.0683 | 0.4844 | 0.6345 | StandardScaler + LogisticRegression pipeline; default params (no Optuna tuning) |
| `outcome_rf.pkl` | 2026-05-20 | Random Forest (tuned) | 1.0209 | 0.5313 | 0.6032 | Best individual model on val set; tuned via Optuna (50 trials) |
| `outcome_xgb.pkl` | 2026-05-20 | XGBoost (tuned) | 1.0333 | 0.5469 | 0.6080 | Highest val accuracy individual model; tuned via Optuna (100 trials) |
| `outcome_ensemble.pkl` | 2026-05-20 | Soft-voting Ensemble | 1.0368 | 0.5313 | 0.6124 | Averages LR + RF + XGB predict_proba; best test-set model (LL=1.0137) |
| `outcome_xgb_calibrated.pkl` | TBD | XGB + Isotonic Calibration | TBD | TBD | TBD | Calibrated with CalibratedClassifierCV(cv='prefit', method='isotonic') on 64-row WC 2022 val set; serialised in Subphase 5.9 |
| `outcome_ensemble_calibrated.pkl` | TBD | Calibrated Ensemble | TBD | TBD | TBD | LR + RF + calibrated XGB; applied only if val log-loss improves; serialised in Subphase 5.9 |
| `home_goals_xgb.pkl` | 2026-05-20 | XGBRegressor (tuned) | — | — | — | Val MAE=1.0777; tuned via Optuna (50 trials) |
| `away_goals_xgb.pkl` | 2026-05-20 | XGBRegressor (tuned) | — | — | — | Val MAE=0.8464; tuned via Optuna (50 trials) |
| `home_goals_poisson.pkl` | 2026-05-20 | Poisson Regressor | — | — | — | Fallback; val MAE=1.1274 |
| `away_goals_poisson.pkl` | 2026-05-20 | Poisson Regressor | — | — | — | Fallback; val MAE=0.8381 |

---

## Calibration Strategy

Calibration is assessed in Subphase 5.8 via reliability diagrams (plots saved to `outputs/plots/calibration_*.png`). The decision to apply `CalibratedClassifierCV` (isotonic, `cv='prefit'`) is conditional: calibration is only applied to the XGBoost outcome model if the resulting ensemble val log-loss does not increase. Isotonic regression is non-parametric and can overfit on small samples (64 rows), so the guard is critical.

Serialised calibrated models are suffixed `_calibrated` (Subphase 5.9). The `MODEL_REGISTRY.md` TBD entries above will be updated after Subphase 5.9 completes.

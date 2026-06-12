# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Machine learning system predicting FIFA World Cup 2026 match outcomes (W/D/L) and scorelines. Served via a Streamlit dashboard at https://fifawc2026predictor.streamlit.app/.

## Commands

```bash
# Activate virtualenv (Windows)
venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_preprocess.py -v

# Launch dashboard locally
streamlit run app/dashboard.py
```

### Pipeline scripts (run in order)

```bash
python scripts/run_feature_engineering.py  # builds data/processed/ parquets (~15 min)
python scripts/run_baseline_models.py      # quick baseline (no tuning)
python scripts/run_tuning.py               # Optuna HPO — 4-6 hrs, skip if pkl exists
python scripts/train.py                    # final ensemble + calibration → models/*.pkl
python scripts/run_backtest.py             # walk-forward validation on WC 2018 & 2022
python scripts/predict.py                  # generates data/processed/wc2026_predictions.csv
python scripts/precompute_predictions.py   # top-3 scoreline probs per match
```

During the live tournament, only the first and last two scripts need to re-run after new results land.

### Live results (actual scores vs predictions)

```bash
python scripts/fetch_results.py            # fetch live/full-time scores -> data/processed/wc2026_results.csv
python scripts/fetch_results.py --summary  # print prediction-vs-result accuracy from the existing CSV
```

The dashboard's Match Predictions page also live-fetches these scores every ~60s (cached), showing a
status badge (Upcoming / 🔴 LIVE / FT), the actual scoreline, a ✅/❌ outcome verdict + ⭐ exact-score
flag, a per-side home-model/away-model goals comparison per card, and a running-accuracy banner
(outcome, exact, and each goals model's hit-rate + MAE). Set `FD_API_KEY` in Streamlit Cloud **secrets** for live
fetch on the hosted app; it falls back to the committed `wc2026_results.csv` when the API/key is absent.

## Architecture

### Data flow

```
data/raw/results.csv          (49k matches, 1872–present)
data/raw/rankings.csv         (70k FIFA ranking records)
data/raw/wc2026_fixtures*.csv (104 WC 2026 fixtures)
        ↓
src/data/preprocess.py  →  33 engineered features per match
        ↓
data/processed/features_train.parquet   (training set, ~26k rows)
data/processed/features_predict.parquet (WC 2026, 104 rows)
        ↓
src/models/ (outcome_model, goals_model, ensemble, tune)
        ↓
models/outcome_model.pkl          (soft-voting ensemble: LR + RF + XGB)
models/home_goals_model.pkl       (XGBoost regressor)
models/away_goals_model.pkl       (XGBoost regressor)
        ↓
data/processed/wc2026_predictions.csv
        ↓
app/dashboard.py  (Streamlit, 4 pages)
```

### Key modules

- **`src/data/preprocess.py`** — All 33 features: ranking diff/ratio, 5- and 10-match form windows, H2H stats, tournament stage ordinal, host-nation flag, rest-day differential. `FEATURE_COLUMNS` list and `REFERENCE_DATE` constant live here.
- **`src/models/ensemble.py`** — Soft-voting over three base classifiers + temperature scaling calibration (T≈1.2, fitted on WC 2022 val set).
- **`src/models/tune.py`** — Optuna HPO with `TPESampler` + `MedianPruner`, 5-fold CV.
- **`src/evaluation/backtest.py`** — Walk-forward protocol that re-trains on all data before each WC tournament; prevents look-ahead bias.
- **`src/betting/edge.py`** — Computes edge = model_prob − implied_prob; flags value (>5%) and avoid (<-5%).
- **`src/evaluation/live_tracking.py`** — Joins live results (`fetch_wc2026_results` in `src/data/ingest.py`) to predictions by `fixture_id` / row index; flags `outcome_correct`, `exact_score_correct`, and per-side `home_goals_correct` / `away_goals_correct` (+ signed errors, since home and away goals come from two separate regressors) per match (excludes `TBD` knockout placeholders). `summarize()` reports running outcome/exact accuracy plus each goals model's exact-hit rate and MAE.
- **`app/dashboard.py`** — Entry point; four pages: predictions, bracket, performance, model info.

### Outcome encoding

`0 = Away Win`, `1 = Draw`, `2 = Home Win` — used throughout models and prediction CSV.

### Sample weights

Training applies `match_importance × recency_weight` where WC matches get weight 3.0, qualifiers 1.5, friendlies 0.5, and recency decays as `0.85^(years_since_match)`.

### Environment

`FD_API_KEY` in `.env` (not committed) is needed by `src/data/ingest.py` to fetch fixtures from football-data.org. All other data is loaded from `data/raw/`.

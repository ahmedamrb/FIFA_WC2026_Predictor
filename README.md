# FIFA World Cup 2026 Match Predictor

A machine learning system that predicts FIFA World Cup 2026 match outcomes and scorelines, combining historical international football results, FIFA rankings, and bookmaker odds to generate calibrated probability estimates for home win, draw, and away win outcomes.

## Live Dashboard

**[https://fifawc2026predictor.streamlit.app/](https://fifawc2026predictor.streamlit.app/)**

Explore match predictions, the tournament bracket, model performance metrics, and data & model information — all in one interactive dashboard.

---

## Features

- **Outcome predictions** — Home win / Draw / Away win probabilities for all 104 WC 2026 fixtures
- **Scoreline predictions** — Expected goals for both teams via XGBoost regression models
- **Soft-voting ensemble** — Logistic Regression + Random Forest + XGBoost (tuned via Optuna)
- **Calibrated probabilities** — Temperature scaling applied post-training
- **Betting edge detection** — Flags value bets where model probability exceeds implied bookmaker probability by > 5%
- **33 engineered features** — Form (last 5 & 10), head-to-head, FIFA rankings, venue, tournament stage, rest days

---

## Model Performance (Backtests)

| Tournament | Log-Loss | Accuracy | Brier Score | Flat-Stake ROI |
|---|---|---|---|---|
| WC 2018 (test) | 1.008 | 51.6% | 0.201 | +3.1% |
| WC 2022 (val) | 1.026 | 54.7% | 0.202 | +9.4% |

---

## Data Sources

| Source | Purpose |
|---|---|
| [Kaggle — martj42/international-football-results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2026) | Historical match results (49,000+ rows, 1872–present) |
| [Kaggle — cashncarry/fifaworldranking](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | FIFA rankings over time (70,000+ rows) |
| [football-data.org API](https://www.football-data.org/) | WC 2026 fixture schedule |
| [openfootball/world-cup](https://github.com/openfootball/world-cup) | Historical WC match data (1998–2022) |

---

## Local Setup

1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your `FD_API_KEY` from football-data.org.

## Pipeline

Run the full pipeline in order:

```bash
# 1. EDA & alignment checks
python scripts/run_eda.py
python scripts/run_alignment_check.py

# 2. Feature engineering
python scripts/run_feature_engineering.py

# 3. Baseline models
python scripts/run_baseline_models.py

# 4. Hyperparameter tuning (slow — 100+ Optuna trials)
python scripts/run_tuning.py

# 5. Full training run
python scripts/train.py

# 6. Backtest evaluation
python scripts/run_backtest.py

# 7. Generate predictions for WC 2026 fixtures
python scripts/predict.py
python scripts/precompute_predictions.py

# 8. Launch dashboard locally
streamlit run app/dashboard.py
```

## Tests

```bash
python -m pytest tests/ -v
```

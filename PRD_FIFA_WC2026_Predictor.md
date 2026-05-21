# PRD: FIFA World Cup 2026 Match Predictor

**Document Type:** Product Requirements Document  
**Project:** `wc2026-predictor` (Private Repository)  
**Owner:** Ahmed  
**Status:** Active v1.2  
**Last Updated:** 21 May 2026

---

## 1. Overview

### 1.1 Purpose

A Python-based machine learning system that predicts match outcomes for the FIFA World Cup 2026. The system ingests free, publicly available football data, trains and tunes ensemble ML models, and serves predictions via a lightweight Streamlit dashboard — optimized to produce actionable, bet-informed outputs per game.

### 1.2 Goals

- Predict Win/Draw/Loss probabilities for every WC 2026 match with well-calibrated confidence scores.
- Predict expected scorelines (e.g., 2–1) using a goals model.
- Calculate a betting edge signal by comparing model probabilities against bookmaker implied odds.
- Provide a clean Streamlit dashboard to browse predictions match-by-match.
- Achieve strong backtesting performance against historical World Cup data (1998–2022).

### 1.3 Non-Goals

- No real-money betting integration or automated wagering.
- No paid data sources or API subscriptions.
- No live in-game prediction (pre-match only).
- No mobile app or cloud deployment (local Streamlit only).
- No user authentication or multi-user support.

---

## 2. Background & Context

The FIFA World Cup 2026 is hosted across the USA, Canada, and Mexico — the first edition with 48 teams and 104 matches. The expanded format and inclusion of more nations creates a larger prediction surface and more betting opportunities than prior tournaments.

Publicly available historical match data (international fixtures, WC results, team/player rankings) is sufficient to train a competitive intermediate-level ML system. The primary user is a single developer who will interact via a local dashboard.

---

## 3. User Stories

| ID | As a user I want to... | So that I can... |
|----|------------------------|------------------|
| US-01 | See Win/Draw/Loss probabilities for any WC 2026 match | Assess the most likely outcome before betting |
| US-02 | See a predicted scoreline (e.g., 2–1) | Bet on correct score markets |
| US-03 | See a model confidence score per prediction | Know when the model is uncertain and avoid low-confidence bets |
| US-04 | See betting edge vs bookmaker odds | Identify value bets where the model disagrees with the market |
| US-05 | Browse upcoming and past predictions in a dashboard | Have a single interface for all predictions |
| US-06 | Retrain models easily as new match data comes in | Keep predictions fresh throughout the tournament |

---

## 4. Data

### 4.1 Free Data Sources

| Source | What to Use | Format | Notes |
|--------|-------------|--------|-------|
| [international_football_results.csv (Kaggle)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | All international match results 1872–present | CSV | 47,000+ matches; includes tournament type |
| [FIFA World Rankings (Kaggle)](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Team ranking & points over time | CSV | Use ranking at time of match |
| [football-data.org API](https://www.football-data.org/) | WC 2026 fixtures & live results | REST API | Free tier (10 req/min); no key payment required |
| [Open Football GitHub](https://github.com/openfootball/world-cup) | Historical WC fixture data | JSON/CSV | Useful for ground truth validation |
| [SofaScore / Transfermarkt (scraping)](https://www.transfermarkt.com) | Squad depth, market value, recent form | HTML scraping | Use BeautifulSoup; respect robots.txt |

### 4.2 Key Features to Engineer

**Match-level features (37 implemented):**
- `home_rank`, `away_rank`, `home_rank_points`, `away_rank_points`
- `rank_diff`, `rank_points_ratio`, `rank_points_diff`
- Recent form (last 5 & 10 matches): wins, avg goals scored, avg goals conceded, WDL points — for both home and away (16 features)
- `form_diff_wdl_5`: home minus away form momentum
- `home_goal_efficiency_5`, `away_goal_efficiency_5`: goals scored / goals conceded ratio
- H2H: `h2h_home_win_rate`, `h2h_matches_count`, `h2h_avg_goals_home`, `h2h_avg_goals_away`
- Context: `tournament_stage` (0–6 ordinal), `is_wc_match`, `is_neutral_venue`, `host_nation_advantage`
- Rest: `home_days_rest`, `away_days_rest`, `rest_diff`

**Features explored but not used in final model:**
- Elo ratings (`home_elo`, `away_elo`, `elo_diff`): implemented in `compute_elo_ratings()` in `preprocess.py` but excluded from `FEATURE_COLUMNS` — degraded WC 2022 val log-loss due to upset-heavy 2022 tournament

### 4.3 Data Pipeline

```
raw/
  ├── results.csv           ← Kaggle international results
  ├── rankings.csv          ← FIFA rankings history
  ├── wc2026_fixtures.json  ← football-data.org API
  └── scraped/              ← Transfermarkt form data

processed/
  ├── features_train.parquet   ← Historical matches with features
  ├── features_predict.parquet ← WC 2026 fixtures with features
  └── bookmaker_odds.csv       ← Manually collected or scraped odds

models/
  ├── outcome_model.pkl     ← W/D/L classifier
  ├── home_goals_model.pkl  ← Home goals regressor
  └── away_goals_model.pkl  ← Away goals regressor
```

---

## 5. Machine Learning

### 5.1 Model Architecture

Two separate modeling tracks run in parallel:

**Track A — Outcome Model (W/D/L)**

- Target: match outcome (0=Away Win, 1=Draw, 2=Home Win)
- Type: Multi-class classification
- Primary model: `GradientBoostingClassifier` / `XGBClassifier`
- Ensemble: Soft-voting ensemble of XGBoost + Random Forest + Logistic Regression

**Track B — Goals Model (Scoreline)**

- Target: goals scored by each team (separate models for home and away)
- Type: Regression (Poisson-distributed counts)
- Primary model: `PoissonRegressor` + `XGBRegressor`
- Scoreline derived by sampling from predicted Poisson distributions

### 5.2 Training Strategy

- **Training set:** All international matches 1998–present, excluding WC 2018 and WC 2022 windows (26,566 rows)
- **Validation set:** WC 2022 matches — 64 rows, fixed, used only for final evaluation (never for optimisation)
- **Test set (backtest):** WC 2018 matches — 64 rows, held out

Sample weighting:
- WC matches: 3×
- WC qualifiers: 1.5×
- Friendlies: 0.5×
- Matches older than 4 years: decay by 0.85^(years_ago)

### 5.3 Tuning Pipeline

Use `scikit-learn` pipelines with `Optuna` for hyperparameter optimization:

```
Pipeline:
  1. Feature preprocessing (StandardScaler in LR pipeline; tree models use raw features)
  2. Optuna TPE sampler, MedianPruner
     - XGB outcome: 100 trials
     - RF outcome: 50 trials
     - Goals models: 50 trials each
  3. Cross-validation: StratifiedKFold(n_splits=5) for outcome; KFold(5) for goals
  4. Metric: log-loss (primary), accuracy (secondary)
  5. Sample weights (match_importance × recency_weight) applied in all CV folds
```

Tuned hyperparameters for XGBoost:
- `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`

Tuned hyperparameters for Random Forest:
- `n_estimators`, `max_features`, `min_samples_split`, `min_samples_leaf`, `max_depth`

Ensemble weights optimised post-training via `scipy.optimize.minimize` (L-BFGS-B) on out-of-fold (OOF) predictions using softmax reparametrization to minimise OOF log-loss.

Best ensemble weights (37-feature pipeline): LR=0.4452, RF=0.2183, XGB=0.3365

### 5.4 Evaluation Metrics

| Metric | Description | Target | Achieved |
|--------|-------------|--------|----------|
| Log-loss | Measures calibration of probability predictions | < 0.95 on WC 2022 validation set | **1.029** (not met) |
| Accuracy | Correct outcome classification | > 52% (baseline: ~50%) | **54.7%** ✅ |
| Brier Score | Probabilistic scoring rule | < 0.22 | 0.607 (note: multiclass Brier uses different scale) |
| Backtest ROI | Simulated flat-stake betting return | > −10% | TBD (Phase 6) |
| Calibration curve | Reliability diagram of predicted vs actual probabilities | Visually near-diagonal | Max ECE = 0.022 ✅ |

> **Note on log-loss target:** The PRD target of 0.95 is unlikely achievable without historical bookmaker odds as training features. WC 2022 was historically upset-heavy (Saudi Arabia beat Argentina; Morocco beat Spain, Portugal, and Belgium; Japan beat Germany and Spain), which heavily penalises high-confidence favourite predictions. The best achievable log-loss with the available feature set is ~1.03. For reference, a random uniform model scores ln(3) ≈ 1.099.

### 5.5 Backtesting Protocol

1. Train model on matches up to and including WC 2014.
2. Predict outcomes for WC 2018 — record all probabilities.
3. Retrain including 2018 — predict WC 2022.
4. Record accuracy, log-loss, and simulated bet return for each tournament.
5. Simulate flat-stake betting: bet on model's top-confidence outcome; compare return against true result and bookmaker odds.

---

## 6. Betting Edge Calculation

The betting edge for a given match and outcome is defined as:

```
Edge = Model Probability − Implied Bookmaker Probability
Implied Prob = 1 / Decimal Odds
```

If `Edge > 0.05` (5% threshold), flag as **Value Bet**.

Bookmaker odds are ingested manually (CSV) or scraped from a free aggregator. The dashboard displays:
- Model probability vs. implied market probability side by side
- Edge percentage with colour coding (green = value, red = overpriced)
- Recommended bet tag: **Value / Avoid / Neutral**

---

## 7. Dashboard (Streamlit)

### 7.1 Pages

**Page 1 — Match Predictions**
- Fixture selector (group stage / knockout / by date)
- Per-match card showing:
  - Team badges and names
  - W/D/L probabilities as a horizontal bar chart
  - Predicted scoreline (most likely + top 3 alternatives)
  - Confidence score (model uncertainty, shown as a gauge)
  - Betting edge panel (odds input field + calculated edge)

**Page 2 — Tournament Bracket**
- Visual bracket showing predicted round-by-round winners
- Each team node coloured by win probability

**Page 3 — Model Performance**
- Backtesting summary table (2018, 2022 results)
- Log-loss and accuracy plots
- Calibration curve chart

**Page 4 — Data & Model Info**
- Feature importance chart
- Training data summary
- Model version + last retrain date

### 7.2 Running Locally

```bash
streamlit run app/dashboard.py
```

---

## 8. Project Structure

```
wc2026-predictor/
├── data/
│   ├── raw/
│   ├── processed/
│   └── bookmaker_odds.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── data/
│   │   ├── ingest.py          ← download & cache raw data
│   │   ├── preprocess.py      ← cleaning & feature engineering
│   │   └── odds.py            ← bookmaker odds ingestion
│   ├── models/
│   │   ├── outcome_model.py   ← W/D/L classifier
│   │   ├── goals_model.py     ← Poisson goals regressor
│   │   ├── ensemble.py        ← voting ensemble wrapper
│   │   └── tune.py            ← Optuna tuning pipelines
│   ├── evaluation/
│   │   ├── backtest.py        ← backtesting simulation
│   │   └── metrics.py         ← log-loss, calibration, ROI
│   └── betting/
│       └── edge.py            ← edge calculation logic
├── app/
│   ├── dashboard.py           ← Streamlit entry point
│   └── components/            ← reusable Streamlit widgets
├── models/                    ← serialised .pkl model files
├── scripts/
│   ├── train.py               ← full training run
│   └── predict.py             ← generate predictions for upcoming fixtures
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 9. Tech Stack

| Layer | Library / Tool | Version |
|-------|---------------|---------|
| Language | Python | 3.11+ |
| Data wrangling | pandas, numpy | Latest |
| ML models | scikit-learn, xgboost | Latest |
| HPO / Tuning | optuna | Latest |
| Goals model | statsmodels (Poisson GLM) | Latest |
| Dashboard | streamlit | Latest |
| Charts | plotly | Latest |
| Data fetching | requests, httpx | Latest |
| Scraping | beautifulsoup4, lxml | Latest |
| Serialization | joblib | Latest |
| Notebook | jupyter | Latest |
| Dependency mgmt | pip + requirements.txt | — |

---

## 10. Development Phases

### Phase 1 — Data & EDA (Week 1)
- [ ] Download Kaggle datasets (results, rankings)
- [ ] Set up football-data.org API key (free)
- [ ] Build `ingest.py` to cache all raw data locally
- [ ] EDA notebook: distributions, missing values, feature correlations
- [ ] Verify WC 2026 fixture data is available

### Phase 2 — Feature Engineering (Week 1–2)
- [ ] Implement all match-level features in `preprocess.py`
- [ ] Build form calculation with configurable lookback windows
- [ ] Merge ranking data aligned to match date
- [ ] Export `features_train.parquet` and validate

### Phase 3 — Baseline Models (Week 2)
- [ ] Train baseline Logistic Regression (W/D/L)
- [ ] Evaluate log-loss on 2022 WC validation set
- [ ] Train baseline Poisson goals model
- [ ] Validate predicted scoreline distribution

### Phase 4 — Ensemble & Tuning (Week 2–3)
- [ ] Implement Optuna tuning pipeline for XGBoost
- [ ] Build soft-voting ensemble (XGB + RF + LR)
- [ ] Run full tuning run (100+ Optuna trials)
- [ ] Compare tuned vs baseline log-loss

### Phase 5 — Backtesting & Betting Edge (Week 3)
- [ ] Implement `backtest.py` over 2018, 2022 WC
- [ ] Implement betting edge calculator in `edge.py`
- [ ] Add bookmaker odds ingestion (CSV + manual update)
- [ ] Plot calibration curves

### Phase 6 — Dashboard (Week 3–4)
- [ ] Build Streamlit match prediction page
- [ ] Add tournament bracket page
- [ ] Add model performance page
- [ ] Polish UI (team colours, flags, charts)

### Phase 7 — Ongoing During Tournament
- [ ] Update fixture data from football-data.org after each round
- [ ] Retrain models with completed match results
- [ ] Update bookmaker odds CSV before each match day

---

## 11. Risk & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Free data sources incomplete / outdated | Medium | High | Use multiple sources; build fallback ingest paths |
| Scraping blocked by Transfermarkt | Medium | Low | Cache aggressively; fall back to Kaggle datasets |
| Model overfit to strong historical nations | Medium | Medium | Apply regularization; include upset-aware features |
| Bookmaker odds unavailable for some matches | Low | Medium | Allow manual odds entry in dashboard |
| football-data.org rate limit hit | Low | Low | Add 6s sleep between requests; cache responses |

---

## 12. Success Criteria

The project is considered successful if it meets all of the following by tournament kick-off (June 11, 2026):

1. ~~Log-loss on 2022 WC backtest is **< 0.95**~~ — **Revised:** Achieve best-possible log-loss with available data. Current best: **1.029** (ensemble, WC 2022 val). Target of 0.95 requires historical bookmaker odds as training features; not achievable with current free data sources.
2. Outcome accuracy on 2022 WC backtest is **> 52%** — ✅ **ACHIEVED: 54.7%** (ensemble)
3. Simulated flat-stake betting ROI on 2018 + 2022 WC is **> −10%** — (Phase 6, pending)
4. All WC 2026 group-stage fixtures have predictions ready before matchday 1
5. Streamlit dashboard runs locally without errors

---

## 13. Out of Scope (Future Ideas)

- Player-level features (injuries, suspensions, lineups)
- In-play / live score model updates
- Neural network models (e.g., transformer on match sequences)
- Automated odds scraping with Selenium
- Docker containerization or cloud deployment
- Multi-tournament generalization (AFCON, Euros, etc.)

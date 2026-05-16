# FIFA World Cup 2026 Match Predictor

A machine learning system that predicts FIFA World Cup 2026 match outcomes and scorelines, combining historical international football results, FIFA rankings, and bookmaker odds to generate calibrated probability estimates for home win, draw, and away win outcomes.

## Setup

1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in your API keys.

## Usage

Run data ingestion:
```bash
python scripts/run_eda.py
```

Train models:
```bash
python scripts/train.py
```

Generate predictions:
```bash
python scripts/predict.py
```

Launch dashboard:
```bash
streamlit run app/dashboard.py
```

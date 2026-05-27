"""Subphase 7.6 End-to-End Dashboard Verification Script.

Validates all resources, data schemas, model behaviors, and computation logic
used by app/dashboard.py without running a Streamlit server. Prints PASS/FAIL
for each checklist item and exits with code 1 if any check fails.

Usage:
    python scripts/verify_dashboard.py
"""

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from unittest.mock import MagicMock

# ── Bootstrap ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Must mock streamlit before any component import that does `import streamlit as st`
# All st.xxx() calls are inside functions, not at module level, so this is safe.
sys.modules["streamlit"] = MagicMock()

import joblib                                                     # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy.stats import entropy                                  # noqa: E402

from src.models.ensemble import WC2026Ensemble                   # noqa: E402
from src.data.preprocess import FEATURE_COLUMNS as CANONICAL_FC  # noqa: E402

# ── Path constants ────────────────────────────────────────────────────────────
_MODELS = REPO_ROOT / "models"
_PROC   = REPO_ROOT / "data" / "processed"
_RAW    = REPO_ROOT / "data" / "raw"

# ── Result tracking ───────────────────────────────────────────────────────────
_pass_count = 0
_fail_count = 0


def run(label: str, fn) -> bool:
    """Execute fn(); print PASS or FAIL with label. Returns True on pass."""
    global _pass_count, _fail_count
    try:
        fn()
        print(f"  PASS  {label}")
        _pass_count += 1
        return True
    except Exception as exc:
        print(f"  FAIL  {label}")
        print(f"        {type(exc).__name__}: {exc}")
        _fail_count += 1
        return False


def _require(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _require_file(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(str(p))


def _import_component(module_name: str, rel_path: str):
    """Import a component module by file path (avoids app/__init__.py requirement)."""
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Section 1 — Required Files
# =============================================================================
print("\n── 1. Required files ───────────────────────────────────────────────")

REQUIRED_FILES = [
    _MODELS / "outcome_lr.pkl",
    _MODELS / "outcome_rf.pkl",
    _MODELS / "outcome_xgb.pkl",
    _MODELS / "home_goals_xgb.pkl",
    _MODELS / "away_goals_xgb.pkl",
    _MODELS / "home_goals_poisson.pkl",
    _MODELS / "away_goals_poisson.pkl",
    _MODELS / "MODEL_REGISTRY.md",
    _PROC / "features_predict.parquet",
    _PROC / "features_train.parquet",
    _PROC / "final_backtest_metrics.json",
    _PROC / "backtest_wc2018.csv",
    _PROC / "backtest_wc2022.csv",
    _RAW  / "wc2026_fixtures_flat.csv",
    _RAW  / "rankings.csv",
    _RAW  / "team_name_map.csv",
    REPO_ROOT / "data" / "bookmaker_odds.csv",
]

for p in REQUIRED_FILES:
    run(f"exists: {p.relative_to(REPO_ROOT)}", lambda _p=p: _require_file(_p))


# =============================================================================
# Section 2 — Model Loading
# =============================================================================
print("\n── 2. Model loading ────────────────────────────────────────────────")

_models: dict = {}

for name in [
    "outcome_lr", "outcome_rf", "outcome_xgb",
    "home_goals_xgb", "away_goals_xgb",
    "home_goals_poisson", "away_goals_poisson",
]:
    def _load(n=name):
        _models[n] = joblib.load(_MODELS / f"{n}.pkl")
    run(f"joblib.load: {name}.pkl", _load)


# =============================================================================
# Section 3 — WC2026Ensemble Construction
# =============================================================================
print("\n── 3. WC2026Ensemble ───────────────────────────────────────────────")

_ensemble = None


def _build_ensemble():
    global _ensemble
    _ensemble = WC2026Ensemble(
        _models["outcome_lr"],
        _models["outcome_rf"],
        _models["outcome_xgb"],
    )


run("WC2026Ensemble initialises (3 models, no weights arg)", _build_ensemble)

# Test that predict_proba works on a dummy row
def _ensemble_proba_shape():
    _require(_ensemble is not None, "ensemble not built")
    # Use features_predict row 0 as a test input
    fp = pd.read_parquet(_PROC / "features_predict.parquet")
    X = fp.iloc[[0]][CANONICAL_FC]
    proba = _ensemble.predict_proba(X)
    _require(proba.shape == (1, 3), f"expected shape (1,3), got {proba.shape}")
    _require(abs(proba[0].sum() - 1.0) < 1e-6, f"proba sum={proba[0].sum()}")

run("WC2026Ensemble.predict_proba returns shape (1,3) summing to 1.0", _ensemble_proba_shape)


# =============================================================================
# Section 4 — Data File Schemas
# =============================================================================
print("\n── 4. Data file schemas ────────────────────────────────────────────")

_fixtures = None
_fp = None
_ft = None
_metrics = None
_bt18 = None
_bt22 = None


def _load_fixtures():
    global _fixtures
    _fixtures = pd.read_csv(_RAW / "wc2026_fixtures_flat.csv", parse_dates=["match_date"])

run("fixtures_flat.csv loads cleanly", _load_fixtures)

run(
    "fixtures_flat.csv has required columns",
    lambda: _require(
        {"match_date", "home_team", "away_team", "stage", "status"}.issubset(_fixtures.columns),
        f"columns={_fixtures.columns.tolist()}",
    ),
)
run(
    "fixtures_flat.csv has 104 rows",
    lambda: _require(len(_fixtures) == 104, f"got {len(_fixtures)}"),
)
run(
    "fixtures_flat.csv has 72 GROUP_STAGE rows",
    lambda: _require(
        (_fixtures["stage"] == "GROUP_STAGE").sum() == 72,
        f"got {(_fixtures['stage'] == 'GROUP_STAGE').sum()}",
    ),
)


def _load_fp():
    global _fp
    _fp = pd.read_parquet(_PROC / "features_predict.parquet")

run("features_predict.parquet loads cleanly", _load_fp)
run(
    "features_predict.parquet has 104 rows",
    lambda: _require(len(_fp) == 104, f"got {len(_fp)}"),
)
run(
    "features_predict.parquet contains all FEATURE_COLUMNS",
    lambda: _require(
        all(c in _fp.columns for c in CANONICAL_FC),
        f"missing: {[c for c in CANONICAL_FC if c not in _fp.columns]}",
    ),
)
run(
    "features_predict row count == fixtures row count (positional alignment)",
    lambda: _require(
        len(_fixtures) == len(_fp),
        f"fixtures={len(_fixtures)}, features_predict={len(_fp)}",
    ),
)


def _load_ft():
    global _ft
    _ft = pd.read_parquet(_PROC / "features_train.parquet")

run("features_train.parquet loads cleanly", _load_ft)
run(
    "features_train.parquet has date, tournament, outcome columns",
    lambda: _require(
        {"date", "tournament", "outcome"}.issubset(_ft.columns),
        "missing: " + str({'date', 'tournament', 'outcome'} - set(_ft.columns)),
    ),
)
run(
    "features_train.parquet contains all FEATURE_COLUMNS",
    lambda: _require(
        all(c in _ft.columns for c in CANONICAL_FC),
        f"missing: {[c for c in CANONICAL_FC if c not in _ft.columns]}",
    ),
)


def _load_metrics():
    global _metrics
    with open(_PROC / "final_backtest_metrics.json", encoding="utf-8") as f:
        _metrics = json.load(f)

run("final_backtest_metrics.json loads cleanly", _load_metrics)
run(
    "final_backtest_metrics.json has wc2018 and wc2022 keys",
    lambda: _require(
        {"wc2018", "wc2022"}.issubset(_metrics.keys()),
        f"keys={list(_metrics.keys())}",
    ),
)
run(
    "final_backtest_metrics.json each tournament has required sub-keys",
    lambda: _require(
        all(
            k in _metrics[t]
            for t in ("wc2018", "wc2022")
            for k in ("log_loss", "accuracy", "brier_score", "flat_stake_roi", "value_bet_roi")
        ),
        f"wc2018={list(_metrics.get('wc2018', {}).keys())}",
    ),
)


def _load_bt():
    global _bt18, _bt22
    _bt18 = pd.read_csv(_PROC / "backtest_wc2018.csv")
    _bt22 = pd.read_csv(_PROC / "backtest_wc2022.csv")

run("backtest_wc2018.csv and backtest_wc2022.csv load cleanly", _load_bt)
run(
    "backtest CSVs both have cumulative_profit column",
    lambda: _require(
        "cumulative_profit" in _bt18.columns and "cumulative_profit" in _bt22.columns,
        "cumulative_profit column missing",
    ),
)
run(
    "bookmaker_odds.csv loads cleanly",
    lambda: pd.read_csv(REPO_ROOT / "data" / "bookmaker_odds.csv"),
)


# =============================================================================
# Section 5 — FEATURE_COLUMNS Sync Check
# =============================================================================
print("\n── 5. FEATURE_COLUMNS sync ─────────────────────────────────────────")

_pc_mod = None
_br_mod = None


def _import_prediction_card():
    global _pc_mod
    _pc_mod = _import_component("prediction_card", "app/components/prediction_card.py")

run("prediction_card.py imports cleanly (streamlit mocked)", _import_prediction_card)

run(
    "prediction_card.FEATURE_COLUMNS matches preprocess.FEATURE_COLUMNS",
    lambda: _require(
        _pc_mod.FEATURE_COLUMNS == CANONICAL_FC,
        f"len(pc)={len(_pc_mod.FEATURE_COLUMNS)} len(canonical)={len(CANONICAL_FC)}",
    ),
)


def _import_bracket():
    global _br_mod
    _br_mod = _import_component("bracket", "app/components/bracket.py")

run("bracket.py imports cleanly (streamlit mocked)", _import_bracket)

run(
    "bracket.FEATURE_COLUMNS matches preprocess.FEATURE_COLUMNS",
    lambda: _require(
        _br_mod.FEATURE_COLUMNS == CANONICAL_FC,
        f"diff: {set(_br_mod.FEATURE_COLUMNS) ^ set(CANONICAL_FC)}",
    ),
)

run(
    "performance_charts.py imports cleanly (streamlit mocked)",
    lambda: _import_component("performance_charts", "app/components/performance_charts.py"),
)

run(
    "model_info.py imports cleanly (streamlit mocked)",
    lambda: _import_component("model_info", "app/components/model_info.py"),
)


# =============================================================================
# Section 6 — Match Predictions Page Logic
# =============================================================================
print("\n── 6. Match Predictions page logic ────────────────────────────────")

_sample_fixture = None
_sample_features = None
_proba = None


def _extract_sample():
    global _sample_fixture, _sample_features
    group_rows = _fixtures[_fixtures["stage"] == "GROUP_STAGE"]
    _sample_fixture = group_rows.iloc[0]
    fixture_idx = _sample_fixture.name
    _require(
        0 <= fixture_idx < len(_fp),
        f"fixture_idx={fixture_idx} out of range for features_predict ({len(_fp)} rows)",
    )
    _sample_features = _fp.iloc[fixture_idx]

run("sample fixture + features row extracted via positional index", _extract_sample)


def _run_predict_proba():
    global _proba
    X = pd.DataFrame(
        _sample_features[CANONICAL_FC].values.reshape(1, -1),
        columns=CANONICAL_FC,
    )
    _proba = _ensemble.predict_proba(X)[0]

run("ensemble.predict_proba(X) returns array of shape (3,)", _run_predict_proba)

run(
    "ensemble.predict_proba output sums to 1.0",
    lambda: _require(abs(_proba.sum() - 1.0) < 1e-6, f"sum={_proba.sum()}"),
)
run(
    "ensemble.predict_proba values all in [0, 1]",
    lambda: _require(
        bool(np.all(_proba >= 0) and np.all(_proba <= 1)),
        f"proba={_proba}",
    ),
)


def _run_goals_models():
    X = pd.DataFrame(
        _sample_features[CANONICAL_FC].values.reshape(1, -1),
        columns=CANONICAL_FC,
    )
    h = float(_models["home_goals_xgb"].predict(X)[0])
    a = float(_models["away_goals_xgb"].predict(X)[0])
    _require(h >= 0, f"home_goals prediction is negative: {h}")
    _require(a >= 0, f"away_goals prediction is negative: {a}")

run("home_goals_xgb and away_goals_xgb predict non-negative values", _run_goals_models)


def _compute_confidence():
    probs_array = np.clip(_proba, 1e-9, 1.0)
    raw_entropy = entropy(probs_array)
    confidence = float(np.clip(1.0 - raw_entropy / np.log(3), 0.0, 1.0))
    _require(0.0 <= confidence <= 1.0, f"confidence={confidence}")

run("confidence score computation yields value in [0, 1]", _compute_confidence)


def _check_no_width_stretch():
    src = (REPO_ROOT / "app" / "components" / "prediction_card.py").read_text(encoding="utf-8")
    _require(
        "width='stretch'" not in src,
        "prediction_card.py still contains width='stretch' — bug not fixed",
    )

run("prediction_card.py does NOT contain width='stretch'", _check_no_width_stretch)

run(
    "prediction_card.py uses use_container_width=True for plotly_chart",
    lambda: _require(
        "use_container_width=True" in (
            REPO_ROOT / "app" / "components" / "prediction_card.py"
        ).read_text(encoding="utf-8"),
        "use_container_width=True not found in prediction_card.py",
    ),
)


# =============================================================================
# Section 7 — Tournament Bracket Page Logic
# =============================================================================
print("\n── 7. Tournament Bracket page logic ────────────────────────────────")


def _test_rank_lookup():
    lookup = _br_mod._build_rank_lookup()
    _require(isinstance(lookup, dict), f"type={type(lookup)}")
    _require(len(lookup) > 0, "rank_lookup is empty")

run("_build_rank_lookup() returns non-empty dict", _test_rank_lookup)


def _test_identify_groups():
    groups = _br_mod._identify_groups(_fixtures)
    _require(len(groups) == 12, f"expected 12 groups, got {len(groups)}")
    for g in groups:
        _require(len(g) == 4, f"group {g[0]!r} has {len(g)} teams")

run("_identify_groups() returns 12 groups of 4 teams", _test_identify_groups)


_bracket_tree = None


def _full_bracket_simulation():
    global _bracket_tree
    rank_lookup = _br_mod._build_rank_lookup()
    groups = _br_mod._identify_groups(_fixtures)
    standings = _br_mod._simulate_group_stage(_fixtures, _fp, _ensemble, groups, rank_lookup)
    _require(len(standings) == 12, f"expected 12 standings, got {len(standings)}")
    qualifiers = _br_mod._select_qualifiers(standings, rank_lookup)
    _require(len(qualifiers) == 32, f"expected 32 qualifiers, got {len(qualifiers)}")
    _bracket_tree = _br_mod._build_bracket_tree(qualifiers, rank_lookup)
    _require("FINAL" in _bracket_tree, "FINAL round missing")
    _require(len(_bracket_tree["FINAL"]) == 1, "FINAL should have 1 match")
    _require("THIRD_PLACE" in _bracket_tree, "THIRD_PLACE round missing")
    winner = _bracket_tree["FINAL"][0]["winner"]
    _require(isinstance(winner, str) and len(winner) > 0, "FINAL winner is empty")

run("full bracket simulation (group → qualifiers → knockout → FINAL)", _full_bracket_simulation)

run(
    "bracket FINAL has a non-empty winner string",
    lambda: _require(
        _bracket_tree is not None and isinstance(_bracket_tree["FINAL"][0]["winner"], str),
        "FINAL winner not set",
    ),
)


# =============================================================================
# Section 8 — Model Performance Page Logic
# =============================================================================
print("\n── 8. Model Performance page logic ─────────────────────────────────")

run(
    "combined log_loss computation is in valid range",
    lambda: _require(
        0.0 < (_metrics["wc2018"]["log_loss"] + _metrics["wc2022"]["log_loss"]) / 2 < 10.0,
        "combined log_loss out of expected range",
    ),
)
run(
    "combined accuracy computation is in valid range [0, 1]",
    lambda: _require(
        0.0 < (_metrics["wc2018"]["accuracy"] + _metrics["wc2022"]["accuracy"]) / 2 < 1.0,
        "combined accuracy out of expected range",
    ),
)
run(
    "backtest_wc2018.csv has at least 60 rows",
    lambda: _require(len(_bt18) >= 60, f"got {len(_bt18)} rows"),
)
run(
    "backtest_wc2022.csv has at least 60 rows",
    lambda: _require(len(_bt22) >= 60, f"got {len(_bt22)} rows"),
)
run(
    "outputs/plots/ directory exists",
    lambda: _require(
        (REPO_ROOT / "outputs" / "plots").is_dir(),
        "outputs/plots/ directory not found",
    ),
)
run(
    "calibration_rf_tuned.png exists in outputs/plots/",
    lambda: _require(
        (REPO_ROOT / "outputs" / "plots" / "calibration_rf_tuned.png").exists(),
        "calibration_rf_tuned.png not found",
    ),
)
run(
    "calibration_ensemble.png exists in outputs/plots/",
    lambda: _require(
        (REPO_ROOT / "outputs" / "plots" / "calibration_ensemble.png").exists(),
        "calibration_ensemble.png not found",
    ),
)


# =============================================================================
# Section 9 — Data & Model Info Page Logic
# =============================================================================
print("\n── 9. Data & Model Info page logic ─────────────────────────────────")

run(
    "outcome_xgb.feature_importances_ has length == len(FEATURE_COLUMNS)",
    lambda: _require(
        hasattr(_models["outcome_xgb"], "feature_importances_")
        and len(_models["outcome_xgb"].feature_importances_) == len(CANONICAL_FC),
        f"got {len(_models['outcome_xgb'].feature_importances_)} vs {len(CANONICAL_FC)}",
    ),
)


def _test_feature_importance_series():
    importances = pd.Series(
        _models["outcome_xgb"].feature_importances_,
        index=CANONICAL_FC,
    )
    top20 = importances.sort_values(ascending=False).head(20)
    _require(len(top20) == 20, f"expected 20, got {len(top20)}")

run("feature_importances_ -> pd.Series -> head(20) produces 20 entries",
    _test_feature_importance_series)

run(
    "features_train date column parseable as datetime",
    lambda: _require(pd.to_datetime(_ft["date"]).notna().all(), "date column has NaT"),
)
run(
    "features_train tournament column is non-empty with multiple values",
    lambda: _require(
        _ft["tournament"].notna().all() and len(_ft["tournament"].unique()) > 1,
        f"unique tournaments={_ft['tournament'].nunique()}",
    ),
)


def _test_registry_parsing():
    import re
    content = (_MODELS / "MODEL_REGISTRY.md").read_text(encoding="utf-8")
    dates = re.findall(r"\|\s*(\d{4}-\d{2}-\d{2})\s*\|", content)
    _require(len(dates) > 0, "no dates found in MODEL_REGISTRY.md")
    last_retrained = max(dates)
    _require(len(last_retrained) == 10, f"unexpected date format: {last_retrained!r}")

run("MODEL_REGISTRY.md can be parsed for a date", _test_registry_parsing)


# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*60}")
print(f"  TOTAL: {_pass_count + _fail_count} checks "
      f"-- {_pass_count} PASS, {_fail_count} FAIL")
print(f"{'='*60}\n")

sys.exit(0 if _fail_count == 0 else 1)

# FPL AI Predictions - Copilot Instructions

## Project Overview

**FPLyzer** is a Fantasy Premier League AI prediction platform deployed as a Flask web app. It uses ML predictions (CSV-based) for team optimization, transfer suggestions, wildcard planning, and AI-powered team reviews. Users authenticate via GCP BigQuery; the app is hosted on **Google Cloud Run** with Gunicorn.

**Tech Stack:**
- Flask + Gunicorn on Google Cloud Run (`Procfile`: 1 worker, 8 threads)
- GCP BigQuery for user data and pre-computed stats storage
- Redis Cloud for caching (in-memory dict fallback when Redis unavailable)
- ML predictions in CSV: `Data/Predictions/{season}/predictions_{season}.csv`
- PuLP for linear optimization; Google Gemini (`google-genai`) for AI reviews
- Flask-Limiter for rate limiting; Flask-WTF for CSRF protection
- Flask-Mail via Gmail SMTP for email verification and password reset

## Running Locally

```bash
# Always activate the venv first
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

# Run Flask app (from repo root)
python src/fpl_web_app.py     # http://localhost:5000
```

Required `.env` variables (see `.env.example`):
- `FLASK_SECRET_KEY` — required; random fallback is used in local dev only
- `GOOGLE_AI_API_KEY` — for AI team review feature
- `MAIL_USERNAME`, `MAIL_PASSWORD`, `MAIL_DEFAULT_SENDER`, `APP_URL` — for email flows
- `REDIS_URL` — optional; falls back to in-memory cache
- `ADMIN_DEV_PASSWORD` (dev) or `ADMIN_PASSWORD_HASH` (prod) — for `/admin`

## Deployment

Primary deployment is **Google Cloud Run** (`K_SERVICE` env var is set automatically).  
`Procfile`: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --chdir src fpl_web_app:app`

Legacy Ubuntu server also supported via `config/fpl_web_app.service` (systemd + unix socket).

```bash
# Systemd (legacy server)
sudo systemctl restart fpl_web_app
sudo journalctl -u fpl_web_app -f
```

## Architecture

### Predictions CSV — Source of Truth

All optimization modules read from `Data/Predictions/2025_26/predictions_2025_26.csv`.

Key columns: `id`, `web_name`, `team`, `position`, `GW`, `predicted_next_points`, `confidence_score`, `next_availability`, `next_status`, `current_price`, `actual_points`

**GW Offset**: To get GW N predictions, filter on `GW == N-1` — the `predicted_next_points` column holds GW N values.

```python
gw_data = df[df['GW'] == target_gw - 1]
# gw_data['predicted_next_points'] → predictions for target_gw
```

### Caching Layer

`cache_get`/`cache_set` in `fpl_web_app.py` transparently use Redis or fall back to a module-level `_memory_cache` dict. Always use these helpers — never call `_redis_client` directly. FPL bootstrap-static is cached for 5 minutes under key `fpl:bootstrap_static`.

### BigQuery Tables

| Dataset | Table | Purpose |
|---------|-------|---------|
| `fpl_users` | `users` | User accounts (auth, subscription) |
| `fpl_users` | `user_auth` | Email verification & password reset tokens |
| `fpl_users` | `invite_codes` | Invite-code gating for registration |
| `fpl_users` | `feature_usage` | Per-user feature usage tracking |
| `fpl_data` | `player_weekly_stats` | Pre-computed player stats (from `WeeklyStatsPrecomputer`) |

`users` schema: `id`, `username`, `email`, `team_id`, `password_hash`, `subscription_type` ('free'/'premium'), timestamps, `email_verified`, `email_opted_in`, `unsubscribe_token`

### Authentication & Sessions

- All routes except `/`, `/login`, `/register`, `/terms`, `/forgot-password`, `/reset-password/*`, `/verify-email/*`, `/unsubscribe/*` require auth via `@app.before_request`
- `session` stores: `user_id`, `username`, `team_id`, `subscription_type`
- Password requirements: ≥12 chars, uppercase, lowercase, digit, symbol, no username/email substrings
- New users must verify email before logging in (token stored in `user_auth` table)
- Invite-code gating is admin-toggleable at runtime (cached in Redis)
- Admin session uses `ADMIN_SESSION_KEY = 'is_admin'`; use `@admin_required` decorator for admin routes; supports user impersonation via `session['is_impersonating']`

### Rate Limiting

Routes use `@limiter.limit(...)` decorators. Public crawler-accessible pages use `@limiter.exempt`. The limiter is backed by Redis when available.

### Feature Gating (Free vs Premium)

`subscription_type` in session controls access. Free users get `/free-analysis` (unauthenticated, limited) and read-only dashboard. Premium users get full optimization, wildcard, transfer planner, AI review, and hidden gems.

### Free Hit / Chip Handling

`determine_squad_gameweek(team_id, target_gw)` returns `(effective_squad_gw, free_hit_used)`. If Free Hit was active, it falls back to the prior GW's squad to avoid recommending "transfers" from the temporary Free Hit team.

## Modules

| Module | Class/Entry | Purpose |
|--------|-------------|---------|
| `fpl_web_app.py` | Flask app | Routes, auth, caching, feature gating |
| `config.py` | constants | All paths, mappings, constants |
| `TeamOptimizer.py` | `FPLTeamOptimizer` | N-transfer team optimization + captain |
| `TransferAssistant.py` | `FPLTransferAssistant`, `CSVPredictionLoader` | Transfer suggestions (V3 algorithm) |
| `WildCardOptimizer.py` | `WildCardOptimizer` | PuLP budget-constrained full team build |
| `LLMTeamReviewer.py` | `LLMTeamReviewer` | Gemini AI team review with Search grounding |
| `EmailService.py` | `init_mail`, helpers | Email verification, password reset, GW reminders |
| `WeeklyPicksGenerator.py` | script | Generates `static/weekly_picks.html` |
| `HiddenGemsGenerator.py` | script | Generates `static/hidden_gems.html` (differentials) |
| `AvailabilityReporter.py` | script | Generates `static/availability_report.html` |
| `SeasonCalendarGenerator.py` | script | Generates `static/season_calendar.json` for frontend |
| `WeeklyStatsPrecomputer.py` | script | Bulk-inserts player stats to BigQuery weekly |
| `TeamDataReporter.py` | helpers | Fetches current squad from FPL API |

**Generator scripts** (run manually before each GW):
```bash
python src/WeeklyPicksGenerator.py
python src/HiddenGemsGenerator.py
python src/AvailabilityReporter.py
python src/SeasonCalendarGenerator.py
python src/WeeklyStatsPrecomputer.py  # needs BigQuery write access
```

## Key Conventions

### Availability Score
```python
# 0.0 = never select, 0.01–0.99 = reduced priority, 2.0 = fully available
def calculate_availability_score(player):
    availability = player.get('next_availability', 100)
    status = player.get('next_status', 'a')
    if status == 'u' or availability == 0:
        return 0.0
    elif status in ['i', 'd'] or availability < 100:
        return max(0.01, min(0.99, availability / 100))
    return 2.0
```

`next_status` values: `'a'` available, `'i'` injured, `'u'` unavailable, `'d'` doubtful, `'s'` suspended. Default min threshold: **75%**.

### Transfer Value Scoring (V3)
```python
import math
cost_penalty = 0.5 * math.log(1 + transfer_cost)
form_penalty = get_form_penalty(out_player)  # Protects in-form players
value_score = points_gain - cost_penalty - form_penalty
# Prioritises absolute points gain over cost efficiency
```

### Budget Units
Budget is stored in **0.1M increments**: £10.0M = 100 units, £100.0M = 1000 units (`BUDGET_LIMIT`). `current_price` in CSV is also in these units.

### Jersey Images
```python
team_id = TEAM_NAME_TO_CODE.get(player['team'], 1)  # Default Arsenal=1
url = f"/static/images/Jersey/{team_id}G.png" if position == 'GK' else f"/static/images/Jersey/{team_id}.png"
```

### FPL API Endpoints (from `config.py`)
```
Bootstrap:   GET https://fantasy.premierleague.com/api/bootstrap-static/
Fixtures:    GET https://fantasy.premierleague.com/api/fixtures/
Live GW:     GET https://fantasy.premierleague.com/api/event/{gw}/live/
Team picks:  GET https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/
Team entry:  GET https://fantasy.premierleague.com/api/entry/{team_id}/
```
Always handle 404 for invalid team IDs. Use `get_cached_bootstrap_data()` instead of calling the API directly (5-min cache).

### HTML Report Styling
- Colors: `#38003C` (purple bg), `#E90052` (pink), `#00FF87` (green)
- Font: Montserrat (Google Fonts)
- Pitch layout: GK `top:80%`, DEF `60%`, MID `40%`, FWD `15%`
- Captain badge: `<div class="captain-badge">C</div>`
- Templates use Jinja2 (`templates/`); partials in `templates/partials/`; email templates in `templates/emails/`

### Gameweek Finished Check
```python
gw_finished = pd.notna(player['actual_points'])
```

### Coding Standards
- Max line length: 100 characters
- Google-style docstrings; module-level docstrings required
- F-strings for formatting; `Path` objects for file paths
- JSON API responses always include `success` (bool) and `message` fields
- Catch specific exceptions — no bare `except:`

## Anti-Patterns

❌ Don't load `.pkl` ML model files — use CSV predictions only  
❌ Don't call FPL API for availability — use CSV `next_availability`/`next_status`  
❌ Don't forget the GW offset: GW N predictions come from GW N-1 rows  
❌ Don't assume availability columns are non-null (check for NaN)  
❌ Don't hardcode team IDs — use `TEAM_NAME_TO_CODE` from `config.py`  
❌ Don't bypass `@app.before_request` auth checks for protected routes  
❌ Don't use simple pts/cost efficiency for transfers — use V3 value scoring  
❌ Don't call `_redis_client` directly — use `cache_get`/`cache_set` helpers  
❌ Don't store the `FLASK_SECRET_KEY` or credentials in source code  

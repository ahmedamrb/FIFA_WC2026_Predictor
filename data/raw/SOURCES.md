# Data Sources

## International Football Results

- **Dataset name:** International Football Results from 1872 to 2026
- **Kaggle dataset ID:** `martj42/international-football-results-from-1872-to-2026`
- **Kaggle URL:** https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2026
- **Local file:** `data/raw/results.csv`
- **Download date:** 2026-05-17
- **Row count:** 49,287
- **Latest match date in dataset:** 2026-06-27 (dataset includes pre-scheduled future fixtures)
- **Columns (9):** `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `city`, `country`, `neutral`

---

## FIFA World Rankings

- **Dataset name:** FIFA World Ranking 1992-2024
- **Kaggle dataset ID:** `cashncarry/fifaworldranking`
- **Kaggle URL:** https://www.kaggle.com/datasets/cashncarry/fifaworldranking
- **Local file:** `data/raw/rankings.csv`
- **Download date:** 2026-05-17
- **Row count:** 70,194
- **Date range:** 1992-12-31 to 2026-04-01
- **Columns (9):** index, `rank`, `country_full`, `country_abrv`, `total_points`, `previous_points`, `rank_change`, `confederation`, `rank_date`

---

## WC 2026 Fixtures (football-data.org API)

- **API name:** football-data.org
- **Endpoint:** https://api.football-data.org/v4/competitions/WC/matches?season=2026
- **Local file:** `data/raw/wc2026_fixtures.json`
- **Fetch date:** 2026-05-17
- **Record count:** 104

---

## Bookmaker Odds

*(To be populated in Subphase 1.7)*

- **Source:** Manual collection from free odds aggregator (e.g., OddsPortal)
- **Local file:** `data/bookmaker_odds.csv`
- **Collection method:** Manual, collected before each match day

---

## openfootball World Cup Historical Data

- **Source:** openfootball/worldcup (GitHub)
- **GitHub URL:** https://github.com/openfootball/worldcup
- **License:** CC0-1.0 (public domain)
- **Local directories:** `data/raw/openfootball/wc{year}/` for year in 1998, 2002, 2006, 2010, 2014, 2018, 2022
- **Files per tournament:** `cup.txt` (group stage, 48 matches), `cup_finals.txt` (knockout stage, 16 matches)
- **Total matches per tournament:** 64
- **Format:** Football.TXT plain text; match lines contain ` @ ` separator for venue
- **Key fields per match line:** date, home team name, away team name, score, venue
- **Retrieval method:** `python scripts/fetch_openfootball_data.py`
- **Fetch date:** 2026-05-17

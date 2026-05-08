#!/usr/bin/env python
# coding: utf-8

# # 06n – GDELT News Mentions (via REST API)
#
# **Signal**: press/news mentions of the company before Series A.
# Uses GDELT DOC 2.0 REST API — free, no BigQuery quota needed.
# Rate limit: 1 request per 5 seconds.
#
# **Features**:
#   - gdelt_mentions_12m_preA : article count in 12m before Series A
#   - gdelt_mentions_3m_preA  : article count in 3m before Series A
#   - gdelt_has_preA_press    : binary
#   - gdelt_note              : audit

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from shared import load_env
load_env()

PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline",
                   Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
PIPE = next((p for p in PIPE_CANDIDATES if p.exists() and (p / ".run_id").exists()), None) or (
    PIPE_CANDIDATES[0] if PIPE_CANDIDATES[0].exists() else
    Path.home() / "Desktop" / "Data_thesis" / "Pipeline")
PIPE.mkdir(parents=True, exist_ok=True)


def run_id_from_filename(name):
    m = re.match(r"^([^_]+)_", Path(name).name)
    if not m:
        raise ValueError(f"Bad filename: {name}")
    return m.group(1)


CHOSEN_RUN_ID = os.environ.get("PIPELINE_RUN_ID", "").strip() or (
    (PIPE / ".run_id").read_text().strip() if (PIPE / ".run_id").exists() else "")

INPUT_CSV = None
_base_override = os.environ.get("PIPELINE_BASE_CSV", "").strip()
if _base_override and Path(_base_override).exists():
    INPUT_CSV = Path(_base_override)
if INPUT_CSV is None:
    candidates = sorted(
        list(PIPE.glob("*_05_*enriched*.csv")) + list(PIPE.glob("*_05_webtraction*.csv")),
        key=lambda x: x.name, reverse=True)
    if candidates:
        INPUT_CSV = candidates[0]
if INPUT_CSV is None:
    raise FileNotFoundError("No input CSV found")

RUN_ID = run_id_from_filename(INPUT_CSV)
OUTPUT_CSV = PIPE / f"{RUN_ID}_04j_gdelt_news__signals.csv"
print("INPUT :", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

# GDELT DOC 2.0 REST API — free, no auth, 1 req per 5 seconds
GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
DELAY = 5.5  # seconds between requests (rate limit = 1 per 5 sec)
MAXRECORDS = 250  # GDELT artlist API cap per request
# GDELT DOC 2.0 coverage starts Feb 2017 — earlier dates return 0 (no data, not no press)
GDELT_COVERAGE_START = pd.Timestamp("2017-02-19")


def query_gdelt(company_name, start_date, end_date):
    """Query GDELT DOC API for article count mentioning the company.
    Returns (count, error_or_None, truncated).
    truncated=True means count hit MAXRECORDS — true count is unknown.
    """
    try:
        params = {
            "query": f'"{company_name}"',
            "mode": "artlist",
            "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
            "maxrecords": str(MAXRECORDS),
            "format": "json",
        }
        r = requests.get(GDELT_API, params=params, timeout=30)
        if r.status_code == 429:
            return 0, "rate_limited", False
        if r.status_code != 200:
            return 0, f"http_{r.status_code}", False
        data = r.json()
        articles = data.get("articles") or []
        count = len(articles)
        return count, None, count >= MAXRECORDS
    except requests.exceptions.Timeout:
        return 0, "timeout", False
    except Exception as e:
        # GDELT sometimes returns HTML instead of JSON for empty results
        return 0, None, False


df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
print(f"Rows: {len(df)}")

df["gdelt_mentions_12m_preA"] = 0
df["gdelt_mentions_3m_preA"] = 0
df["gdelt_has_preA_press"] = 0
df["gdelt_note"] = ""

from tqdm import tqdm

for i in tqdm(df.index, desc="GDELT"):
    name = str(df.loc[i, "Organization_Name"]).strip()
    if not name or len(name) < 3:
        df.at[i, "gdelt_note"] = "no_name"
        continue

    # Skip non-distinctive names — quoted search for "Ramp" still matches
    # every article containing the word, noise cap can't help at MAXRECORDS=250
    if "name_distinctive" in df.columns:
        nd = df.loc[i, "name_distinctive"]
        if pd.notna(nd) and int(nd) == 0:
            df.at[i, "gdelt_note"] = "skipped_non_distinctive"
            continue

    sA_raw = df.loc[i, "seriesA_date"] if "seriesA_date" in df.columns else None
    sA_dt = pd.to_datetime(sA_raw, errors="coerce")
    if pd.isna(sA_dt):
        df.at[i, "gdelt_note"] = "no_seriesA_date"
        continue

    # GDELT DOC 2.0 coverage starts Feb 2017 — skip earlier dates
    if sA_dt < GDELT_COVERAGE_START:
        df.at[i, "gdelt_note"] = "pre_gdelt_coverage"
        continue

    # End 1 day before Series A to exclude day-of announcement articles
    end_date = (sA_dt - pd.Timedelta(days=1)).to_pydatetime()
    start_12m = (sA_dt - pd.Timedelta(days=365)).to_pydatetime()
    start_3m = (sA_dt - pd.Timedelta(days=90)).to_pydatetime()

    # Query 12-month window
    count_12m, err, truncated_12m = query_gdelt(name, start_12m, end_date)
    time.sleep(DELAY)

    if err == "rate_limited":
        # Back off and retry once
        time.sleep(10)
        count_12m, err, truncated_12m = query_gdelt(name, start_12m, end_date)
        time.sleep(DELAY)

    if err:
        df.at[i, "gdelt_note"] = err
        continue

    # Truncation = hit MAXRECORDS limit, true count unknown — likely name collision
    if truncated_12m:
        df.at[i, "gdelt_note"] = f"noise_cap (>={count_12m} articles, truncated)"
        continue

    # Query 3-month window (only if 12m had results)
    count_3m = 0
    if count_12m > 0:
        count_3m, err_3m, truncated_3m = query_gdelt(name, start_3m, end_date)
        time.sleep(DELAY)

        if err_3m == "rate_limited":
            time.sleep(10)
            count_3m, err_3m, truncated_3m = query_gdelt(name, start_3m, end_date)
            time.sleep(DELAY)

        if truncated_3m:
            count_3m = 0  # can't trust truncated count

    df.at[i, "gdelt_mentions_12m_preA"] = count_12m
    df.at[i, "gdelt_mentions_3m_preA"] = count_3m
    df.at[i, "gdelt_has_preA_press"] = 1 if count_12m > 0 else 0
    df.at[i, "gdelt_note"] = "queried"

df.to_csv(OUTPUT_CSV, index=False)
n_press = int(df["gdelt_has_preA_press"].sum())
print(f"\nSaved: {OUTPUT_CSV}")
print(f"GDELT: {n_press}/{len(df)} companies with pre-A press mentions")

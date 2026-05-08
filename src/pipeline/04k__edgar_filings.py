#!/usr/bin/env python
# coding: utf-8

# # 06p – SEC EDGAR Filing Count (beyond Form D)
#
# **Signal**: how many SEC filings did the company make before Series A?
# Companies that file 10-K, 10-Q, 8-K, S-1 before their Series A are more
# financially mature and institutionally sophisticated.
#
# **Source**: data.sec.gov REST API — free, no API key, 10 req/sec.
# Returns JSON with full filing history for a given CIK.
# We already have CIK for 414/547 companies from step 01 (Form D matching).
#
# **Features**:
#   - edgar_filings_preA: total SEC filings before Series A (all types)
#   - edgar_has_10k_preA: binary — filed at least one 10-K/10-Q before A
#   - edgar_note: audit

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
    for pat in ["*_05_enriched*.csv", "*_05_webtraction*.csv"]:
        candidates = sorted(PIPE.glob(pat), key=lambda x: x.name, reverse=True)
        if candidates:
            INPUT_CSV = candidates[0]
            break
if INPUT_CSV is None:
    raise FileNotFoundError("No input CSV found")

RUN_ID = run_id_from_filename(INPUT_CSV)
OUTPUT_CSV = PIPE / f"{RUN_ID}_04k_edgar_filings__signals.csv"
print("INPUT :", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

# SEC EDGAR API — no auth needed
EDGAR_API = "https://data.sec.gov/submissions/CIK{cik}.json"
HEADERS = {"User-Agent": "ThesisResearch/1.0 (arthurschoen@mit.edu)", "Accept": "application/json"}
DELAY = 0.12  # ~8 req/sec (under 10/sec limit)

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
df["edgar_filings_preA"] = 0
df["edgar_has_10k_preA"] = 0
df["edgar_note"] = ""

from tqdm import tqdm

for i in tqdm(df.index, desc="EDGAR filings"):
    cik_raw = df.loc[i, "sec_match_cik"] if "sec_match_cik" in df.columns else None
    if pd.isna(cik_raw) or cik_raw is None or str(cik_raw).strip() in ("", "nan"):
        df.at[i, "edgar_note"] = "no_cik"
        continue

    cik = str(int(float(cik_raw))).zfill(10)  # SEC requires 10-digit zero-padded
    sA_raw = df.loc[i, "seriesA_date"] if "seriesA_date" in df.columns else None
    sA_dt = pd.to_datetime(sA_raw, errors="coerce")
    if pd.isna(sA_dt):
        df.at[i, "edgar_note"] = "no_seriesA_date"
        continue

    sA_str = sA_dt.strftime("%Y-%m-%d")

    try:
        url = EDGAR_API.format(cik=cik)
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            df.at[i, "edgar_note"] = f"http_{r.status_code}"
            time.sleep(DELAY)
            continue

        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        dates = recent.get("filingDate", [])
        forms = recent.get("form", [])

        # Count filings before Series A
        n_preA = 0
        has_10k = 0
        for dt, form in zip(dates, forms):
            if dt and dt < sA_str:  # strict < to exclude the Series A Form D filing itself
                n_preA += 1
                if form in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                    has_10k = 1

        df.at[i, "edgar_filings_preA"] = n_preA
        df.at[i, "edgar_has_10k_preA"] = has_10k
        df.at[i, "edgar_note"] = f"found_{n_preA}_filings"

    except requests.exceptions.Timeout:
        df.at[i, "edgar_note"] = "timeout"
    except Exception as e:
        df.at[i, "edgar_note"] = f"error_{type(e).__name__}"

    time.sleep(DELAY)

df.to_csv(OUTPUT_CSV, index=False)
n_filings = int((df["edgar_filings_preA"] > 0).sum())
n_10k = int(df["edgar_has_10k_preA"].sum())
print(f"\nSaved: {OUTPUT_CSV}")
print(f"EDGAR: {n_filings}/{len(df)} with pre-A filings, {n_10k} with 10-K/10-Q")

#!/usr/bin/env python
# coding: utf-8

# # 04d – OpenAlex Signals (simplified)
#
# Binary + count: does this company have academic publications before Series A?
# Sector-filtered to deeptech only (biotech, AI, quantum, hardware — other sectors
# don't publish research papers).
#
# Company matching: domain-based institution lookup (deterministic, no LLM needed).
# Anti-leakage: publication_date <= seriesA_date, with pagination.
#
# Output: openalex_has_papers_preA (0/1), openalex_works_count, openalex_signal_note

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline", Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
PIPE = next((p for p in PIPE_CANDIDATES if p.exists() and (p / ".run_id").exists()), None) or (PIPE_CANDIDATES[0] if PIPE_CANDIDATES[0].exists() else Path.home() / "Desktop" / "Data_thesis" / "Pipeline")
PIPE.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from shared import load_env
load_env()

def run_id_from_filename(name):
    m = re.match(r"^([^_]+)_", Path(name).name)
    if not m:
        raise ValueError(f"Bad filename: {name}")
    return m.group(1)

CHOSEN_RUN_ID = os.environ.get("PIPELINE_RUN_ID", "").strip() or ((PIPE / ".run_id").read_text().strip() if (PIPE / ".run_id").exists() else "")
_base_override = os.environ.get("PIPELINE_BASE_CSV", "").strip()
if _base_override and Path(_base_override).exists():
    INPUT_CSV = Path(_base_override)
else:
    INPUT_CSV = None
if INPUT_CSV is None and CHOSEN_RUN_ID:
    for suffix in ["_03_enriched_base.csv", "_03_formd__seriesA_dates_enriched__clean.csv",
                    "_05_webtraction__signals.csv"]:
        p = PIPE / f"{CHOSEN_RUN_ID}{suffix}"
        if p.exists():
            INPUT_CSV = p
            break
if INPUT_CSV is None:
    for pattern in ["*_03_enriched_base.csv", "*_03_formd__seriesA_dates_enriched__clean.csv",
                     "*_05_webtraction__signals.csv"]:
        candidates = sorted(PIPE.glob(pattern), key=lambda x: x.name, reverse=True)
        if candidates:
            INPUT_CSV = candidates[0]
            break
if INPUT_CSV is None:
    raise FileNotFoundError("No base CSV found in Pipeline")

RUN_ID = run_id_from_filename(INPUT_CSV)
OUTPUT_CSV = PIPE / f"{RUN_ID}_04d_openalex__signals.csv"
print("INPUT:", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

OPENALEX = "https://api.openalex.org"
HEADERS = {"User-Agent": "mailto:thesis@example.com", "Accept": "application/json"}
DELAY = 0.5

# Sector filter: only query deeptech companies (biotech, AI, hardware publish papers)
OPENALEX_SECTORS = {"deeptech"}


def _normalize_domain(url):
    """Strip protocol, www, and path. 'https://www.Foo.co/bar' -> 'foo.co'."""
    if not url:
        return ""
    u = str(url).strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = u.split("/", 1)[0].split("?", 1)[0]
    u = re.sub(r"^www\.", "", u)
    return u.strip()


def _find_matching_institution(org_name, company_website):
    """Look up company in OpenAlex institutions by name, verify by domain match.
    Returns (institution_id, display_name, note)."""
    cb_domain = _normalize_domain(company_website)
    if not cb_domain:
        return None, "", "no_company_website"
    try:
        r = requests.get(
            f"{OPENALEX}/institutions",
            params={"search": org_name, "per_page": 10},
            headers=HEADERS, timeout=20,
        )
        if r.status_code != 200:
            return None, "", f"http_{r.status_code}"
        results = r.json().get("results", [])
        if not results:
            return None, "", "no_institution_in_openalex"
        for inst in results:
            oa_domain = _normalize_domain(inst.get("homepage_url", ""))
            if oa_domain and oa_domain == cb_domain:
                inst_id = (inst.get("id") or "").replace("https://openalex.org/", "")
                return inst_id, inst.get("display_name", ""), "matched_by_domain"
        rej = ", ".join(_normalize_domain(i.get("homepage_url", "")) or "?" for i in results[:3])
        return None, "", f"domain_mismatch (candidates: {rej})"
    except requests.exceptions.Timeout:
        return None, "", "timeout"
    except Exception as e:
        return None, "", f"err_{type(e).__name__}"


def _count_works_preA(inst_id, end_str):
    """Count all works affiliated with institution published on or before end_str.
    Paginates through all results (cursor-based) to get accurate count."""
    total = 0
    cursor = "*"
    while cursor:
        params = {
            "filter": f"institutions.id:{inst_id},to_publication_date:{end_str}",
            "per-page": 200,
            "cursor": cursor,
            "select": "id,publication_date",
        }
        r = None
        for attempt in range(3):
            try:
                r = requests.get(f"{OPENALEX}/works", params=params, headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if r is None or r.status_code != 200:
            break
        data = r.json()
        results = data.get("results", [])
        # Client-side anti-leakage double-check
        for w in results:
            pub = (w.get("publication_date") or "")[:10]
            if not pub or pub <= end_str:
                total += 1
        cursor = data.get("meta", {}).get("next_cursor")
        if not results:
            break
        time.sleep(0.3)
    return total


def openalex_signals(org_name, series_a_date, company_website=""):
    """Check if company has academic publications before Series A."""
    out = {
        "openalex_has_papers_preA": 0,
        "openalex_works_count": 0,
        "openalex_signal_note": "",
    }
    name = (org_name or "").strip()
    if len(name) < 2:
        out["openalex_signal_note"] = "no_name"
        return out
    try:
        end_d = pd.to_datetime(series_a_date, errors="coerce")
        if pd.isna(end_d):
            out["openalex_signal_note"] = "no_seriesA_date"
            return out
        end_str = end_d.strftime("%Y-%m-%d")

        # Resolve company to OpenAlex institution via domain match
        inst_id, inst_name, match_note = _find_matching_institution(name, company_website)
        if not inst_id:
            out["openalex_signal_note"] = match_note
            return out

        # Count works with pagination
        count = _count_works_preA(inst_id, end_str)
        out["openalex_works_count"] = count
        out["openalex_has_papers_preA"] = 1 if count > 0 else 0
        out["openalex_signal_note"] = f"{match_note}:{inst_name} ({inst_id}), {count} works"
    except Exception as e:
        out["openalex_signal_note"] = f"error: {type(e).__name__}: {str(e)[:80]}"
    return out


# =====================================================================
# Main
# =====================================================================

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

# Initialize output columns
df["openalex_has_papers_preA"] = 0
df["openalex_works_count"] = 0
df["openalex_signal_note"] = ""

col_web = "Website" if "Website" in df.columns else ("domain_clean" if "domain_clean" in df.columns else None)

# Sector filter
has_sector = "startup_sector" in df.columns
if has_sector:
    sector_mask = df["startup_sector"].str.lower().isin(OPENALEX_SECTORS)
    n_query = sector_mask.sum()
    n_skip = (~sector_mask).sum()
    print(f"OpenAlex: {n_query} companies in scope (deeptech), {n_skip} skipped by sector filter")
else:
    sector_mask = pd.Series(True, index=df.index)
    print("OpenAlex: no startup_sector column — querying all companies")

from tqdm import tqdm
for i in tqdm(df.index, desc="OpenAlex"):
    if has_sector and not sector_mask.iloc[i]:
        df.at[i, "openalex_signal_note"] = "skipped_sector"
        continue

    name = str(df.loc[i, "Organization_Name"]).strip()
    sdate = df.loc[i, "seriesA_date"] if "seriesA_date" in df.columns else ""
    web = str(df.loc[i, col_web]).strip() if col_web and pd.notna(df.loc[i, col_web]) else ""

    out = openalex_signals(name, sdate, company_website=web)
    for k, v in out.items():
        df.at[i, k] = v
    time.sleep(DELAY)

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
has_papers = (df["openalex_has_papers_preA"] == 1).sum()
print(f"OpenAlex: {has_papers}/{len(df)} have pre-Series A publications")

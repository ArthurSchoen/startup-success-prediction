#!/usr/bin/env python
# coding: utf-8

# # 04e – Hacker News Signals (simplified)
#
# Binary launch + mention count. No sector filter (HN covers all tech sectors).
# Coverage: ~1% (most startups don't get HN attention before Series A).
#
# Company matching: exact quoted domain search via Algolia API.
# Anti-leakage: created_at_i <= seriesA timestamp, no upvote/comment counts.
#
# Output: hn_launch (0/1), hn_mention_count_preA, hn_note

import os
import re
import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path

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
OUTPUT_CSV = PIPE / f"{RUN_ID}_04e_hacker_news__signals.csv"
print("INPUT:", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

HN_API = "https://hn.algolia.com/api/v1/search"
HEADERS = {"User-Agent": "ThesisResearch/1.0 (Python)"}
DELAY = 0.4

# LLM for HN post verification
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_llm_client = None
if OPENAI_API_KEY:
    try:
        import openai
        _llm_client = openai.OpenAI()
        print("HN: LLM post verification enabled")
    except ImportError:
        print("WARNING: openai not installed")


def _domain_clean(domain):
    s = (domain or "").strip().lower()
    s = re.sub(r"^https?://", "", s).split("/")[0]
    s = re.sub(r"^www\.", "", s)
    return s


def _verify_hn_post_llm(post_title, company_name, company_description):
    """LLM check: is this HN post about this company?"""
    if _llm_client is None:
        return True
    try:
        llm_r = _llm_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=20, temperature=0,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": (
                    f'Is this Hacker News post about the company "{company_name}" '
                    f'({company_description[:100]})?\n'
                    f'HN post title: "{post_title}"\n'
                    f'Return: {{"match": 0 or 1}}'
                )},
            ],
        )
        raw = llm_r.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        return parsed.get("match", 0) == 1
    except Exception:
        return True


def hn_signals(domain, series_a_date, company_name="", company_description="", founded_date=None):
    """Extract HN mention count and binary launch signal, pre-Series A only."""
    out = {"hn_launch": 0, "hn_mention_count_preA": 0, "hn_note": ""}

    d = _domain_clean(domain)
    if not d or len(d) < 3:
        out["hn_note"] = "no_domain"
        return out

    # Compute Series A timestamp for date filter
    max_ts = None
    if series_a_date:
        try:
            sa_dt = pd.to_datetime(series_a_date, errors="coerce")
            if pd.notna(sa_dt):
                max_ts = int(sa_dt.timestamp())
        except Exception:
            pass
    if max_ts is None:
        out["hn_note"] = "no_seriesA_date"
        return out

    # 1) Count total HN posts linking to this domain before Series A
    try:
        r = requests.get(HN_API,
            params={"query": f'"{d}"', "restrictSearchableAttributes": "url",
                    "hitsPerPage": 1, "attributesToRetrieve": [],
                    "numericFilters": f"created_at_i<={max_ts}"},
            headers=HEADERS, timeout=10)
        if r.status_code == 200:
            out["hn_mention_count_preA"] = int(r.json().get("nbHits", 0))
    except Exception:
        pass
    time.sleep(0.2)

    # 2) Find any launch post (Show HN or story) before Series A
    try:
        date_filter = f"created_at_i<={max_ts}"
        hits = []

        # Try Show HN first
        r = requests.get(HN_API,
            params={"query": d, "restrictSearchableAttributes": "url",
                    "tags": "show_hn", "hitsPerPage": 5,
                    "numericFilters": date_filter},
            headers=HEADERS, timeout=10)
        if r.status_code == 200:
            hits = r.json().get("hits", [])

        # Fallback to story (still with date filter)
        if not hits:
            r = requests.get(HN_API,
                params={"query": d, "restrictSearchableAttributes": "url",
                        "tags": "story", "hitsPerPage": 5,
                        "numericFilters": date_filter},
                headers=HEADERS, timeout=10)
            if r.status_code == 200:
                hits = r.json().get("hits", [])

        if not hits:
            out["hn_note"] = "no_posts_preA"
            return out

        # LLM verify the best post
        best = hits[0]
        post_title = best.get("title", "") or best.get("story_title", "")
        if post_title and not _verify_hn_post_llm(post_title, company_name, company_description):
            out["hn_note"] = "llm_rejected"
            return out

        # Founding-date guard: discard if post predates founding by >1 year
        if founded_date:
            try:
                post_ts = best.get("created_at_i", 0)
                founded_dt = pd.to_datetime(founded_date, errors="coerce")
                if pd.notna(founded_dt) and post_ts > 0:
                    post_dt = pd.Timestamp(post_ts, unit="s")
                    if (founded_dt - post_dt).days > 365:
                        out["hn_note"] = "post_predates_founding"
                        return out
            except Exception:
                pass

        out["hn_launch"] = 1
        out["hn_note"] = f"found:{post_title[:60]}"
    except Exception:
        pass

    return out


# =====================================================================
# Main
# =====================================================================

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
col_domain = "domain_clean" if "domain_clean" in df.columns else "Website"

# Initialize output columns
df["hn_launch"] = 0
df["hn_mention_count_preA"] = 0
df["hn_note"] = ""

from tqdm import tqdm
for i in tqdm(df.index, desc="Hacker News"):
    domain = str(df.loc[i, col_domain]).strip() if col_domain in df.columns else ""
    name = str(df.loc[i, "Organization_Name"]).strip()
    sdate = str(df.loc[i, "seriesA_date"]).strip() if "seriesA_date" in df.columns else ""
    desc = str(df.loc[i, "Description"]).strip()[:200] if "Description" in df.columns else ""
    founded = str(df.loc[i, "Founded Date"]).strip() if "Founded Date" in df.columns else ""

    out = hn_signals(domain,
                     sdate if sdate not in ("nan", "NaT", "") else None,
                     company_name=name, company_description=desc,
                     founded_date=founded if founded not in ("nan", "NaT", "") else None)
    for k, v in out.items():
        df.at[i, k] = v
    time.sleep(DELAY)

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
print(f"HN: {int(df['hn_launch'].sum())} launches, "
      f"{(df['hn_mention_count_preA'] > 0).sum()} with mentions")

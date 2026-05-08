#!/usr/bin/env python
# coding: utf-8

# # 04a – Wikipedia Signals (simplified)
#
# Binary signal only: does this company have a Wikipedia page that existed
# before Series A? Coverage is ~0.5%, so sub-features (edit counts, pageviews,
# vandalism, infobox) are not statistically meaningful and were removed.
#
# Output: wikipedia_before_seriesA (0/1), wiki_note

import os
import re
import json
import time
import requests
import pandas as pd
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

INPUT_CSV = None
_base_override = os.environ.get("PIPELINE_BASE_CSV", "").strip()
if _base_override and Path(_base_override).exists():
    INPUT_CSV = Path(_base_override)
if INPUT_CSV is None and CHOSEN_RUN_ID:
    p = PIPE / f"{CHOSEN_RUN_ID}_05_webtraction__signals.csv"
    if p.exists():
        INPUT_CSV = p
if INPUT_CSV is None:
    candidates = sorted(PIPE.glob("*_05_webtraction__signals.csv"), key=lambda x: x.name, reverse=True)
    if candidates:
        INPUT_CSV = candidates[0]
if INPUT_CSV is None:
    raise FileNotFoundError("No _05_webtraction__signals.csv found in Pipeline")

RUN_ID = run_id_from_filename(INPUT_CSV)
OUTPUT_CSV = PIPE / f"{RUN_ID}_04a_wikipedia__signals.csv"
print("INPUT:", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "thesis-research-bot/1.0 (academic)"}
DELAY = 0.5

# LLM for entity verification
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_llm_client = None
if OPENAI_API_KEY:
    try:
        import openai
        _llm_client = openai.OpenAI()
        print("Wikipedia: LLM entity verification enabled")
    except ImportError:
        print("WARNING: openai not installed")


def _wiki_search(query, limit=3):
    """Search Wikipedia, return list of page titles."""
    try:
        r = requests.get(WIKI_API, params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": limit, "format": "json"
        }, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return [res["title"] for res in r.json().get("query", {}).get("search", [])]
    except Exception:
        pass
    return []


def _wiki_extract(title, sentences=3):
    """Get first N sentences of a Wikipedia page."""
    try:
        r = requests.get(WIKI_API, params={
            "action": "query", "titles": title, "prop": "extracts",
            "exsentences": sentences, "explaintext": True, "format": "json"
        }, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            return list(pages.values())[0].get("extract", "")
    except Exception:
        pass
    return ""


def _llm_entity_match(extract, company_name, company_description):
    """LLM check: is this Wikipedia article about this company?"""
    if _llm_client is None or not extract or len(extract) < 20:
        return False
    try:
        llm_r = _llm_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=20, temperature=0,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": (
                    f'Is this Wikipedia article about the company "{company_name}" '
                    f'({company_description[:150]})?\n'
                    f'Wikipedia: {extract[:400]}\n'
                    f'Return: {{"match": 0 or 1}}'
                )},
            ],
        )
        raw = llm_r.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        return parsed.get("match", 0) == 1
    except Exception:
        return False


def _wiki_creation_date(title):
    """Get page creation date (first revision timestamp)."""
    try:
        r = requests.get(WIKI_API, params={
            "action": "query", "titles": title, "prop": "revisions",
            "rvlimit": 1, "rvdir": "newer", "rvprop": "timestamp", "format": "json"
        }, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            revs = list(pages.values())[0].get("revisions", [])
            if revs:
                return revs[0].get("timestamp", "")[:10]
    except Exception:
        pass
    return ""


def wikipedia_signals(company_name, series_a_date, description="", industries=""):
    """Check if company has a Wikipedia page created before Series A."""
    out = {"wikipedia_before_seriesA": 0, "wiki_note": ""}

    name = (company_name or "").strip()
    if not name or len(name) < 2 or name in ("nan", "None"):
        out["wiki_note"] = "no_name"
        return out

    try:
        sa_dt = pd.to_datetime(series_a_date, errors="coerce")
        if pd.isna(sa_dt):
            out["wiki_note"] = "no_seriesA_date"
            return out
        sa_str = sa_dt.strftime("%Y-%m-%d")
    except Exception:
        out["wiki_note"] = "date_parse_error"
        return out

    # Search Wikipedia (name-first, then name + keywords)
    ind_keywords = [k.strip() for k in (industries or "").split(",")[:3] if k.strip() and k.strip() != "nan"]
    searches = [name]
    if ind_keywords:
        searches.append(f"{name} {' '.join(ind_keywords)}")

    matched_title = None
    for query in searches:
        if matched_title:
            break
        titles = _wiki_search(query, limit=3)
        time.sleep(0.3)

        for title in titles:
            extract = _wiki_extract(title, sentences=3)
            time.sleep(0.2)
            if not extract or len(extract) < 20:
                continue
            if _llm_entity_match(extract, name, description):
                matched_title = title
                break
            time.sleep(0.2)

    if not matched_title:
        out["wiki_note"] = "no_match"
        return out

    # Check creation date
    creation_date = _wiki_creation_date(matched_title)
    time.sleep(0.2)

    if not creation_date or creation_date > sa_str:
        out["wiki_note"] = f"created={creation_date} (after seriesA={sa_str})"
        return out

    out["wikipedia_before_seriesA"] = 1
    out["wiki_note"] = f"matched:{matched_title} created={creation_date}"
    return out


# =====================================================================
# Main
# =====================================================================

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

wiki_cols = ["wikipedia_before_seriesA", "wiki_note"]
for c in wiki_cols:
    if c not in df.columns:
        df[c] = "" if c == "wiki_note" else 0

from tqdm import tqdm
for i in tqdm(df.index, desc="Wikipedia"):
    name = str(df.loc[i, "Organization_Name"]).strip()
    sdate = str(df.loc[i, "seriesA_date"]).strip() if "seriesA_date" in df.columns else ""
    desc = str(df.loc[i, "Description"]).strip()[:200] if "Description" in df.columns else ""
    ind = str(df.loc[i, "Industries"]).strip()[:200] if "Industries" in df.columns else ""

    out = wikipedia_signals(name, sdate if sdate not in ("nan", "NaT", "") else None,
                           description=desc, industries=ind)
    for k, v in out.items():
        df.at[i, k] = v
    time.sleep(DELAY)

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
before_sa = int(df["wikipedia_before_seriesA"].sum())
print(f"Wikipedia: {before_sa}/{len(df)} have pre-Series A page")

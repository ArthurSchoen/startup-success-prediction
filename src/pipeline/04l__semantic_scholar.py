#!/usr/bin/env python
# coding: utf-8

# # 06q – Semantic Scholar (academic papers by company authors)
#
# **Signal**: academic paper activity affiliated with the company before Series A.
# Complements OpenAlex with different coverage (Semantic Scholar indexes
# CS/AI/ML conferences better than OpenAlex which is broader).
#
# **Source**: Semantic Scholar API v1 — free, no API key for basic access.
# Rate limit: 1 req/sec (unauthenticated), 10 req/sec (with API key).
#
# **Features**:
#   - ss_papers_preA: papers with company affiliation before Series A
#   - ss_citations_preA: total citations on those papers
#   - ss_has_preA_research: binary
#   - ss_note: audit

import os
import re
import time
import json
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
OUTPUT_CSV = PIPE / f"{RUN_ID}_04l_semantic_scholar__signals.csv"
print("INPUT :", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

# Semantic Scholar API
SS_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")  # optional, for higher rate
HEADERS = {"User-Agent": "ThesisResearch/1.0"}
if SS_KEY:
    HEADERS["x-api-key"] = SS_KEY
    DELAY = 0.12  # 10 req/sec with key
    print("Semantic Scholar: API key loaded (10 req/sec)")
else:
    DELAY = 1.1  # 1 req/sec without key
    print("Semantic Scholar: no API key (1 req/sec — slow)")

# LLM for paper verification (same pattern as Patents/FedAwards/OpenAlex)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_llm_client = None
if OPENAI_API_KEY:
    try:
        import openai
        _llm_client = openai.OpenAI()
        print("Semantic Scholar: LLM verification enabled")
    except ImportError:
        print("WARNING: openai not installed — skipping LLM verification")
else:
    print("WARNING: OPENAI_API_KEY not set — using noise caps only")

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
df["ss_papers_preA"] = 0
df["ss_citations_preA"] = 0
df["ss_has_preA_research"] = 0
df["ss_note"] = ""

# Only query companies with distinctive names (>1 word or name_distinctive=1)
# Single common words return too much noise
from tqdm import tqdm

for i in tqdm(df.index, desc="Semantic Scholar"):
    name = str(df.loc[i, "Organization_Name"]).strip()
    if not name or len(name) < 3:
        df.at[i, "ss_note"] = "no_name"
        continue

    # Skip short single common words (same logic as GDELT)
    words = name.split()
    if len(words) == 1 and len(name) <= 6:
        df.at[i, "ss_note"] = "skipped_short_single_word"
        continue

    sA_raw = df.loc[i, "seriesA_date"] if "seriesA_date" in df.columns else None
    sA_dt = pd.to_datetime(sA_raw, errors="coerce")
    if pd.isna(sA_dt):
        df.at[i, "ss_note"] = "no_seriesA_date"
        continue

    sA_year = sA_dt.year

    try:
        # Search for papers mentioning company name in title/abstract
        params = {
            "query": f'"{name}"',
            "fields": "title,year,citationCount,authors",
            "limit": 100,
            "year": f"-{sA_year}",  # papers up to Series A year
        }
        r = requests.get(SS_API, params=params, headers=HEADERS, timeout=15)

        if r.status_code == 429:
            df.at[i, "ss_note"] = "rate_limited"
            time.sleep(5)
            continue
        if r.status_code != 200:
            df.at[i, "ss_note"] = f"http_{r.status_code}"
            time.sleep(DELAY)
            continue

        data = r.json()
        papers = data.get("data", [])

        # Step 1: Filter by year (anti-leakage)
        pre_a_papers = []
        for p in papers:
            yr = p.get("year")
            if yr and yr > sA_year:
                continue
            pre_a_papers.append(p)

        if not pre_a_papers:
            df.at[i, "ss_note"] = "not_found"
            time.sleep(DELAY)
            continue

        # Step 2: LLM verification — are these papers ACTUALLY from/about
        # this company, or just title/abstract collisions?
        # Same pattern as Patents and Federal Awards verifiers.
        desc = str(df.loc[i, "Description"]).strip()[:300] if "Description" in df.columns else ""
        sample_titles = [str(p.get("title", ""))[:100] for p in pre_a_papers[:5] if p.get("title")]
        total_raw = len(pre_a_papers)
        total_raw_citations = sum(p.get("citationCount") or 0 for p in pre_a_papers)

        verified = False
        if _llm_client and sample_titles:
            try:
                titles_str = "\n".join(f"  - {t}" for t in sample_titles)
                llm_prompt = (
                    f'Startup: "{name}"\n'
                    f'Description: {desc}\n\n'
                    f'These {total_raw} academic papers were found by searching for "{name}" '
                    f'in paper titles/abstracts (total {total_raw_citations} citations):\n'
                    f'{titles_str}\n\n'
                    f'Question: are these papers plausibly FROM or ABOUT this specific startup?\n'
                    f'Accept (match=1) if the paper topics align with the startup\'s domain.\n'
                    f'Reject (match=0) if the papers are clearly about something else '
                    f'(e.g. "Clarity" the startup vs papers about "clarity" the concept, '
                    f'"Corpay" the fintech vs papers about "corporate payments" in general).\n'
                    f'Return: {{"match": 0 or 1}}'
                )
                llm_r = _llm_client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0, max_tokens=30,
                    messages=[
                        {"role": "system", "content": "Return ONLY JSON."},
                        {"role": "user", "content": llm_prompt},
                    ],
                )
                raw = llm_r.choices[0].message.content.strip()
                start, end = raw.find("{"), raw.rfind("}") + 1
                verified = json.loads(raw[start:end]).get("match", 0) == 1
            except Exception:
                verified = False
        elif not _llm_client:
            # Without LLM: apply hard noise caps only
            verified = (total_raw <= 50 and total_raw_citations <= 10_000)

        if verified:
            df.at[i, "ss_papers_preA"] = total_raw
            df.at[i, "ss_citations_preA"] = total_raw_citations
            df.at[i, "ss_has_preA_research"] = 1
            df.at[i, "ss_note"] = f"verified_{total_raw}_papers"
        else:
            df.at[i, "ss_papers_preA"] = 0
            df.at[i, "ss_citations_preA"] = 0
            df.at[i, "ss_has_preA_research"] = 0
            df.at[i, "ss_note"] = f"llm_rejected ({total_raw} papers, {total_raw_citations} citations)"

    except requests.exceptions.Timeout:
        df.at[i, "ss_note"] = "timeout"
    except Exception as e:
        df.at[i, "ss_note"] = f"error_{type(e).__name__}"

    time.sleep(DELAY)

df.to_csv(OUTPUT_CSV, index=False)
n_found = int(df["ss_has_preA_research"].sum())
print(f"\nSaved: {OUTPUT_CSV}")
print(f"Semantic Scholar: {n_found}/{len(df)} with pre-A papers")

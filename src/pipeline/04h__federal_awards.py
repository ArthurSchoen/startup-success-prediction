#!/usr/bin/env python
# coding: utf-8

# # 06l – Federal Awards (USAspending.gov)
#
# **Signal**: federal grants AND contracts awarded to the company before Series A.
# Replaces NIH-only step. USAspending.gov is the canonical US government awards database
# (US Treasury) and supersedes NIH Reporter, NSF Award Search, and SBIR.gov for our purpose.
#
# **Coverage**:
#   - GRANTS (codes 02-05): NIH, NSF, DOE, USDA, EPA, ED, DHS, etc.
#   - CONTRACTS (codes A-D): DoD, DARPA, NASA prime contracts (key for hardware/aerospace)
#   - All US federal agencies, since 2007-10-01 (irrelevant for our dataset, founded ≥2015)
#
# **Anti-leakage**:
#   - `time_period` filter on the API itself (start_date ≤ Series A)
#   - Per-record `Start Date` post-filter as a second guard
#
# **Survivorship bias**: NONE — federal awards are permanent legal records by law.
# Terminated/cancelled awards stay in the database forever.
#
# **Features produced**:
#   - fed_grants_count_preA   : # of grant awards before Series A
#   - fed_contracts_count_preA: # of contract awards before Series A
#   - fed_awards_before_seriesA: 1 if ANY federal grant or contract before Series A
#   - fed_awards_note         : status / debugging
#
# **Verification**: USAspending's `recipient_search_text` is fuzzy substring matching, so
# we run LLM verification on every distinct recipient name returned, exactly like
# we do for NIH and OpenAlex.

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline", Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
PIPE = next((p for p in PIPE_CANDIDATES if p.exists() and (p / ".run_id").exists()), None) or (PIPE_CANDIDATES[0] if PIPE_CANDIDATES[0].exists() else Path.home() / "Desktop" / "Data_thesis" / "Pipeline")
PIPE.mkdir(parents=True, exist_ok=True)

# Load .env so OPENAI_API_KEY is available when invoked directly. Without this,
# the LLM recipient verifier silently doesn't fire, leading to false positives
# like Zippy (online lender) getting "63 federal contracts" from an unrelated
# ZIPPY SHELL INC storage company.
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
# Output filename keeps the _23_ slot the old NIH step occupied so downstream scripts
# that look for "_23_*__signals.csv" still find it.
OUTPUT_CSV = PIPE / f"{RUN_ID}_04h_federal_awards__signals.csv"
print("INPUT:", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

USASPEND_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
HEADERS = {"Content-Type": "application/json", "User-Agent": "thesis-research-bot/1.0"}
DELAY = 0.3

# USAspending award_type_codes
GRANT_CODES = ["02", "03", "04", "05"]      # block grants, project grants, cooperative agreements, etc.
CONTRACT_CODES = ["A", "B", "C", "D"]       # BPA, IDV, contract, definitive contract
# USAspending data floor: searches earlier than 2007-10-01 are rejected
USASPEND_MIN_DATE = "2007-10-01"

# LLM for recipient verification
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_llm_client = None
if OPENAI_API_KEY:
    try:
        import openai, json as _json
        _llm_client = openai.OpenAI()
        print("Federal Awards: LLM recipient verification enabled")
    except ImportError:
        pass


def _verify_recipient_llm(recipient_name, company_name, company_description,
                          founded_date="", n_matches=1):
    """LLM check: is this USAspending recipient the same entity as the company?

    Accepts plausible legal-name variants (e.g. "TRAILHEAD BIOSYSTEMS INC" for
    a company called "Trailhead"). Rejects obvious different-entity collisions
    (Zippy mortgage startup vs "Zippy Shell" storage company).

    Volume context is provided so the LLM can flag implausibly large award
    footprints (>50 awards for a pre-Series-A startup = name collision).
    """
    if _llm_client is None:
        return True
    try:
        import json as _json
        volume_note = (
            f" (this recipient has {n_matches} total awards — "
            f"flag if implausibly many for a pre-Series-A startup)"
            if n_matches > 1 else ""
        )
        llm_r = _llm_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=40, temperature=0,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": (
                    f'Pre-Series-A startup:\n'
                    f'  name: "{company_name}"\n'
                    f'  founded: {founded_date or "unknown"}\n'
                    f'  description: {company_description[:250]}\n\n'
                    f'USAspending.gov recipient: "{recipient_name}"{volume_note}\n\n'
                    f'Question: is this recipient the SAME entity as the startup '
                    f'(possibly under a legal name variant like "XYZ INC", "XYZ LLC", '
                    f'"XYZ THERAPEUTICS, INC.", "XYZ FEDERAL")?\n\n'
                    f'Accept (match=1) when:\n'
                    f'  - The recipient name is a plausible legal variant of the startup name, AND\n'
                    f'  - Nothing about the name obviously conflicts with the startup\'s sector '
                    f'(an insurance startup matched to a clearly agricultural recipient name is suspect).\n\n'
                    f'Reject (match=0) when:\n'
                    f'  - The recipient name contains a qualifier that reveals a different entity '
                    f'("ZIPPY SHELL INC" storage vs "Zippy" mortgage; "ROOTS INC" agricultural vs '
                    f'"Roots" insurance-AI).\n'
                    f'  - The volume is huge (>50 total awards) and the startup is pre-Series A — '
                    f'that\'s almost certainly a large unrelated company with the same short name.\n\n'
                    f'When unsure, prefer ACCEPT — real matches should pass. We only reject on clear collisions.\n\n'
                    f'Return: {{"match": 0 or 1}}'
                )},
            ],
        )
        raw = llm_r.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        return _json.loads(raw[start:end]).get("match", 0) == 1
    except Exception:
        return False


def _query_usaspending(name, type_codes, end_date_str):
    """Query USAspending for awards of a given type to a recipient, with anti-leakage time filter."""
    payload = {
        "filters": {
            "award_type_codes": type_codes,
            "recipient_search_text": [name],
            "time_period": [{"start_date": USASPEND_MIN_DATE, "end_date": end_date_str}],
        },
        "fields": ["Recipient Name", "Award Amount", "Start Date", "Awarding Agency"],
        "limit": 100,
        "page": 1,
    }
    try:
        r = requests.post(USASPEND_URL, json=payload, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        return r.json().get("results", []), None
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception as e:
        return None, f"error_{type(e).__name__}"


def fed_award_signals(company_name, series_a_date, company_description="",
                       founded_date=""):
    out = {
        "fed_grants_count_preA": 0,
        "fed_contracts_count_preA": 0,
        "fed_awards_before_seriesA": 0,
        "fed_awards_note": "",
    }
    name = (company_name or "").strip()
    if not name or len(name) < 3 or name in ("nan", "None"):
        return out

    sa_dt = pd.to_datetime(series_a_date, errors="coerce")
    if pd.isna(sa_dt):
        out["fed_awards_note"] = "no_seriesA_date"
        return out
    sa_str = sa_dt.strftime("%Y-%m-%d")

    # Cache LLM verdicts per unique recipient string across both queries (grants + contracts)
    # Also cache the match count so the LLM gets volume context.
    llm_cache = {}
    recipient_counts = {}  # rec_raw -> total matches observed so far

    def _recipient_ok(rec_raw):
        if rec_raw in llm_cache:
            return llm_cache[rec_raw]
        n_matches = recipient_counts.get(rec_raw, 1)
        if _llm_client is not None:
            ok = _verify_recipient_llm(
                rec_raw, name, company_description,
                founded_date=founded_date, n_matches=n_matches,
            )
        else:
            # Fallback when LLM is unavailable: strict prefix match (lowercased)
            r = (rec_raw or "").lower().strip()
            n = name.lower().strip()
            ok = (r == n or r.startswith(n + " ") or r.startswith(n + ","))
        llm_cache[rec_raw] = ok
        return ok

    # Founding-date gate: reject awards whose Start Date is more than 5 years
    # before the company's Founded Date. Same logic as Patents — catches name
    # collisions where e.g. "ROOTS INC" is an unrelated entity from 2001 while
    # our startup "Roots" was founded in 2018.
    _FOUNDING_GATE_DAYS = 5 * 365
    founded_dt = pd.to_datetime(founded_date, errors="coerce") if founded_date else None

    def _filter_and_count(results):
        """Verify recipients via LLM cache + founding-date gate + anti-leakage on Start Date."""
        if not results:
            return 0
        # First pass: populate recipient_counts so the LLM sees total volume
        for award in results:
            rec = award.get("Recipient Name", "") or ""
            recipient_counts[rec] = recipient_counts.get(rec, 0) + 1
        # Second pass: verify and count
        verified = []
        for award in results:
            rec = award.get("Recipient Name", "") or ""
            if not _recipient_ok(rec):
                continue
            start = (award.get("Start Date") or "")[:10]
            # Anti-leakage: must start on or before Series A
            if not start or start > sa_str:
                continue
            # Founding-date gate: award Start Date must not be >5y before Founded Date
            if founded_dt is not None and not pd.isna(founded_dt):
                award_dt = pd.to_datetime(start, errors="coerce")
                if pd.notna(award_dt) and (founded_dt - award_dt).days > _FOUNDING_GATE_DAYS:
                    continue
            verified.append(award)
        return len(verified)

    notes = []
    total_raw_results = 0
    total_verified_recipients = 0

    # Query 1 — GRANTS
    grant_results, grant_err = _query_usaspending(name, GRANT_CODES, sa_str)
    if grant_err:
        notes.append(f"grants_{grant_err}")
    else:
        total_raw_results += len(grant_results or [])
        out["fed_grants_count_preA"] = _filter_and_count(grant_results)

    time.sleep(0.1)  # tiny breather between the two API calls

    # Query 2 — CONTRACTS (DoD/DARPA/NASA prime contracts)
    contract_results, contract_err = _query_usaspending(name, CONTRACT_CODES, sa_str)
    if contract_err:
        notes.append(f"contracts_{contract_err}")
    else:
        total_raw_results += len(contract_results or [])
        out["fed_contracts_count_preA"] = _filter_and_count(contract_results)

    # Count recipients the LLM accepted (any True in cache)
    total_verified_recipients = sum(1 for v in llm_cache.values() if v)

    out["fed_awards_before_seriesA"] = 1 if (out["fed_grants_count_preA"] > 0
                                              or out["fed_contracts_count_preA"] > 0) else 0

    # Build a clear note explaining the zero result
    if notes:
        out["fed_awards_note"] = "; ".join(notes)
    elif out["fed_awards_before_seriesA"] == 0:
        if total_raw_results == 0:
            out["fed_awards_note"] = "not_found"
        elif total_verified_recipients == 0:
            out["fed_awards_note"] = "all_recipients_rejected_by_llm"
        else:
            out["fed_awards_note"] = "all_awards_post_seriesA"

    return out


df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

for c in ["fed_grants_count_preA", "fed_contracts_count_preA", "fed_awards_before_seriesA"]:
    df[c] = 0
df["fed_awards_note"] = ""

from tqdm import tqdm

# No medical gate — federal awards span all sectors (DoD/DARPA/NASA contracts go
# to deeptech / hardware / aerospace startups, NIH/NSF grants go to medical/basic-science).
print(f"Federal Awards: {len(df)} companies to query (USAspending.gov)")

for i in tqdm(df.index, desc="Federal Awards"):
    name = str(df.loc[i, "Organization_Name"]).strip() if "Organization_Name" in df.columns else ""
    sdate = str(df.loc[i, "seriesA_date"]).strip() if "seriesA_date" in df.columns else ""
    desc = str(df.loc[i, "Description"]).strip()[:250] if "Description" in df.columns else ""
    founded = str(df.loc[i, "Founded Date"]).strip() if "Founded Date" in df.columns else ""
    out = fed_award_signals(
        name,
        sdate if sdate not in ("nan", "NaT", "") else None,
        company_description=desc,
        founded_date=founded if founded not in ("nan", "NaT", "") else "",
    )
    for k, v in out.items():
        df.at[i, k] = v
    time.sleep(DELAY)

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
n_grants = int((df["fed_grants_count_preA"] > 0).sum())
n_contracts = int((df["fed_contracts_count_preA"] > 0).sum())
n_any = int(df["fed_awards_before_seriesA"].sum())
print(f"Federal Awards: {n_any}/{len(df)} companies with any pre-A award "
      f"(grants={n_grants}, contracts={n_contracts})")

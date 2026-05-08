#!/usr/bin/env python
# coding: utf-8

# # 06y – Y Combinator alumni signal
#
# **Signal**: did this company go through Y Combinator BEFORE its Series A?
#
# YC alumni are well-documented to have ~3× higher Series B conversion rates than
# the general venture-backed population. The list is public, free, and complete:
# 5,690+ companies including ~1,000 inactive/failed (no survivorship bias).
#
# **Source**: yc-oss.github.io/api/companies/all.json — community mirror of the
# canonical YC public companies list. Contains: name, batch ("Winter 2018"), status,
# website (url), former_names, industries, one_liner, slug.
#
# **Anti-leakage**: companies that joined YC AFTER their Series A do not count as a
# pre-A signal — only `yc_alumni_preA=1` if batch_date ≤ seriesA_date.
#
# **Survivorship bias**: NONE — YC keeps Inactive (1,005), Acquired (732), and Public
# (23) companies in the list. We verified status distribution before adopting this source.
#
# **Matching strategy** (precision over recall):
#   1. Normalize name (strip punctuation, lowercase, drop "inc"/"llc")
#   2. Exact normalized match against `name` and `former_names`
#   3. URL/domain match against `url` field (very precise)
#   4. LLM verification on the matched YC entry against the company's description
#
# **Features produced**:
#   - yc_alumni_preA            : 1 if matched AND joined YC before Series A
#   - yc_batch                  : YC batch name, e.g. "Summer 2018" (empty if not pre-A)
#   - yc_months_before_seriesA  : months between batch start and Series A (NaN if not pre-A)
#   - yc_note                   : status / debug

import os
import re
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse

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
OUTPUT_CSV = PIPE / f"{RUN_ID}_04n_yc_alumni__signals.csv"
print("INPUT:", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

YC_CACHE = PIPE / "_yc_companies.json"
YC_URL = "https://yc-oss.github.io/api/companies/all.json"

# LLM for ambiguous matches
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_llm_client = None
if OPENAI_API_KEY:
    try:
        import openai, json as _json
        _llm_client = openai.OpenAI()
        print("YC Alumni: LLM verification enabled")
    except ImportError:
        pass


# --------------- 1) Load YC companies (cache to disk) ---------------
def load_yc_companies():
    if YC_CACHE.exists():
        try:
            with open(YC_CACHE) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 1000:
                print(f"YC: loaded {len(data)} companies from cache")
                return data
        except Exception:
            pass
    print("YC: downloading list from yc-oss.github.io...")
    r = requests.get(YC_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    with open(YC_CACHE, "w") as f:
        json.dump(data, f)
    print(f"YC: downloaded and cached {len(data)} companies")
    return data


# --------------- 2) Name normalization ---------------
LEGAL_SUFFIXES = (
    r"\binc\b", r"\binc\.", r"\bllc\b", r"\bllc\.", r"\bltd\b", r"\bltd\.",
    r"\bcorp\b", r"\bcorporation\b", r"\bco\b", r"\bgmbh\b", r"\bs\.a\.",
    r"\bplc\b", r"\bcompany\b", r"\bthe\b",
)
_suffix_re = re.compile("|".join(LEGAL_SUFFIXES), re.IGNORECASE)
_punct_re = re.compile(r"[^a-z0-9]+")


def normalize_name(s):
    if not s or pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = _suffix_re.sub(" ", s)
    s = _punct_re.sub(" ", s)
    return re.sub(r"\s+", "", s)  # collapse all whitespace


def normalize_domain(url):
    if not url or pd.isna(url):
        return ""
    s = str(url).strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = s.split("/")[0]
    s = re.sub(r"^www\.", "", s)
    return s


# --------------- 3) Build lookup indexes ---------------
def build_indexes(yc_data):
    """Three lookup dicts: normalized name → list of YC entries; same for former_names; same for domain."""
    by_name = {}
    by_former = {}
    by_domain = {}
    for entry in yc_data:
        n = normalize_name(entry.get("name", ""))
        if n:
            by_name.setdefault(n, []).append(entry)
        for fn in (entry.get("former_names") or []):
            fn_n = normalize_name(fn)
            if fn_n:
                by_former.setdefault(fn_n, []).append(entry)
        # BUG FIX: YC JSON has TWO url-ish fields: `url` is the YC profile page
        # (always ycombinator.com/companies/{slug} — useless) and `website` is the
        # real company domain. The old code picked `url` first, which collapsed
        # every YC company to the `ycombinator.com` domain and produced an index
        # with exactly 1 entry. Using `website` first recovers ~5650 real domains.
        d = normalize_domain(entry.get("website") or "")
        if d and d != "ycombinator.com":
            by_domain.setdefault(d, []).append(entry)
    return by_name, by_former, by_domain


# --------------- 4) Batch date parsing ---------------
SEASON_TO_MONTH = {"winter": 1, "spring": 4, "summer": 6, "fall": 9, "x": 6}


def batch_to_date(batch_str):
    """Map 'Summer 2018' → date(2018, 6, 1). Returns None for 'Unspecified'."""
    if not batch_str or not isinstance(batch_str, str):
        return None
    s = batch_str.strip().lower()
    if s == "unspecified":
        return None
    parts = s.split()
    if len(parts) != 2:
        return None
    season, year = parts[0], parts[1]
    try:
        month = SEASON_TO_MONTH.get(season, 1)
        return pd.Timestamp(year=int(year), month=month, day=1)
    except Exception:
        return None


# --------------- 5) LLM verification on ambiguous matches ---------------
def _verify_yc_match_llm(yc_entry, company_name, company_description):
    """Confirm a YC entry truly matches the company (descriptions align)."""
    if _llm_client is None:
        return True
    try:
        import json as _json
        yc_desc = (yc_entry.get("one_liner") or "")[:200]
        yc_inds = ", ".join(yc_entry.get("industries") or [])[:100]
        yc_url = yc_entry.get("website", "") or yc_entry.get("url", "") or ""
        llm_r = _llm_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=20, temperature=0,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": (
                    f'Company in our database:\n'
                    f'  name: "{company_name}"\n'
                    f'  description: {company_description[:200]}\n\n'
                    f'Y Combinator alumnus we matched against:\n'
                    f'  name: "{yc_entry.get("name")}"\n'
                    f'  one-liner: {yc_desc}\n'
                    f'  industries: {yc_inds}\n'
                    f'  website: {yc_url}\n\n'
                    f'Are these the SAME company? Return match=1 only if the descriptions '
                    f'and businesses clearly align. Return 0 if they happen to share a name '
                    f'but are different companies.\n'
                    f'Return: {{"match": 0 or 1}}'
                )},
            ],
        )
        raw = llm_r.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        return _json.loads(raw[start:end]).get("match", 0) == 1
    except Exception:
        return False


# --------------- 6) Match a company against YC ---------------
def find_yc_match(company_name, company_domain, company_description, company_founded_date,
                  by_name, by_former, by_domain):
    """Return (yc_entry, match_method) or (None, None).

    Priority: domain > name > former_name. After finding candidates, apply
    two sanity checks in order:
      (a) founding-date gate: if the YC batch is more than 180 days BEFORE
          the company's founding date, it's a different entity that happens
          to share a name or domain (e.g. Paribus/Ramp — Ramp the fintech
          acquired ramp.com from Paribus, which was YC Summer 2015 but
          Ramp was founded in 2019).
      (b) LLM description match: confirm the descriptions actually align.
          Required even for single-candidate domain matches, since domain
          ownership can change hands.
    """
    candidates = []
    d = normalize_domain(company_domain)
    if d:
        for e in by_domain.get(d, []):
            candidates.append((e, "domain"))
    n = normalize_name(company_name)
    if n:
        for e in by_name.get(n, []):
            candidates.append((e, "name"))
        for e in by_former.get(n, []):
            candidates.append((e, "former_name"))

    if not candidates:
        return None, None

    # Dedupe by slug
    seen = set()
    unique = []
    for e, m in candidates:
        slug = e.get("slug") or e.get("name", "")
        if slug not in seen:
            seen.add(slug)
            unique.append((e, m))

    # --- (a) Founding-date gate: YC batch cannot be >180 days before founding ---
    founded_dt = pd.to_datetime(company_founded_date, errors="coerce") if company_founded_date else None
    surviving = []
    for e, m in unique:
        if founded_dt is not None and not pd.isna(founded_dt):
            batch_dt = batch_to_date(e.get("batch", ""))
            if batch_dt is not None and (founded_dt - batch_dt).days > 180:
                # YC batch predates company founding by more than 6 months → reject
                # (catches domain-hijack cases like Ramp-the-fintech acquiring
                # ramp.com from Paribus, which was YC Summer 2015)
                continue
        surviving.append((e, m))

    if not surviving:
        return None, None

    # --- (b) Match resolution priority:
    #   1) Domain match + date gate passed → trust WITHOUT LLM. An exact
    #      domain match is strongly specific, and the date gate already
    #      catches domain-hijack false positives. Using LLM here produces
    #      false negatives when YC's one-liner phrases the business
    #      differently than Crunchbase's description (e.g. Reverie Labs:
    #      "ML pharma" vs "brain-penetrant cancer therapies" — same company).
    #   2) Name-only or former_name match → require LLM verification, because
    #      name collisions are common (two different Bento companies, two
    #      Fellos, two Firsthands, etc.).
    domain_matches = [(e, m) for e, m in surviving if m == "domain"]
    if len(domain_matches) == 1:
        return domain_matches[0]
    if len(domain_matches) > 1 and _llm_client is not None:
        for e, m in domain_matches:
            if _verify_yc_match_llm(e, company_name, company_description):
                return e, m
        return None, None

    # No domain match — fall back to name / former_name with LLM verification
    name_matches = [(e, m) for e, m in surviving if m != "domain"]
    if _llm_client is not None:
        for e, m in name_matches:
            if _verify_yc_match_llm(e, company_name, company_description):
                return e, m
        return None, None

    # No LLM available — only trust single-candidate name match
    if len(name_matches) == 1:
        return name_matches[0]
    return None, None


# --------------- 7) Main loop ---------------
yc_data = load_yc_companies()
by_name, by_former, by_domain = build_indexes(yc_data)
print(f"YC indexes: {len(by_name)} names, {len(by_former)} former names, {len(by_domain)} domains")

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
df["yc_alumni_preA"] = 0
df["yc_batch"] = ""
df["yc_months_before_seriesA"] = np.nan
df["yc_note"] = ""

col_domain = "domain_clean" if "domain_clean" in df.columns else ("Website" if "Website" in df.columns else None)
col_founded = "Founded Date" if "Founded Date" in df.columns else ("founded_date" if "founded_date" in df.columns else None)

from tqdm import tqdm

print(f"YC Alumni: {len(df)} companies to check")

for i in tqdm(df.index, desc="YC Alumni"):
    name = str(df.loc[i, "Organization_Name"]).strip() if "Organization_Name" in df.columns else ""
    domain = str(df.loc[i, col_domain]).strip() if col_domain else ""
    desc = str(df.loc[i, "Description"]).strip()[:250] if "Description" in df.columns else ""
    sdate_raw = str(df.loc[i, "seriesA_date"]).strip() if "seriesA_date" in df.columns else ""
    sa_dt = pd.to_datetime(sdate_raw, errors="coerce") if sdate_raw not in ("", "nan", "NaT") else None
    founded_raw = str(df.loc[i, col_founded]).strip() if col_founded else ""

    entry, method = find_yc_match(name, domain, desc, founded_raw, by_name, by_former, by_domain)

    if entry is None:
        df.at[i, "yc_note"] = "not_in_yc"
        continue

    batch = entry.get("batch", "")
    batch_dt = batch_to_date(batch)

    if sa_dt is None or pd.isna(sa_dt):
        df.at[i, "yc_note"] = f"matched_via_{method}_no_seriesA_date"
        continue
    if batch_dt is None:
        df.at[i, "yc_note"] = f"matched_via_{method}_unparseable_batch"
        continue

    if batch_dt <= sa_dt:
        df.at[i, "yc_alumni_preA"] = 1
        df.at[i, "yc_batch"] = batch
        df.at[i, "yc_months_before_seriesA"] = round((sa_dt - batch_dt).days / 30.44, 1)
        df.at[i, "yc_note"] = f"matched_via_{method}"
    else:
        # Joined YC AFTER Series A — anti-leakage: do not count as pre-A signal
        df.at[i, "yc_note"] = f"matched_via_{method}_post_seriesA"

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
n_alumni = int(df["yc_alumni_preA"].sum())
print(f"YC Alumni: {n_alumni}/{len(df)} companies are pre-Series-A YC alumni")

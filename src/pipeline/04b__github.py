#!/usr/bin/env python
# coding: utf-8

# # 04b – GitHub Signals (simplified)
#
# Binary presence + backlinks count. Sector-filtered to devtools/enterprise only
# (consumer/biotech/hardware companies rarely have meaningful GitHub presence).
#
# Coverage: ~5-8% of devtools/enterprise companies
# Output: github_before_seriesA (0/1), backlinks_github_count_preA, github_note

import os
import re
import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

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
ALLOW_OVERWRITE = True
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
OUTPUT_CSV = PIPE / f"{RUN_ID}_04b_github__signals.csv"
print("INPUT:", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

GITHUB_API = "https://api.github.com"
HEADERS = {"User-Agent": "ThesisResearch/1.0 (Python)", "Accept": "application/vnd.github.v3+json"}
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip() or None
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"
    # Verify token is valid before processing thousands of companies
    _check = requests.get(f"{GITHUB_API}/rate_limit", headers=HEADERS, timeout=10)
    if _check.status_code == 401:
        raise ValueError("GITHUB_TOKEN is invalid/expired — refusing to run (would silently return 0 for all companies)")
    print(f"GITHUB_TOKEN: SET — {_check.json().get('rate',{}).get('limit','?')} req/hour")
else:
    print("GITHUB_TOKEN: NOT SET — rate limit 60 req/hour")
DELAY = 1.2

# Sector filter: only query GitHub for companies likely to have repos
GITHUB_SECTORS = {"devtools", "enterprise"}

# LLM for repo verification
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_llm_client = None
if OPENAI_API_KEY:
    try:
        import openai
        _llm_client = openai.OpenAI()
        print("GitHub: LLM repo verification enabled")
    except ImportError:
        print("WARNING: openai not installed")


def _domain_slug(domain):
    """'www.syntiant.com' -> 'syntiant'."""
    d = re.sub(r"^https?://", "", (domain or "")).split("/")[0].lower()
    d = re.sub(r"^www\.", "", d)
    d = re.sub(r"\.(com|io|ai|co|net|org|tech|app|dev)$", "", d)
    return re.sub(r"[^a-z0-9]", "", d)


def _name_slug(name):
    """'Bear Robotics' -> 'bearrobotics'."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _name_slug_dashed(name):
    """'Bear Robotics' -> 'bear-robotics'."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def _try_org(candidate):
    """Try GitHub org endpoint. Returns org dict or None."""
    if not candidate or len(candidate) < 3:
        return None
    try:
        r = requests.get(f"{GITHUB_API}/orgs/{candidate}", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _verify_repo_llm(owner, repo, company_name, company_description):
    """LLM verification: is this GitHub repo owned by this company?"""
    if _llm_client is None:
        return True
    try:
        r = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}/readme",
                        headers={**HEADERS, "Accept": "application/vnd.github.v3.raw"},
                        timeout=10)
        if r.status_code == 200:
            readme_text = r.text[:500]
        else:
            r2 = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=HEADERS, timeout=10)
            desc = r2.json().get("description", "") if r2.status_code == 200 else ""
            readme_text = f"Repo: {owner}/{repo}. Description: {desc}"

        llm_r = _llm_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=20, temperature=0,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": (
                    f'Is this GitHub repository owned by or about the company "{company_name}" '
                    f'({company_description[:150]})?\n'
                    f'GitHub content: {readme_text[:400]}\n'
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


def _github_backlinks_count(domain, created_before=None):
    """Count GitHub repos referencing this company's domain, created before Series A."""
    d = re.sub(r"^https?://", "", (domain or "")).split("/")[0].lower()
    if not d or len(d) < 3:
        return 0
    try:
        q = d
        if created_before:
            q = f"{d} created:<{created_before}"
        r = requests.get(f"{GITHUB_API}/search/repositories",
                        params={"q": q, "per_page": 1},
                        headers=HEADERS, timeout=10)
        return int(r.json().get("total_count", 0)) if r.status_code == 200 else 0
    except Exception:
        return 0


def _find_github_presence(name, domain, description):
    """Search for company's GitHub org/repo. Returns (owner, repo_name) or None."""
    d_slug = _domain_slug(domain)
    n_slug = _name_slug(name)
    n_dashed = _name_slug_dashed(name)
    candidates = []

    # Try org by domain slug, then name slugs
    for slug in [d_slug, n_slug, n_dashed]:
        if not slug or len(slug) < 3 or slug in [s for s, _ in candidates]:
            continue
        org = _try_org(slug)
        if org:
            login = org.get("login", slug)
            try:
                r = requests.get(f"{GITHUB_API}/orgs/{login}/repos",
                               params={"sort": "stargazers", "per_page": 1},
                               headers=HEADERS, timeout=10)
                if r.status_code == 200 and r.json():
                    repo = r.json()[0]
                    candidates.append((repo.get("owner", {}).get("login", login), repo.get("name", "")))
            except Exception:
                pass
        time.sleep(0.3)

    # Repo search fallback
    if len(candidates) < 3 and name:
        try:
            r = requests.get(f"{GITHUB_API}/search/repositories",
                           params={"q": f'"{name[:50]}" in:name', "per_page": 3, "sort": "stars"},
                           headers=HEADERS, timeout=10)
            if r.status_code == 200:
                for item in r.json().get("items", []):
                    owner = item.get("owner", {}).get("login", "")
                    rname = item.get("name", "")
                    if owner and rname and (owner, rname) not in candidates:
                        candidates.append((owner, rname))
        except Exception:
            pass

    # LLM verify candidates (max 3)
    for owner, rname in candidates[:3]:
        if _verify_repo_llm(owner, rname, name, description):
            return owner, rname
        time.sleep(0.3)

    return None


def _check_repo_before_seriesA(owner, repo, series_a_date):
    """Check if repo was created before Series A date."""
    try:
        r = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            created = r.json().get("created_at", "")[:10]
            if created and series_a_date:
                return created <= str(series_a_date)[:10]
    except Exception:
        pass
    return False


# =====================================================================
# Main
# =====================================================================

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
col_name = "Organization_Name"
col_domain = "domain_clean" if "domain_clean" in df.columns else "Website"

# Initialize output columns
df["github_before_seriesA"] = 0
df["backlinks_github_count_preA"] = 0
df["github_note"] = ""

# Sector filter
has_sector = "startup_sector" in df.columns
if has_sector:
    sector_mask = df["startup_sector"].str.lower().isin(GITHUB_SECTORS)
    n_skip = (~sector_mask).sum()
    n_query = sector_mask.sum()
    print(f"GitHub: {n_query} companies in scope (devtools/enterprise), {n_skip} skipped by sector filter")
else:
    sector_mask = pd.Series(True, index=df.index)
    print("GitHub: no startup_sector column — querying all companies")

from tqdm import tqdm
for i in tqdm(df.index, desc="GitHub"):
    if not sector_mask.iloc[i] if has_sector else False:
        df.at[i, "github_note"] = "skipped_sector"
        continue

    name = str(df.loc[i, col_name]).strip()
    domain = str(df.loc[i, col_domain]).strip()
    description = str(df.loc[i, "Description"]).strip()[:200] if "Description" in df.columns else ""
    series_a_val = str(df.loc[i, "seriesA_date"]).strip() if "seriesA_date" in df.columns else ""
    series_a_val = series_a_val if series_a_val not in ("nan", "NaT", "") else None

    # Backlinks: repos referencing company domain (pre-Series A)
    created_before = None
    if series_a_val:
        try:
            created_before = pd.to_datetime(series_a_val, errors="coerce").strftime("%Y-%m-%d")
        except Exception:
            pass
    if domain:
        df.at[i, "backlinks_github_count_preA"] = _github_backlinks_count(domain, created_before)
        time.sleep(0.3)

    # Own GitHub presence
    match = _find_github_presence(name, domain, description)
    if match:
        owner, rname = match
        if series_a_val and _check_repo_before_seriesA(owner, rname, series_a_val):
            df.at[i, "github_before_seriesA"] = 1
            df.at[i, "github_note"] = f"verified:{owner}/{rname}"
        else:
            df.at[i, "github_note"] = f"found:{owner}/{rname} (after seriesA or no date)"
    else:
        df.at[i, "github_note"] = "no_repo_found"

    time.sleep(DELAY)

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
found = (df["github_before_seriesA"] == 1).sum()
print(f"GitHub: {found}/{len(df)} have pre-Series A repo")

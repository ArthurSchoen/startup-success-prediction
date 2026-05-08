#!/usr/bin/env python
# coding: utf-8

# # 05g – Packages (B: PyPI / npm / crates.io)
#
# **Input** : CSV de l'étape 05f (github), avec `Organization_Name`, `domain_clean`, `seriesA_date`.
#
# **Signal** : si produit dev (package trouvé) :
# - `pkg_first_release_date` : date de publication du premier release
# - `pkg_releases_count` : nb de releases (tous registres confondus)
# - `pkg_downloads_recent` : downloads récents (PyPI via pypistats.org si dispo)
# - `pkg_registry` : PyPI | npm | crates (premier trouvé)
#
# Versionning régulier = exécution. Pour deeptech / dev tools, souvent pertinent.
#
# ⚠️ Matching org/domain → nom de package heuristique (slug du nom, domaine sans TLD).
#
# ---
#
# **⚠️ Biais « Téléchargements »** : Les registres (npm, PyPI) ne donnent généralement que les stats des **30 derniers jours** par défaut. Utiliser les téléchargements de 2026 pour prédire une réussite en 2021 = **leakage critique**. **Correction** : **PePy** (PyPI) ou **npm-stat** pour l'historique mois par mois. En attendant : **pkg_downloads_recent** = valeur actuelle (à interpréter avec prudence). **pkg_momentum_3m** = placeholder (nécessite PePy/npm-stat).
#
# **⚠️ Ambiguïté du mainteneur** : Beaucoup de packages sont créés par des particuliers avant enregistrement de la société. **Recommandation** : **pkg_maintainer_email_domain_match** = 1 si l'email du mainteneur contient le domaine de la startup.

# In[ ]:


import os
import re
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline", Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
PIPE = next((p for p in PIPE_CANDIDATES if p.exists() and (p / ".run_id").exists()), None) or (PIPE_CANDIDATES[0] if PIPE_CANDIDATES[0].exists() else Path.home() / "Desktop" / "Data_thesis" / "Pipeline")
BASE = PIPE.parent
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

def pick_next_file_for_step(pipe_dir, prev_step, next_step_prefix, allow_overwrite=False):
    candidates = []
    for p in pipe_dir.glob(f"*{prev_step}*.csv"):
        if "__midpoint" in p.name:
            continue
        run_id = p.name.split("_", 1)[0]
        next_exists = any(pipe_dir.glob(f"{run_id}{next_step_prefix}*.csv"))
        if allow_overwrite or not next_exists:
            candidates.append((run_id, p))
    if not candidates:
        raise FileNotFoundError(f"No input for prev_step={prev_step}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

CHOSEN_RUN_ID = os.environ.get("PIPELINE_RUN_ID", "").strip() or ((PIPE / ".run_id").read_text().strip() if (PIPE / ".run_id").exists() else "")
ALLOW_OVERWRITE = True
_base_override = os.environ.get("PIPELINE_BASE_CSV", "").strip()
if _base_override and Path(_base_override).exists():
    INPUT_CSV = Path(_base_override)
else:
    INPUT_CSV = None  # set below
if INPUT_CSV is None:
    prev_steps = ["_04b_github__signals", "_04a_wikipedia__signals", "_04i_epo_patents__signals", "_04c_packages__signals", "_04d_openalex__signals", "_04e_hacker_news__signals"]
    INPUT_CSV = None
    if CHOSEN_RUN_ID:
        for fname in [f"{CHOSEN_RUN_ID}_04b_github__signals.csv", f"{CHOSEN_RUN_ID}_04a_wikipedia__signals.csv", f"{CHOSEN_RUN_ID}_04i_epo_patents__signals.csv", f"{CHOSEN_RUN_ID}_04c_packages__signals.csv", f"{CHOSEN_RUN_ID}_04d_openalex__signals.csv", f"{CHOSEN_RUN_ID}_04e_hacker_news__signals.csv"]:
            p = PIPE / fname
            if p.exists():
                INPUT_CSV = p
                break
    if INPUT_CSV is None:
        for prev in prev_steps:
            try:
                INPUT_CSV = pick_next_file_for_step(PIPE, prev, "_04c_packages__", ALLOW_OVERWRITE)
                break
            except FileNotFoundError:
                continue
    if INPUT_CSV is None:
        archive_dir = PIPE / "archive"
        for p in prev_steps:
            pattern = "*" + p.replace("_signals", "").strip("_") + "*.csv"
            candidates = list(archive_dir.glob(f"*/{pattern}"))
            candidates = [x for x in candidates if "__midpoint" not in x.name]
            if candidates:
                candidates.sort(key=lambda p: (p.parent.name, p.name), reverse=True)
                INPUT_CSV = candidates[0]
                print("(entrée depuis l'archive)")
                break
    if INPUT_CSV is None:
        raise FileNotFoundError("No _04*__signals in Pipeline or archive")
RUN_ID = run_id_from_filename(INPUT_CSV)
OUTPUT_CSV = PIPE / f"{RUN_ID}_04c_packages__signals.csv"
print("INPUT:", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

HEADERS = {"User-Agent": "ThesisResearch/1.0 (Python)"}
DELAY = 0.5

def slug(s):
    s = (s or "").strip().lower()
    return re.sub(r"[^a-z0-9]", "", s)[:40]

def domain_stem(domain):
    s = (domain or "").strip().lower()
    s = re.sub(r"^https?://", "", s).split("/")[0]
    return s.split(".")[0] if s else ""


def try_pypi(name_candidate):
    out = {"registry": "", "first_date": "", "releases": 0, "release_dates": [], "release_versions": [], "maintainer_email": ""}
    if not name_candidate or len(name_candidate) < 2:
        return out
    name_candidate = name_candidate.replace("_", "-").lower()[:40]
    try:
        r = requests.get(f"https://pypi.org/pypi/{name_candidate}/json", headers=HEADERS, timeout=10)
        if r.status_code == 404:
            return out
        if r.status_code == 429:
            print(f"  RATE-LIMITED on PyPI for {name_candidate}")
            time.sleep(5)
            return out
        if r.status_code != 200:
            print(f"  PyPI HTTP {r.status_code} for {name_candidate}")
            return out
        d = r.json()
        out["registry"] = "pypi"
        urls = d.get("releases", {})
        out["releases"] = len(urls)
        out["maintainer_email"] = (d.get("info") or {}).get("author_email") or ""
        version_dates = []
        for k, v in urls.items():
            if v and isinstance(v, list):
                up = (v[0].get("upload_time", "") or "")[:10]
                if up:
                    version_dates.append((k, up))
        if version_dates:
            dates_only = [x[1] for x in version_dates]
            out["first_date"] = min(dates_only)
            out["release_dates"] = sorted(dates_only)
            out["release_versions"] = [x[0] for x in sorted(version_dates, key=lambda x: x[1])]
    except requests.exceptions.Timeout:
        print(f"  PyPI timeout for {name_candidate}")
    except requests.exceptions.ConnectionError:
        print(f"  PyPI connection error for {name_candidate}")
    except Exception as e:
        print(f"  PyPI error for {name_candidate}: {e}")
    return out

def try_npm(name_candidate):
    out = {"registry": "", "first_date": "", "releases": 0, "release_dates": [], "release_versions": [], "maintainer_email": ""}
    if not name_candidate or len(name_candidate) < 2:
        return out
    name_candidate = name_candidate.replace("_", "-").lower()[:40]
    try:
        r = requests.get(f"https://registry.npmjs.org/{name_candidate}", headers=HEADERS, timeout=10)
        if r.status_code == 404:
            return out
        if r.status_code == 429:
            print(f"  RATE-LIMITED on npm for {name_candidate}")
            time.sleep(5)
            return out
        if r.status_code != 200:
            print(f"  npm HTTP {r.status_code} for {name_candidate}")
            return out
        d = r.json()
        if d.get("error") == "Not found":
            return out
        out["registry"] = "npm"
        times = d.get("time", {})
        if isinstance(times, dict):
            created = times.get("created", "")
            if created:
                out["first_date"] = created[:10]
            skip = {"created", "modified", "unpublished"}
            version_dates = [(k, (v or "")[:10]) for k, v in times.items() if k not in skip and v]
            if version_dates:
                version_dates.sort(key=lambda x: x[1])
                out["release_dates"] = [x[1] for x in version_dates]
                out["release_versions"] = [x[0] for x in version_dates]
            out["releases"] = len(version_dates)
    except requests.exceptions.Timeout:
        print(f"  npm timeout for {name_candidate}")
    except requests.exceptions.ConnectionError:
        print(f"  npm connection error for {name_candidate}")
    except Exception as e:
        print(f"  npm error for {name_candidate}: {e}")
    return out


def package_signals(name, domain):
    default = {"registry": "", "first_date": "", "releases": 0, "release_dates": [], "release_versions": [], "maintainer_email": "", "matched_candidate": ""}
    candidates = []
    if name:
        candidates.append(slug(name))
    if domain:
        candidates.append(domain_stem(domain))
    candidates = list(dict.fromkeys([c for c in candidates if c and len(c) >= 2]))
    for c in candidates:
        results = []
        for try_fn in [try_pypi, try_npm]:
            out = try_fn(c)
            if out.get("registry"):
                results.append(out)
            time.sleep(0.2)
        if results:
            best = max(results, key=lambda x: x.get("releases", 0))
            best["matched_candidate"] = c
            return best
    return default


# LLM for package verification
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_llm_client = None
if OPENAI_API_KEY:
    try:
        import openai, json as _json
        _llm_client = openai.OpenAI()
        print("Packages: LLM verification enabled")
    except ImportError:
        print("WARNING: openai not installed — skipping LLM verification")


def _verify_package_llm(pkg_name, pkg_registry, company_name, company_description):
    """LLM check: is this package from this company?"""
    if _llm_client is None:
        return True
    try:
        import json as _json
        llm_r = _llm_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=20, temperature=0,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": (
                    f'Is the {pkg_registry} package "{pkg_name}" likely published by '
                    f'the company "{company_name}" ({company_description[:150]})?\n'
                    f'Return: {{"match": 0 or 1}}'
                )},
            ],
        )
        raw = llm_r.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = _json.loads(raw[start:end])
        return parsed.get("match", 0) == 1
    except Exception:
        return True

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
col_name = "Organization_Name"
col_domain = "domain_clean" if "domain_clean" in df.columns else "Website"
df["pkg_first_release_date"] = ""
df["pkg_releases_count_preA"] = 0
df["pkg_before_seriesA"] = 0
df["pkg_registry"] = ""
df["pkg_note"] = ""

from tqdm import tqdm
for i in tqdm(df.index, desc="Packages"):
    name = str(df.loc[i, col_name]).strip() if col_name in df.columns else ""
    domain = str(df.loc[i, col_domain]).strip() if col_domain in df.columns else ""
    series_a_str = str(df.loc[i, "seriesA_date"]).strip() if "seriesA_date" in df.columns else ""
    desc = str(df.loc[i, "Description"]).strip()[:200] if "Description" in df.columns else ""

    out = package_signals(name, domain)

    # PRE-A DATE CHECK — if first_date > Series A, discard
    sA = pd.to_datetime(series_a_str, errors="coerce") if series_a_str not in ("", "nan", "NaT") else None
    first_d = pd.to_datetime(out.get("first_date", ""), errors="coerce") if out.get("first_date") else None

    if out.get("registry") and (first_d is None or pd.isna(first_d) or sA is None or pd.isna(sA) or first_d > sA):
        df.at[i, "pkg_note"] = "post_seriesA_or_no_date"
        time.sleep(DELAY)
        continue

    # LLM verification (use actual matched candidate, not slug(name))
    if out.get("registry") and _llm_client is not None:
        matched = out.get("matched_candidate", "")
        if not _verify_package_llm(matched, out["registry"], name, desc):
            df.at[i, "pkg_note"] = "llm_rejected"
            time.sleep(DELAY)
            continue

    # Founding-date sanity: discard if package predates company by > 1 year
    founded_val = str(df.loc[i, "founded_date"]).strip() if "founded_date" in df.columns else ""
    if out.get("first_date") and founded_val and founded_val not in ("", "nan", "NaT"):
        pkg_dt = pd.to_datetime(out["first_date"], errors="coerce")
        founded_dt = pd.to_datetime(founded_val, errors="coerce")
        if pd.notna(pkg_dt) and pd.notna(founded_dt) and (founded_dt - pkg_dt).days > 365:
            df.at[i, "pkg_note"] = "pkg_predates_company"
            time.sleep(DELAY)
            continue

    # Count releases before Series A
    releases_preA = 0
    if sA and pd.notna(sA) and out.get("release_dates"):
        releases_preA = sum(1 for d in out["release_dates"] if d and pd.to_datetime(d, errors="coerce") <= sA)

    df.at[i, "pkg_first_release_date"] = out.get("first_date", "")
    df.at[i, "pkg_releases_count_preA"] = releases_preA
    df.at[i, "pkg_registry"] = out.get("registry", "")
    time.sleep(DELAY)

df["pkg_before_seriesA"] = (df["pkg_releases_count_preA"] > 0).astype(int)

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
print("Packages with pre-A releases:", (df["pkg_releases_count_preA"] > 0).sum())

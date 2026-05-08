#!/usr/bin/env python
# coding: utf-8

# # 05q – Clinical Trials (ClinicalTrials.gov)
#
# **Signal**: Clinical trial activity before Series A.
# Relevant for biotech, pharma, medical device startups.
# - ct_trials_count_preA: number of trials involving the company before Series A
# - ct_earliest_phase: earliest trial phase (1-4, 0 = not applicable)
# - ct_before_seriesA (0/1): any trial started before Series A
#
# API: ClinicalTrials.gov v2 (free, no key, no rate limit issues)

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
OUTPUT_CSV = PIPE / f"{RUN_ID}_04g_clinical_trials__signals.csv"
print("INPUT:", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

CT_API = "https://clinicaltrials.gov/api/v2/studies"
DELAY = 0.3

PHASE_MAP = {
    "EARLY_PHASE1": 0.5, "PHASE1": 1, "PHASE2": 2, "PHASE3": 3, "PHASE4": 4,
    "NA": 0,
}

# Keywords that mark a company as medical/biotech — Clinical Trials signal is
# irrelevant for non-medical companies and risks name-collision false positives.
MEDICAL_KEYWORDS = (
    "biotech", "biotechnology", "biopharma", "pharma", "pharmaceutic",
    "therapeutic", "medical", "medicine", "health", "clinical",
    "diagnos", "life science", "drug", "oncology", "vaccine", "genomic",
    "neuroscience", "cardio", "immun", "cell therapy", "gene therapy",
)


def is_medical_company(industries_str, startup_sector=""):
    """Return True if the company is biotech/pharma/medical based on Industries field or sector."""
    s = str(industries_str or "").lower()
    if any(k in s for k in MEDICAL_KEYWORDS):
        return True
    # Backstop: deeptech sector companies may have medical without the keyword
    return False


# LLM for sponsor verification (name collision defense)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_llm_client = None
if OPENAI_API_KEY:
    try:
        import openai, json as _json
        _llm_client = openai.OpenAI()
        print("Clinical Trials: LLM sponsor verification enabled")
    except ImportError:
        pass


def _verify_sponsor_llm(sponsor_name, company_name, company_description):
    """LLM check: is this trial sponsor the same entity as the company?
    Handles subsidiary/legal-entity naming (e.g., 'ModernaTX, Inc.' ≡ 'Moderna')."""
    if _llm_client is None:
        return True
    try:
        import json as _json
        llm_r = _llm_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=20, temperature=0,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": (
                    f'Company: "{company_name}" — {company_description[:200]}\n'
                    f'ClinicalTrials.gov sponsor: "{sponsor_name}"\n\n'
                    f'Is the sponsor the SAME company (possibly under a legal / subsidiary '
                    f'name like "XYZ Therapeutics, Inc.", "XYZ Pharmaceuticals", "XYZ Bio", '
                    f'"XYZ Medical, LLC")? Return match=1 if yes, 0 if it is a different '
                    f'organization (hospital, university, unrelated company with similar name).\n'
                    f'Return: {{"match": 0 or 1}}'
                )},
            ],
        )
        raw = llm_r.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        return _json.loads(raw[start:end]).get("match", 0) == 1
    except Exception:
        return False


def ct_signals(company_name, series_a_date, industries="", company_description="",
               name_distinctive=1):
    out = {
        "ct_trials_count_preA": 0,
        "ct_min_phase_preA": np.nan,
        "ct_max_phase_preA": np.nan,
        "ct_before_seriesA": 0,
        "ct_note": "",
    }
    name = (company_name or "").strip()
    if not name or len(name) < 3 or name in ("nan", "None"):
        return out

    # Gate 1: medical/biotech only — skip non-medical companies entirely
    if not is_medical_company(industries):
        out["ct_note"] = "skipped_non_medical"
        return out

    # NOTE: the non-distinctive skip was removed here. ClinicalTrials.gov
    # searches by sponsor legal name, which is structured. The downstream
    # sponsor verifier handles collisions like "Cell Therapeutics" vs the
    # word "cell". The non_medical gate above already filters ~70% of the
    # dataset, so this second gate was just dropping BIO companies with
    # common names from being checked at all.
    try:
        sa_dt = pd.to_datetime(series_a_date, errors="coerce")
        if pd.isna(sa_dt):
            out["ct_note"] = "no_seriesA_date"
            return out
        sa_str = sa_dt.strftime("%Y-%m-%d")

        params = {
            "query.term": name,
            "filter.overallStatus": "ACTIVE_NOT_RECRUITING,COMPLETED,ENROLLING_BY_INVITATION,NOT_YET_RECRUITING,RECRUITING,SUSPENDED,TERMINATED,WITHDRAWN,UNKNOWN",
            "pageSize": 50,
            "format": "json",
        }
        r = requests.get(CT_API, params=params, timeout=15)
        if r.status_code != 200:
            out["ct_note"] = f"http_{r.status_code}"
            return out

        studies = r.json().get("studies", [])
        if not studies:
            out["ct_note"] = "not_found"
            return out

        name_lower = name.lower().strip()

        def _loose_match(org):
            """Loose match: catches subsidiaries like 'ModernaTX, Inc.' for 'Moderna'.
            Returns a quality tier: 'strict' (exact/startswith) or 'loose' (substring)."""
            if not org:
                return None
            if org == name_lower:
                return "strict"
            if org.startswith(name_lower + " ") or org.startswith(name_lower + ","):
                return "strict"
            if name_lower.startswith(org + " ") or name_lower.startswith(org + ","):
                return "strict"
            # Loose substring match (requires LLM verification downstream)
            if name_lower in org or org in name_lower:
                return "loose"
            return None

        # Memoize LLM verdicts per unique sponsor string to avoid duplicate calls
        llm_cache = {}
        def _sponsor_ok(org_raw):
            if org_raw in llm_cache:
                return llm_cache[org_raw]
            ok = _verify_sponsor_llm(org_raw, name, company_description)
            llm_cache[org_raw] = ok
            return ok

        preA_trials = []
        phases = []
        for study in studies:
            proto = study.get("protocolSection", {})
            status = proto.get("statusModule", {})
            start_date = status.get("startDateStruct", {}).get("date", "")

            sponsor_module = proto.get("sponsorCollaboratorsModule", {})
            lead_raw = sponsor_module.get("leadSponsor", {}).get("name", "") or ""
            collab_raw = [c.get("name", "") or "" for c in sponsor_module.get("collaborators", [])]
            all_orgs_raw = [lead_raw] + collab_raw
            all_orgs_lower = [o.lower().strip() for o in all_orgs_raw]

            # Find best-quality match among all sponsors
            best_tier = None
            matched_raw = None
            for raw, lo in zip(all_orgs_raw, all_orgs_lower):
                tier = _loose_match(lo)
                if tier == "strict":
                    best_tier, matched_raw = "strict", raw
                    break
                if tier == "loose" and best_tier is None:
                    best_tier, matched_raw = "loose", raw
            if best_tier is None:
                continue

            # Strict matches are accepted as-is; loose matches need LLM confirmation
            if best_tier == "loose" and _llm_client is not None:
                if not _sponsor_ok(matched_raw):
                    continue
            elif best_tier == "loose" and _llm_client is None:
                # No LLM available — don't trust loose matches
                continue

            # Anti-leakage: trial must have started on or before Series A
            if start_date and start_date[:10] <= sa_str:
                preA_trials.append(study)
                design = proto.get("designModule", {})
                for p in design.get("phases", []) or []:
                    phase_val = PHASE_MAP.get(p, 0)
                    if phase_val > 0:
                        phases.append(phase_val)

        out["ct_trials_count_preA"] = len(preA_trials)
        out["ct_before_seriesA"] = 1 if preA_trials else 0
        if phases:
            out["ct_min_phase_preA"] = min(phases)
            out["ct_max_phase_preA"] = max(phases)

    except Exception as e:
        out["ct_note"] = f"error: {type(e).__name__}"

    return out


df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

for c in ["ct_trials_count_preA", "ct_before_seriesA"]:
    df[c] = 0
df["ct_min_phase_preA"] = np.nan
df["ct_max_phase_preA"] = np.nan
df["ct_note"] = ""

from tqdm import tqdm

# Pre-filter: only medical/biotech companies are eligible. Non-medical companies
# get ct_note="skipped_non_medical" and zero on all other columns — no API call,
# no sleep, no LLM call. This is ~75% of the dataset for typical VC portfolios.
medical_mask = df["Industries"].apply(is_medical_company) if "Industries" in df.columns else pd.Series([True]*len(df), index=df.index)
df.loc[~medical_mask, "ct_note"] = "skipped_non_medical"
eligible_idx = df.index[medical_mask].tolist()
print(f"Clinical Trials: {len(eligible_idx)}/{len(df)} medical companies eligible (skipping {len(df)-len(eligible_idx)} non-medical)")

for i in tqdm(eligible_idx, desc="Clinical Trials"):
    name = str(df.loc[i, "Organization_Name"]).strip()
    sdate = str(df.loc[i, "seriesA_date"]).strip() if "seriesA_date" in df.columns else ""
    inds = str(df.loc[i, "Industries"]).strip() if "Industries" in df.columns else ""
    desc = str(df.loc[i, "Description"]).strip()[:250] if "Description" in df.columns else ""
    nd = df.loc[i, "name_distinctive"] if "name_distinctive" in df.columns else 1
    out = ct_signals(
        name,
        sdate if sdate not in ("nan", "NaT", "") else None,
        industries=inds,
        company_description=desc,
        name_distinctive=nd,
    )
    for k, v in out.items():
        df.at[i, k] = v
    time.sleep(DELAY)

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
found = int(df["ct_before_seriesA"].sum())
print(f"Clinical Trials: {found}/{len(eligible_idx)} medical companies with pre-A trials")

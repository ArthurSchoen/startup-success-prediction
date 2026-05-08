#!/usr/bin/env python
# coding: utf-8

# # 04i – Patent Signals via Google Patents Public Dataset on BigQuery
#
# **Replaces EPO OPS for this step.** The filename stays the same so the
# pipeline orchestrator continues to pick it up; the previous EPO-based
# implementation is archived at Code/_archive/04i__epo_patents.py.bak.YYYYMMDD.
#
# ## Why BigQuery instead of EPO OPS?
#
# EPO OPS free tier has a hard 4 GB/week quota (€2,800/year for the paid tier).
# In the new2k run 43% of companies (236/547) returned `quota_exceeded`,
# which was dropping real patent signals silently. BigQuery's
# `patents-public-data.patents.publications` dataset is an official Google
# Cloud Public Dataset sourced from IFI Claims, covering ~98M patents
# worldwide (including EP filings, so no real coverage loss vs EPO).
#
# ## Cost model
#
# BigQuery free tier: **1 TB/month** of query processing. Our single bulk
# query scans ~6 GB, so we have roughly 170× headroom. We set
# `maximum_bytes_billed=50 GB` as a safety cap in case the query plan
# misestimates.
#
# ## Features produced (same schema as the old EPO step)
#
#   - patent_count_preA_company  : verified pre-Series A patents
#   - has_patent_preA_company    : 0/1
#   - patent_earliest_date_preA  : earliest filing date (YYYY-MM-DD)
#   - patent_note                : audit / trace
#   - patent_matched_assignees   : (new) which assignee strings matched

import os
import re
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

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
if INPUT_CSV is None and CHOSEN_RUN_ID:
    for search_dir in [PIPE, PIPE / "archive" / CHOSEN_RUN_ID]:
        p = search_dir / f"{CHOSEN_RUN_ID}_05_webtraction__signals.csv"
        if p.exists():
            INPUT_CSV = p
            break
if INPUT_CSV is None:
    candidates = sorted(PIPE.glob("*_05_webtraction__signals.csv"),
                        key=lambda x: x.name, reverse=True)
    if candidates:
        INPUT_CSV = candidates[0]
if INPUT_CSV is None:
    candidates = sorted((PIPE / "archive").glob("*/*_05_webtraction__signals.csv"),
                        key=lambda x: x.name, reverse=True)
    if candidates:
        INPUT_CSV = candidates[0]
if INPUT_CSV is None:
    raise FileNotFoundError("No _05_webtraction__signals.csv found in Pipeline or archive")

RUN_ID = run_id_from_filename(INPUT_CSV)
OUTPUT_CSV = PIPE / f"{RUN_ID}_04i_epo_patents__signals.csv"
print("INPUT :", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

# ── Google Cloud / BigQuery setup ──────────────────────────────────────────
GAC = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
if not GAC or not Path(GAC).exists():
    raise ValueError(
        "GOOGLE_APPLICATION_CREDENTIALS must be set in .env and point to a valid "
        "service-account JSON file with BigQuery access. Got: " + repr(GAC))

try:
    from google.cloud import bigquery
except ImportError:
    raise ImportError(
        "google-cloud-bigquery is not installed. Run: pip install google-cloud-bigquery")

bq_client = bigquery.Client()
print(f"BigQuery project: {bq_client.project}")

# Safety cap: if the query plan tries to bill more than 50 GB, fail rather
# than proceed. Our actual scan is ~6 GB; this cap exists only to catch a
# query that accidentally stops using the partition filter.
MAX_BYTES_BILLED = 50 * 1024 * 1024 * 1024  # 50 GB

# ── LLM for applicant disambiguation ───────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_llm_client = None
if OPENAI_API_KEY:
    try:
        import openai
        _llm_client = openai.OpenAI()
        print("LLM applicant verification enabled")
    except ImportError:
        print("WARNING: openai package not installed — skipping LLM verification")
else:
    print("WARNING: OPENAI_API_KEY not set — skipping LLM verification")


# ── Helper: normalize company name for matching ────────────────────────────
_SQL_SAFE = re.compile(r"[^A-Za-z0-9 &'\-.,]")
_LEGAL_SUFFIX = re.compile(r"\b(inc|inc\.|llc|llc\.|ltd|ltd\.|corp|corporation|co|gmbh|plc|limited|company)\b\.?", re.I)


def normalize_for_match(name):
    """Produce an UPPERCASE canonical form suitable for LIKE matching.
    Strips legal suffixes and compresses whitespace. Returns '' if unusable.
    """
    if not name or pd.isna(name):
        return ""
    s = str(name).strip()
    s = _LEGAL_SUFFIX.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _SQL_SAFE.sub("", s)
    return s.upper()


# ── Build the bulk query ───────────────────────────────────────────────────
# We use a LIKE-join on assignee_harmonized.name to catch common legal-suffix
# variations ("ATOM COMPUTING", "ATOM COMPUTING INC", "ATOM COMPUTING INC,",
# etc.). country_code='US' handles one class of false positives
# (MOTION2AI vs MOTION2AI KOREA). The final disambiguation happens in the
# LLM verifier.

BULK_QUERY_TEMPLATE = """
WITH targets AS (
  SELECT * FROM UNNEST([
    {targets_struct_list}
  ])
),
matches AS (
  -- One row per (target, patent-assignee pair). Important: NO GROUP BY here
  -- so that multiple patents filed on the same day stay as separate rows.
  SELECT
    t.target_idx,
    t.target_name,
    t.sA_cutoff,
    a.name AS assignee_name,
    p.publication_number,
    p.filing_date,
    (SELECT tl.text FROM UNNEST(p.title_localized) tl LIMIT 1) AS title_text
  FROM `patents-public-data.patents.publications` p,
       UNNEST(p.assignee_harmonized) a
  JOIN targets t
    ON (
      UPPER(a.name) = t.target_name
      OR UPPER(a.name) LIKE CONCAT(t.target_name, ' INC%')
      OR UPPER(a.name) LIKE CONCAT(t.target_name, ' LLC%')
      OR UPPER(a.name) LIKE CONCAT(t.target_name, ' CORP%')
      OR UPPER(a.name) LIKE CONCAT(t.target_name, ' CO')
      OR UPPER(a.name) LIKE CONCAT(t.target_name, ', INC%')
      OR UPPER(a.name) LIKE CONCAT(t.target_name, ',%')
    )
  WHERE a.country_code = 'US'
    AND p.filing_date IS NOT NULL
    AND p.filing_date BETWEEN 19900101 AND t.sA_cutoff
)
SELECT
  target_idx,
  target_name,
  sA_cutoff,
  assignee_name,
  COUNT(DISTINCT publication_number) AS n_patents,
  MIN(filing_date) AS first_filing,
  MAX(filing_date) AS last_filing,
  ARRAY_AGG(title_text IGNORE NULLS LIMIT 5) AS sample_titles
FROM matches
GROUP BY target_idx, target_name, sA_cutoff, assignee_name
ORDER BY target_idx, n_patents DESC
"""


def build_targets_struct(targets):
    """Turn [(idx, name, sA_int), ...] into a SQL UNNEST([STRUCT(...)]) payload.

    BigQuery string literals: use double-quotes to avoid having to escape
    single-quotes inside names like "Stackin'". Double-quotes and backslashes
    still need to be escaped, but those are vanishingly rare in company names;
    normalize_for_match already strips them via _SQL_SAFE.
    """
    parts = []
    for idx, name, sA_int in targets:
        # Belt-and-suspenders: escape any remaining special chars
        safe = name.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(
            f'STRUCT({idx} AS target_idx, "{safe}" AS target_name, {int(sA_int)} AS sA_cutoff)'
        )
    return ",\n    ".join(parts)


# ── Founding-date hard gate ────────────────────────────────────────────────
# Hard rule: if an assignee's earliest filing is more than 5 years BEFORE the
# company's founding date, it's almost certainly a different entity with the
# same legal name. 5y gives slack for founders filing under their startup's
# name before legal incorporation (Happiest Baby's case) while cleanly
# rejecting 17-year gaps like Roots (2001 patents attributed to a 2018 ag/
# insurance-AI startup).
FOUNDING_GATE_YEARS = 5


def filter_by_founding_date(rows, founded_date_str):
    """Drop rows whose first_filing is more than FOUNDING_GATE_YEARS before
    the company's founding date. Returns the surviving rows and a count of
    rejected ones for audit.
    """
    founded = pd.to_datetime(founded_date_str, errors="coerce")
    if pd.isna(founded):
        return rows, 0  # no founding date to check against
    cutoff = founded - pd.Timedelta(days=FOUNDING_GATE_YEARS * 365)
    cutoff_int = int(cutoff.strftime("%Y%m%d"))
    kept, rejected = [], 0
    for r in rows:
        if r["first_filing"] < cutoff_int:
            rejected += 1
            continue
        kept.append(r)
    return kept, rejected


# ── LLM verifier for a single company + its candidate assignees ────────────
def verify_company_assignees_llm(company_name, company_description,
                                 company_founded_date, rows):
    """Given rows = list of dicts with assignee_name, n_patents, sample_titles,
    ask the LLM which assignees (if any) actually belong to this company.
    Returns a set of accepted assignee_name strings.
    """
    if not _llm_client or not rows:
        # Without LLM, accept any match (caller is responsible for risk).
        return {r["assignee_name"] for r in rows}
    try:
        # Build a compact candidate list — now including first_filing date
        # so the LLM can catch obvious temporal mismatches.
        candidates = []
        for i, r in enumerate(rows[:10]):
            titles = r.get("sample_titles") or []
            titles_str = " | ".join(str(t)[:80] for t in titles[:3])
            ff = str(r.get("first_filing") or "")
            ff_pretty = f"{ff[:4]}-{ff[4:6]}-{ff[6:8]}" if len(ff) == 8 else ff
            candidates.append(
                f"  [{i}] assignee='{r['assignee_name']}' "
                f"n={r['n_patents']} earliest={ff_pretty} "
                f"sample_titles='{titles_str[:200]}'"
            )
        candidates_str = "\n".join(candidates)
        prompt = (
            f"You are verifying which patent assignee entries belong to a specific startup.\n\n"
            f"Startup:\n"
            f"  name: {company_name}\n"
            f"  founded: {company_founded_date or 'unknown'}\n"
            f"  description: {(company_description or '')[:400]}\n\n"
            f"Candidate patent assignees:\n{candidates_str}\n\n"
            f"Your job: accept the indices whose sample_titles are PLAUSIBLY related "
            f"to what the startup does. A hard rule elsewhere already filters out "
            f"obvious temporal mismatches (10+ year gaps), so you don't need to "
            f"worry about dates. Focus purely on semantic alignment.\n\n"
            f"Accept an index when:\n"
            f"  - The titles describe the same general domain as the startup "
            f"(biotech company + pharma/drug titles; robotics company + robotic/sensor titles; "
            f"consumer-product company + product-related titles; software startup + "
            f"software/computing titles).\n"
            f"  - The assignee legal name matches the startup name plausibly "
            f"(XYZ INC, XYZ LLC, XYZ THERAPEUTICS all fine for startup XYZ).\n\n"
            f"Reject an index when:\n"
            f"  - The titles are clearly in a totally unrelated domain "
            f"(e.g. 'Roots' insurance-AI vs agricultural plant patents; "
            f"'Maze' product-research vs biotech gene-therapy patents).\n"
            f"  - The assignee has thousands of patents (>500), which would mean "
            f"a collision with a massive established company, not our pre-A startup.\n\n"
            f"When in doubt, ACCEPT. False negatives (rejecting a real match) are "
            f"as bad as false positives. The hard temporal gate is the main filter; "
            f"you're the semantic check.\n\n"
            f'Return JSON: {{"accept_indices": [list of integers]}}'
        )
        r = _llm_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=80,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = r.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        accept = set(parsed.get("accept_indices", []))
        return {rows[i]["assignee_name"] for i in accept if 0 <= i < len(rows[:10])}
    except Exception as e:
        print(f"  LLM verification error for '{company_name}': {type(e).__name__}: {e}")
        return None  # distinct from set() = deliberate rejection


# ── Main ───────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
if "Organization_Name" not in df.columns:
    raise ValueError("Input is missing 'Organization_Name' column")
if "seriesA_date" not in df.columns:
    raise ValueError("Input is missing 'seriesA_date' column")

df["patent_count_preA_company"] = 0
df["has_patent_preA_company"] = 0
df["patent_earliest_date_preA"] = ""
df["patent_matched_assignees"] = ""
df["patent_note"] = ""

# Build targets list: (row_index, normalized_name, seriesA_int)
targets = []
for i, row in df.iterrows():
    raw_name = row.get("Organization_Name")
    norm = normalize_for_match(raw_name)
    if not norm:
        df.at[i, "patent_note"] = "no_name"
        continue
    sA_raw = row.get("seriesA_date")
    sA_dt = pd.to_datetime(sA_raw, errors="coerce")
    if pd.isna(sA_dt):
        df.at[i, "patent_note"] = "no_seriesA_date"
        continue
    sA_int = int(sA_dt.strftime("%Y%m%d"))
    targets.append((int(i), norm, sA_int))

print(f"Queryable companies: {len(targets)} / {len(df)}")
if not targets:
    df.to_csv(OUTPUT_CSV, index=False)
    print("No queryable rows — wrote output with zeros")
    raise SystemExit(0)

# ── Run the query in batches of 100 ────────────────────────────────────────
# BigQuery has a query size limit (~256 KB). With 500+ companies, the STRUCT
# list exceeds this. Splitting into batches of 100 keeps each query small.
BATCH_SIZE = 100
result = []
failed_indices = set()
t0 = time.time()

for batch_start in range(0, len(targets), BATCH_SIZE):
    batch = targets[batch_start:batch_start + BATCH_SIZE]
    query = BULK_QUERY_TEMPLATE.format(targets_struct_list=build_targets_struct(batch))

    try:
        real_cfg = bigquery.QueryJobConfig(
            maximum_bytes_billed=MAX_BYTES_BILLED,
            use_query_cache=True,
        )
        batch_result = list(bq_client.query(query, job_config=real_cfg).result())
        result.extend(batch_result)
        print(f"  Batch {batch_start}-{batch_start+len(batch)}: {len(batch_result)} rows")
    except Exception as e:
        print(f"  Batch {batch_start}: error: {type(e).__name__}: {str(e)[:100]}")
        failed_indices.update(idx for idx, _, _ in batch)

print(f"Query returned {len(result)} total rows in {time.time()-t0:.1f}s ({len(targets)} companies in {(len(targets)-1)//BATCH_SIZE+1} batches)")

# ── Group rows by company and run LLM verifier per company ────────────────
by_company = defaultdict(list)
for r in result:
    by_company[int(r["target_idx"])].append({
        "assignee_name": r["assignee_name"],
        "n_patents": int(r["n_patents"]),
        "first_filing": int(r["first_filing"]),
        "last_filing": int(r["last_filing"]),
        "sample_titles": list(r["sample_titles"] or []),
    })

print(f"Companies with at least one candidate assignee: {len(by_company)}")

# For each company with candidates, run the founding-date hard gate then LLM verification
verified_total = 0
rejected_by_date_gate = 0
for idx in tqdm(sorted(by_company.keys()), desc="Verification"):
    rows = by_company[idx]
    # Get this company's name, description, founding date
    cname = str(df.loc[idx, "Organization_Name"])
    cdesc = str(df.loc[idx, "Description"]) if "Description" in df.columns else ""
    cfounded = str(df.loc[idx, "Founded Date"]) if "Founded Date" in df.columns else ""

    # Hard gate 1: drop assignees whose first filing is >5y before founded date
    rows, n_gated = filter_by_founding_date(rows, cfounded)
    rejected_by_date_gate += n_gated
    if not rows:
        df.at[idx, "patent_note"] = "date_gate_rejected_all"
        continue

    # Hard gate 2: LLM verification with description + sector + temporal check
    accepted_assignees = verify_company_assignees_llm(cname, cdesc, cfounded, rows)
    if accepted_assignees is None:
        # LLM error — accept all (same as no-LLM mode; founding-date gate already filtered)
        accepted_assignees = {r["assignee_name"] for r in rows}
        llm_note = "llm_error_accepted_all"
    elif not accepted_assignees:
        df.at[idx, "patent_note"] = "llm_rejected_all"
        continue
    else:
        llm_note = None
    # Sum verified patents for this company
    total = 0
    earliest = None
    matched_names = []
    for r in rows:
        if r["assignee_name"] not in accepted_assignees:
            continue
        total += r["n_patents"]
        if earliest is None or r["first_filing"] < earliest:
            earliest = r["first_filing"]
        matched_names.append(r["assignee_name"])
    if total > 0:
        df.at[idx, "patent_count_preA_company"] = total
        df.at[idx, "has_patent_preA_company"] = 1
        if earliest:
            df.at[idx, "patent_earliest_date_preA"] = (
                f"{earliest//10000:04d}-{(earliest%10000)//100:02d}-{earliest%100:02d}"
            )
        df.at[idx, "patent_matched_assignees"] = " | ".join(matched_names[:5])
        note = llm_note or f"verified={len(matched_names)} | via bigquery"
        df.at[idx, "patent_note"] = note
        verified_total += 1
    else:
        df.at[idx, "patent_note"] = llm_note or "llm_rejected_all"

# Companies with no candidate at all
for i in df.index:
    if df.at[i, "patent_note"] == "" and int(i) not in by_company:
        if int(i) in failed_indices:
            df.at[i, "patent_note"] = "bigquery_batch_error"
        else:
            df.at[i, "patent_note"] = "no_assignee_match"

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
found = int(df["has_patent_preA_company"].sum())
print(f"BigQuery Patents: {found}/{len(df)} companies with verified pre-A patents")
print(f"  (hard gate rejected {rejected_by_date_gate} assignee rows for being >5y before founding date)")

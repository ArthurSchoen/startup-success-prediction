#!/usr/bin/env python
# coding: utf-8

# In[2]:


# clean_formd_output.py
# ------------------------------------------------------------
# Post-process Form D enriched output:
# - drop noisy Crunchbase cols
# - reorder columns logically
# - write CLEAN CSV
# ------------------------------------------------------------

import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# Use the Pipeline folder that contains .run_id (written by 07) so we use the same run as 01
PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline", Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
PIPE = None
for p in PIPE_CANDIDATES:
    if p.exists() and (p / ".run_id").exists():
        PIPE = p
        break
if PIPE is None:
    PIPE = PIPE_CANDIDATES[0] if PIPE_CANDIDATES[0].exists() else (Path.home() / "Desktop" / "Data_thesis" / "Pipeline")
BASE = PIPE.parent
ARCHIVE = PIPE / "archive"
PIPE.mkdir(parents=True, exist_ok=True)
ARCHIVE.mkdir(parents=True, exist_ok=True)


def pick_next_file_for_step(pipe_dir: Path, prev_step: str, next_step_prefix: str, allow_overwrite: bool = False) -> Path:
    """allow_overwrite=True : prend le dernier input même si la sortie existe (pour re-run)."""
    candidates = []
    for p in pipe_dir.glob(f"*{prev_step}*.csv"):
        run_id = p.name.split("_", 1)[0]
        next_exists = any(pipe_dir.glob(f"{run_id}{next_step_prefix}*.csv"))
        if allow_overwrite or not next_exists:
            candidates.append((run_id, p))
    if not candidates:
        raise FileNotFoundError(f"No runnable input found for prev_step={prev_step}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# ✅ ICI: on part de l’étape 02, et on génère l’étape 03
CHOSEN_RUN_ID = os.environ.get("PIPELINE_RUN_ID", "").strip() or ((PIPE / ".run_id").read_text().strip() if (PIPE / ".run_id").exists() else "")
ALLOW_OVERWRITE = True
INPUT_02 = PIPE / f"{CHOSEN_RUN_ID}_02_formd__seriesA_dates_enriched.csv" if CHOSEN_RUN_ID else None
if CHOSEN_RUN_ID and INPUT_02 and INPUT_02.exists():
    INPUT_CSV = INPUT_02
else:
    INPUT_CSV = pick_next_file_for_step(
        pipe_dir=PIPE, prev_step="_02_formd__seriesA_dates_enriched", next_step_prefix="_03_formd__seriesA_dates_enriched__clean", allow_overwrite=ALLOW_OVERWRITE
    )

RUN_ID = INPUT_CSV.name.split("_", 1)[0]
OUTPUT_CSV = PIPE / f"{RUN_ID}_03_formd__seriesA_dates_enriched__clean.csv"

print("INPUT :", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

DROP_CRUNCHBASE_COLS = [
    "Founded Date Precision",
    "Exit Date","Exit Date Precision","Closed Date","Closed Date Precision",
    "Estimated Revenue Range","Company Type","Facebook","LinkedIn","Number of Articles","Hub Tags",
    "Last Equity Funding Amount Currency","Last Funding Amount Currency",
    "Total Equity Funding Amount Currency","Total Funding Amount Currency",
    "Investment Stage",
    "Acquired by","Acquired by URL","Announced Date","Announced Date Precision",
    "Transaction Name","Transaction Name URL",
    "IPO Status","IPO Date","Money Raised at IPO","Money Raised at IPO Currency",
    "Money Raised at IPO (in USD)","Valuation at IPO","Valuation at IPO Currency","Valuation at IPO (in USD)",
    "Number of Employees","Number of Lead Investments","Number of Diversity Investments",
    "Number of Exits","Number of Exits (IPO)","Accelerator Program Type",
]

# "ordre intelligent" (tu peux tweak)
ORDERED_COLS = [
    # 1) Identity / core
    "Organization_Name",
    "Organization Name URL",
    "Website",
    "Twitter",
    "Industries",
    "Industry Groups",
    "Headquarters Location",
    "Headquarters Regions",
    "Postal Code",
    "Description",
    "Full Description",
    "Operating Status",

    # 2) Founding / team
    "Founded Date",
    "Number of Founders",
    "Founders",
    "Diversity Spotlight",

    # 3) Crunchbase funding summary (keep amounts USD, types, investors)
    "Funding Status",
    "Number of Funding Rounds",
    "Last Funding Date",
    "Last Funding Type",
    "Last Funding Amount (in USD)",
    "Last Equity Funding Type",
    "Last Equity Funding Amount (in USD)",
    "Total Funding Amount (in USD)",
    "Total Equity Funding Amount (in USD)",
    "Top 5 Investors",
    "Lead Investors",
    "Number of Investors",

    # 4) M&A / IPO (if you keep them)
    "Acquisition Status",
    "Number of Acquisitions",
    "Acquisition Type",

    # 5) Normalized helper cols created by your FormD pipeline
    "Postal_Code",
    "HQ_City",
    "HQ_State",
    "industry_bucket",

    # 6) SEC matching / debug
    "sec_match_cik",
    "sec_match_issuer_name",
    "sec_match_score",
    "sec_founders_overlap_count",
    "sec_founders_overlap_ratio",
    "sec_candidates_debug",

    # 7) Events + inferred Series A
    "sec_events_found",
    "sec_events_summary",
    "seriesA_date",
    "seriesA_quarter",
    "seriesA_amount_usd",
    "seriesA_amount_usd_num",
    "seriesA_security_type",
    "seriesA_comment",
    "seriesA_confidence",

    # 8) Age sanity check
    "founded_date",
    "age_at_seriesA_years",
    "age_at_seriesA_flag",

    # 9) Form D investor features
    "formd_directors_preA",
    "formd_investor_count_preA",
    "formd_prior_rounds",
]

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = list(df.columns)

    # keep only the ordered cols that exist
    ordered_existing = [c for c in ORDERED_COLS if c in existing]

    # append any remaining cols not specified (so you never lose data accidentally)
    remainder = [c for c in existing if c not in ordered_existing]

    return df[ordered_existing + remainder]

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded: {len(df)} rows")

    # ── 1) DROP first, COMPUTE after (don't waste cycles on rows we'll discard) ──

    # Drop rows without seriesA_date
    if "seriesA_date" in df.columns:
        n_before = len(df)
        df = df[df["seriesA_date"].notna() & (df["seriesA_date"].astype(str).str.strip() != "")]
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"⚠️ Dropped {n_dropped} rows without seriesA_date")

    # Drop SUSPECT age rows
    if "age_at_seriesA_flag" in df.columns:
        n_before = len(df)
        df = df[df["age_at_seriesA_flag"].astype(str).str.strip().str.upper() != "SUSPECT"]
        if len(df) < n_before:
            print(f"⚠️ Dropped {n_before - len(df)} rows with age_at_seriesA_flag = SUSPECT")

    # Drop LOW confidence rows early — saves all API calls downstream
    if "seriesA_confidence" in df.columns:
        n_before = len(df)
        df = df[df["seriesA_confidence"].astype(str).str.strip().str.upper() != "LOW"].copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"⚠️ Dropped {n_dropped} rows with seriesA_confidence = LOW")

    print(f"After filtering: {len(df)} rows")

    # ── 2) COMPUTE on surviving rows only ──

    # Form D investor features (only on filtered rows — much faster than doing 2K)
    FORMD_DB = BASE / "Form D SEC" / "formd_all.sqlite"
    if FORMD_DB.exists() and "sec_match_cik" in df.columns and "seriesA_date" in df.columns:
        print("Extracting Form D investor features...")
        conn = sqlite3.connect(str(FORMD_DB))
        # Pre-fetch all accession numbers per CIK (one query, uses index)
        # Then filter by date in Python — avoids per-row full-table scans
        unique_ciks = df["sec_match_cik"].dropna().unique()
        cik_strs = [str(int(float(c))).zfill(10) for c in unique_ciks]
        # Build a CIK→accessions lookup
        placeholders = ",".join("?" * len(cik_strs))
        acc_rows = conn.execute(f"""
            SELECT i.CIK, s.ACCESSIONNUMBER, s.FILING_DATE
            FROM issuers i
            JOIN formdsubmission s ON i.ACCESSIONNUMBER = s.ACCESSIONNUMBER
            WHERE i.CIK IN ({placeholders})
        """, cik_strs).fetchall()
        # CIK → list of (accession, filing_date)
        from collections import defaultdict
        cik_acc = defaultdict(list)
        for cik_val, acc, fdate in acc_rows:
            cik_acc[str(cik_val).strip()].append((acc, fdate or ""))
        # Pre-fetch relatedpersons for relevant accessions
        all_accs = [acc for rows in cik_acc.values() for acc, _ in rows]
        rp_data = {}  # acc → list of (firstname, lastname, rel1, rel2)
        off_data = {}  # acc → TOTALNUMBERALREADYINVESTED
        if all_accs:
            batch_size = 500
            for i in range(0, len(all_accs), batch_size):
                batch = all_accs[i:i+batch_size]
                ph = ",".join("?" * len(batch))
                for row in conn.execute(f"""
                    SELECT ACCESSIONNUMBER, FIRSTNAME, LASTNAME, RELATIONSHIP_1, RELATIONSHIP_2
                    FROM relatedpersons WHERE ACCESSIONNUMBER IN ({ph})
                """, batch).fetchall():
                    rp_data.setdefault(row[0], []).append(row[1:])
                for row in conn.execute(f"""
                    SELECT ACCESSIONNUMBER, TOTALNUMBERALREADYINVESTED
                    FROM offering WHERE ACCESSIONNUMBER IN ({ph})
                    AND TOTALNUMBERALREADYINVESTED IS NOT NULL
                """, batch).fetchall():
                    off_data[row[0]] = row[1]
        conn.close()
        # Now compute per company (all in memory, no more SQL)
        directors_list, investors_list, prior_list = [], [], []
        for _, row in df.iterrows():
            cik = row.get("sec_match_cik")
            sa_date = str(row.get("seriesA_date", "")).strip()
            if pd.isna(cik) or not sa_date or sa_date in ("nan", "NaT", ""):
                directors_list.append(0); investors_list.append(0); prior_list.append(0)
                continue
            cik_str = str(int(float(cik))).zfill(10)
            accs = cik_acc.get(cik_str, [])
            # Two date cutoffs:
            # 1) Buffer (A + 90 days): for directors and investors
            #    The Series A filing is submitted days/weeks after closing.
            #    Directors and investors on that filing are known at prediction time.
            # 2) Strict (before A): for prior rounds count
            #    Prior rounds = funding rounds BEFORE Series A, not including A itself.
            try:
                sa_dt = pd.to_datetime(sa_date)
                buffer_date = (sa_dt + pd.Timedelta(days=90)).strftime("%Y-%m-%d")
            except Exception:
                buffer_date = sa_date
            accs_with_buffer = [acc for acc, fdate in accs if fdate <= buffer_date]
            accs_strict = [acc for acc, fdate in accs if fdate < sa_date]
            # Directors count (with buffer: includes A filing)
            directors = set()
            for acc in accs_with_buffer:
                for fn, ln, r1, r2 in rp_data.get(acc, []):
                    if r1 == "Director" or r2 == "Director":
                        directors.add(f"{fn} {ln}")
            directors_list.append(len(directors))
            # Max investor count (with buffer: includes A filing)
            max_inv = 0
            for acc in accs_with_buffer:
                val = off_data.get(acc)
                if val is not None:
                    try:
                        max_inv = max(max_inv, int(val))
                    except (ValueError, TypeError):
                        pass
            investors_list.append(max_inv)
            # Prior rounds (strict: before A only, not including A)
            prior_list.append(len(accs_strict))
        df["formd_directors_preA"] = directors_list
        df["formd_investor_count_preA"] = investors_list
        df["formd_prior_rounds"] = prior_list
        has_dirs = (df["formd_directors_preA"] > 0).sum()
        print(f"  Directors: {has_dirs}/{len(df)} have pre-A directors")
        print(f"  Investors: {(df['formd_investor_count_preA'] > 0).sum()}/{len(df)}")
        print(f"  Prior rounds: {(df['formd_prior_rounds'] > 0).sum()}/{len(df)}")
    else:
        if not FORMD_DB.exists():
            print("WARNING: formd_all.sqlite not found — skipping investor features")

    # Drop noisy Crunchbase columns
    df = df.drop(columns=DROP_CRUNCHBASE_COLS, errors="ignore")

    # Reorder
    df = reorder_columns(df)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()



# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8
"""
data_pipeline.py — Parallel Signal Orchestration for Startup Success Prediction

Collects pre-Series A signals from 13+ public data sources concurrently.
Each signal collector runs as an independent subprocess; results are merged
on Organization_Name at the end.

Architecture:
  Phase 1: LLM Classification (GPT-4o-mini) — must run first
     └─ deeptech_llm_score (0-10), startup_sector, name_distinctive
  Phase 2: 12 signal collectors in parallel (~5 min vs ~17 min sequential)
     └─ Google Trends, Wikipedia, GitHub, Packages, OpenAlex,
        Hacker News, Wayback Machine, App Store, Product Hunt,
        Clinical Trials, NIH Grants, EPO Patents

Time: ~17 min sequential → ~5 min parallel (3.4x speedup)

Author: Arthur Schoen — MIT / HEC Paris
"""

import os
import re
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Environment ──────────────────────────────────────────────────────────────
# Load .env (API keys for GitHub, OpenAI, EPO, Product Hunt, etc.)
for _env_path in [Path.cwd() / ".env", Path.cwd().parent / ".env"]:
    if _env_path.exists():
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ[_k.strip()] = _v.strip().strip('"').strip("'")
        break

# ── Paths ────────────────────────────────────────────────────────────────────
PIPE     = Path.cwd() / "Pipeline"
CODE_DIR = Path(__file__).parent / "signals"
VENV_PY  = Path(sys.executable)

# ── Run ID & Base CSV ────────────────────────────────────────────────────────
RUN_ID = (PIPE / ".run_id").read_text().strip() if (PIPE / ".run_id").exists() else ""

def find_base_csv() -> Path:
    """Find the most recent step-02 output (cleaned core features)."""
    if RUN_ID:
        for suffix in ["_03_formd__seriesA_dates_enriched__clean.csv",
                        "_05_webtraction__signals.csv"]:
            p = PIPE / f"{RUN_ID}{suffix}"
            if p.exists():
                return p
    # Fallback: glob for latest
    for pattern in ["*_03_formd__seriesA_dates_enriched__clean.csv",
                     "*_05_webtraction__signals.csv"]:
        candidates = sorted(PIPE.glob(pattern), key=lambda x: x.name, reverse=True)
        if candidates:
            return candidates[0]
    raise FileNotFoundError("No base CSV found in Pipeline/")

BASE_CSV = find_base_csv()

# ── Signal Step Definitions ──────────────────────────────────────────────────
# Phase 1: LLM classification (runs first — downstream steps depend on it)
LLM_STEP = ("05", "05__llm_classify.py", f"{RUN_ID}_19_deeptech_llm__signals.csv")

# Phase 2: All signal collectors (run concurrently)
SIGNAL_STEPS = [
    ("06a", "06a__google_trends.py",   f"{RUN_ID}_07_googletrends__signals.csv"),
    ("06b", "06b__wikipedia.py",        f"{RUN_ID}_06b_wikipedia.csv"),
    ("06c", "06c__github.py",           f"{RUN_ID}_11_github__signals.csv"),
    ("06d", "06d__packages.py",         f"{RUN_ID}_12_packages__signals.csv"),
    ("06e", "06e__openalex.py",         f"{RUN_ID}_13_openalex__signals.csv"),
    ("06f", "06f__hacker_news.py",      f"{RUN_ID}_14_producthunt_hn__signals.csv"),
    ("06w", "06w__wayback_unified.py",  f"{RUN_ID}_24_wayback__signals.csv"),
    ("06h", "06h__appstore.py",         f"{RUN_ID}_16_appstore__signals.csv"),
    ("06j", "06j__product_hunt.py",     f"{RUN_ID}_21_producthunt__signals.csv"),
    ("06k", "06k__clinical_trials.py",  f"{RUN_ID}_22_clinical_trials__signals.csv"),
    ("06l", "06l__nih_grants.py",       f"{RUN_ID}_23_nih_grants__signals.csv"),
    ("06m", "06m__epo_patents.py",      f"{RUN_ID}_06_patents__signals.csv"),
]


def run_step(key: str, script: str, expected_output: str,
             base_csv_override: Path = None) -> tuple:
    """
    Run a signal-collection script as a subprocess.

    Each script reads PIPELINE_BASE_CSV (env var) and writes its output CSV
    to the Pipeline directory. Stdout is streamed with [key] prefixes for
    real-time monitoring.
    """
    script_path = CODE_DIR / script
    env = os.environ.copy()
    env["PIPELINE_BASE_CSV"] = str(base_csv_override or BASE_CSV)
    env["PIPELINE_RUN_ID"]   = RUN_ID
    env["PYTHONPATH"]        = str(CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    start = time.time()
    print(f"  [{key}] Starting {script}", flush=True)
    try:
        proc = subprocess.Popen(
            [str(VENV_PY), "-u", str(script_path)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        last_lines = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                last_lines.append(line)
                # Show key progress indicators
                progress_keywords = [
                    "%|", "Saved", "Done", "found", "scored", "midpoint",
                    "error", "ERROR", "WARNING",
                ]
                if any(kw in line for kw in progress_keywords):
                    print(f"  [{key}] {line}", flush=True)
        proc.wait(timeout=7200)
        elapsed = time.time() - start

        if proc.returncode != 0:
            print(f"  [{key}] ERROR after {elapsed:.0f}s (exit {proc.returncode})")
            for l in last_lines[-5:]:
                print(f"  [{key}]   {l}")
            return key, False, elapsed

        print(f"  [{key}] Done in {elapsed:.0f}s", flush=True)
        return key, True, elapsed

    except subprocess.TimeoutExpired:
        print(f"  [{key}] TIMEOUT after {time.time() - start:.0f}s")
        return key, False, time.time() - start
    except Exception as e:
        print(f"  [{key}] EXCEPTION: {e}")
        return key, False, time.time() - start


def merge_outputs(base_csv: Path, steps: list, pipe_dir: Path) -> pd.DataFrame:
    """
    Merge all signal outputs onto the base DataFrame by Organization_Name.

    Each signal script produces a CSV with Organization_Name + new columns.
    We left-join them sequentially onto the base to build the full feature matrix.
    """
    df_merged = pd.read_csv(base_csv)
    df_merged.columns = df_merged.columns.str.strip()

    for key, _script, expected_output in steps:
        out_path = pipe_dir / expected_output
        if not out_path.exists():
            print(f"  [{key}] MISSING {out_path.name} — skipping")
            continue
        try:
            df_sig = pd.read_csv(out_path)
            df_sig.columns = df_sig.columns.str.strip()
            new_cols = [c for c in df_sig.columns
                        if c not in set(df_merged.columns) or c == "Organization_Name"]
            df_merged = df_merged.merge(
                df_sig[new_cols], on="Organization_Name", how="left"
            )
            added = len(new_cols) - 1  # subtract Organization_Name
            print(f"  [{key}] +{added} columns from {out_path.name}")
        except Exception as e:
            print(f"  [{key}] Merge error: {e}")

    return df_merged


def run_pipeline():
    """
    Execute the full data collection pipeline:
      1. Run LLM classification (enriches base CSV with sector/deeptech signals)
      2. Run 12 signal collectors in parallel
      3. Merge all outputs into a single feature matrix
    """
    t0 = time.time()

    # ── Phase 1: LLM Classification ──────────────────────────────────────
    print("\n--- Phase 1: LLM Classification (GPT-4o-mini) ---")
    llm_key, llm_script, llm_out = LLM_STEP
    key, ok, elapsed = run_step(llm_key, llm_script, llm_out)

    # Merge LLM output into base CSV for downstream steps
    enriched_base = BASE_CSV
    if ok:
        llm_path = PIPE / llm_out
        if llm_path.exists():
            df_base = pd.read_csv(BASE_CSV)
            df_base.columns = df_base.columns.str.strip()
            df_llm = pd.read_csv(llm_path)
            df_llm.columns = df_llm.columns.str.strip()
            llm_cols = [c for c in df_llm.columns
                        if c not in set(df_base.columns) or c == "Organization_Name"]
            if len(llm_cols) > 1:
                df_enriched = df_base.merge(df_llm[llm_cols], on="Organization_Name", how="left")
                enriched_base = PIPE / f"{RUN_ID}_05_enriched_base.csv"
                df_enriched.to_csv(enriched_base, index=False)
                print(f"  Enriched base with {len(llm_cols)-1} LLM columns")

    # ── Phase 2: Parallel Signal Collection ──────────────────────────────
    print(f"\n--- Phase 2: Running {len(SIGNAL_STEPS)} signal steps in parallel ---")
    with ThreadPoolExecutor(max_workers=len(SIGNAL_STEPS)) as executor:
        futures = {
            executor.submit(run_step, key, script, out,
                            base_csv_override=enriched_base): key
            for key, script, out in SIGNAL_STEPS
        }
        for future in as_completed(futures):
            future.result()  # propagate exceptions

    # ── Phase 3: Merge ───────────────────────────────────────────────────
    print("\n--- Phase 3: Merging signal outputs ---")
    all_steps = [LLM_STEP] + SIGNAL_STEPS
    df_merged = merge_outputs(BASE_CSV, all_steps, PIPE)

    output_path = PIPE / f"{RUN_ID}_06_merged_signals.csv"
    df_merged.to_csv(output_path, index=False)

    total = time.time() - t0
    print(f"\nMerged: {df_merged.shape[0]} rows x {df_merged.shape[1]} cols")
    print(f"Total wall time: {total:.0f}s")
    return df_merged


if __name__ == "__main__":
    run_pipeline()

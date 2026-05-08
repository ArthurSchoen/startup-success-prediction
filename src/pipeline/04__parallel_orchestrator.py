#!/usr/bin/env python
# coding: utf-8
"""
04__parallel_orchestrator.py — Run all signal-enrichment steps (04a through 04n) in parallel.

Instead of chaining 05→05b→05c→...→05i sequentially (each reading the previous output),
all steps read from the same base CSV (step 02 output) and run concurrently.
Results are merged at the end by Organization_Name.

Time: ~17 min sequential → ~5 min parallel (bottleneck = slowest single step).
"""

import os
import re
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from shared import load_env
load_env()

PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline", Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
PIPE = next((p for p in PIPE_CANDIDATES if p.exists() and (p / ".run_id").exists()), None) or (
    PIPE_CANDIDATES[0] if PIPE_CANDIDATES[0].exists() else Path.home() / "Desktop" / "Data_thesis" / "Pipeline"
)
CODE_DIR = Path(__file__).parent
VENV_PYTHON = Path(sys.executable)

def run_id_from_filename(name):
    m = re.match(r"^([^_]+)_", Path(name).name)
    if not m:
        raise ValueError(f"Bad filename: {name}")
    return m.group(1)

# Find base CSV: step 02 output (step 03/04 removed from sequential pipeline)
# Falls back to step 05 output for backward compatibility with old runs
CHOSEN_RUN_ID = os.environ.get("PIPELINE_RUN_ID", "").strip() or (
    (PIPE / ".run_id").read_text().strip() if (PIPE / ".run_id").exists() else ""
)
BASE_CSV = None
if CHOSEN_RUN_ID:
    # Prefer step 02 output (new pipeline: 01 → 02 → orchestrator)
    for suffix in ["_03_formd__seriesA_dates_enriched__clean.csv",
                    "_05_webtraction__signals.csv"]:
        for search_dir in [PIPE, PIPE / "archive" / CHOSEN_RUN_ID]:
            p = search_dir / f"{CHOSEN_RUN_ID}{suffix}"
            if p.exists():
                BASE_CSV = p
                break
        if BASE_CSV:
            break
if BASE_CSV is None:
    for pattern in ["*_03_formd__seriesA_dates_enriched__clean.csv",
                     "*_05_webtraction__signals.csv"]:
        candidates = sorted(PIPE.glob(pattern), key=lambda x: x.name, reverse=True)
        if candidates:
            BASE_CSV = candidates[0]
            break
if BASE_CSV is None:
    for pattern in ["*/*_03_formd__seriesA_dates_enriched__clean.csv",
                     "*/*_05_webtraction__signals.csv"]:
        candidates = sorted((PIPE / "archive").glob(pattern), key=lambda x: x.name, reverse=True)
        if candidates:
            BASE_CSV = candidates[0]
            print(f"(base CSV found in archive: {BASE_CSV.parent.name})")
            break
if BASE_CSV is None:
    raise FileNotFoundError("No base CSV found in Pipeline or archive (expected _03_formd...clean or _05_webtraction)")

RUN_ID = run_id_from_filename(BASE_CSV)
print(f"Base CSV: {BASE_CSV}")
print(f"Run ID:   {RUN_ID}")

# Phase 2a: LLM Classification (must run first — other steps depend on its output)
LLM_STEP = ("03", "03__llm_classify.py", f"{RUN_ID}_03_llm__signals.csv")

# Phase 2b: Signal enrichment (all run in parallel after 05)
# Output filenames match what each script actually produces
SIGNAL_STEPS = [
    ("04a", "04a__wikipedia.py",          f"{RUN_ID}_04a_wikipedia__signals.csv"),
    ("04b", "04b__github.py",             f"{RUN_ID}_04b_github__signals.csv"),
    ("04c", "04c__packages.py",           f"{RUN_ID}_04c_packages__signals.csv"),
    ("04d", "04d__openalex.py",           f"{RUN_ID}_04d_openalex__signals.csv"),
    ("04e", "04e__hacker_news.py",        f"{RUN_ID}_04e_hacker_news__signals.csv"),
    ("04m", "04m__wayback_unified.py",    f"{RUN_ID}_04m_wayback__signals.csv"),
    ("04f", "04f__product_hunt.py",       f"{RUN_ID}_04f_product_hunt__signals.csv"),
    ("04g", "04g__clinical_trials.py",    f"{RUN_ID}_04g_clinical_trials__signals.csv"),
    ("04h", "04h__federal_awards.py",     f"{RUN_ID}_04h_federal_awards__signals.csv"),
    ("04i", "04i__epo_patents.py",        f"{RUN_ID}_04i_epo_patents__signals.csv"),
    ("04n", "04n__yc_alumni.py",          f"{RUN_ID}_04n_yc_alumni__signals.csv"),
    ("04j", "04j__gdelt_news.py",         f"{RUN_ID}_04j_gdelt_news__signals.csv"),
    ("04k", "04k__edgar_filings.py",      f"{RUN_ID}_04k_edgar_filings__signals.csv"),
]

def run_step(key, script, expected_output, base_csv_override=None):
    """Run a signal script as a subprocess with PIPELINE_BASE_CSV override."""
    script_path = CODE_DIR / script
    output_path = PIPE / expected_output
    env = os.environ.copy()
    env["PIPELINE_BASE_CSV"] = str(base_csv_override or BASE_CSV)
    env["PIPELINE_RUN_ID"] = RUN_ID
    env["PYTHONPATH"] = str(CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    start = time.time()
    print(f"  [{key}] Starting {script}", flush=True)
    try:
        # Stream output in real-time: each line prefixed with [key]
        proc = subprocess.Popen(
            [str(VENV_PYTHON), "-u", str(script_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        last_lines = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                last_lines.append(line)
                # Show progress lines (tqdm, midpoint saves, key results)
                if any(kw in line for kw in ["tqdm", "%|", "Saved", "Done", "found", "scored",
                                              "EPO", "Patent", "GitHub", "Wikipedia", "Google",
                                              "Wayback", "Clinical", "NIH", "AppStore", "Product",
                                              "HN", "OpenAlex", "Package", "midpoint", "LLM",
                                              "DeepTech", "token", "error", "ERROR", "WARNING"]):
                    print(f"  [{key}] {line}", flush=True)
        proc.wait(timeout=7200)
        elapsed = time.time() - start
        if proc.returncode != 0:
            print(f"  [{key}] ERROR after {elapsed:.0f}s (exit code {proc.returncode})", flush=True)
            for l in last_lines[-5:]:
                print(f"  [{key}]   {l}", flush=True)
            return key, False, elapsed, output_path
        print(f"  [{key}] ✓ Done in {elapsed:.0f}s", flush=True)
        return key, True, elapsed, output_path
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  [{key}] TIMEOUT after {elapsed:.0f}s")
        return key, False, elapsed, output_path
    except Exception as e:
        elapsed = time.time() - start
        print(f"  [{key}] EXCEPTION: {e}")
        return key, False, elapsed, output_path

# Phase 2a: Run LLM classification first (other steps depend on it)
print("\n━━━ Phase 2a: LLM Classification ━━━")
t0 = time.time()
results = {}

llm_key, llm_script, llm_out = LLM_STEP
key, ok, elapsed, out_path = run_step(llm_key, llm_script, llm_out)
results[key] = {"ok": ok, "elapsed": elapsed, "output": out_path}

# Merge LLM output into BASE_CSV so downstream steps have startup_sector + name_distinctive
ENRICHED_BASE = BASE_CSV
if ok and out_path.exists():
    try:
        df_base = pd.read_csv(BASE_CSV)
        df_base.columns = df_base.columns.str.strip()
        df_llm = pd.read_csv(out_path)
        df_llm.columns = df_llm.columns.str.strip()
        llm_cols = [c for c in df_llm.columns if c not in set(df_base.columns) or c == "Organization_Name"]
        if len(llm_cols) > 1:
            df_enriched = df_base.merge(df_llm[llm_cols], on="Organization_Name", how="left")
            ENRICHED_BASE = PIPE / f"{RUN_ID}_03_enriched_base.csv"
            df_enriched.to_csv(ENRICHED_BASE, index=False)
            print(f"  Enriched base CSV with {len(llm_cols)-1} LLM columns → {ENRICHED_BASE.name}")
    except Exception as e:
        print(f"  WARNING: Could not merge LLM output: {e}")
else:
    print(f"WARNING: LLM classification failed — downstream steps may lack sector/name_distinctive data")

# Phase 2b: Run all signal steps in parallel
print(f"\n━━━ Phase 2b: Running {len(SIGNAL_STEPS)} signal steps in parallel ━━━")
with ThreadPoolExecutor(max_workers=len(SIGNAL_STEPS)) as executor:
    futures = {
        executor.submit(run_step, key, script, out, base_csv_override=ENRICHED_BASE): key
        for key, script, out in SIGNAL_STEPS
    }
    for future in as_completed(futures):
        key, ok, elapsed, out_path = future.result()
        results[key] = {"ok": ok, "elapsed": elapsed, "output": out_path}

total_elapsed = time.time() - t0
print(f"\n{'━' * 50}")
print(f"All steps complete in {total_elapsed:.0f}s (vs ~{sum(r['elapsed'] for r in results.values()):.0f}s sequential)")

# Merge all outputs onto base CSV
print("\nMerging signal outputs...")
df_base = pd.read_csv(BASE_CSV)
df_base.columns = df_base.columns.str.strip()
base_cols = set(df_base.columns)
key_cols = ["Organization_Name"]

df_merged = df_base.copy()

# Process outputs in order (LLM step first, then signal steps)
ALL_STEPS = [LLM_STEP] + SIGNAL_STEPS
for key, script, expected_output in ALL_STEPS:
    out_path = PIPE / expected_output
    if not out_path.exists():
        print(f"  [{key}] MISSING output {out_path.name} — skipping")
        continue
    try:
        df_sig = pd.read_csv(out_path)
        df_sig.columns = df_sig.columns.str.strip()
        # New columns = columns in this output not already in merged
        new_cols = [c for c in df_sig.columns if c not in set(df_merged.columns) or c in key_cols]
        merge_cols = list(dict.fromkeys(key_cols + [c for c in new_cols if c not in key_cols]))
        df_new = df_sig[merge_cols].copy()
        df_merged = df_merged.merge(df_new, on=key_cols, how="left")
        added = len(merge_cols) - len(key_cols)
        print(f"  [{key}] +{added} columns from {out_path.name}")
    except Exception as e:
        print(f"  [{key}] Merge error: {e}")

# Save merged output as the final signals file (step 07 will read this)
MERGED_OUTPUT = PIPE / f"{RUN_ID}_04_merged_signals.csv"
df_merged.to_csv(MERGED_OUTPUT, index=False)
print(f"\nMerged output ({df_merged.shape[0]} rows × {df_merged.shape[1]} cols): {MERGED_OUTPUT}")
print(f"Total wall time: {total_elapsed:.0f}s")

# Exit with error if any step failed
failed = [k for k, v in results.items() if not v["ok"]]
if failed:
    print(f"\n❌ FAILED STEPS: {', '.join(failed)}")
    sys.exit(1)

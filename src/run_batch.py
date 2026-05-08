#!/usr/bin/env python
# coding: utf-8
"""
run_batch.py — Single orchestrated batch runner for the 16K startup pipeline.

Usage:
    python Code/run_batch.py                           # next 2000 from full_remaining_16k.csv
    python Code/run_batch.py --batch_size 500          # smaller batch
    python Code/run_batch.py --crunchbase_csv path.csv # custom input
    python Code/run_batch.py --skip_step01             # rerun from orchestrator (step 01 already done)
    python Code/run_batch.py --only_step05             # just run step 05 on existing merged signals

Pre-filter workflow (saves ~75% of API calls):
    # Step 1: Pre-filter all 16K (runs once, ~10 min)
    python Code/run_batch.py --prefilter
    # Step 2: Run batches from pre-filtered survivors (each ~4-5 hours)
    python Code/run_batch.py --from_prefiltered --batch_size 2000

Pipeline flow (single process, no background jobs):
    1. Pick next batch → save as semicolon-separated Crunchbase export
    2. Step 01: SEC Form D enrichment  (skipped with --from_prefiltered)
    3. Step 02: Core feature cleaning   (skipped with --from_prefiltered)
    4. Step 03: LLM classify
    5. Step 04: Parallel signal orchestrator (sub-steps 04a-04n)
    6. Step 05: Final dataset merge & clean
    7. Mark batch as done in tracking CSV
"""

import os
import sys
import time
import subprocess
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent
PIPE = BASE / "Pipeline"
CRUNCH = BASE / "Crunchbase_data"
CODE_DIR = Path(__file__).resolve().parent
TRACKING_CSV = PIPE / "_batch_tracking.csv"
FULL_16K = CRUNCH / "full_remaining_16k.csv"
PREFILTERED_CSV = PIPE / "prefiltered_survivors.csv"
ARCHIVE_NEW2K = PIPE / "archive" / "new2k"
VENV_PYTHON = Path(sys.executable)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import load_env
load_env()

PIPE.mkdir(parents=True, exist_ok=True)


# ── Tracking ──────────────────────────────────────────────────────────────

def load_done_names():
    """Return set of already-processed Organization Names."""
    if TRACKING_CSV.exists():
        df = pd.read_csv(TRACKING_CSV)
        return set(df["Organization_Name"].tolist())
    return set()


def mark_done(names, batch_id):
    """Append processed company names to the tracking CSV."""
    new_rows = pd.DataFrame({
        "Organization_Name": list(names),
        "batch_id": batch_id,
        "processed_date": datetime.now().isoformat(),
    })
    if TRACKING_CSV.exists():
        existing = pd.read_csv(TRACKING_CSV)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(TRACKING_CSV, index=False)
    print(f"Tracking updated: {len(names)} companies marked done (total: {len(combined)})")


def seed_tracking_from_archive():
    """Seed _batch_tracking.csv with the 547 companies already done in new2k."""
    archive_csv = ARCHIVE_NEW2K / "new2k_05_enriched_base.csv"
    if not archive_csv.exists():
        archive_csv = ARCHIVE_NEW2K / "new2k_03_formd__seriesA_dates_enriched__clean.csv"
    if not archive_csv.exists():
        print(f"WARNING: No archive file found in {ARCHIVE_NEW2K}, skipping seed")
        return 0

    archive_df = pd.read_csv(archive_csv)
    name_col = "Organization Name" if "Organization Name" in archive_df.columns else "Organization_Name"
    archive_names = archive_df[name_col].dropna().unique().tolist()

    already_done = load_done_names()
    new_names = [n for n in archive_names if n not in already_done]
    if not new_names:
        print(f"Tracking already contains all {len(archive_names)} archive companies")
        return 0

    mark_done(new_names, "new2k_archive")
    print(f"Seeded {len(new_names)} companies from new2k archive into tracking")
    return len(new_names)


# ── Pre-filter ────────────────────────────────────────────────────────────

def run_prefilter(crunchbase_csv):
    """Run steps 01+02 on the full dataset, save survivors as prefiltered_survivors.csv.

    This avoids running expensive signal extraction (steps 03+04) on ~75% of
    companies that get filtered out by Form D matching / confidence checks.
    """
    t0 = time.time()
    prefilter_id = f"prefilter_{datetime.now().strftime('%Y%m%d_%H%M')}"
    print(f"\n{'=' * 60}")
    print(f"  PRE-FILTER MODE")
    print(f"  Input:  {crunchbase_csv}")
    print(f"  Run ID: {prefilter_id}")
    print(f"{'=' * 60}")

    # Load full dataset
    full = pd.read_csv(crunchbase_csv)
    name_col = "Organization Name" if "Organization Name" in full.columns else "Organization_Name"
    print(f"Full dataset: {len(full)} companies")

    # Exclude already-done companies from pre-filter
    done = load_done_names()
    if done:
        remaining = full[~full[name_col].isin(done)]
        print(f"Already processed: {len(done)}, remaining for pre-filter: {len(remaining)}")
    else:
        remaining = full

    if len(remaining) == 0:
        print("All companies already processed!")
        return False

    # Save as semicolon-separated CSV for step 01
    export_path = CRUNCH / f"{prefilter_id}_01_crunchbase__raw_export.csv"
    remaining.to_csv(export_path, sep=";", index=False)
    print(f"Exported {len(remaining)} companies to {export_path}")

    env = {"PIPELINE_RUN_ID": prefilter_id}

    # Step 01: SEC Form D enrichment
    ok = run_script("Pre-filter Step 01: SEC Form D enrichment",
                     "01__formd__seriesA_dates_enrich.py", env, timeout=7200)
    if not ok:
        print("\nPre-filter Step 01 FAILED")
        return False

    # Step 02: Core feature cleaning (this is where the ~75% get filtered)
    ok = run_script("Pre-filter Step 02: Core feature cleaning",
                     "02__clean__core_features.py", env, timeout=3600)
    if not ok:
        print("\nPre-filter Step 02 FAILED")
        return False

    # Read the survivors from step 02 output
    step03_output = PIPE / f"{prefilter_id}_03_formd__seriesA_dates_enriched__clean.csv"
    if not step03_output.exists():
        print(f"\nERROR: Expected output {step03_output} not found")
        return False

    survivors = pd.read_csv(step03_output)
    survivors.to_csv(PREFILTERED_CSV, index=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  PRE-FILTER COMPLETE")
    print(f"  Input:     {len(remaining)} companies")
    print(f"  Survivors: {len(survivors)} companies ({100*len(survivors)/len(remaining):.1f}%)")
    print(f"  Filtered:  {len(remaining) - len(survivors)} companies")
    print(f"  Saved to:  {PREFILTERED_CSV}")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    return True


def pick_batch_from_prefiltered(batch_size):
    """Pick next batch from pre-filtered survivors, skipping already-done companies."""
    if not PREFILTERED_CSV.exists():
        print(f"ERROR: {PREFILTERED_CSV} not found. Run --prefilter first.")
        return None, None, None

    survivors = pd.read_csv(PREFILTERED_CSV)
    name_col = "Organization Name" if "Organization Name" in survivors.columns else "Organization_Name"
    print(f"Pre-filtered survivors: {len(survivors)} companies")

    done = load_done_names()
    print(f"Already processed: {len(done)}")

    remaining = survivors[~survivors[name_col].isin(done)]
    print(f"Remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("All pre-filtered companies processed!")
        return None, None, None

    batch = remaining.head(batch_size)
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # Save as the step 03 output (steps 01+02 already done during prefilter)
    # The orchestrator (step 04) reads from {RUN_ID}_03_formd__...__clean.csv
    step03_path = PIPE / f"{batch_id}_03_formd__seriesA_dates_enriched__clean.csv"
    batch.to_csv(step03_path, index=False)

    # Also write .run_id so downstream scripts can find it
    (PIPE / ".run_id").write_text(batch_id)

    company_names = batch[name_col].tolist()
    print(f"Batch: {batch_id} ({len(batch)} companies from pre-filtered)")
    print(f"Step 03 file: {step03_path}")
    return batch_id, step03_path, company_names


# ── Batch selection ───────────────────────────────────────────────────────

def pick_batch(crunchbase_csv, batch_size):
    """Pick next unprocessed companies, save as semicolon-separated Crunchbase export."""
    full = pd.read_csv(crunchbase_csv)
    print(f"Full dataset: {len(full)} companies")

    done = load_done_names()
    print(f"Already processed: {len(done)}")

    # Filter by both possible column names
    name_col = "Organization Name" if "Organization Name" in full.columns else "Organization_Name"
    remaining = full[~full[name_col].isin(done)]
    print(f"Remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("All companies processed!")
        return None, None, None

    batch = remaining.head(batch_size)
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # Save as semicolon-separated CSV (step 01 expects this)
    export_path = CRUNCH / f"{batch_id}_01_crunchbase__raw_export.csv"
    batch.to_csv(export_path, sep=";", index=False)

    company_names = batch[name_col].tolist()
    print(f"Batch: {batch_id} ({len(batch)} companies)")
    print(f"Export: {export_path}")
    return batch_id, export_path, company_names


# ── Step runner ───────────────────────────────────────────────────────────

def run_script(label, script_name, env_overrides=None, timeout=7200):
    """Run a Python script as a subprocess with real-time output."""
    script_path = CODE_DIR / script_name
    if not script_path.exists():
        print(f"  ERROR: {script_path} not found")
        return False

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    print(f"\n{'━' * 60}")
    print(f"  {label}")
    print(f"  Script: {script_name}")
    print(f"{'━' * 60}")

    start = time.time()
    try:
        proc = subprocess.Popen(
            [str(VENV_PYTHON), "-u", str(script_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  | {line}", flush=True)
        proc.wait(timeout=timeout)
        elapsed = time.time() - start

        if proc.returncode != 0:
            print(f"  FAILED (exit code {proc.returncode}) after {elapsed:.0f}s")
            return False
        print(f"  Done in {elapsed:.0f}s")
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        proc.kill()
        return False
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False


# ── Main pipeline ─────────────────────────────────────────────────────────

def run_pipeline(batch_id, skip_step01=False, only_step07=False):
    """Run the full pipeline for a batch."""
    env = {"PIPELINE_RUN_ID": batch_id}
    t0 = time.time()

    if only_step07:
        # Just run step 05 on existing merged signals
        ok = run_script("Step 05: Final dataset", "05__clean__final_dataset.py", env)
        if not ok:
            print("\nStep 05 FAILED")
            return False
        print(f"\nStep 05 complete in {time.time() - t0:.0f}s")
        return True

    if not skip_step01:
        # Step 01: SEC Form D enrichment
        ok = run_script("Step 01: SEC Form D enrichment", "01__formd__seriesA_dates_enrich.py", env)
        if not ok:
            print("\nStep 01 FAILED — aborting pipeline")
            return False

        # Step 02: Core feature cleaning
        ok = run_script("Step 02: Core feature cleaning", "02__clean__core_features.py", env)
        if not ok:
            print("\nStep 02 FAILED — aborting pipeline")
            return False

    # Steps 03+04: LLM classify + all signal steps in parallel
    ok = run_script(
        "Steps 03+04: LLM classify + parallel signal orchestrator",
        "04__parallel_orchestrator.py", env,
        timeout=14400,  # 4 hours (GDELT is slow at 1 req/5s)
    )
    if not ok:
        print("\nOrchestrator FAILED — step 05 will not run")
        return False

    # Step 05: Final dataset
    ok = run_script("Step 05: Final dataset merge & clean", "05__clean__final_dataset.py", env)
    if not ok:
        print("\nStep 05 FAILED")
        return False

    total = time.time() - t0
    print(f"\n{'━' * 60}")
    print(f"  Pipeline complete: {total:.0f}s ({total/60:.1f} min)")
    print(f"{'━' * 60}")

    # Show output files
    for pattern in [f"{batch_id}_05_final_dataset.csv",
                    f"{batch_id}_05_final_train.csv",
                    f"{batch_id}_05_final_test.csv"]:
        p = PIPE / pattern
        if p.exists():
            df = pd.read_csv(p)
            print(f"  {p.name}: {len(df)} rows × {len(df.columns)} cols")

    return True


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single orchestrated batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pre-filter workflow (recommended for 15K+ datasets):
  python Code/run_batch.py --prefilter                              # run once
  python Code/run_batch.py --from_prefiltered --batch_size 2000     # repeat
        """,
    )
    parser.add_argument("--crunchbase_csv", type=str, default=str(FULL_16K),
                        help="Path to full Crunchbase CSV (default: full_remaining_16k.csv)")
    parser.add_argument("--batch_size", type=int, default=2000,
                        help="Companies per batch (default: 2000)")
    parser.add_argument("--batch_id", type=str, default=None,
                        help="Resume a specific batch ID (skip batch selection)")
    parser.add_argument("--skip_step01", action="store_true",
                        help="Skip steps 01-02, start from orchestrator")
    parser.add_argument("--only_step07", action="store_true",
                        help="Only run step 05 on existing merged signals")
    parser.add_argument("--no_mark_done", action="store_true",
                        help="Don't update tracking CSV after completion")
    parser.add_argument("--status", action="store_true",
                        help="Show batch tracking status and exit")
    # Pre-filter flags
    parser.add_argument("--prefilter", action="store_true",
                        help="Run steps 01+02 on full dataset, save survivors to Pipeline/prefiltered_survivors.csv")
    parser.add_argument("--from_prefiltered", action="store_true",
                        help="Read batches from prefiltered_survivors.csv (skips steps 01+02)")
    args = parser.parse_args()

    if args.status:
        done = load_done_names()
        print(f"Total processed: {len(done)} companies")
        if TRACKING_CSV.exists():
            df = pd.read_csv(TRACKING_CSV)
            print(f"Batches: {df['batch_id'].nunique()}")
            print(df.groupby("batch_id").agg(
                companies=("Organization_Name", "count"),
                date=("processed_date", "first")
            ).to_string())
        if PREFILTERED_CSV.exists():
            pf = pd.read_csv(PREFILTERED_CSV)
            pf_remaining = len(pf) - len(done.intersection(
                set(pf["Organization_Name"].tolist()) if "Organization_Name" in pf.columns
                else set(pf["Organization Name"].tolist())
            ))
            print(f"\nPre-filtered survivors: {len(pf)} total, {pf_remaining} remaining")
        else:
            print("\nNo pre-filtered file found (run --prefilter first)")
        return

    # ── Pre-filter mode ──────────────────────────────────────────────
    if args.prefilter:
        # Seed tracking with already-done new2k archive companies
        seed_tracking_from_archive()
        # Run steps 01+02 on the full dataset
        ok = run_prefilter(args.crunchbase_csv)
        if not ok:
            sys.exit(1)
        return

    # ── From pre-filtered mode ───────────────────────────────────────
    if args.from_prefiltered:
        batch_id, step03_path, company_names = pick_batch_from_prefiltered(args.batch_size)
        if batch_id is None:
            return

        # Run pipeline skipping steps 01+02 (already done in prefilter)
        ok = run_pipeline(batch_id, skip_step01=True, only_step07=args.only_step07)

        if ok and company_names and not args.no_mark_done:
            mark_done(company_names, batch_id)

        if not ok:
            print(f"\nPipeline failed for batch {batch_id}.")
            print(f"To resume: python Code/run_batch.py --batch_id {batch_id} --skip_step01")
            sys.exit(1)
        return

    # ── Standard mode (original behavior) ────────────────────────────

    # Select or resume batch
    if args.batch_id:
        batch_id = args.batch_id
        company_names = None  # Can't mark done without names
        print(f"Resuming batch: {batch_id}")
    else:
        batch_id, export_path, company_names = pick_batch(args.crunchbase_csv, args.batch_size)
        if batch_id is None:
            return

    # Run the pipeline
    ok = run_pipeline(batch_id, skip_step01=args.skip_step01, only_step07=args.only_step07)

    # Mark as done
    if ok and company_names and not args.no_mark_done:
        mark_done(company_names, batch_id)
    elif ok and not company_names and args.batch_id:
        print("(Cannot mark done — no company list available when resuming by batch_id)")

    if not ok:
        print(f"\nPipeline failed for batch {batch_id}.")
        print(f"To resume: python Code/run_batch.py --batch_id {batch_id} --skip_step01")
        sys.exit(1)


if __name__ == "__main__":
    main()

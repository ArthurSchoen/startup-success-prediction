#!/usr/bin/env python
# coding: utf-8
"""
concurrent_runner.py — Checkpoint/Resume Concurrent Processing

Reusable module for running API-bound signal collection with:
  - ThreadPoolExecutor for concurrent requests
  - Checkpoint/resume (saves midpoint CSV every N rows)
  - Per-row error isolation (one failure doesn't crash the batch)
  - Rate limiting (configurable delay between submissions)

Usage:
    from utils.concurrent_runner import process_dataframe

    def my_signal_fn(row_dict):
        name = row_dict["Organization_Name"]
        # ... call API ...
        return {"signal_count": 42, "signal_note": "ok"}

    df = process_dataframe(
        df=df,
        func=my_signal_fn,
        output_csv="output.csv",
        marker_col="signal_note",
        max_workers=5,
        checkpoint_every=50,
        delay=0.5,
    )

Author: Arthur Schoen — MIT / HEC Paris
"""

import time
import threading
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def process_dataframe(
    df: pd.DataFrame,
    func: callable,
    output_csv: str | Path,
    marker_col: str = None,
    max_workers: int = 1,
    checkpoint_every: int = 100,
    delay: float = 0.5,
    desc: str = "Processing",
) -> pd.DataFrame:
    """
    Apply `func` to each row of `df` concurrently, with checkpointing.

    Parameters
    ----------
    df : DataFrame
        Input data. Each row is passed to `func` as a dict.
    func : callable
        Function(row_dict) -> dict of new column values.
    output_csv : Path
        Where to save the final output.
    marker_col : str, optional
        Column name that marks a row as "already processed".
        Rows where this column is non-empty are skipped on resume.
    max_workers : int
        Thread pool size (1 = sequential).
    checkpoint_every : int
        Save a midpoint CSV every N completed rows.
    delay : float
        Seconds between task submissions (rate limiting).
    desc : str
        Description for the progress bar.

    Returns
    -------
    DataFrame with new columns added by `func`.
    """
    output_csv = Path(output_csv)
    midpoint_csv = output_csv.with_suffix(".csv__midpoint.csv")

    # Resume from midpoint if it exists
    if midpoint_csv.exists():
        print(f"  Resuming from midpoint: {midpoint_csv.name}")
        df_mid = pd.read_csv(midpoint_csv)
        # Merge already-processed rows
        if marker_col and marker_col in df_mid.columns:
            done_mask = df_mid[marker_col].notna() & (df_mid[marker_col] != "")
            done_names = set(df_mid.loc[done_mask, "Organization_Name"])
            print(f"  Already processed: {len(done_names)}/{len(df)}")
        else:
            done_names = set()
        df = df_mid.copy()
    else:
        done_names = set()

    # Identify rows to process
    if marker_col and marker_col in df.columns:
        todo_mask = ~df["Organization_Name"].isin(done_names)
    else:
        todo_mask = pd.Series(True, index=df.index)

    todo_indices = df.index[todo_mask].tolist()
    if not todo_indices:
        print(f"  All rows already processed.")
        df.to_csv(output_csv, index=False)
        return df

    print(f"  Processing {len(todo_indices)} rows ({max_workers} workers, {delay}s delay)")

    lock = threading.Lock()
    completed = 0

    def process_row(idx):
        nonlocal completed
        row_dict = df.loc[idx].to_dict()
        try:
            result = func(row_dict)
        except Exception as e:
            result = {marker_col: f"ERROR: {e}"} if marker_col else {}
        with lock:
            for k, v in result.items():
                if k not in df.columns:
                    df[k] = None
                df.at[idx, k] = v
            completed += 1
            if completed % checkpoint_every == 0:
                df.to_csv(midpoint_csv, index=False)
        return idx

    if max_workers == 1:
        # Sequential mode (for debugging or strict rate-limited APIs)
        for idx in tqdm(todo_indices, desc=desc):
            process_row(idx)
            time.sleep(delay)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx in todo_indices:
                f = executor.submit(process_row, idx)
                futures[f] = idx
                time.sleep(delay)
            for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
                f.result()  # propagate exceptions

    # Save final output
    df.to_csv(output_csv, index=False)
    # Clean up midpoint
    if midpoint_csv.exists():
        midpoint_csv.unlink()
    print(f"  Saved: {output_csv.name}")
    return df

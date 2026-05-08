"""
shared/concurrent_runner.py — Reusable concurrent processing with checkpoints.

Usage in any signal step:

    from shared.concurrent_runner import process_dataframe

    def my_signal_fn(row_dict):
        '''Process one company. Returns dict of new column values.'''
        name = row_dict["Organization_Name"]
        return {"my_signal_count": 42, "my_signal_note": "ok"}

    df = process_dataframe(
        df=df,
        func=my_signal_fn,
        output_csv=OUTPUT_CSV,
        marker_col="my_signal_note",   # skip rows where this is already filled (resume)
        max_workers=5,                  # concurrent threads
        checkpoint_every=50,            # save progress every N rows
        delay=0.5,                      # per-request delay (rate limiting)
    )

Features:
  - Concurrent execution with ThreadPoolExecutor
  - Checkpoint/resume: saves midpoint CSV every N rows, skips already-processed rows
  - Rate limiting: configurable delay between requests
  - Progress bar with tqdm
  - Graceful error handling per row (doesn't crash the batch)
"""

import time
import threading
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def process_dataframe(
    df,
    func,
    output_csv,
    marker_col=None,
    max_workers=1,
    checkpoint_every=100,
    delay=0.5,
    desc="Processing",
):
    """
    Apply func to each row of df with concurrency, checkpoints, and resume.

    Args:
        df: DataFrame to process
        func: callable(row_dict) -> dict of new column values
        output_csv: Path to save output (and midpoint checkpoints)
        marker_col: if set, skip rows where this column is already non-empty (resume)
        max_workers: number of concurrent threads (1 = sequential)
        checkpoint_every: save midpoint CSV every N completed rows
        delay: seconds to sleep between submissions (rate limiting)
        desc: tqdm description

    Returns:
        df with new columns populated
    """
    output_csv = Path(output_csv)
    midpoint_csv = output_csv.parent / (output_csv.stem + "__midpoint.csv")

    # Resume: load midpoint if it exists and merge already-processed rows
    if midpoint_csv.exists() and marker_col:
        try:
            df_mid = pd.read_csv(midpoint_csv)
            df_mid.columns = df_mid.columns.str.strip()
            if marker_col in df_mid.columns and "Organization_Name" in df_mid.columns:
                # Get columns that exist in midpoint but not in df (new signal columns)
                new_cols = [c for c in df_mid.columns if c not in df.columns]
                if new_cols:
                    merge_cols = ["Organization_Name"] + new_cols
                    df = df.merge(
                        df_mid[merge_cols].drop_duplicates("Organization_Name"),
                        on="Organization_Name",
                        how="left",
                    )
                else:
                    # Columns already exist in df; update values from midpoint
                    key_to_mid = df_mid.set_index("Organization_Name").to_dict("index")
                    for i in df.index:
                        name = str(df.loc[i, "Organization_Name"]).strip()
                        if name in key_to_mid:
                            mid_row = key_to_mid[name]
                            if marker_col in mid_row and str(mid_row.get(marker_col, "")).strip() not in ("", "nan"):
                                for col, val in mid_row.items():
                                    if col in df.columns:
                                        df.at[i, col] = val
                resumed = df[marker_col].astype(str).str.strip().ne("").sum() if marker_col in df.columns else 0
                print(f"  Resumed from midpoint: {resumed}/{len(df)} already processed")
        except Exception as e:
            print(f"  Midpoint load failed ({e}), starting fresh")

    # Determine which rows need processing
    if marker_col and marker_col in df.columns:
        todo_mask = df[marker_col].isna() | (df[marker_col].astype(str).str.strip().isin(["", "nan"]))
    else:
        todo_mask = pd.Series([True] * len(df), index=df.index)

    todo_indices = df.index[todo_mask].tolist()
    total_todo = len(todo_indices)
    already_done = len(df) - total_todo

    if already_done > 0:
        print(f"  Skipping {already_done} already-processed rows, {total_todo} remaining")

    if total_todo == 0:
        print(f"  All rows already processed!")
        return df

    # Process with concurrency
    completed_count = 0
    lock = threading.Lock()
    pbar = tqdm(total=total_todo, desc=desc, initial=0)

    def _process_one(idx):
        row = df.loc[idx].to_dict()
        try:
            result = func(row)
            return idx, result, None
        except Exception as e:
            return idx, None, str(e)

    if max_workers <= 1:
        # Sequential mode (simpler, for debugging)
        for idx in todo_indices:
            idx, result, err = _process_one(idx)
            if result:
                for k, v in result.items():
                    df.at[idx, k] = v
            elif err:
                if marker_col and marker_col in df.columns:
                    df.at[idx, marker_col] = f"error: {err[:100]}"
            completed_count += 1
            pbar.update(1)
            # Checkpoint
            if checkpoint_every and completed_count % checkpoint_every == 0:
                df.to_csv(midpoint_csv, index=False)
            time.sleep(delay)
    else:
        # Concurrent mode
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx in todo_indices:
                future = executor.submit(_process_one, idx)
                futures[future] = idx
                time.sleep(delay / max_workers)  # Spread submissions

            for future in as_completed(futures):
                idx, result, err = future.result()
                with lock:
                    if result:
                        for k, v in result.items():
                            df.at[idx, k] = v
                    elif err:
                        if marker_col and marker_col in df.columns:
                            df.at[idx, marker_col] = f"error: {err[:100]}"
                    completed_count += 1
                    pbar.update(1)
                    # Checkpoint
                    if checkpoint_every and completed_count % checkpoint_every == 0:
                        df.to_csv(midpoint_csv, index=False)

    pbar.close()

    # Final save
    df.to_csv(midpoint_csv, index=False)

    # Clean up midpoint after successful completion
    if midpoint_csv.exists():
        midpoint_csv.unlink()

    return df

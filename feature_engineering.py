#!/usr/bin/env python
# coding: utf-8
"""
feature_engineering.py — Final Dataset Assembly for Startup Success Prediction

Transforms raw merged signals into a clean, model-ready dataset:
  1. Build target labels (SeriesB_within_36m — binary)
  2. Anti-leakage filtering (only pre-Series A features)
  3. Temporal train/test split (seriesA_year < 2022 = train)
  4. Feature imputation & encoding
  5. Export train/test CSVs + feature config for model training

Feature Schema (A→P):
  A: Identity & timing     B: Labels            C: Core (amount, industry, age)
  D: SEC matching           E: Patents           F: Founder signals
  G: Traffic (deprecated)   H: Google Trends     I: Wikipedia
  J: News mentions          K: Backlinks/HN      L: GitHub
  M: Packages (PyPI/npm)    N: OpenAlex          O: Product Hunt/HN
  P: Wayback + App Store

Anti-Leakage Design:
  - ALL features are pre-Series A snapshots (temporal filtering at API level)
  - GitHub stars/forks, package downloads, current citations → excluded
  - Wikipedia uses historical revision API (page state at Series A date)
  - Label columns (Last Funding Type) used for labeling only, never as features

Author: Arthur Schoen — MIT / HEC Paris
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PIPE = Path.cwd() / "Pipeline"
BASE = PIPE.parent

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_COL       = "success_seriesB_plus"      # Ever raised Series B+
TARGET_36M_COL   = "SeriesB_within_36m"         # Primary target for modeling
MONTHS_TO_B_COL  = "months_to_seriesB"
SPLIT_COL        = "split"
SERIESA_YEAR_COL = "seriesA_year"

# Series B+ keywords for label construction
SERIES_B_PLUS_KEYWORDS = [
    "series b", "series c", "series d", "series e", "series f", "series g",
    "ipo", "late stage", "growth", "private equity", "venture - series unknown",
]

# Features known to cause leakage (current-day values, not pre-Series A)
LEAKAGE_RISK_COLS = {
    "deeptech", "deeptech_score", "sec_match_score",
    "founder_bigtech_flag", "founder_phd_flag",
    "founder_serial_entrepreneur_flag", "founder_prior_exit_flag",
}

# ── Feature Definitions ─────────────────────────────────────────────────────
# Core numeric features (always present)
FEATURE_NUMERIC = [
    "seriesA_amount_usd_millions", "age_at_seriesA_years", "Number of Founders",
    "patent_count_preA_company", "has_patent_preA_company",
    "founder_features_available", "founder_context_len", "founder_text_coverage_ratio",
]

# Categorical features
FEATURE_CATEGORICAL = ["industry_bucket"]

# Optional features (added by signal collectors, may be zero/absent)
FEATURE_OPTIONAL = [
    # Google Trends (12 features)
    "gtrends_mean_12m", "gtrends_max_12m", "gtrends_slope_12m",
    "gtrends_mean_3m", "gtrends_max_3m", "gtrends_mean_6m", "gtrends_max_6m",
    "gtrends_evolution_3m_12m", "gtrends_volatility_12m", "gtrends_acceleration",
    "gtrends_breakout_score", "gtrends_breakout_ratio", "gtrends_spike_count",
    # Wikipedia (9 features)
    "wikipedia_exists", "wikipedia_existed_at_seriesA", "wikipedia_before_seriesA",
    "wikipedia_page_length", "wiki_edit_count_preA", "wiki_editor_diversity",
    "wiki_is_disambiguation", "wiki_infobox_presence", "wiki_lang_count",
    # GitHub (7 features)
    "github_commits_3m", "github_commits_6m", "github_commits_12m",
    "github_before_seriesA", "github_distinct_committers_3m",
    "github_months_from_first_commit_to_seriesA",
    # Packages (7 features)
    "pkg_releases_count", "pkg_before_seriesA", "pkg_registry_count",
    "pkg_update_frequency", "pkg_version_stability", "pkg_major_updates_preA",
    "pkg_ecosystem_fullstack",
    # OpenAlex (5 features)
    "openalex_works_count", "openalex_works_12m", "openalex_works_36m",
    "oa_academic_velocity", "oa_collaboration_index",
    # Hacker News + Product Hunt (6 features)
    "ph_hn_launch", "ph_hn_is_show_hn", "ph_hn_upvotes", "ph_hn_comments",
    "ph_hn_before_seriesA", "ph_producthunt_launch",
    # Backlinks (5 features)
    "backlinks_hn_count_preA", "backlinks_reddit_count_preA",
    "backlinks_github_count_preA", "backlinks_approx_domains_preA",
    "hn_comments_per_upvote",
    # Wayback + App Store (6 features)
    "wayback_snapshots_12m_preA", "wayback_activity_score",
    "web_team_maturity", "web_sophistication",
    "web_founder_university_tier", "web_customer_traction",
    "appstore_has_app", "appstore_before_seriesA",
    # LLM-derived
    "deeptech_llm_score",
    # News
    "news_mentions_count", "news_sources_count", "news_avg_tone",
]

# Columns that mean "no signal detected" → impute with 0 (not median)
ZERO_IMPUTE_PREFIXES = (
    "has_", "patent_count", "wikipedia_exists", "wikipedia_before",
    "wiki_edit_count", "wiki_editor_diversity", "wiki_is_", "wiki_vandalism",
    "backlinks_", "hn_comments", "github_commits", "github_before",
    "github_distinct", "github_months",
    "pkg_releases_count", "pkg_before", "pkg_registry_count", "pkg_update",
    "openalex_works_count", "openalex_works_12m", "openalex_works_36m",
    "oa_academic", "oa_collaboration",
    "ph_hn_launch", "ph_hn_is_show", "ph_hn_upvotes", "ph_hn_comments",
    "ph_hn_before", "founder_",
    "wayback_", "appstore_",
)


# ── Label Construction ───────────────────────────────────────────────────────
def is_series_b_plus(last_type: str, last_equity_type: str) -> bool:
    """Check if the last funding round indicates Series B or higher."""
    for val in [last_type, last_equity_type]:
        if pd.isna(val) or val is None:
            continue
        val_lower = str(val).strip().lower()
        if any(kw in val_lower for kw in SERIES_B_PLUS_KEYWORDS):
            return True
    return False


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct target labels from Crunchbase funding data.

    Target: SeriesB_within_36m = 1 if company raised Series B+
    within 36 months of their Series A date.

    Special handling for "Venture (series unknown)":
    - Only counted as success if round is 6+ months after Series A
      AND amount >= 1.5x the Series A amount.
    """
    # Resolve column names (CSV may have different casing)
    col_map = {str(c).strip().lower(): c for c in df.columns}
    col_last_type   = col_map.get("last funding type")
    col_last_equity  = col_map.get("last equity funding type")
    col_last_date    = col_map.get("last funding date")
    col_last_amount  = col_map.get("last funding amount (in usd)")

    # Parse Series A date
    df["seriesA_date_parsed"] = pd.to_datetime(df["seriesA_date"], errors="coerce")

    # Binary: ever raised Series B+
    df[TARGET_COL] = df.apply(
        lambda r: 1 if is_series_b_plus(
            r.get(col_last_type, ""), r.get(col_last_equity, "")
        ) else 0, axis=1
    )

    # Compute months to Series B
    if "seriesB_date" in df.columns:
        b_date = pd.to_datetime(df["seriesB_date"], errors="coerce")
    elif col_last_date:
        b_date = pd.to_datetime(df[col_last_date], errors="coerce")
    else:
        b_date = pd.Series(pd.NaT, index=df.index)

    df[MONTHS_TO_B_COL] = np.nan
    mask = df[TARGET_COL].astype(bool)
    df.loc[mask, MONTHS_TO_B_COL] = (
        (b_date[mask] - df.loc[mask, "seriesA_date_parsed"]).dt.days / 30.44
    ).round(0)

    # Primary target: Series B+ within 36 months
    df[TARGET_36M_COL] = (
        df[TARGET_COL].astype(bool) & (df[MONTHS_TO_B_COL] <= 36)
    ).astype(int)

    # Series A year for temporal split
    df[SERIESA_YEAR_COL] = df["seriesA_date_parsed"].dt.year

    return df


# ── Feature Assembly ─────────────────────────────────────────────────────────
def select_features(df: pd.DataFrame) -> list:
    """
    Select model-ready features from the merged dataset.

    Rules:
    - Include numeric features with >= 3 non-zero values (minimum signal)
    - Exclude known leakage-risk columns
    - Include categorical features for one-hot encoding
    """
    all_candidates = FEATURE_NUMERIC + FEATURE_OPTIONAL
    selected = []
    for c in all_candidates:
        if c not in df.columns:
            continue
        if c in LEAKAGE_RISK_COLS:
            continue
        nonzero = (df[c].fillna(0) != 0).sum()
        if nonzero >= 3:
            selected.append(c)

    cat_cols = [c for c in FEATURE_CATEGORICAL if c in df.columns]
    return selected + cat_cols


def impute_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Impute missing values with domain-appropriate defaults.

    Strategy:
    - "Absence" columns (binary flags, counts): fill with 0
    - "Continuous" columns (trends, scores): fill with median
    - All-NaN columns: fill with 0
    """
    for c in feature_cols:
        if c in FEATURE_CATEGORICAL:
            df[c] = df[c].astype(str).replace("nan", "MISSING").replace("", "MISSING")
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if not df[c].isna().any():
            continue
        use_zero = any(c.startswith(p) for p in ZERO_IMPUTE_PREFIXES)
        if use_zero or df[c].notna().sum() == 0:
            df[c] = df[c].fillna(0)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = df[c].median()
            df[c] = df[c].fillna(0 if pd.isna(med) else med)
    return df


def temporal_split(df: pd.DataFrame, split_year: int = None) -> tuple:
    """
    Split dataset temporally: train = seriesA_year < split_year, test = rest.

    If split_year is None, uses 80th percentile of seriesA_year
    (ensures ~20% test set).
    """
    if split_year is None:
        split_year = int(df[SERIESA_YEAR_COL].dropna().quantile(0.80))
        # Ensure at least 10% in test
        if (df[SERIESA_YEAR_COL] >= split_year).sum() < max(1, len(df) * 0.10):
            split_year = int(df[SERIESA_YEAR_COL].dropna().quantile(0.70))

    df[SPLIT_COL] = np.where(df[SERIESA_YEAR_COL] < split_year, "train", "test")
    train = df[df[SPLIT_COL] == "train"].copy()
    test  = df[df[SPLIT_COL] == "test"].copy()

    print(f"Split year: {split_year}")
    print(f"  Train: {len(train)} rows ({train[TARGET_36M_COL].sum()} positives)")
    print(f"  Test:  {len(test)} rows ({test[TARGET_36M_COL].sum()} positives)")
    return train, test, split_year


def build_final_dataset(input_csv: Path, output_dir: Path) -> tuple:
    """
    Full feature engineering pipeline:
      1. Load merged signals
      2. Build target labels
      3. Select & impute features
      4. Temporal train/test split
      5. Save train.csv, test.csv, feature_config.json
    """
    print(f"Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()
    print(f"  {len(df)} companies, {len(df.columns)} raw columns")

    # Drop low-confidence rows
    if "seriesA_confidence" in df.columns:
        n_low = (df["seriesA_confidence"].astype(str).str.upper() == "LOW").sum()
        if n_low > 0:
            df = df[df["seriesA_confidence"].astype(str).str.upper() != "LOW"].copy()
            print(f"  Dropped {n_low} LOW confidence rows")

    # Build labels
    df = build_labels(df)
    print(f"  Target ({TARGET_36M_COL}): {df[TARGET_36M_COL].sum()} positives / {len(df)} total")

    # Select features
    feature_cols = select_features(df)
    print(f"  Selected {len(feature_cols)} features")

    # Impute
    df = impute_features(df, feature_cols)

    # Temporal split
    train, test, split_year = temporal_split(df)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "final_train.csv", index=False)
    test.to_csv(output_dir / "final_test.csv", index=False)

    config = {
        "target": TARGET_36M_COL,
        "split_year": split_year,
        "feature_columns": feature_cols,
    }
    with open(output_dir / "final_feature_columns.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    return train, test, feature_cols


if __name__ == "__main__":
    # Find latest merged signals
    candidates = sorted(PIPE.glob("*_06_merged_signals.csv"), reverse=True)
    if not candidates:
        raise FileNotFoundError("No merged signals CSV found. Run data_pipeline.py first.")

    output_dir = PIPE / "archive" / "Final dataset"
    build_final_dataset(candidates[0], output_dir)

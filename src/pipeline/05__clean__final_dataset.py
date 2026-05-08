#!/usr/bin/env python
# coding: utf-8

# # 05 – Final dataset for analysis (06)
#
# **Objectif** : Préparer un jeu de données propre pour la modélisation (étape 06).
#
# - **Label principal** : `SeriesB_within_36m` (target binaire 06). `success_seriesB_plus` = audit / dérivé (Series B+ ever).
# - **Leakage-safe** : uniquement des features connues **avant / au moment** de la Series A (snapshot pré-Series A).
# - **Split temporel** : train = seriesA_year < 2022, test = récent (pour 06 : LogReg, XGBoost/LightGBM, calibration, SHAP).
#
# **Schéma colonnes (A→O)** : A = Identité + timing (index table) ; B = Label (SeriesB_within_36m) ; C = Crunchbase core ; D = SEC ; E = Patents ; F = Founders ; G = Traffic ; I = Wikipedia ; J = News ; K = Backlinks/HN ; L = GitHub ; M = Packages ; N = OpenAlex ; O = PH/HN launch.
#
# **Notes redondance** : Traffic (SEMrush) removed — API unavailable. Founder embeddings removed — overfitting risk. Google Trends removed — noisy, no published benchmark. Wikipedia → wikipedia_before_seriesA = truth time-safe. Backlinks → préférence version pré-A pour éviter leakage.
#
# **Dictionnaire** : `features/FEATURES_FINAL_DATASET.md`, `features/FEATURES_ALL.md`.

# In[75]:


import os
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))
from shared import load_env
load_env()

# Use the Pipeline folder that contains .run_id (written by 07) so we use the same run as 01
PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline", Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
PIPE = next((p for p in PIPE_CANDIDATES if p.exists() and (p / ".run_id").exists()), None) or (PIPE_CANDIDATES[0] if PIPE_CANDIDATES[0].exists() else Path.home() / "Desktop" / "Data_thesis" / "Pipeline")
BASE = PIPE.parent
PIPE.mkdir(parents=True, exist_ok=True)

def run_id_from_filename(name: str) -> str:
    m = re.match(r"^([^_]+)_", Path(name).name)
    if not m:
        raise ValueError(f"Bad filename (expected RUNID_...): {name}")
    return m.group(1)

def pick_next_file_for_step(pipe_dir: Path, prev_step: str, next_step_prefix: str, allow_overwrite: bool = False) -> Path:
    candidates = []
    pipe_matches = list(pipe_dir.glob(f"*{prev_step}*.csv"))
    for p in pipe_matches:
        if "__midpoint" in p.name:
            continue
        run_id = p.name.split("_", 1)[0]
        next_exists = any(pipe_dir.glob(f"{run_id}{next_step_prefix}*.csv"))
        if allow_overwrite or not next_exists:
            candidates.append((run_id, p))
    if not candidates:
        raise FileNotFoundError(f"No runnable input for prev_step={prev_step}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

ALLOW_OVERWRITE = True
CHOSEN_RUN_ID = os.environ.get("PIPELINE_RUN_ID", "").strip() or ((PIPE / ".run_id").read_text().strip() if (PIPE / ".run_id").exists() else "")
def _find_input_csv():
    if CHOSEN_RUN_ID:
        # Prefer the merged output from the parallel orchestrator (has ALL signals)
        merged = PIPE / f"{CHOSEN_RUN_ID}_04_merged_signals.csv"
        if merged.exists():
            return merged, "merged_signals"
        # Also check archive
        merged_archive = PIPE / "archive" / CHOSEN_RUN_ID / f"{CHOSEN_RUN_ID}_04_merged_signals.csv"
        if merged_archive.exists():
            return merged_archive, "merged_signals (archive)"
        # Fallback: individual signal files (legacy, pre-orchestrator)
        for fname, label in [
            (f"{CHOSEN_RUN_ID}_04n_producthunt_hn__signals.csv", "producthunt_hn"),
            (f"{CHOSEN_RUN_ID}_04l_openalex__signals.csv", "openalex"),
            (f"{CHOSEN_RUN_ID}_04k_packages__signals.csv", "packages"),
            (f"{CHOSEN_RUN_ID}_04j_github__signals.csv", "github"),
            (f"{CHOSEN_RUN_ID}_04i_backlinks_light__signals.csv", "backlinks light"),
            (f"{CHOSEN_RUN_ID}_04h_news_mentions__signals.csv", "news mentions"),
            (f"{CHOSEN_RUN_ID}_04g_wikipedia__signals.csv", "wikipedia"),
            (f"{CHOSEN_RUN_ID}_04e_patents__signals.csv", "patents"),
        ]:
            p = PIPE / fname
            if p.exists():
                return p, label
    for prev, label in [
        ("_04n_producthunt_hn__signals", "producthunt_hn"),
        ("_04l_openalex__signals", "openalex"),
        ("_04k_packages__signals", "packages"),
        ("_04j_github__signals", "github"),
        ("_04i_backlinks_light__signals", "backlinks light"),
        ("_04h_news_mentions__signals", "news mentions"),
        ("_04g_wikipedia__signals", "wikipedia"),
        ("_04e_patents__signals", "patents"),
    ]:
        try:
            return pick_next_file_for_step(PIPE, prev, "_05_final_", ALLOW_OVERWRITE), label
        except FileNotFoundError:
            continue
    archive_dir = PIPE / "archive"
    for pattern in ["*_04n_producthunt_hn__signals.csv", "*_04l_openalex__signals.csv", "*_04k_packages__signals.csv", "*_04j_github__signals.csv", "*_04i_backlinks_light__signals.csv", "*_04h_news_mentions__signals.csv", "*_04g_wikipedia__signals.csv", "*_04e_patents__signals.csv"]:
        candidates = list(archive_dir.glob(f"*/{pattern}")) + list(archive_dir.glob(f"*/archive_detaille/{pattern}"))
        candidates = [p for p in candidates if "__midpoint" not in p.name]
        candidates = list(dict.fromkeys(candidates))
        if candidates:
            candidates.sort(key=lambda p: (p.parent.name if p.parent.name != "archive_detaille" else p.parent.parent.name, p.name), reverse=True)
            return candidates[0], "archive"
    raise FileNotFoundError("No _04_merged_signals or _04X_*__signals in Pipeline or archive")
INPUT_CSV, _input_label = _find_input_csv()
print(f"(entrée: {_input_label})")
RUN_ID = run_id_from_filename(INPUT_CSV)
OUTPUT_CSV = PIPE / f"{RUN_ID}_05_final_dataset.csv"
OUTPUT_TRAIN = PIPE / f"{RUN_ID}_05_final_train.csv"
OUTPUT_TEST = PIPE / f"{RUN_ID}_05_final_test.csv"

print("INPUT :", INPUT_CSV)
print("OUTPUT (full):", OUTPUT_CSV)
print("OUTPUT (train):", OUTPUT_TRAIN)
print("OUTPUT (test):", OUTPUT_TEST)


# In[76]:


# --------------- Label: Series B+ = success ---------------
# Last Funding Type / Last Equity Funding Type: Series B, C, D, E, IPO, Late Stage → 1
SERIES_B_PLUS_KEYWORDS = [
    "series b", "series c", "series d", "series e", "series f", "series g",
    "ipo", "late stage", "growth", "private equity", "venture - series unknown",
]

def _resolve_col(df, preferred_names):
    """Return first column in df that matches (strip, case-insensitive) one of preferred_names, else None."""
    if df is None or not hasattr(df, "columns"):
        return None
    colmap = {str(c).strip().lower(): c for c in df.columns}
    for name in preferred_names:
        key = name.strip().lower()
        if key in colmap:
            return colmap[key]
    return None

def is_series_b_plus(last_type, last_equity_type) -> bool:
    """True if Last Funding Type is Series B+; 'venture - series unknown' is handled separately with date/amount check."""
    def check(s) -> bool:
        if s is None or (hasattr(s, "__float__") and pd.isna(s)):
            return False
        t = str(s).strip().lower()
        if not t or t in ("nan", "none", "nat"):
            return False
        if "venture" in t and "series unknown" in t:
            return False  # handled in build_target_with_venture_unknown
        return any(k in t for k in SERIES_B_PLUS_KEYWORDS)
    return check(last_type) or check(last_equity_type)

def _is_venture_series_unknown(last_type, last_equity_type) -> bool:
    for s in (last_type, last_equity_type):
        if s is None or (hasattr(s, "__float__") and pd.isna(s)):
            continue
        t = str(s).strip().lower()
        if "venture" in t and "series unknown" in t:
            return True
    return False

SERIES_B_PLUS_REASON_LABELS = {
    "series b": "Series B", "series c": "Series C", "series d": "Series D",
    "series e": "Series E", "series f": "Series F", "series g": "Series G",
    "ipo": "IPO", "late stage": "Late Stage", "growth": "Growth",
    "private equity": "Private Equity", "venture - series unknown": "Venture (series unknown)",
}

def get_series_b_plus_reason(last_type, last_equity_type) -> str:
    """Return the reason (funding type) that classified as Series B+, or empty string if not success."""
    def first_match(s):
        if s is None or (hasattr(s, "__float__") and pd.isna(s)):
            return ""
        t = str(s).strip().lower()
        if not t or t in ("nan", "none", "nat"):
            return ""
        for k in SERIES_B_PLUS_KEYWORDS:
            if k in t:
                return SERIES_B_PLUS_REASON_LABELS.get(k, k.title())
        return ""
    return first_match(last_type) or first_match(last_equity_type) or ""

# --------------- Temporal split (train = older, test = recent) ---------------
# Dynamic: 80th percentile of seriesA_year so ~20% goes to test; fallback 2022
SPLIT_YEAR = None  # computed after seriesA_year is available

# --------------- Pre–Series A features only (no leakage) ---------------
# Metadata / identifiers (keep for traceability, not as model features in 06)
META_COLS = ["Organization_Name", "seriesA_date", "seriesA_quarter", "Website", "Founded Date"]

# Target and split — label principal = SeriesB_within_36m (target binaire 06); success_seriesB_plus = audit
PRIMARY_TARGET_COL = "SeriesB_within_36m"  # label principal (idéal target binaire)
TARGET_COL = "success_seriesB_plus"  # audit / dérivé (Series B+ ever)
SUCCESS_REASON_COL = "success_seriesB_plus_reason"  # reason classified as success (e.g. Series B, IPO)
SUCCESS_REASON_CONFIDENCE_COL = "success_seriesB_plus_reason_confidence"  # LOW/MEDIUM/HIGH only for Venture (series unknown)
TARGET_36M_COL = PRIMARY_TARGET_COL  # alias
SPLIT_COL = "split"
SERIESA_YEAR_COL = "seriesA_year"
MONTHS_TO_B_COL = "months_to_seriesB"  # for time-to-B analysis in 06 (NaN if never B+)

# Numeric/categorical features known at or before Series A (snapshot)
# seriesA_amount in millions only (merged from seriesA_amount_usd_num in 3c)
FEATURE_NUMERIC = [
    "seriesA_amount_usd_millions",
    "age_at_seriesA_years",
    "Number of Founders",
    # sec_match_score removed — it's a data quality metric, not a company signal
    # (all LOW-confidence rows already dropped by seriesA_confidence filter)
    "founder_features_available",
    # is_delaware_hq dropped: 1% coverage, flat signal
    # founder_name_in_company dropped: 0.4% coverage, flat signal
    # SEMrush traffic features removed — API not available, columns were always empty
]

FEATURE_CATEGORICAL = [
    "industry_bucket",
    "age_at_seriesA_flag",
]

# Optional features — matched to ACTUAL column names produced by signal steps
FEATURE_OPTIONAL = [
        # ── LLM deeptech classification (step 03) ──
        "deeptech_llm_score",                     # 0-10 score, 73% coverage
        # ── Founder features (step 04m Wayback extraction) ──
        # founder_bigtech_flag dropped: inverse signal (0.85x), ex-FAANG founders do worse
        # founder_phd_flag dropped: flat signal (1.07x) on 6% coverage
        "founder_serial_entrepreneur_flag",         # 4% — repeat founder (1.57x B signal)
        # ── Wikipedia — dropped: 0.4% coverage (2 data points), useless ──
        # ── GitHub / Packages / OpenAlex — sparse individually, combined below ──
        # github_commits_12m dropped: inverse signal (0.02x)
        # github_before_seriesA, pkg_before_seriesA, openalex_works_count: folded into has_technical_footprint
        # ── Hacker News (step 04e) — captured by log_hn_mentions ──
        # hn_launch dropped: inverse signal (0.55x), log_hn_mentions already captures HN
        # ── Product Hunt — dropped: 0.2% coverage ──
        # ── Technical footprint (derived: github + packages + openalex) ──
        "has_technical_footprint",                 # binary: any code/papers/packages pre-A (7%)
        # ── Clinical Trials (step 04g) — 1 feature (sector-specific: biotech) ──
        "ct_trials_count_preA",                    # count of registered trials (2%)
        # ── Federal Awards (step 04h) — 2 features (binary dropped: redundant with counts) ──
        "fed_contracts_count_preA",                # count of procurement contracts (7%)
        "fed_grants_count_preA",                   # count of SBIR/STTR grants (5%), r=0.20 with contracts
        # ── YC Alumni (step 04n) — 1 feature ──
        "yc_alumni_preA",                          # binary: in YC before Series A (2%)
        # ── Patents (step 04i) — 2 features ──
        "has_patent_preA_company",                 # binary: at least 1 patent (18%)
        "patent_count_preA_company",               # count of verified patents (18%)
        # ── Wayback (step 04m) — 4 features ──
        "wayback_snapshots_3m_preA",               # recent 3m traffic (52%), rank 11 in model
        "wayback_acceleration",                    # 3m/12m ratio: >1 = traffic accelerating pre-raise
        "has_careers_page_preA",                   # hiring signal (23%)
        "product_stage_preA",                      # 0-3 maturity from website (34%)
        # ── Wayback page extraction (step 04m) ──
        "named_customers_count_preA",              # named customer logos/mentions (9%)
        "founder_school_tier",                     # top school signal (3%)
        # ── Derived features (no API calls) ──
        "desc_word_count",                         # description length (100%)
        "visibility_score",                        # count of active sparse signals 0-3 (46%)
        "name_word_count",                         # words in company name (100%)
        "name_has_tech_suffix",                    # ends in AI/Labs/Tech etc (16%)
        "is_diverse_founded",                      # Crunchbase Diversity Spotlight (18%)
        # ── GDELT news (step 04j) — 2 features ──
        "gdelt_mentions_3m_preA",                  # recent 3m press count (9%), rank 8 in model
        "gdelt_acceleration",                      # 3m/12m ratio: >1 = press accelerating pre-raise
        # ── Form D investors (step 02) — 3 features ──
        "formd_directors_preA",                    # VC board seats pre-A (54%)
        "formd_investor_count_preA",               # total investors from offering table (56%)
        "formd_prior_rounds",                      # prior Form D filings = prior rounds (57%)
        # ── EDGAR filings (step 04k) — 1 feature ──
        "edgar_filings_preA",                      # SEC filing count pre-A (24%)
        # ── Engineered features (derived from existing signals) ──
        "log_patent_count",                        # log(1+patents) — compress outliers
        "log_hn_mentions",                         # log(1+hn_count) — inverse signal (more HN = worse)
        "log_wayback_3m",                          # log(1+snapshots) — compress outliers
        "log_edgar_filings",                       # log(1+filings) — compress outliers
        "log_gdelt_3m",                            # log(1+mentions) — compress outliers
        "directors_x_patent",                      # binary: has BOTH directors AND patent
        # team_growth_rate dropped: inverse signal (B=0 higher), team_size from Wayback
        # LLM is unreliable (confuses board/advisors/investors with employees)
        "directors_per_round",                     # directors / prior_rounds — investor quality
    ]

# Columns used ONLY for label (do not use as features)
LABEL_ONLY_COLS = ["Last Funding Type", "Last Equity Funding Type", "Last Funding Date", "Last Funding Amount (in USD)", "Total Funding Amount (in USD)", "Total Equity Funding Amount (in USD)"]

# Leakage safety net — most leakage features are no longer in FEATURE_OPTIONAL,
# but this catches any that might slip through from upstream CSVs.
# Full list of removed features: see features/FEATURES_REMOVED.md
LEAKAGE_RISK_COLS = [
    # Crunchbase post-Series A columns (used for labeling only)
    "Operating Status", "Number of Funding Rounds",
    "Total Funding Amount", "Total Funding Amount (in USD)",
    "Total Equity Funding Amount", "Total Equity Funding Amount (in USD)",
    "Last Funding Date", "Last Funding Type", "Last Funding Amount", "Last Funding Amount (in USD)",
    "Last Equity Funding Type", "Last Equity Funding Amount", "Last Equity Funding Amount (in USD)",
    "Number of Investors", "Acquisition Status", "Acquisition Type", "Number of Acquisitions",
    "Funding Status",
    # Deprecated
    "sec_match_score",
]


# In[77]:


df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
print("Rows loaded:", len(df))
print("Columns:", list(df.columns)[:20], "...")


# In[78]:


# --------------- 1) Drop rows without seriesA_date ---------------
if "seriesA_date" not in df.columns:
    raise ValueError("Missing seriesA_date column")
before = len(df)
df = df[df["seriesA_date"].notna() & (df["seriesA_date"].astype(str).str.strip() != "")].copy()
print(f"Dropped {before - len(df)} rows without seriesA_date")

# --------------- 1b) Exclure les startups avec age_at_seriesA_flag = SUSPECT ---------------
if "age_at_seriesA_flag" in df.columns:
    before_suspect = len(df)
    df = df[df["age_at_seriesA_flag"].astype(str).str.strip().str.upper() != "SUSPECT"].copy()
    if len(df) < before_suspect:
        print(f"Dropped {before_suspect - len(df)} rows with age_at_seriesA_flag = SUSPECT")

# --------------- 1c) Exclure les rows avec seriesA_confidence = LOW ---------------
if "seriesA_confidence" not in df.columns:
    formd_path = PIPE / f"{RUN_ID}_02_formd__seriesA_dates_enriched.csv"
    if not formd_path.exists():
        for arch in [PIPE / "archive", PIPE / "archive" / "archive_detaille"]:
            if arch.exists():
                found = list(arch.rglob(f"{RUN_ID}_02_formd__seriesA_dates_enriched*.csv"))
                if found:
                    formd_path = found[0]
                    break
    if formd_path.exists():
        df_formd = pd.read_csv(formd_path)
        df_formd.columns = df_formd.columns.str.strip()
        if "seriesA_confidence" in df_formd.columns and "Organization_Name" in df_formd.columns:
            conf = df_formd[["Organization_Name", "seriesA_confidence"]].drop_duplicates("Organization_Name")
            df = df.merge(conf, on="Organization_Name", how="left")
# Après merge, pandas peut créer seriesA_confidence_x / _y si les deux tables avaient la colonne → unifier
if "seriesA_confidence_y" in df.columns:
    df["seriesA_confidence"] = df["seriesA_confidence_y"].fillna(df.get("seriesA_confidence_x", pd.Series(dtype=object)))
    df = df.drop(columns=["seriesA_confidence_x", "seriesA_confidence_y"], errors="ignore")
elif "seriesA_confidence_x" in df.columns:
    df["seriesA_confidence"] = df["seriesA_confidence_x"]
    df = df.drop(columns=["seriesA_confidence_x"], errors="ignore")
# LOW confidence rows are now filtered in step 02 (early filtering to save API calls)
# Keep this as a safety net in case step 02 wasn't run
if "seriesA_confidence" in df.columns:
    n_low = (df["seriesA_confidence"].astype(str).str.strip().str.upper() == "LOW").sum()
    if n_low > 0:
        df = df[df["seriesA_confidence"].astype(str).str.strip().str.upper() != "LOW"].copy()
        print(f"Dropped {n_low} remaining LOW confidence rows (safety net)")

# --------------- 2) Build target: success = Series B+ ---------------
# Resolve column names (CSV may have different casing or spacing)
col_last_type = _resolve_col(df, ["Last Funding Type", "Last funding type"])
col_last_equity = _resolve_col(df, ["Last Equity Funding Type", "Last equity funding type"])
col_last_date = _resolve_col(df, ["Last Funding Date", "Last funding date"])
col_last_amount = _resolve_col(df, ["Last Funding Amount (in USD)", "Last funding amount (in USD)"])

if col_last_type is None and col_last_equity is None:
    raise ValueError(
        "Neither 'Last Funding Type' nor 'Last Equity Funding Type' found in CSV. "
        "Check column names (current: " + ", ".join(df.columns[:30]) + " ...)"
    )

# Ensure seriesA_date_parsed and seriesA_amount (USD) exist for Venture (series unknown) check
df["seriesA_date_parsed"] = pd.to_datetime(df["seriesA_date"], errors="coerce")
if "seriesA_amount_usd_num" in df.columns and "seriesA_amount_usd_millions" not in df.columns:
    df["seriesA_amount_usd_millions"] = (pd.to_numeric(df["seriesA_amount_usd_num"], errors="coerce").fillna(0) / 1e6).round(2)

def _parse_amount_usd(val):
    if pd.isna(val) or val is None or val == "":
        return np.nan
    s = str(val).strip().replace(",", "").replace("$", "").strip()
    return pd.to_numeric(s, errors="coerce")

def build_target_with_venture_unknown(r):
    lt, le = _get_val(r, col_last_type), _get_val(r, col_last_equity)
    reason = get_series_b_plus_reason(lt, le)
    if reason == "Venture (series unknown)":
        # Success only if: round at least 6 months after Series A and amount >= 1.5 * Series A
        seriesA_d = r.get("seriesA_date_parsed")
        if pd.isna(seriesA_d):
            return 0, reason, ""
        last_d = pd.to_datetime(r.get(col_last_date), errors="coerce") if col_last_date else pd.NaT
        if pd.isna(last_d) or last_d < seriesA_d + pd.DateOffset(months=6):
            return 0, "", ""
        seriesA_usd = (r.get("seriesA_amount_usd_millions") or 0) * 1e6
        last_usd = _parse_amount_usd(r.get(col_last_amount)) if col_last_amount else np.nan
        if pd.isna(last_usd) or seriesA_usd <= 0 or last_usd < 1.5 * seriesA_usd:
            return 0, "", ""
        ratio = last_usd / seriesA_usd
        if ratio >= 3:
            conf = "HIGH"
        elif ratio >= 2:
            conf = "MEDIUM"
        else:
            conf = "LOW"
        return 1, reason, conf
    success = 1 if is_series_b_plus(lt, le) else 0
    return success, reason if success else "", ""

def _get_val(r, col):
    return r.get(col, "") if col else ""

rows_target = df.apply(build_target_with_venture_unknown, axis=1)
df[TARGET_COL] = [x[0] for x in rows_target]
df[SUCCESS_REASON_COL] = [x[1] for x in rows_target]
df[SUCCESS_REASON_CONFIDENCE_COL] = [x[2] for x in rows_target]
print(f"Target columns used: Last Type={col_last_type!r}, Last Equity={col_last_equity!r}")
print(f"Target: {df[TARGET_COL].sum()} success (Series B+), {len(df) - df[TARGET_COL].sum()} not")
if df[TARGET_COL].sum() == 0 and len(df) > 0:
    print("WARNING: 0 success — check that 'Last Funding Type' / 'Last Equity Funding Type' contain values like 'Series B', 'Series C', 'IPO', 'Late Stage'.")

# --------------- 3) Series A year and temporal split ---------------
df["seriesA_date_parsed"] = pd.to_datetime(df["seriesA_date"], errors="coerce")
df[SERIESA_YEAR_COL] = df["seriesA_date_parsed"].dt.year
# Compute dynamic split year: 80th percentile → ~20% test; minimum test size = 10%
if SPLIT_YEAR is None:
    p80 = int(df[SERIESA_YEAR_COL].dropna().quantile(0.80))
    # Ensure at least 10% in test
    if (df[SERIESA_YEAR_COL] >= p80).sum() < max(1, len(df) * 0.10):
        p80 = int(df[SERIESA_YEAR_COL].dropna().quantile(0.70))
    SPLIT_YEAR = p80
    print(f"Split year (auto): {SPLIT_YEAR} (80th pctl of seriesA_year)")
df[SPLIT_COL] = np.where(df[SERIESA_YEAR_COL] < SPLIT_YEAR, "train", "test")
print(f"Split: train = seriesA_year < {SPLIT_YEAR} -> {(df[SPLIT_COL] == 'train').sum()}, test -> {(df[SPLIT_COL] == 'test').sum()}")

# --------------- 3b) Label: raised_series_B (no time window, 4 detection sources) ---------------
# We do NOT impose a 36-month window. Instead we rely on 5+ years of observation
# (2015-2020 cohorts analyzed in 2026) to ensure most B raises have occurred.

_last_date_col = col_last_date if col_last_date else "Last Funding Date"
_last_funding_dt = pd.to_datetime(df.get(_last_date_col), errors="coerce")
_months_from_a = ((_last_funding_dt - df["seriesA_date_parsed"]).dt.days / 30.44)

# months_to_seriesB (for analysis, not for label filtering)
df[MONTHS_TO_B_COL] = np.nan
if "seriesB_date" in df.columns:
    seriesB_date_parsed = pd.to_datetime(df["seriesB_date"], errors="coerce")
    use_seriesB_date = seriesB_date_parsed.notna() & (seriesB_date_parsed >= df["seriesA_date_parsed"])
    b_date = seriesB_date_parsed.where(use_seriesB_date).fillna(_last_funding_dt)
else:
    b_date = _last_funding_dt
mask_b_plus = df[TARGET_COL].astype(bool)
df.loc[mask_b_plus, MONTHS_TO_B_COL] = (
    (b_date[mask_b_plus] - df.loc[mask_b_plus, "seriesA_date_parsed"]).dt.days / 30.44
).round(0)
df[MONTHS_TO_B_COL] = pd.to_numeric(df[MONTHS_TO_B_COL], errors="coerce").round(0).astype("Int64")

# --- Source 1: Crunchbase Last Equity Funding Type = B+ ---
# Use Last EQUITY Funding Type (not Last Funding Type) to catch companies
# that raised B then took debt (Last Funding Type = "Debt" hides the B).
_eq_type_col = "Last Equity Funding Type"
_eq_type = df.get(_eq_type_col, pd.Series([""] * len(df))).fillna("").astype(str)
_b_plus_types = ["Series B", "Series C", "Series D", "Series E", "Series F", "Series G", "Series H"]
_cb_bplus = _eq_type.isin(_b_plus_types).astype(int)

# --- Source 2: Form D detected B-range filing after Series A ---
_formd_b = (df.get("seriesB_source", pd.Series([""] * len(df))).fillna("") == "formd").astype(int)

# --- Source 3: Venture Unknown with step-up or >= $7M, gap >= 6 months ---
# VU = Crunchbase knows they raised equity but doesn't know which series.
# We count it as B if the amount suggests progression beyond A.
_is_vu = _eq_type.str.lower().str.contains("venture", na=False) & ~_eq_type.isin(_b_plus_types)
_lfa_col = col_last_amount if col_last_amount else "Last Funding Amount (in USD)"
_last_eq_amt = pd.to_numeric(df.get("Last Equity Funding Amount (in USD)"), errors="coerce").fillna(0)
_sa_usd = pd.to_numeric(df.get("seriesA_amount_usd_num"), errors="coerce").fillna(0)
_vu_stepup = _last_eq_amt > _sa_usd  # raised more than their Series A
_vu_big = _last_eq_amt >= 7_000_000   # or in typical B range
_vu_positive = (_is_vu & (_months_from_a >= 6) & (_vu_stepup | _vu_big)).astype(int)

# --- Source 4: Equity growth — Total Equity > 2x Series A, gap >= 6 months ---
# Catches companies where Crunchbase still says "Series A" or "Venture Unknown"
# but total equity proves they raised significant follow-on.
_total_equity = pd.to_numeric(df.get("Total Equity Funding Amount (in USD)"), errors="coerce").fillna(0)
_sa_usd_m = pd.to_numeric(df.get("seriesA_amount_usd_millions"), errors="coerce").fillna(0) * 1e6
_eq_growth = (
    (_sa_usd_m > 0) &
    (_total_equity > 2.0 * _sa_usd_m) &
    (_months_from_a >= 6) &
    ~_cb_bplus.astype(bool) &
    ~_formd_b.astype(bool) &
    ~_vu_positive.astype(bool)
).astype(int)

# --- COMBINED: any of the 4 sources → positive ---
df[TARGET_36M_COL] = ((_cb_bplus == 1) | (_formd_b == 1) | (_vu_positive == 1) | (_eq_growth == 1)).astype(int)

# Breakdown
_n_cb = int(_cb_bplus.sum())
_n_formd_new = int(((_formd_b == 1) & (_cb_bplus == 0)).sum())
_n_vu_new = int(((_vu_positive == 1) & (_cb_bplus == 0) & (_formd_b == 0)).sum())
_n_eq_new = int(_eq_growth.sum())
print(f"Label ({TARGET_36M_COL}): {df[TARGET_36M_COL].sum()} success = "
      f"CB_Equity_B+({_n_cb}) + FormD(+{_n_formd_new}) + VU_smart(+{_n_vu_new}) + Equity_growth(+{_n_eq_new}) "
      f"= {df[TARGET_36M_COL].mean()*100:.1f}%")

# Keep raised_B_plus for backwards compatibility
_series_b_plus_strict = df[TARGET_COL].astype(bool)
_strong_followon = (_sa_usd_m > 0) & (_total_equity >= 2 * _sa_usd_m)
df["raised_B_plus"] = (_series_b_plus_strict | _strong_followon).astype(int)
print(f"Label (legacy) raised_B_plus: {int(df['raised_B_plus'].sum())} ({df['raised_B_plus'].mean()*100:.1f}%)")

# --------------- 3c) Montant Series A : une seule colonne en millions USD ---------------
if "seriesA_amount_usd_num" in df.columns:
    df["seriesA_amount_usd_millions"] = (pd.to_numeric(df["seriesA_amount_usd_num"], errors="coerce").fillna(0) / 1e6).round(2)
if "seriesA_amount_usd_millions" not in df.columns and "seriesA_amount_usd_num" in df.columns:
    df["seriesA_amount_usd_millions"] = (pd.to_numeric(df["seriesA_amount_usd_num"], errors="coerce").fillna(0) / 1e6).round(2)
# Cap at $200M: anything above is almost certainly not a real Series A
# (e.g. Perch at $775M = total funding mislabeled as Series A in Crunchbase).
# US median Series A was $8.9M (2017-2022, Crunchbase); p99 ≈ $150M.
SERIES_A_CAP_MILLIONS = 200
if "seriesA_amount_usd_millions" in df.columns:
    n_capped = int((df["seriesA_amount_usd_millions"] > SERIES_A_CAP_MILLIONS).sum())
    df["seriesA_amount_usd_millions"] = df["seriesA_amount_usd_millions"].clip(upper=SERIES_A_CAP_MILLIONS)
    if n_capped > 0:
        print(f"seriesA_amount capped at ${SERIES_A_CAP_MILLIONS}M: {n_capped} companies clipped")
if "seriesA_amount_usd_num" in df.columns:
    df = df.drop(columns=["seriesA_amount_usd_num"], errors="ignore")


# --------------- 3d2) Derived features (session April 12, 2026) ---------------
# 5 features derived from existing data, zero API calls:
# 1. desc_word_count: proxy for company maturity (longer = more established).
#    Description text is mostly set at company creation, low leakage risk.
# 2. visibility_score: counts how many sparse external signals are nonzero
#    (patents, YC, HN, clinical trials, federal awards, GitHub, etc.)
#    Score 0-3. Companies with score 3 have 47% Series B rate vs 25% at score 0.
# 3. name_word_count: 1-word vs 2+ word names
# 4. name_has_tech_suffix: name ends in AI, Labs, Tech, etc.
# 5. is_diverse_founded: Crunchbase Diversity Spotlight flag

# 1. Description word count
_desc = df.get("Description", pd.Series([""] * len(df))).fillna("").astype(str)
_full_desc = df.get("Full Description", pd.Series([""] * len(df))).fillna("").astype(str)
df["desc_word_count"] = (_desc + " " + _full_desc).apply(lambda x: len(x.split()))
print(f"desc_word_count: mean={df['desc_word_count'].mean():.0f} words")

# 2. Visibility score (count of active sparse external signals, capped at 3)
_ext_signal_cols = [
    "has_patent_preA_company", "yc_alumni_preA", "hn_mention_count_preA",
    "ct_trials_count_preA", "openalex_works_count",
    "fed_grants_count_preA", "fed_contracts_count_preA",
    "github_before_seriesA", "pkg_releases_count_preA",
    "wikipedia_before_seriesA", "backlinks_github_count_preA",
]
_n_active = sum(
    (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0).astype(int)
    for c in _ext_signal_cols if c in df.columns
)
df["visibility_score"] = _n_active.clip(upper=3).astype(int)
print(f"visibility_score: {df['visibility_score'].value_counts().sort_index().to_dict()}")

# 3. Name word count — strip common suffixes BEFORE counting
#    "XYZ AI" → strip "AI" → "XYZ" → 1 word
#    "Happiest Baby" → no suffix → 2 words
#    "Atom Computing" → no suffix → 2 words (Computing is the real name)
_NAME_SUFFIXES_RE = re.compile(
    r"\b(inc|llc|ltd|corp|co|company|gmbh|plc|"     # legal
    r"ai|labs?|tech|technologies|systems|digital|"    # generic tech
    r"software|platform|app|studio|studios|"          # generic product
    r"network|networks|cloud|data|"                   # generic infra
    r"solutions|services|group|ventures|partners)\b\.?",  # generic business
    re.IGNORECASE,
)
_names = df.get("Organization_Name", pd.Series([""] * len(df))).fillna("").astype(str)
def _core_word_count(name):
    """Count meaningful words after stripping generic suffixes."""
    stripped = _NAME_SUFFIXES_RE.sub("", name).strip()
    stripped = re.sub(r"\s+", " ", stripped).strip()
    words = [w for w in stripped.split() if len(w) >= 2]  # drop single-char remnants
    return max(1, len(words))  # minimum 1 (never report 0 words)
df["name_word_count"] = _names.apply(_core_word_count)
print(f"name_word_count: {df['name_word_count'].value_counts().sort_index().to_dict()}")

# 4. Name has tech suffix (same regex, binary flag)
df["name_has_tech_suffix"] = _names.str.lower().str.contains(
    r"\b(ai|labs?|tech|technologies|systems|robotics|bio|therapeutics|health)\b",
    regex=True,
).astype(int)
print(f"name_has_tech_suffix: {int(df['name_has_tech_suffix'].sum())}/{len(df)}")

# 5. Diversity spotlight
if "Diversity Spotlight" in df.columns:
    df["is_diverse_founded"] = df["Diversity Spotlight"].fillna("").astype(str).str.contains(
        "Women|Diverse", case=False, regex=True
    ).astype(int)
else:
    df["is_diverse_founded"] = 0
print(f"is_diverse_founded: {int(df['is_diverse_founded'].sum())}/{len(df)}")

# 6. Acceleration features: 3m/12m ratio captures momentum right before the raise
#    High ratio = traffic/press is accelerating. Low = flat or declining.
_wb_12m = pd.to_numeric(df.get("wayback_snapshots_12m_preA"), errors="coerce").fillna(0)
_wb_3m = pd.to_numeric(df.get("wayback_snapshots_3m_preA"), errors="coerce").fillna(0)
# 3m is 25% of 12m window, so ratio > 0.25 means accelerating
# Use 3m / (12m + 1) to avoid division by zero, multiply by 4 to normalize (so 1.0 = flat)
df["wayback_acceleration"] = np.where(_wb_12m > 0, (_wb_3m * 4) / _wb_12m, 0.0).round(3)
print(f"wayback_acceleration: non-zero={int((df['wayback_acceleration'] > 0).sum())}/{len(df)}, mean={df['wayback_acceleration'].mean():.2f}")

_gdelt_12m = pd.to_numeric(df.get("gdelt_mentions_12m_preA"), errors="coerce").fillna(0)
_gdelt_3m = pd.to_numeric(df.get("gdelt_mentions_3m_preA"), errors="coerce").fillna(0)
df["gdelt_acceleration"] = np.where(_gdelt_12m > 0, (_gdelt_3m * 4) / _gdelt_12m, 0.0).round(3)
print(f"gdelt_acceleration: non-zero={int((df['gdelt_acceleration'] > 0).sum())}/{len(df)}, mean={df['gdelt_acceleration'].mean():.2f}")

# 6b. Technical footprint: combines github + packages + academic papers into one binary
_gh = pd.to_numeric(df.get("github_before_seriesA"), errors="coerce").fillna(0) > 0
_pkg = pd.to_numeric(df.get("pkg_before_seriesA"), errors="coerce").fillna(0) > 0
_oa = pd.to_numeric(df.get("openalex_works_count"), errors="coerce").fillna(0) > 0
df["has_technical_footprint"] = (_gh | _pkg | _oa).astype(int)
print(f"has_technical_footprint: {int(df['has_technical_footprint'].sum())}/{len(df)}")

# 7. Log-transform heavy-tailed counts
#    Compresses outliers: log(1 + 488 patents) = 6.2, log(1 + 1 patent) = 0.7
#    Helps the model distinguish "none vs some" from "some vs lots"
for _raw, _log_name in [
    ("patent_count_preA_company", "log_patent_count"),
    ("hn_mention_count_preA", "log_hn_mentions"),
    ("wayback_snapshots_3m_preA", "log_wayback_3m"),
    ("edgar_filings_preA", "log_edgar_filings"),
    ("gdelt_mentions_3m_preA", "log_gdelt_3m"),
]:
    if _raw in df.columns:
        df[_log_name] = np.log1p(pd.to_numeric(df[_raw], errors="coerce").fillna(0))
print(f"Log-transformed: patent, hn, wayback, edgar, gdelt")

# 8. Interaction: directors × patent (two strongest signals combined)
_dirs = pd.to_numeric(df.get("formd_directors_preA"), errors="coerce").fillna(0)
_has_pat = pd.to_numeric(df.get("has_patent_preA_company"), errors="coerce").fillna(0)
df["directors_x_patent"] = (_dirs > 0).astype(int) * _has_pat.astype(int)
print(f"directors_x_patent: {int(df['directors_x_patent'].sum())}/{len(df)} have BOTH")

# 9. Investor quality: directors per round
_rounds = pd.to_numeric(df.get("formd_prior_rounds"), errors="coerce").fillna(0)
df["directors_per_round"] = np.where(_rounds > 0, _dirs / _rounds, 0.0).round(2)
print(f"directors_per_round: non-zero={int((df['directors_per_round'] > 0).sum())}/{len(df)}, mean={df['directors_per_round'].mean():.1f}")

# --------------- 3d) Advisor-suggested metadata features ---------------
# Two binary features suggested by advisor (meeting 2026-04-10):
#   - is_delaware_hq: HQ physically in Delaware (not just incorporated there)
#   - founder_name_in_company: the company name contains one of the founders' first/last names
# Both computed from Crunchbase columns only — no external API calls.

def _is_delaware_hq(hq):
    """Return 1 if HQ location contains 'Delaware' (case-insensitive). 0 otherwise."""
    if pd.isna(hq):
        return 0
    return int("delaware" in str(hq).lower())


def _founder_name_in_company(company_name, founders):
    """Return 1 if any founder's first or last name (≥4 chars) appears as a
    word-boundary match inside the company name. Used to test the hypothesis
    that companies named after their founder tend to underperform."""
    if not company_name or not founders:
        return 0
    if pd.isna(company_name) or pd.isna(founders):
        return 0
    name = str(company_name).strip().lower()
    founders_str = str(founders).strip()
    if not founders_str or founders_str.lower() == "nan":
        return 0
    # Strip common legal/industry suffixes from the company name before matching
    clean_name = re.sub(
        r'\b(inc|llc|ltd|corp|co|company|technologies|tech|labs|health|bio|ai|io|app)\b\.?',
        '', name, flags=re.IGNORECASE,
    )
    clean_name = re.sub(r'[^a-z]+', ' ', clean_name).strip()
    # For each founder, extract name tokens and check for word-boundary match
    for founder in founders_str.split(','):
        f = founder.strip().lower()
        if not f or f == 'nan':
            continue
        parts = re.sub(r'[^a-z ]', ' ', f).split()
        for token in parts:
            # Require ≥4 chars to avoid matching short/common words
            if len(token) < 4:
                continue
            if token in ('dr', 'mr', 'mrs', 'ms', 'prof', 'jr', 'sr', 'the', 'and', 'for',
                         'with', 'from', 'inc', 'llc', 'ltd', 'corp', 'phd', 'mba', 'ceo', 'cto'):
                continue
            if re.search(r'\b' + re.escape(token) + r'\b', clean_name):
                return 1
    return 0


if "Headquarters Location" in df.columns:
    df["is_delaware_hq"] = df["Headquarters Location"].apply(_is_delaware_hq).astype(int)
else:
    df["is_delaware_hq"] = 0
print(f"is_delaware_hq: {int(df['is_delaware_hq'].sum())}/{len(df)} companies HQ in Delaware")

_company_col = "Organization_Name" if "Organization_Name" in df.columns else "Organization Name"
_founders_col = "Founders" if "Founders" in df.columns else None
if _company_col in df.columns and _founders_col in df.columns:
    df["founder_name_in_company"] = df.apply(
        lambda r: _founder_name_in_company(r.get(_company_col), r.get(_founders_col)),
        axis=1,
    ).astype(int)
else:
    df["founder_name_in_company"] = 0
print(f"founder_name_in_company: {int(df['founder_name_in_company'].sum())}/{len(df)} companies named after a founder")


# --------------- 4) Build feature set (pre–Series A only) ---------------
all_features = []
for c in FEATURE_NUMERIC + FEATURE_CATEGORICAL:
    if c in df.columns:
        all_features.append(c)
for c in FEATURE_OPTIONAL:
    if c in df.columns and c not in all_features:
        all_features.append(c)

meta_available = [c for c in META_COLS if c in df.columns]
EXCLUDE_FROM_OUTPUT = ["run_id", "domain_clean", "seriesA_amount_usd_num", "seriesB_amount_usd_num", "sec_founders_overlap_count", "sec_founders_overlap_ratio", "sec_events_found", "founder_context_len", "founder_text_coverage_ratio", "seriesA_confidence"]
all_features = [c for c in all_features if c not in EXCLUDE_FROM_OUTPUT]
SERIESB_OUT_COLS = ["seriesB_date", "seriesB_amount_usd", "seriesB_source", "seriesB_comment"]  # seriesB_amount_usd_num excluded
seriesB_avail = [c for c in SERIESB_OUT_COLS if c in df.columns]
TEXT_NOTE_COLS = ("wikipedia_excerpt", "news_mentions_query_used", "wikipedia_note", "news_mentions_note", "backlinks_signal_note", "github_note", "pkg_note", "hn_note", "ph_note", "openalex_signal_note", "nlp_note")
# Schema A→O: A = Identity+timing (index table), B = label (SeriesB_within_36m), C→O = feature groups (ordre exact)
ORDER_A = [
    "Organization_Name", "Website", "Founded Date", "seriesA_date", SERIESA_YEAR_COL, "seriesA_quarter",
    TARGET_36M_COL, "raised_B_plus", MONTHS_TO_B_COL, "seriesB_date", "seriesB_amount_usd", "seriesB_source", "seriesB_comment",
    TARGET_COL, SUCCESS_REASON_COL, SUCCESS_REASON_CONFIDENCE_COL, SPLIT_COL,
]
ORDER_A_FULL = ["Organization_Name", "Website", "run_id", "Founded Date", "seriesA_date", "seriesA_year", "seriesA_quarter",
    "SeriesB_within_36m", "months_to_seriesB", "seriesB_date", "seriesB_amount_usd", "seriesB_source", "seriesB_comment",
    "success_seriesB_plus", "success_seriesB_plus_reason", "success_seriesB_plus_reason_confidence", "split"]
ORDER_C = ["seriesA_amount_usd_millions", "industry_bucket", "deeptech_llm_score", "startup_sector", "age_at_seriesA_years", "age_at_seriesA_flag", "Number of Founders",
           # Advisor-suggested metadata features (2026-04-10 meeting)
           "is_delaware_hq", "founder_name_in_company"]
ORDER_D = ["formd_directors_preA", "formd_investor_count_preA", "formd_prior_rounds"]
ORDER_E = ["has_patent_preA_company", "patent_count_preA_company", "patent_earliest_date_preA", "patent_matched_assignees", "patent_note"]
ORDER_F = [
    "founder_features_available",
    # Wayback LLM-extracted founder signals with citation verification
    "founder_school_tier", "founder_phd_flag", "founder_bigtech_flag",
    "founder_serial_entrepreneur_flag", "founder_prior_exit_flag",
]
ORDER_G = []  # SEMrush traffic features removed
ORDER_H = []  # Google Trends removed
ORDER_I = ["wikipedia_before_seriesA"]
ORDER_J = ["news_mentions_count", "news_sources_count", "news_mentions_query_used", "news_avg_tone", "news_velocity_3m", "news_lexical_diversity", "news_global_reach", "news_tone_slope", "news_cooccurrence_vc", "news_goldstein_scale", "news_mentions_note"]
ORDER_K = ["backlinks_hn_count_preA", "backlinks_hn_before_seriesA", "backlinks_reddit_count_preA", "backlinks_github_count_preA", "backlinks_approx_domains_preA", "backlinks_signal_note"]
ORDER_L = ["github_before_seriesA", "backlinks_github_count_preA", "github_note"]
ORDER_M = ["pkg_registry", "pkg_first_release_date", "pkg_releases_count_preA", "pkg_signal_date", "pkg_before_seriesA", "pkg_major_updates_preA", "pkg_note"]
ORDER_N = ["openalex_has_papers_preA", "openalex_works_count", "openalex_signal_note"]
ORDER_O = [
    # Hacker News (04e) — leaky upvote/comment columns removed, kept mention_count + launch binary
    "hn_mention_count_preA", "hn_launch", "hn_note",
    # Product Hunt (04n) — upvote/comment columns removed (API returns current totals, leaky)
    "ph_launch", "ph_launch_date", "ph_before_seriesA", "ph_featured",
    "ph_launches_count_preA", "ph_note",
]
ORDER_P = [
    # Wayback CDX counts
    "wayback_first_snapshot_date", "wayback_snapshots_12m_preA", "wayback_snapshots_3m_preA",
    # Wayback CDX binary (careers page exists pre-A)
    "has_careers_page_preA",
    # Wayback LLM-extracted traction signals (citation-verified)
    "named_customers_count_preA", "product_stage_preA",
]
# New source groups (added during 2026-04-10 audit)
ORDER_Q_CLINICAL = [
    "ct_trials_count_preA", "ct_before_seriesA", "ct_min_phase_preA", "ct_max_phase_preA", "ct_note",
]
ORDER_R_FED = [
    "fed_grants_count_preA", "fed_contracts_count_preA", "fed_awards_before_seriesA", "fed_awards_note",
]
ORDER_S_YC = [
    "yc_alumni_preA", "yc_batch", "yc_months_before_seriesA", "yc_note",
]
# New sources + derived features (session 2026-04-12)
ORDER_T_EDGAR = [
    "edgar_filings_preA", "edgar_has_10k_preA", "edgar_note",
]
ORDER_U_GDELT = [
    "gdelt_mentions_12m_preA", "gdelt_mentions_3m_preA", "gdelt_has_preA_press", "gdelt_note",
]
ORDER_V_DERIVED = [
    "desc_word_count", "visibility_score", "name_word_count", "name_has_tech_suffix",
    "is_diverse_founded", "wayback_acceleration", "gdelt_acceleration", "has_technical_footprint",
    "log_patent_count", "log_hn_mentions", "log_wayback_3m", "log_edgar_filings", "log_gdelt_3m",
    "directors_x_patent", "directors_per_round",
]
# Ordre final strict : A puis C→V (une seule liste, pas de remaining_features)
NOTES_EXTRA = [c for c in TEXT_NOTE_COLS if c not in (ORDER_I + ORDER_J + ORDER_K + ORDER_L + ORDER_M + ORDER_N + ORDER_O + ORDER_P + ORDER_Q_CLINICAL + ORDER_R_FED + ORDER_S_YC + ORDER_T_EDGAR + ORDER_U_GDELT)]
FULL_ORDER = list(ORDER_A) + ORDER_C + ORDER_D + ORDER_E + ORDER_F + ORDER_G + ORDER_H + ORDER_I + ORDER_J + ORDER_K + ORDER_L + ORDER_M + ORDER_N + ORDER_O + ORDER_P + ORDER_Q_CLINICAL + ORDER_R_FED + ORDER_S_YC + ORDER_T_EDGAR + ORDER_U_GDELT + ORDER_V_DERIVED + list(NOTES_EXTRA)
out_cols = [c for c in FULL_ORDER if c in df.columns and c not in EXCLUDE_FROM_OUTPUT]
out_cols = list(dict.fromkeys(out_cols))

df_out = df[out_cols].copy()

# --------------- 4b) Dédupliquer : une ligne par (Organization_Name, seriesA_date) ---------------
dup_cols = [c for c in ["Organization_Name", "seriesA_date"] if c in df_out.columns]
if dup_cols:
    before = len(df_out)
    df_out = df_out.drop_duplicates(subset=dup_cols, keep="first").reset_index(drop=True)
    if len(df_out) < before:
        print(f"Dedup: {before - len(df_out)} doublons supprimés -> {len(df_out)} lignes")

# --------------- 5) Impute missing numerics ---------------
# Strategy:
#   - "absence" columns (binary flags, counts, scores where NaN = signal not found): fill with 0
#   - "continuous" columns (page_length, etc.): fill with median
#   - all-NaN columns (signal never fired): fill with 0 (not median, which would be NaN → warning)
#
# Columns that mean "no signal detected" → impute 0 (not median)
ZERO_IMPUTE_PREFIXES = (
    "has_", "patent_count", "wikipedia_before",
    "backlinks_", "hn_comments", "github_before",
    "pkg_releases_count", "pkg_before", "pkg_registry_count", "pkg_update",
    "pkg_version", "pkg_major", "pkg_maintainer", "pkg_ecosystem",
    "openalex_works_count", "openalex_has_papers",
    "hn_launch", "hn_is_show", "hn_mention_count", "hn_before",
    "ph_launch", "ph_featured", "ph_before", "ph_launches",
    "founder_",
    "wayback_", "nlp_",
    "fed_grants", "fed_contracts", "fed_awards",  # USAspending federal awards (04h)
    "yc_alumni", "yc_months",  # YC: both binary and months → 0 if not YC
    "ct_trials", "ct_before", "ct_min_phase", "ct_max_phase",  # Clinical trials → 0 if none
    "edgar_filings", "edgar_has",  # EDGAR → 0 if no filings
    "gdelt_mentions", "gdelt_has",  # GDELT → 0 if no mentions
    "formd_directors", "formd_investor", "formd_prior",  # Form D → 0 if no data
    "wayback_acceleration", "gdelt_acceleration",  # derived ratios → 0 if no data
    "log_patent", "log_hn", "log_wayback", "log_edgar", "log_gdelt",  # log-transformed
    "directors_x_patent", "directors_per_round",
)
import warnings
numeric_features = [c for c in FEATURE_NUMERIC + FEATURE_OPTIONAL if c in df_out.columns and c in all_features and c not in TEXT_NOTE_COLS and c not in LEAKAGE_RISK_COLS]
for c in numeric_features:
    try:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
        if not df_out[c].isna().any():
            continue
        use_zero = any(c.startswith(p) for p in ZERO_IMPUTE_PREFIXES) or df_out[c].notna().sum() == 0
        if use_zero:
            df_out[c] = df_out[c].fillna(0)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = df_out[c].median()
            df_out[c] = df_out[c].fillna(0 if pd.isna(med) else med)
    except Exception:
        pass

# --------------- 6) Categorical: fill missing with "MISSING" ---------------
for c in FEATURE_CATEGORICAL:
    if c in df_out.columns:
        df_out[c] = df_out[c].astype(str).replace("nan", "MISSING").replace("", "MISSING")

print("Feature columns (for 06):", all_features)
print("Output shape:", df_out.shape)


# In[80]:


# --------------- 7) Save full dataset + train / test ---------------
df_out["run_id"] = RUN_ID
order_cols = [c for c in ORDER_A_FULL if c in df_out.columns] + [c for c in df_out.columns if c not in ORDER_A_FULL]
df_out = df_out[order_cols]
df_out.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)

train_df = df_out[df_out[SPLIT_COL] == "train"].copy()
test_df = df_out[df_out[SPLIT_COL] == "test"].copy()
train_df.to_csv(OUTPUT_TRAIN, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)
print("Saved train:", OUTPUT_TRAIN, "->", len(train_df), "rows")
print("Saved test:", OUTPUT_TEST, "->", len(test_df), "rows")

print("\n--- Summary for 06 ---")
print(f"Target (06): {PRIMARY_TARGET_COL} (1 = Series B+ within 36m)")
print(f"Split: {SPLIT_COL} (train = seriesA_year < {SPLIT_YEAR})")
print(f"Features (pre–Series A): {len(all_features)} cols")
print(f"Train success rate ({PRIMARY_TARGET_COL}): {train_df[PRIMARY_TARGET_COL].mean():.2%}")
print(f"Test success rate ({PRIMARY_TARGET_COL}):  {test_df[PRIMARY_TARGET_COL].mean():.2%}")


# In[81]:


# --------------- 8) Export feature list for 06 (model step) ---------------
import json
FEATURE_LIST_PATH = PIPE / f"{RUN_ID}_05_feature_columns.json"
config_06 = {
    "target": PRIMARY_TARGET_COL,
    "target_audit": TARGET_COL,
    "success_reason_col": SUCCESS_REASON_COL,
    "success_reason_confidence_col": SUCCESS_REASON_CONFIDENCE_COL,
    "target_36m": TARGET_36M_COL,
    "months_to_seriesB": MONTHS_TO_B_COL,
    "split_col": SPLIT_COL,
    "seriesA_year_col": SERIESA_YEAR_COL,
    "split_year": SPLIT_YEAR,
    "feature_columns": all_features,
    "meta_columns": [c for c in META_COLS if c in df_out.columns],
}
with open(FEATURE_LIST_PATH, "w") as f:
    json.dump(config_06, f, indent=2)
print("Config for 06:", FEATURE_LIST_PATH)

# --------------- 9) Archive : tout directement dans archive/YYYYMMDD/ (détail de chaque étape) ---------------
import shutil

ARCHIVE_ROOT = PIPE / "archive" / RUN_ID  # YYYYMMDD
ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

# Fichiers détaillés (chaque étape) directement dans archive/YYYYMMDD/
for src in PIPE.glob(f"{RUN_ID}_*"):
    if src.is_file():
        shutil.copy2(src, ARCHIVE_ROOT / src.name)

# --------------- 10) Pipeline/archive/Final dataset/ = stack (N, N-1), train, test, config ---------------
FINAL_DATASET_DIR = PIPE / "archive" / "Final dataset"
FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
STACKED_CSV = FINAL_DATASET_DIR / "final_dataset_stacked.csv"       # N (actuel)
STACKED_PREVIOUS_CSV = FINAL_DATASET_DIR / "final_dataset_stacked_previous.csv"  # N-1
# Migration : si l'ancien stack est encore dans archive/ (sans sous-dossier), le lire une fois
_legacy_stack = PIPE / "archive" / "final_dataset_stacked.csv"
if not STACKED_CSV.exists() and _legacy_stack.exists():
    shutil.copy2(_legacy_stack, STACKED_CSV)

df_out["run_id"] = RUN_ID
order_cols = [c for c in ORDER_A_FULL if c in df_out.columns] + [c for c in df_out.columns if c not in ORDER_A_FULL]
df_out = df_out[order_cols]
n_prev_rows = None
if STACKED_CSV.exists():
    # Sauvegarder l'actuel (N) en N-1 avant de le modifier (ancien N-1 est écrasé)
    shutil.copy2(STACKED_CSV, STACKED_PREVIOUS_CSV)
    # Dédupliquer N-1 (au cas où l'ancien N contenait des doublons)
    df_prev = pd.read_csv(STACKED_PREVIOUS_CSV)
    dup_cols_prev = [c for c in ["Organization_Name", "seriesA_date"] if c in df_prev.columns]
    if dup_cols_prev:
        df_prev = df_prev.drop_duplicates(subset=dup_cols_prev, keep="last").reset_index(drop=True)
        df_prev.to_csv(STACKED_PREVIOUS_CSV, index=False)
    n_prev_rows = len(df_prev)
    df_stacked = pd.read_csv(STACKED_CSV)
    df_stacked = pd.concat([df_stacked, df_out], ignore_index=True)
else:
    df_stacked = df_out.copy()
# Dédupliquer le stack : (Organization_Name, seriesA_date) unique, garder la dernière version
dup_cols_stack = [c for c in ["Organization_Name", "seriesA_date"] if c in df_stacked.columns]
if dup_cols_stack:
    df_stacked = df_stacked.drop_duplicates(subset=dup_cols_stack, keep="last").reset_index(drop=True)
# Exclure du stack les rows avec seriesA_confidence = LOW (anciennes runs ou stack historique)
if "seriesA_confidence_y" in df_stacked.columns:
    df_stacked["seriesA_confidence"] = df_stacked["seriesA_confidence_y"].fillna(df_stacked.get("seriesA_confidence_x", pd.Series(dtype=object)))
    df_stacked = df_stacked.drop(columns=["seriesA_confidence_x", "seriesA_confidence_y"], errors="ignore")
elif "seriesA_confidence_x" in df_stacked.columns:
    df_stacked["seriesA_confidence"] = df_stacked["seriesA_confidence_x"]
    df_stacked = df_stacked.drop(columns=["seriesA_confidence_x"], errors="ignore")
if "seriesA_confidence" in df_stacked.columns:
    before_low = len(df_stacked)
    df_stacked = df_stacked[df_stacked["seriesA_confidence"].astype(str).str.strip().str.upper() != "LOW"].copy()
    if len(df_stacked) < before_low:
        print(f"Dropped {before_low - len(df_stacked)} rows with seriesA_confidence = LOW from stack")
# Remove columns that must not appear in final output (stack may have them from legacy rows)
cols_drop_stack = [c for c in EXCLUDE_FROM_OUTPUT if c != "run_id" and c in df_stacked.columns]
if cols_drop_stack:
    df_stacked = df_stacked.drop(columns=cols_drop_stack, errors="ignore")
# Réordonner le stack : A (avec run_id) puis C→O (même ordre que final dataset)
stack_order = [c for c in ORDER_A_FULL if c in df_stacked.columns] + [c for c in FULL_ORDER if c in df_stacked.columns and c not in ORDER_A_FULL] + [c for c in df_stacked.columns if c not in ORDER_A_FULL and c not in FULL_ORDER]
df_stacked = df_stacked[stack_order]
df_stacked.to_csv(STACKED_CSV, index=False)
print(f"\n--- Stack (dataset qui grandit à chaque run) ---")
print(f"N (actuel): {STACKED_CSV} | total rows: {len(df_stacked)} (+ {len(df_out)} ce run)")
print(f"N-1 (version précédente): {STACKED_PREVIOUS_CSV}" + (f" | {n_prev_rows} rows" if n_prev_rows is not None else " (créé au prochain run)"))

# --------------- 11) Préparer train / test pour les analyses ---------------
# Use THIS RUN's data (not stacked) — stacking dilutes features from different signal versions
FINAL_TRAIN_CSV = FINAL_DATASET_DIR / "final_train.csv"
FINAL_TEST_CSV = FINAL_DATASET_DIR / "final_test.csv"
train_run = df_out[df_out[SPLIT_COL] == "train"].copy()
test_run = df_out[df_out[SPLIT_COL] == "test"].copy()
train_run.to_csv(FINAL_TRAIN_CSV, index=False)
test_run.to_csv(FINAL_TEST_CSV, index=False)
CONFIG_ARCHIVE = FINAL_DATASET_DIR / "final_feature_columns.json"
with open(CONFIG_ARCHIVE, "w") as f:
    json.dump(config_06, f, indent=2)
print(f"Train (pour 06): {FINAL_TRAIN_CSV} | rows: {len(train_run)}")
print(f"Test (pour 06):  {FINAL_TEST_CSV} | rows: {len(test_run)}")
print(f"Config 06: {CONFIG_ARCHIVE}")
print(f"(Using this run only — not stacked — to preserve feature coverage)")

# --------------- 12) Vider le Pipeline (après archivage) ---------------
# Supprimer tous les fichiers de ce run dans PIPE (glob, pour ne rien oublier)
removed = 0
for p in list(PIPE.glob(f"{RUN_ID}_*")):
    if p.is_file():
        p.unlink()
        removed += 1
print(f"\n--- Pipeline vidé ---")
print(f"Fichiers supprimés de {PIPE}: {removed}")

print("\n--- Archive ---")
print("archive/YYYYMMDD/ (détail de chaque étape):", ARCHIVE_ROOT)
print("archive/Final dataset/ (stack N, N-1, train, test, config 06):", FINAL_DATASET_DIR)


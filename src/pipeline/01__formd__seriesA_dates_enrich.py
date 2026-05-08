#!/usr/bin/env python
# coding: utf-8

# In[2]:


# sec_formd_features_v2.py
# ------------------------------------------------------------
# Match startups -> SEC Form D issuers (CIK), build events from
# ACCESSIONNUMBER join, dedupe events (prefer non-amendments),
# infer Series A candidate via Equity clusters, compute pre-A
# features + confidence + revenue ranges.
# ------------------------------------------------------------

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

import re
from pathlib import Path

# Use the Pipeline folder that contains .run_id (written by 07) so output is where 02 will look
PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline", Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
PIPE = None
for p in PIPE_CANDIDATES:
    if p.exists() and (p / ".run_id").exists():
        PIPE = p
        break
if PIPE is None:
    PIPE = PIPE_CANDIDATES[0] if PIPE_CANDIDATES[0].exists() else (Path.home() / "Desktop" / "Data_thesis" / "Pipeline")
BASE = PIPE.parent
CRUNCH = BASE / "Crunchbase_data"
FORMD  = BASE / "Form D SEC"
ARCHIVE = PIPE / "archive"

PIPE.mkdir(parents=True, exist_ok=True)
ARCHIVE.mkdir(parents=True, exist_ok=True)

SQLITE_DB = FORMD / "formd_all.sqlite"
SEP = ";"  # Crunchbase export chez toi

def run_id_from_filename(name: str) -> str:
    m = re.match(r"^([^_]+)_", name)
    if not m:
        raise ValueError(f"Bad filename (expected RUNID_...): {name}")
    return m.group(1)

def pick_next_file_for_step(pipe_dir: Path, prev_step: str, next_step_prefix: str) -> Path:
    """
    prev_step: string that must be in filename, ex: "_01_crunchbase__raw_export"
    next_step_prefix: prefix that indicates next step exists, ex: "_02_formd__"
    Rule: choose the newest RUN_ID where prev exists and next doesn't.
    """
    candidates = []
    for p in pipe_dir.glob(f"*{prev_step}*.csv"):
        run_id = p.name.split("_", 1)[0]
        # next exists?
        next_exists = any(pipe_dir.glob(f"{run_id}{next_step_prefix}*.csv"))
        if not next_exists:
            candidates.append((run_id, p))

    if not candidates:
        raise FileNotFoundError(f"No runnable input found for {prev_step} (all already processed?)")

    # newest date wins (YYYYMMDD sort works)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def pick_next_crunchbase_export(crunch_dir: Path, pipe_dir: Path, allow_overwrite: bool = False) -> Path:
    """allow_overwrite=True : re-run et écraser la sortie step 02 (modifications)."""
    exports = sorted(crunch_dir.glob("*_01_crunchbase__raw_export.csv"), key=lambda p: p.name, reverse=True)
    if not exports:
        raise FileNotFoundError("No Crunchbase raw export found in Crunchbase_data/")

    for p in exports:
        run_id = run_id_from_filename(p.name)
        already = any(pipe_dir.glob(f"{run_id}_02_formd__*.csv"))
        if allow_overwrite or not already:
            return p

    raise FileNotFoundError("All Crunchbase exports already processed to step 02.")

# ----------------------------
# Interactive file selection (or auto-pick via PIPELINE_RUN_ID env var)
# ----------------------------
ALLOW_OVERWRITE = True

def _interactive_select(crunch_dir: Path) -> Path:
    """List all Crunchbase exports and let the user pick one."""
    exports = sorted(crunch_dir.glob("*_01_crunchbase__raw_export.csv"), key=lambda p: p.name)
    exports = [e for e in exports if "(1)" not in e.name]  # skip duplicates
    if not exports:
        raise FileNotFoundError("No Crunchbase raw export found in Crunchbase_data/")

    print("\n" + "=" * 70)
    print("  Available Crunchbase exports:")
    print("=" * 70)
    for idx, p in enumerate(exports, 1):
        rid = run_id_from_filename(p.name)
        try:
            n_rows = sum(1 for _ in open(p)) - 1  # fast line count minus header
        except Exception:
            n_rows = "?"
        already = "✓ enriched" if any(PIPE.glob(f"{rid}_02_formd__*.csv")) or any((PIPE / "archive").glob(f"*/{rid}_02_formd__*.csv")) else ""
        print(f"  [{idx:>2}]  {p.name:<55}  ({n_rows} rows)  {already}")

    print(f"  [ 0]  Auto-select (newest unprocessed)")
    print("=" * 70)

    while True:
        try:
            choice = input("Select export [0 = auto]: ").strip()
            if choice == "" or choice == "0":
                return pick_next_crunchbase_export(crunch_dir, PIPE, allow_overwrite=ALLOW_OVERWRITE)
            num = int(choice)
            if 1 <= num <= len(exports):
                return exports[num - 1]
            print(f"  Enter a number between 0 and {len(exports)}")
        except (ValueError, EOFError):
            print("  Enter a valid number")

CHOSEN_RUN_ID = os.environ.get("PIPELINE_RUN_ID", "").strip() or (
    (PIPE / ".run_id").read_text().strip() if (PIPE / ".run_id").exists() else ""
)

# If run_pipeline.sh set PIPELINE_RUN_ID, skip interactive selection
if os.environ.get("PIPELINE_RUN_ID", "").strip():
    chosen_path = CRUNCH / f"{CHOSEN_RUN_ID}_01_crunchbase__raw_export.csv"
    if chosen_path.exists():
        INPUT_CSV = chosen_path
    else:
        INPUT_CSV = pick_next_crunchbase_export(CRUNCH, PIPE, allow_overwrite=ALLOW_OVERWRITE)
else:
    # Interactive mode: show all exports and let user pick
    INPUT_CSV = _interactive_select(CRUNCH)

RUN_ID = run_id_from_filename(INPUT_CSV.name)
OUTPUT_CSV = PIPE / f"{RUN_ID}_02_formd__seriesA_dates_enriched.csv"

# Update .run_id so downstream steps use the same run
(PIPE / ".run_id").write_text(RUN_ID)

# ============================================================
# Matching helpers
# ============================================================

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "l.l.c", "ltd", "limited",
    "corp", "corporation", "co", "company",
    "plc", "gmbh", "sarl", "sas", "sa", "bv",
    "holdings", "holding", "group", "partners", "partner",
    "technologies", "technology", "systems", "system",
}

def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def company_tokens(name: str) -> List[str]:
    t = norm_text(name)
    toks = [x for x in t.split() if len(x) >= 2]
    toks = [x for x in toks if x not in LEGAL_SUFFIXES]
    return toks

def simple_similarity(a: str, b: str) -> float:
    ta = set(company_tokens(a))
    tb = set(company_tokens(b))
    if not ta or not tb:
        return 0.0
    if ta == tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / max(1, union)

def is_ambiguous_name(name: str) -> bool:
    # 1 token => très ambigu (“Nova”, “Apex”…)
    return len(company_tokens(name)) <= 1

def norm_postal(s: str) -> str:
    s = (s or "").strip()
    m = re.match(r"([A-Za-z0-9]+)", s.replace(" ", ""))
    return m.group(1) if m else ""

def fmt_usd(x: float) -> str:
    try:
        v = float(x)
        if v <= 0:
            return ""
        if v >= 1_000_000_000:
            return f"${v/1_000_000_000:.2f}B"
        if v >= 1_000_000:
            return f"${v/1_000_000:.2f}M"
        if v >= 1_000:
            return f"${v/1_000:.0f}k"
        return f"${v:.0f}"
    except Exception:
        return ""

# ============================================================
# Parsers
# ============================================================

def parse_any_date(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s[:19], fmt)
        except Exception:
            pass

    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", "")
        if not s or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None

def coalesce_amount(sold: Any, offered: Any) -> Optional[float]:
    a = safe_float(sold)
    if a is not None and a > 0:
        return a
    b = safe_float(offered)
    if b is not None and b > 0:
        return b
    return None


# ============================================================
# Business helpers
# ============================================================

def industry_bucket(industries_raw: str) -> str:
    t = norm_text(industries_raw)
    if any(k in t for k in ["biotech", "biotechnology", "genetics", "life science", "medical", "pharma", "therapeutics"]):
        return "BIO"
    if any(k in t for k in ["aerospace", "industrial", "machinery", "manufacturing", "robotics", "battery", "energy", "semiconductor", "sensor", "cleantech"]):
        return "HARD"
    if any(k in t for k in ["fintech", "financial", "wealth management", "crypto", "cryptocurrency", "blockchain"]):
        return "FIN"
    if any(k in t for k in ["cyber", "telecommunications", "quantum", "infrastructure"]):
        return "INFRA"
    return "SAAS"

def split_hq_location(x: str) -> Tuple[str, str]:
    s = str(x or "").strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    city, state = "", ""
    if len(parts) >= 2:
        city, state = parts[0], parts[1]
    elif len(parts) == 1:
        city = parts[0]
    return city, state


def fill_seriesA_from_crunchbase(row: Any) -> Dict[str, Any]:
    """
    When Crunchbase Last Funding Type or Last Equity Funding Type = Series A,
    fill seriesA_date (from Last Funding Date) and amount (Last Funding/Equity Amount).
    No CIK/legal name from Crunchbase; Patents step uses Organization_Name only.
    """
    def get(row: Any, key: str, default: str = "") -> str:
        try:
            v = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
            return str(v).strip() if v is not None and not (isinstance(v, float) and pd.isna(v)) else default
        except Exception:
            return default

    last_type = get(row, "Last Funding Type", "")
    last_equity_type = get(row, "Last Equity Funding Type", "")
    if "series a" not in last_type.lower() and "series a" not in last_equity_type.lower():
        return {}

    out: Dict[str, Any] = {
        "seriesA_source": "crunchbase",
        "seriesA_date": "",
        "seriesA_quarter": "",
        "seriesA_amount_usd": "",
        "seriesA_amount_usd_num": "",
        "seriesA_security_type": "Equity",
        "seriesA_comment": "crunchbase_seriesA",
        "seriesA_confidence": "LOW",
        "founded_date": "",
        "age_at_seriesA_years": "",
        "age_at_seriesA_flag": "UNKNOWN",
    }

    series_dt = parse_any_date(get(row, "Last Funding Date", ""))
    if not series_dt:
        return out

    amt = safe_float(row.get("Last Equity Funding Amount (in USD)")) if hasattr(row, "get") else None
    if amt is None or amt <= 0:
        amt = safe_float(row.get("Last Funding Amount (in USD)")) if hasattr(row, "get") else None
    if amt is None:
        amt = 0.0

    out["seriesA_confidence"] = "MED" if amt > 0 else "LOW"
    out["seriesA_date"] = series_dt.date().isoformat()
    out["seriesA_quarter"] = f"{series_dt.year}Q{((series_dt.month - 1) // 3) + 1}"
    out["seriesA_amount_usd_num"] = amt
    out["seriesA_amount_usd"] = fmt_usd(amt)

    founded_dt = parse_any_date(get(row, "Founded Date", ""))
    if founded_dt:
        out["founded_date"] = founded_dt.date().isoformat()
        try:
            d0 = founded_dt.date()
            d1 = series_dt.date()
            age_years = (d1 - d0).days / 365.25
            out["age_at_seriesA_years"] = round(age_years, 2)
            out["age_at_seriesA_flag"] = "SUSPECT" if age_years >= 6 else "OK"
        except Exception:
            pass
    return out

# --------------- Series B from Crunchbase (priority: if Last Funding Type = Series B+ with amount+date, use it first) ---------------
SERIES_B_PLUS_KEYWORDS = ["series b", "series c", "series d", "series e", "series f", "series g", "ipo", "late stage", "growth", "private equity", "venture - series unknown"]

def fill_seriesB_from_crunchbase(row: Any) -> Dict[str, Any]:
    """
    When Crunchbase Last Funding Type or Last Equity = Series B (or B+: C, D, IPO, Late Stage)
    and we have Last Funding Date + amount, fill seriesB_date and seriesB_amount.
    Use this first to save time; Form D is used only when Crunchbase does not provide B+ with amount+date.
    """
    def get(row: Any, key: str, default: str = "") -> str:
        try:
            v = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
            return str(v).strip() if v is not None and not (isinstance(v, float) and pd.isna(v)) else default
        except Exception:
            return default

    last_type = get(row, "Last Funding Type", "").lower()
    last_equity = get(row, "Last Equity Funding Type", "").lower()
    if not any(k in last_type or k in last_equity for k in SERIES_B_PLUS_KEYWORDS):
        return {}

    series_dt = parse_any_date(get(row, "Last Funding Date", ""))
    if not series_dt:
        return {}

    amt = safe_float(row.get("Last Equity Funding Amount (in USD)")) if hasattr(row, "get") else None
    if amt is None or amt <= 0:
        amt = safe_float(row.get("Last Funding Amount (in USD)")) if hasattr(row, "get") else None
    if amt is None or amt <= 0:
        return {}

    return {
        "seriesB_date": series_dt.date().isoformat(),
        "seriesB_amount_usd": fmt_usd(amt),
        "seriesB_amount_usd_num": amt,
        "seriesB_source": "crunchbase",
        "seriesB_comment": "crunchbase_seriesB_plus",
    }

def invalidate_seriesB_if_before_A(r: Dict[str, Any]) -> None:
    """If seriesB_date < seriesA_date, the Series B date is an error (e.g. wrong round in Crunchbase). Clear seriesB_* and flag."""
    if not r.get("seriesA_date") or not r.get("seriesB_date"):
        return
    a_dt = parse_any_date(r["seriesA_date"])
    b_dt = parse_any_date(r["seriesB_date"])
    if a_dt and b_dt and b_dt < a_dt:
        r["seriesB_date"] = ""
        r["seriesB_amount_usd"] = ""
        r["seriesB_amount_usd_num"] = ""
        r["seriesB_source"] = ""
        r["seriesB_comment"] = "error_B_before_A"

def truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}

def classify_security_from_row(row: pd.Series) -> str:
    raw = str(row.get("security_raw", "") or "").strip()
    if raw:
        t = norm_text(raw)
        if any(k in t for k in ["debt", "note", "bond", "loan", "promissory", "debenture"]):
            return "Debt"
        if any(k in t for k in ["convertible", "safe", "simple agreement", "warrant", "option"]):
            return "Convertible"
        if any(k in t for k in ["equity", "common", "preferred", "stock", "shares", "membership interest"]):
            return "Equity"
        return "Other"

    if truthy(row.get("ISEQUITYTYPE")):
        return "Equity"
    if truthy(row.get("ISDEBTTYPE")):
        return "Debt"
    if truthy(row.get("ISOPTIONTOACQUIRETYPE")):
        return "Convertible"
    if truthy(row.get("ISOTHERSECURITYTYPE")):
        return "Other"
    return "Unknown"

def amount_confidence(row: pd.Series) -> str:
    sold = safe_float(row.get("amount_sold"))
    offered = safe_float(row.get("amount_offered"))
    if sold is not None and sold > 0:
        return "HIGH"
    if offered is not None and offered > 0:
        return "MED"
    return "LOW"


# ============================================================
# Series A config (ranges)
# ============================================================

@dataclass
class SeriesAConfig:
    target_low: float
    target_high: float
    mega_cap: float

# Median US Series A round size (empirical, from Crunchbase data 2017-2022).
# Previous value was $14M (2022 peak) which was 57% too high for the full dataset.
# Our dataset: overall median = $8.9M; by sector: SAAS $7.7M, FIN $9.1M,
# HARD $8.8M, BIO $13.9M, INFRA $7.5M. We use $9M as the baseline and derive
# sector multipliers from the actual ratios (BIO ~1.6×, FIN ~1.0×, etc.)
# Source: Crunchbase US median Series A was $8.6M (2018-2020), $14M (2022 peak).
# Year-specific Series A medians (from Crunchbase data)
# Series A sizes grew significantly: $7M in 2017 to $12.5M in 2022
MEDIAN_SERIES_A_BY_YEAR = {
    2015: 5_000_000,
    2016: 7_000_000,
    2017: 7_000_000,
    2018: 7_500_000,
    2019: 7_000_000,
    2020: 8_000_000,
    2021: 12_000_000,
    2022: 12_500_000,
}
BASE_MEDIAN_SERIES_A_USD = 9_000_000  # fallback for unknown years

# Sector multipliers: derived from actual dataset medians relative to overall.
# BIO ($13.9M / $8.9M = 1.56), FIN ($9.1M / $8.9M = 1.02), HARD ($8.8M / $8.9M = 0.99),
# INFRA ($7.5M / $8.9M = 0.84). Rounded for stability.
BUCKET_MULT = {
    "SAAS":  1.00,   # $9M median → range $2.25M – $13.5M
    "FIN":   1.00,   # $9.1M actual ≈ same as overall
    "HARD":  1.00,   # $8.8M actual ≈ same as overall
    "BIO":   1.55,   # $13.9M actual → range $3.5M – $20.9M
    "INFRA": 0.85,   # $7.5M actual → range $1.9M – $11.5M
}

def cfg_from_median(median_usd: float) -> SeriesAConfig:
    return SeriesAConfig(
        target_low=0.25 * median_usd,
        target_high=1.5 * median_usd,
        mega_cap=5.0 * median_usd,
    )

# ✅ IMPORTANT: defined (tu l’avais perdu)
SERIES_A_CFG = {
    b: cfg_from_median(BASE_MEDIAN_SERIES_A_USD * m)
    for b, m in BUCKET_MULT.items()
}

# --------------- Series B config (median US + ranges, for Form D fallback) ---------------
@dataclass
class SeriesBConfig:
    target_low: float
    target_high: float
    mega_cap: float

# Median US Series B: Crunchbase reports ~$25-30M (2020-2022).
# Previous value $35M was slightly high. Using $27M as a compromise.
# Note: Series B identification from Form D is secondary to Crunchbase labeling
# and only affects the seriesB_date/amount columns, not the primary target label.
BASE_MEDIAN_SERIES_B_USD = 27_000_000

def cfg_b_from_median(median_usd: float) -> SeriesBConfig:
    return SeriesBConfig(
        target_low=0.25 * median_usd,
        target_high=1.5 * median_usd,
        mega_cap=5.0 * median_usd,
    )

SERIES_B_CFG = {
    b: cfg_b_from_median(BASE_MEDIAN_SERIES_B_USD * m)
    for b, m in BUCKET_MULT.items()
}

SERIES_B_MIN_USD = 10_000_000   # below -> likely not Series B (could be A or bridge)
SERIES_B_LATE_STAGE_USD = 200_000_000  # above -> late stage, not typical B


# ============================================================
# Series A via Equity clusters
# ============================================================

SERIES_A_CLUSTER_WINDOW_DAYS = 120  # 90-180 typical

def _conf_rank(x: str) -> int:
    return {"LOW": 0, "MED": 1, "HIGH": 2}.get(str(x or "").upper(), 0)

def build_equity_clusters(events: pd.DataFrame, window_days: int = SERIES_A_CLUSTER_WINDOW_DAYS) -> List[pd.DataFrame]:
    """
    Clusters NON-chaînés:
    - on regroupe les equity events tant qu'ils restent dans window_days
      par rapport au DEBUT du cluster (pas par rapport à l'event précédent).
    => empêche les clusters multi-années.
    """
    if events.empty:
        return []

    ev = events.copy()
    ev = ev[ev["event_date"].notna()].copy()
    ev = ev[ev["security_type_class"].isin(["Equity", "Convertible", "Unknown", ])].copy()
    ev = ev[ev["amount_usd"].fillna(0) > 0].copy()
    ev = ev.sort_values("event_date")

    if ev.empty:
        return []

    clusters: List[pd.DataFrame] = []
    cur_idx: List[int] = []
    cluster_start: Optional[datetime] = None

    for idx, r in ev.iterrows():
        dt = r["event_date"]
        if cluster_start is None:
            cluster_start = dt
            cur_idx = [idx]
            continue

        # ✅ compare to cluster_start, not last_dt
        if (dt - cluster_start).days <= int(window_days):
            cur_idx.append(idx)
        else:
            clusters.append(ev.loc[cur_idx].copy())
            cluster_start = dt
            cur_idx = [idx]

    if cur_idx:
        clusters.append(ev.loc[cur_idx].copy())

    return clusters

# ============================================================
# Series A selection v2 (smart filters + scoring)
# Paste this block in your file (replace pick_series_a_cluster + compute_features_from_events call)
# ============================================================

import math
from typing import Optional, Tuple, List

# --- tweakable thresholds ---
SEED_MAX_USD = 2_000_000                # keep at $2M (year-aware filtering handles the rest)
EARLY_YEARS_MAX = 1.0                   # "too close to founding"
EARLY_AMOUNT_MAX_USD = 10_000_000       # if <10M and <1y, likely seed/early
LATE_STAGE_MIN_USD = 100_000_000        # huge rounds are usually NOT Series A
LATE_STAGE_MIN_AGE_YEARS = 5.0          # if company is old + huge -> late stage
MAX_PLAUSIBLE_A_AGE_YEARS = 12.0        # after that, very unlikely a "true A"

# bucket-based "typical" A range multipliers around your median
A_LOW_MULT  = 0.25
A_HIGH_MULT = 2.50  # raised from 1.5 to 2.5: captures larger Series A rounds ($20M+)

def _age_years(founded_dt: Optional[datetime], round_dt: datetime) -> Optional[float]:
    if not founded_dt:
        return None
    return (round_dt.date() - founded_dt.date()).days / 365.25

def _gauss_score(x: float, mu: float, sigma: float) -> float:
    # smooth score in [0,1], peak at mu
    return math.exp(-0.5 * ((x - mu) / max(1e-9, sigma)) ** 2)

def _amount_score(amount: float, bucket: str, filing_year: int = 0) -> float:
    cfg = SERIES_A_CFG.get(bucket, SERIES_A_CFG["SAAS"])
    # Use year-specific median if available, else fallback
    base_median = MEDIAN_SERIES_A_BY_YEAR.get(filing_year, BASE_MEDIAN_SERIES_A_USD)
    lo = A_LOW_MULT * (base_median * BUCKET_MULT.get(bucket, 1.0))
    hi = A_HIGH_MULT * (base_median * BUCKET_MULT.get(bucket, 1.0))
    mid = (lo + hi) / 2

    # hard reject if absurd
    if amount <= 0:
        return 0.0
    if amount < SEED_MAX_USD:
        return 0.0
    if amount > max(LATE_STAGE_MIN_USD, cfg.mega_cap):
        # not impossible, but very unlikely for Series A
        return 0.05

    # score peaks near middle of target band
    # sigma ~ band width / 2 (soft)
    sigma = (hi - lo) / 2 if hi > lo else mid * 0.5
    return float(_gauss_score(amount, mid, sigma))

def _age_score(age_years: Optional[float]) -> float:
    # if no founded date => neutral-ish (don’t kill)
    if age_years is None:
        return 0.55
    # peak around 3.5 years, sigma 2 years
    s = float(_gauss_score(age_years, mu=3.5, sigma=2.0))
    # hard downweight extreme ages
    if age_years < 0.5:
        s *= 0.2
    if age_years > MAX_PLAUSIBLE_A_AGE_YEARS:
        s *= 0.05
    return s

def _cluster_meta(c: pd.DataFrame, founded_dt: Optional[datetime]) -> dict:
    c = c.sort_values("event_date")
    dt = c["event_date"].min()
    amt = float(c["amount_usd"].fillna(0).sum())
    age = _age_years(founded_dt, dt) if dt else None
    return {"dt": dt, "amt": amt, "age": age, "n": int(c.shape[0])}

def pick_series_a_cluster_v2(
    events: pd.DataFrame,
    bucket: str,
    founded_dt: Optional[datetime],
) -> Tuple[Optional[pd.DataFrame], str]:

    if events is None or events.empty:
        return None, "no_events"

    clusters = build_equity_clusters(events, window_days=SERIES_A_CLUSTER_WINDOW_DAYS)
    if not clusters:
        return None, "no_equity_clusters"

    cfg = SERIES_A_CFG.get(bucket, SERIES_A_CFG["SAAS"])

    metas: List[Tuple[pd.DataFrame, dict]] = []
    for c in clusters:
        m = _cluster_meta(c, founded_dt)  # expects {"dt","amt","age"}
        if m.get("dt") is not None and float(m.get("amt") or 0) > 0:
            metas.append((c, m))

    if not metas:
        return None, "no_valid_clusters"

    # -------------------------
    # 1) filter obvious NON-A (year-aware)
    # -------------------------
    kept: List[Tuple[pd.DataFrame, dict]] = []
    for c, m in metas:
        amt = float(m["amt"])
        age = m.get("age")  # years or None
        filing_yr = m["dt"].year if m.get("dt") else 0
        yr_median = MEDIAN_SERIES_A_BY_YEAR.get(filing_yr, BASE_MEDIAN_SERIES_A_USD) * BUCKET_MULT.get(bucket, 1.0)
        yr_seed_max = max(SEED_MAX_USD, 0.40 * yr_median)  # year-aware seed floor

        # seed-like: reject if below year-aware seed threshold AND early (< 2 years)
        if amt < yr_seed_max:
            if age is None or age < 2.0:
                continue

        # late-stage-like
        if (amt >= LATE_STAGE_MIN_USD) and (age is not None) and (age >= LATE_STAGE_MIN_AGE_YEARS):
            continue

        # too old for A
        if (age is not None) and (age > MAX_PLAUSIBLE_A_AGE_YEARS):
            continue

        kept.append((c, m))

    if not kept:
        return None, "filtered_all_clusters_seed_or_late_stage"

    # sort by time
    kept_by_time = sorted(kept, key=lambda x: x[1]["dt"])

    # -------------------------
    # 2) Combined scoring: amount_in_range + age + step-up ratio
    # Series A is the first BIG raise: significant step-up from previous round,
    # amount in sector-specific range, at typical company age (2-4 years)
    # -------------------------

    # Compute step-up ratio for each cluster (vs previous cluster by date)
    for i, (c, m) in enumerate(kept_by_time):
        if i == 0:
            m["step_up"] = 1.0  # first cluster, no comparison
        else:
            prev_amt = float(kept_by_time[i - 1][1]["amt"])
            cur_amt = float(m["amt"])
            m["step_up"] = cur_amt / max(prev_amt, 1.0)

    def _combined_score(cm):
        c, m = cm
        amt = float(m["amt"])
        age = m.get("age")
        step_up = m.get("step_up", 1.0)

        # Amount score: is it in the sector-specific, year-specific Series A range?
        filing_yr = m["dt"].year if m.get("dt") else 0
        a_s = _amount_score(amt, bucket, filing_year=filing_yr)

        # Age score: peak at 2-4 years, penalize extremes
        age_factor = 1.0
        if age is not None:
            if age < 1.5:
                age_factor = 0.4
            elif age < 2.0:
                age_factor = 0.7
            elif age <= 4.0:
                age_factor = 1.0
            elif age <= 6.0:
                age_factor = 0.8
            else:
                age_factor = 0.5

        # Step-up bonus: a 3x+ jump from previous round strongly suggests Series A
        step_factor = 1.0
        if step_up >= 3.0:
            step_factor = 1.5   # big jump = likely the graduation to A
        elif step_up >= 2.0:
            step_factor = 1.2   # moderate jump
        elif step_up < 0.5:
            step_factor = 0.6   # amount went DOWN, unusual for A

        return a_s * age_factor * step_factor

    scored = [((_combined_score((c, m)), c, m)) for c, m in kept_by_time]

    # When scores are close (within 30%), prefer the LATER filing
    # Earlier filings are often seeds that score well but aren't the real A
    scored.sort(key=lambda x: x[0], reverse=True)
    best_s, best_c, best_m = scored[0]

    if len(scored) > 1:
        # When our best pick is early (age < 2 years), strongly consider later filings
        # Series A is almost never the first/earliest filing
        best_age = best_m.get("age")
        best_is_early = best_age is not None and best_age < 2.0

        for s, c, m in scored[1:]:
            age = m.get("age")
            is_later = m["dt"] > best_m["dt"]
            is_typical_age = age is not None and 1.5 <= age <= 6.0

            if not is_later or not is_typical_age:
                continue

            # If our best pick is early (<2y), accept any later filing with >40% of best score
            # If our best pick is at typical age, only accept if >70% of best score
            threshold = 0.40 if best_is_early else 0.70
            if s >= best_s * threshold:
                best_s, best_c, best_m = s, c, m
                break

    age_str = f"|age={best_m['age']:.1f}" if best_m.get('age') is not None else ""
    step_str = f"|step_up={best_m.get('step_up', 1.0):.1f}x"

    return best_c, (
        f"picked_by_combined_score|score={best_s:.2f}"
        f"|amt={fmt_usd(best_m['amt'])}"
        f"|dt={best_m['dt'].date().isoformat()}"
        f"{age_str}{step_str}"
    )

# ============================================================
# Series B from Form D (events after Series A, amount in B range)
# ============================================================

def pick_series_b_from_formd(
    events: pd.DataFrame,
    seriesA_date_str: str,
    bucket: str,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Find a probable Series B round in Form D events: events AFTER seriesA_date
    with amount in Series B range (median US + bucket). Returns earliest in range.
    """
    if events is None or events.empty:
        return None, "no_events"
    seriesA_dt = parse_any_date(seriesA_date_str)
    if not seriesA_dt:
        return None, "no_seriesA_date"

    ev = events.copy()
    ev = ev[ev["event_date"].notna()].copy()
    ev = ev[ev["event_date"] > seriesA_dt].copy()
    ev = ev[ev["security_type_class"].isin(["Equity", "Convertible", "Unknown"])].copy()
    ev = ev[ev["amount_usd"].fillna(0) > 0].copy()
    ev = ev.sort_values("event_date")

    if ev.empty:
        return None, "no_events_after_seriesA"

    cfg = SERIES_B_CFG.get(bucket, SERIES_B_CFG["SAAS"])
    mult = BUCKET_MULT.get(bucket, 1.0)
    base_med = BASE_MEDIAN_SERIES_B_USD * mult
    target_low = max(SERIES_B_MIN_USD, 0.25 * base_med)
    target_high = min(SERIES_B_LATE_STAGE_USD, 1.5 * base_med)

    # Build clusters (same window as A) for events after A
    clusters = build_equity_clusters(ev, window_days=SERIES_A_CLUSTER_WINDOW_DAYS)
    if not clusters:
        return None, "no_clusters_after_seriesA"

    candidates: List[Tuple[datetime, float, pd.DataFrame]] = []
    for c in clusters:
        c = c.sort_values("event_date")
        dt = c["event_date"].min()
        amt = float(c["amount_usd"].fillna(0).sum())
        if amt < target_low or amt > target_high:
            continue
        candidates.append((dt, amt, c))

    if not candidates:
        return None, "no_cluster_in_seriesB_range"

    candidates.sort(key=lambda x: x[0])
    best_dt, best_amt, best_c = candidates[0]
    return best_c, (
        f"formd_seriesB|amt={fmt_usd(best_amt)}|dt={best_dt.date().isoformat()}"
    )

# ============================================================
# Update compute_features_from_events: use v2 picker
# ============================================================

def compute_features_from_events(
    events: pd.DataFrame,
    bucket: str,
    match_score: float,
    founded_dt: Optional[datetime],
) -> Dict[str, Any]:

    out: Dict[str, Any] = {
        "seriesA_date": "",
        "seriesA_quarter": "",
        "seriesA_amount_usd": "",
        "seriesA_amount_usd_num": "",
        "seriesA_security_type": "",
        "seriesA_comment": "",
        "seriesA_confidence": "",

        "founded_date": founded_dt.date().isoformat() if founded_dt else "",
        "age_at_seriesA_years": "",
        "age_at_seriesA_flag": "UNKNOWN",
    }

    if events is None or events.empty:
        out["seriesA_comment"] = "no_events"
        out["seriesA_confidence"] = "LOW"
        return out

    cluster, comment = pick_series_a_cluster_v2(events, bucket, founded_dt)
    if cluster is None or cluster.empty:
        out["seriesA_comment"] = comment
        out["seriesA_confidence"] = "LOW"
        return out

    cluster = cluster.sort_values("event_date")
    series_amt = float(cluster["amount_usd"].fillna(0).sum())
    series_dt = cluster["event_date"].min()

    out["seriesA_date"] = series_dt.date().isoformat()
    out["seriesA_quarter"] = f"{series_dt.year}Q{((series_dt.month - 1)//3) + 1}"
    out["seriesA_amount_usd_num"] = series_amt
    out["seriesA_amount_usd"] = fmt_usd(series_amt)
    out["seriesA_security_type"] = "Equity"
    out["seriesA_comment"] = comment

    # confidence: based on entity match quality, not amount
    # A perfect name+founder match is MED even if amount is small
    out["seriesA_confidence"] = "MED" if match_score >= 0.50 else "LOW"

    # founded sanity
    if founded_dt:
        age_years = _age_years(founded_dt, series_dt)
        out["age_at_seriesA_years"] = round(age_years, 2) if age_years is not None else ""
        out["age_at_seriesA_flag"] = "SUSPECT" if (age_years is not None and age_years >= 6) else "OK"
        if out["age_at_seriesA_flag"] == "SUSPECT":
            out["seriesA_confidence"] = "LOW"
            out["seriesA_comment"] += "|age_suspect>=6y"

    return out

def seriesA_confidence(score, series_comment, series_amt, series_sec, amt_conf, num_investors):
    amt = float(series_amt or 0)

    if series_sec != "Equity":
        return "LOW"

    # only worst case forces LOW
    if "very_low_confidence" in series_comment:
        return "LOW"

    # HIGH: strong match + sold amount
    if score > 0.50 and amt_conf == "HIGH" and amt > 0:
        return "HIGH"

    # MED: your rule
    if score >= 0.50 and amt > 0:
        return "MED"

    return "LOW"

# ============================================================
# SQLite schema helpers
# ============================================================

def connect_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def first_existing(col_candidates: List[str], existing_cols: List[str]) -> Optional[str]:
    existing_map = {c.lower(): c for c in existing_cols}
    for cand in col_candidates:
        hit = existing_map.get(cand.lower())
        if hit:
            return hit
    return None

def show_schema(conn: sqlite3.Connection):
    print("\n🔎 DB schema quick view:")
    for t in ["issuers", "formdsubmission", "offering"]:
        try:
            cols = table_columns(conn, t)
            print(f" - {t}: {len(cols)} cols | sample={cols[:12]}")
        except Exception as e:
            print(f" - {t}: ERROR -> {e}")

# ============================================================
# Founders matching (RELATEDPERSONS)
# ============================================================

def candidate_entity_names_str(conn, company, city, state, postal, top_k=4) -> str:
    cand = get_candidate_issuers(conn, company, city, state, postal, limit=80)
    if cand is None or cand.empty:
        return ""

    # garde uniquement les noms (évite les doublons)
    names = (
        cand["_issuer_name"]
        .astype(str)
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )
    names = names[:top_k]
    return " / ".join(names)

def debug_candidates_for_startup(conn, startup_row, top_k=8) -> str:
    company = str(startup_row.get("Organization_Name", "") or "").strip()
    city = str(startup_row.get("HQ_City", "") or "").strip()
    state = str(startup_row.get("HQ_State", "") or "").strip()
    postal = str(startup_row.get("Postal_Code", "") or "").strip()
    founders_raw = str(startup_row.get("Founders", "") or "").strip()

    cand = get_candidate_issuers(conn, company, city, state, postal, limit=80)
    if cand is None or cand.empty:
        return "no_candidates"

    # Keep only CIK candidates
    cand["_cik_clean"] = cand["_cik"].astype(str).str.strip()
    cand = cand[~cand["_cik_clean"].isin(["", "nan", "none"])].copy()
    if cand.empty:
        return "no_candidates_with_cik"

    # Compute founder last-name overlap for each CIK
    f_last = parse_founders_lastnames(founders_raw)
    rows = []
    for _, rr in cand.head(top_k).iterrows():
        cik = str(rr["_cik_clean"])
        issuer = str(rr["_issuer_name"])
        score = float(rr.get("match_score", 0.0))

        r_last = fetch_related_lastnames_for_cik(conn, cik)
        ov_cnt, ov_ratio = founder_overlap_score(f_last, r_last)

        rows.append(f"{issuer} | CIK={cik.zfill(10)} | score={score:.2f} | ov={ov_cnt}")

    return " || ".join(rows)

def parse_founders_lastnames(founders_raw: str) -> List[str]:
    # "Jacob Leverich, Mike Speiser" -> ["LEVERICH","SPEISER"]
    if not founders_raw or str(founders_raw).lower() in {"nan", "none"}:
        return []
    parts = [p.strip() for p in str(founders_raw).split(",") if p.strip()]
    out = []
    for p in parts:
        toks = [t for t in re.split(r"\s+", p) if t]
        if not toks:
            continue
        ln = toks[-1].strip().upper()
        ln = re.sub(r"[^A-Z\-']", "", ln)
        if len(ln) >= 2:
            out.append(ln)
    return sorted(list(set(out)))

_rp_sql_cache = {}  # cached SQL query + column resolution (avoids schema lookups per call)

def fetch_related_lastnames_for_cik(conn: sqlite3.Connection, cik: str) -> List[str]:
    # Use cached SQL to avoid 3 schema queries per call
    if "sql" not in _rp_sql_cache:
        tables = {r[0].lower(): r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        rp_table = tables.get("relatedpersons")
        if not rp_table:
            _rp_sql_cache["sql"] = None
            return []
        rp_cols = table_columns(conn, rp_table)
        last_col = first_existing(["LASTNAME", "lastName", "LAST_NAME"], rp_cols)
        acc_col  = first_existing(["ACCESSIONNUMBER", "accessionNumber"], rp_cols)
        iss_cols = table_columns(conn, "issuers")
        iss_cik_col = first_existing(["CIK", "cik"], iss_cols)
        iss_acc_col = first_existing(["ACCESSIONNUMBER", "accessionNumber"], iss_cols)
        if not (last_col and acc_col and iss_cik_col and iss_acc_col):
            _rp_sql_cache["sql"] = None
            return []
        _rp_sql_cache["sql"] = f"""
            SELECT DISTINCT rp.{last_col} AS lname
            FROM {rp_table} rp
            JOIN issuers u ON rp.{acc_col} = u.{iss_acc_col}
            WHERE u.{iss_cik_col} = ?
              AND rp.{last_col} IS NOT NULL
        """

    sql = _rp_sql_cache.get("sql")
    if sql is None:
        return []

    df = pd.read_sql_query(sql, conn, params=[str(cik).strip()])
    if df.empty:
        return []

    out = []
    for x in df["lname"].astype(str).tolist():
        ln = _clean_lastname(x)
        if len(ln) >= 2:
            out.append(ln)
    return sorted(list(set(out)))


# Suffixes that should be stripped from last names before comparison
_LASTNAME_SUFFIXES = re.compile(
    r'[,\s]+(JR\.?|SR\.?|II|III|IV|V|MD|PHD|DVM|DDS|MBA|CPA|ESQ\.?|PE|JD|CFA|'
    r'CFP|RN|DO|PHARMD|MSC|BSC|DPHIL|FACS|FRCS|MRCOG|MRCP)\b\.?',
    re.IGNORECASE,
)

def _clean_lastname(name: str) -> str:
    """Strip suffixes (Jr., PhD, MD, etc.) and non-alpha chars from a last name."""
    s = name.strip().upper()
    # Remove everything after a comma (catches "MURPHY, JR." and "SMITH, MD, PHD")
    if ',' in s:
        s = s.split(',')[0].strip()
    # Also remove known suffixes without commas (catches "MURPHY JR")
    s = _LASTNAME_SUFFIXES.sub('', s).strip()
    # Keep only letters, hyphens, apostrophes
    s = re.sub(r"[^A-Z\-']", "", s)
    return s


def founder_overlap_score(founder_lastnames: List[str], related_lastnames: List[str]) -> Tuple[int, float]:
    if not founder_lastnames:
        return 0, 0.0
    # Clean both sides before comparing
    f = set(_clean_lastname(n) for n in founder_lastnames)
    r = set(_clean_lastname(n) for n in related_lastnames)
    f.discard('')
    r.discard('')
    ov = len(f & r)
    return ov, ov / max(1, len(f))

# ============================================================
# In-memory issuer index (5000x faster than SQL LIKE)
# ============================================================
# Loaded once at startup, shared across all workers via fork.

_ISSUER_INDEX = None  # {"all_issuers": DataFrame, "token_index": {token: set(row_indices)}, "col_*": str}

def _build_issuer_index(conn: sqlite3.Connection) -> dict:
    """Load all 474K issuers into memory and build an inverted token index."""
    issuers_cols = table_columns(conn, "issuers")
    col_name   = first_existing(["ENTITYNAME","entityName","issuerName","name"], issuers_cols)
    col_city   = first_existing(["CITY","issuerCity","city"], issuers_cols)
    col_state  = first_existing(["STATEORCOUNTRY","issuerStateOrCountry","state"], issuers_cols)
    col_postal = first_existing(["ZIPCODE","issuerZipCode","zip","postalCode"], issuers_cols)
    col_cik    = first_existing(["CIK","cik","issuerCik"], issuers_cols)

    print("  Loading issuers into memory for fast name matching...")
    all_issuers = pd.read_sql(f"SELECT * FROM issuers", conn)
    all_issuers["_issuer_name"] = all_issuers[col_name].astype(str)
    all_issuers["_cik"] = all_issuers[col_cik].astype(str) if col_cik and col_cik in all_issuers.columns else ""
    all_issuers["_city"]   = all_issuers[col_city].astype(str) if col_city and col_city in all_issuers.columns else ""
    all_issuers["_state"]  = all_issuers[col_state].astype(str) if col_state and col_state in all_issuers.columns else ""
    all_issuers["_postal"] = all_issuers[col_postal].astype(str) if col_postal and col_postal in all_issuers.columns else ""

    # Build inverted index: token → set of row indices
    token_index = {}
    for i, name in enumerate(all_issuers["_issuer_name"].values):
        for tok in norm_text(str(name)).split():
            if len(tok) >= 3 and tok not in LEGAL_SUFFIXES:
                if tok not in token_index:
                    token_index[tok] = set()
                token_index[tok].add(i)

    print(f"  Loaded {len(all_issuers)} issuers, {len(token_index)} unique tokens")
    return {
        "all_issuers": all_issuers,
        "token_index": token_index,
        "col_name": col_name, "col_city": col_city,
        "col_state": col_state, "col_postal": col_postal, "col_cik": col_cik,
    }

def _ensure_issuer_index(conn: sqlite3.Connection):
    global _ISSUER_INDEX
    if _ISSUER_INDEX is None:
        _ISSUER_INDEX = _build_issuer_index(conn)
    return _ISSUER_INDEX


# ============================================================
# Matching issuers (CIK) — uses in-memory inverted index
# ============================================================

def get_candidate_issuers(
    conn: sqlite3.Connection,
    company_name: str,
    city: str,
    state: str,
    postal: str,
    limit: int = 200,
    top_k: int = 10,
) -> pd.DataFrame:
    idx = _ensure_issuer_index(conn)
    all_issuers = idx["all_issuers"]
    token_index = idx["token_index"]

    tokens = [t for t in norm_text(company_name).split() if len(t) >= 3 and t not in LEGAL_SUFFIXES][:4]
    if not tokens:
        return pd.DataFrame()

    # Intersect posting lists — instant lookup
    candidate_indices = None
    for tok in tokens:
        if tok in token_index:
            if candidate_indices is None:
                candidate_indices = token_index[tok].copy()
            else:
                candidate_indices &= token_index[tok]
        else:
            return pd.DataFrame()

    if not candidate_indices:
        return pd.DataFrame()

    indices = sorted(candidate_indices)[:limit]
    df = all_issuers.iloc[indices].copy()

    city_n = norm_text(city)
    state_n = norm_text(state)
    postal_n = norm_postal(postal)

    scores, zip_hits, city_hits, state_hits = [], [], [], []

    for _, rr in df.iterrows():
        base = simple_similarity(company_name, rr["_issuer_name"])

        zip_hit   = bool(postal_n and (postal_n == norm_postal(rr["_postal"])))
        city_hit  = bool(city_n and (city_n in norm_text(rr["_city"])))
        state_hit = bool(state_n and (state_n in norm_text(rr["_state"])))

        zip_hits.append(int(zip_hit))
        city_hits.append(int(city_hit))
        state_hits.append(int(state_hit))

        bonus = 0.0
        if zip_hit: bonus += 0.25
        if city_hit: bonus += 0.15
        if state_hit: bonus += 0.10

        final = base if base < 0.50 else min(1.0, base + bonus)
        if is_ambiguous_name(company_name) and not zip_hit:
            final = min(final, 0.50)

        scores.append(float(final))

    df["match_score"] = scores
    df["loc_zip_hit"] = zip_hits
    df["loc_city_hit"] = city_hits
    df["loc_state_hit"] = state_hits

    df = df.sort_values("match_score", ascending=False).head(top_k).copy()
    return df

_sbi_timings = {"get_candidates": [], "scoring_loop": [], "lastname_lookups": [], "n_candidates": [], "n_cache_miss": []}

def select_best_issuer(conn: sqlite3.Connection, startup_row: Dict[str, Any], rp_lastname_cache=None):
    import time as _t
    if rp_lastname_cache is None:
        rp_lastname_cache = {}

    company = str(startup_row.get("Organization_Name", "") or "").strip()
    city    = str(startup_row.get("HQ_City", "") or "").strip()
    state   = str(startup_row.get("HQ_State", "") or "").strip()
    postal  = str(startup_row.get("Postal_Code", "") or "").strip()
    founders_raw = str(startup_row.get("Founders", "") or "").strip()

    if not company:
        return None, "", 0.0, 0, 0.0

    _t0 = _t.time()
    cand = get_candidate_issuers(conn, company, city, state, postal, limit=200, top_k=10)
    _sbi_timings["get_candidates"].append(_t.time() - _t0)

    if cand is None or cand.empty:
        _sbi_timings["n_candidates"].append(0)
        return None, "", 0.0, 0, 0.0

    cand["_cik_clean"] = cand["_cik"].astype(str).str.strip()
    cand = cand[~cand["_cik_clean"].isin(["", "nan", "none"])].copy()
    if cand.empty:
        _sbi_timings["n_candidates"].append(0)
        return None, "", 0.0, 0, 0.0

    _sbi_timings["n_candidates"].append(len(cand))
    founder_ln = parse_founders_lastnames(founders_raw)

    passing = []
    _t_loop = _t.time()
    _t_lastname_total = 0
    _cache_misses = 0
    for _, row in cand.iterrows():
        cik = str(row["_cik_clean"])
        issuer_name = str(row["_issuer_name"])
        name_score = float(row.get("match_score", 0.0))

        rp_ln = rp_lastname_cache.get(cik)
        if rp_ln is None:
            _t_ln = _t.time()
            rp_ln = fetch_related_lastnames_for_cik(conn, cik)
            _t_lastname_total += _t.time() - _t_ln
            rp_lastname_cache[cik] = rp_ln
            _cache_misses += 1

        ov_cnt, ov_ratio = founder_overlap_score(founder_ln, rp_ln)

        if ov_cnt >= 1 and name_score >= 0.50:
            # Scenario 1: founder overlap + reasonable name match
            passing.append((ov_cnt, name_score, cik, issuer_name, ov_ratio))
        elif ov_cnt == 0 and name_score == 1.0 and (row.get("loc_city_hit", 0) == 1 or row.get("loc_zip_hit", 0) == 1):
            # Scenario 2: exact name match + city or ZIP, no founder overlap
            passing.append((0, name_score, cik, issuer_name, 0.0))

    _sbi_timings["scoring_loop"].append(_t.time() - _t_loop)
    _sbi_timings["lastname_lookups"].append(_t_lastname_total)
    _sbi_timings["n_cache_miss"].append(_cache_misses)

    if not passing:
        top = cand.iloc[0]
        return None, str(top["_issuer_name"]), float(top.get("match_score", 0.0)), 0, 0.0

    passing.sort(key=lambda x: (x[0], x[1]), reverse=True)
    ov_cnt, name_score, cik, issuer_name, ov_ratio = passing[0]
    return cik.zfill(10), issuer_name, float(name_score), int(ov_cnt), float(ov_ratio)

# ============================================================
# Fetch events by CIK using ACCESSIONNUMBER join
# + DEDUPE by accession (no groupby.apply warning)
# ============================================================

def fetch_events_for_cik(conn: sqlite3.Connection, cik: str) -> pd.DataFrame:
    """
    Fetch SEC Form D "events" for a given CIK by joining:
      issuers (CIK, ACCESSIONNUMBER)
        -> formdsubmission (FILING_DATE)
        -> offering (SALE_DATE, amounts, flags, investors, security type, amendments)

    Then we clean + normalize into one row per "round event":
      1) Build event_date = SALE_DATE if present else FILING_DATE
      2) Build amount_usd = coalesce(TOTALAMOUNTSOLD, TOTALOFFERINGAMOUNT)
      3) Classify security_type_class (Equity / Convertible / Debt / Other / Unknown)
      4) Dedupe duplicates created by SQL joins (one row per accession)
      5) Collapse amendments:
         - If a row is an amendment, PREVIOUSACCESSIONNUMBER points to the original
         - We treat amendment as an update, NOT additional money
         - So we keep MAX(amount) across the original+amend chain
    """

    # ---- detect schema columns (cached to avoid repeated PRAGMA calls) ----
    if not hasattr(fetch_events_for_cik, "_col_cache"):
        fetch_events_for_cik._col_cache = {
            "issuers": table_columns(conn, "issuers"),
            "formdsubmission": table_columns(conn, "formdsubmission"),
            "offering": table_columns(conn, "offering"),
        }
    iss_cols = fetch_events_for_cik._col_cache["issuers"]
    sub_cols = fetch_events_for_cik._col_cache["formdsubmission"]
    off_cols = fetch_events_for_cik._col_cache["offering"]

    iss_cik_col = first_existing(["CIK", "cik", "issuerCik"], iss_cols)
    iss_acc_col = first_existing(["ACCESSIONNUMBER", "accessionNumber"], iss_cols)

    sub_acc_col = first_existing(["ACCESSIONNUMBER", "accessionNumber"], sub_cols)
    filing_col = first_existing(["FILING_DATE", "filingDate"], sub_cols)

    off_acc_col = first_existing(["ACCESSIONNUMBER", "accessionNumber"], off_cols)
    sold_col = first_existing(["TOTALAMOUNTSOLD", "totalAmountSold", "amountSold"], off_cols)
    offered_col = first_existing(["TOTALOFFERINGAMOUNT", "totalOfferingAmount", "amountOffered"], off_cols)
    first_sale_col = first_existing(["SALE_DATE", "dateOfFirstSale", "firstSaleDate"], off_cols)

    inv_col = first_existing(["TOTALNUMBERALREADYINVESTED", "numberOfInvestors", "TOTALNUMBERINVESTORS"], off_cols)
    sec_raw_col = first_existing(["TYPESOFSECURITIESOFFERED", "securitiesOffered", "securityType"], off_cols)

    eq_flag = first_existing(["ISEQUITYTYPE"], off_cols)
    debt_flag = first_existing(["ISDEBTTYPE"], off_cols)
    opt_flag = first_existing(["ISOPTIONTOACQUIRETYPE"], off_cols)
    other_flag = first_existing(["ISOTHERSECURITYTYPE"], off_cols)

    rev_col = first_existing(["REVENUERANGE", "revenueRange"], off_cols)

    amend_flag = first_existing(["ISAMENDMENT"], off_cols)
    prev_acc_col = first_existing(["PREVIOUSACCESSIONNUMBER"], off_cols)

    if not (iss_cik_col and iss_acc_col and sub_acc_col and off_acc_col):
        raise ValueError(
            f"Missing required join columns. "
            f"issuers(cik={iss_cik_col}, acc={iss_acc_col}) "
            f"formdsubmission(acc={sub_acc_col}) offering(acc={off_acc_col})"
        )

    # ---- SQL join ----
    select_parts = [
        f"u.{iss_acc_col} AS accession",
        f"s.{filing_col} AS filing_date" if filing_col else "NULL AS filing_date",
        f"o.{first_sale_col} AS first_sale_date" if first_sale_col else "NULL AS first_sale_date",
        f"o.{sold_col} AS amount_sold" if sold_col else "NULL AS amount_sold",
        f"o.{offered_col} AS amount_offered" if offered_col else "NULL AS amount_offered",
        f"o.{inv_col} AS num_investors" if inv_col else "NULL AS num_investors",
        f"o.{sec_raw_col} AS security_raw" if sec_raw_col else "NULL AS security_raw",
        f"o.{eq_flag} AS ISEQUITYTYPE" if eq_flag else "NULL AS ISEQUITYTYPE",
        f"o.{debt_flag} AS ISDEBTTYPE" if debt_flag else "NULL AS ISDEBTTYPE",
        f"o.{opt_flag} AS ISOPTIONTOACQUIRETYPE" if opt_flag else "NULL AS ISOPTIONTOACQUIRETYPE",
        f"o.{other_flag} AS ISOTHERSECURITYTYPE" if other_flag else "NULL AS ISOTHERSECURITYTYPE",
        f"o.{amend_flag} AS ISAMENDMENT" if amend_flag else "NULL AS ISAMENDMENT",
        f"o.{prev_acc_col} AS PREVIOUSACCESSIONNUMBER" if prev_acc_col else "NULL AS PREVIOUSACCESSIONNUMBER",
        f"o.{rev_col} AS revenue_range" if rev_col else "NULL AS revenue_range",
    ]

    sql = f"""
    SELECT {", ".join(select_parts)}
    FROM issuers u
    LEFT JOIN formdsubmission s
      ON u.{iss_acc_col} = s.{sub_acc_col}
    LEFT JOIN offering o
      ON u.{iss_acc_col} = o.{off_acc_col}
    WHERE u.{iss_cik_col} = ?
    """

    df = pd.read_sql_query(sql, conn, params=[str(cik).strip()])
    if df.empty:
        return df

    # ---- normalize dates ----
    df["filing_dt"] = df["filing_date"].apply(parse_any_date)
    df["first_sale_dt"] = df["first_sale_date"].apply(parse_any_date)
    df["event_date"] = df["first_sale_dt"].where(df["first_sale_dt"].notna(), df["filing_dt"])

    # ---- normalize amounts / investors / types ----
    df["amount_usd"] = df.apply(lambda r: coalesce_amount(r.get("amount_sold"), r.get("amount_offered")), axis=1)
    df["num_investors_clean"] = df["num_investors"].apply(lambda x: int(safe_float(x) or 0))
    df["security_type_class"] = df.apply(classify_security_from_row, axis=1)
    df["amount_confidence"] = df.apply(amount_confidence, axis=1)

    # keep only valid dated events
    df = df[df["event_date"].notna()].copy()

    # ---- amendment flag ----
    df["ISAMENDMENT_CLEAN"] = df["ISAMENDMENT"].apply(lambda x: 1 if truthy(x) else 0) if "ISAMENDMENT" in df.columns else 0

    # =========================================================
    # 1) DEDUPE WITHIN ACCESSION (fix duplicates from joins)
    #    IMPORTANT: avoid _x/_y by removing overlap columns
    # =========================================================
    if "accession" in df.columns:
        # numeric versions for robust max
        df["amount_sold_num"] = df["amount_sold"].apply(lambda x: safe_float(x) or 0.0)
        df["amount_offered_num"] = df["amount_offered"].apply(lambda x: safe_float(x) or 0.0)
        df["amount_usd_num"] = df["amount_usd"].apply(lambda x: safe_float(x) or 0.0)

        # pick best categorical row per accession: prefer non-amendment then earliest event
        df_sorted = df.sort_values(
            ["accession", "ISAMENDMENT_CLEAN", "event_date"],
            ascending=[True, True, True],
        )
        best = df_sorted.drop_duplicates("accession", keep="first").copy()

        # aggregate numeric fields per accession (rename -> avoid _x/_y)
        agg = (
            df.groupby("accession", as_index=False)
              .agg({
                  "event_date": "min",
                  "filing_dt": "min",
                  "first_sale_dt": "min",
                  "amount_sold_num": "max",
                  "amount_offered_num": "max",
                  "amount_usd_num": "max",
                  "num_investors_clean": "max",
                  "ISAMENDMENT_CLEAN": "max",
              })
              .rename(columns={
                  "event_date": "event_date_min",
                  "filing_dt": "filing_dt_min",
                  "first_sale_dt": "first_sale_dt_min",
                  "amount_sold_num": "amount_sold_max",
                  "amount_offered_num": "amount_offered_max",
                  "amount_usd_num": "amount_usd_max",
                  "num_investors_clean": "num_investors_max",
                  "ISAMENDMENT_CLEAN": "had_amendment",
              })
        )

        # drop overlapping cols from best to prevent suffixes
        best = best.drop(
            columns=[
                "event_date", "filing_dt", "first_sale_dt",
                "amount_sold", "amount_offered", "amount_usd",
                "amount_sold_num", "amount_offered_num", "amount_usd_num",
                "num_investors_clean",
            ],
            errors="ignore"
        )

        df = best.merge(agg, on="accession", how="left")

        # restore unified columns
        df["event_date"] = df["event_date_min"]
        df["filing_dt"] = df["filing_dt_min"]
        df["first_sale_dt"] = df["first_sale_dt_min"]
        df["amount_sold"] = df["amount_sold_max"]
        df["amount_offered"] = df["amount_offered_max"]
        df["amount_usd"] = df["amount_usd_max"]
        df["num_investors_clean"] = df["num_investors_max"]

        df = df.drop(
            columns=[
                "event_date_min", "filing_dt_min", "first_sale_dt_min",
                "amount_sold_max", "amount_offered_max", "amount_usd_max",
                "num_investors_max",
                "amount_sold_num", "amount_offered_num", "amount_usd_num",
            ],
            errors="ignore"
        )

        df["amount_confidence"] = df.apply(amount_confidence, axis=1)

    return df.sort_values("event_date")

def summarize_events_for_debug(
    events: pd.DataFrame,
    max_lines: int = 10,
    show_accession: bool = False
) -> str:
    """
    Compact readable summary of events for one startup.
    Keeps it CSV-friendly (single string).
    """
    if events is None or events.empty:
        return "no_events"

    ev = events.copy()
    ev = ev[ev["event_date"].notna()].copy()
    ev = ev.sort_values("event_date")

    def _fmt_amt(x):
        try:
            v = float(x)
            if v <= 0:
                return ""
            # short format: 12.3M / 450k
            if v >= 1_000_000:
                return f"{v/1_000_000:.2f}M"
            if v >= 1_000:
                return f"{v/1_000:.0f}k"
            return f"{v:.0f}"
        except Exception:
            return ""

    lines = []
    for _, r in ev.head(max_lines).iterrows():
        dt = r.get("event_date")
        dt_s = dt.date().isoformat() if hasattr(dt, "date") else str(dt)

        sec = str(r.get("security_type_class", "") or "")
        amt = _fmt_amt(r.get("amount_usd", 0))
        inv = str(int(r.get("num_investors_clean", 0) or 0))
        conf = str(r.get("amount_confidence", "") or "")
        amend = "Amd" if int(r.get("ISAMENDMENT_CLEAN", 0) or 0) == 1 else ""
        rev = str(r.get("revenue_range", "") or "")

        acc = str(r.get("accession", "") or "")
        acc_s = f" acc={acc[-6:]}" if (show_accession and acc) else ""

        # format: 2021-05-10 | Equity | 3.13M | inv=20 | HIGH | Amd | rev=...
        parts = [dt_s, sec, amt, f"inv={inv}", conf]
        if amend:
            parts.append(amend)
        if rev:
            parts.append(f"rev={rev}")
        line = " | ".join([p for p in parts if p])
        lines.append(line + acc_s)

    extra = ""
    if len(ev) > max_lines:
        extra = f" (+{len(ev)-max_lines} more)"

    return " || ".join(lines) + extra
# ============================================================
# Feature engineering
# ============================================================

def compute_preA_estimated_raised(pre: pd.DataFrame, bucket: str) -> float:
    cfg = SERIES_A_CFG.get(bucket, SERIES_A_CFG["SAAS"])
    total = 0.0
    for _, r in pre.iterrows():
        sold = safe_float(r.get("amount_sold"))
        offered = safe_float(r.get("amount_offered"))

        if sold is not None and sold > 0:
            amt = sold
        elif offered is not None and offered > 0:
            amt = offered * 0.60
        else:
            amt = 0.0

        amt = min(amt, cfg.mega_cap * 0.25)
        total += amt
    return float(total)

# ============================================================
# Logging
# ============================================================

def log_line(i, total, org, cand_names, bucket, cik, score, events_found, seriesA_date, seriesA_amt, seriesA_conf, comment):
    print(
    f"[{i}/{total}] {org[:28]:28s} | "
    f"{(cand_names or '-')[:60]:60s} | "
    f"bucket={bucket:5s} | match={score:0.2f} | CIK={str(cik)[:10]:10s} | "
    f"events={events_found:4d} | A={seriesA_date or '-':10s} | "
    f"A_amt={str(seriesA_amt)[:11]:11s} | conf={seriesA_conf:4s} | {comment}"
)

# ============================================================
# Main pipeline
# ============================================================

def run():
    print("✅ RUN STARTED")
    print("input =", INPUT_CSV)
    print("db    =", SQLITE_DB)
    print("out   =", OUTPUT_CSV)

    if not SQLITE_DB.exists():
        raise FileNotFoundError(f"SQLite DB not found: {SQLITE_DB}")

    # ----------------------------
    # 1) Load input
    # ----------------------------
    df = pd.read_csv(str(INPUT_CSV), sep=SEP, engine="python", on_bad_lines="skip")

    # Drop junk column if present
    df = df.drop(columns=["Unnamed: 11"], errors="ignore")

    # --- FIX Crunchbase column name ---
    df.columns = [c.strip() for c in df.columns]
    if "Organization Name" in df.columns and "Organization_Name" not in df.columns:
        df = df.rename(columns={"Organization Name": "Organization_Name"})

    # Required columns
    required = ["Organization_Name", "Industries", "Headquarters Location", "Postal Code", "Founders", "Founded Date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    # ----------------------------
    # Filter: Last Funding = Series A -> date <= 2022-12-31; Series A <= 2023-01-01 (36m before 2026-01-01)
    # ----------------------------
    CUTOFF_36_MONTHS = datetime(2023, 1, 1).date()
    CUTOFF_LAST_FUNDING_SA = datetime(2022, 12, 31).date()
    n_before = len(df)
    def _is_series_a_last(r):
        lt = str(r.get("Last Funding Type", "") or "").strip().lower()
        le = str(r.get("Last Equity Funding Type", "") or "").strip().lower()
        return "series a" in lt or "series a" in le
    def _last_funding_date_ok(r):
        if not _is_series_a_last(r):
            return True
        dt = parse_any_date(r.get("Last Funding Date", "") or "")
        if not dt:
            return True
        return dt.date() <= CUTOFF_LAST_FUNDING_SA
    df = df[df.apply(_last_funding_date_ok, axis=1)].copy()
    n_after = len(df)
    print(f"  Filter (Last Funding = Series A -> date <= 2022-12-31): dropped {n_before - n_after} rows, kept {n_after}")

    # Normalize location fields
    df["Postal_Code"] = df["Postal Code"].astype(str).fillna("")
    df[["HQ_City", "HQ_State"]] = df["Headquarters Location"].apply(
        lambda x: pd.Series(split_hq_location(x))
    )

    # ----------------------------
    # 2) DB connect + build inverted index ONCE
    # ----------------------------
    conn = connect_sqlite(SQLITE_DB)
    show_schema(conn)

    # Build the issuer index once in main thread — workers reuse it
    _ensure_issuer_index(conn)
    print(f"  Issuer index ready (shared across all workers)")

    # ----------------------------
    # 3) Main loop — PARALLELIZED
    # ----------------------------
    _timing_totals = {}  # accumulate timings across all rows
    def process_one_row(row_dict, idx, total):
        """Process a single startup row. Uses shared issuer index + own SQLite connection for CIK lookups."""
        import time as _t
        _t0_all = _t.time()
        r = row_dict.copy()
        class _RowProxy:
            def __init__(self, d): self._d = d
            def get(self, k, default=""): return self._d.get(k, default)
            def __getattr__(self, k): return self._d.get(k)
        proxy = _RowProxy(r)

        _t0 = _t.time()
        r.update(fill_seriesA_from_crunchbase(proxy))
        r.update(fill_seriesB_from_crunchbase(proxy))
        _timing_totals.setdefault("crunchbase_fallback", []).append(_t.time() - _t0)

        for k in ("seriesB_date", "seriesB_amount_usd", "seriesB_amount_usd_num", "seriesB_source", "seriesB_comment"):
            if k not in r or r.get(k) is None or (isinstance(r.get(k), float) and pd.isna(r.get(k))):
                r[k] = ""

        org = str(r.get("Organization_Name", "") or "")

        _t0 = _t.time()
        bucket = industry_bucket(str(r.get("Industries", "")))
        r["industry_bucket"] = bucket
        founded_dt = parse_any_date(r.get("Founded Date"))
        _timing_totals.setdefault("industry+parse", []).append(_t.time() - _t0)

        _conn = conn
        try:
            _t0 = _t.time()
            cik, issuer_name, score, ov_cnt, ov_ratio = select_best_issuer(_conn, r, rp_lastname_cache=rp_cache)
            _timing_totals.setdefault("select_best_issuer", []).append(_t.time() - _t0)

            r["sec_match_cik"] = cik or ""
            r["sec_match_issuer_name"] = issuer_name or ""
            r["sec_match_score"] = float(score)
            r["sec_founders_overlap_count"] = int(ov_cnt)
            r["sec_founders_overlap_ratio"] = float(ov_ratio)
            r["sec_candidates_debug"] = ""  # skip debug (was calling get_candidate_issuers a 2nd time)
            cand_names = issuer_name or "-"  # skip 3rd call to get_candidate_issuers

            if not cik:
                if r.get("seriesA_source") == "crunchbase":
                    r["sec_events_found"] = 0
                    r["sec_events_summary"] = "no_formd_match_crunchbase_seriesA"
                    invalidate_seriesB_if_before_A(r)
                    log_line(idx, total, org, cand_names, bucket, "-", score, 0,
                             r.get("seriesA_date", ""), r.get("seriesA_amount_usd", ""),
                             r.get("seriesA_confidence", "LOW"), "crunchbase_seriesA_no_cik")
                    return r
                else:
                    log_line(idx, total, org, cand_names, bucket, "-", score, 0,
                             "", "", "LOW", "filtered_out_no_founder_match")
                    return None

            try:
                _t0 = _t.time()
                events = fetch_events_for_cik(_conn, cik)
                r["sec_events_found"] = int(len(events))
                r["sec_events_summary"] = summarize_events_for_debug(events, max_lines=12, show_accession=False)
                _timing_totals.setdefault("fetch_events", []).append(_t.time() - _t0)
            except Exception as e:
                log_line(idx, total, org, cand_names, bucket, cik, score, 0,
                         "", "", "LOW", f"filtered_out_events_error={str(e)[:60]}")
                return None

            _t0 = _t.time()
            feats = compute_features_from_events(events=events, bucket=bucket,
                                                  match_score=score, founded_dt=founded_dt)
            r.update(feats)
            r["seriesA_source"] = "formd"
            _timing_totals.setdefault("compute_features", []).append(_t.time() - _t0)

            _t0 = _t.time()
            if not r.get("seriesB_date") and r.get("seriesA_date"):
                cluster_b, comment_b = pick_series_b_from_formd(events, r["seriesA_date"], bucket)
                if cluster_b is not None and not cluster_b.empty:
                    cluster_b = cluster_b.sort_values("event_date")
                    series_b_dt = cluster_b["event_date"].min()
                    series_b_amt = float(cluster_b["amount_usd"].fillna(0).sum())
                    r["seriesB_date"] = series_b_dt.date().isoformat()
                    r["seriesB_amount_usd"] = fmt_usd(series_b_amt)
                    r["seriesB_amount_usd_num"] = series_b_amt
                    r["seriesB_source"] = "formd"
                    r["seriesB_comment"] = comment_b

            _timing_totals.setdefault("seriesB_formd", []).append(_t.time() - _t0)

            # Form D investor features (directors, investor count, prior rounds)
            # are now computed in step 02 on the filtered set (~570 rows instead
            # of 2K here) — much faster, same results.

            invalidate_seriesB_if_before_A(r)

            _timing_totals.setdefault("TOTAL", []).append(_t.time() - _t0_all)

            log_line(idx, total, org, cand_names, bucket, cik, score, len(events),
                     r.get("seriesA_date", ""), r.get("seriesA_amount_usd", ""),
                     r.get("seriesA_confidence", "LOW"), r.get("seriesA_comment", ""))
            return r
        except Exception:
            _timing_totals.setdefault("TOTAL", []).append(_t.time() - _t0_all)
            return None

    # Single-threaded with shared connection + shared inverted index
    # (faster than threads because no connection open/close overhead per company)
    total = len(df)
    row_dicts = [row.to_dict() for _, row in df.iterrows()]
    print(f"\n  Processing {total} companies (single-thread, shared index + connection)...")

    rows_out = []
    rp_cache = {}  # shared lastname cache across all companies
    for i, rd in enumerate(row_dicts):
        result = process_one_row(rd, i + 1, total)
        if result is not None:
            rows_out.append(result)

    # ── TIMING: select_best_issuer internals ──
    if _sbi_timings["get_candidates"]:
        n = len(_sbi_timings["get_candidates"])
        print("\n  ┌─── INSIDE select_best_issuer (%d calls) ───" % n)
        for key in ["get_candidates", "scoring_loop", "lastname_lookups"]:
            vals = _sbi_timings[key]
            avg = sum(vals) / len(vals) * 1000
            tot = sum(vals) * 1000
            print("  │  %-25s  avg=%7.1fms  total=%8.0fms" % (key, avg, tot))
        avg_cand = sum(_sbi_timings["n_candidates"]) / max(1, len(_sbi_timings["n_candidates"]))
        avg_miss = sum(_sbi_timings["n_cache_miss"]) / max(1, len(_sbi_timings["n_cache_miss"]))
        print("  │  avg candidates: %.1f, avg cache misses: %.1f" % (avg_cand, avg_miss))
        print("  └────────────────────────────────────────")

    # ── TIMING SUMMARY ──
    if _timing_totals:
        print("\n  ┌─── TIMING BREAKDOWN (per company avg) ───")
        n = len(_timing_totals.get("TOTAL", [1]))
        for key in ["crunchbase_fallback", "industry+parse", "select_best_issuer", "fetch_events", "compute_features", "seriesB_formd", "TOTAL"]:
            vals = _timing_totals.get(key, [])
            if vals:
                avg = sum(vals) / len(vals) * 1000
                tot = sum(vals) * 1000
                print(f"  │  {key:<25s}  avg={avg:>7.1f}ms  total={tot:>8.0f}ms")
        print("  └────────────────────────────────────────")

    # Filter: seriesA_date <= 2023-01-01 (36 months before 2026-01-01)
    n_before_out = len(rows_out)
    rows_out = [r for r in rows_out if (not r.get("seriesA_date")) or (parse_any_date(r.get("seriesA_date")) and parse_any_date(r.get("seriesA_date")).date() <= CUTOFF_36_MONTHS)]
    print(f"  Filter (seriesA_date <= 2023-01-01): dropped {n_before_out - len(rows_out)} rows, kept {len(rows_out)}")

    # ----------------------------
    # 4) Output cleanup
    # ----------------------------
    out_df = pd.DataFrame(rows_out)

    # Remove / stand-by columns you don't want
    drop_cols = [
        "sec_match_comment",                 # remove
        "seriesA_amount_confidence",         # remove (you kept seriesA_confidence)
        "seriesA_revenue_range",             # remove
        "preA_revenue_range_mode",           # remove

        # freeze ALL preA block
        "preA_event_count","preA_total_raised_usd","preA_total_sold_usd","preA_estimated_raised_usd",
        "preA_amount_median_usd","preA_velocity_days","preA_num_investors_total","preA_num_investors_max",
        "preA_round_size_avg_usd","preA_round_size_max_usd","preA_capital_efficiency",
        "preA_sec_timing_gap_days_avg","preA_sec_timing_gap_days_med",
    ]
    out_df = out_df.drop(columns=drop_cols, errors="ignore")

    # Save
    out_df.to_csv(str(OUTPUT_CSV), index=False)
    conn.close()


    print("\n✅ Saved ->", OUTPUT_CSV)

if __name__ == "__main__":
    run()



# ![image.png](attachment:image.png)

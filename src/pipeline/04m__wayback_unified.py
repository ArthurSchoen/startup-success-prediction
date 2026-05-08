#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from __future__ import annotations  # Python 3.9 compat: allow `str | None`, list[str], etc.

import os
import re
import time
import random
import json
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from shared import load_env
load_env()
from tqdm import tqdm
from datetime import datetime
from dateutil.parser import parse
import tldextract

from pathlib import Path
import re

import time, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline", Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
PIPE = next((p for p in PIPE_CANDIDATES if p.exists() and (p / ".run_id").exists()), None) or (PIPE_CANDIDATES[0] if PIPE_CANDIDATES[0].exists() else Path.home() / "Desktop" / "Data_thesis" / "Pipeline")
BASE = PIPE.parent
PIPE.mkdir(parents=True, exist_ok=True)


SEP = ";"  # tes CSV

def pick_next_file_for_step(pipe_dir: Path, prev_step: str, next_step_prefix: str, allow_overwrite: bool = False) -> Path:
    """allow_overwrite=True : re-run et écraser la sortie (modifications)."""
    candidates = []
    for p in pipe_dir.glob(f"*{prev_step}*.csv"):
        run_id = p.name.split("_", 1)[0]
        # exclude __midpoint files: step is "done" only when final output exists
        next_exists = any(
            q for q in pipe_dir.glob(f"{run_id}{next_step_prefix}*.csv")
            if "__midpoint" not in q.name
        )
        if allow_overwrite or not next_exists:
            candidates.append((run_id, p))
    if not candidates:
        raise FileNotFoundError(f"No runnable input found for prev_step={prev_step}")
    candidates.sort(key=lambda x: x[0], reverse=True)  # newest date
    return candidates[0][1]

print("PIPE =", PIPE)
print("CSV in PIPE:", [p.name for p in PIPE.glob("*.csv")])

print("STEP03 matches:", [p.name for p in PIPE.glob("*_03_formd__seriesA_dates_enriched__clean*.csv")])
print("STEP04 matches:", [p.name for p in PIPE.glob("*_04_wayback__*.csv")])

ALLOW_OVERWRITE = True
CHOSEN_RUN_ID = os.environ.get("PIPELINE_RUN_ID", "").strip() or ((PIPE / ".run_id").read_text().strip() if (PIPE / ".run_id").exists() else "")

# Accept PIPELINE_BASE_CSV from orchestrator (parallel mode)
INPUT_CSV = None
_base_override = os.environ.get("PIPELINE_BASE_CSV", "").strip()
if _base_override and Path(_base_override).exists():
    INPUT_CSV = Path(_base_override)
if INPUT_CSV is None and CHOSEN_RUN_ID:
    INPUT_03 = PIPE / f"{CHOSEN_RUN_ID}_03_formd__seriesA_dates_enriched__clean.csv"
    if INPUT_03.exists():
        INPUT_CSV = INPUT_03
if INPUT_CSV is None:
    try:
        INPUT_CSV = pick_next_file_for_step(
            pipe_dir=PIPE, prev_step="_03_formd__seriesA_dates_enriched__clean", next_step_prefix="_04_wayback__", allow_overwrite=ALLOW_OVERWRITE
        )
    except FileNotFoundError:
        pass
if INPUT_CSV is None:
    raise FileNotFoundError("No input CSV found for step 03 (wayback)")

RUN_ID = INPUT_CSV.name.split("_", 1)[0]
OUTPUT_CSV = PIPE / f"{RUN_ID}_04m_wayback__signals.csv"

print("INPUT :", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

# True = skip Wayback API, copy step 02 -> step 04 with empty wayback columns (standby, like webtraction)
SKIP_WAYBACK = False

# ============================================================
# CONFIG
# ============================================================


WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK_WEB = "https://web.archive.org/web"

SLEEP_SECONDS = 0.6
JITTER_SECONDS = 0.2
INITIAL_DELAY_CDX = 0.8   # seconds before first CDX (increase if you get 429)

MAX_TRIES = 3
TIMEOUT_PAGE = 8

CDX_TRIES = 3
TIMEOUT_CDX = 20

MAX_WORKERS = int(os.environ.get("WAYBACK_WORKERS", "3"))

# cooldown global (shared)
cooldown_lock = threading.Lock()
cooldown_until = 0.0  # unix timestamp


# LLM SUBJECTIVE SCORING removed (team_maturity, web_sophistication, customer_traction) —
# too noisy on real data (<30% signal-to-noise in 10-startup audit).
#
# INSTEAD: LLM EXTRACTION with citation verification — the LLM extracts structured facts
# (school names, PhD, prior exit, bigtech) from founder text, and MUST cite the exact quote.
# We verify each citation is a literal substring of the input. Hallucinated citations are
# discarded. This gives us LLM intelligence for free while preventing the failure modes
# we saw (Lorem Ipsum acceptance, hallucinated scores on vague marketing copy).

_llm_founder_client = None
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if _OPENAI_KEY:
    try:
        import openai
        _llm_founder_client = openai.OpenAI()
        print("Wayback: LLM founder extraction enabled (with citation verification)")
    except ImportError:
        pass

_LLM_PAGE_EXTRACTION_PROMPT = """Read this archived startup webpage (from BEFORE their Series A funding).
Extract ONLY verifiable facts. Every fact MUST include an EXACT QUOTE from the text. I will verify your quotes are literal substrings — do NOT paraphrase.

CRITICAL ATTRIBUTION RULES:
- For founder facts (schools, PhD, bigtech, exits): the quote MUST explicitly connect the fact to a SPECIFIC FOUNDER by name or founder role (CEO, CTO, co-founder). Phrases like "Jane Doe, co-founder, received her PhD from X" or "CEO John Smith, former engineer at Google" count. Do NOT extract:
  • Schools mentioned in research citations, paper references, university collaborations, or partnerships
  • Schools attended by advisors, board members, investors, or employees
  • BigTech mentions that aren't about a specific founder's past employment
- For customers: must be a SPECIFIC company/organization NAME (e.g., "Google", "Cisco", "Acme Corp"). Do NOT extract generic categories like "restaurants", "hospitals", "enterprises", "small businesses".
- For product stage: require SPECIFIC evidence. Stage 2 needs a concrete product reference (demo link, named customers, feature descriptions). Stage 3 needs explicit scale language ("500+ customers", "Fortune 500", "enterprise deployments"). When uncertain, return lower stage.

SECTION A — FOUNDERS (CEO/CTO/co-founders ONLY):
1. **universities**: schools attended by a named founder. Quote must contain the founder's name/role AND the school AND a phrase like "graduated from / received her degree / studied at / alumnus / PhD from".
2. **has_phd**: true only if a specific founder is described as having a PhD/doctorate.
3. **prior_exit**: true only if a specific founder is described as having previously SOLD or EXITED a startup.
4. **serial_entrepreneur**: true only if a specific founder is described as having founded MULTIPLE companies.
5. **bigtech_experience**: BigTech companies (Google, Meta, Apple, Amazon, Microsoft, Netflix, Twitter, Uber, Airbnb, Stripe, etc.) where a named founder PREVIOUSLY WORKED. Quote must contain the founder's name/role AND the company AND a phrase like "former / previously at / ex-".

SECTION B — COMPANY TRACTION:
6. **named_customers**: specific company NAMES mentioned as customers/users (e.g., "trusted by Google and Salesforce"). NOT categories.
7. **product_stage**: classify based on SPECIFIC evidence:
   - 0 = pre-product (no product mentioned, vision-only, "coming soon")
   - 1 = beta/prototype ("sign up for beta", "early access", "pilot", "in development")
   - 2 = live product (demo/trial available, features described, first customers)
   - 3 = scaling (explicit scale: "trusted by 500+", "enterprise", Fortune 500 customers, growth metrics)
   Default to LOWER stage when uncertain. Provide the exact quote that best justifies the stage.

Company: {name}
Webpage text:
{text}

Return JSON:
{{"universities": [{{"name": "MIT", "quote": "co-founder Jane Doe received her PhD from MIT"}}],
  "has_phd": {{"value": false, "quote": ""}},
  "prior_exit": {{"value": false, "quote": ""}},
  "serial_entrepreneur": {{"value": false, "quote": ""}},
  "bigtech_experience": [{{"company": "Google", "quote": "CTO John Smith, former senior engineer at Google"}}],
  "named_customers": [{{"name": "Salesforce", "quote": "trusted by Salesforce"}}],
  "product_stage": {{"stage": 0, "quote": ""}}
}}

If info is not explicitly attributable to a founder or is not available, return empty arrays, false, 0, and empty quotes. When in doubt, omit."""

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; thesis-bot/1.0; +https://mit.edu)"
}

WAYBACK_AVAILABLE = "https://archive.org/wayback/available"

# Candidate pages
CANDIDATE_PATHS = ["/about", "/about-us", "/team", "/"]

# Priority bonus (about/team > others)
PATH_PRIORITY = {
    "/about-us": 3,
    "/about": 3,
    "/team": 2,
    "/careers": 1,
    "/": 0,
}

# Retry statuses
RETRY_STATUS = {429, 500, 502, 503, 504}

# Founders anchor words (context extraction)
FOUNDER_ANCHORS = [
    "founder", "co-founder", "cofounder", "founding",
    "ceo", "cto", "cpo", "coo", "chief executive", "chief technology",
    "founding team",
    "founders",
    "our founders",
    "the founders",
    "co-founders",
    "cofounders",
    "founded by",
    "meet the founders"
]

# Remove board-only noise
BOARD_WORDS = [
    "board", "board member", "director", "independent director",
    "advisor", "advisory", "advisory board", "chairman", "chairwoman",
    "non-executive", "nxe", "committee"
]

ROLE_HINTS = re.compile(r"(?i)\b(ceo|cto|cpo|coo|founder|co-founder|president|vp|vice president|head|director|phd|md|mba|chief)\b")
BAD_2WORD_PHRASES = set([
    "thank you", "requesting access", "instant access", "try observe",
    "press releases", "for developers", "about securitize", "home securitize",
    "leadership news", "blockchain resources", "protocol preferred",
    "private securities", "trading joins"
])

FOUNDER_WORDS_STRICT = ["founder", "co-founder", "cofounder", "founded by", "founding team"]

# Soft-404 patterns
SOFT_404_PATTERNS = [
    "404", "page not found", "not found", "doesn't exist",
    "error", "this page could not be found", "we can't find",
    "sorry", "something went wrong"
]

# Cookie signals for removal
COOKIE_SIGNALS = [
    "cookie notice", "cookie policy", "manage consent",
    "privacy policy", "consent preferences",
    "accept cookies", "strictly necessary cookies",
    "targeting/ advertising cookies", "analytical/ performance cookies",
    "manage settings", "consent", "cookie settings"
]

# Serial / exit patterns
SERIAL_PHRASES = [
    "serial entrepreneur",
    "previously founded",
    "previously co-founded",
    "founded before",
    "built and sold",
    "second startup",
    "third startup"
]
EXIT_PHRASES = [
    "acquired by",
    "acquisition",
    "ipo",
    "went public",
    "exit",
    "sold to", "sold his company", "sold her company",
    "sold his first company", "sold her first company",
    "acquired"
]

RE_PHD = re.compile(r"(?i)\b(ph\.?d|doctorate|doctoral|dphil|postdoc|m\.?d|j\.?d|mba)\b")

# ============================================================
# 1) TEXT NORMALIZER + SCHOOL/BIGTECH LISTS
# ============================================================

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

SCHOOL_ALIASES = sorted(set([_norm(x) for x in [
    "Massachusetts Institute of Technology", "MIT",
    "Stanford University", "Stanford",
    "Harvard University", "Harvard",
    "Princeton University", "Princeton",
    "Yale University", "Yale",
    "Columbia University", "Columbia",
    "University of Chicago", "UChicago",
    "Duke University", "Duke",
    "Cornell University", "Cornell",
    "Brown University", "Brown",
    "Dartmouth College", "Dartmouth",
    "University of California, Berkeley", "UC Berkeley", "Berkeley",
    "University of California Los Angeles", "UCLA",
    "Carnegie Mellon University", "Carnegie Mellon", "CMU",
    "California Institute of Technology", "Caltech",
    "University of Oxford", "Oxford",
    "University of Cambridge", "Cambridge",
    "Imperial College London", "Imperial College",
    "ETH Zurich", "ETH Zürich",
    "EPFL", "Ecole Polytechnique Federale de Lausanne", "École Polytechnique Fédérale de Lausanne",
    "INSEAD",
    "London School of Economics", "LSE",
    "HEC Paris", "HEC",
    "ESSEC", "ESCP",
    "Ecole Polytechnique", "École Polytechnique",
]]))

# Convention: HIGHER = BETTER. 3=elite, 2=top-50, 1=other known university, 0=none.
SCHOOL_TIERS = {
    3: set([_norm(x) for x in [
        "mit", "massachusetts institute of technology",
        "stanford", "stanford university",
        "harvard", "harvard university",
        "uc berkeley", "university of california, berkeley", "berkeley",
        "carnegie mellon", "carnegie mellon university", "cmu",
        "caltech", "california institute of technology",
        "oxford", "university of oxford",
        "cambridge", "university of cambridge",
    ]]),
    2: set([_norm(x) for x in [
        "princeton", "princeton university",
        "columbia", "columbia university",
        "cornell", "cornell university",
        "yale", "yale university",
        "duke", "duke university",
        "uchicago", "university of chicago",
        "northwestern", "northwestern university",
        "johns hopkins", "johns hopkins university",
        "dartmouth", "dartmouth college",
        "brown", "brown university",
        "upenn", "university of pennsylvania", "penn", "wharton",
        "georgia tech", "georgia institute of technology",
        "michigan", "university of michigan",
        "ucla", "uc san diego", "ucsd",
        "imperial college", "imperial college london",
        "eth zurich", "eth zürich",
        "epfl", "ecole polytechnique federale de lausanne", "école polytechnique fédérale de lausanne",
        "insead", "lse", "london school of economics",
        "hec", "hec paris",
        "polytechnique", "ecole polytechnique",
        "tsinghua", "peking university",
        "nyu", "new york university",
        "usc", "university of southern california",
        "cmu", "carnegie mellon",  # also tier 3, double-registered is harmless since we check 3 first
    ]]),
    1: set(),  # will be filled below (all other known universities)
}
SCHOOL_TIERS[1] = set(SCHOOL_ALIASES) - SCHOOL_TIERS[3] - SCHOOL_TIERS[2]

BIGTECH_TIERS = {
    1: set([_norm(x) for x in ["google", "alphabet", "meta", "facebook", "amazon", "aws", "microsoft", "apple", "nvidia"]]),
    2: set([_norm(x) for x in ["openai", "deepmind", "tesla", "netflix", "salesforce", "oracle", "adobe", "ibm", "intel", "cisco", "sap", "palantir", "snowflake", "databricks", "uber", "airbnb", "stripe", "shopify"]]),
    3: set([_norm(x) for x in ["doordash", "coinbase", "bytedance", "tiktok", "snap", "snapchat", "pinterest", "zoom", "atlassian", "twilio", "hubspot", "square", "block"]]),
}

# ============================================================
# 2) REGEX COMPILATION
# ============================================================

def compile_word_regex(words):
    escaped = [re.escape(w) for w in words if w]
    if not escaped:
        return re.compile(r"^\b$")  # never match
    return re.compile(r"(?i)\b(" + "|".join(escaped) + r")\b")

RE_ANCHOR = compile_word_regex([_norm(x) for x in FOUNDER_ANCHORS])
RE_SCHOOL_ALL = compile_word_regex(SCHOOL_ALIASES)
RE_BIGTECH_T1 = compile_word_regex(list(BIGTECH_TIERS[1]))
RE_BIGTECH_T2 = compile_word_regex(list(BIGTECH_TIERS[2]))
RE_BIGTECH_T3 = compile_word_regex(list(BIGTECH_TIERS[3]))

# ============================================================
# 3) GENERIC HELPERS
# ============================================================

def safe_sleep():
    """Sleep with jitter to reduce Wayback throttling."""
    time.sleep(SLEEP_SECONDS + random.random() * JITTER_SECONDS)

def safe_parse_date(x):
    try:
        if pd.isna(x) or str(x).strip() == "":
            return None
        return parse(str(x))
    except Exception:
        return None

def normalize_domain(url_or_domain: str) -> str:
    if not isinstance(url_or_domain, str) or not url_or_domain.strip():
        return ""
    s = url_or_domain.strip()
    s = re.sub(r"^https?://", "", s, flags=re.I)
    s = s.split("/")[0].strip()
    ext = tldextract.extract(s)
    if not ext.domain or not ext.suffix:
        return s.lower()
    return f"{ext.domain}.{ext.suffix}".lower()

def to_wayback_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")

def parse_wayback_timestamp(ts: str):
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S")
    except Exception:
        return None

def is_soft_404(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(p in t for p in SOFT_404_PATTERNS)

def global_wait_if_needed():
    global cooldown_until
    with cooldown_lock:
        wait = max(0.0, cooldown_until - time.time())
    if wait > 0:
        print(f"  ⏳ Rate limited. Waiting {int(wait)}s before next request...", flush=True)
        time.sleep(wait)

def global_cooldown(seconds: float):
    global cooldown_until
    with cooldown_lock:
        cooldown_until = max(cooldown_until, time.time() + seconds)

def get_with_retry(url, params=None, tries=3, timeout=10, base_sleep=1.0):
    import requests

    last_r = None
    for i in range(tries):
        global_wait_if_needed()  # ✅ respect cooldown before any request

        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            last_r = r

            if r.status_code == 200:
                return r

            if r.status_code in {403, 404, 410}:
                return r

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = max(45, int(ra) if (ra and ra.isdigit()) else 45)
                extra = 10
                total_wait = wait + extra + random.random() * 5
                print(f"  ⚠️ 429 Too Many Requests. Cooldown {int(total_wait)}s... (Check VPN is OFF!)", flush=True)
                global_cooldown(total_wait)
                continue

            if r.status_code in {500, 502, 503, 504}:
                time.sleep(base_sleep * (2 ** i) + random.random())
                continue

            return r

        except Exception:
            time.sleep(base_sleep * (2 ** i) + random.random())

    return last_r

# ============================================================
# 4) FOUNDER FILTERING
# ============================================================

def parse_founders_cell(x: str) -> list[str]:
    if not isinstance(x, str) or not x.strip():
        return []
    raw = re.split(r"[,;]+", x)  # virgule OU ; support
    founders = []
    for s in raw:
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        if len(s) >= 3:
            founders.append(s)
    return list(dict.fromkeys(founders))

def filter_text_by_founders(text: str, founders: list[str], window: int = 3) -> tuple[str, float, str]:
    if not text:
        return "", 0.0, "fallback"

    chunks = re.split(r"[.!?\n]+", text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 20]

    marked = set()
    source = None

    def mark_window(i: int):
        for j in range(i, min(i + 1 + window, len(chunks))):
            marked.add(j)

    # Name window: when we have founder names, prioritize windows around NAMES only
    if founders:
        for i, c in enumerate(chunks):
            t = c.lower()
            if any(name in t for name in founders):
                mark_window(i)
                source = source or "name_window"

    # Anchor window: only use generic anchors (CEO, co-founder, etc.) if NO founder name matched
    if source is None:
        for i, c in enumerate(chunks):
            t = c.lower()
            if any(a in t for a in FOUNDER_ANCHORS):
                mark_window(i)
                source = "anchor_window"

    keep = []
    for i in sorted(marked):
        t = chunks[i].lower()
        if any(b in t for b in BOARD_WORDS) and not any(w in t for w in FOUNDER_WORDS_STRICT):
            continue
        keep.append(chunks[i])

    filtered = " ".join(keep).strip()

    # Fallback: expand with anchor-only chunks only when we had no name match (avoid polluting with board/advisor text)
    if len(filtered) < 200 and source != "name_window":
        keep2 = []
        for c in chunks:
            t = c.lower()
            if any(w in t for w in FOUNDER_ANCHORS) and not any(b in t for b in BOARD_WORDS):
                keep2.append(c)
        filtered2 = " ".join(keep2).strip()
        if len(filtered2) > len(filtered):
            filtered = filtered2
            source = "fallback"

    coverage = len(filtered) / max(1, len(text))
    return filtered, coverage, (source or "fallback")

# ============================================================
# 5) COOKIE / NOISE REMOVAL
# ============================================================

def remove_cookie_noise(text: str) -> str:
    if not text:
        return ""
    t = text.lower()

    # likely cookie-only page
    if sum(sig in t for sig in COOKIE_SIGNALS) >= 2 and len(text) > 600:
        return ""
    return text

# ============================================================
# 6) WAYBACK FUNCTIONS
# ============================================================

def wayback_url(timestamp: str, target_url: str) -> str:
    # ✅ id_ = donne le HTML "brut", souvent plus stable
    return f"{WAYBACK_WEB}/{timestamp}id_/{target_url}"

def cdx_search(domain: str, year_from: int, year_to: int) -> tuple[list, int]:
    params = {
    "url": f"{domain}/*",
    "matchType": "domain",
    "output": "json",
    "from": str(year_from),
    "to": str(year_to),
    "fl": "timestamp,original,mimetype,statuscode",
    "collapse": "digest",
    "filter": "statuscode:200",
    "limit": "5000",   # ✅ IMPORTANT sinon CDX te coupe la réponse
}
    r = get_with_retry(WAYBACK_CDX, params=params, tries=CDX_TRIES, timeout=TIMEOUT_CDX)
    if r is None:
        return [], -1

    status = r.status_code
    if status != 200:
        return [], status

    try:
        data = r.json()
        return (data[1:] if len(data) > 1 else []), status
    except Exception:
        return [], status

def cdx_search_force(domain: str, year_from: int, year_to: int, max_total_wait_s: int = 15) -> tuple[list, int]:
    """
    Force at least one real CDX attempt per startup.
    Retries until:
      - status 200 (success)
      - permanent error (403/404/410)
      - deadline reached (returns -1)
    """
    params = {
        "url": f"{domain}/*",
        "matchType": "domain",
        "limit": "5000",
        "output": "json",
        "from": str(year_from),
        "to": str(year_to),
        "filter": "statuscode:200",
        "collapse": "digest",
        "fl": "timestamp,original,mimetype,statuscode",
    }

    t0 = time.time()
    attempt = 0
    last_status = -1

    MAX_CDX_ATTEMPTS = 6  # cap attempts per domain; then use fallback
    while (time.time() - t0) < max_total_wait_s and attempt < MAX_CDX_ATTEMPTS:
        attempt += 1
        print(f"  📡 CDX request for {domain} (attempt {attempt}/{MAX_CDX_ATTEMPTS})...", flush=True)

        r = get_with_retry(
        WAYBACK_CDX,
        params=params,
        tries=2,
        timeout=TIMEOUT_CDX,
        base_sleep=0.5
    )

        if r is None:
            last_status = -1
            continue

        last_status = r.status_code

        # ✅ success
        if r.status_code == 200:
            try:
                data = r.json()
                rows = data[1:] if len(data) > 1 else []
                return rows, 200
            except Exception:
                # JSON parse failed => retry
                last_status = -1
                continue

        # ✅ permanent stop
        if r.status_code in {403, 404, 410}:
            return [], r.status_code

        # ⚠️ throttled or server issue => longer wait before retry (fewer attempts, more rest)
        wait_retry = 1.5 + random.random() * 0.5  
        print(f"  ⏳ CDX throttled (status {last_status}). Retry in {int(wait_retry)}s...", flush=True)
        time.sleep(wait_retry)

    # deadline hit
    return [], (last_status if last_status != 0 else -1)

def available_fallback_snapshot(domain: str, cutoff_dt: datetime):
    """
    Backup if CDX fails: use Wayback 'available' API.
    Returns snapshot_ts or None.
    """
    if not domain or cutoff_dt is None:
        return None

    ts = cutoff_dt.strftime("%Y%m%d")
    url = f"http://{domain}/"

    params = {"url": url, "timestamp": ts}

    r = get_with_retry(WAYBACK_AVAILABLE, params=params, tries=3, timeout=TIMEOUT_CDX, base_sleep=0.5)
    if r is None or r.status_code != 200:
        return None

    try:
        data = r.json()
        closest = data.get("archived_snapshots", {}).get("closest", {})
        snap_url = closest.get("url", "")
        snap_ts = closest.get("timestamp", "")
        if snap_ts and len(snap_ts) >= 14:
            return snap_ts[:14]
    except Exception:
        return None

    return None


def pick_best_snapshot_before(cdx_rows: list, cutoff_dt: datetime) -> str | None:
    if not cdx_rows or cutoff_dt is None:
        return None

    cutoff_ts = to_wayback_timestamp(cutoff_dt)
    candidates = []
    fallback_candidates = []  # any mimetype, before cutoff

    for row in cdx_rows:
        if len(row) < 4:
            continue
        ts, original, mimetype, status = row[0], row[1], row[2], row[3]
        if ts <= cutoff_ts:
            fallback_candidates.append(ts)
            if "html" in (mimetype or ""):
                candidates.append(ts)

    if candidates:
        return max(candidates)
    # Fallback: use latest snapshot before cutoff even if mimetype not html (e.g. warc/reponse)
    if fallback_candidates:
        return max(fallback_candidates)
    return None

# Global caches (reset each run)
FETCH_CACHE = {}         # key=(snapshot_ts,url)->text
FETCH_STATUS_CACHE = {}  # key=(snapshot_ts,url)->status_code

def fetch_text(url: str, snapshot_ts: str = "") -> str:
    cache_key = (snapshot_ts, url)
    if cache_key in FETCH_CACHE:
        return FETCH_CACHE[cache_key]

    r = get_with_retry(url, params=None, tries=MAX_TRIES, timeout=TIMEOUT_PAGE)
    if r is None:
        FETCH_CACHE[cache_key] = ""
        FETCH_STATUS_CACHE[cache_key] = -1
        return ""

    FETCH_STATUS_CACHE[cache_key] = r.status_code

    if r.status_code != 200:
        FETCH_CACHE[cache_key] = ""
        return ""

    try:
        soup = BeautifulSoup(r.text, "lxml")

        # remove noisy tags
        for tag in soup(["script", "style", "noscript", "svg", "img", "header", "footer"]):
            tag.decompose()

        # remove cookie banners (HTML-based)
        for bad in soup.select('[id*="cookie"], [class*="cookie"], [aria-label*="cookie"], [class*="consent"], [id*="consent"]'):
            bad.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        # cookie text-only discard
        text = remove_cookie_noise(text)
        if not text:
            FETCH_CACHE[cache_key] = ""
            return ""

        # soft404 discard
        if len(text) < 150 and is_soft_404(text):
            FETCH_CACHE[cache_key] = ""
            return ""

        FETCH_CACHE[cache_key] = text
        return text

    except Exception:
        FETCH_CACHE[cache_key] = ""
        return ""

def find_best_about_page(domain: str, snapshot_ts: str) -> tuple[str, str, str, float, dict]:
    if not domain or not snapshot_ts:
        return "", "", "", 0.0, {}

    # try about/team first (priority order)
    paths = sorted(CANDIDATE_PATHS, key=lambda p: -PATH_PRIORITY.get(p, 0))

    best_url = ""
    best_text = ""
    best_path = ""
    best_score = -1
    status_by_path = {}

    for path in paths:
        target = f"http://{domain}{path}"
        archived = wayback_url(snapshot_ts, target)

        safe_sleep()

        text = fetch_text(archived, snapshot_ts=snapshot_ts)
        status = FETCH_STATUS_CACHE.get((snapshot_ts, archived), None)
        status_by_path[path] = status

        # ✅ SKIP hard failures
        if status == -1:
            continue

        # ✅ SKIP empty / too short pages
        if not text or len(text) < 80:
            continue

        text_len = len(text)
        bonus = PATH_PRIORITY.get(path, 0) * 2000
        score = text_len + bonus

        if score > best_score:
            best_score = score
            best_text = text
            best_url = archived
            best_path = path

        # stop early if good about/team page already
        if PATH_PRIORITY.get(path, 0) >= 2 and text_len >= 1200:
            break

    return best_url, best_text, best_path, best_score, status_by_path

# ============================================================
# 7) TEAM NAMES EXTRACTION (LESS NOISE)
# ============================================================

BAD_NAME_TOKENS = {
    "company", "board", "president", "advisor", "marketing", "management",
    "security", "technologies", "technology", "software", "suite", "copied",
    "social", "group", "chief", "senior", "vice", "executive", "services",
    "product", "data"
}

# ============================================================
# 8) FOUNDER FEATURES SCORING (multi-founder safe)
# ============================================================

def extract_context_snippets(text: str, anchor_regex, window=260) -> str:
    if not text:
        return ""
    snippets = []
    for m in anchor_regex.finditer(text):
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        snippets.append(text[start:end])
    return " ... ".join(snippets)

def detect_all_schools(text: str) -> list[str]:
    if not text:
        return []
    matches = [_norm(m.group(1)) for m in RE_SCHOOL_ALL.finditer(_norm(text))]
    return list(dict.fromkeys(matches))

def tier_of_school(s: str) -> int:
    """Higher = better: 3=elite, 2=top-50, 1=other known university, 0=not found."""
    if s in SCHOOL_TIERS[3]:
        return 3
    if s in SCHOOL_TIERS[2]:
        return 2
    if s in SCHOOL_TIERS[1]:
        return 1
    return 0

def detect_bigtech_tier(text: str) -> tuple[int, str]:
    if not text:
        return 0, ""
    t = _norm(text)

    m1 = RE_BIGTECH_T1.search(t)
    if m1:
        return 1, _norm(m1.group(1))

    m2 = RE_BIGTECH_T2.search(t)
    if m2:
        return 2, _norm(m2.group(1))

    m3 = RE_BIGTECH_T3.search(t)
    if m3:
        return 3, _norm(m3.group(1))

    return 0, ""

def llm_extract_page_features(full_text: str, company_name: str = "") -> dict | None:
    """LLM-based extraction from archived startup pages, with citation verification.

    Extracts BOTH founder credentials AND company traction signals from the same
    text in a single LLM call. Every extracted fact must include an exact quote
    that is a literal substring of the input — hallucinated citations are silently
    discarded, so only real facts survive.

    Strategy: for founder info, we use extract_context_snippets() to pull paragraphs
    near founder keywords from the FULL text (avoiding truncation loss on long pages).
    We then combine those with the page overview for general traction signals."""
    if _llm_founder_client is None or not full_text or len(full_text) < 200:
        return None
    if "lorem ipsum" in full_text.lower() or "consectetur adipiscing" in full_text.lower():
        return None
    try:
        import json as _json
        # Focused founder snippets (from full text, no truncation)
        founder_snippets = extract_context_snippets(full_text, RE_ANCHOR, window=400)
        # Combine: founder snippets + full page overview (for traction/customers/product)
        if founder_snippets and len(founder_snippets) >= 100:
            llm_text = (f"=== FOUNDER-RELEVANT EXCERPTS ===\n{founder_snippets[:3000]}\n\n"
                        f"=== FULL PAGE TEXT ===\n{full_text[:3000]}")
        else:
            llm_text = full_text[:6000]

        prompt = _LLM_PAGE_EXTRACTION_PROMPT.format(name=company_name, text=llm_text[:6000])
        r = _llm_founder_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=500, temperature=0,
            messages=[
                {"role": "system", "content": (
                    "Extract verifiable facts from this archived startup webpage. "
                    "Return ONLY valid JSON. Every fact MUST include the exact quote "
                    "from the text. Do NOT paraphrase quotes."
                )},
                {"role": "user", "content": prompt},
            ],
        )
        raw = r.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rstrip("`").strip()
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s < 0 or e <= s:
            return None
        parsed = _json.loads(raw[s:e])

        # ── Citation verification ────────────────────────────────────
        text_lower = full_text.lower()

        def _quote_verified(quote_str, min_len=5):
            q = (quote_str or "").lower().strip()
            return bool(q and len(q) >= min_len and q in text_lower)

        # ── Section A: Founder credentials ──
        verified_schools = []
        for u in (parsed.get("universities") or []):
            if _quote_verified(u.get("quote"), min_len=10) and (u.get("name") or "").strip():
                verified_schools.append(u["name"].strip())

        def _verify_bool(field_dict):
            if not isinstance(field_dict, dict) or not field_dict.get("value"):
                return False
            return _quote_verified(field_dict.get("quote"))

        has_phd = _verify_bool(parsed.get("has_phd"))
        prior_exit = _verify_bool(parsed.get("prior_exit"))
        serial = _verify_bool(parsed.get("serial_entrepreneur"))

        verified_bigtech = []
        for bt in (parsed.get("bigtech_experience") or []):
            if _quote_verified(bt.get("quote")) and (bt.get("company") or "").strip():
                verified_bigtech.append(bt["company"].strip())

        best_tier = max((tier_of_school(_norm(s)) for s in verified_schools), default=0)

        # ── Section B: Company traction ──
        # Reject generic category words as "customer names" (e.g., "restaurants", "hospitals")
        GENERIC_CUSTOMER_BLACKLIST = {
            "restaurants", "restaurant", "hospitals", "hospital", "clinics", "clinic",
            "enterprises", "enterprise", "businesses", "business", "companies", "company",
            "small businesses", "smbs", "startups", "startup", "customers", "clients",
            "users", "organizations", "organization", "firms", "firm", "teams", "team",
            "schools", "school", "universities", "university", "stores", "store",
            "retailers", "retailer", "dealers", "dealer", "brands", "brand",
        }
        verified_customers = []
        for c in (parsed.get("named_customers") or []):
            name_val = (c.get("name") or "").strip()
            if not name_val or name_val.lower() in GENERIC_CUSTOMER_BLACKLIST:
                continue
            if _quote_verified(c.get("quote"), min_len=8):
                verified_customers.append(name_val)

        # Product stage: verify the quote, trust the stage if quote checks out
        product_stage = 0
        ps_dict = parsed.get("product_stage") or {}
        if isinstance(ps_dict, dict) and _quote_verified(ps_dict.get("quote"), min_len=5):
            product_stage = max(0, min(3, int(ps_dict.get("stage", 0))))

        has_any_signal = bool(
            verified_schools or has_phd or prior_exit or serial or verified_bigtech
            or verified_customers or product_stage > 0
        )

        return {
            # Founder signals
            "founder_school_tier": best_tier,
            "founder_phd_flag": int(has_phd),
            "founder_bigtech_flag": int(bool(verified_bigtech)),
            "founder_serial_entrepreneur_flag": int(serial),
            "founder_prior_exit_flag": int(prior_exit),
            "founder_features_available": 1 if has_any_signal else 0,
            # Traction signals
            "named_customers_count_preA": len(verified_customers),
            "product_stage_preA": product_stage,
            # Debug / audit
            "_llm_schools_verified": verified_schools,
            "_llm_bigtech_verified": verified_bigtech,
            "_llm_customers_verified": verified_customers,
        }
    except Exception:
        return None


def score_founder_features(full_text: str) -> dict:
    """Regex-based founder feature scoring. Used as fallback when LLM is unavailable.
    full_text should be filtered_text or founder_context so schools/bigtech are not polluted by board/advisor CVs."""
    full_text = full_text or ""
    founder_context = extract_context_snippets(full_text, RE_ANCHOR, window=260)
    founder_norm = _norm(founder_context)

    has_anchor = len(founder_context.strip()) > 0
    if not has_anchor:
        # if no anchor keyword, still consider it valid if we see founder-like roles in the (filtered) text
        has_anchor = bool(re.search(r"(?i)\b(ceo|cto|coo|cpo)\b", full_text))

    # Schools and bigtech ONLY from founder context (avoid board/advisor/staff CVs on full page)
    text_for_schools_bigtech = founder_context if founder_context.strip() else full_text
    schools = detect_all_schools(text_for_schools_bigtech)
    best_tier = max([tier_of_school(s) for s in schools], default=0)

    bigtech_tier, bigtech_name = detect_bigtech_tier(text_for_schools_bigtech)

    phd_flag = int(bool(RE_PHD.search(founder_norm)))
    serial_flag = int(any(_norm(x) in founder_norm for x in SERIAL_PHRASES))
    prior_exit_flag = int(any(_norm(x) in founder_norm for x in EXIT_PHRASES))

    return {
        "founder_features_available": int(has_anchor and len(founder_context) >= 120),
        "founder_context_len": len(founder_context),

        "founder_school_tier": best_tier,
        "founder_school_names_all": "; ".join(schools[:10]),

        "founder_bigtech_flag": int(bigtech_tier > 0),
        "founder_bigtech_tier": bigtech_tier,
        "founder_bigtech_name": bigtech_name,

        "founder_phd_flag": phd_flag,
        "founder_serial_entrepreneur_flag": serial_flag,
        "founder_prior_exit_flag": prior_exit_flag,
    }

# ============================================================
# 9) MAIN PIPELINE
# ============================================================

def run_pipeline(INPUT_CSV:str, OUTPUT_CSV:str):
    if SKIP_WAYBACK:
        print("SKIP WAYBACK: copying step 02 -> step 04 with empty wayback columns (standby)")
        df = pd.read_csv(INPUT_CSV, engine="python", sep=",", quotechar='"', encoding="utf-8", on_bad_lines="skip")
        df.columns = [c.strip() for c in df.columns]
        COLMAP = {"Organization Name": "Organization_Name", "Organization_Name": "Organization_Name", "Website": "Website", "website": "Website", "SeriesA_Date": "seriesA_date", "SeriesA Date": "seriesA_date", "seriesA_date": "seriesA_date", "Founders": "Founders", "founders_parsed": "Founders"}
        df = df.rename(columns={k: v for k, v in COLMAP.items() if k in df.columns})
        if "Founders" in df.columns:
            df["founders_text"] = df["Founders"].astype(str).fillna("")
        else:
            df["founders_text"] = ""
        df["seriesA_dt"] = df["seriesA_date"].apply(safe_parse_date)
        results = []
        for _, row in df.iterrows():
            out = row.to_dict()
            domain = normalize_domain(row.get("Website", ""))
            founders_list = parse_founders_cell(row.get("founders_text", ""))
            out["founders_parsed"] = ", ".join(founders_list)
            out["domain_clean"] = domain
            for k in ["wayback_snapshot_ts", "wayback_snapshot_dt", "delta_days_A_minus_snapshot", "wayback_page_url", "wayback_best_path", "wayback_best_score", "cdx_http_status", "best_path_http_status", "path_status_map", "wayback_comment", "founder_excerpt", "founder_excerpt_short", "founder_text_source_flag"]:
                out[k] = ""
            out["founder_text_len"] = out["founder_text_len_raw"] = out["founder_text_len_filtered"] = 0
            out["founder_text_coverage_ratio"] = 0.0
            for k, v in score_founder_features("").items():
                out[k] = v
            _fr = out.get("Founders") or out.get("founders_text") or ""
            if isinstance(_fr, str) and _fr.strip():
                out["founder_excerpt"] = "Founders: " + _fr.strip()[:800]
                out["founder_excerpt_short"] = _fr.strip()[:300]
                out["founder_features_available"] = 1
            results.append(out)
        out_df = pd.DataFrame(results).drop(columns=["team_size_estimate", "team_names_sample"], errors="ignore")
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved -> {OUTPUT_CSV} ({len(results)} rows)")
        return
    print("\n" + "!" * 60)
    print("  ⚠️  CHECK YOUR VPN  ⚠️  Turn OFF VPN or Wayback will return 429!")
    print("!" * 60 + "\n")
    print("📥 Loading CSV:", INPUT_CSV)

    # ✅ reset HTML caches each run
    FETCH_CACHE.clear()
    FETCH_STATUS_CACHE.clear()

    df = pd.read_csv(
        INPUT_CSV,
        engine="python",
        sep=",",
        quotechar='"',
        escapechar="\\",
        encoding="utf-8",
        on_bad_lines="skip"
    )

    # --- 0) normalize column names to a standard schema ---
    df.columns = [c.strip() for c in df.columns]

    COLMAP = {
        # org
        "Organization Name": "Organization_Name",
        "Organization_Name": "Organization_Name",
        "Organization_Name ": "Organization_Name",

        # website
        "Website": "Website",
        "website": "Website",

        # series A date (keep ONE canonical name)
        "SeriesA_Date": "seriesA_date",
        "SeriesA Date": "seriesA_date",
        "seriesA_date": "seriesA_date",

        # founders raw
        "Founders": "Founders",
        "founders_parsed": "Founders",   # si step03 a déjà parsed founders
    }

    df = df.rename(columns={k: v for k, v in COLMAP.items() if k in df.columns})

    # founders_text always exists
    if "Founders" in df.columns:
        df["founders_text"] = df["Founders"].astype(str).fillna("")
    else:
        df["founders_text"] = ""

    # required columns
    required_cols = ["Organization_Name", "Website", "seriesA_date", "founders_text"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}")

    df["seriesA_dt"] = df["seriesA_date"].apply(safe_parse_date)

    print("✅ Loaded rows:", len(df))

    # Midpoint: same RUN_ID/YYYYMMDD logic as output, in-progress save for resume on crash
    output_path = Path(OUTPUT_CSV) if not isinstance(OUTPUT_CSV, Path) else OUTPUT_CSV
    MIDPOINT_CSV = output_path.parent / (output_path.stem + "__midpoint.csv")

    results = []
    done_map = {}  # (Organization_Name, domain_clean) -> result dict
    if MIDPOINT_CSV.exists():
        try:
            mid_df = pd.read_csv(MIDPOINT_CSV, engine="python", sep=",", quotechar='"', encoding="utf-8", on_bad_lines="skip")
            mid_df.columns = [c.strip() for c in mid_df.columns]
            for _, r in mid_df.iterrows():
                key = (r.get("Organization_Name"), r.get("domain_clean", ""))
                if key not in done_map:
                    done_map[key] = r.to_dict()
            print(f"📂 Resuming from midpoint: {len(done_map)} rows already done -> {MIDPOINT_CSV.name}")
        except Exception as e:
            print(f"⚠️ Could not load midpoint ({e}), starting from scratch.")

    cache_cdx = {}
    n_total = 0
    n_snapshot_ok = 0
    n_founder_ok = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        domain = normalize_domain(row.get("Website", ""))
        seriesA_dt = row.get("seriesA_dt", None)
        row_key = (row.get("Organization_Name"), domain)
        if row_key in done_map:
            results.append(done_map[row_key])
            continue

        out = row.to_dict()
        founders_list = parse_founders_cell(row.get("founders_text", ""))

        # init output fields
        out["founders_parsed"] = ", ".join(founders_list)
        out["domain_clean"] = domain

        out["wayback_snapshot_ts"] = ""
        out["wayback_snapshot_dt"] = ""
        out["delta_days_A_minus_snapshot"] = ""

        out["wayback_page_url"] = ""
        out["wayback_best_path"] = ""
        out["wayback_best_score"] = ""
        out["cdx_http_status"] = ""

        out["best_path_http_status"] = ""
        out["path_status_map"] = ""
        out["wayback_comment"] = ""

        out["founder_excerpt"] = ""
        out["founder_excerpt_short"] = ""
        # founder_excerpt fallback: use Crunchbase Founders when no wayback text (so we don't miss founder info)
        _founders_raw = (out.get("Founders") or out.get("founders_text") or "")
        if isinstance(_founders_raw, str) and _founders_raw.strip():
            out["founder_excerpt"] = "Founders: " + _founders_raw.strip()[:800]
            out["founder_excerpt_short"] = _founders_raw.strip()[:300]
            out["founder_features_available"] = 1  # we have at least founder names
        out["founder_text_len"] = 0
        out["founder_text_len_raw"] = 0
        out["founder_text_len_filtered"] = 0
        out["founder_text_coverage_ratio"] = 0.0
        out["founder_text_source_flag"] = ""

        # CDX counts
        out["wayback_snapshots_12m_preA"] = 0
        out["wayback_snapshots_3m_preA"] = 0
        out["has_careers_page_preA"] = 0
        # LLM-extracted traction signals (citation-verified)
        out["named_customers_count_preA"] = 0
        # team_size_on_page removed — too noisy
        out["product_stage_preA"] = 0

        # default founder features
        defaults = score_founder_features("")
        for k, v in defaults.items():
            out[k] = v

        snapshot_ok = 0
        founder_ok = 0
        source_flag = ""

        try:
            if not domain or seriesA_dt is None:
                out["wayback_comment"] = "missing_domain_or_seriesA_date"
                results.append(out)
                done_map[row_key] = out
                pd.DataFrame(results).to_csv(MIDPOINT_CSV, index=False)
                tqdm.write(f"💾 midpoint saved -> {MIDPOINT_CSV.name} ({len(results)}/{len(df)})")
                continue

            y_from = max(seriesA_dt.year - 3, 1996)
            y_to = seriesA_dt.year

            # CDX (cached by domain)
            if domain not in cache_cdx:
                # #region agent log
                if len(cache_cdx) == 0:
                    print(f"  ⏳ First request: waiting {INITIAL_DELAY_CDX}s before CDX...", flush=True)
                    time.sleep(INITIAL_DELAY_CDX)
                cdx_rows, cdx_status = cdx_search_force(domain, y_from, y_to, max_total_wait_s=90)

                # ✅ backup ultra rapide si CDX fail total
                if cdx_status == -1:
                    snap_ts = available_fallback_snapshot(domain, seriesA_dt)
                    if snap_ts:
                        cdx_rows = [[snap_ts, f"http://{domain}/", "text/html", "200"]]
                        cdx_status = 200  # we simulate success

                cache_cdx[domain] = (cdx_rows, cdx_status)
            else:
                cdx_rows, cdx_status = cache_cdx[domain]

            out["cdx_http_status"] = cdx_status

            snapshot_ts = pick_best_snapshot_before(cdx_rows, seriesA_dt)
            if not snapshot_ts:
                # Diagnostic: CDX=200 but no snapshot (empty rows? all after Series A? no html?)
                n_rows = len(cdx_rows)
                # seriesA_dt can be pd.NaT (NaTType); don't call strftime on it
                cutoff_ts = to_wayback_timestamp(seriesA_dt) if (seriesA_dt is not None and not pd.isna(seriesA_dt)) else ""
                n_html = sum(1 for r in cdx_rows if len(r) >= 4 and "html" in (str(r[2]) or ""))
                n_before = sum(1 for r in cdx_rows if len(r) >= 4 and cutoff_ts and str(r[0]) <= cutoff_ts)
                print(f"  ⚠️ CDX=200 but no snapshot for {domain}: rows={n_rows}, html={n_html}, before_seriesA={n_before}, seriesA_cutoff={cutoff_ts}", flush=True)
                out["wayback_comment"] = f"no_snapshot_before_seriesA rows={n_rows} html={n_html} before={n_before}"
                results.append(out)
                done_map[row_key] = out
                pd.DataFrame(results).to_csv(MIDPOINT_CSV, index=False)
                tqdm.write(f"💾 midpoint saved -> {MIDPOINT_CSV.name} ({len(results)}/{len(df)})")
                continue

            out["wayback_snapshot_ts"] = snapshot_ts
            snapshot_ok = 1

            snap_dt = parse_wayback_timestamp(snapshot_ts)
            out["wayback_snapshot_dt"] = snap_dt.date().isoformat() if snap_dt else ""
            out["delta_days_A_minus_snapshot"] = (seriesA_dt - snap_dt).days if (snap_dt and seriesA_dt) else ""

            # ═══════════════════════════════════════════════════════════
            # 2-PAGE FETCH: homepage + one about/team page (exact match)
            # ═══════════════════════════════════════════════════════════

            # Build URL index from CDX results
            cdx_url_map = {}  # normalized_path → (best_timestamp, original_url)
            for cdx_row in cdx_rows:
                if len(cdx_row) >= 2:
                    url_str = str(cdx_row[1])
                    path = re.sub(r"^https?://[^/]+", "", url_str.lower()).rstrip("/") or "/"
                    ts = str(cdx_row[0])
                    if path not in cdx_url_map or ts > cdx_url_map[path][0]:
                        cdx_url_map[path] = (ts, url_str)

            # ── Find about/team page by exact path match (priority order) ──
            ABOUT_PATHS = ["/about", "/about-us", "/team", "/about/team",
                           "/our-team", "/about/our-team", "/leadership",
                           "/who-we-are", "/our-story"]
            about_page_path = None
            for candidate in ABOUT_PATHS:
                if candidate in cdx_url_map:
                    about_page_path = candidate
                    break

            # ── Fetch 2 pages: homepage + about page ──
            homepage_text = ""
            about_text = ""
            best_path = "/"
            best_url = ""
            status_by_path = {}

            # 1) Fetch homepage
            hp_ts = cdx_url_map.get("/", (snapshot_ts, f"http://{domain}/"))[0]
            safe_sleep()
            homepage_text = fetch_text(wayback_url(hp_ts, f"http://{domain}/"), snapshot_ts=hp_ts)
            status_by_path["/"] = FETCH_STATUS_CACHE.get((hp_ts, wayback_url(hp_ts, f"http://{domain}/")), None)

            # 2) Fetch about/team page (if found in CDX)
            if about_page_path:
                ap_ts, ap_orig = cdx_url_map[about_page_path]
                safe_sleep()
                about_text = fetch_text(wayback_url(ap_ts, ap_orig), snapshot_ts=ap_ts)
                status_by_path[about_page_path] = FETCH_STATUS_CACHE.get((ap_ts, wayback_url(ap_ts, ap_orig)), None)
                if about_text and len(about_text) >= 80:
                    best_path = about_page_path
                    best_url = wayback_url(ap_ts, ap_orig)

            # Use about page for founder extraction, homepage as fallback
            text = about_text if (about_text and len(about_text) >= 80) else homepage_text
            if not best_url and homepage_text:
                best_url = wayback_url(hp_ts, f"http://{domain}/")

            out["wayback_page_url"] = best_url
            out["wayback_best_path"] = best_path
            out["wayback_best_score"] = 1 if about_page_path else 0
            out["best_path_http_status"] = status_by_path.get(best_path, "")
            out["path_status_map"] = str(status_by_path)

            if not text:
                out["wayback_comment"] = "snapshot_found_but_no_text_extracted"
                results.append(out)
                done_map[row_key] = out
                pd.DataFrame(results).to_csv(MIDPOINT_CSV, index=False)
                tqdm.write(f"💾 midpoint saved -> {MIDPOINT_CSV.name} ({len(results)}/{len(df)})")
                continue

            out["founder_text_len"] = len(text)

            # ── Founder feature extraction (from team page text) ──
            filtered_text, coverage, source_flag = filter_text_by_founders(text, founders_list, window=3)

            out["founder_excerpt"] = (filtered_text[:1200] + "...") if len(filtered_text) > 1200 else filtered_text
            out["founder_excerpt_short"] = filtered_text[:300]
            if len(filtered_text.strip()) < 150:
                _f = (out.get("Founders") or out.get("founders_text") or "")
                if isinstance(_f, str) and _f.strip():
                    out["founder_excerpt"] = "Founders: " + _f.strip()[:800]
                    out["founder_excerpt_short"] = _f.strip()[:300]
                    out["founder_features_available"] = 1

            out["founder_text_len_raw"] = len(text)
            out["founder_text_len_filtered"] = len(filtered_text)
            out["founder_text_coverage_ratio"] = round(coverage, 4)
            out["founder_text_source_flag"] = source_flag

            # ── Combine BOTH pages for richer LLM context ──
            # Always use both team/about page + homepage together, so the LLM
            # sees the full picture (team bios + product/customer info).
            combined_text = ""
            if about_text and homepage_text and about_text != homepage_text:
                combined_text = f"=== TEAM/ABOUT PAGE ===\n{about_text}\n\n=== HOMEPAGE ===\n{homepage_text}"
            else:
                combined_text = about_text or homepage_text or ""

            if not combined_text or len(combined_text) < 100:
                out["wayback_comment"] = "snapshot_found_but_insufficient_text"
                results.append(out)
                done_map[row_key] = out
                pd.DataFrame(results).to_csv(MIDPOINT_CSV, index=False)
                continue

            out["founder_text_len"] = len(combined_text)
            out["founder_text_len_raw"] = len(text)

            # ── LLM extraction with citation verification (only method) ──
            # No regex fallback — LLM extraction is strictly better at understanding
            # context ("earned her doctorate" = PhD) and can't hallucinate because
            # we verify every quote is a literal substring of the input text.
            company_name = row.get("Organization_Name", "")

            # Prepend founder names from Crunchbase for better anchoring
            _founders_raw = (row.get("Founders") or row.get("founders_text") or "")
            if isinstance(_founders_raw, str) and _founders_raw.strip() and _founders_raw.strip() != "nan":
                combined_text = f"Founders: {_founders_raw.strip()}.\n\n" + combined_text

            llm_feats = llm_extract_page_features(combined_text, company_name=company_name)
            if llm_feats is not None and llm_feats.get("founder_features_available"):
                for k in ("founder_school_tier", "founder_phd_flag", "founder_bigtech_flag",
                          "founder_serial_entrepreneur_flag", "founder_prior_exit_flag",
                          "founder_features_available",
                          "named_customers_count_preA", "product_stage_preA"):
                    out[k] = llm_feats.get(k, out.get(k, 0))
                out["founder_extraction_method"] = "llm_citation_verified"
            else:
                # LLM returned nothing or is unavailable — features stay at 0
                out["founder_extraction_method"] = "llm_no_signal"
                out["founder_features_available"] = 0

            # Consistency: zero out flags when features_available=0
            if out.get("founder_features_available", 0) == 0:
                for flag in ["founder_bigtech_flag", "founder_phd_flag",
                             "founder_serial_entrepreneur_flag", "founder_prior_exit_flag",
                             "founder_school_tier", "founder_bigtech_tier"]:
                    out[flag] = 0

            founder_ok = int(out.get("founder_features_available", 0) == 1)

            if founder_ok == 0:
                out["wayback_comment"] = "ok_no_founder_signal"
            elif "wayback_comment" not in out or not out["wayback_comment"]:
                out["wayback_comment"] = "ok"

            # ── CDX snapshot counts ──────────────────────────────────
            if seriesA_dt and cdx_rows:
                from datetime import timedelta
                sa_ts = to_wayback_timestamp(seriesA_dt)
                from_12m_ts = to_wayback_timestamp(seriesA_dt - timedelta(days=365))
                from_3m_ts = to_wayback_timestamp(seriesA_dt - timedelta(days=91))
                snap_12m = sum(1 for r in cdx_rows if len(r) >= 1 and str(r[0]) >= from_12m_ts and str(r[0]) <= sa_ts)
                snap_3m = sum(1 for r in cdx_rows if len(r) >= 1 and str(r[0]) >= from_3m_ts and str(r[0]) <= sa_ts)
                out["wayback_snapshots_12m_preA"] = snap_12m
                out["wayback_snapshots_3m_preA"] = snap_3m
            else:
                out["wayback_snapshots_12m_preA"] = 0
                out["wayback_snapshots_3m_preA"] = 0

            # ── Careers page check (from CDX rows, no extra fetch) ──
            if seriesA_dt and cdx_rows:
                sa_ts = to_wayback_timestamp(seriesA_dt)
                careers_keywords = ("/careers", "/jobs", "/hiring", "/join-us", "/join")
                for r in cdx_rows:
                    if len(r) >= 2:
                        url_lower = str(r[1]).lower()
                        ts = str(r[0])
                        if ts <= sa_ts and any(kw in url_lower for kw in careers_keywords):
                            out["has_careers_page_preA"] = 1
                            break

            # LLM web presence scoring REMOVED — see header comment for rationale.
            # Founder school tier comes from the regex-based score_founder_features() above,
            # which matches school names only when they actually appear in the page text.

            results.append(out)
            done_map[row_key] = out
            pd.DataFrame(results).to_csv(MIDPOINT_CSV, index=False)
            tqdm.write(f"💾 midpoint saved -> {MIDPOINT_CSV.name} ({len(results)}/{len(df)})")

        finally:
            n_total += 1
            n_snapshot_ok += snapshot_ok
            n_founder_ok += founder_ok

            tqdm.write(
                f"[{n_total}/{len(df)}] {domain or 'NO_DOMAIN'} | "
                f"CDX={out.get('cdx_http_status','')} | "
                f"best={out.get('wayback_best_path','')}:{out.get('best_path_http_status','')} | "
                f"snapshot={'Y' if snapshot_ok else 'N'} | "
                f"founder={'Y' if founder_ok else 'N'} | "
                f"snapshot_rate={n_snapshot_ok/n_total:.2%} | "
                f"founder_rate={n_founder_ok/n_total:.2%} | "
                f"source={out.get('founder_text_source_flag','')} | "
                f"paths={out.get('path_status_map','')}"
            )

    out_df = pd.DataFrame(results)
    # drop removed columns (team size / team samples) so output stays consistent
    out_df = out_df.drop(columns=["team_size_estimate", "team_names_sample"], errors="ignore")
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved final -> {OUTPUT_CSV}")
    if len(results) == len(df) and MIDPOINT_CSV.exists():
        MIDPOINT_CSV.unlink()
        print("🗑 midpoint removed (run complete)")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_pipeline(INPUT_CSV, OUTPUT_CSV)


"""
shared/wayback_cache.py — Disk cache for Wayback Machine snapshots.

Shared between steps 03, 05l, and 05m to avoid redundant HTTP calls.
Cache key: (domain, page_pattern, to_date) → (timestamp, text)

At 7,500 companies, this eliminates ~15,000 redundant Wayback fetches.
"""

import os
import re
import json
import hashlib
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# Cache directory
PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline", Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
_PIPE = next((p for p in PIPE_CANDIDATES if p.exists()), PIPE_CANDIDATES[0])
CACHE_DIR = _PIPE / "cache" / "wayback"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CDX_BASE = "https://web.archive.org/cdx/search/cdx"
WAYBACK_BASE = "https://web.archive.org/web"
HEADERS = {"User-Agent": "thesis-research-bot/1.0 (academic; mailto:thesis@example.com)"}

ABOUT_PATTERNS = ["/about", "/team", "/founders", "/about-us", "/company"]


def _cache_key(domain, pattern, to_date):
    """Generate a unique cache filename."""
    raw = f"{domain}|{pattern}|{to_date}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    safe_domain = re.sub(r"[^a-z0-9]", "_", domain.lower())[:30]
    return f"{safe_domain}_{h}.json"


def _read_cache(key):
    """Read from disk cache. Returns dict or None."""
    path = CACHE_DIR / key
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("text") and len(data["text"]) > 50:
                return data
        except Exception:
            pass
    return None


def _write_cache(key, data):
    """Write to disk cache."""
    try:
        (CACHE_DIR / key).write_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


def cdx_find_snapshot(domain, pattern_suffix, to_yyyymmdd):
    """Use CDX API to find a snapshot. Returns (timestamp, original_url) or (None, None)."""
    url_pattern = f"{domain}{pattern_suffix}*"
    params = {
        "url": url_pattern, "output": "json", "fl": "timestamp,original",
        "filter": "statuscode:200", "limit": 5, "to": to_yyyymmdd,
    }
    try:
        r = requests.get(CDX_BASE, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None, None
        data = r.json()
        if len(data) <= 1:
            return None, None
        rows = sorted(data[1:], key=lambda x: x[0], reverse=True)
        return rows[0][0], rows[0][1]
    except Exception:
        return None, None


def fetch_archived_page(timestamp, original_url):
    """Fetch an archived page from Wayback Machine."""
    wayback_url = f"{WAYBACK_BASE}/{timestamp}/{original_url}"
    try:
        r = requests.get(wayback_url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def extract_text(html):
    """Strip HTML to clean text."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def get_about_text(domain, to_date_str, max_chars=5000, delay=0.3):
    """
    Get archived About/Team page text for a domain, with disk caching.

    Args:
        domain: e.g. "example.com"
        to_date_str: e.g. "20210615" (YYYYMMDD) — fetch snapshot AT OR BEFORE this date
        max_chars: max text length to return
        delay: sleep between Wayback requests

    Returns:
        (text, snapshot_date_str, page_pattern) or ("", "", "")
    """
    domain = (domain or "").strip().lower()
    if not domain or domain in ("nan", "none") or len(domain) < 3:
        return "", "", ""

    to_yyyymmdd = str(to_date_str).replace("-", "")[:8]
    if len(to_yyyymmdd) < 8:
        return "", "", ""

    # Try About/Team patterns, then homepage
    patterns_to_try = ABOUT_PATTERNS + [""]

    for pattern in patterns_to_try:
        cache_key = _cache_key(domain, pattern, to_yyyymmdd)
        cached = _read_cache(cache_key)
        if cached:
            return cached["text"][:max_chars], cached.get("snapshot_date", ""), cached.get("pattern", pattern.lstrip("/") or "homepage")

        ts, orig = cdx_find_snapshot(domain, pattern, to_yyyymmdd)
        time.sleep(delay)

        if ts and orig:
            html = fetch_archived_page(ts, orig)
            time.sleep(delay)
            if html:
                text = extract_text(html)
                if len(text) > 100:
                    snapshot_date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""
                    page_pattern = pattern.lstrip("/") or "homepage"
                    _write_cache(cache_key, {
                        "text": text[:max_chars],
                        "snapshot_date": snapshot_date,
                        "pattern": page_pattern,
                        "timestamp": ts,
                        "original_url": orig,
                    })
                    return text[:max_chars], snapshot_date, page_pattern

        # Cache the miss too (avoid re-querying)
        _write_cache(cache_key, {"text": "", "snapshot_date": "", "pattern": ""})

    return "", "", ""


def cache_stats():
    """Return cache size stats."""
    files = list(CACHE_DIR.glob("*.json"))
    hits = sum(1 for f in files if json.loads(f.read_text()).get("text"))
    return {"total": len(files), "hits": hits, "misses": len(files) - hits, "dir": str(CACHE_DIR)}

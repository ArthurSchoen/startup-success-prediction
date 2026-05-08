#!/usr/bin/env python
# coding: utf-8

# # 05p – Product Hunt Signals
#
# **Input** : CSV with `Organization_Name`, `domain_clean`, `seriesA_date`.
#
# **Signal** : Product Hunt launch data before Series A.
# - `ph_launch` (0/1) : was the product launched on Product Hunt?
# - `ph_upvotes` : upvotes on the launch post
# - `ph_comments` : comments on the launch post
# - `ph_launch_date` : date of the PH launch
# - `ph_before_seriesA` (0/1) : was the PH launch before Series A?
# - `ph_featured` (0/1) : was the product featured (Product of the Day/Week)?
#
# **API** : Product Hunt GraphQL API v2 (requires OAuth token).
# Rate limit: 500 requests/day (free developer token).
# Batch search: up to 20 results per query → 7,500 companies in ~375 requests (1 day).
#
# **Setup** : Register at https://api.producthunt.com/v2/oauth/applications
# Then set PRODUCTHUNT_TOKEN in .env

import os
import re
import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path

PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline", Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
PIPE = next((p for p in PIPE_CANDIDATES if p.exists() and (p / ".run_id").exists()), None) or (PIPE_CANDIDATES[0] if PIPE_CANDIDATES[0].exists() else Path.home() / "Desktop" / "Data_thesis" / "Pipeline")
PIPE.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from shared import load_env
load_env()

def run_id_from_filename(name):
    m = re.match(r"^([^_]+)_", Path(name).name)
    if not m:
        raise ValueError(f"Bad filename: {name}")
    return m.group(1)

CHOSEN_RUN_ID = os.environ.get("PIPELINE_RUN_ID", "").strip() or ((PIPE / ".run_id").read_text().strip() if (PIPE / ".run_id").exists() else "")

INPUT_CSV = None
_base_override = os.environ.get("PIPELINE_BASE_CSV", "").strip()
if _base_override and Path(_base_override).exists():
    INPUT_CSV = Path(_base_override)
if INPUT_CSV is None and CHOSEN_RUN_ID:
    p = PIPE / f"{CHOSEN_RUN_ID}_05_webtraction__signals.csv"
    if p.exists():
        INPUT_CSV = p
if INPUT_CSV is None:
    candidates = sorted(PIPE.glob("*_05_webtraction__signals.csv"), key=lambda x: x.name, reverse=True)
    if candidates:
        INPUT_CSV = candidates[0]
if INPUT_CSV is None:
    raise FileNotFoundError("No _05_webtraction__signals.csv found in Pipeline")

RUN_ID = run_id_from_filename(INPUT_CSV)
OUTPUT_CSV = PIPE / f"{RUN_ID}_04f_product_hunt__signals.csv"
print("INPUT:", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

PH_API = "https://api.producthunt.com/v2/api/graphql"
PRODUCTHUNT_TOKEN = os.environ.get("PRODUCTHUNT_TOKEN", "").strip()
if not PRODUCTHUNT_TOKEN:
    print("WARNING: PRODUCTHUNT_TOKEN not set. Product Hunt signals will be skipped.")
    print("  → Register at https://api.producthunt.com/v2/oauth/applications")
    print("  → Add PRODUCTHUNT_TOKEN=your_token to .env")
else:
    print(f"Product Hunt: token set (len={len(PRODUCTHUNT_TOKEN)})")

# LLM for post-match verification (domain + description cross-check)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_llm_client = None
if OPENAI_API_KEY:
    try:
        import openai, json as _json
        _llm_client = openai.OpenAI()
        print("Product Hunt: LLM verification enabled")
    except ImportError:
        pass

# GraphQL: lookup by slug. Drops votesCount and commentsCount — those are CURRENT totals
# at query time (not historical), so using them leaks post-Series-A activity into the feature.
# Same leakage pattern we caught and fixed in Hacker News (06f).
#
# PH API limitations (confirmed via introspection 2026-04):
#   - No free-text search
#   - No query by real website URL (all URL fields are PH redirects)
#   - Products have unpredictable slugs (Replika → "replikaai", Fauna → "faunadb")
#   - Launches of the same product use different slugs ("replika-2", "replika-4")
# Strategy: try ~10 slug variants, LLM-verify each hit, enumerate numbered re-launches.
SLUG_QUERY = """query($slug: String!) {
  post(slug: $slug) {
    id
    name
    slug
    tagline
    description
    featuredAt
    createdAt
    url
  }
}"""


def _verify_ph_post_llm(post_name, post_tagline, post_desc, company_name, company_description):
    """LLM check: is this Product Hunt post about the same company?
    Used only when the post's website doesn't cleanly match the company's domain."""
    if _llm_client is None:
        return True  # Fail open when LLM is unavailable
    try:
        import json as _json
        llm_r = _llm_client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=20, temperature=0,
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": (
                    f'Company: "{company_name}" — {company_description[:200]}\n\n'
                    f'Product Hunt post:\n'
                    f'  name: "{post_name}"\n'
                    f'  tagline: "{post_tagline[:150]}"\n'
                    f'  description: "{(post_desc or "")[:200]}"\n\n'
                    f'Is this Product Hunt post describing the SAME company? '
                    f'Return match=1 only if the post\'s product clearly matches the company\'s business.\n'
                    f'Return: {{"match": 0 or 1}}'
                )},
            ],
        )
        raw = llm_r.choices[0].message.content.strip()
        s, e = raw.find("{"), raw.rfind("}") + 1
        return _json.loads(raw[s:e]).get("match", 0) == 1
    except Exception:
        return True


def _make_slugs(company_name, domain):
    """Exhaustive slug candidate generation for PH.

    Covers patterns observed in real data:
    - Direct name: 'Replika' → 'replika'
    - Name + TLD stem: 'Replika' (.ai) → 'replikaai', 'replika-ai'
    - Name + category: 'Fauna' (database) → 'faunadb', 'fauna-db'
    - Domain stem: 'workramp.com' → 'workramp'
    - No-hyphen compaction: 'Hinge Health' → 'hinge-health', 'hingehealth'
    """
    slugs = []
    name = (company_name or "").strip().lower()

    # Extract domain TLD (e.g., "ai" from "replika.ai") and stem
    domain_tld = ""
    domain_stem = ""
    if domain:
        d = re.sub(r"^https?://", "", domain).split("/")[0].lower()
        d = re.sub(r"^www\.", "", d)
        parts = d.rsplit(".", 1)
        if len(parts) == 2:
            domain_tld = parts[1]  # "ai", "com", "io", ...
        domain_stem = re.sub(r"\.[a-z]{2,6}$", "", d)  # "replika" from "replika.ai"
        domain_stem = re.sub(r"[^a-z0-9-]", "-", domain_stem).strip("-")

    if name:
        base = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
        if base:
            # Priority 1: most common slugs — direct base and compact form
            slugs.append(base)                         # "replika", "hinge-health"
            compact = base.replace("-", "")
            if compact != base:
                slugs.append(compact)                  # "hingehealth"
            # Priority 2: TLD-suffix variants (catches "replikaai" from replika.ai)
            if domain_tld and domain_tld not in ("com", "net", "org"):
                slugs.append(compact + domain_tld)     # "replikaai"
                slugs.append(base + "-" + domain_tld)  # "replika-ai"
            # Priority 3: common numbered re-launches (Replika was found at "replika-2")
            slugs.append(base + "-2")
            slugs.append(base + "-3")
            # Priority 4: category suffixes (catches "faunadb")
            for suf in ("db", "io", "app", "hq", "ai"):
                if suf != domain_tld:
                    slugs.append(compact + suf)
    # Domain stem comes last — only useful if the company name doesn't match the domain
    if domain_stem and domain_stem not in slugs:
        slugs.append(domain_stem)
        if "-" in domain_stem:
            slugs.append(domain_stem.replace("-", ""))

    # Dedup preserving order, filter tiny slugs
    return list(dict.fromkeys(s for s in slugs if s and len(s) >= 2))[:12]


def _fetch_post_by_slug(headers, slug):
    """Single PH lookup. Returns the post dict or None. Returns 'RATE_LIMITED' sentinel on 429."""
    try:
        r = requests.post(PH_API, headers=headers, json={
            "query": SLUG_QUERY, "variables": {"slug": slug},
        }, timeout=15)
        if r.status_code == 429:
            return "RATE_LIMITED"
        if r.status_code == 401:
            print(f"  PH AUTH ERROR for {slug} — check PRODUCTHUNT_TOKEN")
            return None
        if r.status_code != 200:
            return None
        return r.json().get("data", {}).get("post")
    except requests.exceptions.Timeout:
        print(f"  PH timeout for {slug}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"  PH connection error for {slug}")
        return None
    except Exception as e:
        print(f"  PH error for {slug}: {e}")
        return None


def _extract_launch_base(post):
    """Extract the 'base' launch slug from a post's url field.
    e.g. '.../launches/replika-4' → 'replika'  (strip trailing -N)
    This lets us enumerate sibling launches (replika, replika-2, ...)."""
    url = (post or {}).get("url", "") or ""
    m = re.search(r"/launches/([a-z0-9-]+)", url)
    if not m:
        return (post or {}).get("slug", "")
    slug = m.group(1)
    return re.sub(r"-\d+$", "", slug)


def ph_signals(company_name, domain, series_a_date, company_description="", name_distinctive=1):
    """Query Product Hunt with exhaustive slug variants, LLM-verify each hit,
    enumerate sibling launches of the same product, pick the earliest pre-Series-A
    launch, and report per-launch featured status.

    Returns features computed from VERIFIED pre-Series-A launches only."""
    out = {
        "ph_launch": 0,
        "ph_launch_date": "",
        "ph_before_seriesA": 0,
        "ph_featured": 0,
        "ph_launches_count_preA": 0,
        "ph_note": "",
    }

    if not PRODUCTHUNT_TOKEN:
        out["ph_note"] = "no_token"
        return out

    name = (company_name or "").strip()
    if not name or name in ("nan", "None"):
        out["ph_note"] = "no_name"
        return out

    if pd.notna(name_distinctive) and int(name_distinctive) == 0:
        out["ph_note"] = "skipped_non_distinctive"
        return out

    headers = {
        "Authorization": f"Bearer {PRODUCTHUNT_TOKEN}",
        "Content-Type": "application/json",
    }

    # ── Step 1: slug lookup with EARLY STOP on first verified hit ──
    # Instead of trying all 12 slugs blindly, we verify each hit immediately via LLM.
    # As soon as we get one verified match, we stop searching slugs and move to
    # enumerating sibling launches of THAT product. This saves ~half the API calls
    # and keeps us under PH's rate limit.
    slugs_to_try = _make_slugs(name, domain)
    verified_hits = []
    seen_ids = set()
    product_verdict_cache = {}

    for slug in slugs_to_try:
        # Sleep FIRST (before each call) — guaranteed throttling even on early stop
        time.sleep(0.6)  # ~1.5 req/s, safely under PH's rate limit
        post = _fetch_post_by_slug(headers, slug)
        if post == "RATE_LIMITED":
            out["ph_note"] = "rate_limited"
            return out
        if not post or not post.get("id") or post["id"] in seen_ids:
            continue
        seen_ids.add(post["id"])

        # Extract product slug from URL
        m = re.search(r"/products/([a-z0-9-]+)/launches/", post.get("url", "") or "")
        product_slug = m.group(1) if m else post.get("slug", "")

        # Verify via LLM (cached per product)
        if product_slug not in product_verdict_cache:
            product_verdict_cache[product_slug] = _verify_ph_post_llm(
                post.get("name", "") or "",
                post.get("tagline", "") or "",
                post.get("description", "") or "",
                name, company_description,
            )
        if product_verdict_cache[product_slug]:
            verified_hits.append(post)
            break  # ← EARLY STOP: we found a verified match, no need to keep trying slugs

    if not verified_hits:
        out["ph_note"] = "not_found" if not product_verdict_cache else "all_rejected_by_llm"
        return out

    # ── Step 2: enumerate sibling launches of the verified product ──
    # If we found "replika-2", also try "replika", "replika-3", etc. to find all launches.
    base = _extract_launch_base(verified_hits[0])
    if base:
        for candidate in [base] + [f"{base}-{i}" for i in range(2, 7)]:
            if any(h.get("slug") == candidate for h in verified_hits):
                continue  # already have it
            time.sleep(0.6)  # throttle BEFORE the call
            post = _fetch_post_by_slug(headers, candidate)
            if post == "RATE_LIMITED":
                break  # keep what we have
            if post and post.get("id") and post["id"] not in seen_ids:
                # Verify this new launch belongs to the same product
                m = re.search(r"/products/([a-z0-9-]+)/launches/", post.get("url", "") or "")
                prod = m.group(1) if m else post.get("slug", "")
                if prod not in product_verdict_cache:
                    product_verdict_cache[prod] = _verify_ph_post_llm(
                        post.get("name", "") or "",
                        post.get("tagline", "") or "",
                        post.get("description", "") or "",
                        (company_name or "").strip(), company_description,
                    )
                if product_verdict_cache.get(prod, False):
                    seen_ids.add(post["id"])
                    verified_hits.append(post)

    # ── Step 3: filter to pre-Series-A launches, compute features ──
    sa_dt = pd.to_datetime(series_a_date, errors="coerce") if series_a_date else None
    pre_a_launches = []
    for post in verified_hits:
        created = (post.get("createdAt") or "")[:10]
        if not created:
            continue
        if sa_dt is not None and not pd.isna(sa_dt):
            try:
                ph_dt = pd.to_datetime(created, errors="coerce")
                if pd.isna(ph_dt) or ph_dt > sa_dt:
                    continue
            except Exception:
                continue
        pre_a_launches.append(post)

    if not pre_a_launches:
        out["ph_note"] = f"{len(verified_hits)}_launches_all_post_seriesA"
        return out

    pre_a_launches.sort(key=lambda p: p.get("createdAt", ""))
    earliest = pre_a_launches[0]
    any_featured = any(p.get("featuredAt") for p in pre_a_launches)

    out["ph_launch"] = 1
    out["ph_launch_date"] = (earliest.get("createdAt") or "")[:10]
    out["ph_before_seriesA"] = 1
    out["ph_featured"] = 1 if any_featured else 0
    out["ph_launches_count_preA"] = len(pre_a_launches)
    out["ph_note"] = f"verified_{len(pre_a_launches)}_launches_preA"
    return out


# Load data
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

# Only clean pre-Series-A columns. ph_upvotes / ph_comments REMOVED —
# Product Hunt GraphQL returns CURRENT vote/comment counts, not historical,
# which would leak post-Series-A activity into the feature.
ph_cols = ["ph_launch", "ph_launch_date", "ph_before_seriesA", "ph_featured",
           "ph_launches_count_preA", "ph_note"]
for c in ph_cols:
    if c not in df.columns:
        df[c] = "" if c in ("ph_launch_date", "ph_note") else 0

# Sector filter: skip deeptech (biotech/hardware rarely launch on PH)
SKIP_SECTORS = {"deeptech"}
has_sector = "startup_sector" in df.columns

if not PRODUCTHUNT_TOKEN:
    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved (empty — no token):", OUTPUT_CSV)
else:
    skipped = 0
    queried = 0
    from tqdm import tqdm
    for i in tqdm(df.index, desc="Product Hunt"):
        # Skip sectors where PH is irrelevant (saves API calls)
        if has_sector:
            sector = str(df.loc[i, "startup_sector"]).strip().lower()
            if sector in SKIP_SECTORS:
                df.at[i, "ph_note"] = "sector_skip"
                skipped += 1
                continue

        name = str(df.loc[i, "Organization_Name"]).strip()
        domain = str(df.loc[i, "domain_clean"]).strip() if "domain_clean" in df.columns else ""
        sdate = str(df.loc[i, "seriesA_date"]).strip() if "seriesA_date" in df.columns else ""
        desc = str(df.loc[i, "Description"]).strip()[:250] if "Description" in df.columns else ""
        nd = df.loc[i, "name_distinctive"] if "name_distinctive" in df.columns else 1

        out = ph_signals(
            name, domain,
            sdate if sdate not in ("nan", "NaT", "") else None,
            company_description=desc,
            name_distinctive=nd,
        )
        for k, v in out.items():
            df.at[i, k] = v
        queried += 1

    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved:", OUTPUT_CSV)
    launched = int(df["ph_launch"].sum())
    before_sa = int(df["ph_before_seriesA"].sum())
    featured = int(df["ph_featured"].sum())
    print(f"Product Hunt: queried {queried}, skipped {skipped} (sector filter)")
    print(f"  Launches found: {launched}, before Series A: {before_sa}, featured: {featured}")
    print(f"  API calls used: {queried}/500 daily limit")

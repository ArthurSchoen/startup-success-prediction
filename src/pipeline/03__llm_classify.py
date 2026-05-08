#!/usr/bin/env python
# coding: utf-8

# # 05n – DeepTech LLM Score
#
# **Input** : CSV de l'étape 05 (webtraction), avec `Organization_Name`, `Description`, `Industries`.
#
# **Signal** : LLM-based deeptech score (0-10) using company description + industries.
# Replaces the rule-based `deeptech` / `deeptech_score` from step 02.
#
# - `deeptech_llm_score` : 0 = pure consumer/SaaS, 10 = breakthrough science/hardware
#
# Cost: ~$0.50 for 7,500 companies with GPT-4o-mini.

import os
import re
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

PIPE_CANDIDATES = [Path.cwd().parent / "Pipeline", Path.cwd() / "Pipeline", Path.home() / "Desktop" / "Data_thesis" / "Pipeline"]
PIPE = next((p for p in PIPE_CANDIDATES if p.exists() and (p / ".run_id").exists()), None) or (PIPE_CANDIDATES[0] if PIPE_CANDIDATES[0].exists() else Path.home() / "Desktop" / "Data_thesis" / "Pipeline")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from shared import load_env
load_env()
PIPE.mkdir(parents=True, exist_ok=True)

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
OUTPUT_CSV = PIPE / f"{RUN_ID}_03_llm__signals.csv"
print("INPUT:", INPUT_CSV)
print("OUTPUT:", OUTPUT_CSV)

# LLM setup
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_client = None
if OPENAI_API_KEY:
    try:
        import openai
        _client = openai.OpenAI()
        print("DeepTech LLM: using OpenAI GPT-4o-mini")
    except ImportError:
        print("WARNING: openai package not installed.")
else:
    print("WARNING: OPENAI_API_KEY not set. DeepTech scoring will be skipped.")

SYSTEM_PROMPT = """You are classifying startups. Return ONLY a JSON object, no other text.

Deeptech = technology that requires significant R&D, scientific breakthroughs, or hardware innovation.
- 0: Pure consumer app, marketplace, or media (Uber, Airbnb)
- 2: SaaS with standard engineering (CRM, analytics dashboards)
- 4: Applied AI/ML on existing data (NLP tools, recommendation engines)
- 6: Novel AI architectures, advanced sensors, or medical devices
- 8: Biotech, quantum computing, novel materials, robotics with R&D
- 10: Fundamental science breakthrough (fusion energy, gene therapy, quantum hardware)

Sector categories:
- devtools: developer tools, infrastructure, APIs, open-source, cloud platforms
- enterprise: B2B SaaS, fintech, healthcare IT, legal tech, HR tech
- consumer: food, fashion, beauty, retail, marketplace, social, media, gaming
- deeptech: biotech, pharma, hardware, robotics, quantum, advanced materials, energy
- other: anything that does not clearly fit the above"""

USER_PROMPT_TEMPLATE = """Classify this startup:

Company: {name}
Description: {description}
Industries: {industries}

Also assess the company name. Use this DECISION TREE in order — stop at the first match:

STEP 1: Is the name multi-word (2+ words, possibly with "&" or hyphens)?
  → YES → name_distinctive = 1. STOP.
  Examples: "Dharma Platform"=1, "Elevated Returns"=1, "Lyra Health"=1, "Kettle & Fire"=1,
            "Crypt TV"=1, "Dog Haus"=1, "Human Inc"=1 (the "Inc" counts as a word).

STEP 2: Is the name a single token that is NOT a standard English dictionary word?
  (Made-up words, portmanteaus, misspellings, technical coinages, foreign words,
   names with numbers/odd capitalization all qualify as "not a dictionary word".)
  → YES → name_distinctive = 1. STOP.
  Examples: "Headout"=1 (portmanteau), "Atomist"=1, "Cogniac"=1, "EnsoData"=1,
            "LoftSmart"=1, "FeedMob"=1, "Syntiant"=1, "b8ta"=1, "boostr"=1,
            "Zendesk"=1, "Airbnb"=1, "Clearbit"=1, "Gencove"=1.

STEP 3: The name IS a single standard English dictionary word.
  Is this name a MEGA-BRAND that dominates Google results for that word
  (household name, billions in valuation, everyone would recognize)?
  → YES → name_distinctive = 1. STOP.
  Examples (only these kinds of huge brands): "Notion"=1, "Peloton"=1, "Slack"=1,
            "Square"=1, "Stripe"=1, "Calm"=1, "Uber"=1.

STEP 4: Single dictionary word, NOT a mega-brand.
  → name_distinctive = 0.
  Examples: "Drift"=0, "Grove"=0, "Fig"=0, "Headset"=0, "Manta"=0, "Ethic"=0,
            "Pilot"=0, "Ramp"=0, "Haus"=0 (common German word), "Fauna"=0 (Latin/English word),
            "Human"=0, "Kong"=0.

Be honest about STEP 3 — very few startups qualify as mega-brands. If you haven't heard
of the company or it's only known within a niche, it's NOT a mega-brand → step 4 → 0.

Return JSON: {{"deeptech_score": <integer 0-10>, "startup_sector": "<devtools|enterprise|consumer|deeptech|other>", "name_distinctive": <0 or 1>, "reasoning": "<one sentence>"}}"""

DELAY = 0.3


def score_deeptech(name, description, industries):
    """Call GPT-4o-mini to score deeptech + sector. Returns (score, sector, name_distinctive, reasoning)."""
    if _client is None:
        return None, None, 1, "no_llm"

    desc = (description or "").strip()[:500]
    ind = (industries or "").strip()[:200]
    if not desc and not ind:
        return None, None, 1, "no_description"

    prompt = USER_PROMPT_TEMPLATE.format(name=name, description=desc, industries=ind)
    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=100,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content if response.choices else ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        parsed = json.loads(raw)
        score = int(parsed.get("deeptech_score", 0))
        score = max(0, min(10, score))
        sector = str(parsed.get("startup_sector", "other")).strip().lower()
        if sector not in ("devtools", "enterprise", "consumer", "deeptech", "other"):
            sector = "other"
        name_distinctive = int(parsed.get("name_distinctive", 1))
        reasoning = parsed.get("reasoning", "")
        return score, sector, name_distinctive, reasoning
    except json.JSONDecodeError:
        return None, None, 1, "json_parse_error"
    except Exception as e:
        return None, None, 1, f"error: {type(e).__name__}"


df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

df["deeptech_llm_score"] = np.nan
df["startup_sector"] = ""
df["name_distinctive"] = 1
df["deeptech_llm_note"] = ""

# Description column: try multiple possible names
desc_col = None
for candidate in ["Full Description", "Description", "full_description", "description"]:
    if candidate in df.columns:
        desc_col = candidate
        break

ind_col = "Industries" if "Industries" in df.columns else None

from tqdm import tqdm
for i in tqdm(df.index, desc="DeepTech LLM"):
    name = str(df.loc[i, "Organization_Name"]).strip()
    desc = str(df.loc[i, desc_col]).strip() if desc_col else ""
    ind = str(df.loc[i, ind_col]).strip() if ind_col else ""

    if desc in ("nan", "None", ""):
        desc = ""
    if ind in ("nan", "None", ""):
        ind = ""

    score, sector, name_dist, note = score_deeptech(name, desc, ind)
    if score is not None:
        df.at[i, "deeptech_llm_score"] = score
    if sector:
        df.at[i, "startup_sector"] = sector
    df.at[i, "name_distinctive"] = name_dist
    df.at[i, "deeptech_llm_note"] = str(note)
    time.sleep(DELAY)

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
scored = df["deeptech_llm_score"].notna().sum()
print(f"DeepTech LLM: {scored}/{len(df)} companies scored")
if scored > 0:
    print(f"Distribution: mean={df['deeptech_llm_score'].mean():.1f}, median={df['deeptech_llm_score'].median():.0f}")
    print(f"  0-2 (consumer/SaaS): {(df['deeptech_llm_score'] <= 2).sum()}")
    print(f"  3-5 (applied tech):  {((df['deeptech_llm_score'] >= 3) & (df['deeptech_llm_score'] <= 5)).sum()}")
    print(f"  6-8 (deep tech):     {((df['deeptech_llm_score'] >= 6) & (df['deeptech_llm_score'] <= 8)).sum()}")
    print(f"  9-10 (frontier):     {(df['deeptech_llm_score'] >= 9).sum()}")

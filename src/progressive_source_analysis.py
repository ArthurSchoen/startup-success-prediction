#!/usr/bin/env python3
"""
Progressive Source Addition Analysis
-------------------------------------
Start with Form D only, then add each source group one by one.
Measures the incremental AUC contribution of each data source.
Produces:
  - Table of progressive AUC values
  - Figure: progressive_source_addition.png
"""

import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
DATA = BASE / "Pipeline" / "archive" / "build_2993" / "final_dataset_full.csv"
FIGURES = BASE / "Pipeline" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_csv(DATA)
TARGET = "label_B"
y = df[TARGET].values.astype(int)

# Define feature groups by source (order: strongest first based on ablation)
SOURCE_GROUPS = [
    ("Form D", [
        "seriesA_amount_usd_millions", "formd_directors_preA",
        "formd_investor_count_preA", "formd_prior_rounds", "directors_per_round"
    ]),
    ("Crunchbase", [
        "age_at_seriesA_years", "Number of Founders",
        "name_has_tech_suffix", "name_word_count"
    ]),
    ("Wayback Machine", [
        "wayback_snapshots_12m_preA", "has_careers_page_preA",
        "founder_bigtech_flag", "founder_phd_flag", "founder_school_tier",
        "founder_serial_entrepreneur_flag", "founder_prior_exit_flag",
        "founder_features_available", "named_customers_count_preA",
        "product_stage_preA"
    ]),
    ("LLM (deeptech)", ["deeptech_llm_score"]),
    ("Patents", ["patent_count_preA_company"]),
    ("Clinical Trials", ["ct_trials_count_preA"]),
    ("Federal Awards", ["fed_awards_count_preA"]),
    ("YC Alumni", ["yc_alumni_preA"]),
    ("GDELT News", ["gdelt_mentions_3m_preA"]),
    ("EDGAR SEC", ["edgar_filings_preA"]),
    ("Hacker News", ["hn_mention_count_preA"]),
    ("GitHub", ["github_before_seriesA", "backlinks_github_count_preA"]),
    ("Wikipedia", ["wikipedia_before_seriesA"]),
    ("OpenAlex", ["openalex_works_count"]),
    ("Packages", ["pkg_before_seriesA"]),
    ("Composites", ["institutional_footprint", "sector_match", "community_footprint"]),
]

# Verify all features exist
all_feats = [f for _, feats in SOURCE_GROUPS for f in feats]
missing = [f for f in all_feats if f not in df.columns]
if missing:
    print(f"WARNING: Missing features: {missing}")

def cv_auc(features, X_df, y, n_splits=5):
    """5-fold stratified CV AUC for XGBoost on given features."""
    X = X_df[features].fillna(0).values
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true_all, y_score_all = [], []
    for train_idx, val_idx in skf.split(X, y):
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.8, scale_pos_weight=pos_weight,
            eval_metric="logloss", verbosity=0, random_state=42
        )
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[val_idx])[:, 1]
        y_true_all.extend(y[val_idx].tolist())
        y_score_all.extend(probs.tolist())
    return roc_auc_score(np.array(y_true_all), np.array(y_score_all))

# Progressive addition
print("Progressive Source Addition Analysis")
print("=" * 65)
print(f"{'Step':<4} {'Source Added':<20} {'# Features':>10} {'CV AUC':>8} {'Delta':>8}")
print("-" * 65)

results = []
cumulative_features = []

for i, (source_name, features) in enumerate(SOURCE_GROUPS):
    cumulative_features.extend(features)
    auc = cv_auc(cumulative_features, df, y)
    delta = auc - results[-1]["auc"] if results else auc - 0.500
    results.append({
        "step": i + 1,
        "source": source_name,
        "n_features": len(cumulative_features),
        "auc": auc,
        "delta": delta,
    })
    print(f"{i+1:<4} +{source_name:<19} {len(cumulative_features):>10} {auc:>8.3f} {delta:>+8.3f}")

print("=" * 65)

# Also compute: each source ALONE (not cumulative)
print("\nIndividual Source AUC (each source alone):")
print(f"{'Source':<20} {'# Features':>10} {'CV AUC':>8}")
print("-" * 42)
individual_results = []
for source_name, features in SOURCE_GROUPS:
    if len(features) == 0:
        continue
    auc = cv_auc(features, df, y)
    individual_results.append({"source": source_name, "n_features": len(features), "auc": auc})
    print(f"{source_name:<20} {len(features):>10} {auc:>8.3f}")

# ── Figure: Progressive Source Addition ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

sources = [r["source"] for r in results]
aucs = [r["auc"] for r in results]
deltas = [r["delta"] for r in results]

# Bar chart with cumulative AUC
x = np.arange(len(sources))
bars = ax.bar(x, aucs, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.5)

# Color the first bar (Form D) darker
bars[0].set_facecolor("#2c5282")

# Add delta labels on top
for i, (auc_val, delta_val) in enumerate(zip(aucs, deltas)):
    ax.text(i, auc_val + 0.005, f"{auc_val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    if i > 0:
        color = "#2ca02c" if delta_val > 0.001 else ("#d62728" if delta_val < -0.001 else "#888888")
        ax.text(i, auc_val + 0.018, f"({delta_val:+.3f})", ha="center", va="bottom", fontsize=6.5, color=color)

# Reference lines
ax.axhline(0.500, ls="--", color="gray", lw=1, alpha=0.5, label="Random baseline (0.500)")
ax.axhline(aucs[-1], ls=":", color="#C44E52", lw=1, alpha=0.5, label=f"Full model ({aucs[-1]:.3f})")

ax.set_xticks(x)
ax.set_xticklabels(sources, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Cross-Validated AUC", fontsize=11)
ax.set_title("Progressive Model Building: Adding Sources One by One\n"
             "Each bar shows cumulative AUC after adding that source", fontsize=11)
ax.set_ylim([0.48, max(aucs) + 0.04])
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
out = FIGURES / "progressive_source_addition.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nSaved: {out}")

# ── Figure 2: Individual source AUC (each alone) ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ind_sources = [r["source"] for r in individual_results]
ind_aucs = [r["auc"] for r in individual_results]

# Sort by AUC descending
sorted_idx = np.argsort(ind_aucs)[::-1]
ind_sources_sorted = [ind_sources[i] for i in sorted_idx]
ind_aucs_sorted = [ind_aucs[i] for i in sorted_idx]

x = np.arange(len(ind_sources_sorted))
colors = ["#2c5282" if auc > 0.55 else "#4C72B0" if auc > 0.52 else "#aec7e8" for auc in ind_aucs_sorted]
bars = ax.bar(x, ind_aucs_sorted, color=colors, alpha=0.85, edgecolor="white")

for i, auc_val in enumerate(ind_aucs_sorted):
    ax.text(i, auc_val + 0.003, f"{auc_val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.axhline(0.500, ls="--", color="gray", lw=1, alpha=0.5, label="Random baseline (0.500)")
ax.set_xticks(x)
ax.set_xticklabels(ind_sources_sorted, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Cross-Validated AUC (source alone)", fontsize=11)
ax.set_title("Individual Source Predictive Power\n"
             "Each bar: XGBoost trained on features from that source only", fontsize=11)
ax.set_ylim([0.45, max(ind_aucs_sorted) + 0.04])
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
out = FIGURES / "individual_source_auc.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# Save results to JSON for thesis reference
results_data = {
    "progressive": results,
    "individual": individual_results,
}
out_json = FIGURES / "progressive_source_results.json"
with open(out_json, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"Saved: {out_json}")

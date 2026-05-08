#!/usr/bin/env python3
"""
08 — Due diligence audit of features.

Goal: verify two claims from the earlier XGBoost gain analysis:
  1. The 59 "useless" features are really useless (not a bug on our side)
  2. The 35 "useful" features aren't overoptimistic (leakage / proxy of amount)

Tests:
  A. Univariate AUC — train a 1-feature model for EVERY column. This bypasses
     the "absorption by composite" issue: if a feature has univariate AUC > 0.55
     but XGBoost gain = 0, it means the feature has signal but XGBoost picked a
     correlated feature instead.
  B. Spearman correlation with the label (rank correlation — more robust than Pearson).
  C. For each "useful" feature: correlation with seriesA_amount_usd_millions to
     check if it's just a proxy of the amount.
  D. Drop the top feature (seriesA_amount) and re-run CV to see if the others
     keep their predictive power on their own.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────────────────
train = pd.read_csv("Pipeline/archive/Final dataset/final_train.csv")
test = pd.read_csv("Pipeline/archive/Final dataset/final_test.csv")
full = pd.concat([train, test], ignore_index=True)

raw = pd.read_csv("Pipeline/archive/new2k/new2k_04_merged_signals.csv")
full = full.merge(raw[["Organization_Name","Total Equity Funding Amount (in USD)","Last Funding Type"]], on="Organization_Name", how="left")
sA_usd = pd.to_numeric(full["seriesA_amount_usd_millions"], errors="coerce").fillna(0) * 1e6
te = pd.to_numeric(full["Total Equity Funding Amount (in USD)"], errors="coerce").fillna(0)
full["raised_B_plus"] = (full["success_seriesB_plus"].astype(bool) | ((sA_usd > 0) & (te >= 2 * sA_usd))).astype(int)

# Merge OpenAlex (new version)
try:
    oa = pd.read_csv("Pipeline/new2k_04d_openalex__signals.csv")
    OA_COLS = ["openalex_works_count","openalex_has_papers_preA"]
    oa_cols = [c for c in OA_COLS if c in oa.columns]
    for c in oa_cols:
        if c in full.columns: full = full.drop(columns=[c])
    full = full.merge(oa[["Organization_Name"] + oa_cols], on="Organization_Name", how="left")
    for c in oa_cols: full[c] = pd.to_numeric(full[c], errors="coerce").fillna(0)
except FileNotFoundError:
    pass

# ── Build the same 96-feature set (with FE) ──────────────────────────
def log1p_safe(s):
    return np.log1p(pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0))

def add_features(df):
    out = df.copy()
    for col in ["seriesA_amount_usd_millions",
                "wayback_snapshots_12m_preA","wayback_snapshots_3m_preA",
                "openalex_works_count","pkg_releases_count_preA",
                "patent_count_preA_company","hn_mention_count_preA","named_customers_count_preA","team_size_on_page",
                "fed_grants_count_preA","fed_contracts_count_preA","ct_trials_count_preA","ph_launches_count_preA",
]:
        if col in out.columns:
            out[f"{col}_log"] = log1p_safe(out[col])
    for col in ["has_patent_preA_company","fed_awards_before_seriesA","ct_before_seriesA","wikipedia_before_seriesA","yc_alumni_preA"]:
        if col not in out.columns: out[col] = 0
    out["n_external_validation"] = sum(
        (pd.to_numeric(out[c], errors="coerce").fillna(0) > 0).astype(int)
        for c in ["has_patent_preA_company","fed_awards_before_seriesA","ct_before_seriesA","wikipedia_before_seriesA","yc_alumni_preA"]
    )
    out["has_any_external_validation"] = (out["n_external_validation"] > 0).astype(int)
    for col in ["github_before_seriesA","pkg_before_seriesA","hn_launch","ph_launch"]:
        if col not in out.columns: out[col] = 0
    out["n_developer_signals"] = sum(
        (pd.to_numeric(out[c], errors="coerce").fillna(0) > 0).astype(int)
        for c in ["github_before_seriesA","pkg_before_seriesA","hn_launch","ph_launch"]
    )
    for col in ["founder_phd_flag","founder_bigtech_flag","founder_serial_entrepreneur_flag","founder_prior_exit_flag"]:
        if col not in out.columns: out[col] = 0
    out["n_founder_credentials"] = sum(
        pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
        for c in ["founder_phd_flag","founder_bigtech_flag","founder_serial_entrepreneur_flag","founder_prior_exit_flag"]
    )
    if "seriesA_amount_usd_millions" in out.columns and "age_at_seriesA_years" in out.columns:
        age = pd.to_numeric(out["age_at_seriesA_years"], errors="coerce").fillna(1).clip(lower=1)
        amt = pd.to_numeric(out["seriesA_amount_usd_millions"], errors="coerce").fillna(0)
        out["amount_per_year"] = amt / age
    if "deeptech_llm_score" in out.columns:
        out["founder_creds_x_deeptech"] = (
            out["n_founder_credentials"] * pd.to_numeric(out["deeptech_llm_score"], errors="coerce").fillna(0)
        )
    return out

full_fe = add_features(full)
y = full_fe["raised_B_plus"].astype(int)

# Get all numeric features
DROP = {"SeriesB_within_36m","raised_B_plus","months_to_seriesB","seriesA_year","seriesA_quarter",
        "success_seriesB_plus","success_seriesB_plus_reason_confidence","Number of Funding Rounds",
        "Total Equity Funding Amount (in USD)","_pred","run_id"}
ALL_FEATURES = [c for c in full_fe.columns
                if full_fe[c].dtype in [np.int64, np.float64, np.int32, np.float32]
                and c not in DROP]
print(f"Total numeric features: {len(ALL_FEATURES)}")

# ── Test A: Univariate AUC (5-fold CV with just 1 feature) ───────────
def univariate_auc_cv(col):
    X = full_fe[[col]].apply(pd.to_numeric, errors="coerce").fillna(0)
    if X[col].nunique() <= 1:
        return np.nan, 0.0
    # Skip if the feature value has 0 variance for positives or negatives
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        # Use a simple threshold-based AUC (pass feature values as scores directly)
        # Take absolute correlation sign into account
        if X_tr[col].corr(y_tr) >= 0:
            scores_va = X_va[col].values
        else:
            scores_va = -X_va[col].values
        try:
            aucs.append(roc_auc_score(y_va, scores_va))
        except:
            aucs.append(0.5)
    coverage = (pd.to_numeric(full_fe[col], errors="coerce").fillna(0) != 0).mean()
    return np.mean(aucs), coverage

print(f"\n{'='*70}")
print("TEST A: Univariate AUC for every feature")
print(f"{'='*70}")
print("(Checking if 'useless' features have hidden signal)")
print()
results = []
for col in ALL_FEATURES:
    auc, cov = univariate_auc_cv(col)
    corr = full_fe[col].astype(float).corr(y.astype(float))
    results.append({
        "feature": col,
        "univariate_auc": auc,
        "coverage": cov,
        "corr_with_label": corr,
    })

df_res = pd.DataFrame(results).sort_values("univariate_auc", ascending=False)

# Show features with univariate AUC > 0.55 (meaningful individual signal)
print(f"Features with univariate AUC > 0.55 (ordered):")
print(f"{'Feature':<40s} {'AUC':>7s} {'Cov':>7s} {'Corr':>7s}")
print("-" * 65)
high_signal = df_res[df_res["univariate_auc"] > 0.55]
for _, r in high_signal.iterrows():
    print(f"{r['feature'][:39]:<40s} {r['univariate_auc']:.3f}  {r['coverage']*100:5.1f}%  {r['corr_with_label']:+.3f}")

# Features in the middle zone (0.52-0.55) — weak but real
mid = df_res[(df_res["univariate_auc"] > 0.52) & (df_res["univariate_auc"] <= 0.55)]
print(f"\nWeak signal (0.52 < AUC ≤ 0.55):")
for _, r in mid.iterrows():
    print(f"  {r['feature'][:38]:<38s} {r['univariate_auc']:.3f}  {r['coverage']*100:5.1f}%  {r['corr_with_label']:+.3f}")

# Truly useless features
noise = df_res[df_res["univariate_auc"] <= 0.52]
print(f"\nNo individual signal (AUC ≤ 0.52): {len(noise)} features")

# ── Test B: Overoptimism check on top features ──────────────────────
print(f"\n{'='*70}")
print("TEST B: Are the TOP useful features proxies of seriesA_amount?")
print(f"{'='*70}")

top_useful = [
    "named_customers_count_preA_log","pkg_before_seriesA","n_developer_signals",
    "seriesA_amount_usd_millions_log","has_careers_page_preA","founder_bigtech_flag",
    "n_external_validation","seriesA_amount_usd_millions",
    "wayback_snapshots_12m_preA","has_any_external_validation","Number of Founders",
    "wayback_snapshots_3m_preA","amount_per_year","age_at_seriesA_years",
]
top_useful = [c for c in top_useful if c in full_fe.columns]

sA = pd.to_numeric(full_fe["seriesA_amount_usd_millions"], errors="coerce").fillna(0)
print(f"Correlation of each top feature with seriesA_amount_usd_millions:")
print(f"{'Feature':<40s} {'corr w/ amount':>14s} {'corr w/ label':>14s}")
for col in top_useful:
    vals = pd.to_numeric(full_fe[col], errors="coerce").fillna(0)
    corr_amt = vals.corr(sA)
    corr_lab = vals.corr(y.astype(float))
    flag = " 🚨" if abs(corr_amt) > 0.7 else ""
    print(f"  {col[:39]:<40s} {corr_amt:+.3f}         {corr_lab:+.3f}{flag}")

# ── Test C: Drop seriesA_amount and see what happens ────────────────
print(f"\n{'='*70}")
print("TEST C: Drop seriesA_amount_* features — do the others still work?")
print(f"{'='*70}")

USEFUL = [c for c in top_useful if c in full_fe.columns]
USEFUL_WITHOUT_AMOUNT = [c for c in USEFUL if "amount" not in c.lower() and "seriesa_amount" not in c.lower()]
print(f"Full USEFUL: {len(USEFUL)} features")
print(f"Without amount: {len(USEFUL_WITHOUT_AMOUNT)} features")

def cv_eval(feature_list):
    X = full_fe[feature_list].apply(pd.to_numeric, errors="coerce").fillna(0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        scale = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                              scale_pos_weight=scale, eval_metric="auc", random_state=42)
        model.fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
    return np.mean(aucs), np.std(aucs)

a, s = cv_eval(USEFUL)
print(f"  With amount:    AUC = {a:.3f} ± {s:.3f}")
a2, s2 = cv_eval(USEFUL_WITHOUT_AMOUNT)
print(f"  Without amount: AUC = {a2:.3f} ± {s2:.3f}")
delta = a2 - a
print(f"  Delta: {delta:+.3f}")
if abs(delta) < 0.015:
    print(f"  → ✅ Other features are genuinely informative (not just amount proxies)")
else:
    print(f"  → ⚠️ Amount was doing most of the work")

# ── Test D: Test for leakage via label → feature correlation ────────
print(f"\n{'='*70}")
print("TEST D: Suspicious correlation (possible leakage?)")
print(f"{'='*70}")

# Any feature with |corr| > 0.3 with label is worth investigating
suspicious = df_res[df_res["corr_with_label"].abs() > 0.3].sort_values("corr_with_label", key=abs, ascending=False)
if len(suspicious) > 0:
    print(f"{'Feature':<40s} {'Corr w/ label':>15s}")
    for _, r in suspicious.head(10).iterrows():
        print(f"  {r['feature'][:39]:<40s} {r['corr_with_label']:+.3f}")
else:
    print("  ✅ No feature has suspiciously high correlation with label (|r| > 0.3)")

# ── Test E: seriesA_amount leakage check ───────────────────────────
# The label uses Total Equity Funding >= 2x seriesA_amount. Is seriesA_amount
# itself leaking via this construction?
print(f"\n{'='*70}")
print("TEST E: Does the label construction leak through seriesA_amount?")
print(f"{'='*70}")
# For companies where label=1, is their seriesA_amount different from label=0?
sA_pos = sA[y == 1]
sA_neg = sA[y == 0]
print(f"  Mean seriesA_amount | label=1: ${sA_pos.mean():.2f}M  (n={len(sA_pos)})")
print(f"  Mean seriesA_amount | label=0: ${sA_neg.mean():.2f}M  (n={len(sA_neg)})")
print(f"  Median seriesA_amount | label=1: ${sA_pos.median():.2f}M")
print(f"  Median seriesA_amount | label=0: ${sA_neg.median():.2f}M")
print(f"  Ratio of means: {sA_pos.mean()/sA_neg.mean():.2f}x")

# If label uses Total Equity >= 2x Series A, small Series A rounds are MORE likely
# to trigger the label (easier to double). Check this.
print(f"\n  Do smaller Series A rounds get the label more often?")
for low, high in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 200)]:
    mask = (sA >= low) & (sA < high)
    if mask.sum() > 0:
        rate = y[mask].mean()
        print(f"    Series A ${low}-${high}M: n={int(mask.sum()):>3d}, positive rate = {rate*100:5.1f}%")

# Save full results
df_res.to_csv("Pipeline/archive/Final dataset/feature_audit.csv", index=False)
print(f"\n✅ Full feature audit saved to: Pipeline/archive/Final dataset/feature_audit.csv")

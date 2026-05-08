#!/usr/bin/env python3
"""
07 — Feature ablation: really test if the 59 "useless" features are useless.

We train 3 models on 5-fold CV and compare:
  A. All 96 features (with FE)
  B. Only 37 "useful" features (gain >= 0.005 from 08)
  C. Only the 59 "useless" features (to see if they have ANY signal)
  D. Just the top 15 most useful features

Also compares two label definitions to show sensitivity.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────────────────
train = pd.read_csv("Pipeline/archive/Final dataset/final_train.csv")
test = pd.read_csv("Pipeline/archive/Final dataset/final_test.csv")
full = pd.concat([train, test], ignore_index=True)

# Build raised_B_plus label
raw = pd.read_csv("Pipeline/archive/new2k/new2k_04_merged_signals.csv")
full = full.merge(raw[["Organization_Name","Total Equity Funding Amount (in USD)","Last Funding Type"]], on="Organization_Name", how="left")
sA_usd = pd.to_numeric(full["seriesA_amount_usd_millions"], errors="coerce").fillna(0) * 1e6
te = pd.to_numeric(full["Total Equity Funding Amount (in USD)"], errors="coerce").fillna(0)
full["raised_B_plus"] = (full["success_seriesB_plus"].astype(bool) | ((sA_usd > 0) & (te >= 2 * sA_usd))).astype(int)
print(f"raised_B_plus: {int(full['raised_B_plus'].sum())}/{len(full)} ({full['raised_B_plus'].mean()*100:.1f}%)")

# Merge new OpenAlex
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

# ── Feature engineering (same as 09) ──────────────────────────────────
def log1p_safe(s):
    return np.log1p(pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0))

def add_features(df):
    out = df.copy()
    SKEWED = ["seriesA_amount_usd_millions",
              "wayback_snapshots_12m_preA","wayback_snapshots_3m_preA",
              "openalex_works_count","pkg_releases_count_preA",
              "patent_count_preA_company","hn_mention_count_preA","named_customers_count_preA","team_size_on_page",
              "fed_grants_count_preA","fed_contracts_count_preA","ct_trials_count_preA","ph_launches_count_preA"]
    for col in SKEWED:
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

# Useful vs useless feature lists (from 08's gain analysis)
USEFUL = [
    "named_customers_count_preA_log","pkg_before_seriesA","n_developer_signals","seriesA_amount_usd_millions_log",
    "has_careers_page_preA","founder_bigtech_flag","n_external_validation","seriesA_amount_usd_millions",
    "founder_creds_x_deeptech","wayback_snapshots_12m_preA","wayback_snapshots_12m_preA_log","has_any_external_validation",
    "Number of Founders","wayback_snapshots_3m_preA","amount_per_year","age_at_seriesA_years","named_customers_count_preA",
    "github_before_seriesA","wayback_snapshots_3m_preA_log","product_stage_preA","deeptech_llm_score",
    "backlinks_github_count_preA","n_founder_credentials","hn_mention_count_preA","pkg_releases_count_preA",
    "founder_prior_exit_flag","team_size_on_page_log","founder_features_available","fed_awards_before_seriesA",
    "team_size_on_page","founder_serial_entrepreneur_flag",
    "founder_phd_flag","hn_mention_count_preA_log","yc_alumni_preA","pkg_releases_count_preA_log",
]
TOP15 = USEFUL[:15]

USELESS = [
    # OpenAlex (was 0 in original, now has 3 hits after fix)
    "openalex_works_count","openalex_has_papers_preA","openalex_works_count_log",
    # Wikipedia (sparse binary only)
    "wikipedia_before_seriesA",
    # Individual sparse features absorbed by composites
    "has_patent_preA_company","patent_count_preA_company","patent_count_preA_company_log",
    "ct_before_seriesA","ct_trials_count_preA","ct_trials_count_preA_log",
    "fed_grants_count_preA","fed_contracts_count_preA","fed_grants_count_preA_log","fed_contracts_count_preA_log",
    "hn_launch","ph_launch","ph_before_seriesA","ph_featured","ph_launches_count_preA","ph_launches_count_preA_log",
    "is_delaware_hq","founder_name_in_company","founder_school_tier",
    "yc_months_before_seriesA",
]

USEFUL = [c for c in USEFUL if c in full_fe.columns]
USELESS = [c for c in USELESS if c in full_fe.columns]
TOP15 = [c for c in TOP15 if c in full_fe.columns]
ALL_FEATURES = [c for c in full_fe.columns if full_fe[c].dtype in [np.int64, np.float64, np.int32, np.float32]]
DROP_COLS = {"SeriesB_within_36m","raised_B_plus","months_to_seriesB","seriesA_year","seriesA_quarter","success_seriesB_plus","Number of Funding Rounds","Total Equity Funding Amount (in USD)","_pred"}
ALL_FEATURES = [c for c in ALL_FEATURES if c not in DROP_COLS]

print(f"\nFeature sets:")
print(f"  ALL: {len(ALL_FEATURES)} features")
print(f"  USEFUL: {len(USEFUL)} features")
print(f"  TOP15: {len(TOP15)} features")
print(f"  USELESS: {len(USELESS)} features")

y = full_fe["raised_B_plus"].astype(int)

def cv_eval(feature_list, label):
    X = full_fe[feature_list].apply(pd.to_numeric, errors="coerce").fillna(0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, aps = [], []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        scale = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                              scale_pos_weight=scale, eval_metric="auc", random_state=42)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, p))
        aps.append(average_precision_score(y_va, p))
    return label, len(feature_list), np.mean(aucs), np.std(aucs), np.mean(aps)

print(f"\n{'='*70}")
print("ABLATION: 5-fold CV with different feature sets")
print(f"{'='*70}")
print(f"{'Feature set':<35s} {'N':>4s}  {'AUC ± std':>15s}  {'AP':>7s}")
print(f"{'-'*35} {'-'*4}  {'-'*15}  {'-'*7}")
results = []
for name, features in [
    ("A. ALL features (96)", ALL_FEATURES),
    ("B. USEFUL only (37)", USEFUL),
    ("C. TOP 15 only", TOP15),
    ("D. USELESS only (59)", USELESS),
]:
    r = cv_eval(features, name)
    results.append(r)
    name, n, auc_m, auc_s, ap_m = r
    print(f"{name:<35s} {n:>4d}  {auc_m:.3f} ± {auc_s:.3f}  {ap_m:.3f}")

# Verdict
print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")
r_all = results[0]; r_useful = results[1]; r_top15 = results[2]; r_useless = results[3]
print(f"Are the 59 'useless' features really useless?")
print(f"  USELESS-only model AUC: {r_useless[2]:.3f} ± {r_useless[3]:.3f}")
print(f"  Random baseline:         0.500")
print(f"  Lift above random:       +{r_useless[2]-0.5:.3f}")
if r_useless[2] < 0.55:
    print(f"  → ✅ CONFIRMED: the 59 features are essentially noise (AUC close to random)")
elif r_useless[2] < 0.58:
    print(f"  → ⚠️ They have SOME signal but very weak")
else:
    print(f"  → 🚨 They actually have signal — not truly useless")
print()
print(f"Can we drop them without losing performance?")
print(f"  ALL (96):       AUC = {r_all[2]:.3f} ± {r_all[3]:.3f}")
print(f"  USEFUL (37):    AUC = {r_useful[2]:.3f} ± {r_useful[3]:.3f}")
delta = r_useful[2] - r_all[2]
print(f"  Delta: {delta:+.3f}")
if delta >= -0.005:
    print(f"  → ✅ YES: dropping useless features doesn't hurt (or slightly helps)")
else:
    print(f"  → ⚠️ There's a small cost ({-delta:.3f}) — some 'useless' features contribute marginally")

print()
print(f"Is TOP 15 enough?")
print(f"  TOP 15:   AUC = {r_top15[2]:.3f} ± {r_top15[3]:.3f}")
print(f"  USEFUL 37:AUC = {r_useful[2]:.3f} ± {r_useful[3]:.3f}")
delta2 = r_top15[2] - r_useful[2]
if delta2 >= -0.01:
    print(f"  → ✅ TOP 15 alone captures the signal ({delta2:+.3f})")
else:
    print(f"  → The other 22 features contribute ({-delta2:.3f})")

# ── Label sensitivity ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print("LABEL SENSITIVITY (ablation on USEFUL feature set)")
print(f"{'='*70}")
# Recompute full table with multiple labels
labels_to_try = {
    "SeriesB_within_36m (old, 13.9%)": full_fe["SeriesB_within_36m"].astype(int),
    "success_seriesB_plus (27.6%)":    full_fe["success_seriesB_plus"].astype(int),
    "raised_B_plus (current, 46.6%)":  full_fe["raised_B_plus"].astype(int),
}
# Add aggressive and generous
te_col = pd.to_numeric(full_fe["Total Equity Funding Amount (in USD)"], errors="coerce").fillna(0)
sA_col = pd.to_numeric(full_fe["seriesA_amount_usd_millions"], errors="coerce").fillna(0) * 1e6
b_plus = full_fe["success_seriesB_plus"].astype(bool)
labels_to_try["aggressive (1.5x, 55.8%)"] = (b_plus | ((sA_col > 0) & (te_col >= 1.5 * sA_col))).astype(int)

X_useful = full_fe[USEFUL].apply(pd.to_numeric, errors="coerce").fillna(0)
for lab_name, y_lab in labels_to_try.items():
    pos_pct = 100 * y_lab.mean()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, va_idx in skf.split(X_useful, y_lab):
        X_tr, X_va = X_useful.iloc[tr_idx], X_useful.iloc[va_idx]
        y_tr, y_va = y_lab.iloc[tr_idx], y_lab.iloc[va_idx]
        scale = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                              scale_pos_weight=scale, eval_metric="auc", random_state=42)
        model.fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
    print(f"  {lab_name:<35s}  AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

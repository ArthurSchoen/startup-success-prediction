#!/usr/bin/env python
# coding: utf-8
"""
06__model_training.py — Model comparison for Series B prediction (pre-Series A signals)

Target: SeriesB_within_36m (1 = raised Series B or higher within 36 months of Series A)

Evaluation: 5-fold stratified cross-validation on train set.
AUC from 5-fold CV provides a reasonable estimate; interpret with bootstrap CIs.

Models compared (literature-aligned — tree ensembles dominate VC prediction papers):
  1. Logistic Regression  L2  (ridge)   — baseline, well-calibrated
  2. Logistic Regression  L1  (lasso)   — sparse, built-in feature selection
  3. Random Forest                       — most cited in startup-success literature
  4. Gradient Boosting  (sklearn GBM)   — standard deeptech benchmark
  5. XGBoost                            — modern benchmark
  6. LightGBM                           — fast, handles sparsity well

Plots saved to Pipeline/figures/:
  01_roc_curves.png          — 5-fold CV ROC curves, all models
  02_pr_curves.png           — Precision-Recall curves (better metric for imbalanced data)
  03_auc_comparison.png      — AUC bar chart with bootstrap CI
  04_feature_importance.png  — Top-15 features per tree model
  05_shap_beeswarm.png       — SHAP values for best model
  06_probability_ranking.png — All companies ranked by predicted probability
  07_test_set_curves.png     — Test-set ROC + PR curves
  08_calibration.png         — Reliability diagrams (5-fold CV + test set)
  09_pipeline_diagram.png    — Data collection pipeline diagram (thesis figure)
  10_test_auc_comparison.png — Test-set AUC bar chart comparison
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance

import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
PIPE    = BASE / "Pipeline"
FIGURES = PIPE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Auto-discover: env var > Pipeline root > archive/Final dataset
_final_override = os.environ.get("PIPELINE_FINAL_DIR", "").strip()
if _final_override:
    FINAL = Path(_final_override)
else:
    # Look for latest final dataset in Pipeline root
    _candidates = sorted(PIPE.glob("*_05_final_dataset.csv"), key=lambda p: p.name, reverse=True)
    if _candidates:
        FINAL = _candidates[0].parent
        _prefix = _candidates[0].name.split("_05_")[0]
    else:
        FINAL = BASE / "Pipeline" / "archive" / "Final dataset"

# ── Load data ─────────────────────────────────────────────────────────────────
# Try to find the train/test files
_train_candidates = sorted(FINAL.glob("*final_train.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
_test_candidates = sorted(FINAL.glob("*final_test.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
_feat_candidates = sorted(FINAL.glob("*feature_columns.json"), key=lambda p: p.stat().st_mtime, reverse=True)

train_df = pd.read_csv(_train_candidates[0] if _train_candidates else FINAL / "final_train.csv")
test_df = pd.read_csv(_test_candidates[0] if _test_candidates else FINAL / "final_test.csv")
feat_cfg = json.loads((_feat_candidates[0] if _feat_candidates else FINAL / "final_feature_columns.json").read_text())

TARGET    = feat_cfg["target"]           # SeriesB_within_36m
ALL_FEATS = feat_cfg["feature_columns"]  # 120 declared features

print(f"Train: {train_df.shape}  |  Test: {test_df.shape}")
print(f"Target: {TARGET}")
print(f"Train positives: {int(train_df[TARGET].sum())}/{len(train_df)}")
print(f"Test  positives: {int(test_df[TARGET].sum())}/{len(test_df)}")

# ── Feature selection ─────────────────────────────────────────────────────────
# Keep numeric features with >= 3 non-zero values in train (minimum signal)
NONZERO_THRESHOLD = 3
CATEGORICAL_COLS  = ["industry_bucket"]

# Anti-leakage safeguard: exclude features that use current/post-Series A data
# These are snapshot values (stars, forks, downloads) queried TODAY, not at Series A date.
# Leakage features are already excluded in step 07.
# This is a minimal safety net for any columns that might slip through.
LEAKAGE_RISK_COLS = {
    "deeptech", "deeptech_score", "sec_match_score",
    # founder_bigtech_flag, founder_phd_flag, founder_serial_entrepreneur_flag,
    # founder_prior_exit_flag are extracted from Wayback snapshots dated before
    # Series A — they are NOT leakage and are kept as valid features.
}

usable_numeric = []
excluded_leakage = 0
for c in ALL_FEATS:
    if c not in train_df.columns:
        continue
    if c in LEAKAGE_RISK_COLS:
        excluded_leakage += 1
        continue
    if train_df[c].dtype == object:
        continue  # handled separately
    nonzero = (train_df[c].fillna(0) != 0).sum()
    if nonzero >= NONZERO_THRESHOLD:
        usable_numeric.append(c)

cat_cols = [c for c in CATEGORICAL_COLS if c in train_df.columns]
FEATURE_COLS = usable_numeric + cat_cols

print(f"\nSelected features: {len(usable_numeric)} numeric + {len(cat_cols)} categorical = {len(FEATURE_COLS)} total")
print(f"  (dropped {excluded_leakage} leakage-risk features, {len(ALL_FEATS) - len(FEATURE_COLS) - excluded_leakage} all-zero or missing)")

# ── Prepare X, y ─────────────────────────────────────────────────────────────
def prepare(df, feature_cols, fit_transformer=None):
    X_raw = df[feature_cols].copy()
    # Fill NaN with 0 (signal absence — already the correct semantics for all signal cols)
    for c in usable_numeric:
        if c in X_raw.columns:
            X_raw[c] = X_raw[c].fillna(0)
    return X_raw

X_train_raw = prepare(train_df, FEATURE_COLS)
X_test_raw  = prepare(test_df,  FEATURE_COLS)
y_train = train_df[TARGET].values.astype(int)
y_test  = test_df[TARGET].values.astype(int)

# Preprocessing: scale numeric, one-hot categorical
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), usable_numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
], remainder="drop")

preprocessor.fit(X_train_raw)
X_train = preprocessor.transform(X_train_raw)
X_test  = preprocessor.transform(X_test_raw)

# Feature names after preprocessing
ohe_names = []
if cat_cols:
    ohe_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
feat_names = usable_numeric + ohe_names

n_features = X_train.shape[1]
print(f"Final feature matrix: {X_train.shape}  (after OHE: {n_features} features)")

# ── Model definitions ─────────────────────────────────────────────────────────
# class_weight='balanced' compensates for class imbalance
# XGBoost: scale_pos_weight = n_negative / n_positive
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

MODELS = {
    "LogReg_L2": LogisticRegression(
        penalty="l2", C=0.1, class_weight="balanced",
        solver="lbfgs", max_iter=1000, random_state=42
    ),
    "LogReg_L1": LogisticRegression(
        penalty="l1", C=0.05, class_weight="balanced",
        solver="liblinear", max_iter=1000, random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, max_depth=3, min_samples_leaf=2,
        class_weight="balanced", random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.8, scale_pos_weight=pos_weight,
        eval_metric="logloss", verbosity=0, random_state=42
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.8, class_weight="balanced",
        verbose=-1, random_state=42
    ),
}

MODEL_COLORS = {
    "LogReg_L2":        "#4C72B0",
    "LogReg_L1":        "#7cb9e8",
    "RandomForest":     "#DD8452",
    "GradientBoosting": "#55A868",
    "XGBoost":          "#C44E52",
    "LightGBM":         "#8172B3",
}

# ── 5-fold CV ────────────────────────────────────────────────────────────────
print(f"\nRunning 5-fold stratified CV ({len(MODELS)} models)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}  # model_name -> {y_true, y_score, auc, ap}
for name, model in MODELS.items():
    y_true_all, y_score_all = [], []
    for train_idx, val_idx in skf.split(X_train, y_train):
        Xtr, Xval = X_train[train_idx], X_train[val_idx]
        ytr, yval = y_train[train_idx], y_train[val_idx]
        # Skip fold if only one class in ytr (can't fit some models)
        if len(np.unique(ytr)) < 2:
            y_true_all.extend(yval.tolist())
            y_score_all.extend([0.5] * len(yval))
            continue
        m = model.__class__(**model.get_params())
        m.fit(Xtr, ytr)
        probs = m.predict_proba(Xval)[:, 1]
        y_true_all.extend(yval.tolist())
        y_score_all.extend(probs.tolist())

    y_true_arr  = np.array(y_true_all)
    y_score_arr = np.array(y_score_all)

    if len(np.unique(y_true_arr)) < 2:
        auc = float("nan")
        ap  = float("nan")
    else:
        auc = roc_auc_score(y_true_arr, y_score_arr)
        ap  = average_precision_score(y_true_arr, y_score_arr)

    cv_results[name] = {
        "y_true":  y_true_arr,
        "y_score": y_score_arr,
        "auc":     auc,
        "ap":      ap,
    }
    print(f"  {name:<20}  ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")

# Bootstrap CI for AUC (1000 resamples of CV predictions)
def bootstrap_auc_ci(y_true, y_score, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    lo = np.percentile(aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(aucs, (1 + ci) / 2 * 100)
    return lo, hi

print("\nBootstrap 95% CI (1000 resamples of CV predictions):")
for name, res in cv_results.items():
    lo, hi = bootstrap_auc_ci(res["y_true"], res["y_score"])
    cv_results[name]["ci_lo"] = lo
    cv_results[name]["ci_hi"] = hi
    print(f"  {name:<20}  [{lo:.3f}, {hi:.3f}]")

# ── Fit final models on full train ────────────────────────────────────────────
final_models = {}
for name, model in MODELS.items():
    m = model.__class__(**model.get_params())
    m.fit(X_train, y_train)
    final_models[name] = m

# ── Plot 1: ROC curves (5-fold CV) ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Random (AUC=0.50)")

for name, res in cv_results.items():
    fpr, tpr, _ = roc_curve(res["y_true"], res["y_score"])
    auc = res["auc"]
    ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2,
            label=f"{name}  (AUC={auc:.2f}  [{res['ci_lo']:.2f}–{res['ci_hi']:.2f}])")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title(f"ROC Curves — 5-fold CV (n={len(y_train)}, {int(y_train.sum())} positives)",
             fontsize=11)
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
fig.tight_layout()
out = FIGURES / "01_roc_curves.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nSaved: {out}")

# ── Plot 2: Precision-Recall curves ──────────────────────────────────────────
# PR is more informative than ROC with severe imbalance (10.5% positive rate)
fig, ax = plt.subplots(figsize=(8, 6))
baseline = y_train.mean()
ax.axhline(baseline, color="k", ls="--", alpha=0.4, lw=1,
           label=f"Random baseline (precision={baseline:.2f})")

for name, res in cv_results.items():
    prec, rec, _ = precision_recall_curve(res["y_true"], res["y_score"])
    ap = res["ap"]
    ax.plot(rec, prec, color=MODEL_COLORS[name], lw=2,
            label=f"{name}  (AP={ap:.2f})")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves — 5-fold CV\n"
             "(Better metric than ROC for imbalanced classes)", fontsize=11)
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
fig.tight_layout()
out = FIGURES / "02_pr_curves.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# ── Plot 3: AUC bar chart ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

names  = list(cv_results.keys())
aucs   = [cv_results[n]["auc"]   for n in names]
aps    = [cv_results[n]["ap"]    for n in names]
ci_lo  = [cv_results[n]["ci_lo"] for n in names]
ci_hi  = [cv_results[n]["ci_hi"] for n in names]
colors = [MODEL_COLORS[n] for n in names]
yerr   = np.array([[a - lo, hi - a] for a, lo, hi in zip(aucs, ci_lo, ci_hi)]).T

x = np.arange(len(names))
axes[0].bar(x, aucs, color=colors, alpha=0.85, yerr=yerr, capsize=4, error_kw={"lw": 1.5})
axes[0].axhline(0.5, ls="--", color="gray", lw=1, label="Random")
axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
axes[0].set_ylabel("ROC-AUC (5-fold CV)"); axes[0].set_title("ROC-AUC Comparison")
axes[0].set_ylim([0, 1]); axes[0].legend(fontsize=8)
for i, v in enumerate(aucs):
    axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

axes[1].bar(x, aps, color=colors, alpha=0.85)
axes[1].axhline(baseline, ls="--", color="gray", lw=1, label=f"Random ({baseline:.2f})")
axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
axes[1].set_ylabel("Average Precision (5-fold CV)"); axes[1].set_title("PR-AUC Comparison")
axes[1].set_ylim([0, 1]); axes[1].legend(fontsize=8)
for i, v in enumerate(aps):
    axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

fig.suptitle(f"Model Comparison — 5-fold CV on Train Set (n={len(y_train)})", fontsize=12, fontweight="bold")
fig.tight_layout()
out = FIGURES / "03_auc_comparison.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# ── Plot 4: Feature importance (tree models) ──────────────────────────────────
tree_models = {k: v for k, v in final_models.items()
               if k in ("RandomForest", "GradientBoosting", "XGBoost", "LightGBM")}
TOP_N = 15

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (name, model) in zip(axes, tree_models.items()):
    importances = model.feature_importances_
    actual_n = min(TOP_N, len(importances))
    idx = np.argsort(importances)[-actual_n:]
    top_names  = [feat_names[i] for i in idx]
    top_values = importances[idx]

    ax.barh(range(actual_n), top_values, color=MODEL_COLORS[name], alpha=0.85)
    ax.set_yticks(range(actual_n))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{name} — Top {actual_n} Features")

fig.suptitle("Feature Importance (Tree Models, trained on full train set)", fontsize=12, fontweight="bold")
fig.tight_layout()
out = FIGURES / "04_feature_importance.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# ── Plot 5: SHAP — best model by AUC ─────────────────────────────────────────
best_name = max(cv_results, key=lambda n: cv_results[n]["auc"])
best_model = final_models[best_name]
print(f"\nBest model by 5-fold CV AUC: {best_name} ({cv_results[best_name]['auc']:.3f})")

try:
    if best_name in ("RandomForest", "GradientBoosting"):
        explainer = shap.TreeExplainer(best_model)
    elif best_name == "XGBoost":
        explainer = shap.TreeExplainer(best_model)
    elif best_name == "LightGBM":
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.LinearExplainer(best_model, X_train, feature_perturbation="interventional")

    shap_values = explainer.shap_values(X_train)

    # For classifiers that return [neg, pos] SHAP values, take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_train,
        feature_names=feat_names,
        show=False, max_display=15, plot_type="dot",
    )
    plt.title(f"SHAP Values — {best_name} (best model)\nRed = pushes prediction toward Series B success",
              fontsize=11)
    plt.tight_layout()
    out = FIGURES / "05_shap_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
except Exception as e:
    print(f"SHAP plot skipped ({e})")

# ── Plot 6: Probability ranking — all companies ───────────────────────────────
# Use best model to score all companies (train + test)
all_df = pd.concat([train_df, test_df], ignore_index=True)
X_all_raw = prepare(all_df, FEATURE_COLS)
X_all = preprocessor.transform(X_all_raw)

probs_all = best_model.predict_proba(X_all)[:, 1]
y_all     = all_df[TARGET].values
split_all = all_df["split"].values if "split" in all_df.columns else ["train"] * len(all_df)

ranking = pd.DataFrame({
    "Company":  all_df["Organization_Name"].values,
    "Prob":     probs_all,
    "Actual":   y_all,
    "Split":    split_all,
}).sort_values("Prob", ascending=False).reset_index(drop=True)

print(f"\nProbability Ranking (all {len(ranking)} companies):")
print(ranking.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
bar_colors = []
for _, row in ranking.iterrows():
    if row["Actual"] == 1:
        bar_colors.append("#2ca02c")     # green = actual success
    elif row["Split"] == "test":
        bar_colors.append("#ff7f0e")     # orange = test set
    else:
        bar_colors.append("#aec7e8")     # light blue = train negative

y_pos = range(len(ranking))
ax.barh(y_pos, ranking["Prob"], color=bar_colors, alpha=0.9, edgecolor="white")
ax.set_yticks(y_pos)
ax.set_yticklabels(ranking["Company"], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Predicted Probability of Series B within 36m", fontsize=11)
ax.set_title(f"Company Rankings — {best_name} (best model)\n"
             "🟢 Actual success  🟠 Test set (unknown outcome at time of A)  🔵 Train negative",
             fontsize=10)
ax.axvline(0.5, ls="--", color="gray", lw=1, alpha=0.6, label="P=0.5 threshold")

# Legend
legend_elements = [
    mpatches.Patch(color="#2ca02c", alpha=0.9, label="Train — actual success (Series B)"),
    mpatches.Patch(color="#aec7e8", alpha=0.9, label="Train — no Series B yet"),
    mpatches.Patch(color="#ff7f0e", alpha=0.9, label="Test set"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
fig.tight_layout()
out = FIGURES / "06_probability_ranking.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"MODEL COMPARISON SUMMARY (5-fold CV, n={len(y_train)} train)")
print("=" * 60)
print(f"{'Model':<22} {'ROC-AUC':>8} {'95% CI':>16} {'PR-AUC':>8}")
print("-" * 60)
for name in sorted(cv_results, key=lambda n: -cv_results[n]["auc"]):
    r = cv_results[name]
    print(f"{name:<22} {r['auc']:>8.3f} [{r['ci_lo']:.3f}–{r['ci_hi']:.3f}] {r['ap']:>8.3f}")
print("=" * 60)
print(f"\nBest model: {best_name} (AUC={cv_results[best_name]['auc']:.3f})")

# ── Test set evaluation (if test has positives) ────────────────────────────────
test_results = {}
if len(y_test) > 0 and len(np.unique(y_test)) >= 2:
    print(f"\n{'─' * 60}")
    print(f"TEST SET EVALUATION (n={len(y_test)}, {int(y_test.sum())} positives)")
    print(f"{'─' * 60}")
    print(f"{'Model':<22} {'Test AUC':>9} {'Test AP':>9}")
    print("-" * 44)
    for name in sorted(cv_results, key=lambda n: -cv_results[n]["auc"]):
        m = final_models[name]
        test_probs = m.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)
        test_ap  = average_precision_score(y_test, test_probs)
        test_results[name] = {"y_score": test_probs, "auc": test_auc, "ap": test_ap}
        print(f"{name:<22} {test_auc:>9.3f} {test_ap:>9.3f}")
    print(f"{'─' * 60}")
else:
    print(f"\n  Test set: {len(y_test)} rows, {int(y_test.sum())} positives → test AUC not computed")

# ── Plot 7: Test-set ROC + PR curves ────────────────────────────────────────
if test_results:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Random (AUC=0.50)")
    for name, res in test_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_score"])
        ax1.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2,
                 label=f"{name}  (AUC={res['auc']:.2f})")
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title(f"ROC Curves — Test Set (n={len(y_test)}, {int(y_test.sum())} positives)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlim([-0.02, 1.02]); ax1.set_ylim([-0.02, 1.02])

    # PR
    test_baseline = y_test.mean()
    ax2.axhline(test_baseline, color="k", ls="--", alpha=0.4, lw=1,
                label=f"Random (precision={test_baseline:.2f})")
    for name, res in test_results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_score"])
        ax2.plot(rec, prec, color=MODEL_COLORS[name], lw=2,
                 label=f"{name}  (AP={res['ap']:.2f})")
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Precision-Recall Curves — Test Set")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlim([-0.02, 1.02]); ax2.set_ylim([-0.02, 1.02])

    fig.suptitle("Held-Out Test Set Evaluation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "07_test_set_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

# ── Plot 8: Calibration plot ────────────────────────────────────────────────
from sklearn.calibration import calibration_curve

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Calibration on 5-fold CV predictions (train)
ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Perfectly calibrated")
for name, res in cv_results.items():
    try:
        n_bins = min(5, max(2, int(len(res["y_true"]) / 10)))
        prob_true, prob_pred = calibration_curve(
            res["y_true"], res["y_score"], n_bins=n_bins, strategy="uniform"
        )
        ax1.plot(prob_pred, prob_true, "o-", color=MODEL_COLORS[name], lw=1.5,
                 markersize=5, label=name)
    except Exception:
        pass
ax1.set_xlabel("Mean predicted probability", fontsize=11)
ax1.set_ylabel("Fraction of positives", fontsize=11)
ax1.set_title(f"Calibration — 5-fold CV (n={len(y_train)})")
ax1.legend(fontsize=8, loc="lower right")
ax1.set_xlim([-0.02, 1.02]); ax1.set_ylim([-0.02, 1.02])

# Calibration on test set
if test_results:
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Perfectly calibrated")
    for name, res in test_results.items():
        try:
            n_bins = min(5, max(2, int(len(y_test) / 10)))
            prob_true, prob_pred = calibration_curve(
                y_test, res["y_score"], n_bins=n_bins, strategy="uniform"
            )
            ax2.plot(prob_pred, prob_true, "o-", color=MODEL_COLORS[name], lw=1.5,
                     markersize=5, label=name)
        except Exception:
            pass
    ax2.set_xlabel("Mean predicted probability", fontsize=11)
    ax2.set_ylabel("Fraction of positives", fontsize=11)
    ax2.set_title(f"Calibration — Test Set (n={len(y_test)})")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_xlim([-0.02, 1.02]); ax2.set_ylim([-0.02, 1.02])
else:
    ax2.text(0.5, 0.5, "No test positives\n(calibration undefined)",
             ha="center", va="center", fontsize=12, transform=ax2.transAxes)
    ax2.set_title("Calibration — Test Set (N/A)")

fig.suptitle("Reliability Diagrams (Calibration)", fontsize=13, fontweight="bold")
fig.tight_layout()
out = FIGURES / "08_calibration.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# ── Plot 9: Data Collection Pipeline Diagram ────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
ax.text(8, 9.6, "Data Collection & Feature Engineering Pipeline",
        ha="center", va="center", fontsize=16, fontweight="bold",
        family="sans-serif")
ax.text(8, 9.25, "Can Public Data Predict Who Gets Series B?",
        ha="center", va="center", fontsize=11, fontstyle="italic", color="#555555")

# --- Input sources (left column) ---
sources = [
    ("Crunchbase",          "Company profiles,\nfunding rounds",    "#1f77b4"),
    ("SEC Form D",          "Regulatory filings,\nSeries A verification", "#ff7f0e"),
    ("Wayback Machine",     "Historical snapshots,\nweb presence",  "#2ca02c"),
    ("GitHub API",          "Repos, stars, commits,\ncontributors", "#d62728"),
    ("Wikipedia",           "Page existence,\nedit history",        "#8c564b"),
    ("Hacker News",         "Launches, upvotes,\ncomments",         "#e377c2"),
    ("Product Hunt",        "Product launches,\nmaker reputation",  "#9467bd"),
    ("OpenAlex",            "Academic papers,\ncitations, h-index", "#bcbd22"),
    ("Package Registries",  "PyPI/npm releases,\ndownloads",        "#17becf"),
    ("EDGAR / SEC",         "Filing counts,\ncorporate filings",    "#17becf"),
    ("Federal Awards",      "NIH, NSF, DARPA\ngrants & contracts",  "#bcbd22"),
    ("Clinical Trials",     "ClinicalTrials.gov\nregistrations",    "#aec7e8"),
    ("Y Combinator",        "Accelerator alumni,\nbatch matching",  "#e377c2"),
    ("GDELT News",          "Media mentions,\ntone analysis",       "#7f7f7f"),
]

box_h = 0.48
y_start = 8.5
gap = 0.53
src_x = 1.2

for i, (name, desc, color) in enumerate(sources):
    y = y_start - i * gap
    rect = plt.Rectangle((src_x - 1.1, y - box_h/2), 2.2, box_h,
                          facecolor=color, alpha=0.15, edgecolor=color, lw=1.5,
                          transform=ax.transData, zorder=2)
    ax.add_patch(rect)
    ax.text(src_x, y + 0.08, name, ha="center", va="center",
            fontsize=8, fontweight="bold", color=color, zorder=3)
    ax.text(src_x, y - 0.18, desc, ha="center", va="center",
            fontsize=5.5, color="#444444", zorder=3)

# --- Central processing boxes ---
proc_x = 5.8
proc_items = [
    ("01  SEC Matching",      "Match companies to\nForm D filings",  8.3),
    ("02  Feature Cleaning",  "Core features,\ndeeptech flag",       7.3),
    ("03  Wayback Extract",   "HTML → text\nfeatures",               6.3),
    ("05  LLM Classify",      "GPT-4o-mini\ndeeptech score",         5.3),
    ("06  Parallel Signals",  "13 signals queried\nconcurrently",      4.1),
    ("07  Final Dataset",     "Merge, label,\ntemporal split",        2.9),
]

for label, desc, y in proc_items:
    rect = plt.Rectangle((proc_x - 1.3, y - 0.35), 2.6, 0.7,
                          facecolor="#f0f0f0", edgecolor="#333333", lw=1.5,
                          transform=ax.transData, zorder=2)
    ax.add_patch(rect)
    ax.text(proc_x, y + 0.08, label, ha="center", va="center",
            fontsize=8, fontweight="bold", zorder=3)
    ax.text(proc_x, y - 0.18, desc, ha="center", va="center",
            fontsize=6, color="#555555", zorder=3)

# Arrows: sources → processing
for i in range(len(sources)):
    y_src = y_start - i * gap
    if i < 2:
        y_tgt = 8.3
    elif i == 2:
        y_tgt = 6.3
    else:
        y_tgt = 4.1
    ax.annotate("", xy=(proc_x - 1.3, y_tgt), xytext=(src_x + 1.1, y_src),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8),
                zorder=1)

# Vertical arrows between processing steps
proc_ys = [8.3, 7.3, 6.3, 5.3, 4.1, 2.9]
for i in range(len(proc_ys) - 1):
    ax.annotate("", xy=(proc_x, proc_ys[i+1] + 0.35),
                xytext=(proc_x, proc_ys[i] - 0.35),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
                zorder=1)

# --- Right side: Model training & output ---
model_x = 10.5
# Final dataset box
rect = plt.Rectangle((model_x - 1.5, 2.55), 3.0, 0.7,
                      facecolor="#e8f4e8", edgecolor="#2ca02c", lw=2,
                      transform=ax.transData, zorder=2)
ax.add_patch(rect)
ax.text(model_x, 2.98, "Final Dataset", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#2ca02c", zorder=3)
ax.text(model_x, 2.72, "~15K startups", ha="center", va="center",
        fontsize=7, color="#555555", zorder=3)

# Arrow from step 07 to final dataset
ax.annotate("", xy=(model_x - 1.5, 2.9), xytext=(proc_x + 1.3, 2.9),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5), zorder=1)

# Model comparison box
rect = plt.Rectangle((model_x - 1.5, 1.3), 3.0, 0.9,
                      facecolor="#fff3e0", edgecolor="#ff7f0e", lw=2,
                      transform=ax.transData, zorder=2)
ax.add_patch(rect)
ax.text(model_x, 1.9, "06  Model Training", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#e65100", zorder=3)
ax.text(model_x, 1.6, "LogReg · RF · GBM\nXGBoost · LightGBM", ha="center", va="center",
        fontsize=7, color="#555555", zorder=3)

ax.annotate("", xy=(model_x, 2.2), xytext=(model_x, 2.55),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5), zorder=1)

# Outputs
out_items = [
    ("ROC / PR Curves", 8.5),
    ("Feature Importance", 7.8),
    ("SHAP Explainability", 7.1),
    ("Calibration Plots", 6.4),
    ("Company Rankings", 5.7),
]
out_x = 13.5

rect_bg = plt.Rectangle((out_x - 1.3, 5.2), 2.6, 3.8,
                          facecolor="#f5f0ff", edgecolor="#8172B3", lw=1.5,
                          transform=ax.transData, zorder=1)
ax.add_patch(rect_bg)
ax.text(out_x, 8.85, "Outputs", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#8172B3", zorder=3)

for label, y in out_items:
    ax.text(out_x, y, f"▸ {label}", ha="center", va="center",
            fontsize=8, color="#444444", zorder=3)

# Arrow from model to outputs
ax.annotate("", xy=(out_x - 1.3, 7.0), xytext=(model_x + 1.5, 1.75),
            arrowprops=dict(arrowstyle="->", color="#8172B3", lw=1.5,
                            connectionstyle="arc3,rad=-0.3"), zorder=1)

# Key stats box
rect = plt.Rectangle((model_x - 1.5, 0.15), 5.3, 0.85,
                      facecolor="#fafafa", edgecolor="#cccccc", lw=1,
                      transform=ax.transData, zorder=2)
ax.add_patch(rect)
ax.text(model_x + 1.15, 0.72, "Key Numbers", ha="center", fontsize=8,
        fontweight="bold", color="#333333", zorder=3)
ax.text(model_x + 1.15, 0.38,
        "13 data sources  ·  ~15K startups  ·  Temporal train/test split  ·  SHAP interpretability",
        ha="center", fontsize=7, color="#666666", zorder=3)

fig.tight_layout()
out = FIGURES / "09_pipeline_diagram.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Plot 10: Test-set AUC comparison bar chart ──────────────────────────────
if test_results:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names_sorted = sorted(test_results, key=lambda n: -test_results[n]["auc"])
    t_aucs  = [test_results[n]["auc"] for n in names_sorted]
    t_aps   = [test_results[n]["ap"]  for n in names_sorted]
    t_colors = [MODEL_COLORS[n] for n in names_sorted]
    x = np.arange(len(names_sorted))

    ax1.bar(x, t_aucs, color=t_colors, alpha=0.85)
    ax1.axhline(0.5, ls="--", color="gray", lw=1, label="Random")
    ax1.set_xticks(x); ax1.set_xticklabels(names_sorted, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("ROC-AUC"); ax1.set_title("ROC-AUC — Test Set")
    ax1.set_ylim([0, 1]); ax1.legend(fontsize=8)
    for i, v in enumerate(t_aucs):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    test_baseline = y_test.mean()
    ax2.bar(x, t_aps, color=t_colors, alpha=0.85)
    ax2.axhline(test_baseline, ls="--", color="gray", lw=1, label=f"Random ({test_baseline:.2f})")
    ax2.set_xticks(x); ax2.set_xticklabels(names_sorted, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Average Precision"); ax2.set_title("PR-AUC — Test Set")
    ax2.set_ylim([0, 1]); ax2.legend(fontsize=8)
    for i, v in enumerate(t_aps):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle(f"Model Comparison — Test Set (n={len(y_test)}, {int(y_test.sum())} positives)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "10_test_auc_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

print(f"\n  CAVEATS:")
print(f"   - n={len(y_train)} train, {int(y_train.sum())} positives ({y_train.mean():.1%} positive rate)")
if len(y_test) > 0:
    print(f"   - n={len(y_test)} test, {int(y_test.sum())} positives ({y_test.mean():.1%} positive rate)")
print(f"   - Check bootstrap CIs for overlap — indicates whether model differences are significant")
print(f"   - AUC > 0.5 is encouraging; inspect calibration and SHAP for actionable insights")
print(f"\nFigures saved to: {FIGURES}")

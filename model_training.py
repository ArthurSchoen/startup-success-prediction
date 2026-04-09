#!/usr/bin/env python
# coding: utf-8
"""
model_training.py — Model Comparison for Series B Prediction (Pre-Series A Signals)

Target: SeriesB_within_36m (1 = raised Series B+ within 36 months of Series A)

HARD CONSTRAINTS:
  - Train: 19 samples, 2 positives (10.5%) — extremely imbalanced & tiny
  - Test: 6 samples, 0 positives → test AUC undefined; shown as probability rankings
  - Evaluation: Leave-One-Out CV (LOO-CV) — the only valid strategy at n=19
  - AUC from LOO-CV is coarse (19 prediction points; interpret directionally)

Models compared (literature-aligned — tree ensembles dominate VC prediction papers):
  1. Logistic Regression L2 (Ridge)   — baseline, well-calibrated
  2. Logistic Regression L1 (Lasso)   — sparse, built-in feature selection
  3. Random Forest                     — most cited in startup-success literature
  4. Gradient Boosting (sklearn GBM)   — standard deeptech benchmark
  5. XGBoost                           — modern benchmark with native imbalance handling
  6. LightGBM                          — fast, handles sparsity well

Author: Arthur Schoen — MIT / HEC Paris
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, average_precision_score

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
FINAL   = BASE / "Pipeline" / "archive" / "Final dataset"
FIGURES = BASE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load Data ────────────────────────────────────────────────────────────────
def load_data(data_dir: Path = FINAL):
    """Load train/test CSVs and feature configuration."""
    train_df = pd.read_csv(data_dir / "final_train.csv")
    test_df  = pd.read_csv(data_dir / "final_test.csv")
    feat_cfg = json.loads((data_dir / "final_feature_columns.json").read_text())

    target    = feat_cfg["target"]           # SeriesB_within_36m
    all_feats = feat_cfg["feature_columns"]  # ~120 declared features

    print(f"Train: {train_df.shape}  |  Test: {test_df.shape}")
    print(f"Target: {target}")
    print(f"Train positives: {int(train_df[target].sum())}/{len(train_df)}")
    print(f"Test  positives: {int(test_df[target].sum())}/{len(test_df)}")
    return train_df, test_df, target, all_feats


# ── Feature Selection ────────────────────────────────────────────────────────
# Minimum non-zero values to keep a feature (avoids all-zero noise columns)
NONZERO_THRESHOLD = 3
CATEGORICAL_COLS  = ["industry_bucket"]

# Anti-leakage safety net (already filtered in feature_engineering.py)
LEAKAGE_RISK_COLS = {
    "deeptech", "deeptech_score", "sec_match_score",
    "founder_bigtech_flag", "founder_phd_flag",
    "founder_serial_entrepreneur_flag", "founder_prior_exit_flag",
}


def select_usable_features(train_df, all_feats):
    """Keep numeric features with >= NONZERO_THRESHOLD non-zero values in train."""
    usable_numeric = []
    for c in all_feats:
        if c not in train_df.columns or c in LEAKAGE_RISK_COLS:
            continue
        if train_df[c].dtype == object:
            continue
        nonzero = (train_df[c].fillna(0) != 0).sum()
        if nonzero >= NONZERO_THRESHOLD:
            usable_numeric.append(c)

    cat_cols = [c for c in CATEGORICAL_COLS if c in train_df.columns]
    feature_cols = usable_numeric + cat_cols
    print(f"Selected features: {len(usable_numeric)} numeric + {len(cat_cols)} categorical")
    return usable_numeric, cat_cols, feature_cols


# ── Preprocessing ────────────────────────────────────────────────────────────
def prepare_matrices(train_df, test_df, usable_numeric, cat_cols, feature_cols, target):
    """Build preprocessed feature matrices (scale numeric, one-hot categorical)."""
    def fill_na(df):
        X = df[feature_cols].copy()
        for c in usable_numeric:
            if c in X.columns:
                X[c] = X[c].fillna(0)
        return X

    X_train_raw = fill_na(train_df)
    X_test_raw  = fill_na(test_df)
    y_train = train_df[target].values.astype(int)
    y_test  = test_df[target].values.astype(int)

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

    print(f"Feature matrix: {X_train.shape} (after OHE: {len(feat_names)} features)")
    return X_train, X_test, y_train, y_test, feat_names, preprocessor


# ── Model Definitions ────────────────────────────────────────────────────────
def get_models(y_train):
    """
    Define 6 models spanning the complexity-interpretability spectrum.

    Class weight balancing compensates for extreme imbalance (2/19 positives):
      - LogReg/RF/LightGBM: class_weight='balanced' (auto neg×0.56, pos×4.75)
      - XGBoost: scale_pos_weight = n_negative/n_positive ≈ 8.5
    """
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models = {
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
    return models


MODEL_COLORS = {
    "LogReg_L2":        "#4C72B0",
    "LogReg_L1":        "#7cb9e8",
    "RandomForest":     "#DD8452",
    "GradientBoosting": "#55A868",
    "XGBoost":          "#C44E52",
    "LightGBM":         "#8172B3",
}


# ── Leave-One-Out Cross-Validation ──────────────────────────────────────────
def run_loo_cv(models, X_train, y_train):
    """
    Evaluate all models using Leave-One-Out CV.

    Why LOO-CV? With n=19 (2 positives), any K-fold split leaves too few
    samples per fold. LOO-CV:
      - Uses 18/19 samples for training each fold
      - Produces 19 prediction points for curve estimation
      - Is the only unbiased strategy at this sample size

    Returns dict: model_name -> {y_true, y_score, auc, ap, ci_lo, ci_hi}
    """
    print(f"\nRunning LOO-CV ({len(X_train)} folds x {len(models)} models)...")
    loo = LeaveOneOut()
    results = {}

    for name, model in models.items():
        y_true_all, y_score_all = [], []
        for train_idx, val_idx in loo.split(X_train):
            Xtr, Xval = X_train[train_idx], X_train[val_idx]
            ytr = y_train[train_idx]

            if len(np.unique(ytr)) < 2:
                y_true_all.append(y_train[val_idx][0])
                y_score_all.append(0.5)
                continue

            m = model.__class__(**model.get_params())
            m.fit(Xtr, ytr)
            prob = m.predict_proba(Xval)[0, 1]
            y_true_all.append(y_train[val_idx][0])
            y_score_all.append(prob)

        y_true_arr  = np.array(y_true_all)
        y_score_arr = np.array(y_score_all)

        if len(np.unique(y_true_arr)) < 2:
            auc, ap = float("nan"), float("nan")
        else:
            auc = roc_auc_score(y_true_arr, y_score_arr)
            ap  = average_precision_score(y_true_arr, y_score_arr)

        # Bootstrap 95% CI (1000 resamples of LOO predictions)
        ci_lo, ci_hi = bootstrap_auc_ci(y_true_arr, y_score_arr)

        results[name] = {
            "y_true": y_true_arr, "y_score": y_score_arr,
            "auc": auc, "ap": ap, "ci_lo": ci_lo, "ci_hi": ci_hi,
        }
        print(f"  {name:<20}  ROC-AUC={auc:.3f}  [{ci_lo:.3f}-{ci_hi:.3f}]  PR-AUC={ap:.3f}")

    return results


def bootstrap_auc_ci(y_true, y_score, n_boot=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for AUC."""
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


def fit_final_models(models, X_train, y_train):
    """Fit all models on the full training set (for SHAP, feature importance, etc.)."""
    final = {}
    for name, model in models.items():
        m = model.__class__(**model.get_params())
        m.fit(X_train, y_train)
        final[name] = m
    return final


# ── Main ─────────────────────────────────────────────────────────────────────
def train_all(data_dir: Path = FINAL):
    """
    Full training pipeline:
      1. Load data
      2. Select features
      3. Preprocess
      4. Define 6 models
      5. LOO-CV evaluation
      6. Fit final models
    """
    train_df, test_df, target, all_feats = load_data(data_dir)
    usable_numeric, cat_cols, feature_cols = select_usable_features(train_df, all_feats)
    X_train, X_test, y_train, y_test, feat_names, preprocessor = prepare_matrices(
        train_df, test_df, usable_numeric, cat_cols, feature_cols, target
    )

    models = get_models(y_train)
    loo_results = run_loo_cv(models, X_train, y_train)
    final_models = fit_final_models(models, X_train, y_train)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"MODEL COMPARISON SUMMARY (LOO-CV, n={len(y_train)} train)")
    print(f"{'=' * 60}")
    print(f"{'Model':<22} {'ROC-AUC':>8} {'95% CI':>16} {'PR-AUC':>8}")
    print("-" * 60)
    for name in sorted(loo_results, key=lambda n: -loo_results[n]["auc"]):
        r = loo_results[name]
        print(f"{name:<22} {r['auc']:>8.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}] {r['ap']:>8.3f}")
    print("=" * 60)

    best = max(loo_results, key=lambda n: loo_results[n]["auc"])
    print(f"\nBest model: {best} (AUC={loo_results[best]['auc']:.3f})")

    return {
        "loo_results": loo_results,
        "final_models": final_models,
        "feat_names": feat_names,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "train_df": train_df, "test_df": test_df,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "target": target,
    }


if __name__ == "__main__":
    train_all()

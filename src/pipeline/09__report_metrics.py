"""
Stage 09: report metrics.

Recomputes, from the saved dataset, every number quoted in the README and
regenerates the ROC and precision-recall figures. Runs in about a minute and
needs no API key.

Inputs
    data/processed/final_dataset_full.csv
    data/processed/feature_config_35.json
Outputs
    reports/metrics.json
    reports/METRICS.md
    reports/figures/roc_curves_v2.png
    reports/figures/pr_curves_v2.png

Target: label_B, any follow-on round after the Series A, no time window
(see "The label" in the README).
Features: the 34 reported in the README, i.e. the 36 listed in
feature_config_35.json minus is_diverse_founded and directors_per_round.

Two evaluation bases are reported and kept apart:
  - full-data 5-fold CV on all 2,993 startups (the basis of the per-source
    AUCs and of precision at 5%),
  - an 80/20 stratified split, with 5-fold CV inside the 2,394 training rows
    and a single score on the 599 held-out rows.
Scaling happens inside each fold (a Pipeline), which fixes the
"scaler fit before the split" limit for the two logistic regressions.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "final_dataset_full.csv"
CONFIG = ROOT / "data" / "processed" / "feature_config_35.json"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SEED = 42
N_BOOT = 1000

df = pd.read_csv(DATA)
cfg = json.load(open(CONFIG))
FEATURES = [f for f in cfg["features"] if f not in ("is_diverse_founded", "directors_per_round")]
y = df["label_B"].values.astype(int)


def matrix(cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values


X = matrix(FEATURES)
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def xgb_model(yy):
    pw = (yy == 0).sum() / max((yy == 1).sum(), 1)
    return xgb.XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
                             scale_pos_weight=pw, eval_metric="logloss", verbosity=0,
                             random_state=SEED)


MODELS = {
    "XGBoost": lambda yy: xgb_model(yy),
    "LightGBM": lambda yy: lgb.LGBMClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                              subsample=0.8, class_weight="balanced", verbose=-1,
                                              random_state=SEED),
    "Gradient boosting": lambda yy: GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                                               learning_rate=0.05, subsample=0.8,
                                                               random_state=SEED),
    "Random forest": lambda yy: RandomForestClassifier(n_estimators=500, max_depth=3,
                                                       min_samples_leaf=2, class_weight="balanced",
                                                       random_state=SEED),
    "Logistic regression (L2)": lambda yy: make_pipeline(
        StandardScaler(), LogisticRegression(penalty="l2", C=0.1, class_weight="balanced",
                                             max_iter=1000, random_state=SEED)),
    "Logistic regression (L1)": lambda yy: make_pipeline(
        StandardScaler(), LogisticRegression(penalty="l1", C=0.05, class_weight="balanced",
                                             solver="liblinear", max_iter=1000, random_state=SEED)),
}


def oof(make, Xm, yy):
    """Out-of-fold scores from 5-fold stratified CV."""
    s = np.zeros(len(yy))
    for a, b in skf.split(Xm, yy):
        m = make(yy[a])
        m.fit(Xm[a], yy[a])
        s[b] = m.predict_proba(Xm[b])[:, 1]
    return s


def precision_at(yy, ss, q):
    k = max(1, int(len(yy) * q))
    top = np.argsort(-ss)[:k]
    return float(yy[top].mean()), int(k), int(yy[top].sum())


def r(x, n=3):
    return float(round(x, n))


out = {"target": "label_B", "n_startups": int(len(y)), "base_rate": r(y.mean(), 4),
       "n_features": len(FEATURES), "split": {"train": int(len(tr)), "test": int(len(te))}}

# 1. Six models on the 80/20 split ---------------------------------------------------------
print("1/5 model comparison")
models, curves = {}, {}
for name, make in MODELS.items():
    s_cv = oof(make, X[tr], y[tr])
    m = make(y[tr])
    m.fit(X[tr], y[tr])
    s_te = m.predict_proba(X[te])[:, 1]
    models[name] = {"cv_auc": r(roc_auc_score(y[tr], s_cv)), "cv_pr_auc": r(average_precision_score(y[tr], s_cv)),
                    "test_auc": r(roc_auc_score(y[te], s_te)), "test_pr_auc": r(average_precision_score(y[te], s_te))}
    curves[name] = s_cv
    if name == "XGBoost":
        xgb_test_scores = s_te
out["models_80_20"] = models
out["precision_at_test"] = {f"top_{int(q * 100)}pct": dict(zip(("precision", "k", "hits"), precision_at(y[te], xgb_test_scores, q)))
                            for q in (0.05, 0.10, 0.20)}

# 2. Figures ---------------------------------------------------------------------------------
print("2/5 figures")
FIGURES.mkdir(parents=True, exist_ok=True)
palette = ["#1E6BB8", "#5E82FF", "#7A9CC6", "#A8A398", "#C2410C", "#E0A43A"]
for kind in ("roc", "pr"):
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=200)
    for (name, s), c in zip(curves.items(), palette):
        if kind == "roc":
            fpr, tpr, _ = roc_curve(y[tr], s)
            ax.plot(fpr, tpr, color=c, lw=2, label=f"{name}  AUC {models[name]['cv_auc']:.3f}")
        else:
            p, rc, _ = precision_recall_curve(y[tr], s)
            ax.plot(rc, p, color=c, lw=2, label=f"{name}  AP {models[name]['cv_pr_auc']:.3f}")
    if kind == "roc":
        ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="Coin flip  AUC 0.500")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves, six models\n5-fold CV on the training set (n = 2,394), label_B")
    else:
        ax.axhline(y[tr].mean(), color="#999999", lw=1, ls="--", label=f"Base rate  {y[tr].mean():.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim(0.4, 1.02)
        ax.set_title("Precision-recall curves, six models\n5-fold CV on the training set (n = 2,394), label_B")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#EEEEEE", lw=1)
    ax.legend(frameon=False, fontsize=9, loc="lower right" if kind == "roc" else "upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{kind}_curves_v2.png")
    plt.close(fig)

# 3. Full-data CV: headline AUC, precision at k, bootstrap -------------------------------------
print("3/5 full-data CV and bootstrap")
s_full = oof(MODELS["XGBoost"], X, y)
out["full_cv_auc"] = r(roc_auc_score(y, s_full))
out["precision_at_full_cv"] = {f"top_{int(q * 100)}pct": dict(zip(("precision", "k", "hits"), precision_at(y, s_full, q)))
                               for q in (0.01, 0.05, 0.10, 0.20)}
rng = np.random.default_rng(SEED)
aucs, p5 = [], []
for _ in range(N_BOOT):
    i = rng.integers(0, len(y), len(y))
    aucs.append(roc_auc_score(y[i], s_full[i]))
    p5.append(precision_at(y[i], s_full[i], 0.05)[0])
out["bootstrap_95ci"] = {"full_cv_auc": [r(v) for v in np.percentile(aucs, [2.5, 97.5])],
                         "full_cv_precision_at_5pct": [r(v) for v in np.percentile(p5, [2.5, 97.5])],
                         "note": "Resamples out-of-fold predictions; does not capture refit variance."}

# 4. Source ablation (35-feature grouping of src/progressive_source_analysis.py) -------------
print("4/5 source ablation")
SOURCE_GROUPS = [
    ("Form D", ["seriesA_amount_usd_millions", "formd_directors_preA", "formd_investor_count_preA",
                "formd_prior_rounds", "directors_per_round"]),
    ("Crunchbase", ["age_at_seriesA_years", "Number of Founders", "name_has_tech_suffix", "name_word_count"]),
    ("Wayback Machine", ["wayback_snapshots_12m_preA", "has_careers_page_preA", "founder_bigtech_flag",
                         "founder_phd_flag", "founder_school_tier", "founder_serial_entrepreneur_flag",
                         "founder_prior_exit_flag", "founder_features_available", "named_customers_count_preA",
                         "product_stage_preA"]),
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
all35 = [f for _, fs in SOURCE_GROUPS for f in fs]


def auc_of(cols, yy=y):
    return r(roc_auc_score(yy, oof(MODELS["XGBoost"], matrix(cols), yy)))


alone = {name: auc_of(fs) for name, fs in SOURCE_GROUPS}
cumulative, acc = [], []
for name, fs in SOURCE_GROUPS:
    acc += fs
    cumulative.append({"added": name, "auc": auc_of(acc)})
full35 = auc_of(all35)
drop_one = {name: r(auc_of([f for f in all35 if f not in fs]) - full35) for name, fs in SOURCE_GROUPS}
out["ablation"] = {"basis": "full-data 5-fold CV, XGBoost, 35 features", "all_sources_auc": full35,
                   "source_alone_auc": alone, "cumulative": cumulative, "drop_one_delta": drop_one}

# 5. Robustness ------------------------------------------------------------------------------
print("5/5 robustness")
no_amount = [f for f in FEATURES if f != "seriesA_amount_usd_millions"]
rob = {"without_seriesA_amount_auc": auc_of(no_amount),
       "seriesA_amount_alone_auc": auc_of(["seriesA_amount_usd_millions"]),
       "formd_without_amount_auc": auc_of(["formd_directors_preA", "formd_investor_count_preA", "formd_prior_rounds"])}
a = pd.to_datetime(df["seriesA_date"], errors="coerce")
b = pd.to_datetime(df["seriesB_date"], errors="coerce")
months = (b - a).dt.days / 30.44
y36 = ((df["label_B"] == 1) & (months <= 36)).astype(int).values
rob["strict_36m_label"] = {"definition": "label_B and a dated follow-on within 36 months of the Series A",
                           "base_rate": r(y36.mean(), 4), "auc": auc_of(FEATURES, y36)}
gap = months[(df["label_B"] == 1) & months.notna()]
rob["months_A_to_followon_among_dated_positives"] = {"median": r(gap.median(), 1),
                                                      "share_within_36m": r((gap <= 36).mean(), 3)}
out["robustness"] = rob

# Which of the four label sources fired (positives can match several)
eq_type = df["Last Equity Funding Type"].fillna("").astype(str)
b_plus = eq_type.isin(["Series B", "Series C", "Series D", "Series E", "Series F", "Series G", "Series H"])
formd_b = df["seriesB_source"].fillna("").eq("formd")
total_eq = pd.to_numeric(df["Total Equity Funding Amount (in USD)"], errors="coerce").fillna(0)
sa = pd.to_numeric(df["seriesA_amount_usd_millions"], errors="coerce").fillna(0) * 1e6
equity_2x = (sa > 0) & (total_eq > 2 * sa)
pos = df["label_B"] == 1
out["label_sources"] = {"positives": int(pos.sum()),
                        "crunchbase_series_b_or_later": int((pos & b_plus).sum()),
                        "later_formd_filing": int((pos & formd_b).sum()),
                        "only_equity_or_unknown_series_rules": int((pos & ~b_plus & ~formd_b).sum()),
                        "note": "The last count covers positives found only by the 'Venture, unknown series' "
                                "or 'total equity > 2x Series A' rules."}

REPORTS.mkdir(exist_ok=True)
(REPORTS / "metrics.json").write_text(json.dumps(out, indent=2))

# Markdown report ----------------------------------------------------------------------------
L = ["# Metrics", "",
     "Generated by `src/pipeline/09__report_metrics.py` from `data/processed/final_dataset_full.csv`. "
     "Target `label_B` (any follow-on round after the Series A, no time window), "
     f"{out['n_startups']:,} startups, base rate {out['base_rate']:.1%}, {out['n_features']} features.", "",
     "## Models (80/20 stratified split, seed 42)", "",
     "| Model | CV AUC (train) | CV PR-AUC | Test AUC | Test PR-AUC |", "|---|---|---|---|---|"]
for n, m in models.items():
    L.append(f"| {n} | {m['cv_auc']:.3f} | {m['cv_pr_auc']:.3f} | {m['test_auc']:.3f} | {m['test_pr_auc']:.3f} |")
L += ["", "## Precision among the highest-ranked startups", "",
      f"Full-data 5-fold CV, XGBoost. AUC {out['full_cv_auc']:.3f} "
      f"(bootstrap 95% CI {out['bootstrap_95ci']['full_cv_auc'][0]:.3f} to {out['bootstrap_95ci']['full_cv_auc'][1]:.3f}).", "",
      "| Group | Startups | Raised again | Precision |", "|---|---|---|---|"]
for k, v in out["precision_at_full_cv"].items():
    L.append(f"| {k.replace('top_', 'Top ').replace('pct', '%')} | {v['k']} | {v['hits']} | {v['precision']:.1%} |")
L.append(f"| Everyone | {len(y)} | {int(y.sum())} | {y.mean():.1%} |")
L += ["", "Held-out test set (599 startups): " + ", ".join(
    f"{k.replace('top_', 'top ').replace('pct', '%')} {v['precision']:.1%} ({v['hits']}/{v['k']})"
    for k, v in out["precision_at_test"].items()) + ".", "",
      "## Sources", "", f"Full-data 5-fold CV, XGBoost, 35 features. All sources: AUC {full35:.3f}.", "",
      "| Source | Alone | Cumulative, in this order | Change when removed |", "|---|---|---|---|"]
for (name, _), c in zip(SOURCE_GROUPS, cumulative):
    L.append(f"| {name} | {alone[name]:.3f} | {c['auc']:.3f} | {drop_one[name]:+.3f} |")
L += ["", "## Robustness", "",
      f"- Without the Series A amount: AUC {rob['without_seriesA_amount_auc']:.3f}. The amount alone: {rob['seriesA_amount_alone_auc']:.3f}. "
      f"Form D without the amount: {rob['formd_without_amount_auc']:.3f}.",
      f"- Strict 36-month label ({rob['strict_36m_label']['definition']}): base rate {rob['strict_36m_label']['base_rate']:.1%}, "
      f"AUC {rob['strict_36m_label']['auc']:.3f}.",
      f"- Among positives with a dated follow-on, the median gap from Series A is "
      f"{rob['months_A_to_followon_among_dated_positives']['median']} months; "
      f"{rob['months_A_to_followon_among_dated_positives']['share_within_36m']:.0%} fall within 36 months.",
      f"- Label sources among {out['label_sources']['positives']:,} positives: Crunchbase Series B or later "
      f"{out['label_sources']['crunchbase_series_b_or_later']:,}; later Form D filing {out['label_sources']['later_formd_filing']:,}; "
      f"found only by the unknown-series or equity rules {out['label_sources']['only_equity_or_unknown_series_rules']:,}.", ""]
(REPORTS / "METRICS.md").write_text("\n".join(L))
print("wrote reports/metrics.json, reports/METRICS.md and two figures")

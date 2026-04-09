#!/usr/bin/env python
# coding: utf-8
"""
evaluation.py — Visualization & Evaluation for Startup Success Prediction

Generates all evaluation plots from trained models:
  01_roc_curves.png          — LOO-CV ROC curves with 95% CI
  02_pr_curves.png           — Precision-Recall curves (primary metric for imbalanced data)
  03_auc_comparison.png      — AUC bar chart comparison
  04_feature_importance.png  — Top-15 features per tree model
  05_shap_beeswarm.png       — SHAP values for best model
  06_probability_ranking.png — All companies ranked by predicted probability
  08_calibration.png         — Reliability diagrams

Author: Arthur Schoen — MIT / HEC Paris
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
import shap

from model_training import MODEL_COLORS, train_all, FIGURES

# ── Plot 1: ROC Curves (LOO-CV) ─────────────────────────────────────────────
def plot_roc_curves(loo_results, y_train, output_dir=FIGURES):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Random (AUC=0.50)")

    for name, res in loo_results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_score"])
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2,
                label=f"{name}  (AUC={res['auc']:.2f}  [{res['ci_lo']:.2f}-{res['ci_hi']:.2f}])")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — LOO-CV (n={len(y_train)}, {int(y_train.sum())} positives)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(output_dir / "01_roc_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved: 01_roc_curves.png")


# ── Plot 2: Precision-Recall Curves ─────────────────────────────────────────
def plot_pr_curves(loo_results, y_train, output_dir=FIGURES):
    """PR curves are more informative than ROC for severe imbalance (10.5% positive)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_train.mean()
    ax.axhline(baseline, color="k", ls="--", alpha=0.4, lw=1,
               label=f"Random baseline (precision={baseline:.2f})")

    for name, res in loo_results.items():
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_score"])
        ax.plot(rec, prec, color=MODEL_COLORS[name], lw=2,
                label=f"{name}  (AP={res['ap']:.2f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — LOO-CV\n"
                 "(Better metric than ROC for imbalanced classes)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(output_dir / "02_pr_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved: 02_pr_curves.png")


# ── Plot 3: AUC Comparison Bar Chart ────────────────────────────────────────
def plot_auc_comparison(loo_results, y_train, output_dir=FIGURES):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names  = list(loo_results.keys())
    aucs   = [loo_results[n]["auc"]   for n in names]
    aps    = [loo_results[n]["ap"]    for n in names]
    ci_lo  = [loo_results[n]["ci_lo"] for n in names]
    ci_hi  = [loo_results[n]["ci_hi"] for n in names]
    colors = [MODEL_COLORS[n] for n in names]
    yerr   = np.array([[a - lo, hi - a] for a, lo, hi in zip(aucs, ci_lo, ci_hi)]).T

    x = np.arange(len(names))

    # ROC-AUC
    axes[0].bar(x, aucs, color=colors, alpha=0.85, yerr=yerr, capsize=4)
    axes[0].axhline(0.5, ls="--", color="gray", lw=1, label="Random")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("ROC-AUC (LOO-CV)"); axes[0].set_title("ROC-AUC Comparison")
    axes[0].set_ylim([0, 1]); axes[0].legend(fontsize=8)
    for i, v in enumerate(aucs):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    # PR-AUC
    baseline = y_train.mean()
    axes[1].bar(x, aps, color=colors, alpha=0.85)
    axes[1].axhline(baseline, ls="--", color="gray", lw=1, label=f"Random ({baseline:.2f})")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Average Precision (LOO-CV)"); axes[1].set_title("PR-AUC Comparison")
    axes[1].set_ylim([0, 1]); axes[1].legend(fontsize=8)
    for i, v in enumerate(aps):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle(f"Model Comparison — LOO-CV (n={len(y_train)})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "03_auc_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: 03_auc_comparison.png")


# ── Plot 4: Feature Importance (Tree Models) ────────────────────────────────
def plot_feature_importance(final_models, feat_names, output_dir=FIGURES, top_n=15):
    tree_models = {k: v for k, v in final_models.items()
                   if k in ("RandomForest", "GradientBoosting", "XGBoost", "LightGBM")}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, model) in zip(axes, tree_models.items()):
        importances = model.feature_importances_
        actual_n = min(top_n, len(importances))
        idx = np.argsort(importances)[-actual_n:]
        top_names  = [feat_names[i] for i in idx]
        top_values = importances[idx]

        ax.barh(range(actual_n), top_values, color=MODEL_COLORS[name], alpha=0.85)
        ax.set_yticks(range(actual_n))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"{name} — Top {actual_n} Features")

    fig.suptitle("Feature Importance (Tree Models)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "04_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"Saved: 04_feature_importance.png")


# ── Plot 5: SHAP Beeswarm ──────────────────────────────────────────────────
def plot_shap(final_models, loo_results, X_train, feat_names, output_dir=FIGURES):
    """SHAP values for the best model by LOO-CV AUC."""
    best_name = max(loo_results, key=lambda n: loo_results[n]["auc"])
    best_model = final_models[best_name]
    print(f"SHAP: {best_name} (AUC={loo_results[best_name]['auc']:.3f})")

    try:
        if best_name in ("LogReg_L2", "LogReg_L1"):
            explainer = shap.LinearExplainer(best_model, X_train,
                                              feature_perturbation="interventional")
        else:
            explainer = shap.TreeExplainer(best_model)

        shap_values = explainer.shap_values(X_train)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, feature_names=feat_names,
                          show=False, max_display=15, plot_type="dot")
        plt.title(f"SHAP Values — {best_name} (best model)\n"
                  "Red = pushes prediction toward Series B success", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / "05_shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: 05_shap_beeswarm.png")
    except Exception as e:
        print(f"SHAP plot skipped ({e})")


# ── Plot 6: Probability Ranking ─────────────────────────────────────────────
def plot_probability_ranking(final_models, loo_results, train_df, test_df,
                              preprocessor, feature_cols, target,
                              output_dir=FIGURES):
    """Rank all companies by predicted probability of Series B success."""
    best_name = max(loo_results, key=lambda n: loo_results[n]["auc"])
    best_model = final_models[best_name]

    all_df = pd.concat([train_df, test_df], ignore_index=True)
    X_raw = all_df[feature_cols].copy()
    for c in feature_cols:
        if c != "industry_bucket" and c in X_raw.columns:
            X_raw[c] = X_raw[c].fillna(0)
    X_all = preprocessor.transform(X_raw)

    probs = best_model.predict_proba(X_all)[:, 1]
    y_all = all_df[target].values
    split = all_df["split"].values if "split" in all_df.columns else ["train"] * len(all_df)

    ranking = pd.DataFrame({
        "Company": all_df["Organization_Name"].values,
        "Prob": probs, "Actual": y_all, "Split": split,
    }).sort_values("Prob", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    bar_colors = []
    for _, row in ranking.iterrows():
        if row["Actual"] == 1:
            bar_colors.append("#2ca02c")
        elif row["Split"] == "test":
            bar_colors.append("#ff7f0e")
        else:
            bar_colors.append("#aec7e8")

    ax.barh(range(len(ranking)), ranking["Prob"], color=bar_colors, alpha=0.9, edgecolor="white")
    ax.set_yticks(range(len(ranking)))
    ax.set_yticklabels(ranking["Company"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted P(Series B within 36m)", fontsize=11)
    ax.set_title(f"Company Rankings — {best_name}", fontsize=10)
    ax.axvline(0.5, ls="--", color="gray", lw=1, alpha=0.6)

    legend_elements = [
        mpatches.Patch(color="#2ca02c", alpha=0.9, label="Actual success (Series B)"),
        mpatches.Patch(color="#aec7e8", alpha=0.9, label="No Series B yet"),
        mpatches.Patch(color="#ff7f0e", alpha=0.9, label="Test set"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "06_probability_ranking.png", dpi=150)
    plt.close(fig)
    print(f"Saved: 06_probability_ranking.png")


# ── Plot 8: Calibration ────────────────────────────────────────────────────
def plot_calibration(loo_results, y_train, output_dir=FIGURES):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Perfectly calibrated")

    for name, res in loo_results.items():
        try:
            n_bins = min(5, max(2, int(len(res["y_true"]) / 10)))
            prob_true, prob_pred = calibration_curve(
                res["y_true"], res["y_score"], n_bins=n_bins, strategy="uniform"
            )
            ax.plot(prob_pred, prob_true, "o-", color=MODEL_COLORS[name],
                    lw=1.5, markersize=5, label=name)
        except Exception:
            pass

    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title(f"Calibration — LOO-CV (n={len(y_train)})")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "08_calibration.png", dpi=150)
    plt.close(fig)
    print(f"Saved: 08_calibration.png")


# ── Run All Plots ───────────────────────────────────────────────────────────
def generate_all_plots(results: dict, output_dir: Path = FIGURES):
    """Generate all evaluation plots from training results."""
    r = results
    plot_roc_curves(r["loo_results"], r["y_train"], output_dir)
    plot_pr_curves(r["loo_results"], r["y_train"], output_dir)
    plot_auc_comparison(r["loo_results"], r["y_train"], output_dir)
    plot_feature_importance(r["final_models"], r["feat_names"], output_dir)
    plot_shap(r["final_models"], r["loo_results"], r["X_train"], r["feat_names"], output_dir)
    plot_probability_ranking(
        r["final_models"], r["loo_results"], r["train_df"], r["test_df"],
        r["preprocessor"], r["feature_cols"], r["target"], output_dir
    )
    plot_calibration(r["loo_results"], r["y_train"], output_dir)
    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    results = train_all()
    generate_all_plots(results)

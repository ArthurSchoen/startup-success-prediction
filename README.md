# Predicting Series B Funding from Public Data

**MIT Sloan School of Management — Master's Thesis (May 2026)**
Arthur Schoen | Advisor: Michael A. Cusumano

---

Can public data predict which startups will raise Series B after Series A?

This thesis builds a machine learning pipeline that integrates **14 public data sources** — SEC filings, patents, archived websites, news coverage, code repositories, and more — into a unified feature set for startup outcome prediction. We introduce a **citation-verified LLM extraction** method that recovers structured facts from archived web pages while programmatically rejecting hallucinated outputs.

## Key Results

| Metric | Value |
|---|---|
| Cross-validated AUC | **0.720** |
| Precision @ top 5% | **91.9%** (vs 52.1% base rate) |
| Dataset | 2,993 US startups (Series A 2015–2021) |
| Model features | 34 |
| Improvement over random | +44% |

The most significant finding: **SEC Form D filings alone achieve AUC 0.693**. Adding 13 additional sources improves this by only 0.026 — structured regulatory data, freely available on SEC EDGAR, carries the vast majority of predictive signal.

## Pipeline Architecture

```
Crunchbase (15,899 startups)
        |
  Step 01: SEC Form D Matching ──── Confirm Series A + extract date/amount
        |
  Step 02: Feature Cleaning ─────── Filter US, realistic age, match confidence
        |
  Step 03: LLM Classification ───── GPT-4o-mini: sector + deeptech score
        |                                                    2,993 startups
  Step 04: Parallel Signal Extraction ── 12 extractors x 14 public sources
        |
  Step 05: Feature Engineering ──── 34 features, 80/20 stratified split
        |
  Step 06: Model Training ────────── XGBoost, LightGBM, RF, GBM, LogReg (5-fold CV)
        |
  Step 07: Ablation Study ────────── Per-source contribution analysis
        |
  Step 08: Anti-Leakage Audit ───── Temporal + survivorship + entity checks
        |
  Results: CV AUC 0.720 | Test AUC 0.718 | P@5%: 91.9%
```

## Data Sources

| # | Source | Signal Type | Coverage |
|---|---|---|---|
| 1 | Crunchbase | Company metadata | 100% |
| 2 | SEC Form D | Regulatory filings | 100% |
| 3 | EDGAR SEC | Regulatory filings | 52.0% |
| 4 | Wayback Machine | Archived websites | 50.6% |
| 5 | Patents (BigQuery) | Patent filings | 25.1% |
| 6 | GDELT | News coverage | 18.3% |
| 7 | GitHub | Code activity | 12.2% |
| 8 | Hacker News | Community attention | 10.3% |
| 9 | Federal Awards | Grants/contracts | 9.8% |
| 10 | npm/PyPI/crates.io | Package releases | 3.4% |
| 11 | YC Alumni List | Accelerator network | 3.3% |
| 12 | Clinical Trials | Clinical trials | 2.9% |
| 13 | OpenAlex | Academic papers | 0.9% |
| 14 | Wikipedia | Notability | 0.2% |

## Project Structure

```
├── data/
│   ├── raw/                  # Crunchbase exports + SEC Form D (not tracked)
│   ├── processed/            # Final 2,993-company dataset with 34 features
│   └── cache/                # Wayback Machine API cache (not tracked)
├── src/
│   ├── pipeline/             # Steps 01–08: full reproducible pipeline
│   │   ├── 01__formd__seriesA_dates_enrich.py
│   │   ├── 02__clean__core_features.py
│   │   ├── 03__llm_classify.py
│   │   ├── 04__parallel_orchestrator.py
│   │   ├── 04a–04n: 14 signal extractors
│   │   ├── 05__clean__final_dataset.py
│   │   ├── 06__model_training.py
│   │   ├── 07__ablation.py
│   │   └── 08__audit_features.py
│   └── shared/               # Utilities (env, concurrency, caching)
├── reports/figures/           # 8 figures used in the thesis
├── docs/                     # Feature definitions + API documentation
└── .gitignore
```

## Thesis Figures

| Figure | Description |
|---|---|
| `sector_chart.png` | Sector distribution (n = 2,993) |
| `pipeline_v6_final.png` | Data collection and feature engineering pipeline |
| `roc_curves_v2.png` | ROC curves for all six models (5-fold CV) |
| `pr_curves_v2.png` | Precision-Recall curves for all six models |
| `sparse_sources_b_rate.png` | Series B rate by signal presence across sources |
| `wayback_features_b_rate.png` | Wayback Machine features: Series B rate by presence |
| `top5_profile_ratio.png` | Top 5% vs bottom 50% company profile |
| `individual_source_auc.png` | Individual source predictive power (AUC trained alone) |

## Citation

> Arthur Schoen. *Predicting Series B Funding from Public Data.* Master's thesis, MIT Sloan School of Management, May 2026.

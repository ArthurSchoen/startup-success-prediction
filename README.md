# Predicting Series B Funding from Public Data

**MIT Sloan School of Management — Master's Thesis (May 2026)**
Arthur Schoen | Advisor: Michael A. Cusumano

---

## TL;DR

- Built an end-to-end ML pipeline predicting which Series A startups go on to raise a follow-on round (Series B or equivalent), using only public data: 14 sources (SEC filings, patents, web archives, news, GitHub, and more) on 2,993 US startups.
- Result: cross-validated AUC 0.720 and 91.9% precision in the top 5% of predictions, against a 52.1% base rate.
- Headline finding: free SEC Form D filings alone carry almost all the signal (AUC 0.693), and most "alternative data" adds surprisingly little.
- Introduced a citation-verified LLM extraction method that pulls structured facts from archived web pages and programmatically rejects hallucinated outputs.

Can public data predict which startups will raise a follow-on round after their Series A?

This thesis builds a machine learning pipeline that integrates **14 public data sources** — SEC filings, patents, archived websites, news coverage, code repositories, and more — into a unified feature set for startup outcome prediction. We introduce a **citation-verified LLM extraction** method that recovers structured facts from archived web pages: every fact the model returns must carry a quote that appears verbatim in the fetched page, and anything whose quote cannot be found is discarded before it becomes a feature. Attributing each fact to the right founder is left to the LLM (see Limits).

## Key Results

| Metric | Value |
|---|---|
| Cross-validated AUC | **0.720** |
| Precision @ top 5% | **91.9%** (vs 52.1% base rate) |
| Dataset | 2,993 US startups (Series A 2015–2021) |
| Model features | 34 |
| Improvement over random | +44% |
| Held-out test AUC (599 startups) | 0.717 (0.718 in the submitted thesis) |

Every number in this README is recomputed from the saved dataset by
`src/pipeline/09__report_metrics.py`, which writes [`reports/METRICS.md`](reports/METRICS.md):
the six-model comparison, precision at 1/5/10/20%, bootstrap intervals, per-source
ablation and the robustness checks below.

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
  Step 09: Report Metrics ───────── Every reported number + ROC/PR figures
        |
  Results: CV AUC 0.720 | Test AUC 0.717 | P@5%: 91.9%
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
│   ├── pipeline/             # Steps 01–09: full reproducible pipeline
│   │   ├── 01__formd__seriesA_dates_enrich.py
│   │   ├── 02__clean__core_features.py
│   │   ├── 03__llm_classify.py
│   │   ├── 04__parallel_orchestrator.py
│   │   ├── 04a–04n: 14 signal extractors
│   │   ├── 05__clean__final_dataset.py
│   │   ├── 06__model_training.py
│   │   ├── 07__ablation.py
│   │   ├── 08__audit_features.py
│   │   └── 09__report_metrics.py
│   └── shared/               # Utilities (env, concurrency, caching)
├── reports/                  # METRICS.md, metrics.json, figures/ (8 figures used in the thesis)
├── docs/                     # Feature definitions + API documentation
└── .gitignore
```

## Thesis Figures

| Figure | Description |
|---|---|
| `sector_chart.png` | Sector distribution (n = 2,993) |
| `pipeline_v6_final.png` | Data collection and feature engineering pipeline |
| `roc_curves_v2.png` | ROC curves for all six models (5-fold CV on the training set), regenerated by stage 09 |
| `pr_curves_v2.png` | Precision-Recall curves for all six models, regenerated by stage 09 |
| `sparse_sources_b_rate.png` | Series B rate by signal presence across sources |
| `wayback_features_b_rate.png` | Wayback Machine features: Series B rate by presence |
| `top5_profile_ratio.png` | Top 5% vs bottom 50% company profile |
| `individual_source_auc.png` | Individual source predictive power (AUC trained alone) |

## Running it

```bash
pip install -r requirements.txt
```

The stages are numbered and run in order. Stages 01 to 04 collect data and are
the slow part, several hours wall clock, most of it waiting on public APIs.
Stages 05 to 08 run on the assembled dataset in a few minutes.

```bash
python src/pipeline/01__formd__seriesA_dates_enrich.py   # SEC Form D matching
python src/pipeline/02__clean__core_features.py          # cohort filtering
python src/pipeline/03__llm_classify.py                  # sector + deeptech score
python src/pipeline/04__parallel_orchestrator.py         # the 14 extractors
python src/pipeline/05__clean__final_dataset.py          # features + train/test split
python src/pipeline/06__model_training.py                # six models, 5-fold CV
python src/pipeline/07__ablation.py                      # per-source contribution
python src/pipeline/08__audit_features.py                # leakage and proxy checks
python src/pipeline/09__report_metrics.py                # every reported number + ROC/PR figures
```

Two API keys are needed, both read from the environment by `src/shared/env_loader.py`:
an OpenAI key for the sector classification in stage 03 and the Wayback extraction
in stage 04m, and a Crunchbase export as the starting cohort. Every other source
is free and unauthenticated: SEC EDGAR, the Wayback Machine, GDELT, OpenAlex,
Hacker News, ClinicalTrials.gov, USAspending, npm, PyPI and crates.io.

The scripts expect to be run from the repository root and read and write under a
`Pipeline/` directory that is not tracked here, since it holds the Crunchbase
export and the intermediate CSVs.

## The label

The model is trained on `label_B`. A startup is positive if any of four signals shows
a follow-on round after its Series A:

1. Crunchbase lists its last equity round as Series B or later.
2. A later SEC Form D filing in the Series B range.
3. A Crunchbase "Venture, series unknown" round at least 6 months after the Series A,
   larger than the Series A or at least $7M.
4. Total equity above twice the Series A, at least 6 months after it.

There is no time window: the cohort raised its Series A between 2015 and 2021 and is
observed in 2026. The base rate is 52.1% (1,559 of 2,993). Of those positives, 860 are
flagged by Crunchbase and 59 by a later Form D (the two overlap slightly); 647 come only
from rules 3 or 4.

With a strict window instead (a dated follow-on within 36 months), the base rate drops
to 21.9% and the AUC barely moves: 0.709 against 0.720. Note that the stage 05 column
named `SeriesB_within_36m` holds the no-window label, despite its name.

`success_seriesB_plus` is a wider audit label used by stages 07 and 08: the company ever
reached Series B or beyond, optionally widened to total equity at least twice the
Series A. Its conclusions are evidence about the features, not a validation of the
trained target.

## Limits and next steps

- **The label is partly indirect.** About 40% of positives (647 of 1,559) are inferred from
  funding growth rather than a dated Series B. Next step: rebuild the label from dated
  rounds only.
- **Founder attribution relies on the LLM.** The code verifies that every quote exists in
  the page, which blocks invented facts. Whether the quote is about a founder rather than
  an advisor is left to the LLM's judgment, as instructed by the prompt, and is not
  re-checked in code. Next step: require the founder's name or role inside the quote.

## Citation

> Arthur Schoen. *Predicting Series B Funding from Public Data.* Master's thesis, MIT Sloan School of Management, May 2026.

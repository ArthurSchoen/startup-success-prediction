# Predicting Series B Funding from Public Data

**MIT Sloan School of Management — Master's Thesis (May 2026)**
Arthur Schoen | Advisor: Michael A. Cusumano

---

## TL;DR

- Built an end-to-end ML pipeline predicting which Series A startups raise a Series B, using only public data: 14 sources (SEC filings, patents, web archives, news, GitHub, and more) on 2,993 US startups.
- Result: cross-validated AUC 0.720 and 91.9% precision in the top 5% of predictions, against a 52.1% base rate.
- Headline finding: free SEC Form D filings alone carry almost all the signal (AUC 0.693), and most "alternative data" adds surprisingly little.
- Introduced a citation-verified LLM extraction method that pulls structured facts from archived web pages and programmatically rejects hallucinated outputs.

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
```

Two API keys are needed, both read from the environment by `src/shared/env_loader.py`:
an OpenAI key for the sector classification in stage 03 and the Wayback extraction
in stage 04m, and a Crunchbase export as the starting cohort. Every other source
is free and unauthenticated: SEC EDGAR, the Wayback Machine, GDELT, OpenAlex,
Hacker News, ClinicalTrials.gov, USAspending, npm, PyPI and crates.io.

The scripts expect to be run from the repository root and read and write under a
`Pipeline/` directory that is not tracked here, since it holds the Crunchbase
export and the intermediate CSVs.

## A note on the two labels

There are two definitions of success in this repository and they are not
interchangeable.

`SeriesB_within_36m` is the trained target: the company raised a Series B within
36 months of its Series A. It is time bounded, which is what makes it a
prediction rather than a description, and it is what stages 06 and 07 use.

`success_seriesB_plus` is a wider audit label: the company ever reached Series B
or beyond. Stage 08 goes further and also counts a company as positive when its
total equity funding reached twice the Series A. That definition was used to
stress the features against the most generous reading of success, so its
conclusions are evidence about the features, not a validation of the trained
target.

## Limits, and what I would do differently

**The leakage audit checks a different label than the model trains on.** Stage 08
is the most careful part of this work: it tests every feature univariately, checks
whether the useful ones are just proxies of the Series A amount, drops that amount
and re-runs, and asks whether the label construction itself leaks. All of that runs
on the wider label. Re-running it on `SeriesB_within_36m` is the first thing I would
do, and until then the leakage conclusions are one step removed from the headline
numbers.

**Stage 08 also finds a bias I did not remove.** Its last test shows that the
"twice the Series A" rule is easier to trigger on small rounds, so smaller Series A
companies are more likely to be labelled positive for a reason that has nothing to
do with their prospects. The diagnostic prints the positive rate by round size and
stops there. The trained label does not have this problem, which is part of why it
is the one the results are reported on.

**Hyperparameters were chosen, not searched.** Shallow trees, strong
regularisation, 100 to 500 estimators. The reasoning is that roughly 3,000 rows and
120 candidate features punish depth, and the shallow settings are what kept
cross-validation and test performance close (0.720 against 0.718). But there is no
grid search and no nested cross-validation, so I cannot claim these are optimal,
only that they are conservative.

**The scaler is fit before the cross-validation split.** Each fold's standardisation
has therefore seen its own validation rows. Four of the six models are trees and
ignore scaling entirely, so the effect is small, but the two logistic regressions
are very slightly flattered. The fix is to wrap the preprocessing and the model in a
single pipeline object and hand that to the cross-validator.

**Coverage is the real ceiling, not the algorithm.** Six of the fourteen sources
cover under 11% of the cohort, and three cover under 4%. The ablation reflects this:
SEC Form D alone reaches 0.693 and all thirteen other sources together add 0.026.
Adding more sources of the same kind will not move this. Better resolution on the
sources that already have coverage would.

**The bootstrap intervals are narrower than they look.** They resample the
cross-validated predictions, which captures sampling noise but not the variance
from refitting the models. A proper interval would repeat the whole
cross-validation across seeds.

**There are no tests, and the pipeline is not idempotent.** Stages read whatever
CSV they find with a glob and a modification-time sort, which is convenient when
iterating and fragile when reproducing. A manifest recording which input produced
which output would be worth more here than unit tests.

## Citation

> Arthur Schoen. *Predicting Series B Funding from Public Data.* Master's thesis, MIT Sloan School of Management, May 2026.

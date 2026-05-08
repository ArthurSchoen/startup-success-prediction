# All Pipeline Features

This file provides a high-level map of what each pipeline step produces. For the 34 features actually used in the model, see [FEATURES_FINAL_DATASET.md](FEATURES_FINAL_DATASET.md).

## Pipeline steps

| Step | Script | Output |
|------|--------|--------|
| 01 | `01__formd__seriesA_dates_enrich.py` | Series A date/amount, Form D match (CIK, directors, investors, prior rounds) |
| 02 | `02__clean__core_features.py` | Filters (US domicile, realistic age, match confidence), core Crunchbase features |
| 03 | `03__llm_classify.py` | Sector classification, deeptech score (GPT-4o-mini) |
| 04 | `04__parallel_orchestrator.py` | Runs 14 signal extractors in parallel (04a–04n) |
| 05 | `05__clean__final_dataset.py` | Merges signals, engineers composites, creates labels, train/test split |
| 06 | `06__model_training.py` | Trains 6 models (XGBoost, LightGBM, RF, GBM, LogReg L1/L2) with 5-fold CV |
| 07 | `07__ablation.py` | Per-source ablation study |
| 08 | `08__audit_features.py` | Anti-leakage audit on each feature |

## Signal extractors (step 04)

| Script | Source | Key signals | Coverage |
|--------|--------|-------------|----------|
| `04a__wikipedia` | Wikipedia | Page exists before A | 0.2% |
| `04b__github` | GitHub API | Repo exists, backlink count | 12.2% |
| `04c__packages` | PyPI/npm/crates | Package exists before A | 3.4% |
| `04d__openalex` | OpenAlex | Publication count | 0.9% |
| `04e__hacker_news` | HN Algolia | Mention count before A | 10.3% |
| `04f__product_hunt` | Product Hunt API | Launch before A | excluded (0% coverage) |
| `04g__clinical_trials` | ClinicalTrials.gov | Trial count before A | 2.9% |
| `04h__federal_awards` | USAspending | Grant/contract count | 9.8% |
| `04i__epo_patents` | BigQuery | Patent count before A | 25.1% |
| `04j__gdelt_news` | GDELT | News mentions 3m/12m before A | 18.3% |
| `04k__edgar_filings` | SEC EDGAR | Non-Form-D filing count | 52.0% |
| `04l__semantic_scholar` | Semantic Scholar | Academic papers | excluded (redundant with OpenAlex) |
| `04m__wayback_unified` | Wayback + GPT-4o-mini | Snapshots, careers page, founder credentials, customers, product stage | 50.6% |
| `04n__yc_alumni` | YC public API | YC batch before A | 3.3% |

## For removed/excluded features

See [FEATURES_REMOVED.md](FEATURES_REMOVED.md).

## For detailed collection methodology

See [DATA_COLLECTION_METHODS.md](DATA_COLLECTION_METHODS.md).

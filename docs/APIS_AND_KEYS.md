# API Keys and External Services

## Required keys (environment variables)

| Key | Used by | Purpose |
|-----|---------|---------|
| `OPENAI_API_KEY` | `03__llm_classify`, `04m__wayback_unified`, all LLM verifications | Sector classification + citation-verified fact extraction |
| `GOOGLE_APPLICATION_CREDENTIALS` | `04i__epo_patents` | BigQuery patent queries |

## Optional keys

| Key | Used by | Without it |
|-----|---------|------------|
| `GITHUB_TOKEN` | `04b__github` | 60 req/hr instead of 5,000 |
| `PRODUCTHUNT_TOKEN` | `04f__product_hunt` | Product Hunt features stay at 0 |

## Free APIs (no auth)

| API | Script |
|-----|--------|
| Wayback Machine CDX | `04m__wayback_unified` |
| Wikipedia MediaWiki | `04a__wikipedia` |
| GDELT DOC 2.0 | `04j__gdelt_news` |
| Hacker News Algolia | `04e__hacker_news` |
| ClinicalTrials.gov v2 | `04g__clinical_trials` |
| USAspending.gov | `04h__federal_awards` |
| SEC EDGAR submissions | `04k__edgar_filings` |
| OpenAlex | `04d__openalex` |
| PyPI / npm / crates.io | `04c__packages` |
| YC public API | `04n__yc_alumni` |

## Setup

All keys are read from environment variables via `os.environ.get()`. Store them in `.env` at the project root (gitignored). Never commit keys.

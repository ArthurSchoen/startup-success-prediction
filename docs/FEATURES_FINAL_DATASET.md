# Final Model Features (34 features)

The 34 features used in the thesis model (Table 13, Appendix C). All features are known at or before the Series A date — no temporal leakage.

## Crunchbase core (4)

| Feature | Description |
|---------|-------------|
| `age_at_seriesA_years` | Company age at Series A |
| `Number of Founders` | Count of listed founders |
| `name_has_tech_suffix` | Name ends in AI, Labs, Tech |
| `name_word_count` | Distinctive words in name |

## LLM classification (1)

| Feature | Description |
|---------|-------------|
| `deeptech_llm_score` | Deeptech score 0-10 (GPT-4o-mini) |

## SEC Form D (4)

| Feature | Description |
|---------|-------------|
| `seriesA_amount_usd_millions` | Series A amount |
| `formd_directors_preA` | Directors at Series A filing |
| `formd_investor_count_preA` | Investors at Series A filing |
| `formd_prior_rounds` | Prior SEC filings count |

## Wayback Machine (10)

| Feature | Description |
|---------|-------------|
| `wayback_snapshots_12m_preA` | Archived snapshots in 12m before A |
| `has_careers_page_preA` | Careers page exists on archived site |
| `founder_bigtech_flag` | Ex-Google/Meta founder (LLM + citation verified) |
| `founder_phd_flag` | PhD founder (LLM + citation verified) |
| `founder_school_tier` | University tier 0-3 (LLM + citation verified) |
| `founder_serial_entrepreneur_flag` | Serial entrepreneur (LLM + citation verified) |
| `founder_prior_exit_flag` | Prior exit (LLM + citation verified) |
| `founder_features_available` | Any verified founder signal from page |
| `named_customers_count_preA` | Named company customers (LLM + citation verified) |
| `product_stage_preA` | Product maturity 0-3 (LLM + citation verified) |

## Patents — BigQuery (1)

| Feature | Description |
|---------|-------------|
| `patent_count_preA_company` | Patents filed before Series A |

## Clinical Trials (1)

| Feature | Description |
|---------|-------------|
| `ct_trials_count_preA` | Registered trials before Series A |

## Federal Awards (1)

| Feature | Description |
|---------|-------------|
| `fed_awards_count_preA` | Federal grants + contracts before A |

## YC Alumni (1)

| Feature | Description |
|---------|-------------|
| `yc_alumni_preA` | YC batch before Series A |

## GDELT News (1)

| Feature | Description |
|---------|-------------|
| `gdelt_mentions_3m_preA` | News mentions 3m before A |

## EDGAR SEC (1)

| Feature | Description |
|---------|-------------|
| `edgar_filings_preA` | Non-Form-D SEC filings before A |

## Hacker News (1)

| Feature | Description |
|---------|-------------|
| `hn_mention_count_preA` | HN post count before A |

## GitHub (2)

| Feature | Description |
|---------|-------------|
| `github_before_seriesA` | Public repo exists before A |
| `backlinks_github_count_preA` | Other repos linking company domain |

## Wikipedia (1)

| Feature | Description |
|---------|-------------|
| `wikipedia_before_seriesA` | Wiki page exists before A |

## OpenAlex (1)

| Feature | Description |
|---------|-------------|
| `openalex_works_count` | Academic publications before A |

## Packages (1)

| Feature | Description |
|---------|-------------|
| `pkg_before_seriesA` | npm/PyPI package exists before A |

## Composite features (3)

| Feature | Description |
|---------|-------------|
| `institutional_footprint` | Count of institutional sources with signal (patents, EDGAR, clinical trials, federal awards) |
| `sector_match` | Right signal type for the sector |
| `community_footprint` | Count of community sources with signal (GitHub, packages, HN, YC) |

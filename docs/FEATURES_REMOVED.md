# Removed Features

Features excluded from the final dataset and model. They are still computed by upstream pipeline steps but are **not declared in FEATURE_OPTIONAL** and **not passed to step 09**, unless noted otherwise.

---

## Leakage: current-snapshot values (not available at Series A)

| Feature | Source | Reason |
|---------|--------|--------|
| `github_stars`, `github_forks`, `github_watchers`, `github_contributors` | GitHub API | Today's counts, not historical |
| `github_releases`, `github_tags` | GitHub API | Same — no historical API |
| `github_issues_open`, `github_issues_closed` | GitHub API | Same |
| `github_bus_factor`, `github_issue_resolution_time`, `github_releases_ratio` | GitHub API | Derived from current snapshot |
| `github_star_velocity`, `github_dependency_count`, `github_tech_validation_score` | GitHub API | Placeholders / uses current data |
| `github_stars_per_download` | GitHub + PyPI | Both components are leaky |
| `pkg_downloads_recent` | PyPI/npm | Last 30 days, not pre-A |
| `pkg_momentum_3m`, `pkg_dependency_graph` | PyPI/npm | Placeholders, need historical data |
| `openalex_cited_by_count`, `oa_h_index_preA` | OpenAlex | Citations accumulate over years |
| `oa_top_tier_ratio`, `oa_patent_citations`, `oa_author_prestige` | OpenAlex | Placeholders, never populated |
| `appstore_rating`, `appstore_rating_count` | App Store | Current user reviews |
| `wiki_lang_count`, `wiki_founder_prominence` | Wikipedia | Current revision data |
| `wikipedia_nb_links_out` | Wikipedia | Current internal links |
| `news_mentions_count_recent`, `news_sources_count_recent` | GDELT | Current 3-month window |
| `backlinks_hn_count`, `backlinks_reddit_count`, `backlinks_github_count`, `backlinks_approx_domains` | HN/Reddit/GitHub | All-time counts (no date filter) |

## Removed: overfitting / noise

| Feature | Source | Reason |
|---------|--------|--------|
| `emb_founder_0` through `emb_founder_15` | Wayback + PCA | 16 dimensions overfit on small N; dominated feature importance with noise |
| `emb_founder_text_len` | Wayback | Correlated with embedding quality, not startup quality |

## Removed: API unavailable

| Feature | Source | Reason |
|---------|--------|--------|
| `visits_preA_3m_avg`, `visits_preA_12m_avg`, `traffic_momentum`, `log_visits_preA`, `log_visits_bucket` | SEMrush | API not configured (SKIP_SEMRUSH=True), columns always empty |

## Removed: deprecated / replaced

| Feature | Source | Reason |
|---------|--------|--------|
| `deeptech`, `deeptech_score` | Rule-based | Replaced by `deeptech_llm_score` (GPT-4o-mini) |
| `sec_match_score` | Form D | Data quality metric, not a company signal |
| `founder_bigtech_flag`, `founder_phd_flag`, `founder_serial_entrepreneur_flag`, `founder_prior_exit_flag` | Wayback heuristic | Replaced by NLP-extracted founder features |

## Removed: source deleted from pipeline

| Feature | Source | Reason |
|---------|--------|--------|
| `gtrends_mean_3m`, `gtrends_max_3m`, `gtrends_mean_6m`, `gtrends_max_6m`, `gtrends_mean_12m`, `gtrends_max_12m`, `gtrends_slope_12m`, `gtrends_evolution_3m_12m`, `gtrends_volatility_12m`, `gtrends_std_12m`, `gtrends_spike_count`, `gtrends_weeks_above_90`, `gtrends_is_active`, `gtrends_acceleration`, `gtrends_breakout_score`, `gtrends_breakout_ratio`, `gtrends_brand_specificity_ratio`, `gtrends_seasonal_adjusted_mean`, `gtrends_geo_concentration`, `gtrends_correlation_topic`, `gtrends_share_of_search`, `gtrends_category_momentum_delta` | Google Trends (pytrends) | Entire Google Trends source removed from the pipeline. The extraction script (formerly 05b) was deleted and pytrends dependency dropped. Reasons: unofficial API with aggressive rate-limiting, indices are relative (0-100) making cross-startup comparison unreliable, and most startups returned all-zero values (volume too low). Not just excluded from the model -- the step no longer exists in the pipeline. |

## Placeholders (never populated)

| Feature | Source | Reason |
|---------|--------|--------|
| `ph_hn_cohesion_index`, `ph_medal_count`, `ph_maker_reputation` | Product Hunt | Need PH API v2 |
| `hn_comment_sentiment`, `hn_hiring_mentions` | Hacker News | Need NLP pipeline |

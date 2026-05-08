# Removed Features

Features excluded from the final 34-feature model. They are still computed by upstream pipeline steps but are not passed to step 06 (model training).

## Leakage: current-snapshot values (not available at Series A)

| Feature | Source | Reason |
|---------|--------|--------|
| `github_stars`, `github_forks`, `github_watchers`, `github_contributors` | GitHub API | Today's counts, not historical |
| `github_releases`, `github_tags` | GitHub API | No historical API |
| `github_issues_open`, `github_issues_closed` | GitHub API | Current snapshot |
| `pkg_downloads_recent` | PyPI/npm | Last 30 days, not pre-A |
| `openalex_cited_by_count` | OpenAlex | Citations accumulate over years |
| `wikipedia_nb_links_out` | Wikipedia | Current internal links |
| `backlinks_hn_count`, `backlinks_reddit_count`, `backlinks_github_count` | HN/Reddit/GitHub | All-time counts without date filter |

## Removed: API unavailable

| Feature | Source | Reason |
|---------|--------|--------|
| `visits_preA_3m_avg`, `visits_preA_12m_avg`, `traffic_momentum`, `log_visits_preA` | SEMrush | API not configured, columns always empty |

## Removed: source excluded from pipeline

| Feature | Source | Reason |
|---------|--------|--------|
| Google Trends features (22 columns) | Google Trends / pytrends | Unofficial API, aggressive rate-limiting, relative indices unreliable for cross-company comparison, most startups return all-zero values |
| App Store features | App Store (iTunes) | Survivorship bias: only returns currently available apps |
| Semantic Scholar features | Semantic Scholar | Redundant with OpenAlex |
| NIH Reporter features | NIH Reporter | Too narrow, replaced by USAspending.gov |
| Product Hunt features | Product Hunt | 0% coverage at current sample size |

# Data Collection Methods — Full Pipeline Reference

**Last updated:** April 24, 2026
**Purpose:** Document exactly how each data source is queried, matched, filtered, and anti-leakage-protected in the thesis pipeline.

---

## Pipeline Overview

```
Crunchbase CSV (raw export)
    |
    v
Step 01: Form D Matching (SEC EDGAR)
    |
    v
Step 02: Cleaning + Filtering
    |
    v
Step 03: LLM Sector Classification (GPT-4o-mini)
    |
    v
Step 04: Parallel Signal Extraction (13 extractors)
    |   04a Wikipedia          04h Federal Awards
    |   04b GitHub             04i Patents (BigQuery)
    |   04c Packages           04j GDELT News
    |   04d OpenAlex           04k EDGAR Filings
    |   04e Hacker News        04m Wayback Machine
    |   04f Product Hunt       04n YC Alumni
    |   04g Clinical Trials
    |
    v
Step 05: Feature Engineering + Train/Test Split
    |
    v
Step 06: Model Training (XGBoost)
```

---

## Step 01 — SEC Form D Matching

**Script:** `01__formd__seriesA_dates_enrich.py`

**Source:** SEC Form D filings stored in local SQLite database (`Form D SEC/formd_all.sqlite`), built from quarterly EDGAR bulk downloads (2017-2025).

**How it works:**
1. For each Crunchbase company, search the Form D ISSUERS table by fuzzy name match (Jaccard similarity)
2. Cross-validate matches by checking if Crunchbase founder last names appear in the Form D RELATEDPERSONS table
3. Among validated CIK matches, find the filing cluster that corresponds to a Series A equity raise
4. Extract the earliest non-amendment filing date as `seriesA_date`

**Company matching:**
- Jaccard similarity on normalized company names (strip legal suffixes: Inc, LLC, Corp, etc.)
- Founder name overlap validation (at least 1 founder surname must appear in Form D related persons)
- Match confidence score = Jaccard score * founder overlap ratio

**Anti-leakage:**
- Only pre-Series-A Form D filings used for investor/director counts
- `formd_directors_preA`: count of directors listed in filings before Series A
- `formd_investor_count_preA`: count of investors in filings before Series A
- `formd_prior_rounds`: count of distinct funding events before the Series A filing

**Output columns:**
- `seriesA_date`, `seriesA_quarter`, `seriesA_amount_usd`, `seriesA_security_type`
- `sec_match_cik`, `sec_match_score`, `sec_founders_overlap_ratio`, `sec_founders_overlap_count`
- `formd_directors_preA`, `formd_investor_count_preA`, `formd_prior_rounds`

---

## Step 02 — Cleaning + Core Features

**Script:** `02__clean__core_features.py`

**What it does:**
1. Drops companies without a valid `seriesA_date`
2. Drops companies with low match confidence (Jaccard < threshold)
3. Drops companies with unrealistic age (founded > 20 years before Series A)
4. Computes rule-based deeptech flag from Crunchbase `Industry Groups` (biotech, hardware, quantum, robotics keywords)
5. Computes `age_at_seriesA_years = (seriesA_date - Founded Date) / 365.25`

**Filtering result:** ~30-35% of Form D-matched companies survive cleaning (e.g., 2,000 input -> ~550 clean).

---

## Step 03 — LLM Sector Classification

**Script:** `03__llm_classify.py`

**API:** OpenAI GPT-4o-mini
**Auth:** `OPENAI_API_KEY` environment variable
**Cost:** ~$0.50 per 7,500 companies

**How it works:**
1. Send company name + Crunchbase description + industries to GPT-4o-mini
2. Ask for: deeptech score (0-10), sector classification, name distinctiveness flag
3. Single API call per company, 0.3s delay between calls

**Output columns:**
- `deeptech_llm_score` (0-10 continuous)
- `startup_sector` (devtools | enterprise | consumer | deeptech | other)
- `name_distinctive` (0/1 — is the company name unique enough for reliable API lookups?)

---

## Step 04a — Wikipedia

**Script:** `04a__wikipedia.py`

**API:** Wikipedia MediaWiki API (`https://en.wikipedia.org/w/api.php`)
- Action `query` with `list=search` for article lookup
- Action `query` with `prop=revisions` for edit history
- Wikimedia Pageviews API (`https://wikimedia.org/api/rest_v1/metrics/pageviews/`)

**Auth:** None required
**Rate limit:** 0.5s delay between requests

**Company matching:**
1. Search Wikipedia by company name
2. Try company name + industry keywords if no result
3. LLM verification: send first 3 sentences of the Wikipedia article to GPT-4o-mini, ask "is this about {company_name} the startup?"
4. Check article creation date < Series A date

**Anti-leakage:**
- Only counts edits that occurred before Series A date
- Only counts pageviews in windows before Series A
- Rejects articles created after Series A

**Output columns:**
- `wikipedia_before_seriesA` (0/1)
- `wiki_edit_count_preA` — number of edits before Series A
- `wiki_editor_diversity` — unique contributors before Series A
- `wiki_vandalism_hit` — reverts/undos detected
- `wiki_infobox_presence` — structured data box present
- `wiki_views_growth_6m_preA` — pageview growth rate (6m vs prior 6m)
- `wiki_note`

**Coverage:** ~0.5% (very few startups have Wikipedia pages before Series A)

---

## Step 04b — GitHub

**Script:** `04b__github.py`

**API:** GitHub REST API v3 (`https://api.github.com`)
- `/orgs/{org}/repos` for organization repo listing
- `/repos/{owner}/{repo}/commits` with `since`/`until` date filters
- `/search/repositories` for name-based lookup

**Auth:** `GITHUB_TOKEN` optional (5,000 req/hour with token, 60/hour without)
**Rate limit:** 1.2s delay between requests, 4 concurrent workers with token

**Company matching:**
1. Extract domain slug (e.g., `stripe.com` -> `stripe`)
2. Try as GitHub org: `GET /orgs/stripe`
3. Try as GitHub user: `GET /users/stripe`
4. Search repos by company name
5. LLM verification on top 3 candidates (verifies the org/repo belongs to the company)

**Anti-leakage:**
- Commit counts use `since` and `until` params anchored to Series A date
- `github_commits_3m`: commits in 3 months before Series A
- `github_commits_6m`: commits in 6 months before Series A
- `github_commits_12m`: commits in 12 months before Series A
- Repo creation date checked: if created after Series A, all counts zeroed

**Output columns:**
- `github_commits_3m`, `github_commits_6m`, `github_commits_12m`
- `github_distinct_committers_3m`
- `github_first_commit_date`, `github_before_seriesA`
- `github_months_from_first_commit_to_seriesA`
- `github_commit_frequency` (commits_12m / 12)
- `github_commit_acceleration` (3m rate vs 12m rate)
- `backlinks_github_count_preA` — other repos referencing company domain
- `github_note`

**Coverage:** ~5-8%

---

## Step 04c — Package Registries

**Script:** `04c__packages.py`

**APIs:**
- PyPI JSON API (`https://pypi.org/pypi/{package}/json`)
- npm Registry (`https://registry.npmjs.org/{package}`)
- Crates.io API (`https://crates.io/api/v1/crates/{package}`)

**Auth:** None required
**Rate limit:** 0.2-0.5s delay between registry checks

**Company matching:**
1. Generate slug variants: company name lowercased, hyphenated, domain stem
2. Try scoped packages: `@company/sdk`, `@company/client`, `@company/core`
3. Check all 3 registries for each variant
4. LLM verification: does the package description match the company?
5. Founding-date sanity: package can't predate company by >1 year

**Anti-leakage:**
- Only counts releases with upload date before Series A
- Rejects packages whose first release is after Series A

**Output columns:**
- `pkg_first_release_date`, `pkg_releases_count_preA`
- `pkg_before_seriesA` (0/1)
- `pkg_registry` (pypi | npm | crates)
- `pkg_note`

**Coverage:** ~7%

---

## Step 04d — OpenAlex

**Script:** `04d__openalex.py`

**API:** OpenAlex API (`https://api.openalex.org`)
- `/institutions` — match company to academic institution
- `/works` — retrieve publications filtered by institution + date

**Auth:** None required (polite pool: add `mailto` param)
**Rate limit:** 0.5s delay, 10 req/s limit

**Company matching:**
1. Search OpenAlex institutions by company domain (homepage_url field)
2. Fallback: search by exact company name
3. If multiple candidates, pick the one whose homepage matches company domain

**Anti-leakage:**
- Publications filtered by `publication_date <= seriesA_date`
- `openalex_works_12m`: papers in 12 months before Series A
- `openalex_works_36m`: papers in 36 months before Series A

**Output columns:**
- `openalex_works_count` — total publications before Series A
- `openalex_works_12m`, `openalex_works_36m`
- `openalex_matched_institution`
- `openalex_signal_note`

**Coverage:** ~16% (strong for biotech/AI companies)

---

## Step 04e — Hacker News

**Script:** `04e__hacker_news.py`

**API:** Hacker News Algolia API (`https://hn.algolia.com/api/v1/search`)
- Query: exact domain in quotes (e.g., `"stripe.com"`)
- Filters: `show_hn` tag first, then `story` fallback

**Auth:** None required
**Rate limit:** 0.4s delay between requests

**Company matching:**
- Search by quoted domain to avoid substring false positives
- LLM verification of post title to confirm it's about the target company

**Anti-leakage:**
- Post `created_at` must be before Series A date
- Founding-date guard: discard posts >1 year before founding
- Upvote/comment counts are NOT extracted (they reflect current state, not post-time state)
- Only binary signals and post dates are kept

**Output columns:**
- `hn_launch` (0/1 — any HN post found)
- `hn_is_show_hn` (0/1 — was it a Show HN)
- `hn_post_date`
- `hn_mention_count_preA` — number of distinct HN posts before Series A
- `hn_before_seriesA` (0/1)
- `hn_note`

**Coverage:** ~1%

---

## Step 04f — Product Hunt

**Script:** `04f__product_hunt.py`

**API:** Product Hunt GraphQL API v2 (`https://api.producthunt.com/v2/api/graphql`)
- Query: `posts(slug: "...")` to look up product pages

**Auth:** `PRODUCTHUNT_TOKEN` optional (500 req/day)
**Rate limit:** 0.6s delay between requests

**Company matching:**
1. Generate ~12 slug variants: name-slug, compact form, with TLD suffix, numbered variants (`drift`, `drift-2`, `drift-3`)
2. Try each slug via GraphQL query
3. On first hit, enumerate sibling launches (same product, different versions)
4. LLM verification: does the PH product match the company?
5. Sector filter: skips deeptech companies (rarely launch on PH)

**Anti-leakage:**
- Only counts launches with date before Series A
- Current votes/comments NOT extracted (they accumulate over time)
- Only binary launch status and dates

**Output columns:**
- `ph_launch` (0/1), `ph_launch_date`
- `ph_before_seriesA` (0/1)
- `ph_featured` (0/1 — was the launch featured by PH editors?)
- `ph_launches_count_preA`
- `ph_note`

**Coverage:** ~5-10%

---

## Step 04g — Clinical Trials

**Script:** `04g__clinical_trials.py`

**API:** ClinicalTrials.gov API v2 (`https://clinicaltrials.gov/api/v2/studies`)
- Query: sponsor name search
- Filter: `query.spons={company_name}`

**Auth:** None required
**Rate limit:** 0.3s delay

**Company matching:**
1. Search by company name in sponsor field
2. Try variant names (with/without legal suffixes)
3. LLM verification to distinguish same-company legal variants from name collisions
4. Medical/biotech gate: skips non-medical companies entirely (based on `startup_sector` and industry keywords)

**Anti-leakage:**
- Trial `StartDate` must be <= Series A date
- Only counts trials that started before Series A

**Output columns:**
- `ct_trials_count_preA` — number of registered trials before Series A
- `ct_min_phase_preA`, `ct_max_phase_preA` — earliest and latest trial phases
- `ct_before_seriesA` (0/1)
- `ct_note`

**Coverage:** ~3% of medical companies

---

## Step 04h — Federal Awards (USAspending)

**Script:** `04h__federal_awards.py`

**API:** USAspending.gov API (`https://api.usaspending.gov/api/v2/search/spending_by_award/`)
- Two separate queries per company: grants (award type codes 02-05) and contracts (codes A-D)

**Auth:** None required
**Rate limit:** 0.3s delay between requests

**Company matching:**
1. Search recipient name by company name (fuzzy substring)
2. Volume guard: if >50 awards returned, likely a name collision (e.g., "Ramp" matches many unrelated federal contractors)
3. LLM verification with volume context for ambiguous matches
4. Founding-date hard gate: awards can't be >5 years before company founding

**Anti-leakage:**
- Award `start_date` must be <= Series A date
- Only counts awards that started before Series A

**Output columns:**
- `fed_grants_count_preA` — NIH/NSF/DOE/USDA grants
- `fed_contracts_count_preA` — DoD/DARPA/NASA contracts
- `fed_awards_before_seriesA` (0/1)
- `fed_awards_note`

**Coverage:** ~3-5%

---

## Step 04i — Patents (BigQuery)

**Script:** `04i__epo_patents.py`

**Source:** Google BigQuery `patents-public-data.patents.publications`
- SQL query with LIKE-join on assignee names

**Auth:** `GOOGLE_APPLICATION_CREDENTIALS` (service account JSON)
**Cost:** ~6 GB per query (~$0.03 at $5/TB), capped at 50 GB safety limit

**Company matching:**
1. Normalize company name: uppercase, strip legal suffixes (INC, LLC, CORP, CO, LTD)
2. Generate LIKE patterns with legal suffix variants (e.g., `%STRIPE INC%`, `%STRIPE LLC%`, `%STRIPE CORP%`)
3. Query BigQuery for matching assignees
4. Founding-date hard gate: reject if earliest patent filing is >5 years before founding
5. LLM semantic verification: does the patent title relate to the company's domain? (catches name collisions like "Roots Insurance" matching agricultural patents)

**Anti-leakage:**
- `filing_date` must be <= Series A date
- Only counts patents filed before Series A

**Companies are batched:** 100 companies per BigQuery query for efficiency.

**Output columns:**
- `patent_count_preA_company` — number of patents filed before Series A
- `has_patent_preA_company` (0/1)
- `patent_earliest_date_preA`
- `patent_matched_assignees` — which assignee names matched
- `patent_note`

**Coverage:** ~2%

---

## Step 04j — GDELT News

**Script:** `04j__gdelt_news.py`

**API:** GDELT DOC 2.0 REST API (`https://api.gdeltproject.org/api/v2/doc/doc`)
- Query: quoted company name, mode `ArtList`, format `json`
- Time window params: `startdatetime`, `enddatetime`

**Auth:** None required
**Rate limit:** 5.5s delay (API limit: 1 request per 5 seconds). Retry once on 429.

**Company matching:**
- Exact quoted company name (e.g., `"Stripe"`) to avoid substring false positives
- Noise cap: if >500 mentions returned, flag as likely name collision

**Anti-leakage:**
- Two time windows, both anchored to Series A:
  - `gdelt_mentions_12m_preA`: articles in 12 months before Series A
  - `gdelt_mentions_3m_preA`: articles in 3 months before Series A
- No current-state data used

**Output columns:**
- `gdelt_mentions_12m_preA`, `gdelt_mentions_3m_preA`
- `gdelt_has_preA_press` (0/1)
- `gdelt_note`

**Coverage:** ~30-40%

---

## Step 04k — EDGAR SEC Filings

**Script:** `04k__edgar_filings.py`

**API:** SEC EDGAR submissions API (`https://data.sec.gov/submissions/CIK{cik}.json`)
- Uses the CIK already matched in step 01 (Form D)

**Auth:** None required (User-Agent header with contact email required by SEC)
**Rate limit:** 0.12s delay (8 req/sec, under SEC's 10/sec limit)

**Company matching:**
- No new matching needed: uses `sec_match_cik` from step 01
- Companies without a CIK match get `edgar_filings_preA = 0`

**Anti-leakage:**
- Filing date strictly < Series A date
- Excludes the Series A Form D filing itself

**Output columns:**
- `edgar_filings_preA` — total non-Form-D SEC filings before Series A
- `edgar_has_10k_preA` (0/1) — has annual report filing
- `edgar_note`

**Coverage:** ~5-10%

---

## Step 04m — Wayback Machine (Internet Archive)

**Script:** `04m__wayback_unified.py`

**APIs:**
1. **CDX API** (`https://web.archive.org/cdx/search/cdx`) — index of all archived snapshots for a domain
2. **Wayback page fetch** (`https://web.archive.org/web/{timestamp}/{url}`) — retrieve archived HTML
3. **OpenAI GPT-4o-mini** — extract structured facts from archived page text

**Auth:** `OPENAI_API_KEY` for LLM extraction
**Rate limit:** 3 parallel workers max. CDX retries with 1.5s backoff on throttling. Fallback to `available` API if CDX fails.

**How it works (per company):**
1. **CDX search:** Query all archived snapshots of the company domain, filtered to `statuscode:200`, from founding year to Series A year
2. **Pick best snapshot:** Select the most recent snapshot before Series A date from the homepage
3. **Fetch + parse:** Download the archived HTML, extract text via BeautifulSoup
4. **Find team/about page:** Search CDX results for `/about`, `/team`, `/people`, `/leadership` URLs. Fetch the best match.
5. **LLM fact extraction:** Send page text to GPT-4o-mini with citation-verification prompt
6. **Citation verification:** Every extracted fact must include an exact quote that is a literal substring of the original page text. Hallucinated quotes are silently dropped.

**Company matching:**
- Domain-based (uses `Website` from Crunchbase, cleaned via tldextract)
- No name matching needed (we're fetching the company's own website)

**Anti-leakage:**
- CDX query filtered to snapshots with `timestamp <= seriesA_date`
- Snapshot counts computed in pre-A windows only
- `wayback_snapshots_12m_preA`: number of archived snapshots in 12 months before Series A
- `wayback_snapshots_3m_preA`: number of archived snapshots in 3 months before Series A (proxy for web activity)
- `wayback_acceleration`: ratio of 3m to 12m snapshot density (increasing = company is updating website more frequently)

**LLM extraction (citation-verified):**
The LLM extracts structured JSON with these fields, each requiring an exact quote:
- `universities`: founder university names + exact quote proving it
- `has_phd`: boolean + quote
- `bigtech_experience`: list of big tech companies + quotes
- `serial_entrepreneur`: boolean + quote
- `prior_exit`: boolean + quote
- `named_customers`: list of customer names + quotes
- `product_stage`: 0-3 scale + quote (0=pre-product, 1=beta, 2=launched, 3=revenue/growth)

**Verification logic:**
```python
def _quote_verified(quote_str, min_len=5):
    q = (quote_str or "").lower().strip()
    return bool(q and len(q) >= min_len and q in text_lower)
```
Each extracted fact's quote must be a literal substring of the page text. If not, the fact is dropped.

**Generic customer blacklist:** Words like "restaurants", "hospitals", "enterprises", "businesses" are rejected as named customers — only real company names (e.g., "Walmart", "Google") pass.

**School tier mapping:** Verified university names are mapped to tiers (1-4) based on a lookup table (Ivy League/Stanford/MIT = tier 4, state schools = tier 1).

**Output columns:**
- `wayback_snapshots_12m_preA`, `wayback_snapshots_3m_preA`
- `wayback_acceleration`
- `has_careers_page_preA` (0/1) — careers/jobs URL found in CDX index before Series A
- `founder_school_tier` (0-4)
- `founder_phd_flag` (0/1)
- `founder_bigtech_flag` (0/1)
- `founder_serial_entrepreneur_flag` (0/1)
- `founder_prior_exit_flag` (0/1)
- `founder_features_available` (0/1) — any verified founder signal found
- `named_customers_count_preA` — count of citation-verified customer names
- `product_stage_preA` (0-3)
- `wayback_comment`

**Coverage:** ~60% have at least one usable snapshot

---

## Step 04n — Y Combinator Alumni

**Script:** `04n__yc_alumni.py`

**Source:** YC public API (`https://yc-oss.github.io/api/companies/all.json`), cached locally at `Pipeline/_yc_companies.json` (~5,700 companies)

**Auth:** None required
**Rate limit:** N/A (single download, cached locally)

**Company matching (priority order):**
1. Domain match: company website domain matches YC company domain
2. Name match: normalized company name matches YC company name
3. Former names match: check YC `former_names` field
4. Founding-date gate: YC batch date can't be >180 days before company founding (catches domain acquisitions, e.g., Ramp acquiring ramp.com from Paribus)
5. LLM verification for ambiguous matches

**Anti-leakage:**
- YC batch date must be <= Series A date
- `yc_alumni_preA` is 1 only if the company went through YC before raising Series A

**Output columns:**
- `yc_alumni_preA` (0/1)
- `yc_batch` (e.g., "S18" for Summer 2018)
- `yc_months_before_seriesA` — time between YC batch and Series A
- `yc_note`

**Coverage:** ~6%

---

## Step 05 — Feature Engineering + Final Dataset

**Script:** `05__clean__final_dataset.py`

**What it does:**
1. Reads base CSV + all signal CSVs from step 04
2. Merges all signals on `Organization_Name`
3. Creates the label: `SeriesB_within_36m` using the 4-condition expanded definition (see THESIS_LABEL_DEFINITION.md)
4. Engineers composite features:
   - `has_technical_footprint`: derived from GitHub, packages, and backlinks
   - `visibility_score`: composite of GDELT, HN, Wikipedia signals
   - Log transforms: `log_patent_count`, `log_hn_mentions`, `log_wayback_3m`, `log_edgar_filings`, `log_gdelt_3m`
   - Interaction terms: `directors_x_patent`, `directors_per_round`
5. Creates temporal train/test split: `train` if `seriesA_year < 2022`, `test` otherwise
6. Selects the final 39 model features (defined in `final_feature_columns.json`)
7. Writes `final_dataset.csv`, `final_train.csv`, `final_test.csv`

**Removed sources (code still present for documentation but output is empty):**
- `ORDER_G = []` — SEMrush traffic (API unavailable)
- `ORDER_H = []` — Google Trends (noisy, no benchmark)

---

## Anti-Leakage Summary

Every extractor follows the same temporal discipline:

| Pattern | What it prevents | Used by |
|---------|------------------|---------|
| `date <= seriesA_date` | Using post-A data | All 13 extractors |
| Founding-date gate | Name collisions with older entities | Patents, Federal Awards, YC, Clinical Trials, HN |
| Volume cap | Generic name false positives | GDELT (>500), Federal Awards (>50) |
| LLM entity verification | Wrong-company matches | Wikipedia, GitHub, Packages, PH, Clinical Trials, Federal Awards, Patents, YC |
| Citation verification | LLM hallucination | Wayback Machine |
| Current-state exclusion | Survivorship bias | HN (no upvotes), PH (no votes), App Store (removed entirely) |

---

## API Keys Required

| Key | Source | Used by |
|-----|--------|---------|
| `OPENAI_API_KEY` | OpenAI | Step 03 (LLM classify), Step 04m (Wayback LLM extraction), all LLM verifications |
| `GITHUB_TOKEN` | GitHub | Step 04b (optional, 5K req/hr vs 60/hr) |
| `PRODUCTHUNT_TOKEN` | Product Hunt | Step 04f (optional, 500 req/day) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google Cloud | Step 04i (BigQuery patents) |

All other APIs (Wikipedia, OpenAlex, HN, ClinicalTrials.gov, USAspending, GDELT, EDGAR, Wayback CDX, YC) are free and require no authentication.

---

## Runtime Estimates (per batch of ~550 clean companies)

| Step | Approx time | Bottleneck |
|------|-------------|------------|
| 01 Form D matching | ~1 min | Local SQLite queries |
| 02 Cleaning | ~2 sec | Pure pandas |
| 03 LLM classify | ~9 min | OpenAI API (0.3s/company) |
| 04a Wikipedia | ~20 min | 0.5s/company + pageview queries |
| 04b GitHub | ~11 min | 1.2s/company, auth-dependent |
| 04c Packages | ~5 min | 3 registries x 0.3s each |
| 04d OpenAlex | ~7 min | 0.5s/company |
| 04e Hacker News | ~4 min | 0.4s/company |
| 04f Product Hunt | ~15 min | 0.6s/company, ~12 slug variants |
| 04g Clinical Trials | ~3 min | Only queries medical companies |
| 04h Federal Awards | ~12 min | 2 queries/company (grants + contracts) |
| 04i Patents (BigQuery) | ~5 min | Batched SQL, single query per 100 companies |
| 04j GDELT News | ~50 min | **5.5s/company** (strict rate limit) |
| 04k EDGAR Filings | ~2 min | 0.12s/company, needs CIK from step 01 |
| 04m Wayback Machine | **~3-4 hours** | **CDX + 2 page fetches + LLM per company, 3 workers** |
| 04n YC Alumni | ~90 sec | Local JSON lookup |
| 05 Feature engineering | ~30 sec | Pure pandas |
| **Total (parallel)** | **~4-5 hours** | **Wayback is the bottleneck** |

*Steps 04a-04n run in parallel via the orchestrator. Wall time = slowest step (Wayback).*

# Toutes les features du pipeline (étapes 01 → 05)

Ce fichier décrit **toutes** les colonnes produites ou transmises à chaque étape du pipeline (vue exhaustive). Pour uniquement les colonnes du **jeu final** utilisé en 08, voir [FEATURES_FINAL_DATASET.md](FEATURES_FINAL_DATASET.md).

---

## Vue d'ensemble des étapes

| Étape | Notebook | Fichier CSV principal | Rôle |
|-------|----------|------------------------|------|
| 01 | 01__formd__seriesA_dates_enrich | *_02_formd__seriesA_dates_enriched.csv | Dates Series A, match Form D SEC |
| 02 | 02__clean__core_features | *_03_...__clean.csv | Core features, fondateurs, industry |
| 03 | 03__wayback__snapshots_extract | *_04_wayback__snapshots.csv | URLs Wayback |
| 04 | 04__webtraction__signals_extract | *_05_webtraction__signals.csv | Trafic, fondateurs enrichis |
| 03 | 03__patentsview__signals | *_06_patents__signals.csv | Brevets pré–Series A |
| 04a | 04a__wikipedia__signals | *_08_wikipedia__signals.csv | Wikipedia |
| 04b | 04b__news_mentions__signals | *_09_news_mentions__signals.csv | GDELT news |
| 04c | 04c__backlinks_light__signals | *_10_backlinks_light__signals.csv | HN, Reddit, GitHub backlinks |
| 04d | 04d__github__signals | *_11_github__signals.csv | GitHub repo stats |
| 04e | 04e__packages__signals | *_12_packages__signals.csv | PyPI / npm / crates |
| 04f | 04f__openalex__signals | *_13_openalex__signals.csv | OpenAlex works/citations |
| 04g | 04g__producthunt_hn__signals | *_14_producthunt_hn__signals.csv | Product Hunt / HN |
| 05 | 05__clean__final_dataset | *_07_final_dataset.csv | Jeu final pour 08 |

---

## 01 – Form D / Series A dates

### Colonnes produites

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **seriesA_date** | ISO date | Date du tour Series A : priorité Crunchbase (*Last Funding Date* si *Last Funding Type* = Series A), sinon déduite des événements Form D SEC (cluster de montant dans la fourchette Series A US). | Référence temporelle pour toutes les fenêtres « pré–Series A ». |
| **seriesA_quarter** | str | Trimestre (ex. 2021Q3). | Agrégation temporelle ; utile pour split ou analyse par période. |
| **seriesA_confidence** | LOW / MED | LOW si pas de montant ou match faible ; MED si match_score ≥ 0,50 et montant ≥ seuil Series A. | Plus la confiance est haute, plus la date/montant Series A est fiable. |
| **seriesA_amount_usd_num** | float | Montant en USD (brut). Source : *Last Equity/Last Funding Amount* (Crunchbase) ou Form D (TOTALAMOUNTSOLD / TOTALOFFERINGAMOUNT). | Taille du tour ; en 05 on utilise souvent une version en millions. |
| **sec_match_score** | float [0–1] | Similarité **Jaccard** entre les tokens du nom de la société (Crunchbase) et du nom de l’émetteur Form D (issuer_name) : `|intersection| / |union|`. Attribué au **meilleur** candidat CIK qui a au moins un fondateur en commun (voir ci‑dessous). | 1 = noms quasi identiques ; 0,5 = recouvrement partiel. N’est renseigné que si un match Form D avec overlap fondateurs a été trouvé. |
| **sec_founders_overlap_count** | int | Nombre de **noms de famille** des fondateurs Crunchbase qui apparaissent parmi les *related persons* du Form D (CIK retenu). | Nombre de fondateurs « vérifiés » par le dépôt SEC ; 0 si pas de match Form D. |
| **sec_founders_overlap_ratio** | float [0–1] | `sec_founders_overlap_count / nombre de fondateurs Crunchbase`. | Proportion des fondateurs recoupés ; 1 = tous recoupés. |
| **sec_events_found** | int | Nombre d’événements Form D (dépôts / amendements) récupérés pour le CIK retenu. | Indicateur d’activité de financement côté SEC. |

**Series B (01)** : **seriesB_date**, **seriesB_amount_usd**, **seriesB_amount_usd_num**, **seriesB_source**, **seriesB_comment**. Priorité : Crunchbase si *Last Funding Type* = Series B (ou B+ : C, D, IPO, Late Stage) avec date + montant ; sinon Form D (médiane ~35M USD, recherche d’un tour après seriesA_date). **Validation** : si seriesB_date &lt; seriesA_date → seriesB_* invalidé (seriesB_comment = error_B_before_A). Utilisé en 05 pour **months_to_seriesB** et **SeriesB_within_36m**.

---

## 02 – Core features

### Colonnes produites

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **industry_bucket** | cat | Agrégation des *Industries* Crunchbase en 5 buckets : **BIO** (biotech, life science, pharma…), **HARD** (aerospace, industrial, robotics, energy, semiconductor…), **FIN** (fintech, crypto, blockchain…), **INFRA** (cyber, telecom, quantum…), **SAAS** (reste). Premier mot-clé trouvé dans la chaîne détermine le bucket. | Secteur simplifié pour analyse ou variables catégorielles. |
| **deeptech_score** | int | Score additif : +2 par token « fort » (science and engineering, hardware, biotech, energy, quantum, aerospace, robotics, semiconductor, medical device, life science, cleantech, agtech), +1 par token « soft » (manufacturing, data and analytics…). Règle « AI tout seul ≠ deeptech » : si uniquement Science and Engineering + (AI ou Software) → +1 seulement. −2 si purement non‑deeptech (sales, real estate, advertising, payments…). Tokens issus de *Industry Groups* + *Industries*. | Score ordinal ; plus il est élevé, plus la société est considérée comme deeptech. |
| **deeptech** | 0/1 | 1 si `deeptech_score ≥ 2`, 0 sinon. | Indicateur binaire deeptech (définition SHAPE). |
| **age_at_seriesA_years** | float | Âge de la société à la date Series A : `(seriesA_date - Founded Date) / 365.25` en années. Vide si pas de *Founded Date*. | Maturité au moment du tour ; souvent &lt; 6 ans pour des Series A « classiques ». |
| **age_at_seriesA_flag** | OK / SUSPECT / UNKNOWN | **OK** si age_at_seriesA_years &lt; 6 ; **SUSPECT** si ≥ 6 ; **UNKNOWN** si pas de Founded Date. En 05 les lignes SUSPECT peuvent être exclues. | Qualité de l’âge : SUSPECT signale une possible erreur de date ou société très mature. |
| **Number of Founders** | int | Nombre de fondateurs (Crunchbase). | Taille de l’équipe fondatrice. |

Héritage 01 : seriesA_date, seriesA_quarter, seriesA_confidence, seriesA_amount_usd_num, sec_*, seriesB_*, etc.

---

## 03 – Wayback snapshots

Colonnes liées aux **snapshots** (URLs Wayback, dates de capture). Utilisées par l’étape 04 pour récupérer le contenu des pages au plus tard avant la Series A. Pas de table dédiée dans ce fichier ; le détail (wayback URLs, snapshot dates) est dans les CSVs d’étape 03/04.

---

## 04 – Webtraffic / fondateurs

### Colonnes produites

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **visits_preA_3m_avg** | float | Moyenne des **visites mensuelles** (SEMrush) sur les **3 derniers mois** avant seriesA_date, pour le domaine du site (domain_clean). | Trafic récent ; unité = visites/mois. |
| **visits_preA_12m_avg** | float | Moyenne des visites mensuelles sur les **12 mois** avant Series A. | Trafic sur l’année ; moins volatil que 3m. |
| **traffic_momentum** | float | `visits_preA_3m_avg / (visits_preA_12m_avg + 1)`. | &gt; 1 = trafic en hausse sur la fin ; &lt; 1 = baisse. |
| **log_visits_preA** | float | `log1p(visits_preA_3m_avg)`. | Trafic en échelle log pour modèles ; réduit l’impact des très gros sites. |
| **log_visits_bucket** | cat | Percentiles sur `log_visits_preA` (sur l’échantillon) : **top_10%** (≥ p90), **top_30%** (≥ p70), **bottom_70%** (&lt; p70). Vide si pas de données trafic. | Classement relatif de trafic dans la cohorte. |

**Fondateurs (03/04)** : **founder_features_available** (1 si contexte fondateurs ≥ 120 caractères et ancre rôle type CEO/CTO), **founder_context_len**, **founder_text_coverage_ratio** (part du texte « filtré fondateurs » par rapport au texte total), **founder_bigtech_flag**, **founder_phd_flag**, **founder_serial_entrepreneur_flag**, **founder_prior_exit_flag**. Détection à partir du texte Wayback (contexte autour des noms/rôles fondateurs ; big tech / PhD / serial / exit par regex et listes de mots-clés).

**Pourquoi `log_visits_bucket` (et parfois `log_visits_preA`) peut être 0 partout** : si les données trafic (ex. SEMrush) sont absentes ou non chargées, `log_visits_preA` reste NaN/0 → bucket vide ou unique.

---

## 03 – PatentsView

### Colonnes produites

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **patent_count_preA_company** | int | Nombre de brevets (PatentsView) associés à la **société** (assignee) dont la **date de dépôt** (application) est **avant** seriesA_date. | Volume d’IP pré–Series A ; 0 = aucun brevet. |
| **has_patent_preA_company** | 0/1 | 1 si patent_count_preA_company ≥ 1, 0 sinon. | Indicateur binaire « au moins un brevet pré–A ». |
| **patent_earliest_date_preA** | ISO date | Date de dépôt du **plus ancien** brevet trouvé avant la Series A pour cette société (un seul par startup). Vide si aucun brevet pré–A. | Ancienneté de l’IP ; permet de mesurer « depuis quand » la société dépose. |

Hérite de 04 (et donc 01–04).

---

## 04a – Wikipedia

### Biais et risques (à prendre en compte)

1. **Biais de notoriété** : Wikipedia ne conserve que les pages ayant une couverture « significative et indépendante ». Pour des milliers de startups, le taux de **wikipedia_exists = 0** peut être très élevé (>90 %). Si le signal est trop rare, le modèle risque de ne rien apprendre. **Recommandation** : ne pas sur-pondérer Wikipedia ; l’utiliser en complément d’autres signaux.

2. **wikipedia_nb_links_out = leakage** : L’API ne fournit pas l’historique des liens. Une page créée en 2017 qui faisait 500 octets peut avoir aujourd’hui 50 liens. Utiliser le nombre de liens **actuel** pour une prédiction en 2017 est un **leakage majeur**. Cette colonne est **exclue du modèle** (étape 05 : `LEAKAGE_RISK_COLS` ; pas d’imputation ni de feature en 08). Elle reste exportée pour traçabilité uniquement.

3. **Biais d’homonymie** : Chercher « Kong » ou « Extend » peut renvoyer la page gorille ou logiciel. **Mitigations en place** : (a) validation **domaine OU au moins un fondateur** dans l’extrait ; (b) **mots-clés contexte entreprise** dans l’extrait (startup, headquarters, venture capital, company, founded, based in, inc., ltd, corporation, headquartered, software company, technology company) ; (c) **catégories Wikipedia** (ex. « Companies established in [Year] », « American companies », « Technology companies ») ; (d) **wiki_is_disambiguation** = 1 si la page est une page d’homonymie → exists = 0.

### Colonnes produites

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **wikipedia_exists** | 0/1 | 1 si une page Wikipedia (actuelle) existe pour la société (recherche par nom/alias). | Notoriété actuelle. |
| **wikipedia_existed_at_seriesA** | 0/1 | 1 si une révision de la page existait **à ou avant** la date Series A (API avec date). | Notoriété pré–Series A (anti‑leakage). |
| **wikipedia_signal_date** | ISO date | Date de **création** de la page (première révision). | Ancienneté de la présence Wikipedia. |
| **wikipedia_before_seriesA** | 0/1 | 1 si la page a été créée avant ou à la Series A, 0 sinon. | Signal binaire « Wikipedia avant la A ». |
| **wikipedia_page_length** | int | Taille en **octets** de la **dernière révision au plus tard à la date Series A** (API `rvend` + `rvdir=older`). | Importance de la page à l’époque ; évite le leakage. |
| **wikipedia_nb_links_out** | int | Nombre de **liens internes** (vers d’autres pages Wikipedia) dans la révision actuelle (l’API ne fournit pas les liens par révision historique). | **Ne pas utiliser en prédiction** : leakage (liens actuels, pas historiques). Exclu du modèle (05). Traçabilité uniquement. |
| **wikipedia_excerpt** | str | Extrait du contenu (~500 car.) **uniquement pour les pages qui passent** la validation (domaine ou fondateur présent). | Vérification manuelle du bon fonctionnement. |
| **wikipedia_note** | str | Commentaire si signal trouvé **après** Series A (features mises à 0). | Traçabilité anti‑leakage. |
| **wiki_edit_count_preA** | int | Nombre de **modifications** de la page avant la Series A (révisions à ou avant seriesA_date). | Activité et intérêt de la communauté. |
| **wiki_editor_diversity** | int | Nombre d’**éditeurs uniques** avant la Series A. 1 seul = souvent employé/PR ; plusieurs = intérêt public. | Diversité des contributeurs. |
| **wiki_is_disambiguation** | 0/1 | 1 si le titre ou les catégories indiquent une **page d’homonymie**. Si 1 → exists = 0 (bruit). | Marque faible ou nom trop commun. |

### Features Wikipedia (implémentées)

| Feature | Définition | Pourquoi c’est prédictif |
|---------|------------|---------------------------|
| **wiki_vandalism_hit** | Nombre de réversions (undo) avant Series A. | Notoriété : plus on est connu, plus on est ciblé. |
| **wiki_infobox_presence** | 1 si la page contient une « Infobox Company ». | Structure professionnelle (standards Wikipedia). |
| **wiki_views_growth_6m_preA** | Croissance des **pageviews** (API pageviews) sur les 6 mois avant Series A. | Due diligence investisseurs / futurs employés. |
| **wiki_founder_prominence** | Les fondateurs ont-ils eux-mêmes une page Wikipedia ? | Signal « serial entrepreneur ». |
| **wiki_lang_count** | Nombre de **langues** dans lesquelles la page existe avant Series A (langlinks). | Expansion internationale, domination de marché. |

---

## 04b – News mentions (GDELT)

### Biais et erreurs (implémentation actuelle : API DOC 2.0, fenêtre rolling 3 mois)

1. **Rolling 3 mois = survivorship bias** : Si tu interroges GDELT **aujourd’hui** (ex. 2026) pour une startup qui a levé en 2021, tu obtiens les news **des 3 derniers mois (2026)**. **Conséquence** : tu ne prédis pas la réussite, tu confirmes que la boîte existe encore 5 ans plus tard. Le signal actuel = « buzz récent / société encore active », pas « buzz pré–Series A ». **Solution** : utiliser **BigQuery (Google Cloud)** pour les **archives historiques** GDELT (gratuit jusqu’à 1 To/mois).

2. **Ambiguïté des noms** : Une recherche sur « Extend » ou « Kong » génère des milliers de faux positifs. **Solution** : opérateurs de proximité ou **mots-clés obligatoires** dans la requête (ex. `"startup" OR "funding" OR "CEO"` en plus de `"Nom" company`).

### Colonnes produites (état actuel)

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **news_mentions_count** | int | Nombre de mentions GDELT pour la requête (nom société / domaine) sur une fenêtre **rolling 3 mois** (API gratuite ; pas d’historique strict pré–Series A). | Buzz **récent** ; à interpréter avec prudence (survivorship bias). |
| **news_sources_count** | int | Nombre de sources distinctes. | Diversité de la couverture. |
| **news_mentions_query_used** | str | Requête effectivement utilisée. | Reproductibilité. |
| **news_mentions_note** | str | Commentaire (ex. fenêtre rolling, pas d’historique). | Traçabilité. |

**Requête GDELT** : on essaie d'abord `"Nom" (startup OR funding OR CEO)` (réduit ambiguïté des noms), puis `"Nom" company`, puis `"Nom"` seul. Colonnes **news_avg_tone**, **news_velocity_3m**, **news_lexical_diversity**, **news_global_reach**, **news_tone_slope**, **news_cooccurrence_vc**, **news_goldstein_scale** : *placeholder* np.nan tant qu'on n'utilise pas BigQuery.

### Features recommandées (GDELT Historical via BigQuery – à remplir)

À implémenter si passage à **BigQuery** pour l’historique (fenêtre **avant** seriesA_date).

| Colonne | Type | Intérêt pour la prédiction |
|---------|------|----------------------------|
| **news_avg_tone** | float | Score de ton GDELT (-100 à +100). Ton très positif pré-levée = signal fort. |
| **news_velocity_3m** | float | Ratio mentions_last_30_days / mentions_last_90_days. Accélération soudaine précède souvent l’annonce de levée. |
| **news_lexical_diversity** | float | Nombre de mots uniques dans les articles. Vocabulaire riche = articles de fond plutôt que communiqués robotisés. |
| **news_global_reach** | int | Nombre de pays/langues où la startup est mentionnée. Fort pour Deeptech à visée mondiale. |
| **news_tone_slope** | float | *Tone momentum* : ton de plus en plus positif sur les 6 mois avant Series A → « lune de miel » médiatique qui attire les investisseurs. |
| **news_cooccurrence_vc** | float / int | La startup apparaît-elle dans le **même article** que des fonds (Sequoia, a16z) ou concurrents leaders ? GDELT = entités citées ensemble (effet réseau). |
| **news_goldstein_scale** | float | Score CAMEO d’impact sur la stabilité. Score élevé = soutien institutionnel ou partenariats majeurs (ex. contrats gouvernementaux). |

---

## 04c – Backlinks light

### Biais et erreurs (implémentation actuelle)

1. **Biais temporel (Backlinks vs Series A)** : Les comptes HN / Reddit / GitHub sont calculés **au moment de la requête** (ex. 2026). Pour une Series A en 2021, `backlinks_*` reflètent la visibilité **actuelle**, pas pré–Series A → **leakage**. Idem pour un vrai « nombre de domaines référents » (Ahrefs/Moz) : le chiffre serait celui de 2026. **Solution** : utiliser des APIs **filtrées par date** — **Algolia HN** (filtre `numericFilters` sur `created_at_i`), **Pushshift** pour Reddit (par date). Pour un compte global de domaines, seul un accès **historique** (Semrush/Ahrefs Historical, ou proxy via Wayback) est valide.

2. **Bruit des « domains »** : Si on intègre un jour un vrai comptage de domaines (crawl / Ahrefs), beaucoup de backlinks viennent de **spam** ou répertoires automatiques. **Recommandation** : ne pas compter uniquement les domaines ; ajouter un **trusted_backlinks_ratio** (ex. liens depuis .edu, .gov, ou top 10k Alexa).

**Note** : `backlinks_approx_domains` actuel = nombre de **plateformes** (HN, Reddit, GitHub) avec au moins une mention (0–3), pas un comptage de domaines racine du web.

### Colonnes produites (état actuel)

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **backlinks_hn_count** | int | Nombre de hits Algolia HN (URL ou texte contenant le domaine). **Sans filtre date** → signal « aujourd’hui ». | Visibilité HN ; à interpréter avec prudence (leakage si Series A ancienne). |
| **backlinks_reddit_count** | int | Nombre de résultats recherche Reddit (domaine). **Sans filtre date**. | Visibilité Reddit ; idem. |
| **backlinks_github_count** | int | Nombre de repos GitHub dont la requête contient le domaine. **Sans filtre date**. | Liens depuis des repos. |
| **backlinks_approx_domains** | int | Nombre de plateformes (0–3) avec ≥ 1 mention (HN, Reddit, GitHub). Proxy « diversité », pas vrai nb de domaines. | Diversité des canaux. |
| **backlinks_hn_count_preA** | float | Nombre de hits HN **créés avant ou à** seriesA_date (Algolia `numericFilters` sur `created_at_i`). NaN si pas de seriesA_date. | Signal pré–Series A (anti‑leakage). |
| **backlinks_hn_before_seriesA** | float | 1 si backlinks_hn_count_preA > 0, 0 sinon. NaN si pas de seriesA_date. | Binaire : présence HN avant levée. |
| **hn_comments_count_preA** | float | Somme des **commentaires** sur les stories HN (domaine) créées avant seriesA_date. NaN si pas de seriesA_date. | Due diligence sociale pré–Series A. |
| **backlinks_reddit_count_preA** | float | Nombre de soumissions Reddit (domaine) **avant** seriesA_date, via **Pushshift** (si dispo). NaN si pas de date ou API indisponible. | Signal pré–Series A Reddit. |
| **backlinks_github_count_preA** | float | Nombre de repos GitHub **créés avant** seriesA_date (`created:<date`). NaN si pas de seriesA_date. | Repos existants avant levée. |
| **backlinks_approx_domains_preA** | float | Nombre de plateformes (0–3) avec ≥ 1 mention **pré–Series A** (HN, Reddit, GitHub). NaN si pas de seriesA_date. | Diversité pré–Series A. |
| **backlinks_signal_note** | str | Indique si filtre date utilisé (HN/GitHub préA, Reddit Pushshift) ou « pas de seriesA_date ». | Traçabilité. |

### Features recommandées (APIs sécurisées, filtrage par date)

À implémenter avec **Algolia HN** (filtre `created_at_i` ≤ date cible) et **Pushshift** (Reddit par date) pour un signal **pré–Series A** valide.

| Colonne | Type | Intérêt pour la prédiction |
|---------|------|----------------------------|
| **hn_frontpage_count** | int | Nombre de fois où le domaine a atteint la **Front Page** HN avant Series A. | Signal de « hype » technologique massive. |
| **hn_comments_count** | int | Nombre de **commentaires** sur les posts liés à la startup (Algolia). | Due diligence sociale : débats techniques = validation. |
| **github_repo_association** | 0/1 | Le domaine est-il lié à un repo GitHub avec **&gt; 100 stars** (avant Series A) ? | Preuve d’une approche Open Source réussie (clé Deeptech). |
| **github_readme_mentions** | int | Nombre de **README** d’autres projets citant la startup. | Dépendance / outil de référence = Product–Market Fit technique. |
| **reddit_sub_diversity** | int | Nombre de **subreddits** différents mentionnant la boîte (ex. r/MachineLearning, r/Python). | Intérêt multi-communautés vs une seule niche. |
| **backlink_velocity_preA** | float | Croissance du nombre de domaines référents (ou de mentions HN+Reddit) sur les **6 mois** avant la levée. | Accélération du réseau = prédicteur d’intérêt VC. |
| **trusted_backlinks_ratio** | float | Part des backlinks depuis domaines « de confiance » (.edu, .gov, top 10k Alexa). | Réduit le bruit spam/répertoires. |
| **referral_entropy** | float | **Entropie de Shannon** sur la distribution des sources (HN, Reddit, GitHub, blogs). | Élevée = connu partout ; faible = auto-promotion sur un canal. |

### Architecture de données recommandée (signaux « alpha »)

| Catégorie | Signal à privilégier | Outil suggéré |
|-----------|----------------------|---------------|
| **Tech Authority** | GitHub mentions / stars, README citations | GitHub API |
| **Intellectual Hype** | HN stories, front page, commentaires (filtrés par date) | Algolia HN API |
| **Public Validation** | Reddit sentiment / mentions par sub (filtrés par date) | Pushshift.io |
| **Web Authority** | Root domains (historique) | Semrush (trial) / Ahrefs Historical / Wayback |

---

## 04d – GitHub

### Biais « Snapshot »

Stars, forks, contributors sont des **valeurs actuelles** (au moment de la requête) → **data leakage** si on utilise pour prédire la levée. **Solution** : **GitHub Archive** (BigQuery) pour l'état du repo à T−1 mois avant Series A. En attendant : **github_before_seriesA**, **github_months_from_first_commit_to_seriesA**, et features « alpha » ci‑dessous.

### Colonnes produites

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **github_stars** | int | Nombre d’étoiles du repo principal (lié au domaine / nom). | Popularité. |
| **github_forks** | int | Nombre de forks. | Réutilisation du code. |
| **github_watchers** | int | Watchers. | Suivi. |
| **github_releases** | int | Nombre de releases. | Maturité produit. |
| **github_tags** | int | Nombre de tags. | Versions / milestones. |
| **github_commits_3m / 6m / 12m** | int | Nombre de commits sur les 3, 6 et 12 derniers mois (avant la date de requête). | Activité de développement. |
| **github_contributors** | int | Nombre de contributeurs. | Taille de l’équipe open source. |
| **github_issues_open** | int | Issues ouvertes. | Charge / transparence. |
| **github_issues_closed** | int | Issues fermées. | Résolution. |
| **github_first_commit_date** | ISO date | Date du premier commit. | Ancienneté du repo. |
| **github_signal_date** | ISO date | Date de la donnée (snapshot). | Reproductibilité. |
| **github_before_seriesA** | 0/1 | 1 si le repo / premier commit est avant ou à la Series A. | Anti‑leakage. |
| **github_note** | str | Commentaire si signal après Series A. | Traçabilité. |
| **github_bus_factor** | int | Nombre minimal de contributeurs possédant ≥ 50 % des commits. | Projet robuste vs dépendant d'une personne. |
| **github_issue_resolution_time** | float | Temps moyen (jours) entre ouverture et fermeture des issues fermées (hors PR). | Efficacité opérationnelle. |
| **github_distinct_committers_3m** | int | Nombre de développeurs distincts ayant committé sur les 3 derniers mois. | Équipe scalable. |
| **github_releases_ratio** | float | releases / max(tags, 1). Beaucoup de tags sans releases = désorganisé. | Maturité (releases avec notes). |
| **github_months_from_first_commit_to_seriesA** | float | Mois entre premier commit et Series A. 6 mois = momentum exceptionnel. | Ancienneté / vitesse. |
| **github_tech_validation_score** | float | 0,2×log(Stars+1) + 0,5×velocity_proxy + 0,3×contrib_norm. | Signal « alpha » pondéré. |
| **github_star_velocity** | float | *Placeholder* np.nan : (Stars_T − Stars_T−3m)/3. Nécessite GitHub Archive. | Accélération popularité avant levée. |
| **github_dependency_count** | float | *Placeholder* np.nan : nb de fois où le repo est listé dans package.json/requirements.txt d'autres projets. | Preuve infrastructure. |

---

## 04e – Packages

### Biais « Téléchargements » et mainteneur

**pkg_downloads_recent** : les registres (npm, PyPI) ne donnent généralement que les **30 derniers jours** → utiliser les téléchargements actuels pour prédire une levée passée = **leakage**. **Correction** : **PePy** (PyPI) ou **npm-stat** pour l’historique mois par mois. **pkg_momentum_3m** = placeholder (nécessite PePy/npm-stat). **Ambiguïté du mainteneur** : beaucoup de packages sont créés par des particuliers avant enregistrement de la société → **pkg_maintainer_email_domain_match** = 1 si l’email du mainteneur contient le domaine de la startup.

### Colonnes produites

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **pkg_first_release_date** | ISO date | Date du **premier** release (PyPI / npm / crates) lié à la société. | Ancienneté du package. |
| **pkg_releases_count** | int | Nombre de releases. | Maturité. |
| **pkg_downloads_recent** | int | Téléchargements récents (30 j par défaut ; **snapshot actuel** → risque leakage). | Adoption. |
| **pkg_registry** | str | Registre du **premier** trouvé (PyPI, npm, crates). | Source. |
| **pkg_signal_date** | ISO date | Date du signal. | Reproductibilité. |
| **pkg_before_seriesA** | 0/1 | 1 si premier release avant ou à la Series A. | Anti‑leakage. |
| **pkg_note** | str | Commentaire si signal après Series A. | Traçabilité. |
| **pkg_registry_count** | int | Nombre de **registres** où le package a été trouvé (même nom). | Multi‑registre. |
| **pkg_registries** | str | Liste des registres (ex. PyPI,npm). | Reproductibilité. |
| **pkg_update_frequency** | float | Nombre **moyen de jours** entre deux releases. | Équipe agile (cycle court = réactif). |
| **pkg_version_stability** | float | Jours écoulés avant d’atteindre la version **1.0.0**. NaN si jamais 1.0. | Maturité (v1.0 = confiance). |
| **pkg_major_updates_preA** | int | Nombre de **changements de version majeure** (v1, v2, v3…) avant Series A. Trop = instabilité/pivot. | Stabilité produit. |
| **pkg_maintainer_email_domain_match** | 0/1 | 1 si l’email du mainteneur contient le **domaine** de la startup. | Lien package ↔ société. |
| **pkg_ecosystem_fullstack** | 0/1 | 1 si le package existe sur **au moins 2 registres** (ex. npm + PyPI = full stack). | Solution complexe / full stack. |
| **pkg_momentum_3m** | float | *Placeholder* np.nan : croissance % des téléchargements sur les 3 mois pré‑Series A. Nécessite **PePy** / **npm-stat**. | Product‑Market Fit technique. |
| **pkg_dependency_graph** | float | *Placeholder* np.nan : nb d’autres packages qui listent ce package comme dépendance. Nécessite libraries.io ou équivalent. | Ancrage dans l’écosystème. |
| **github_stars_per_download** | float | Ratio github_stars / (pkg_downloads_recent + 1). Beaucoup d’étoiles et peu de downloads = hype ; peu d’étoiles et beaucoup de downloads = outil critique (intérêt VC). | Corrélation GitHub–Package. |

---

## 04f – OpenAlex

### Biais « Affiliation » et « Time-to-Citation »

**Ambiguïté d'affiliation** : les startups changent souvent de nom ou sont enregistrées sous des noms légaux différents (ex. « DeepMind » vs « Google DeepMind »). En ne cherchant que le nom commercial, on peut rater une part importante des publications. **Piste** : utiliser l'identifiant **ROR** (Research Organization Registry) qu'OpenAlex intègre (`institutions.ror`) et mapper le domaine / la société vers un ROR pour une extraction plus propre. **Biais time-to-citation** : les citations mettent des années à s'accumuler ; une startup très innovante qui a publié 3 mois avant sa Série A peut avoir 0 citation. **Piste** : ne pas s'appuyer uniquement sur `cited_by_count` ; utiliser aussi le Journal Impact Factor ou le **SJR** (SCImago Journal Rank) des revues où elles publient — `oa_top_tier_ratio` = placeholder (nécessite données SJR/IF externes).

### Colonnes produites

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **openalex_works_count** | int | Nombre de travaux (publications) liés à la société, **publiés à ou avant** la date Series A (anti‑leakage). | Production académique / R&D. |
| **openalex_cited_by_count** | int | Citations totales de ces travaux (pré‑Series A). | Impact. |
| **openalex_works_12m** | int | Travaux publiés dans les 12 mois avant Series A. | Activité récente. |
| **openalex_works_36m** | int | Travaux publiés dans les 36 mois avant Series A. | Activité sur 3 ans. |
| **oa_academic_velocity** | float | Ratio works_12m / (works_36m + ε). Si > 0,33, la production scientifique s'accélère à l'approche de la levée. | Signal « accélération ». |
| **oa_collaboration_index** | int | Nombre d'**institutions distinctes** (co‑signataires) sur les works. Plus une startup collabore avec des universités, plus elle est crédible aux yeux des VCs. | Collaboration universités. |
| **oa_h_index_preA** | int | H‑index calculé **uniquement** sur les travaux avant la Series A. Mesure la solidité de l'équipe de recherche fondatrice. | Qualité équipe recherche. |
| **oa_top_tier_ratio** | float | *Placeholder* np.nan : % de publications dans le top 10 % des journaux cités (SJR/IF). Nécessite données externes (ex. SCImago). | Distingue « vraie science » du « marketing blanc ». |
| **oa_patent_citations** | float | *Placeholder* np.nan : nb de fois les travaux sont cités par des brevets (signal science→marché / M&A). OpenAlex ou sources brevets. | Signal M&A / Série A massive. |
| **oa_author_prestige** | float | *Placeholder* np.nan : score de citations global d'un auteur « star » affilié à la startup. Nécessite appels API auteurs. | Recrutement star = prédicteur levée. |
| **openalex_signal_note** | str | Commentaire si date Series A manquante ou autre (traçabilité anti‑leakage). | Traçabilité. |

---

## 04g – Product Hunt / Hacker News

### Biais « Lancement fantôme » et « Upvotes PH »

**Lancement fantôme** : sur HN, beaucoup de startups sont mentionnées sans que ce soit un « Show HN » officiel. Une simple mention (type story) = signal de notoriété ; un **Show HN** = signal d'exécution volontaire (lancement assumé). **Correction** : distinguer les posts `show_hn` des posts `story` via l'API Algolia (`tags: "show_hn"`) → colonne **ph_hn_is_show_hn** (1 si le meilleur match est un Show HN, 0 si mention seule). **Upvotes PH** : les upvotes sur Product Hunt sont souvent critiqués pour être « organisés » par le marketing. **Correction** : ne pas s'appuyer uniquement sur le score brut ; privilégier le **rang du jour** (ex. Product of the Day #1) quand l'API le fournit → **ph_medal_count** = placeholder (nécessite champs PH dédiés).

### Colonnes produites

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **ph_hn_launch** | 0/1 | 1 si un post HN (URL domaine) a été trouvé. | Lancement / notoriété HN. |
| **ph_hn_is_show_hn** | 0/1 | 1 si le meilleur match est un **Show HN** (lancement volontaire), 0 si story seule (mention). | Signal d'exécution vs notoriété. |
| **ph_hn_upvotes** | int | Nombre de points du post. | Engouement. |
| **ph_hn_comments** | int | Nombre de commentaires. | Discussion. |
| **ph_hn_post_date** | ISO date | Date du post. | Timing. |
| **hn_comments_per_upvote** | float | commentaires / (upvotes + 1). Ratio élevé = débat / complexité ; faible = outil simple / gadget. En deeptech, ratio moyen à élevé souvent recherché. | Densité de discussion. |
| **hn_points_velocity** | float | *Proxy* : points / 24 (points par jour). Viralité pure ; version précise = points / heures sur front page. | Viralité. |
| **ph_hn_cohesion_index** | float | *Placeholder* np.nan : écart en jours entre le post PH et le post HN. Même semaine = stratégie « Blitz » structurée. Nécessite date de lancement PH. | Cohérence lancement PH–HN. |
| **ph_medal_count** | float | *Placeholder* np.nan : nb de badges (Product of the Day/Week/Month). Mesure la domination du produit sur sa fenêtre de lancement. Nécessite API PH. | Rang / domination PH. |
| **ph_maker_reputation** | float | *Placeholder* np.nan : le Maker (fondateur) a-t-il déjà lancé d'autres produits à succès sur PH ? Prédicteur « Serial Makers ». Nécessite API PH. | Réputation Maker. |
| **hn_comment_sentiment** | float | *Placeholder* np.nan : score NLP des commentaires HN avant Série A. Sentiment positif / constructif = validation technique. Nécessite récupération commentaires + NLP. | Validation technique. |
| **hn_hiring_mentions** | float | *Placeholder* np.nan : 1 si la startup apparaît dans les posts mensuels « Who is hiring? » sur HN avant Série A (signal de croissance organique). Nécessite recherche Algolia ask_hn. | Recrutement HN. |
| **ph_producthunt_launch** | 0/1 | 1 si présence sur Product Hunt (API token optionnel). | Lancement PH. |
| **ph_hn_signal_date** | ISO date | Date du signal (post). | Reproductibilité. |
| **ph_hn_before_seriesA** | 0/1 | 1 si le post est avant ou à la Series A. | Anti‑leakage. |
| **ph_hn_note** | str | Commentaire si signal après Series A. | Traçabilité. |

---

## 05 – Final dataset

Sélection des colonnes pour 08 : **labels** (success_seriesB_plus, SeriesB_within_36m, months_to_seriesB), **méta** (Organization_Name, seriesA_date, seriesA_year, split, etc.), **FEATURE_NUMERIC**, **FEATURE_CATEGORICAL**, **FEATURE_OPTIONAL** (dont colonnes « note »). Imputation des numériques, déduplication, split train/test. Sortie : *_07_final_dataset.csv*, *_07_final_train.csv*, *_07_final_test.csv*.

---

## Colonnes « note » (traçabilité anti-leakage)

**wikipedia_note**, **news_mentions_note**, **github_note**, **pkg_note**, **ph_hn_note** : texte explicatif quand un signal a été trouvé mais **après** la date de la Series A (features correspondantes mises à 0). Conservées dans le CSV final, exclues de l’imputation numérique et non utilisées comme variables numériques en 08.

# Features du jeu de données final (étape 08)

Ce fichier décrit **uniquement** les colonnes présentes dans le CSV final (sortie de l’étape 05) utilisées pour la modélisation (étape 08). Toutes les features sont connues **à ou avant** la date de la Series A (pas de leakage).

Pour une vue de **toutes** les colonnes produites à chaque étape du pipeline, voir [FEATURES_ALL.md](FEATURES_ALL.md).

---

## Labels (cibles)

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **success_seriesB_plus** | 0/1 | 1 si la société a atteint au moins une Series B / C / D / IPO / Late Stage (d’après *Last Funding Type* ou *Last Equity Funding Type* Crunchbase, ou Form D). 0 sinon. | Succès « au moins Series B+ ». |
| **success_seriesB_plus_reason** | str | Raison du classement : type de financement qui a déclenché le succès (ex. « Series B », « IPO », « Late Stage »). Vide si non success. | Traçabilité du label. |
| **SeriesB_within_36m** | 0/1 | 1 si Series B+ atteinte **dans les 36 mois** suivant la date de la Series A ; 0 sinon. **Cible principale** pour l’étape 08. | Succès rapide après la A. |
| **months_to_seriesB** | float | Nombre de mois entre **seriesA_date** et la date du dernier tour Series B+ (ou C/D/IPO/Late Stage). NaN si jamais Series B+. | Vitesse d’accès au tour suivant. |

---

## Méta & split

| Colonne | Type | Définition précise | Interprétation |
|---------|------|--------------------|----------------|
| **Organization_Name** | str | Nom de la société (Crunchbase). | Identifiant lisible. |
| **seriesA_date** | ISO date | Date de la Series A (référence temporelle). | Point d’ancrage pour toutes les fenêtres « pré–A ». |
| **seriesA_year** | int | Année de la Series A. | Utilisé pour le **split temporel** (train / test). |
| **split** | train / test | **train** si seriesA_year &lt; 2022, **test** sinon. | Évite le leakage temporel en 08. |
| **seriesA_quarter** | str | Trimestre de la Series A (ex. 2021Q3), si présent. | Agrégation temporelle. |
| **Website** | str | Site web (Crunchbase). | Domaine / URL. |

---

## Features numériques (core)

| Colonne | Source | Définition précise | Interprétation |
|---------|--------|--------------------|----------------|
| **seriesA_amount_usd_millions** | 01 | Montant de la Series A en **millions USD** (dérivé de seriesA_amount_usd_num en 06). | Taille du tour ; comparabilité entre sociétés. |
| **age_at_seriesA_years** | 01 / Crunchbase | Âge de la société à la Series A : `(seriesA_date - Founded Date) / 365.25` en années. Vide si pas de Founded Date. | Maturité au moment du tour ; souvent &lt; 6 ans. |
| **Number of Founders** | Crunchbase | Nombre de fondateurs. | Taille de l’équipe fondatrice. |
| **sec_match_score** | 01 / Form D | Similarité **Jaccard** (0–1) entre le nom société Crunchbase et le nom de l’émetteur Form D retenu (meilleur CIK avec au moins 1 fondateur en commun). | 1 = noms quasi identiques ; 0 = pas de match ou pas de overlap fondateurs. |
| **sec_founders_overlap_count** | 01 / Form D | Nombre de **noms de famille** des fondateurs Crunchbase qui apparaissent parmi les *related persons* du Form D (CIK retenu). | Nombre de fondateurs « vérifiés » par le dépôt SEC. |
| **sec_founders_overlap_ratio** | 01 / Form D | `sec_founders_overlap_count / nombre de fondateurs Crunchbase`. | Proportion des fondateurs recoupés ; 1 = tous recoupés. |
| **sec_events_found** | 01 / Form D | Nombre d’événements Form D (dépôts / amendements) pour le CIK retenu. | Activité de financement côté SEC. |
| **patent_count_preA_company** | 03 PatentsView | Nombre de brevets (assignee = société) dont la **date de dépôt** est **avant** seriesA_date. | Volume d’IP pré–Series A. |
| **has_patent_preA_company** | 03 PatentsView | 1 si patent_count_preA_company ≥ 1, 0 sinon. | Indicateur binaire « au moins un brevet pré–A ». |
| **founder_features_available** | 02 / 04 | 1 si contexte fondateurs ≥ 120 caractères et ancre rôle (CEO/CTO/…) détectée ; 0 sinon. | Qualité des données fondateurs (Wayback). |
| **founder_context_len** | 02 / 04 | Longueur (caractères) du **contexte fondateurs** extrait du texte Wayback (snippets autour des noms/rôles). | Quantité d’info fondateurs. |
| **founder_text_coverage_ratio** | 03 / 04 | Part du texte « filtré fondateurs » (contexte autour des noms) par rapport au texte total de la page. | Couverture du texte pertinent. |
| **founder_bigtech_flag** | 02 / 04 | 1 si au moins un fondateur a une expérience **big tech** (Google, Meta, Amazon, Apple, Microsoft, etc.) détectée dans le contexte ; 0 sinon. | Signal « alumni big tech ». |
| **founder_phd_flag** | 02 / 04 | 1 si au moins un fondateur avec **PhD** détecté dans le contexte ; 0 sinon. | Signal formation recherche. |
| **founder_serial_entrepreneur_flag** | 02 / 04 | 1 si au moins un fondateur **serial entrepreneur** (phrases type « founded X before », « previously started ») détecté ; 0 sinon. | Expérience entrepreneuriale. |
| **founder_prior_exit_flag** | 02 / 04 | 1 si au moins un fondateur avec **exit antérieur** (acquisition, IPO) détecté ; 0 sinon. | Track record. |
| **visits_preA_3m_avg** | 04 Webtraffic | Moyenne des **visites mensuelles** (SEMrush) sur les **3 derniers mois** avant Series A pour le domaine. | Trafic récent ; unité = visites/mois. |
| **visits_preA_12m_avg** | 04 Webtraffic | Moyenne des visites sur les **12 mois** avant Series A. | Trafic sur l’année. |
| **traffic_momentum** | 04 Webtraffic | `visits_preA_3m_avg / (visits_preA_12m_avg + 1)`. | &gt; 1 = hausse ; &lt; 1 = baisse. |
| **log_visits_preA** | 04 Webtraffic | `log1p(visits_preA_3m_avg)`. | Trafic en échelle log pour modèles. |

---

## Features catégorielles

| Colonne | Définition précise | Interprétation |
|---------|--------------------|----------------|
| **industry_bucket** | Secteur agrégé : **BIO** (biotech, life science, pharma…), **HARD** (aerospace, industrial, robotics, energy, semiconductor…), **FIN** (fintech, crypto…), **INFRA** (cyber, telecom, quantum…), **SAAS** (reste). Basé sur *Industries* Crunchbase. | Secteur simplifié pour modèles catégoriels. |
| **age_at_seriesA_flag** | **OK** si age_at_seriesA_years &lt; 6 ; **SUSPECT** si ≥ 6 ; **UNKNOWN** si pas de Founded Date. En 05 les lignes SUSPECT peuvent être exclues. | Qualité de l’âge ; SUSPECT = possible erreur ou société très mature. |

---

## Features optionnelles (si présentes)

Ces colonnes sont dans **FEATURE_OPTIONAL** ; elles peuvent être absentes selon les étapes exécutées. Détail complet dans [FEATURES_ALL.md](FEATURES_ALL.md). Résumé par source :

### Deeptech & trafic

| Colonne | Définition courte | Interprétation |
|---------|-------------------|----------------|
| **patent_earliest_date_preA** | Date ISO du **plus ancien** brevet pré–Series A (une par startup). | Ancienneté de l’IP. |
| **deeptech** | 1 si deeptech_score ≥ 2, 0 sinon. | Indicateur binaire deeptech. |
| **deeptech_score** | Score additif Industry Groups + Industries (+2 fort, +1 soft, −2 non‑deeptech ; règle « AI seul ≠ deeptech »). | Score ordinal deeptech. |
| **log_visits_bucket** | **top_10%** / **top_30%** / **bottom_70%** selon percentiles de log_visits_preA sur l’échantillon. | Classement relatif de trafic. |

### Wikipedia

| Colonne | Définition courte | Interprétation |
|---------|-------------------|----------------|
| **wikipedia_exists** | 1 si page existe (actuelle). | Notoriété actuelle. |
| **wikipedia_existed_at_seriesA** | 1 si révision existait à ou avant Series A. | Notoriété pré–A (anti‑leakage). |
| **wikipedia_signal_date** | Date de création de la page. | Ancienneté. |
| **wikipedia_before_seriesA** | 1 si page créée avant ou à la Series A. | Signal binaire. |
| **wikipedia_page_length** | Taille (octets) de la **dernière révision au plus tard à la Series A**. | Importance de la page à l’époque. |
| **wikipedia_nb_links_out** | Nombre de **liens internes** (vers d’autres pages Wikipedia) dans la révision actuelle. | Exclue du modèle (leakage). Traçabilité uniquement. |
| **wikipedia_excerpt** | Extrait (~500 car.) **uniquement si** domaine ou fondateur + contexte entreprise. | Vérification manuelle du bon fonctionnement. |
| **wiki_edit_count_preA** | Nombre de modifications de la page avant Series A. | Activité / intérêt communauté. |
| **wiki_editor_diversity** | Nombre d’éditeurs uniques avant Series A. | 1 = souvent PR ; plusieurs = intérêt public. |
| **wiki_is_disambiguation** | 0/1 : page d’homonymie → exists = 0. | Marque faible ou nom trop commun. |

**Biais notoriété** : le taux wikipedia_exists = 0 peut être >90 % ; ne pas sur-pondérer. **wikipedia_nb_links_out** = exclue du modèle (leakage ; 05 `LEAKAGE_RISK_COLS`).

### News (GDELT)

| Colonne | Définition courte | Interprétation |
|---------|-------------------|----------------|
| **news_mentions_count** | Nombre de mentions GDELT (fenêtre rolling 3 mois). | Exposition médias. |
| **news_sources_count** | Nombre de sources distinctes. | Diversité. |
| **news_mentions_query_used** | Requête utilisée. | Reproductibilité. |

### Backlinks

**Biais** : comptes « toutes périodes » = risque de leakage ; les colonnes **pré–Series A** (si seriesA_date présent) sont à privilégier pour le modèle. Détail : FEATURES_ALL.md § 04c.

| Colonne | Définition courte | Interprétation |
|---------|-------------------|----------------|
| **backlinks_hn_count** | Mentions HN (Algolia) ; toutes périodes. | Visibilité HN. |
| **backlinks_reddit_count** | Mentions Reddit ; toutes périodes. | Visibilité Reddit. |
| **backlinks_github_count** | Repos GitHub contenant le domaine ; toutes périodes. | Liens depuis repos. |
| **backlinks_approx_domains** | Nb de plateformes (0–3) avec ≥ 1 mention ; toutes périodes. | Diversité des canaux. |
| **backlinks_hn_count_preA** | Mentions HN **avant** seriesA_date (Algolia filtre date). | Signal pré–Series A. |
| **backlinks_hn_before_seriesA** | 1 si au moins une mention HN avant Series A. | Binaire pré–Series A. |
| **hn_comments_count_preA** | Somme des commentaires sur les posts HN pré–Series A. | Due diligence sociale. |
| **backlinks_reddit_count_preA** | Mentions Reddit avant seriesA_date (Pushshift si dispo). | Signal pré–Series A Reddit. |
| **backlinks_github_count_preA** | Repos GitHub **créés avant** seriesA_date. | Repos existants avant levée. |
| **backlinks_approx_domains_preA** | Nb de plateformes (0–3) avec mention pré–Series A. | Diversité pré–Series A. |
| **backlinks_signal_note** | Indique si filtre date utilisé ou pas de seriesA_date. | Traçabilité. |

### GitHub

| Colonne | Définition courte | Interprétation |
|---------|-------------------|----------------|
| **github_stars**, **github_forks**, **github_watchers** | Métriques du repo principal. | Popularité / réutilisation. |
| **github_releases**, **github_tags** | Releases / tags. | Maturité. |
| **github_commits_3m**, **github_commits_6m**, **github_commits_12m** | Commits sur 3 / 6 / 12 mois. | Activité de développement. |
| **github_contributors** | Nombre de contributeurs. | Taille équipe open source. |
| **github_issues_open**, **github_issues_closed** | Issues ouvertes / fermées. | Charge / résolution. |
| **github_first_commit_date**, **github_signal_date** | Premier commit / date du signal. | Ancienneté / reproductibilité. |
| **github_before_seriesA** | 1 si repo / premier commit avant ou à la Series A. | Anti‑leakage. |

### Packages (PyPI / npm / crates)

| Colonne | Définition courte | Interprétation |
|---------|-------------------|----------------|
| **pkg_first_release_date**, **pkg_signal_date** | Premier release / date du signal. | Ancienneté. |
| **pkg_releases_count**, **pkg_downloads_recent** | Nombre de releases / téléchargements récents. | Maturité / adoption. |
| **pkg_registry** | PyPI, npm, crates. | Source. |
| **pkg_before_seriesA** | 1 si premier release avant ou à la Series A. | Anti‑leakage. |

### OpenAlex

| Colonne | Définition courte | Interprétation |
|---------|-------------------|----------------|
| **openalex_works_count**, **openalex_cited_by_count** | Nombre de travaux / citations (pré‑Series A). | Production / impact académique. |
| **openalex_works_12m**, **openalex_works_36m** | Travaux sur 12 / 36 mois avant Series A. | Activité récente. |
| **oa_academic_velocity** | Ratio works_12m / works_36m. | Accélération production scientifique. |
| **oa_collaboration_index** | Nb d'institutions distinctes co‑signataires. | Collaboration universités. |
| **oa_h_index_preA** | H‑index sur les travaux avant Series A. | Solidité équipe recherche. |
| **oa_top_tier_ratio**, **oa_patent_citations**, **oa_author_prestige** | *Placeholders* (SJR/IF, citations brevets, prestige auteur). | Qualité revues / science→marché / star scientists. |
| **openalex_signal_note** | Traçabilité (ex. date Series A manquante). | Anti‑leakage. |

### Product Hunt / Hacker News

| Colonne | Définition courte | Interprétation |
|---------|-------------------|----------------|
| **ph_hn_launch** | 1 si post HN (URL domaine) trouvé. | Lancement / notoriété HN. |
| **ph_hn_is_show_hn** | 1 si meilleur match = Show HN (lancement volontaire), 0 si story seule. | Exécution vs mention. |
| **ph_hn_upvotes**, **ph_hn_comments** | Points / commentaires du post. | Engouement / discussion. |
| **hn_comments_per_upvote** | commentaires / (upvotes + 1). | Densité de discussion (deeptech : ratio moyen à élevé). |
| **hn_points_velocity** | *Proxy* points / 24 (points par jour). | Viralité pure. |
| **ph_hn_post_date**, **ph_hn_signal_date** | Date du post / du signal. | Timing. |
| **ph_hn_cohesion_index**, **ph_medal_count**, **ph_maker_reputation**, **hn_comment_sentiment**, **hn_hiring_mentions** | *Placeholders* (écart PH–HN, badges PH, réputation Maker, NLP commentaires, « Who is hiring? »). | Blitz / rang / Serial Makers / validation technique / recrutement HN. |
| **ph_producthunt_launch** | 1 si présence Product Hunt. | Lancement PH. |
| **ph_hn_before_seriesA** | 1 si post avant ou à la Series A. | Anti‑leakage. |

### Colonnes « note » (traçabilité)

**wikipedia_note**, **news_mentions_note**, **github_note**, **pkg_note**, **ph_hn_note** : texte explicatif quand un signal a été trouvé **après** la date de la Series A (features correspondantes mises à 0). Conservées dans le CSV final, **exclues de l’imputation numérique** et **non utilisées comme variables numériques** en 08.

---

## Pipeline

01 → 02 → … → 05 (nettoyage final) → **08** (modèle). Les fichiers de signaux en Pipeline sont numérotés 06–14.

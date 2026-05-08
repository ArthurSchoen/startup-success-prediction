# Résumé des API utilisées dans le pipeline

Ce document liste **toutes les API externes** utilisées dans le code, le **nom de la clé/token** (variable d’environnement) et les **risques** éventuels (clé en dur, fuite, etc.).

---

## 1. APIs sans clé (gratuites, pas d’auth)

| API | URL / source | Notebook | Clé | Note |
|-----|--------------|----------|-----|------|
| **Wayback Machine (CDX)** | `https://web.archive.org/cdx/search/cdx` | 03__wayback | Aucune | User-Agent uniquement. |
| **Wayback available** | `https://archive.org/wayback/available` | 03__wayback | Aucune | |
| **Wikipedia** | `https://en.wikipedia.org/w/api.php` | 04a__wikipedia | Aucune | User-Agent requis (politique d’usage). |
| **Wikimedia Pageviews** | `https://wikimedia.org/api/rest_v1/metrics/pageviews/...` | 04a__wikipedia | Aucune | |
| **GDELT** | `https://api.gdeltproject.org/api/v2/doc/doc` | 04b__news_mentions | Aucune | |
| **Hacker News (Algolia)** | `https://hn.algolia.com/api/v1/search` | 04c__backlinks, 04g__producthunt_hn | Aucune | |
| **Reddit** | `https://www.reddit.com/search.json` | 04c__backlinks | Aucune | Rate limit possible. |
| **Pushshift (Reddit)** | `https://api.pushshift.io/reddit/search/submission/` | 04c__backlinks | Aucune | Service parfois instable. |
| **PyPI** | `https://pypi.org/pypi/{name}/json` | 04e__packages | Aucune | |
| **PyPistats** | `https://pypistats.org/api/packages/{name}/recent` | 04e__packages | Aucune | |
| **npm** | `https://registry.npmjs.org/{name}` | 04e__packages | Aucune | |
| **Crates.io** | `https://crates.io/api/v1/crates/{name}` | 04e__packages | Aucune | |
| **OpenAlex** | `https://api.openalex.org` | 04f__openalex | Aucune | User-Agent + mailto recommandés (politique d’usage). |

---

## 2. APIs avec clé / token (variable d’environnement)

| API | URL | Notebook | Nom de la clé | Où c’est lu | Risque |
|-----|-----|----------|----------------|-------------|--------|
| **Semrush** | `https://api.semrush.com/analytics/ta/api/v3/summary` | 04__webtraction | `SEMRUSH_API_KEY` | `os.getenv("SEMRUSH_API_KEY", "").strip()` | Aucun : pas de clé en dur. Obligatoire si `SKIP_SEMRUSH=False`. |
| **PatentsView** | `https://search.patentsview.org/api/v1` | 03__patentsview | `PATENTSVIEW_API_KEY` | `os.getenv("PATENTSVIEW_API_KEY")` | Set via environment variable. |
| **GitHub** | `https://api.github.com` | 04d__github, 04c__backlinks | `GITHUB_TOKEN` | Dans 04d : `GITHUB_TOKEN = None  # optional: os.environ.get("GITHUB_TOKEN")` (non branché par défaut) | Aucun : pas de clé en dur. Sans token = 60 req/h. |
| **Product Hunt** | `https://api.producthunt.com/v2/api/graphql` | 04g__producthunt_hn | `PRODUCTHUNT_TOKEN` | `token = None  # os.environ.get("PRODUCTHUNT_TOKEN")` (désactivé) | Aucun : pas de clé en dur ; sans token la feature PH reste à 0. |

---

## 3. Synthèse des clés à configurer (double check)

- **SEMRUSH_API_KEY** : nécessaire seulement si tu exécutes l’étape 04 sans `SKIP_SEMRUSH`. À mettre dans l’env (ex. `export SEMRUSH_API_KEY='...'`). Ne jamais committer la clé.
- **PATENTSVIEW_API_KEY** : utilisé en 03. **À corriger** : retirer la valeur par défaut dans le notebook et utiliser uniquement `os.getenv("PATENTSVIEW_API_KEY")` ; si absent, lever une erreur claire. Ne jamais committer la clé.
- **GITHUB_TOKEN** : optionnel ; pour 04d (et plus de requêtes en 04c si tu l’utilises). À définir via `export GITHUB_TOKEN=’...’` et à brancher dans le code (ex. `GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")`). Ne jamais committer le token.
- **PRODUCTHUNT_TOKEN** : optionnel ; pour 04g (Product Hunt). Actuellement désactivé. Si tu l’actives, utiliser `os.environ.get("PRODUCTHUNT_TOKEN")` et ne jamais committer le token.

---

## 4. Variable de pipeline (pas une API)

- **PIPELINE_RUN_ID** : utilisé dans plusieurs notebooks pour choisir le run (fichier `.run_id` ou env). Ce n’est pas un secret API.

---

## 5. Action recommandée (risque identifié)

**PatentsView (03__patentsview__signals.ipynb)** : une clé API par défaut est codée en dur :

```python
API_KEY = os.getenv("PATENTSVIEW_API_KEY", "").strip()
```

- **Risque** : la clé peut être commitée, copiée, ou réutilisée par quelqu’un d’autre.
- **Correction** : utiliser uniquement `os.getenv("PATENTSVIEW_API_KEY")` (sans valeur par défaut) et lever une erreur si la clé est absente quand l’API est utilisée. Obtenir sa propre clé sur [PatentsView](https://patentsview.org/apis/api-key).

Après correction, faire un **double check** : recherche dans tout le repo de chaînes du type `skeLQVxl`, `PATENTSVIEW`, `Bearer `, `api_key`, `secret` pour s’assurer qu’aucune autre clé n’est en dur.

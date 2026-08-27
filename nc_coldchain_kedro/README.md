# NC Cold-Chain — End-to-End Kedro Data, Graph, Monte-Carlo & Application System

A single **connected data product** for Northern Cape cold-chain telematics. Raw
and synthetic data move through a Kedro pipeline lifecycle — extraction → cleaning
→ **audit** → **audit feedback** → graph construction → **graph optimisation** →
Monte-Carlo risk → reporting — and a FastAPI backend + Streamlit frontend consume
the versioned artifacts the pipeline produces. The graph is a *living output of the
data*, not a hard-coded file: when graph-relevant data changes, the audit detects
it, the graph is rebuilt and re-optimised, a new version is stamped, and the app
shows the new optimal graph.

```
docker compose up --build
```

runs the whole thing with no manual `kedro run` / `python backend.py` /
`npm run dev` steps.

---

## 1. Architecture

```
                DATA SOURCES
   map registry ┆ synthetic streams (A/B/C) ┆ driver reports
        │                │                        │
        ▼                ▼                        ▼
 ┌─────────────────────────────────────────────────────────┐
 │  KEDRO — orchestration & data-processing backbone        │
 │                                                          │
 │  data_extraction ─┐                                      │
 │                   ├─► preprocessing ─► data_audit ──┐    │
 │  synthetic_data ──┘                                 │    │
 │                                    audit_feedback ◄─┘    │
 │                                        │                 │
 │                                        ▼                 │
 │                                   graph (build/validate) │
 │                                        │                 │
 │                                        ▼                 │
 │                               graph_optimization         │
 │                              (versioned optimal graph)   │
 │                                   │          │           │
 │                          ┌────────┘          └────────┐  │
 │                          ▼                            ▼  │
 │                     monte_carlo                   reporting│
 └─────────────────────────┬───────────────────────────┬───┘
                           ▼                            ▼
                    data/07,08 artifacts        data/09 reports + system_status
                           └─────────────┬──────────────┘
                                         ▼
                              FastAPI backend  (reads artifacts only)
                                         ▼
                              Streamlit frontend (latest optimal graph)
```

The backend holds **no** processing logic; it serves artifacts. The frontend loads
**no** files directly; it calls the backend. Synthetic data is **not** an
afterthought — it is generated inside the pipeline and flows the same path as any
real data.

## 2. Data lifecycle (Kedro layers)

```
data/01_raw          map registry, boundary, synthetic streams, driver reports
data/02_intermediate standardised tables (schemas, types, booleans)
data/03_primary      cleaned tables (dedup, imputed)
data/04_feature      graph node table + graph edge table
data/05_model_input  monte_carlo_params.json (ambient grounded in observed data)
data/06_models       (reserved — see §11 Model serving)
data/07_model_output monte_carlo_results.json
data/08_graph        raw/ (graph.pkl, validation) · processed/ (metrics)
                     optimized/ (optimal_graph.pkl+.json, metadata, candidates)
                     visualizations/ (optimal_graph.geojson)
data/09_reporting    audit/ · data_quality · graph_change · simulation · system_status · data_lineage.log
```

Raw sources are never silently overwritten; generated layers are the ones Docker
persists in the `nc_data` volume. `.gitignore` tracks raw, ignores generated.

## 3. Pipelines & nodes

| Pipeline | Key nodes | Produces |
|---|---|---|
| `data_extraction` | extract_osm_roads (downloads OSM + slices to NC), extract_town_registry, extract_boundary, extract_drivers_reports, fetch_live_weather | raw OSM extract + map/boundary/driver/weather data |
| `synthetic_data` | generate_synthetic_streams (real Streams A/B/C generator) | raw weather/road/sensor CSVs |
| `preprocessing` | standardize_schema, clean_data, assemble_graph_inputs (real OSM / synthetic), prepare_model_data | intermediate/primary/feature/model_input |
| `data_audit` | audit_run_checks, audit_generate_report, audit_generate_feedback | audit_results, audit_report, **audit_feedback** |
| `graph` | construct_graph (boundary-clipped), validate_graph, calculate_graph_metrics | raw_graph, validation, metrics |
| `graph_optimization` | evaluate_candidates, build_optimal_graph (versioned), publish_graph | optimal_graph(.pkl/.json), metadata, geojson |
| `monte_carlo` | prepare_monte_carlo_inputs, run_monte_carlo | monte_carlo_results |
| `reporting` | data_quality, graph_change, simulation, system_status | 09_reporting artifacts |

Registered slices: `kedro run` (full), `kedro run --pipeline synthetic_data`,
`--pipeline rebuild` (audit → reporting, for when new data arrives).

## 4. Data catalog

Every dataset is declared in `conf/base/catalog.yml` with an explicit layer; no
node hard-codes a path. Types used: `pandas.CSVDataset`, `pandas.ParquetDataset`,
`json.JSONDataset`, `pickle.PickleDataset`, and a **custom
`NetworkXGraphDataset`** (node-link JSON) for the published graph.

## 5. Parameters

All thresholds, seeds, weights, counts, town coordinates, optimisation candidates
and MC constants live in `conf/base/parameters.yml`. Nothing operational is
hard-coded in Python.

## 6. Data-audit mechanism

`data_audit` is a **first-class pipeline**, not a helper. It produces
machine-readable `audit_results.json` (per-table schema, missing fractions,
duplicates, outliers via z-score, range violations, distributions, plus
graph-input stats), a flat `audit_report.csv`, and — critically —
`audit_feedback.json`.

## 7. Feedback mechanism (real, not fake)

`audit_generate_feedback` compares the **current** graph inputs against the
**previously published** graph's `metadata.json` and emits:

```json
{ "data_valid": true, "first_build": false,
  "graph_rebuild_required": true, "optimization_required": true,
  "reason": "graph-relevant change detected",
  "deltas": {"node_delta": 5, "edge_delta": 8, "mean_edge_weight_pct": 0.0},
  "changed_nodes": ["node_count 32 -> 37"], "changed_edges": [...],
  "quality_issues": [], "previous": {"graph_version": "g-..."} }
```

Verified behaviour: identical inputs → `graph_rebuild_required=false`
("no material change"); +5 nodes → `true` with the detected delta. The downstream
graph node consumes this object and logs the rebuild reason.

## 8. Graph generation & 9. optimisation

**The graph is the REAL OpenStreetMap road network, and it is pulled from OSM
by the pipeline itself** (`network.road_source: osm`, `network.download_osm: true`).
`extract_osm_roads` is the Kedro-native port of your `fast_nc_roads.py`: it
downloads the Geofabrik South Africa road layer, slices it to the Northern Cape
bbox, and writes `data/01_raw/map/northern_cape_roads.gpkg` (idempotent — it
reuses an existing extract). `assemble_graph_inputs` then calls your validated
`nc_road_network_improved.build_clean_network()` on it — load → clean/enrich
(province-clipped) → 150 m snap → component stitch → largest routable component —
and snaps the seven towns onto their nearest real road nodes.

This produces a **single source of truth**: the bundle
`{G_main, main_nodes, cluster_coord, town_nodes}` — the exact object your
`build_clean_network()` returns. Everything downstream derives from it: the audit
feature tables, the metrics, the optimisation, the Monte-Carlo, and the published
graph. There is no second, divergent topology anywhere.

`evaluate_candidates` re-weights that real graph's `spoilage_cost_edge` under each
candidate blend and scores it by mean optimal spoilage cost over the OD pairs
(Dijkstra over the actual road network); `build_optimal_graph` bakes the winning
weights in and emits **`data/08_graph/optimized/nc_road_graph.pkl`** — the artifact
`live_backend.py` loads.

Paths and the bbox live in `parameters.yml` (never hard-coded). If `road_source:
osm`, `download_osm: false`, and no extract is present, the run **fails loudly**
unless you set `network.allow_synthetic_fallback: true`. `road_source: synthetic`
is a clearly-logged offline corridor abstraction (used by CI).

## 10. Graph versioning

`metadata.json` carries `graph_version`, `created_at`, `source_data_version`,
`audit_version`, `optimization_version`, `objective_score`, `node_count`,
`edge_count`, weights and `rebuild_reason`. The frontend header shows the active
version; `graph_change_report.json` shows previous → new.

## 11. Model serving

The "model" here is the optimal graph + Monte-Carlo risk model. The pipeline
**produces** these artifacts (data/07, data/08); the backend **consumes** them and
never recomputes. `data/06_models` is reserved for a future trained predictor;
training would be its own pipeline feeding `06_models`, with serving reading the
artifact — the split is already in the layer layout.

## 12. Backend

FastAPI (`app/backend/main.py`). Endpoints: `/health`, `/status`, `/graph`,
`/graph/geojson`, `/graph/metadata`, `/graph/metrics`, `/graph/candidates`,
`/audit`, `/monte-carlo`, `/reports`, `/reports/graph-change`. A missing artifact
returns **503**, never fabricated data.

## 13/23. Frontend & graph

Streamlit (`app/frontend/streamlit_app.py`) renders the latest optimal graph on a
Folium/OSM map from `/graph/geojson`, plus audit, Monte-Carlo and change-report
tabs. No static graph; it always reflects the newest published artifact.

## 14. Docker architecture

Three services from one image (`docker-compose.yml`):

```
pipeline  (kedro run) ──completes──► backend (uvicorn) ──healthy──► frontend (streamlit)
                 └──────────── shared nc_data volume (artifacts) ───────────┘
```

`backend` uses `depends_on: pipeline: service_completed_successfully` so it never
serves before artifacts exist; `frontend` waits for the backend healthcheck.

## 15. Environment variables

`ENVIRONMENT, DATA_PATH, API_HOST/API_PORT, FRONTEND_PORT, RUN_PIPELINE_ON_START,
USE_SYNTHETIC_DATA, RANDOM_SEED, LOG_LEVEL` — see `.env.example`. An optional real
`API_KEY` (OpenWeatherMap) is read from the environment only, never stored or
committed.

## 15b. Weather API key & credentials (and the "key not working" fix)

**The bug you hit:** `conf/local/credentials.yml` overrides `conf/base`, and it
shipped with a placeholder (`.........`). So even with a real key in `conf/base`,
the placeholder in `conf/local` won overriding → the hook saw a placeholder and
logged *"No real OpenWeatherMap key configured"*. Fixed: `conf/local` now ships
**commented out** (no override).

**To set your key**, uncomment the two lines in **`conf/local/credentials.yml`**
(git-ignored, secure) and paste it:

```yaml
openweathermap:
  API_KEY: "your-real-key"
```

(Or paste it into `conf/base/credentials.yml` — committed, less secure.) At
startup `CredentialsToEnvHook` exports the first real (non-placeholder) key it
finds so `weather_engine.resolve_owm_api_key()` uses it — the provider becomes
`openweathermap`; without a key it falls back to Open-Meteo. The key is never
logged, returned, or (in `conf/local`) committed. Enable a live-weather call that
feeds the Monte-Carlo ambient with `runtime.use_live_weather: true`. Confirm it
worked: the startup log prints `[WEATHER] OpenWeatherMap key loaded from
credentials`.

## 16. Development setup & running the app

```bash
pip install -r requirements.txt
pip install -e .

# 1) run the pipeline (produces all artifacts under data/)
kedro run                              # full: real-OSM graph -> optimise -> MC -> reports
kedro run --pipeline synthetic_data    # just regenerate synthetic streams
kedro run --pipeline rebuild           # audit->reporting only (after new data lands)

# 2) start YOUR live app (AI + weather + telematics + routing), reading the
#    graph the pipeline just produced:
NC_ROAD_GRAPH=data/08_graph/optimized/nc_road_graph.pkl \
  uvicorn app.live_backend:app --host 0.0.0.0 --port 8000
#    (AI needs: pip install transformers torch)

# 3) start your 5-tab Streamlit app against that backend:
BACKEND_URL=http://localhost:8000 streamlit run app/live_frontend.py   # -> :8501

# optional pipeline-output-only view instead of the live app:
uvicorn app.backend.main:app --port 8100
BACKEND_URL=http://localhost:8100 streamlit run app/frontend/streamlit_app.py

# optional: inspect the pipeline DAG / lineage
kedro viz run                          # http://localhost:4141
```

Or run the whole product in one command with Docker (§17):
`docker compose up --build` → pipeline → live-backend (:8000) → live-frontend (:8501).

## 17. Docker setup

```bash
cp .env.example .env      # optional; defaults work
docker compose up --build
# frontend  -> http://localhost:8501
# backend   -> http://localhost:8000/health
```

## 18. Testing

```bash
pytest                    # unit (pipelines) + integration + app-contract
```

`tests/integration/test_end_to_end_logic.py` runs every node in dependency order
using the real parameters — **without** the Kedro runtime — proving the connected
product produces a versioned optimal graph and Monte-Carlo result from synthetic
data alone. Node functions are import-guarded so they test without Kedro installed.

## 19. Introducing new data

Drop new files into `data/01_raw/**` (bind-mounted in compose), then
`kedro run` (or `kedro run --pipeline rebuild`). The audit detects graph-relevant
change, feedback flags a rebuild, the graph is re-optimised, a **new version** is
stamped, Monte-Carlo re-runs, and the backend serves the new graph on the next
frontend refresh.

## 20. How a new optimal graph is produced

`new data → preprocess → audit → feedback(rebuild=true) → construct_graph →
evaluate_candidates → build_optimal_graph(new version) → publish → monte_carlo →
reporting → backend → frontend`. Incremental vs full: raw ingestion and audit are
cheap and always run; graph/optimisation/Monte-Carlo rerun when their inputs
change (Kedro dataset dependencies drive this, not timestamp hacks).

---

### Deliverable acceptance

- **Critical test** (`docker compose up --build`): pipeline generates synthetic
  data → cleans → audits → feedback → builds & optimises graph → Monte-Carlo →
  reports → backend serves → frontend shows the new optimal graph. Verified in
  pure-Python form by the integration test.
- **New-data test**: verified — unchanged inputs produce no rebuild; changed inputs
  produce a new graph version.

## Where is the AI? (now integrated)

Your **Qwen strategy/report LLM, the weather tab, driver action items, the
streaming Monte-Carlo UI and Traccar telematics** live in `app/live_backend.py` /
`app/live_frontend.py` — your original real-time app, now the **primary product**
of this repo. It is wired to consume the pipeline's output: `live_backend.py`
loads its graph from `data/08_graph/optimized/nc_road_graph.pkl` (via the
`NC_ROAD_GRAPH` env var) instead of building its own at startup. `docker compose
up --build` runs `pipeline → live-backend → live-frontend`, so the AI app sits on
top of the Kedro-produced real graph. The lightweight artifacts API + graph viewer
(`app/backend`, `app/frontend`) remain available under `--profile artifacts`.

The AI LLM (Qwen) is loaded lazily and fail-safe by `live_backend.py`; to enable
it install the extra `pip install transformers torch` (kept out of the base image
to stay lean). Without it, routing/risk/weather still work and AI calls degrade
gracefully.

### Known limitations / what to verify on your machine

The OSM download + `build_clean_network` + the live app all require your
environment (geopandas/pyogrio, internet to Geofabrik, and for AI transformers/
torch) and were **not runnable in the authoring sandbox**, so verify on first run:
(1) `data/08_graph/processed/graph_metrics.json` shows a **thousands-node** graph,
not the ~30-node synthetic one; (2) `data/08_graph/optimized/nc_road_graph.pkl`
exists and `live_backend.py` loads it (its startup log prints the node count);
(3) the weather tab shows `source: openweathermap` once your key is set (§15b).
A trained ML predictor in `06_models` and an automatic file-watcher/API trigger
for continuous re-runs are scaffolded but not wired to run continuously.

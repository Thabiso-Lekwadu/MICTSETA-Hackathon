# Northern Cape Freight Optimizer

MICT SETA Skills Development Hackathon, Northern Cape, 28-29 Aug 2026
Track: Transport, Trade and Fisheries

Spoilage-risk-aware route optimization for cold-chain freight on Northern Cape
roads, built on a real OpenStreetMap extract, with a live GPS-tracked dispatch
dashboard and crowd-sourced driver ground-truth reporting.

## What this actually is

Two things, kept deliberately separate:

1. **A Kedro data pipeline** (`src/nc_freight_optimizer/pipelines/`) that turns a
   raw OSM road extract into a routable network graph, with a documented, tiered
   speed-imputation method and a full structural/connectivity audit. Reproducible
   with one command.
2. **Live applications** (`src/nc_freight_optimizer/apps/`) that consume that
   graph: a FastAPI backend serving live vehicle telemetry (real Traccar Client
   phone GPS or a built-in simulator) and spoilage-optimized routing, and a
   Streamlit dashboard for dispatch tracking and field reporting.

See `docs/architecture.md` for why these are separate and how they share code, and
`docs/data_audit_findings.md` for what the data actually says (including its
limitations) before you quote a number from it.

## Project structure

```
conf/base/            catalog.yml (dataset definitions), parameters.yml (all tunable constants)
data/01_raw/           the raw recovered OSM road extract
data/02_intermediate/  validated + cleaned/enriched road table
data/03_primary/       the routable graph artifacts, incl. nc_road_graph.pkl (what the live backend loads)
data/08_reporting/      route_sensitivity_scan.csv -- the batch route-comparison report
docs/                   architecture notes and data audit findings
notebooks/              exploratory notebooks (historical, superseded by the pipeline -- see note inside each)
src/nc_freight_optimizer/
  road_network_core.py  pure data-cleaning functions (imputation, graph construction)
  routing.py             spatial indices + route-scoring, shared by the pipeline and the live backend
  pipelines/
    road_network/        extract -> validate -> clean -> build graph -> bundle
    route_reporting/      batch Standard-vs-Optimized sensitivity scan
  apps/
    live_backend.py       FastAPI server (port 8000)
    live_frontend.py      Streamlit dashboard
tests/                   pytest suite, including a real end-to-end pipeline run
```

## Setup

```bash
pip install -e ".[dev]"
```

or with conda: `conda env create -f environment.yml`

## Running the data pipeline

```bash
kedro run
```

Regenerates everything under `data/02_intermediate/`, `data/03_primary/`, and
`data/08_reporting/` from the raw extract in `data/01_raw/`. Takes about 10 seconds.
Run `kedro test` to run the pytest suite (17 tests, including a full pipeline
execution test).

## Running the live demo

Two terminals, from the project root:

```bash
# Terminal 1
uv run src/nc_freight_optimizer/apps/live_backend.py

# Terminal 2
uv run streamlit run src/nc_freight_optimizer/apps/live_frontend.py
```

The backend must have `data/03_primary/nc_road_graph.pkl` available -- run
`kedro run` first if you haven't already.

### Demo tip

Not every road segment triggers a visible reroute when reported as impassable --
the network is mostly a single paved corridor with few real alternatives at
province scale (see `docs/data_audit_findings.md`). A verified location that does
reliably trigger a visible detour: report **lat -29.307839, lon 17.138515** (N7 near
Springbok) as "Impassable / Washed Out" in the Driver Ground-Truth Form.

## Status

`live_frontend.py` and `live_backend.py` are under active revision (dual telemetry
feed support: real Traccar Client hardware GPS alongside the built-in simulator).
The versions in this repo are integrated and smoke-tested against the real graph,
but expect them to keep changing.

# Architecture

## Why Kedro, and where it stops

This project has two genuinely different kinds of code, and they are kept apart
deliberately:

1. **A batch data pipeline** (`src/nc_freight_optimizer/pipelines/`): load the raw
   OSM road extract, clean it, build a routable graph, run a sensitivity report.
   This is reproducible, deterministic, and re-runnable with `kedro run`. Kedro is
   the right tool for exactly this shape of problem.

2. **Live services** (`src/nc_freight_optimizer/apps/`): a FastAPI backend with
   request-scoped mutable state (vehicle position, active driver reports) and a
   Streamlit dashboard. These are long-running processes that respond to real-time
   events, not batch transformations with a fixed input and output. Forcing them
   into Kedro nodes would be dishonest engineering -- a node is supposed to be a
   pure, re-runnable function over versioned data, and a live GPS feed is neither
   pure nor versioned.

The apps consume the pipeline's output artifact (`data/03_primary/nc_road_graph.pkl`)
directly. They do not go through a `KedroSession` at runtime -- that machinery is for
orchestrating the batch DAG, not for a process that's already running and serving
requests.

## Shared logic, not duplicated logic

Both sides import from the same two library modules, so "how a route is scored" is
defined in exactly one place:

- `road_network_core.py` -- pure data-cleaning functions (imputation, graph
  construction). Used by the pipeline nodes and importable directly from a notebook.
- `routing.py` -- spatial indices and the route-scoring weight functions
  (`effective_weight`, `baseline_weight`, `compare_routes`). Used by
  `pipelines/route_reporting` for the batch sensitivity scan AND by
  `apps/live_backend.py` for live routing. A driver report changes both the same way.

The Kedro `nodes.py` files in `pipelines/` are intentionally thin: they exist to
adapt catalog inputs/parameters to the core functions, not to hold logic themselves.

## Data flow

```
[internet]  Geofabrik south-africa-latest-free.shp.zip
        |  extract_raw_roads (download, validate, extract shapefile, clip to NC bbox)
        v
data/01_raw/extracted_full_roads.pkl
        |  validate_raw_roads (structural/bbox audit)
        v
data/02_intermediate/validated_roads.pkl
        |  clean_and_enrich_roads (filter, simplify, tiered speed imputation)
        v
data/02_intermediate/cleaned_roads.pkl, speed_lookup.pkl
        |  build_road_graph (endpoint snapping, vertex-chain graph construction)
        v
data/03_primary/road_graph_full.pkl, cluster_coord.pkl
        |  extract_main_component (largest connected component)
        v
data/03_primary/road_graph_main.pkl, main_nodes.pkl
        |  bundle_graph_artifacts
        v
data/03_primary/nc_road_graph.pkl  <-- apps/live_backend.py loads this at startup
        |  run_route_sensitivity_scan (route_reporting pipeline)
        v
data/08_reporting/route_sensitivity_scan.csv
```

Run the whole thing with `kedro run` from the project root. Run `kedro viz` (if
installed) to see this as an interactive graph.

## Known, documented limitations

See `docs/data_audit_findings.md` for the full detail. In short: ~65% of the road
network's speed values are imputed defaults rather than observed data, and the raw
extract's bounding box legitimately bleeds slightly into North West Province. Both
are stated explicitly rather than hidden, because a judge or teammate asking "where
does this number come from" deserves an honest answer.

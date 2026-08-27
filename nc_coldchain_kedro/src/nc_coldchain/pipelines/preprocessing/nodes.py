"""preprocessing nodes: standardize -> clean -> prepare graph & model inputs."""
from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .._common import geodesic_km, point_in_boundary

logger = logging.getLogger("nc_coldchain")

_BOOL_COLS = {"heat_alert", "rain_alert", "failure_event", "moving", "spoilage_breach"}


# ----------------------------------------------------------------- standardize --
def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "timestamp_iso" in df.columns:
        df["timestamp_iso"] = df["timestamp_iso"].astype(str)
    for c in df.columns:
        if c in _BOOL_COLS:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def standardize_schema(weather, road, sensors, drivers):
    logger.info("[PREPROCESSING] Standardising schemas of 4 raw tables ...")
    return _standardize(weather), _standardize(road), _standardize(sensors), _standardize(drivers)


# ----------------------------------------------------------------------- clean --
def _clean(df: pd.DataFrame, params: dict, key_cols) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    present_keys = [k for k in key_cols if k in df.columns]
    if present_keys:
        df = df.drop_duplicates(subset=present_keys, keep="first")
    strat = params.get("impute_numeric_strategy", "median")
    for c in df.select_dtypes(include="number").columns:
        if df[c].isna().any():
            fill = df[c].median() if strat == "median" else df[c].mean()
            df[c] = df[c].fillna(fill)
    logger.info("[PREPROCESSING] cleaned %d -> %d rows (dropped %d dup)",
                before, len(df), before - len(df))
    return df


def clean_data(weather, road, sensors, drivers, preprocessing_params):
    keys = preprocessing_params.get("drop_duplicate_keys", ["hour_index", "timestamp_iso"])
    return (
        _clean(weather, preprocessing_params, keys),
        _clean(road, preprocessing_params, keys),
        _clean(sensors, preprocessing_params, keys),
        _clean(drivers, preprocessing_params, ["report_id"]),
    )


# --------------------------------------------------- prepare graph node table ---
def build_graph_nodes(town_registry: dict, boundary: dict, network_params: dict) -> pd.DataFrame:
    """Town nodes + synthetic corridor waypoints, all inside the NC boundary."""
    poly = boundary.get("boundary", [])
    towns = town_registry["towns"]
    rows = []
    for name, (lon, lat) in towns.items():
        rows.append({"node_id": f"T_{name}", "name": name, "lon": float(lon),
                     "lat": float(lat), "kind": "town"})
    # waypoints are added by the edge builder; here we emit towns only.
    df = pd.DataFrame(rows)
    if poly:
        inside = df.apply(lambda r: point_in_boundary(r["lon"], r["lat"], poly), axis=1)
        outside = int((~inside).sum())
        if outside:
            logger.warning("[PREPROCESSING] %d town(s) outside boundary (kept, buffered)", outside)
    logger.info("[PREPROCESSING] graph node table: %d towns", len(df))
    return df


# --------------------------------------------------- prepare graph edge table ---
def _road_roughness(road: pd.DataFrame) -> float:
    if "iri" in road.columns and len(road):
        return float(pd.to_numeric(road["iri"], errors="coerce").dropna().mean() or 2.0)
    return 2.0


def build_graph_edges(nodes: pd.DataFrame, road: pd.DataFrame, network_params: dict,
                      graph_params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a connected corridor network (MST + nearest-neighbour) with waypoints.

    Returns (edges, augmented_nodes) — waypoint nodes are appended to the node
    table so downstream graph construction sees a complete, connected topology.
    """
    towns = nodes[nodes["kind"] == "town"].reset_index(drop=True)
    default_speed = float(graph_params.get("default_speed_kmh", 90.0))
    base_rough = _road_roughness(road)
    wpp = int(network_params.get("waypoints_per_edge", 2))

    # complete distance graph -> MST guarantees connectivity, then add each town's
    # 2 nearest neighbours for realistic redundancy.
    dg = nx.Graph()
    coords = {r["node_id"]: (r["lon"], r["lat"]) for _, r in towns.iterrows()}
    ids = list(coords)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            km = geodesic_km(coords[a][0], coords[a][1], coords[b][0], coords[b][1])
            dg.add_edge(a, b, weight=km)
    keep = set(nx.minimum_spanning_tree(dg, weight="weight").edges())
    for a in ids:
        nbrs = sorted(((geodesic_km(*coords[a], *coords[b]), b) for b in ids if b != a))[:2]
        for _, b in nbrs:
            keep.add(tuple(sorted((a, b))))

    edge_rows, wp_rows = [], []
    for a, b in keep:
        (alon, alat), (blon, blat) = coords[a], coords[b]
        chain = [a]
        for k in range(1, wpp + 1):
            f = k / (wpp + 1)
            wlon, wlat = alon + (blon - alon) * f, alat + (blat - alat) * f
            wid = f"W_{a}_{b}_{k}"
            wp_rows.append({"node_id": wid, "name": "", "lon": round(wlon, 5),
                            "lat": round(wlat, 5), "kind": "waypoint"})
            chain.append(wid)
        chain.append(b)
        allc = {**coords, **{r["node_id"]: (r["lon"], r["lat"]) for r in wp_rows}}
        for u, v in zip(chain[:-1], chain[1:]):
            km = geodesic_km(*allc[u], *allc[v])
            edge_rows.append({
                "u": u, "v": v, "length_km": round(km, 3),
                "maxspeed": default_speed,
                "travel_time_hr": round(km / default_speed, 4),
                "roughness": round(base_rough, 3),
            })
    edges = pd.DataFrame(edge_rows)
    aug_nodes = pd.concat([nodes, pd.DataFrame(wp_rows)], ignore_index=True)
    logger.info("[PREPROCESSING] graph edges=%d, waypoints=%d, total nodes=%d",
                len(edges), len(wp_rows), len(aug_nodes))
    return edges, aug_nodes


# ============================================================================
# PRIMARY graph assembly. Produces the SAME bundle live_backend.py consumes
# ({G_main, main_nodes, cluster_coord, town_nodes}) plus the derived feature
# tables for the audit/metrics/viz. REAL OSM by default; synthetic only as a
# clearly-flagged offline fallback.
# ============================================================================
def assemble_graph_inputs(town_registry: dict, osm_ready: dict, road: pd.DataFrame,
                          network_params: dict, graph_params: dict):
    """Return (road_graph_bundle, feat_graph_edges, feat_graph_nodes).

    road_source='osm'  -> build the REAL network from network.osm_road_path via
                          build_clean_network. Missing extract FAILS LOUDLY unless
                          network.allow_synthetic_fallback is explicitly true.
    road_source='synthetic' -> corridor abstraction (clearly logged).

    `osm_ready` is the marker from the OSM-extraction node, threaded in only to
    order this node after extraction; its 'path' (if present) overrides the param.
    """
    from . import graph_build as gb

    towns = town_registry["towns"]
    source = str(network_params.get("road_source", "osm")).lower()
    if source == "osm":
        path = Path((osm_ready or {}).get("path") or network_params["osm_road_path"])
        if path.exists():
            bundle = gb.build_osm_bundle(str(path), towns,
                                         use_spatial_knn=True, stitch=True)
        else:
            msg = (f"road_source='osm' but the OSM extract was not found at '{path}'. "
                   f"Enable network.download_osm to fetch it, or place your "
                   f"northern_cape_roads.gpkg / extracted_full_roads.geojson there.")
            if network_params.get("allow_synthetic_fallback", False):
                logger.warning("[PREPROCESSING] %s FALLING BACK to SYNTHETIC — NOT the "
                               "real road network.", msg)
                bundle = gb.build_synthetic_bundle(towns, network_params, graph_params, road)
            else:
                raise FileNotFoundError(
                    msg + " To permit a synthetic fallback (NOT for production) set "
                    "network.allow_synthetic_fallback: true in parameters.yml.")
    else:
        logger.warning("[PREPROCESSING] road_source='%s' — SYNTHETIC corridor topology "
                       "(illustrative, not real OSM routes).", source)
        bundle = gb.build_synthetic_bundle(towns, network_params, graph_params, road)

    nodes_df, edges_df = gb.bundle_to_tables(
        bundle["G_main"], bundle["cluster_coord"], bundle["town_nodes"])
    logger.info("[PREPROCESSING] feature tables: %d nodes / %d edges (7 towns snapped)",
                len(nodes_df), len(edges_df))
    return bundle, edges_df, nodes_df


# ------------------------------------------------------- prepare model inputs ---
def prepare_model_data(sensors: pd.DataFrame, live_weather: dict,
                       monte_carlo_params: dict) -> dict:
    """Assemble Monte-Carlo inputs. Ambient priority: live OpenWeatherMap reading
    (when enabled) > observed sensor 90th percentile > configured scenario."""
    ambient = monte_carlo_params.get("ambient_temp_c", 32.0)
    ambient_source = "scenario_param"
    if "ambient_temp_c" in sensors.columns and len(sensors):
        obs = pd.to_numeric(sensors["ambient_temp_c"], errors="coerce").dropna()
        if len(obs):
            ambient = float(round(obs.quantile(0.9), 2))  # plan for a hot-ish day
            ambient_source = "observed_sensors_p90"
    if isinstance(live_weather, dict) and live_weather.get("enabled") \
            and live_weather.get("ambient_temp_c") is not None:
        ambient = float(live_weather["ambient_temp_c"])
        ambient_source = f"live_{live_weather.get('source', 'openweathermap')}"
    out = dict(monte_carlo_params)
    out["ambient_temp_c"] = ambient
    out["ambient_source"] = ambient_source
    logger.info("[PREPROCESSING] MC ambient=%.1f C (source=%s)", ambient, ambient_source)
    out["observed_peak_cargo_c"] = (
        float(pd.to_numeric(sensors.get("cargo_temp_c"), errors="coerce").max())
        if "cargo_temp_c" in sensors.columns else None
    )
    return out

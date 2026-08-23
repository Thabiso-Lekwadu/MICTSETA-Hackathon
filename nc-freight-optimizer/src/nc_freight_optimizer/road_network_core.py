"""
road_network_core.py

Pure, catalog-agnostic functions for cleaning and structuring the Northern Cape road
network extract. These are the actual data transformations; the Kedro nodes in
pipelines/road_network/nodes.py are thin wrappers around this module that handle the
catalog/parameter boundary. Kept separate so the same logic is importable directly
from a notebook, a test, or a live application without going through a Kedro session.
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.spatial import cKDTree

GEOD = Geod(ellps="WGS84")

TOWNS: dict[str, tuple[float, float]] = {
    "Kimberley": (24.7499, -28.7282), "Upington": (21.2561, -28.4478),
    "Springbok": (17.8865, -29.6644), "Kuruman": (23.4333, -27.4531),
    "De Aar": (24.0129, -30.6497), "Calvinia": (19.7761, -31.4707),
    "Port Nolloth": (16.8667, -29.2500),
}
BORDER_POSTS: dict[str, tuple[float, float]] = {
    "Vioolsdrift": (17.7500, -28.7500),
}


# ---------------------------------------------------------------------------
# Structural / completeness audit (used by the road_network pipeline's
# validation node, and directly by Data_Audit.ipynb)
# ---------------------------------------------------------------------------
def audit_structure(gdf: gpd.GeoDataFrame, expected_bbox: dict[str, float],
                     bbox_tolerance_deg: float = 0.5) -> dict:
    bounds = gdf.total_bounds
    in_bbox = (
        bounds[0] >= expected_bbox["min_lon"] - bbox_tolerance_deg
        and bounds[2] <= expected_bbox["max_lon"] + bbox_tolerance_deg
        and bounds[1] >= expected_bbox["min_lat"] - bbox_tolerance_deg
        and bounds[3] <= expected_bbox["max_lat"] + bbox_tolerance_deg
    )
    return {
        "row_count": len(gdf),
        "bbox_within_expected_extent": bool(in_bbox),
        "duplicate_osm_id_count": int(gdf["osm_id"].duplicated().sum()),
        "invalid_geometry_count": int((~gdf.geometry.is_valid).sum()),
        "empty_geometry_count": int(gdf.geometry.is_empty.sum()),
    }


def validate_raw_roads(gdf: gpd.GeoDataFrame, expected_bbox: dict[str, float],
                        bbox_tolerance_deg: float = 0.5) -> gpd.GeoDataFrame:
    """
    The 'extract' step's validation gate: audits structural integrity and raises if the
    raw data doesn't look like a Northern Cape road extract at all (wrong bbox, no rows).
    Passes the GeoDataFrame through unchanged on success -- this node exists for the
    audit trail and fail-fast behaviour, not to transform anything.
    """
    audit = audit_structure(gdf, expected_bbox, bbox_tolerance_deg)
    if audit["row_count"] == 0:
        raise ValueError("Raw road extract is empty.")
    if not audit["bbox_within_expected_extent"]:
        raise ValueError(
            f"Raw road extract bounding box {gdf.total_bounds} falls outside the "
            f"expected Northern Cape extent {expected_bbox}. Wrong region extracted?"
        )
    return gdf


# ---------------------------------------------------------------------------
# Tiered maxspeed imputation (see docstring for the full method description)
# ---------------------------------------------------------------------------
def impute_maxspeed(
    gdf: gpd.GeoDataFrame,
    default_speed_kmh: dict[str, float],
    min_group_n: int = 10,
    min_fclass_n: int = 30,
    min_parent_n: int = 5,
    speed_floor_kmh: float = 5.0,
) -> gpd.GeoDataFrame:
    """
    OSM's `maxspeed == 0` means "unknown", not "0 km/h" -- a vehicle cannot travel at
    0 km/h, so treating the zero literally would break every downstream travel-time
    calculation. Every 0/missing value is replaced using the most specific real signal
    available, falling back progressively:

      Level 0  observed          -- a real, non-zero maxspeed value. Trusted as-is.
      Level 1  fclass_ref_median -- median of real values sharing (fclass, ref prefix),
                                     e.g. ("primary", "R") for provincial R-roads.
                                     Used only where >= min_group_n real samples exist.
      Level 2  fclass_median     -- median of real values for the fclass alone, used
                                     only where >= min_fclass_n real samples exist.
      Level 3  parent_class      -- *_link classes borrow their base class's median
                                     (trunk_link -> trunk). Deliberately excludes
                                     track_gradeN: real samples per individual grade are
                                     too sparse and noisy to trust a combined average,
                                     which would erase the point of grading at all.
      Level 4  domain_default    -- documented (not fitted) South African road-design-
                                     speed assumption table. Last resort.

    Every row keeps a `speed_source` label recording which tier it came from.
    """
    df = gdf.copy()
    df["maxspeed_valid"] = df["maxspeed"].replace(0, np.nan)
    df["ref_prefix"] = df["ref"].astype(str).str.extract(r"^([A-Za-z]+)")[0]

    df["imputed_speed_kmh"] = df["maxspeed_valid"]
    df["speed_source"] = np.where(df["maxspeed_valid"].notna(), "observed", pd.NA)

    grp1 = df.groupby(["fclass", "ref_prefix"])["maxspeed_valid"].agg(["median", "count"])
    for (fclass, ref_prefix), row in grp1.iterrows():
        if row["count"] >= min_group_n:
            mask = ((df["fclass"] == fclass) & (df["ref_prefix"] == ref_prefix)
                     & df["imputed_speed_kmh"].isna())
            df.loc[mask, "imputed_speed_kmh"] = row["median"]
            df.loc[mask, "speed_source"] = "fclass_ref_median"

    grp2 = df.groupby("fclass")["maxspeed_valid"].agg(["median", "count"])
    for fclass, row in grp2.iterrows():
        if row["count"] >= min_fclass_n:
            mask = (df["fclass"] == fclass) & df["imputed_speed_kmh"].isna()
            df.loc[mask, "imputed_speed_kmh"] = row["median"]
            df.loc[mask, "speed_source"] = "fclass_median"

    parent_class = {
        "trunk_link": "trunk", "primary_link": "primary",
        "secondary_link": "secondary", "tertiary_link": "tertiary",
    }
    for child, parent in parent_class.items():
        mask = (df["fclass"] == child) & df["imputed_speed_kmh"].isna()
        if not mask.any():
            continue
        parent_real = df.loc[(df["fclass"] == parent) & df["maxspeed_valid"].notna(), "maxspeed_valid"]
        value = parent_real.median() if len(parent_real) >= min_parent_n else np.nan
        if pd.notna(value):
            df.loc[mask, "imputed_speed_kmh"] = value
            df.loc[mask, "speed_source"] = "parent_class_median"

    mask = df["imputed_speed_kmh"].isna()
    df.loc[mask, "imputed_speed_kmh"] = df.loc[mask, "fclass"].map(default_speed_kmh).fillna(40)
    df.loc[mask, "speed_source"] = "domain_default"

    df["imputed_speed_kmh"] = df["imputed_speed_kmh"].clip(lower=speed_floor_kmh)
    return df


# ---------------------------------------------------------------------------
# Cleaning + enrichment
# ---------------------------------------------------------------------------
def clean_and_enrich(
    gdf: gpd.GeoDataFrame,
    exclude_fclass: list[str],
    simplify_tolerance_deg: float,
    default_speed_kmh: dict[str, float],
    roughness_multiplier: dict[str, float],
) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    gdf = gdf[~gdf["fclass"].isin(exclude_fclass)].copy()
    gdf["geometry"] = gdf.geometry.simplify(tolerance=simplify_tolerance_deg, preserve_topology=True)
    gdf["length_km"] = gdf.geometry.apply(lambda g: GEOD.geometry_length(g) / 1000)

    gdf = impute_maxspeed(gdf, default_speed_kmh)

    gdf["travel_time_hr"] = gdf["length_km"] / gdf["imputed_speed_kmh"]
    gdf["roughness"] = gdf["fclass"].map(roughness_multiplier).fillna(1.3)

    speed_lookup = gdf.groupby("fclass")["imputed_speed_kmh"].median().to_dict()
    return gdf, speed_lookup


# ---------------------------------------------------------------------------
# Routable graph construction
# ---------------------------------------------------------------------------
def build_graph(gdf: gpd.GeoDataFrame, snap_tolerance_m: float) -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
    raw_nodes = set()
    raw_edges = []
    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        for i in range(len(coords) - 1):
            u = (round(coords[i][0], 6), round(coords[i][1], 6))
            v = (round(coords[i + 1][0], 6), round(coords[i + 1][1], 6))
            raw_nodes.add(u)
            raw_nodes.add(v)
            raw_edges.append((u, v, row["fclass"], row["imputed_speed_kmh"], row["roughness"]))

    node_list = list(raw_nodes)
    mean_lat = np.mean([n[1] for n in node_list])
    mx = 111_320 * np.cos(np.radians(mean_lat))
    my = 110_540
    pts_m = np.array([[n[0] * mx, n[1] * my] for n in node_list])

    tree = cKDTree(pts_m)
    pairs = tree.query_pairs(r=snap_tolerance_m)

    parent = list(range(len(node_list)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in pairs:
        union(i, j)

    node_to_cluster = {n: find(idx) for idx, n in enumerate(node_list)}
    cluster_coord: dict[int, tuple[float, float]] = {}
    for n in node_list:
        c = node_to_cluster[n]
        cluster_coord.setdefault(c, n)

    G = nx.Graph()
    for u, v, fclass, speed, roughness in raw_edges:
        cu, cv = node_to_cluster[u], node_to_cluster[v]
        if cu == cv or G.has_edge(cu, cv):
            continue
        seg_len_km = GEOD.line_length([u[0], v[0]], [u[1], v[1]]) / 1000
        seg_time_hr = seg_len_km / speed
        G.add_edge(cu, cv, length_km=seg_len_km, travel_time=seg_time_hr,
                   fclass=fclass, roughness=roughness)

    return G, cluster_coord


def largest_component_subgraph(G: nx.Graph) -> tuple[nx.Graph, set[int]]:
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy(), largest


def projection_scale(mean_lat: float) -> tuple[float, float]:
    mx = 111_320 * np.cos(np.radians(mean_lat))
    my = 110_540
    return mx, my


def nearest_node(lon: float, lat: float, cluster_coord: dict[int, tuple[float, float]],
                  candidate_clusters, mx: float, my: float) -> tuple[int, float]:
    candidates = [c for c in candidate_clusters if c in cluster_coord]
    coords = np.array([cluster_coord[c] for c in candidates])
    pts_m = np.column_stack([coords[:, 0] * mx, coords[:, 1] * my])
    p_m = np.array([lon * mx, lat * my])
    dists = np.linalg.norm(pts_m - p_m, axis=1)
    idx = dists.argmin()
    return candidates[idx], dists[idx] / 1000

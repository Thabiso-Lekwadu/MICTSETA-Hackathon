"""
nc_road_network.py
Shared processing library for the Northern Cape road network.

Both Data_Audit.ipynb and Visualization_Simulation.ipynb import this module
so the cleaning/graph-building logic lives in exactly one place and both
notebooks stay in sync. This is what makes the audit "reproducible" — rerun
build_clean_network() and every downstream artifact regenerates identically.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from pyproj import Geod
from scipy.spatial import cKDTree

GEOD = Geod(ellps="WGS84")

# ---------------------------------------------------------------------------
# Domain knowledge tables, documented assumptions, not fitted values.
# ---------------------------------------------------------------------------
DEFAULT_SPEED_KMH = {
    "trunk": 100, "trunk_link": 60,
    "primary": 100, "primary_link": 60,
    "secondary": 80, "secondary_link": 50,
    "tertiary": 60, "tertiary_link": 40,
    "unclassified": 50, "living_street": 20,
    "track_grade1": 40, "track_grade2": 30, "track_grade3": 25,
    "track_grade4": 20, "track_grade5": 15, "track": 30,
}

ROUGHNESS_MULTIPLIER = {
    "trunk": 1.00, "trunk_link": 1.00,
    "primary": 1.05, "primary_link": 1.05,
    "secondary": 1.10, "secondary_link": 1.10,
    "tertiary": 1.15, "tertiary_link": 1.15,
    "unclassified": 1.30, "living_street": 1.20,
    "track_grade1": 1.40, "track_grade2": 1.60, "track_grade3": 1.90,
    "track_grade4": 2.20, "track_grade5": 2.60, "track": 2.00,
}

# footway/steps/pedestrian/cycleway/bridleway/path aren't truck-drivable;
# residential/service are in-town local streets, irrelevant at province scale.
EXCLUDE_FCLASS = ["residential", "service", "footway", "path", "steps",
                   "pedestrian", "cycleway", "bridleway"]

SNAP_TOLERANCE_M = 150        # merge endpoints within this distance into one node
# Widened from the original 25m: at 25m the largest connected component covers
# only ~26% of the network (14,830 disconnected islands total), which forces
# routing onto huge, geographically nonsensical detours whenever the real
# connector between two points falls into a separate island. 150m closes most
# of that fragmentation. Going wider (300m+) closes more of it but starts
# producing IMPOSSIBLE routes shorter than the straight-line distance between
# their endpoints — i.e. it starts falsely merging nodes that aren't actually
# the same junction (parallel carriageways, nearby-but-unconnected roads).
# 150m was chosen because, checked against every town pair in TOWNS, it never
# produces a route shorter than straight-line distance; wider tolerances did.
# Use validate_route_plausibility() / flag_implausible_pairs() below to check
# any remaining gaps rather than pushing this value higher.
SIMPLIFY_TOLERANCE_DEG = 0.0005  # ~55m; cuts vertex count without changing topology

TOWNS = {
    "Kimberley": (24.7499, -28.7282), "Upington": (21.2561, -28.4478),
    "Springbok": (17.8865, -29.6644), "Kuruman": (23.4333, -27.4531),
    "De Aar": (24.0129, -30.6497), "Calvinia": (19.7761, -31.4707),
    "Port Nolloth": (16.8667, -29.2500),
}
# Approximate, the extract's bbox edge is close to the border, so this snaps
# ~9-10km from the true crossing. Good enough at province scale; a production
# version should fetch the real point via Overpass (barrier=border_control).
BORDER_POSTS = {
    "Vioolsdrift": (17.7500, -28.7500),
}
BORDER_DELAY_HR_DEFAULT = 5.0
IDLE_HEAT_RISK_FACTOR = 1.2


# ---------------------------------------------------------------------------
# STEP 1 — load
# ---------------------------------------------------------------------------
def load_raw(path="extracted_full_roads.parquet"):
    """
    Loads the raw road network. Point this at your own
    northern_cape_roads.gpkg (layer='edges') once you've re-run
    fast_nc_roads.py — the rest of the pipeline is identical either way.
    Also accepts .geojson / .parquet for the recovered dataset shipped with this kit.
    """
    if path.endswith(".gpkg"):
        return gpd.read_file(path, layer="edges", engine="pyogrio")
    if path.endswith(".parquet"):
        return gpd.read_parquet(path)
    return gpd.read_file(path)


# ---------------------------------------------------------------------------
# STEP 2 — clean + enrich the edge table
# ---------------------------------------------------------------------------
def impute_maxspeed(gdf, min_group_n=10, min_fclass_n=30, min_parent_n=5, speed_floor_kmh=5):
    """
    Tiered / hierarchical maxspeed imputation.

    OSM's `maxspeed == 0` means "unknown", not "0 km/h" — a vehicle cannot travel at
    0 km/h, so treating it literally breaks every downstream travel-time calculation
    (division by zero, or roads that look infinitely fast). This replaces every 0/missing
    value using the most specific real signal available, falling back progressively:

      Level 0  observed          : a real, non-zero maxspeed value. Trusted as-is.
      Level 1  fclass_ref_median : median of real values sharing the same (fclass, ref
                                    prefix), e.g. ("primary", "R") for provincial R-roads.
                                    Used only where >= min_group_n real samples exist.
      Level 2  fclass_median     : median of real values for the fclass alone, used only
                                    where >= min_fclass_n real samples exist.
      Level 3  parent_class      : borrows from a related class: *_link classes fall back
                                    to their base class (trunk_link -> trunk); track_gradeN
                                    classes fall back to the combined median across all
                                    track* grades. Used only where the parent itself has
                                    >= min_parent_n real samples.
      Level 4  domain_default    : a documented, not-fitted assumption table based on
                                    typical South African road design speeds. Last resort.

    Every row keeps a `speed_source` label recording which tier it came from, this is
    what makes the imputation auditable rather than a black box.
    """
    df = gdf.copy()
    df["maxspeed_valid"] = df["maxspeed"].replace(0, np.nan)
    df["ref_prefix"] = df["ref"].astype(str).str.extract(r"^([A-Za-z]+)")[0]

    df["imputed_speed_kmh"] = df["maxspeed_valid"]
    df["speed_source"] = np.where(df["maxspeed_valid"].notna(), "observed", pd.NA)

    # Level 1: (fclass, ref_prefix) group median
    grp1 = df.groupby(["fclass", "ref_prefix"])["maxspeed_valid"].agg(["median", "count"])
    for (fclass, ref_prefix), row in grp1.iterrows():
        if row["count"] >= min_group_n:
            mask = ((df["fclass"] == fclass) & (df["ref_prefix"] == ref_prefix)
                     & df["imputed_speed_kmh"].isna())
            df.loc[mask, "imputed_speed_kmh"] = row["median"]
            df.loc[mask, "speed_source"] = "fclass_ref_median"

    # Level 2: fclass-only median (computed from ALL real observations for that fclass,
    # not just what's left unfilled, so the sample size check is against the true support)
    grp2 = df.groupby("fclass")["maxspeed_valid"].agg(["median", "count"])
    for fclass, row in grp2.iterrows():
        if row["count"] >= min_fclass_n:
            mask = (df["fclass"] == fclass) & df["imputed_speed_kmh"].isna()
            df.loc[mask, "imputed_speed_kmh"] = row["median"]
            df.loc[mask, "speed_source"] = "fclass_median"

    # Level 3: parent-class borrowing. Deliberately excludes track_gradeN: real samples
    # per grade are too sparse (2-33 observations) and noisy to trust a combined "all
    # track grades" average, that would erase the whole point of grading (a maintained
    # grade1 track and a barely-passable grade5 track would wrongly get the same speed).
    # Ungraded *_link classes reasonably do share their parent's character, so those still
    # borrow; track_gradeN instead falls through to the grade-differentiated domain
    # defaults in Level 4.
    PARENT_CLASS = {
        "trunk_link": "trunk", "primary_link": "primary",
        "secondary_link": "secondary", "tertiary_link": "tertiary",
    }
    for child, parent in PARENT_CLASS.items():
        mask = (df["fclass"] == child) & df["imputed_speed_kmh"].isna()
        if not mask.any():
            continue
        parent_real = df.loc[(df["fclass"] == parent) & df["maxspeed_valid"].notna(),
                              "maxspeed_valid"]
        value = parent_real.median() if len(parent_real) >= min_parent_n else np.nan
        if pd.notna(value):
            df.loc[mask, "imputed_speed_kmh"] = value
            df.loc[mask, "speed_source"] = "parent_class_median"

    # Level 4: documented domain-default table (last resort)
    mask = df["imputed_speed_kmh"].isna()
    df.loc[mask, "imputed_speed_kmh"] = df.loc[mask, "fclass"].map(DEFAULT_SPEED_KMH).fillna(40)
    df.loc[mask, "speed_source"] = "domain_default"

    # A vehicle cannot travel at ~0 km/h even in principle, floor any residual extreme
    # low value (e.g. a genuine but implausible observed maxspeed=2 on one segment).
    df["imputed_speed_kmh"] = df["imputed_speed_kmh"].clip(lower=speed_floor_kmh)

    return df


def clean_and_enrich(gdf, exclude_fclass=EXCLUDE_FCLASS,
                      simplify_tol=SIMPLIFY_TOLERANCE_DEG):
    gdf = gdf[~gdf["fclass"].isin(exclude_fclass)].copy()
    gdf["geometry"] = gdf.geometry.simplify(tolerance=simplify_tol, preserve_topology=True)

    # accurate geodesic length, avoids UTM-zone distortion across the province
    gdf["length_km"] = gdf.geometry.apply(lambda g: GEOD.geometry_length(g) / 1000)

    # tiered maxspeed imputation (see impute_maxspeed docstring for the method)
    gdf = impute_maxspeed(gdf)

    gdf["travel_time_hr"] = gdf["length_km"] / gdf["imputed_speed_kmh"]
    gdf["roughness"] = gdf["fclass"].map(ROUGHNESS_MULTIPLIER).fillna(1.3)

    speed_lookup = gdf.groupby("fclass")["imputed_speed_kmh"].median().to_dict()
    return gdf, speed_lookup


# ---------------------------------------------------------------------------
# STEP 3 — build a routable graph with endpoint snapping
# ---------------------------------------------------------------------------
def build_graph(gdf, snap_tol_m=SNAP_TOLERANCE_M):
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
    mx = 111320 * np.cos(np.radians(mean_lat)) # convert the lon/lat degree coordinates to meters for snapping
    my = 110540
    pts_m = np.array([[n[0] * mx, n[1] * my] for n in node_list])

    tree = cKDTree(pts_m)
    pairs = tree.query_pairs(r=snap_tol_m)

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
    # representative coordinate per cluster (for plotting / nearest-node lookups)
    cluster_coord = {}
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


def largest_component_subgraph(G):
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy(), largest


# ---------------------------------------------------------------------------
# Nearest-node lookup, restricted to a routable node set
# ---------------------------------------------------------------------------
def nearest_node(lon, lat, cluster_coord, candidate_clusters, mx, my):
    candidates = [c for c in candidate_clusters if c in cluster_coord]
    coords = np.array([cluster_coord[c] for c in candidates])
    pts_m = np.column_stack([coords[:, 0] * mx, coords[:, 1] * my])
    p_m = np.array([lon * mx, lat * my])
    dists = np.linalg.norm(pts_m - p_m, axis=1)
    idx = dists.argmin()
    return candidates[idx], dists[idx] / 1000  # (cluster_id, distance_km)


def projection_scale(mean_lat):
    mx = 111320 * np.cos(np.radians(mean_lat))
    my = 110540
    return mx, my


# ---------------------------------------------------------------------------
# Routing: weight functions and Standard vs Fisheries-Optimized comparison
# ---------------------------------------------------------------------------
def make_time_weight(border_nodes):
    def weight_fn(u, v, d):
        return d["travel_time"] + border_nodes.get(v, 0.0)
    return weight_fn


def make_spoilage_weight(border_nodes):
    def weight_fn(u, v, d):
        spoilage = d["travel_time"] * d.get("roughness", 1.3)
        if v in border_nodes:
            spoilage += border_nodes[v] * IDLE_HEAT_RISK_FACTOR
        return spoilage
    return weight_fn


def evaluate_path(G, path, border_nodes, spoilage_threshold=20.0, shipment_value_rand=450_000):
    total_time, total_spoilage = 0.0, 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]
        total_time += d["travel_time"]
        total_spoilage += d["travel_time"] * d.get("roughness", 1.3)
        if v in border_nodes:
            delay = border_nodes[v]
            total_time += delay
            total_spoilage += delay * IDLE_HEAT_RISK_FACTOR
    spoilage_pct = min(1.0, total_spoilage / spoilage_threshold)
    return {
        "total_time_hr": round(total_time, 2),
        "spoilage_index": round(total_spoilage, 2),
        "spoilage_risk_pct": round(spoilage_pct * 100, 1),
        "expected_loss_rand": round(spoilage_pct * shipment_value_rand, -2),
    }


def compare_routes(G, origin_node, destination_node, border_nodes,
                    spoilage_threshold=20.0, shipment_value_rand=450_000):
    time_weight = make_time_weight(border_nodes)
    spoilage_weight = make_spoilage_weight(border_nodes)

    standard_path = nx.shortest_path(G, origin_node, destination_node, weight=time_weight)
    optimized_path = nx.shortest_path(G, origin_node, destination_node, weight=spoilage_weight)

    standard = evaluate_path(G, standard_path, border_nodes, spoilage_threshold, shipment_value_rand)
    optimized = evaluate_path(G, optimized_path, border_nodes, spoilage_threshold, shipment_value_rand)
    standard["path"] = standard_path
    optimized["path"] = optimized_path
    return standard, optimized


# ---------------------------------------------------------------------------
# Route plausibility validation — a route can never be shorter than the
# straight-line (great-circle) distance between its endpoints. A ratio close
# to 1.0 for a long haul is a red flag in the opposite direction: it usually
# means two genuinely different roads got falsely merged into one node by
# graph-building's snap tolerance. A high ratio (default threshold 1.6x)
# usually means the real, direct connector is missing/disconnected from the
# main routable component, forcing a detour like the Kimberley->De Aar case
# this was built to catch. Use this after any change to SNAP_TOLERANCE_M, and
# before picking town pairs for a live demo.
# ---------------------------------------------------------------------------
def straight_line_km(lon1, lat1, lon2, lat2):
    _, _, distance_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return distance_m / 1000.0


def validate_route_plausibility(G, cluster_coord, origin_node, destination_node,
                                 origin_lonlat=None, destination_lonlat=None, max_ratio=1.6):
    """Runs the time-shortest path between two already-resolved graph nodes
    and compares its distance against the straight-line distance between
    their real-world coordinates (origin_lonlat/destination_lonlat — falls
    back to the node's own snapped coordinate if not supplied, e.g. for a
    town whose true location differs slightly from its nearest graph node)."""
    path = nx.shortest_path(G, origin_node, destination_node, weight='travel_time')
    route_km = sum(G[u][v]['length_km'] for u, v in zip(path[:-1], path[1:]))
    route_hr = sum(G[u][v]['travel_time'] for u, v in zip(path[:-1], path[1:]))

    o_lon, o_lat = origin_lonlat if origin_lonlat else cluster_coord[origin_node]
    d_lon, d_lat = destination_lonlat if destination_lonlat else cluster_coord[destination_node]
    straight_km = straight_line_km(o_lon, o_lat, d_lon, d_lat)

    ratio = route_km / straight_km if straight_km > 0 else float('inf')
    return {
        "route_km": round(route_km, 1),
        "route_time_hr": round(route_hr, 2),
        "straight_line_km": round(straight_km, 1),
        "ratio": round(ratio, 2),
        # impossible: no real road can be shorter than the straight line.
        # Signals a false merge (snap tolerance too wide), not a real gap.
        "impossible_shortcut": ratio < 1.0,
        # plausible-but-suspicious detour, most likely a missing connector.
        "flagged_as_detour": ratio >= max_ratio,
    }


def flag_implausible_pairs(G, cluster_coord, towns=TOWNS, max_ratio=1.6):
    """Runs validate_route_plausibility() across every pair in `towns` (or any
    {name: (lon, lat)} dict) and returns one row per pair. Intended for a
    quick sanity-check cell in Data_Audit.ipynb — run this any time the graph
    is rebuilt, and treat any 'impossible_shortcut' row as urgent (the graph
    is falsely connecting two different roads) and any 'flagged_as_detour'
    row as a candidate to avoid for a live demo until investigated."""
    import itertools

    def nearest_node(lon, lat):
        best, best_dist = None, float('inf')
        for node in G.nodes():
            clon, clat = cluster_coord[node]
            dist = (clon - lon) ** 2 + (clat - lat) ** 2
            if dist < best_dist:
                best_dist = dist
                best = node
        return best

    node_lookup = {name: nearest_node(lon, lat) for name, (lon, lat) in towns.items()}
    rows = []
    for a, b in itertools.combinations(towns.keys(), 2):
        result = validate_route_plausibility(
            G, cluster_coord, node_lookup[a], node_lookup[b],
            origin_lonlat=towns[a], destination_lonlat=towns[b], max_ratio=max_ratio,
        )
        rows.append({"origin": a, "destination": b, **result})
    return rows


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------
def build_clean_network(raw_path="extracted_full_roads.geojson"):
    gdf_raw = load_raw(raw_path)
    gdf_clean, speed_lookup = clean_and_enrich(gdf_raw)
    G_full, cluster_coord = build_graph(gdf_clean)
    G_main, main_nodes = largest_component_subgraph(G_full)
    return {
        "gdf_raw": gdf_raw,
        "gdf_clean": gdf_clean,
        "speed_lookup": speed_lookup,
        "G_full": G_full,
        "G_main": G_main,
        "main_nodes": main_nodes,
        "cluster_coord": cluster_coord,
    }

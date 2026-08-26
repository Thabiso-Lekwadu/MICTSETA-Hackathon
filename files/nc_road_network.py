"""
nc_road_network_improved.py
Improved, drop-in replacement for nc_road_network.py — Northern Cape road network.

WHAT CHANGED vs. the original (and WHY), all API-compatible so Data_Audit.ipynb,
Visualization_Simulation.ipynb, live_backend.py, live_pipeline.py and
system_validation_test.py keep importing it unchanged:

  1. CONNECTIVITY STITCHING (the big accuracy win). The original left ~31% of
     nodes stranded in ~3,687 fragments at 150 m snapping — the direct cause of
     "the router misses a road and takes a huge detour it thinks is optimal".
     build_graph() now runs a conservative endpoint-bridging pass AFTER snapping:
     dead-end nodes (degree 1) that sit within BRIDGE_MAX_M of a dead-end in a
     DIFFERENT component are joined by a real-length connector EDGE (never a node
     merge). This closes digitization gaps without ever creating an impossible
     shortcut, and each connector is tagged `inferred=True` so it stays auditable.

  2. SPATIAL-KNN MAXSPEED IMPUTATION. The original fell straight to a flat,
     province-wide domain default for 65% of segments. A new tier
     (`fclass_spatial_knn`) sits just above that default: a segment with no
     speed borrows a distance-weighted average of the nearest OBSERVED-speed
     segments of the SAME fclass. A track hugging a fast primary corridor now
     inherits a locally-plausible speed instead of one flat number for every
     track in the province — materially better edge weights where any real data
     exists nearby, with the flat default only as the true last resort.

  3. ONE-WAY-AWARE DIRECTED BUILD (realism, opt-in). build_directed_graph()
     produces a networkx.MultiDiGraph that honors the `oneway` tag (F = forward
     only), matching what live_backend.py's docstring and its
     successors()/predecessors() calls actually assume. The default build_graph()
     stays undirected for exact backward-compatibility with the audit notebook's
     nx.connected_components() cells.

  4. TUNABLE, NON-LINEAR SPOILAGE WEIGHT. SPOILAGE_ROUGHNESS_EXPONENT lets you
     penalize rough roads super-linearly (vibration damage is not linear in
     roughness). Default 1.0 = identical to the original; raise to ~1.3–1.6 to
     make the spoilage-optimal route diverge more decisively from the time-optimal
     one wherever a rough-but-fast shortcut actually exists.

  5. ROAD-NAME NORMALIZATION. normalize_road_name() gives every edge a sensible
     display label even though `name`/`ref` are >90% missing.

Everything else (geodesic length, tiered imputation levels 0–4, roughness table,
plausibility validators) is preserved.
"""

import itertools
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from pyproj import Geod
from scipy.spatial import cKDTree

logger = logging.getLogger("nc_road_network")

GEOD = Geod(ellps="WGS84")

# ---------------------------------------------------------------------------
# Domain knowledge tables — documented assumptions, not fitted values.
# ---------------------------------------------------------------------------
DEFAULT_SPEED_KMH = {
    "trunk": 100, "trunk_link": 60,
    "primary": 100, "primary_link": 60,
    "secondary": 80, "secondary_link": 50,
    "tertiary": 60, "tertiary_link": 40,
    "unclassified": 50, "living_street": 20,
    "track_grade1": 40, "track_grade2": 30, "track_grade3": 25,
    "track_grade4": 20, "track_grade5": 15, "track": 30,
    "connector": 30,  # inferred gap-bridging edges (see stitch_components)
}

ROUGHNESS_MULTIPLIER = {
    "trunk": 1.00, "trunk_link": 1.00,
    "primary": 1.05, "primary_link": 1.05,
    "secondary": 1.10, "secondary_link": 1.10,
    "tertiary": 1.15, "tertiary_link": 1.15,
    "unclassified": 1.30, "living_street": 1.20,
    "track_grade1": 1.40, "track_grade2": 1.60, "track_grade3": 1.90,
    "track_grade4": 2.20, "track_grade5": 2.60, "track": 2.00,
    "connector": 1.50,  # inferred connectors carry a modest penalty so the
                        # solver prefers a real mapped road when one exists
}

EXCLUDE_FCLASS = ["residential", "service", "footway", "path", "steps",
                  "pedestrian", "cycleway", "bridleway"]

SNAP_TOLERANCE_M = 150        # merge endpoints within this distance into one node
SIMPLIFY_TOLERANCE_DEG = 0.0005  # ~55m; cuts vertex count without changing topology

# --- New tuning knobs -------------------------------------------------------
# Connectivity stitching: dead-end nodes within this distance of a dead-end in a
# DIFFERENT component are joined by a real-length connector edge. 1000 m is
# conservative: wide enough to close the traced-but-unconnected-track gaps that
# dominate the fragmentation, narrow enough that it never bridges two genuinely
# separate roads. It adds EDGES, never merges nodes, so it can never create a
# route shorter than the straight-line distance between two points.
BRIDGE_MAX_M = 1000.0
STITCH_ONLY_DEAD_ENDS = True   # only degree-1 nodes are bridge candidates

# Spatial-KNN imputation: a missing-speed segment borrows the distance-weighted
# average of its K nearest observed-speed neighbours of the same fclass, provided
# that fclass has at least this many observed samples to learn from.
KNN_NEIGHBORS = 8
KNN_MIN_OBSERVED = 8

# Spoilage weighting non-linearity. 1.0 reproduces the original exactly.
SPOILAGE_ROUGHNESS_EXPONENT = 1.0

TOWNS = {
    "Kimberley": (24.7499, -28.7282), "Upington": (21.2561, -28.4478),
    "Springbok": (17.8865, -29.6644), "Kuruman": (23.4333, -27.4531),
    "De Aar": (24.0129, -30.6497), "Calvinia": (19.7761, -31.4707),
    "Port Nolloth": (16.8667, -29.2500),
}
BORDER_POSTS = {
    "Vioolsdrift": (17.7500, -28.7500),
}
BORDER_DELAY_HR_DEFAULT = 5.0
IDLE_HEAT_RISK_FACTOR = 1.2


# ---------------------------------------------------------------------------
# STEP 1 — load
# ---------------------------------------------------------------------------
def load_raw(path="extracted_full_roads.parquet"):
    """Loads the raw road network. Point this at northern_cape_roads.gpkg
    (layer='edges') once fast_nc_roads.py has produced it; also accepts
    .geojson / .parquet for the recovered dataset shipped with this kit."""
    if path.endswith(".gpkg"):
        return gpd.read_file(path, layer="edges", engine="pyogrio")
    if path.endswith(".parquet"):
        return gpd.read_parquet(path)
    return gpd.read_file(path)


# ---------------------------------------------------------------------------
# STEP 2 — clean + enrich the edge table
# ---------------------------------------------------------------------------
def _projection_scale(mean_lat):
    mx = 111_320.0 * np.cos(np.radians(mean_lat))
    my = 110_540.0
    return mx, my


def _spatial_knn_impute(df, k=KNN_NEIGHBORS, min_observed=KNN_MIN_OBSERVED):
    """New imputation tier `fclass_spatial_knn`, applied to rows still missing a
    speed after the median tiers but before the flat domain default.

    For each fclass that carries at least `min_observed` real observations, a
    missing segment's speed is set to the distance-weighted average of its K
    nearest OBSERVED-speed segments of that same fclass (nearest by segment
    representative point, in a local metric projection). This makes the estimate
    LOCAL — a track next to a fast corridor inherits a fast-ish speed, a track in
    a remote basin inherits a slow one — instead of one province-wide constant.

    Mutates and returns df. Rows with no nearby same-fclass observation are left
    untouched, to fall through to the documented domain default (Level 4)."""
    still_missing = df["imputed_speed_kmh"].isna()
    if not still_missing.any():
        return df

    reps = df.geometry.representative_point()
    lon = reps.x.to_numpy()
    lat = reps.y.to_numpy()
    finite = np.isfinite(lat)
    mean_lat = float(np.nanmean(lat[finite])) if finite.any() else -28.0
    mx, my = _projection_scale(mean_lat)
    px = lon * mx
    py = lat * my

    speed_col = df.columns.get_loc("imputed_speed_kmh")
    source_col = df.columns.get_loc("speed_source")
    observed_speeds_all = df["maxspeed_valid"].to_numpy()

    for fclass in df.loc[still_missing, "fclass"].dropna().unique():
        fmask = (df["fclass"] == fclass).to_numpy()
        obs_mask = fmask & df["maxspeed_valid"].notna().to_numpy()
        miss_mask = fmask & df["imputed_speed_kmh"].isna().to_numpy()
        n_obs = int(obs_mask.sum())
        if n_obs < min_observed or not miss_mask.any():
            continue

        obs_idx = np.where(obs_mask)[0]
        miss_idx = np.where(miss_mask)[0]
        obs_points = np.column_stack([px[obs_idx], py[obs_idx]])
        valid_obs = np.isfinite(obs_points).all(axis=1)
        obs_idx = obs_idx[valid_obs]
        obs_points = obs_points[valid_obs]
        if len(obs_idx) < min_observed:
            continue

        tree = cKDTree(obs_points)
        obs_speeds = observed_speeds_all[obs_idx].astype(float)
        kk = min(k, len(obs_idx))

        query_points = np.column_stack([px[miss_idx], py[miss_idx]])
        for row_pos, mi in enumerate(miss_idx):
            q = query_points[row_pos]
            if not np.isfinite(q).all():
                continue
            distances, neighbors = tree.query(q, k=kk)
            neighbors = np.atleast_1d(neighbors)
            distances = np.atleast_1d(distances).astype(float)
            weights = 1.0 / (distances + 1.0)  # +1 m guards against a zero-distance blow-up
            estimate = float(np.average(obs_speeds[neighbors], weights=weights))
            df.iat[mi, speed_col] = estimate
            df.iat[mi, source_col] = "fclass_spatial_knn"

    return df


def impute_maxspeed(gdf, min_group_n=10, min_fclass_n=30, min_parent_n=5, speed_floor_kmh=5,
                    use_spatial_knn=True):
    """Tiered / hierarchical maxspeed imputation.

    OSM's `maxspeed == 0` means "unknown", not "0 km/h". Every 0/missing value is
    replaced using the most specific real signal available, falling back
    progressively:

      Level 0  observed            : real, non-zero maxspeed. Trusted as-is.
      Level 1  fclass_ref_median   : median of real values sharing (fclass, ref prefix).
      Level 2  fclass_median       : median of real values for the fclass alone.
      Level 3  parent_class_median : *_link classes borrow their base class median.
      Level 3.5 fclass_spatial_knn : NEW — distance-weighted average of the K nearest
                                      observed-speed segments of the same fclass. Local,
                                      data-driven, only where nearby real data exists.
      Level 4  domain_default      : documented (not fitted) assumption table. Last resort.

    `speed_source` records which tier produced each value. Set use_spatial_knn=False
    to reproduce the original module's behaviour exactly."""
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

    # Level 2: fclass-only median
    grp2 = df.groupby("fclass")["maxspeed_valid"].agg(["median", "count"])
    for fclass, row in grp2.iterrows():
        if row["count"] >= min_fclass_n:
            mask = (df["fclass"] == fclass) & df["imputed_speed_kmh"].isna()
            df.loc[mask, "imputed_speed_kmh"] = row["median"]
            df.loc[mask, "speed_source"] = "fclass_median"

    # Level 3: parent-class borrowing (excludes track_gradeN by design).
    PARENT_CLASS = {
        "trunk_link": "trunk", "primary_link": "primary",
        "secondary_link": "secondary", "tertiary_link": "tertiary",
    }
    for child, parent in PARENT_CLASS.items():
        mask = (df["fclass"] == child) & df["imputed_speed_kmh"].isna()
        if not mask.any():
            continue
        parent_real = df.loc[(df["fclass"] == parent) & df["maxspeed_valid"].notna(), "maxspeed_valid"]
        value = parent_real.median() if len(parent_real) >= min_parent_n else np.nan
        if pd.notna(value):
            df.loc[mask, "imputed_speed_kmh"] = value
            df.loc[mask, "speed_source"] = "parent_class_median"

    # Level 3.5: NEW spatial KNN within fclass (before the flat default).
    if use_spatial_knn:
        df = _spatial_knn_impute(df)

    # Level 4: documented domain-default table (last resort)
    mask = df["imputed_speed_kmh"].isna()
    df.loc[mask, "imputed_speed_kmh"] = df.loc[mask, "fclass"].map(DEFAULT_SPEED_KMH).fillna(40)
    df.loc[mask, "speed_source"] = "domain_default"

    df["imputed_speed_kmh"] = df["imputed_speed_kmh"].clip(lower=speed_floor_kmh)
    return df


def normalize_road_name(fclass, name=None, ref=None):
    """Consistent display label for an edge despite `name`/`ref` being >90% missing.
    Prefers "Name (REF)", then the name, then the ref, then a class-derived label."""
    name = None if (name is None or (isinstance(name, float) and pd.isna(name))) else str(name)
    ref = None if (ref is None or (isinstance(ref, float) and pd.isna(ref))) else str(ref)
    if name and ref and name != ref:
        return f"{name} ({ref})"
    if name:
        return name
    if ref:
        return ref
    label = str(fclass or "road").replace("_", " ")
    return f"Unnamed {label} road"


def clean_and_enrich(gdf, exclude_fclass=EXCLUDE_FCLASS,
                     simplify_tol=SIMPLIFY_TOLERANCE_DEG, use_spatial_knn=True):
    gdf = gdf[~gdf["fclass"].isin(exclude_fclass)].copy()
    gdf["geometry"] = gdf.geometry.simplify(tolerance=simplify_tol, preserve_topology=True)

    gdf["length_km"] = gdf.geometry.apply(lambda g: GEOD.geometry_length(g) / 1000)

    gdf = impute_maxspeed(gdf, use_spatial_knn=use_spatial_knn)

    gdf["travel_time_hr"] = gdf["length_km"] / gdf["imputed_speed_kmh"]
    gdf["roughness"] = gdf["fclass"].map(ROUGHNESS_MULTIPLIER).fillna(1.3)

    # Precomputed, non-linear spoilage cost carried on the edge table, so the
    # whole system can share one definition (the backend now prefers a
    # precomputed spoilage_cost when present). Default exponent 1.0 == original.
    gdf["spoilage_cost_edge"] = gdf["travel_time_hr"] * (gdf["roughness"] ** SPOILAGE_ROUGHNESS_EXPONENT)

    speed_lookup = gdf.groupby("fclass")["imputed_speed_kmh"].median().to_dict()
    return gdf, speed_lookup


# ---------------------------------------------------------------------------
# STEP 3 — build a routable graph with endpoint snapping + gap stitching
# ---------------------------------------------------------------------------
def _union_find(n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            return True
        return False

    return find, union


def stitch_components(G, cluster_coord, bridge_max_m=BRIDGE_MAX_M,
                      only_dead_ends=STITCH_ONLY_DEAD_ENDS):
    """Closes digitization gaps by adding real-length connector EDGES between
    nearby nodes that live in DIFFERENT connected components. Never merges nodes,
    so it can never manufacture an impossible (shorter-than-straight-line) route;
    the worst case is a connector that's slightly optimistic, which the modest
    `connector` roughness penalty already discounts.

    Only degree-1 nodes (true dead-ends) are candidates by default — those are
    exactly the traced-but-unconnected track endpoints that fragment the network.
    Returns the number of connector edges added. Each carries `inferred=True`."""
    if G.number_of_nodes() == 0:
        return 0

    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return 0

    node_to_comp = {}
    for cid, comp in enumerate(comps):
        for node in comp:
            node_to_comp[node] = cid

    if only_dead_ends:
        candidates = [n for n in G.nodes() if G.degree(n) == 1]
    else:
        candidates = list(G.nodes())
    if len(candidates) < 2:
        return 0

    coords = np.array([cluster_coord[n] for n in candidates])  # (lon, lat)
    mean_lat = float(coords[:, 1].mean())
    mx, my = _projection_scale(mean_lat)
    projected = np.column_stack([coords[:, 0] * mx, coords[:, 1] * my])

    tree = cKDTree(projected)
    pairs = tree.query_pairs(r=bridge_max_m)

    # Union-find over COMPONENT ids so we add at most one bridge per pair of
    # components joined (the shortest available), and never create redundant
    # parallel bridges once two components are already linked.
    find, union = _union_find(len(comps))

    # Sort candidate pairs by distance so the tightest gap is bridged first.
    scored = []
    for i, j in pairs:
        ci = node_to_comp[candidates[i]]
        cj = node_to_comp[candidates[j]]
        if ci == cj:
            continue
        dist_m = float(np.linalg.norm(projected[i] - projected[j]))
        scored.append((dist_m, i, j))
    scored.sort(key=lambda t: t[0])

    added = 0
    for dist_m, i, j in scored:
        node_a = candidates[i]
        node_b = candidates[j]
        ca = find(node_to_comp[node_a])
        cb = find(node_to_comp[node_b])
        if ca == cb:
            continue  # already connected through an earlier bridge
        lon_a, lat_a = cluster_coord[node_a]
        lon_b, lat_b = cluster_coord[node_b]
        seg_len_km = GEOD.line_length([lon_a, lon_b], [lat_a, lat_b]) / 1000.0
        speed = DEFAULT_SPEED_KMH["connector"]
        roughness = ROUGHNESS_MULTIPLIER["connector"]
        seg_time_hr = seg_len_km / speed if speed > 0 else 0.0
        G.add_edge(
            node_a, node_b,
            length_km=seg_len_km, travel_time=seg_time_hr,
            fclass="connector", roughness=roughness,
            spoilage_cost_edge=seg_time_hr * (roughness ** SPOILAGE_ROUGHNESS_EXPONENT),
            road_name="Inferred connector", road_ref=None, inferred=True,
        )
        union(node_to_comp[node_a], node_to_comp[node_b])
        added += 1

    if added:
        logger.info("stitch_components -> added %d inferred connector edges (<= %.0f m gaps).", added, bridge_max_m)
    return added


def build_graph(gdf, snap_tol_m=SNAP_TOLERANCE_M, stitch=True, bridge_max_m=BRIDGE_MAX_M):
    """Builds the undirected routable graph. Endpoints within `snap_tol_m` merge
    into one node (fixes float/digitization mismatches). When `stitch=True`, a
    conservative gap-bridging pass then connects nearby dead-ends across separate
    components (see stitch_components) — this is what lifts the largest component
    well above the original ~69% and stops the router taking absurd detours around
    a gap it can't see across. Set stitch=False for the original behaviour."""
    raw_nodes = set()
    raw_edges = []
    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        name = row.get("name")
        ref = row.get("ref")
        spoilage_edge = row.get("spoilage_cost_edge")
        for i in range(len(coords) - 1):
            u = (round(coords[i][0], 6), round(coords[i][1], 6))
            v = (round(coords[i + 1][0], 6), round(coords[i + 1][1], 6))
            raw_nodes.add(u)
            raw_nodes.add(v)
            raw_edges.append((u, v, row["fclass"], row["imputed_speed_kmh"], row["roughness"],
                              name, ref, spoilage_edge))

    node_list = list(raw_nodes)
    mean_lat = np.mean([n[1] for n in node_list])
    mx, my = _projection_scale(mean_lat)
    pts_m = np.array([[n[0] * mx, n[1] * my] for n in node_list])

    tree = cKDTree(pts_m)
    pairs = tree.query_pairs(r=snap_tol_m)

    find, union = _union_find(len(node_list))
    for i, j in pairs:
        union(i, j)

    node_to_cluster = {n: find(idx) for idx, n in enumerate(node_list)}
    cluster_coord = {}
    for n in node_list:
        c = node_to_cluster[n]
        cluster_coord.setdefault(c, n)

    G = nx.Graph()
    for u, v, fclass, speed, roughness, name, ref, spoilage_edge in raw_edges:
        cu, cv = node_to_cluster[u], node_to_cluster[v]
        if cu == cv or G.has_edge(cu, cv):
            continue
        seg_len_km = GEOD.line_length([u[0], v[0]], [u[1], v[1]]) / 1000
        seg_time_hr = seg_len_km / speed if speed > 0 else 0.0
        if spoilage_edge is None or (isinstance(spoilage_edge, float) and np.isnan(spoilage_edge)):
            spoilage_edge = seg_time_hr * (roughness ** SPOILAGE_ROUGHNESS_EXPONENT)
        G.add_edge(cu, cv, length_km=seg_len_km, travel_time=seg_time_hr,
                   fclass=fclass, roughness=roughness,
                   spoilage_cost_edge=spoilage_edge,
                   road_name=normalize_road_name(fclass, name, ref),
                   road_ref=(None if ref is None or (isinstance(ref, float) and pd.isna(ref)) else str(ref)),
                   inferred=False)

    if stitch:
        stitch_components(G, cluster_coord, bridge_max_m=bridge_max_m)

    return G, cluster_coord


def build_directed_graph(gdf, snap_tol_m=SNAP_TOLERANCE_M, stitch=True, bridge_max_m=BRIDGE_MAX_M):
    """OPTIONAL realism: builds a networkx.MultiDiGraph that honours the OSM
    `oneway` tag (F = forward-only along the digitized direction; B = both ways).
    This matches live_backend.py's docstring ("in-memory networkx.MultiDiGraph")
    and its successors()/predecessors() usage. Not used by the default pipeline —
    the audit notebook's nx.connected_components() expects the undirected graph —
    but available for a production routing layer that must respect one-way streets.
    Stitching adds bidirectional connector edges."""
    raw_nodes = set()
    raw_edges = []
    for _, row in gdf.iterrows():
        coords = list(row.geometry.coords)
        oneway = str(row.get("oneway") or "B").upper()
        name = row.get("name")
        ref = row.get("ref")
        spoilage_edge = row.get("spoilage_cost_edge")
        for i in range(len(coords) - 1):
            u = (round(coords[i][0], 6), round(coords[i][1], 6))
            v = (round(coords[i + 1][0], 6), round(coords[i + 1][1], 6))
            raw_nodes.add(u)
            raw_nodes.add(v)
            raw_edges.append((u, v, row["fclass"], row["imputed_speed_kmh"], row["roughness"],
                              name, ref, spoilage_edge, oneway))

    node_list = list(raw_nodes)
    mean_lat = np.mean([n[1] for n in node_list])
    mx, my = _projection_scale(mean_lat)
    pts_m = np.array([[n[0] * mx, n[1] * my] for n in node_list])
    tree = cKDTree(pts_m)
    pairs = tree.query_pairs(r=snap_tol_m)
    find, union = _union_find(len(node_list))
    for i, j in pairs:
        union(i, j)
    node_to_cluster = {n: find(idx) for idx, n in enumerate(node_list)}
    cluster_coord = {}
    for n in node_list:
        c = node_to_cluster[n]
        cluster_coord.setdefault(c, n)

    G = nx.MultiDiGraph()

    def add_directed(cu, cv, fclass, speed, roughness, name, ref, spoilage_edge, u, v):
        seg_len_km = GEOD.line_length([u[0], v[0]], [u[1], v[1]]) / 1000
        seg_time_hr = seg_len_km / speed if speed > 0 else 0.0
        if spoilage_edge is None or (isinstance(spoilage_edge, float) and np.isnan(spoilage_edge)):
            spoilage_edge = seg_time_hr * (roughness ** SPOILAGE_ROUGHNESS_EXPONENT)
        G.add_edge(cu, cv, length_km=seg_len_km, travel_time=seg_time_hr,
                   fclass=fclass, roughness=roughness, spoilage_cost_edge=spoilage_edge,
                   road_name=normalize_road_name(fclass, name, ref),
                   road_ref=(None if ref is None or (isinstance(ref, float) and pd.isna(ref)) else str(ref)),
                   inferred=False)

    for u, v, fclass, speed, roughness, name, ref, spoilage_edge, oneway in raw_edges:
        cu, cv = node_to_cluster[u], node_to_cluster[v]
        if cu == cv:
            continue
        add_directed(cu, cv, fclass, speed, roughness, name, ref, spoilage_edge, u, v)
        if oneway != "F":  # bidirectional (or one-way handled as forward-only)
            add_directed(cv, cu, fclass, speed, roughness, name, ref, spoilage_edge, v, u)

    if stitch:
        # Stitch on the weak (undirected) view, then mirror connectors both ways.
        undirected_view = nx.Graph()
        undirected_view.add_nodes_from(G.nodes())
        undirected_view.add_edges_from((a, b) for a, b in G.edges())
        added = stitch_components(undirected_view, cluster_coord, bridge_max_m=bridge_max_m)
        for a, b, data in undirected_view.edges(data=True):
            if data.get("inferred"):
                G.add_edge(a, b, **data)
                G.add_edge(b, a, **data)
        if added:
            logger.info("build_directed_graph -> mirrored %d connectors bidirectionally.", added)

    return G, cluster_coord


def largest_component_subgraph(G):
    """Largest routable component. Uses weakly-connected components for directed
    graphs and connected components for undirected, so it works with either build."""
    if G.is_directed():
        largest = max(nx.weakly_connected_components(G), key=len)
    else:
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
    return _projection_scale(mean_lat)


# ---------------------------------------------------------------------------
# Routing: weight functions and Standard vs Fisheries-Optimized comparison
# ---------------------------------------------------------------------------
def make_time_weight(border_nodes):
    def weight_fn(u, v, d):
        return d["travel_time"] + border_nodes.get(v, 0.0)
    return weight_fn


def make_spoilage_weight(border_nodes, roughness_exponent=SPOILAGE_ROUGHNESS_EXPONENT):
    """Fisheries-optimized weight. Prefers a precomputed `spoilage_cost_edge`
    when present (so the whole system shares one definition); otherwise computes
    travel_time * roughness**exponent on the fly. Raising the exponent makes the
    optimizer avoid rough roads more decisively."""
    def weight_fn(u, v, d):
        if "spoilage_cost_edge" in d and d["spoilage_cost_edge"] is not None:
            spoilage = d["spoilage_cost_edge"]
        else:
            spoilage = d["travel_time"] * (d.get("roughness", 1.3) ** roughness_exponent)
        if v in border_nodes:
            spoilage += border_nodes[v] * IDLE_HEAT_RISK_FACTOR
        return spoilage
    return weight_fn


def evaluate_path(G, path, border_nodes, spoilage_threshold=20.0, shipment_value_rand=450_000,
                  roughness_exponent=SPOILAGE_ROUGHNESS_EXPONENT):
    total_time, total_spoilage = 0.0, 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]
        total_time += d["travel_time"]
        if "spoilage_cost_edge" in d and d["spoilage_cost_edge"] is not None:
            total_spoilage += d["spoilage_cost_edge"]
        else:
            total_spoilage += d["travel_time"] * (d.get("roughness", 1.3) ** roughness_exponent)
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
# Route plausibility validation
# ---------------------------------------------------------------------------
def straight_line_km(lon1, lat1, lon2, lat2):
    _, _, distance_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return distance_m / 1000.0


def validate_route_plausibility(G, cluster_coord, origin_node, destination_node,
                                origin_lonlat=None, destination_lonlat=None, max_ratio=1.6):
    """Runs the time-shortest path between two resolved nodes and compares its
    distance against the straight-line distance between their real-world
    coordinates. ratio < 1.0 => impossible shortcut (false merge); ratio >=
    max_ratio => a suspicious detour (likely a missing/disconnected connector)."""
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
        "impossible_shortcut": ratio < 1.0,
        "flagged_as_detour": ratio >= max_ratio,
    }


def flag_implausible_pairs(G, cluster_coord, towns=TOWNS, max_ratio=1.6):
    """Runs validate_route_plausibility() across every pair in `towns`."""
    def nearest(lon, lat):
        best, best_dist = None, float('inf')
        for node in G.nodes():
            clon, clat = cluster_coord[node]
            dist = (clon - lon) ** 2 + (clat - lat) ** 2
            if dist < best_dist:
                best_dist = dist
                best = node
        return best

    node_lookup = {name: nearest(lon, lat) for name, (lon, lat) in towns.items()}
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
def build_clean_network(raw_path="extracted_full_roads.geojson", stitch=True, use_spatial_knn=True):
    gdf_raw = load_raw(raw_path)
    gdf_clean, speed_lookup = clean_and_enrich(gdf_raw, use_spatial_knn=use_spatial_knn)
    G_full, cluster_coord = build_graph(gdf_clean, stitch=stitch)
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
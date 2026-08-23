"""
system_validation_test.py

Automated validation harness for the Northern Cape Transport, Trade & Fisheries
routing system. Validates the compiled network graph (nc_road_graph.pkl),
cross-examines simulated telemetry against a live-hardware-style GPS session,
computes concrete performance metrics, and prints a Validation Audit Ledger.

Runs standalone:
    uv run system_validation_test.py

If nc_road_graph.pkl is not present in the working directory, the script logs
a warning and falls back to a small synthetic Northern Cape topology (same
schema: length_km, travel_time, fclass, roughness, base_time_mins,
spoilage_cost) so every test suite still executes meaningfully.

Exit code is 0 if every check PASSed or WARNed, 1 if any check FAILed —
suitable for wiring into a CI pipeline.
"""

from __future__ import annotations

import json
import logging
import pickle
import random
import sys
import textwrap
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from pyproj import Geod
from scipy.spatial import cKDTree

try:
    import nc_road_network as ncr
    HAVE_NCR = True
except ImportError:
    HAVE_NCR = False


# ---------------------------------------------------------------------------
# Constants — reused from nc_road_network.py when importable, with local
# fallbacks (identical values) so this script stays self-contained.
# ---------------------------------------------------------------------------
_FALLBACK_DEFAULT_SPEED_KMH = {
    "trunk": 100, "trunk_link": 60, "primary": 100, "primary_link": 60,
    "secondary": 80, "secondary_link": 50, "tertiary": 60, "tertiary_link": 40,
    "unclassified": 50, "living_street": 20, "track_grade1": 40, "track_grade2": 30,
    "track_grade3": 25, "track_grade4": 20, "track_grade5": 15, "track": 30,
}
_FALLBACK_ROUGHNESS_MULTIPLIER = {
    "trunk": 1.00, "trunk_link": 1.00, "primary": 1.05, "primary_link": 1.05,
    "secondary": 1.10, "secondary_link": 1.10, "tertiary": 1.15, "tertiary_link": 1.15,
    "unclassified": 1.30, "living_street": 1.20, "track_grade1": 1.40, "track_grade2": 1.60,
    "track_grade3": 1.90, "track_grade4": 2.20, "track_grade5": 2.60, "track": 2.00,
}
_FALLBACK_TOWNS = {
    "Kimberley": (24.7499, -28.7282), "Upington": (21.2561, -28.4478),
    "Springbok": (17.8865, -29.6644), "Kuruman": (23.4333, -27.4531),
    "De Aar": (24.0129, -30.6497), "Calvinia": (19.7761, -31.4707),
    "Port Nolloth": (16.8667, -29.2500),
}

DEFAULT_SPEED_KMH: dict[str, float] = ncr.DEFAULT_SPEED_KMH if HAVE_NCR else _FALLBACK_DEFAULT_SPEED_KMH
ROUGHNESS_MULTIPLIER: dict[str, float] = ncr.ROUGHNESS_MULTIPLIER if HAVE_NCR else _FALLBACK_ROUGHNESS_MULTIPLIER
SNAP_TOLERANCE_M: float = float(ncr.SNAP_TOLERANCE_M) if HAVE_NCR else 25.0
TOWNS: dict[str, tuple[float, float]] = ncr.TOWNS if HAVE_NCR else _FALLBACK_TOWNS
GEOD: Geod = ncr.GEOD if HAVE_NCR else Geod(ellps="WGS84")

GRAPH_PATH = Path("nc_road_graph.pkl")

# --- Test tolerances. Documented assumptions, not fitted values — adjust to
# your organization's acceptance criteria. ---
SPOILAGE_REDUCTION_TOLERANCE_PCT = 5.0     # optimizer must beat the time-only router by at least this much
SPATIAL_RMSE_TOLERANCE_M = 30.0            # acceptable GPS-to-route deviation
SNAP_ACCURACY_TOLERANCE_PCT = 80.0         # % of pings that must land within SNAP_TOLERANCE_M
TEMPORAL_CONVERGENCE_TOLERANCE_PCT = 20.0  # acceptable drift between predicted and real trip duration
CALIBRATION_DEVIATION_THRESHOLD_PCT = 10.0 # per-road-class speed drift that triggers a recalibration flag
INCIDENT_SEVERITY_FRACTION = 0.5           # incident penalty = this fraction of the baseline route's OWN
                                            # total cost, so the stress test stays meaningful at any network
                                            # size instead of a fixed absolute penalty (e.g. a flat +3h) that's
                                            # a rounding error on a long real-world route.
MIN_OD_SEPARATION_KM = 50.0                # cold-chain sampling only considers O/D pairs at least this far
                                            # apart, so results reflect long-haul trucking between towns/ports
                                            # rather than being diluted by geographically adjacent pairs where
                                            # the standard and spoilage-optimal routes are trivially identical

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("system_validation")


# ---------------------------------------------------------------------------
# Audit ledger
# ---------------------------------------------------------------------------
@dataclass
class LedgerEntry:
    suite: str
    metric: str
    score: str
    tolerance: str
    status: str  # PASS / WARN / FAIL
    note: str = ""


AUDIT_LEDGER: list[LedgerEntry] = []


def record(suite: str, metric: str, score: str, tolerance: str, status: str, note: str = "") -> LedgerEntry:
    entry = LedgerEntry(suite, metric, score, tolerance, status, note)
    AUDIT_LEDGER.append(entry)
    logger.info("[%s] %s -> %s (tolerance %s) : %s%s",
                suite, metric, score, tolerance, status, f" | {note}" if note else "")
    return entry


def print_ledger() -> None:
    suite_w, metric_w, score_w, tol_w, status_w = 26, 30, 15, 34, 7
    total_w = suite_w + metric_w + score_w + tol_w + status_w

    def cell(text: str, width: int) -> str:
        text = str(text)
        if len(text) > width - 1:
            text = text[: width - 2] + "\u2026"  # ellipsis, guarantees no column bleed
        return f"{text:<{width}}"

    print("\n" + "=" * total_w)
    print("VALIDATION AUDIT LEDGER".center(total_w))
    print("=" * total_w)
    print(cell("TEST SUITE", suite_w) + cell("METRIC", metric_w) + cell("SCORE", score_w)
          + cell("TOLERANCE", tol_w) + cell("STATUS", status_w))
    print("-" * total_w)

    for e in AUDIT_LEDGER:
        print(cell(e.suite, suite_w) + cell(e.metric, metric_w) + cell(e.score, score_w)
              + cell(e.tolerance, tol_w) + cell(e.status, status_w))
        if e.note:
            for line in textwrap.wrap(e.note, width=total_w - 6):
                print(f"      -> {line}")

    print("=" * total_w)
    passed = sum(1 for e in AUDIT_LEDGER if e.status == "PASS")
    warned = sum(1 for e in AUDIT_LEDGER if e.status == "WARN")
    failed = sum(1 for e in AUDIT_LEDGER if e.status == "FAIL")
    print(f"SUMMARY: {passed} PASS | {warned} WARN | {failed} FAIL  (total checks: {len(AUDIT_LEDGER)})")
    print("=" * total_w + "\n")


# ---------------------------------------------------------------------------
# 1. System state loading
# ---------------------------------------------------------------------------
def _ensure_routing_weights(G: nx.Graph) -> None:
    """Derives base_time_mins / spoilage_cost on every edge if not already
    present, mirroring live_backend.py's initialize_baseline_impedance()."""
    for _u, _v, attrs in G.edges(data=True):
        travel_time_hr = attrs.get("travel_time")
        if travel_time_hr is None:
            length_km = attrs.get("length_km", 1.0)
            speed = attrs.get("imputed_speed_kmh") or DEFAULT_SPEED_KMH.get(attrs.get("fclass", "unclassified"), 40)
            travel_time_hr = length_km / speed if speed > 0 else 0.0
            attrs["travel_time"] = travel_time_hr
        roughness = attrs.get("roughness", ROUGHNESS_MULTIPLIER.get(attrs.get("fclass", "unclassified"), 1.3))
        attrs.setdefault("roughness", roughness)
        attrs.setdefault("base_time_mins", travel_time_hr * 60.0)
        attrs.setdefault("spoilage_cost", travel_time_hr * roughness)


def _build_synthetic_topology() -> tuple[nx.Graph, set, dict[int, tuple[float, float]]]:
    """Small representative Northern Cape corridor graph, used only when
    nc_road_graph.pkl isn't available, so the validation suite is still
    fully exercisable standalone."""
    town_names = list(TOWNS.keys())
    node_ids = {name: idx for idx, name in enumerate(town_names)}
    cluster_coord = {idx: TOWNS[name] for name, idx in node_ids.items()}

    edges = [
        ("Port Nolloth", "Springbok", 95, "primary"),
        ("Springbok", "Upington", 400, "primary"),
        ("Springbok", "Calvinia", 320, "secondary"),
        ("Calvinia", "De Aar", 380, "track_grade1"),
        ("De Aar", "Kimberley", 210, "primary"),
        ("Kimberley", "Upington", 150, "track_grade3"),  # rough shortcut: shorter but much rougher
        ("Upington", "Kuruman", 260, "secondary"),
        ("Kuruman", "Kimberley", 260, "secondary"),      # paved detour: longer but smooth
    ]

    G = nx.Graph()
    G.add_nodes_from(cluster_coord.keys())
    for a, b, length_km, fclass in edges:
        if a not in node_ids or b not in node_ids:
            continue
        speed = DEFAULT_SPEED_KMH.get(fclass, 60)
        roughness = ROUGHNESS_MULTIPLIER.get(fclass, 1.3)
        travel_time_hr = length_km / speed
        G.add_edge(
            node_ids[a], node_ids[b],
            length_km=float(length_km), travel_time=travel_time_hr, fclass=fclass, roughness=roughness,
            base_time_mins=travel_time_hr * 60.0, spoilage_cost=travel_time_hr * roughness,
            road_name=None, road_ref=None,
        )

    main_nodes = set(G.nodes())
    return G, main_nodes, cluster_coord


def load_topology(graph_path: Path = GRAPH_PATH) -> tuple[nx.Graph, set, dict[int, tuple[float, float]], bool]:
    """Returns (G_main, main_nodes, cluster_coord, is_synthetic_fallback)."""
    if graph_path.exists():
        try:
            with graph_path.open("rb") as f:
                artifacts = pickle.load(f)
            G_main = artifacts["G_main"]
            main_nodes = artifacts.get("main_nodes", set(G_main.nodes()))
            cluster_coord = artifacts["cluster_coord"]
            _ensure_routing_weights(G_main)
            logger.info("Loaded real network topology from %s (%d nodes, %d edges)",
                        graph_path, G_main.number_of_nodes(), G_main.number_of_edges())
            return G_main, main_nodes, cluster_coord, False
        except Exception as exc:
            logger.error("Failed to load %s (%s); falling back to synthetic topology.", graph_path, exc)
    else:
        logger.warning("%s not found; falling back to a synthetic Northern Cape topology for validation.",
                        graph_path)

    G, main_nodes, cluster_coord = _build_synthetic_topology()
    return G, main_nodes, cluster_coord, True


def _nearest_named_node(cluster_coord: dict[int, tuple[float, float]], town_name: str) -> int:
    if town_name not in TOWNS:
        raise KeyError(f"Unknown town '{town_name}'. Known towns: {sorted(TOWNS.keys())}")
    target_lon, target_lat = TOWNS[town_name]
    mean_lat = float(np.mean([c[1] for c in cluster_coord.values()]))
    mx = 111_320.0 * np.cos(np.radians(mean_lat))
    my = 110_540.0
    tx, ty = target_lon * mx, target_lat * my

    node_ids = list(cluster_coord.keys())
    coords = np.array([cluster_coord[n] for n in node_ids])
    projected = np.column_stack([coords[:, 0] * mx, coords[:, 1] * my])
    dists = np.sum((projected - np.array([tx, ty])) ** 2, axis=1)
    return node_ids[int(np.argmin(dists))]


def _sum_attr(G: nx.Graph, path: list[int], attr: str) -> float:
    return sum(G[u][v].get(attr, 0.0) for u, v in zip(path[:-1], path[1:]))


# ---------------------------------------------------------------------------
# 2a. Resilience test — betweenness/path-divergence under a simulated incident
# ---------------------------------------------------------------------------
def resilience_test(G: nx.Graph, cluster_coord: dict[int, tuple[float, float]],
                     origin_name: str = "Port Nolloth", destination_name: str = "Upington",
                     incident_severity_fraction: float = INCIDENT_SEVERITY_FRACTION) -> dict:
    origin_node = _nearest_named_node(cluster_coord, origin_name)
    destination_node = _nearest_named_node(cluster_coord, destination_name)

    baseline_path = nx.shortest_path(G, origin_node, destination_node, weight="spoilage_cost")
    if len(baseline_path) < 3:
        raise ValueError("Baseline route too short to simulate a mid-route incident.")

    idx = len(baseline_path) // 2
    u, v = baseline_path[idx - 1], baseline_path[idx]
    original_attrs = dict(G[u][v])

    # Scale the incident penalty to the route's OWN baseline cost rather
    # than a fixed absolute value. A flat "+3 hours" is a huge relative
    # shock on a small demo graph but a rounding error on a 284-hop real
    # route with a large accumulated baseline cost — which would make the
    # test read as "100% overlap, no reroute" regardless of whether the
    # network actually has redundancy, rather than reflecting the network.
    baseline_spoilage_cost = _sum_attr(G, baseline_path, "spoilage_cost")
    baseline_time_mins = _sum_attr(G, baseline_path, "base_time_mins")
    incident_penalty_cost = baseline_spoilage_cost * incident_severity_fraction
    incident_penalty_mins = baseline_time_mins * incident_severity_fraction

    G[u][v]["spoilage_cost"] = original_attrs.get("spoilage_cost", 0.0) + incident_penalty_cost
    G[u][v]["base_time_mins"] = original_attrs.get("base_time_mins", 0.0) + incident_penalty_mins
    try:
        perturbed_path = nx.shortest_path(G, origin_node, destination_node, weight="spoilage_cost")
    finally:
        G[u][v].clear()
        G[u][v].update(original_attrs)

    baseline_set = set(baseline_path)
    perturbed_set = set(perturbed_path)
    overlap_pct = len(baseline_set & perturbed_set) / len(baseline_set) * 100.0

    incident_still_used = any(
        {a, b} == {u, v} for a, b in zip(perturbed_path[:-1], perturbed_path[1:])
    )

    return {
        "origin": origin_name, "destination": destination_name,
        "baseline_hops": len(baseline_path) - 1, "perturbed_hops": len(perturbed_path) - 1,
        "overlap_pct": overlap_pct, "incident_u": u, "incident_v": v,
        "incident_severity_fraction": incident_severity_fraction,
        "incident_penalty_hours": incident_penalty_mins / 60.0,
        "rerouted": not incident_still_used,
    }


# ---------------------------------------------------------------------------
# 2b. Cold-chain spoilage cost delta across randomized origin/destination pairs
# ---------------------------------------------------------------------------
def spoilage_delta_test(
    G: nx.Graph, main_nodes: set, cluster_coord: dict[int, tuple[float, float]],
    num_pairs: int = 20, seed: int = 42, min_separation_km: float = MIN_OD_SEPARATION_KM,
) -> dict:
    rng = random.Random(seed)
    nodes = list(main_nodes) if main_nodes else list(G.nodes())
    if len(nodes) < 2:
        raise ValueError("Not enough nodes in the topology for origin/destination sampling.")

    pct_reductions: list[float] = []
    time_penalties: list[float] = []
    separations_km: list[float] = []
    sampled = 0
    attempts = 0
    # Higher cap than before: filtering on separation as well as path
    # existence means more candidate pairs get rejected per successful
    # sample, especially on a large real network dominated by local roads.
    max_attempts = num_pairs * 200

    while sampled < num_pairs and attempts < max_attempts:
        attempts += 1
        o, d = rng.sample(nodes, 2)

        # Reject pairs that are too close together. Uniform random sampling
        # over a large node set mostly picks geographically adjacent pairs
        # on the same corridor, where the standard and spoilage-optimal
        # routes are trivially identical (0% reduction) — that dilutes the
        # mean toward zero and doesn't reflect the actual use case of
        # long-haul cold-chain trucking between towns/ports.
        lon_o, lat_o = cluster_coord[o]
        lon_d, lat_d = cluster_coord[d]
        _, _, dist_m = GEOD.inv(lon_o, lat_o, lon_d, lat_d)
        separation_km = dist_m / 1000.0
        if separation_km < min_separation_km:
            continue

        try:
            standard_path = nx.shortest_path(G, o, d, weight="base_time_mins")
            optimized_path = nx.shortest_path(G, o, d, weight="spoilage_cost")
        except nx.NetworkXNoPath:
            continue

        standard_spoilage = _sum_attr(G, standard_path, "spoilage_cost")
        optimized_spoilage = _sum_attr(G, optimized_path, "spoilage_cost")
        standard_time = _sum_attr(G, standard_path, "base_time_mins")
        optimized_time = _sum_attr(G, optimized_path, "base_time_mins")

        if standard_spoilage <= 0:
            continue

        pct_reductions.append((standard_spoilage - optimized_spoilage) / standard_spoilage * 100.0)
        time_penalties.append(optimized_time - standard_time)
        separations_km.append(separation_km)
        sampled += 1

    if sampled == 0:
        raise ValueError(
            f"Could not sample any O/D pairs at least {min_separation_km:.0f} km apart "
            "with a connecting path."
        )

    return {
        "sampled_pairs": sampled,
        "mean_spoilage_reduction_pct": float(np.mean(pct_reductions)),
        "mean_time_penalty_min": float(np.mean(time_penalties)),
        "max_spoilage_reduction_pct": float(np.max(pct_reductions)),
        "min_spoilage_reduction_pct": float(np.min(pct_reductions)),
        "mean_separation_km": float(np.mean(separations_km)),
        "min_separation_km_used": min_separation_km,
    }


# ---------------------------------------------------------------------------
# 3. Live hardware vs. simulation cross-validation
# ---------------------------------------------------------------------------
@dataclass
class TelemetryPing:
    lat: float
    lon: float
    timestamp: float  # unix seconds
    speed_kmh: Optional[float] = None


def _point_to_segment_distance_m(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ap = p - a
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq == 0.0:
        return float(np.linalg.norm(ap))
    t = max(0.0, min(1.0, float(np.dot(ap, ab)) / ab_len_sq))
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))


class EdgeIndex:
    """cKDTree-backed nearest-road-segment lookup with true point-to-segment
    distance (not just nearest midpoint), mirroring the EdgeSpatialIndex
    pattern used in live_backend.py."""

    def __init__(self, G: nx.Graph, cluster_coord: dict[int, tuple[float, float]]):
        self.G = G
        self.cluster_coord = cluster_coord
        self.edges: list[tuple[int, int]] = list(G.edges())
        if not self.edges:
            raise ValueError("Graph has no edges to index.")

        lats = [cluster_coord[n][1] for n in G.nodes()]
        mean_lat = float(np.mean(lats))
        self.mx = 111_320.0 * np.cos(np.radians(mean_lat))
        self.my = 110_540.0

        midpoints = []
        for u, v in self.edges:
            lon_u, lat_u = cluster_coord[u]
            lon_v, lat_v = cluster_coord[v]
            midpoints.append(((lon_u + lon_v) / 2.0 * self.mx, (lat_u + lat_v) / 2.0 * self.my))
        self.tree = cKDTree(np.array(midpoints))

    def _to_m(self, lon: float, lat: float) -> np.ndarray:
        return np.array([lon * self.mx, lat * self.my])

    def nearest_edge(self, lat: float, lon: float, k: int = 8) -> tuple[int, int, float, str]:
        """Returns (u, v, distance_m, fclass) for the closest road segment,
        refined via point-to-segment projection among the k nearest
        midpoint candidates."""
        query = self._to_m(lon, lat)
        k = min(k, len(self.edges))
        _, idxs = self.tree.query(query, k=k)
        idxs = np.atleast_1d(idxs)

        best: Optional[tuple[int, int, float, str]] = None
        for idx in idxs:
            u, v = self.edges[int(idx)]
            lon_u, lat_u = self.cluster_coord[u]
            lon_v, lat_v = self.cluster_coord[v]
            p1 = self._to_m(lon_u, lat_u)
            p2 = self._to_m(lon_v, lat_v)
            dist_m = _point_to_segment_distance_m(query, p1, p2)
            if best is None or dist_m < best[2]:
                fclass = self.G[u][v].get("fclass", "unclassified")
                best = (u, v, dist_m, fclass)
        assert best is not None
        return best


def generate_synthetic_traccar_session(
    G: nx.Graph, cluster_coord: dict[int, tuple[float, float]],
    origin_name: str, destination_name: str,
    num_pings: int = 15, gps_noise_std_m: float = 12.0, speed_variance: float = 0.85, seed: int = 7,
) -> list[TelemetryPing]:
    """Builds a plausible 'real-world' GPS ping sequence by walking the
    optimized route and perturbing it with GPS noise plus a systematically
    slower real-world speed. Used only as a stand-in when no real Traccar
    export is supplied — swap in your own list[TelemetryPing] built from a
    real session to validate against actual hardware."""
    rng = random.Random(seed)
    origin_node = _nearest_named_node(cluster_coord, origin_name)
    destination_node = _nearest_named_node(cluster_coord, destination_name)
    path = nx.shortest_path(G, origin_node, destination_node, weight="spoilage_cost")
    coords = [cluster_coord[n] for n in path]  # (lon, lat)
    if len(coords) < 2:
        raise ValueError("Route too short to synthesize a telemetry session.")

    # Precompute per-hop distance, real speed, and *integrated* elapsed time.
    # Each hop crosses one road segment with its own fclass/speed, so elapsed
    # time has to accumulate hop-by-hop at each hop's own real speed — not
    # get back-calculated by dividing the *cumulative* distance-so-far by
    # whichever single segment a ping happens to land on. That back-calc was
    # the bug: a slow segment late in a long, road-class-mixed route would
    # retroactively apply its speed to the entire preceding distance,
    # producing a runaway inflated "real" trip time.
    cumulative_km = [0.0]
    cumulative_hr = [0.0]
    seg_real_speeds: list[float] = []
    for i, ((lon1, lat1), (lon2, lat2)) in enumerate(zip(coords[:-1], coords[1:])):
        _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
        seg_km = dist_m / 1000.0
        cumulative_km.append(cumulative_km[-1] + seg_km)

        u, v = path[i], path[i + 1]
        fclass = G[u][v].get("fclass", "unclassified")
        assumed_speed = DEFAULT_SPEED_KMH.get(fclass, 40)
        real_speed = assumed_speed * speed_variance
        seg_real_speeds.append(real_speed)
        seg_hr = seg_km / real_speed if real_speed > 0 else 0.0
        cumulative_hr.append(cumulative_hr[-1] + seg_hr)
    total_km = cumulative_km[-1]

    pings: list[TelemetryPing] = []
    start_time = time.time()
    for i in range(num_pings):
        frac = i / (num_pings - 1) if num_pings > 1 else 0.0
        target_km = frac * total_km

        idx = bisect_right(cumulative_km, target_km) - 1
        idx = max(0, min(idx, len(cumulative_km) - 2))
        seg_start_km, seg_end_km = cumulative_km[idx], cumulative_km[idx + 1]
        seg_frac = (target_km - seg_start_km) / (seg_end_km - seg_start_km) if seg_end_km > seg_start_km else 0.0

        lon0, lat0 = coords[idx]
        lon1, lat1 = coords[idx + 1]
        lat = lat0 + (lat1 - lat0) * seg_frac
        lon = lon0 + (lon1 - lon0) * seg_frac

        noise_bearing = rng.uniform(0, 360)
        noise_dist = abs(rng.gauss(0, gps_noise_std_m))
        noisy_lon, noisy_lat, _ = GEOD.fwd(lon, lat, noise_bearing, noise_dist)

        real_speed = seg_real_speeds[idx]
        # Elapsed time = all prior hops' own integrated time, plus a linear
        # fraction of *this* hop's time (constant speed within a hop, so
        # fraction-of-distance == fraction-of-time here).
        elapsed_hr = cumulative_hr[idx] + seg_frac * (cumulative_hr[idx + 1] - cumulative_hr[idx])
        timestamp = start_time + elapsed_hr * 3600.0

        pings.append(TelemetryPing(lat=noisy_lat, lon=noisy_lon, timestamp=timestamp, speed_kmh=real_speed))

    return pings


def cross_validate_telemetry(
    G: nx.Graph, cluster_coord: dict[int, tuple[float, float]], pings: list[TelemetryPing],
) -> dict:
    """Compares a raw telemetry session (real Traccar pings or the synthetic
    stand-in) against the internal routing model: spatial deviation (MSE/RMSE),
    snapping accuracy, and temporal convergence."""
    if len(pings) < 2:
        raise ValueError("Need at least 2 telemetry pings to cross-validate.")

    edge_index = EdgeIndex(G, cluster_coord)

    spatial_errors_m: list[float] = []
    snapped_flags: list[bool] = []
    matched_speeds: list[float] = []
    matched_fclasses: list[str] = []

    for ping in pings:
        try:
            u, v, dist_m, fclass = edge_index.nearest_edge(ping.lat, ping.lon)
        except Exception as exc:
            logger.error("Snapping failed for ping (%.5f, %.5f): %s", ping.lat, ping.lon, exc)
            continue
        spatial_errors_m.append(dist_m)
        snapped_flags.append(dist_m <= SNAP_TOLERANCE_M)
        matched_speeds.append(DEFAULT_SPEED_KMH.get(fclass, 40.0))
        matched_fclasses.append(fclass)

    if not spatial_errors_m:
        raise ValueError("No telemetry pings could be snapped to the topology.")

    mse = float(np.mean(np.square(spatial_errors_m)))
    rmse = float(np.sqrt(mse))
    snap_accuracy_pct = float(np.mean(snapped_flags)) * 100.0

    sorted_pings = sorted(pings, key=lambda p: p.timestamp)
    total_real_minutes = (sorted_pings[-1].timestamp - sorted_pings[0].timestamp) / 60.0
    total_real_km = 0.0
    for p1, p2 in zip(sorted_pings[:-1], sorted_pings[1:]):
        _, _, dist_m = GEOD.inv(p1.lon, p1.lat, p2.lon, p2.lat)
        total_real_km += dist_m / 1000.0

    avg_matched_speed = float(np.mean(matched_speeds)) if matched_speeds else 40.0
    predicted_minutes = (total_real_km / avg_matched_speed) * 60.0 if avg_matched_speed > 0 else 0.0
    temporal_convergence_pct_diff = (
        (total_real_minutes - predicted_minutes) / predicted_minutes * 100.0 if predicted_minutes > 0 else 0.0
    )

    real_speeds = [p.speed_kmh for p in pings if p.speed_kmh is not None]

    return {
        "num_pings": len(pings),
        "num_snapped_pings": len(spatial_errors_m),
        "spatial_mse_m2": mse,
        "spatial_rmse_m": rmse,
        "snap_accuracy_pct": snap_accuracy_pct,
        "total_real_minutes": total_real_minutes,
        "predicted_minutes": predicted_minutes,
        "temporal_convergence_pct_diff": temporal_convergence_pct_diff,
        "avg_matched_speed_kmh": avg_matched_speed,
        "avg_real_speed_kmh": float(np.mean(real_speeds)) if real_speeds else None,
        "matched_fclasses": matched_fclasses,
    }


# ---------------------------------------------------------------------------
# 4. Self-calibration layer
# ---------------------------------------------------------------------------
def calibrate_speed_model(
    pings: list[TelemetryPing], matched_fclasses: list[str],
    deviation_threshold_pct: float = CALIBRATION_DEVIATION_THRESHOLD_PCT,
) -> dict:
    """Running-average error filter: for each road class the telemetry
    touched, compares the vehicle's actual recorded speed against
    DEFAULT_SPEED_KMH's assumption. Classes whose real speed deviates by more
    than `deviation_threshold_pct` get a calibrated replacement value."""
    class_speeds: dict[str, list[float]] = {}
    for ping, fclass in zip(pings, matched_fclasses):
        if ping.speed_kmh is None:
            continue
        class_speeds.setdefault(fclass, []).append(ping.speed_kmh)

    calibrated_vector: dict[str, float] = {}
    notes: list[str] = []
    alpha = 0.3  # exponential running-average smoothing factor

    for fclass, speeds in class_speeds.items():
        assumed = DEFAULT_SPEED_KMH.get(fclass)
        if assumed is None or not speeds:
            continue

        running_avg = speeds[0]
        for s in speeds[1:]:
            running_avg = alpha * s + (1 - alpha) * running_avg

        deviation_pct = (running_avg - assumed) / assumed * 100.0
        if abs(deviation_pct) >= deviation_threshold_pct:
            calibrated_vector[fclass] = round(running_avg, 1)
            direction = "slower" if deviation_pct < 0 else "faster"
            notes.append(
                f"'{fclass}': assumed {assumed:.0f} km/h, telemetry averages {running_avg:.1f} km/h "
                f"({abs(deviation_pct):.1f}% {direction}) -> recalibrate"
            )

    return {
        "calibrated_speed_vector": calibrated_vector,
        "notes": notes,
        "deviation_threshold_pct": deviation_threshold_pct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 90)
    logger.info("NORTHERN CAPE TRANSPORT / FISHERIES ROUTING -- SYSTEM VALIDATION SUITE")
    logger.info("=" * 90)

    G, main_nodes, cluster_coord, is_synthetic = load_topology()
    if is_synthetic:
        logger.warning(
            "Running against a SYNTHETIC FALLBACK topology (%d nodes) -- nc_road_graph.pkl "
            "was not found or failed to load. Results validate the pipeline's LOGIC, not "
            "the real network's data quality.",
            G.number_of_nodes(),
        )

    # --- 2a. Resilience / incident response --------------------------------
    try:
        resilience = resilience_test(G, cluster_coord)
        status = "PASS" if resilience["rerouted"] else "WARN"
        record(
            "Resilience / Incident", "Node Overlap After Incident",
            f"{resilience['overlap_pct']:.1f}%", "Reroute expected (overlap < 100%)", status,
            f"{resilience['origin']} -> {resilience['destination']}: baseline "
            f"{resilience['baseline_hops']} hops, perturbed {resilience['perturbed_hops']} hops. "
            f"Incident on segment ({resilience['incident_u']} -> {resilience['incident_v']}), "
            f"penalty +{resilience['incident_penalty_hours']:.1f}h "
            f"({resilience['incident_severity_fraction']:.0%} of baseline route cost).",
        )
    except Exception as exc:
        record("Resilience / Incident", "Node Overlap After Incident", "ERROR", "n/a", "FAIL", str(exc))

    # --- 2b. Cold-chain spoilage cost delta ---------------------------------
    try:
        spoilage = spoilage_delta_test(G, main_nodes, cluster_coord)
        status = "PASS" if spoilage["mean_spoilage_reduction_pct"] >= SPOILAGE_REDUCTION_TOLERANCE_PCT else "FAIL"
        record(
            "Cold-Chain Optimizer", "Mean Spoilage Cost Reduction",
            f"{spoilage['mean_spoilage_reduction_pct']:.1f}%",
            f">= {SPOILAGE_REDUCTION_TOLERANCE_PCT:.0f}%", status,
            f"Sampled {spoilage['sampled_pairs']} O/D pairs (mean separation "
            f"{spoilage['mean_separation_km']:.0f} km, min {spoilage['min_separation_km_used']:.0f} km). "
            f"Mean extra travel time {spoilage['mean_time_penalty_min']:.1f} min. Range "
            f"[{spoilage['min_spoilage_reduction_pct']:.1f}%, {spoilage['max_spoilage_reduction_pct']:.1f}%].",
        )
    except Exception as exc:
        record("Cold-Chain Optimizer", "Mean Spoilage Cost Reduction", "ERROR", "n/a", "FAIL", str(exc))

    # --- 3. Telemetry cross-validation + 4. self-calibration ----------------
    try:
        sim_origin = "Port Nolloth" if "Port Nolloth" in TOWNS else list(TOWNS.keys())[0]
        sim_destination = "Upington" if "Upington" in TOWNS else list(TOWNS.keys())[-1]

        pings = generate_synthetic_traccar_session(G, cluster_coord, sim_origin, sim_destination)
        cross = cross_validate_telemetry(G, cluster_coord, pings)

        rmse_status = "PASS" if cross["spatial_rmse_m"] <= SPATIAL_RMSE_TOLERANCE_M else "FAIL"
        record(
            "Telemetry Cross-Validation", "Spatial Deviation (RMSE)",
            f"{cross['spatial_rmse_m']:.1f} m", f"<= {SPATIAL_RMSE_TOLERANCE_M:.0f} m", rmse_status,
            f"MSE={cross['spatial_mse_m2']:.1f} m^2 over {cross['num_snapped_pings']}/{cross['num_pings']} pings.",
        )

        snap_status = "PASS" if cross["snap_accuracy_pct"] >= SNAP_ACCURACY_TOLERANCE_PCT else "FAIL"
        record(
            "Telemetry Cross-Validation", "Snapping Accuracy Rate",
            f"{cross['snap_accuracy_pct']:.1f}%",
            f">= {SNAP_ACCURACY_TOLERANCE_PCT:.0f}% (within {SNAP_TOLERANCE_M:.0f}m)", snap_status,
        )

        temporal_status = (
            "PASS" if abs(cross["temporal_convergence_pct_diff"]) <= TEMPORAL_CONVERGENCE_TOLERANCE_PCT else "WARN"
        )
        record(
            "Telemetry Cross-Validation", "Temporal Convergence",
            f"{cross['temporal_convergence_pct_diff']:+.1f}%",
            f"within +/-{TEMPORAL_CONVERGENCE_TOLERANCE_PCT:.0f}%", temporal_status,
            f"Real={cross['total_real_minutes']:.1f}min vs Predicted={cross['predicted_minutes']:.1f}min "
            f"(avg matched speed {cross['avg_matched_speed_kmh']:.1f} km/h).",
        )

        calibration = calibrate_speed_model(pings, cross["matched_fclasses"])
        if calibration["calibrated_speed_vector"]:
            record(
                "Self-Calibration Engine", "Speed Model Drift Detected",
                f"{len(calibration['calibrated_speed_vector'])} class(es)",
                f">= {calibration['deviation_threshold_pct']:.0f}% deviation", "WARN",
                " | ".join(calibration["notes"]),
            )
            logger.info("Calibrated Speed Vector: %s", json.dumps(calibration["calibrated_speed_vector"], indent=2))
        else:
            record(
                "Self-Calibration Engine", "Speed Model Drift Detected",
                "0 classes", f">= {calibration['deviation_threshold_pct']:.0f}% deviation", "PASS",
                "No road class deviated beyond threshold; DEFAULT_SPEED_KMH remains valid.",
            )
    except Exception as exc:
        record("Telemetry Cross-Validation", "Full Suite", "ERROR", "n/a", "FAIL", str(exc))

    print_ledger()

    any_fail = any(e.status == "FAIL" for e in AUDIT_LEDGER)
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
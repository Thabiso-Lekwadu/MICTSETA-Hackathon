"""graph_optimization: search spoilage-weight blends on the REAL graph, select +
version the optimal graph, and emit nc_road_graph.pkl — the exact artifact
live_backend.py loads. Optimisation runs Dijkstra over the real topology.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import networkx as nx

logger = logging.getLogger("nc_coldchain")


def _reweight(G: nx.Graph, thermal: float, mech: float, exp: float) -> nx.Graph:
    """Copy G and recompute spoilage_cost_edge/weight under the given blend."""
    H = G.copy()
    for _, _, d in H.edges(data=True):
        tt = float(d.get("travel_time", d.get("travel_time_hr", 0.0)))
        rough = float(d.get("roughness", 1.3))
        cost = tt * (thermal + mech * (rough ** exp))
        d["spoilage_cost_edge"] = cost
        d["weight"] = cost
    return H


def _objective(G: nx.Graph, town_nodes: dict, pairs) -> float:
    """Mean optimal spoilage cost over the evaluation OD pairs (lower is better)."""
    costs = []
    for a, b in pairs:
        na, nb = town_nodes.get(a), town_nodes.get(b)
        if na is None or nb is None:
            continue
        try:
            costs.append(nx.shortest_path_length(G, na, nb, weight="spoilage_cost_edge"))
        except Exception as exc:
            logger.warning("[GRAPH OPTIMIZATION] pair %s->%s unroutable: %s", a, b, exc)
    return sum(costs) / len(costs) if costs else float("inf")


def evaluate_candidates(bundle: dict, graph_validation: dict, opt_params: dict,
                        graph_params: dict) -> dict:
    if not graph_validation.get("valid", False):
        raise ValueError("cannot optimise: graph failed validation")
    G = bundle["G_main"]
    town_nodes = bundle["town_nodes"]
    pairs = opt_params["evaluation_pairs"]
    exp = float(graph_params["roughness_exponent"])
    results = []
    for cand in opt_params["candidate_weights"]:
        H = _reweight(G, float(cand["thermal"]), float(cand["mech"]), exp)
        score = _objective(H, town_nodes, pairs)
        results.append({"thermal": cand["thermal"], "mech": cand["mech"],
                        "objective_score": round(score, 6)})
        logger.info("[GRAPH OPTIMIZATION] candidate thermal=%.2f mech=%.2f -> %.4f",
                    cand["thermal"], cand["mech"], score)
    results.sort(key=lambda r: r["objective_score"])
    return {"objective": opt_params.get("objective", "min_mean_spoilage_cost"),
            "evaluation_pairs": pairs, "candidates": results, "best": results[0]}


def _short(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def build_optimal_graph(bundle: dict, candidate_graph_scores: dict, graph_params: dict,
                        audit_feedback: dict, repro_params: dict):
    """Bake the winning weights into the real graph. Returns (nc_road_graph, metadata).

    nc_road_graph is {G_main, main_nodes, cluster_coord, town_nodes} — the dict
    live_backend.py loads (it reads G_main / main_nodes / cluster_coord).
    """
    best = candidate_graph_scores["best"]
    H = _reweight(bundle["G_main"], float(best["thermal"]), float(best["mech"]),
                  float(graph_params["roughness_exponent"]))
    created = (datetime(2026, 8, 26, 6, 0, tzinfo=timezone.utc)
               if repro_params.get("deterministic", True) else datetime.now(timezone.utc))
    version = f"g-{_short(H.number_of_nodes(), H.number_of_edges(), best['thermal'], best['mech'])}"
    mean_w = sum(d.get("spoilage_cost_edge", 0.0) for _, _, d in H.edges(data=True)) \
        / max(1, H.number_of_edges())
    metadata = {
        "graph_version": version,
        "created_at": created.isoformat(),
        "source_data_version": _short(H.number_of_nodes(), H.number_of_edges()),
        "audit_version": _short(audit_feedback.get("reason"), audit_feedback.get("deltas")),
        "optimization_version": f"t{best['thermal']}-m{best['mech']}",
        "objective": candidate_graph_scores["objective"],
        "objective_score": best["objective_score"],
        "node_count": H.number_of_nodes(),
        "edge_count": H.number_of_edges(),
        "thermal_weight": best["thermal"],
        "mech_weight": best["mech"],
        "rebuild_reason": audit_feedback.get("reason"),
        "mean_edge_weight": round(mean_w, 6),
    }
    nc_road_graph = {"G_main": H, "main_nodes": set(bundle["main_nodes"]),
                     "cluster_coord": bundle["cluster_coord"],
                     "town_nodes": bundle["town_nodes"]}
    logger.info("[GRAPH OPTIMIZATION] Optimal graph %s (score=%.4f, %d nodes/%d edges)",
                version, best["objective_score"], H.number_of_nodes(), H.number_of_edges())
    return nc_road_graph, metadata


def publish_graph(nc_road_graph: dict, opt_params: dict):
    """Publish node-link JSON + a visualisation GeoJSON with optimal OD routes."""
    G = nc_road_graph["G_main"]
    cc = nc_road_graph["cluster_coord"]
    town_nodes = nc_road_graph["town_nodes"]

    routes = {}
    for a, b in opt_params["evaluation_pairs"]:
        na, nb = town_nodes.get(a), town_nodes.get(b)
        if na is None or nb is None:
            continue
        try:
            path = nx.shortest_path(G, na, nb, weight="spoilage_cost_edge")
            cost = nx.shortest_path_length(G, na, nb, weight="spoilage_cost_edge")
            routes[f"{a}->{b}"] = {"path": [str(n) for n in path], "cost": round(cost, 4),
                                   "coords": [list(cc[n]) for n in path]}
        except Exception:
            continue

    # published node-link graph (ids as strings for JSON portability)
    H = nx.Graph()
    node_of_town = {v: k for k, v in town_nodes.items()}
    for n in G.nodes():
        lon, lat = cc[n]
        H.add_node(str(n), lon=lon, lat=lat, name=node_of_town.get(n, ""),
                   kind="town" if n in node_of_town else "junction")
    for u, v, d in G.edges(data=True):
        H.add_edge(str(u), str(v), length_km=d.get("length_km"),
                   spoilage_cost=round(d.get("spoilage_cost_edge", 0.0), 5))
    published = nx.node_link_data(H, edges="links")

    # GeoJSON: only the town nodes + optimal routes (full edge set can be huge)
    features = []
    for name, n in town_nodes.items():
        lon, lat = cc[n]
        features.append({"type": "Feature",
                         "geometry": {"type": "Point", "coordinates": [lon, lat]},
                         "properties": {"name": name, "kind": "town"}})
    for label, r in routes.items():
        features.append({"type": "Feature",
                         "geometry": {"type": "LineString", "coordinates": r["coords"]},
                         "properties": {"route": label, "spoilage_cost": r["cost"]}})
    geojson = {"type": "FeatureCollection", "features": features, "optimal_routes": routes}
    return published, geojson

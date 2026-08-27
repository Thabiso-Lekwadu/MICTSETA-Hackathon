"""graph nodes: validate + measure the REAL bundle ({G_main, cluster_coord, ...}).

The bundle is already the constructed, boundary-clipped, largest-component graph
(from build_clean_network for OSM, or the synthetic builder). These nodes fail
loudly on structural problems and publish metrics — they do not rebuild it.
"""
from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger("nc_coldchain")


def validate_graph(bundle: dict, graph_params: dict) -> dict:
    """Fail loudly on structural problems; return a validation record."""
    G = bundle["G_main"]
    town_nodes = bundle["town_nodes"]
    problems = []
    if G.number_of_nodes() == 0:
        problems.append("graph has no nodes")
    for u, v, d in G.edges(data=True):
        w = d.get("spoilage_cost_edge", d.get("weight"))
        if w is None or w != w or w < 0:  # None / NaN / negative
            problems.append(f"invalid spoilage weight on edge {u}-{v}: {w}")
            break
    connected = nx.is_connected(G) if G.number_of_nodes() else False
    if graph_params.get("require_connected", True) and not connected:
        problems.append(f"graph not connected ({nx.number_connected_components(G)} components)")
    missing_towns = [t for t, n in town_nodes.items() if n not in G]
    if missing_towns:
        problems.append(f"towns not on graph: {missing_towns}")

    record = {"valid": not problems, "connected": connected,
              "component_count": nx.number_connected_components(G) if G.number_of_nodes() else 0,
              "problems": problems}
    if problems:
        logger.error("[GRAPH] Validation FAILED: %s", problems)
        raise ValueError(f"Graph validation failed: {problems}")
    logger.info("[GRAPH] Validation passed (connected=%s, %d towns on graph)",
                connected, len(town_nodes))
    return record


def calculate_graph_metrics(bundle: dict) -> dict:
    """Structural metrics of the real routable graph."""
    G = bundle["G_main"]
    total_km = sum(d.get("length_km", 0.0) for _, _, d in G.edges(data=True))
    degrees = [d for _, d in G.degree()]
    metrics = {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "town_count": len(bundle["town_nodes"]),
        "is_connected": nx.is_connected(G) if G.number_of_nodes() else False,
        "component_count": nx.number_connected_components(G) if G.number_of_nodes() else 0,
        "avg_degree": round(sum(degrees) / max(1, len(degrees)), 3),
        "total_length_km": round(total_km, 2),
        "inferred_connectors": sum(1 for _, _, d in G.edges(data=True) if d.get("inferred")),
    }
    logger.info("[GRAPH] Metrics: %d nodes / %d edges / %.0f km / connected=%s",
                metrics["node_count"], metrics["edge_count"],
                metrics["total_length_km"], metrics["is_connected"])
    return metrics

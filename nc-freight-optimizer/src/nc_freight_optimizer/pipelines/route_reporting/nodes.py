"""Kedro nodes for the route_reporting pipeline.

Runs the Standard-vs-Fisheries-Optimized route comparison across every town pair in
the network and produces a reporting-layer CSV. This is the same sensitivity scan
built interactively in Visualization_Simulation.ipynb, now runnable as a
non-interactive, reproducible `kedro run` step.
"""

from __future__ import annotations

import itertools

import networkx as nx
import pandas as pd

from nc_freight_optimizer import road_network_core as core
from nc_freight_optimizer.routing import compare_routes, initialize_baseline_impedance


def run_route_sensitivity_scan(
    nc_road_graph_bundle: dict,
    towns: dict[str, list[float]],
    border_posts: dict[str, list[float]],
    border_delay_hr: float,
) -> pd.DataFrame:
    graph_main: nx.Graph = nc_road_graph_bundle["G_main"]
    main_nodes = nc_road_graph_bundle["main_nodes"]
    cluster_coord = nc_road_graph_bundle["cluster_coord"]

    initialize_baseline_impedance(graph_main)

    mean_lat = sum(c[1] for c in cluster_coord.values()) / len(cluster_coord)
    mx, my = core.projection_scale(mean_lat)

    town_nodes = {}
    for name, (lon, lat) in towns.items():
        node_id, _ = core.nearest_node(lon, lat, cluster_coord, main_nodes, mx, my)
        town_nodes[name] = node_id

    border_nodes = {}
    for name, (lon, lat) in border_posts.items():
        node_id, _ = core.nearest_node(lon, lat, cluster_coord, main_nodes, mx, my)
        border_nodes[node_id] = border_delay_hr

    rows = []
    for a, b in itertools.combinations(town_nodes.keys(), 2):
        standard, optimized = compare_routes(graph_main, town_nodes[a], town_nodes[b], border_nodes)
        rows.append({
            "origin": a, "destination": b,
            "routes_differ": standard["path"] != optimized["path"],
            "standard_time_hr": standard["total_time_hr"],
            "optimized_time_hr": optimized["total_time_hr"],
            "standard_spoilage_pct": standard["spoilage_risk_pct"],
            "optimized_spoilage_pct": optimized["spoilage_risk_pct"],
            "rand_saved": standard["expected_loss_rand"] - optimized["expected_loss_rand"],
        })

    return pd.DataFrame(rows).sort_values("rand_saved", ascending=False).reset_index(drop=True)

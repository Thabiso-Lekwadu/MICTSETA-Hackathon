"""
routing.py

Spatial snapping and route-comparison logic shared by two consumers:
  1. pipelines/route_reporting/nodes.py -- the batch sensitivity scan that runs as
     part of `kedro run` and produces the data/08_reporting/ artifact.
  2. apps/live_backend.py -- the live FastAPI service, which uses the same
     spatial-index and weight-function logic against a live-mutable copy of the graph.

Kept as one module so the routing math is identical in both places -- a change here
changes what both the batch report and the live demo actually compute.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


class NodeSpatialIndex:
    """Nearest-neighbor snapping of raw GPS coordinates onto the closest node
    in the active routable topology."""

    def __init__(self, topology: nx.Graph, cluster_coord: dict[int, tuple[float, float]]):
        self._node_ids: list[int] = list(topology.nodes())
        coordinates = np.array([cluster_coord[node_id] for node_id in self._node_ids])
        mean_latitude = float(coordinates[:, 1].mean())
        self._mx = 111_320.0 * np.cos(np.radians(mean_latitude))
        self._my = 110_540.0
        projected = np.column_stack([coordinates[:, 0] * self._mx, coordinates[:, 1] * self._my])
        self._tree = cKDTree(projected)

    def snap(self, latitude: float, longitude: float) -> tuple[int, float]:
        query_point = np.array([longitude * self._mx, latitude * self._my])
        distance_m, index = self._tree.query(query_point)
        return self._node_ids[int(index)], float(distance_m) / 1000.0


class EdgeSpatialIndex:
    """Nearest-neighbor snapping of a reported GPS fix onto the closest road
    segment (edge), via each edge's midpoint coordinate."""

    def __init__(self, topology: nx.Graph, cluster_coord: dict[int, tuple[float, float]]):
        self._edges: list[tuple[int, int]] = []
        midpoints = []
        mean_latitude = float(np.mean([c[1] for c in cluster_coord.values()]))
        self._mx = 111_320.0 * np.cos(np.radians(mean_latitude))
        self._my = 110_540.0

        for u, v in topology.edges():
            lon_u, lat_u = cluster_coord[u]
            lon_v, lat_v = cluster_coord[v]
            midpoints.append(((lon_u + lon_v) / 2.0, (lat_u + lat_v) / 2.0))
            self._edges.append((u, v))

        projected = np.array([[lon * self._mx, lat * self._my] for lon, lat in midpoints])
        self._tree = cKDTree(projected)

    def snap(self, latitude: float, longitude: float) -> tuple[int, int, float]:
        query_point = np.array([longitude * self._mx, latitude * self._my])
        distance_m, index = self._tree.query(query_point)
        u, v = self._edges[int(index)]
        return u, v, float(distance_m) / 1000.0


def initialize_baseline_impedance(topology: nx.Graph) -> None:
    """
    Derives spoilage_cost and base_time_mins from the raw edge attributes
    (length_km, travel_time, roughness). Adds an `override` slot (None by default)
    to every edge so a report/incident can be layered on top without destroying the
    original tiered-imputation baseline.
    """
    for _u, _v, edge_attrs in topology.edges(data=True):
        travel_time_hr = edge_attrs["travel_time"]
        roughness = edge_attrs.get("roughness", 1.3)
        length_km = edge_attrs["length_km"]
        edge_attrs["imputed_speed_kmh"] = length_km / travel_time_hr if travel_time_hr > 0 else 0.0
        edge_attrs["base_time_mins"] = travel_time_hr * 60.0
        edge_attrs["spoilage_cost"] = travel_time_hr * roughness
        edge_attrs["override"] = None


def effective_weight(u: int, v: int, edge_attrs: dict) -> float:
    """Fallback-hierarchy weight: a report/override on this segment takes priority
    over the tiered-imputation baseline."""
    override = edge_attrs.get("override")
    if override is not None:
        return override["spoilage_cost"]
    return edge_attrs["spoilage_cost"]


def baseline_weight(u: int, v: int, edge_attrs: dict) -> float:
    """Ignores any active override -- used only to detect whether an override has
    actually changed the optimal route."""
    return edge_attrs["spoilage_cost"]


def compare_routes(
    topology: nx.Graph,
    origin_node: int,
    destination_node: int,
    border_nodes: dict[int, float],
    idle_heat_risk_factor: float = 1.2,
    spoilage_threshold: float = 20.0,
    shipment_value_rand: float = 450_000.0,
) -> tuple[dict, dict]:
    """Standard (time-only) vs spoilage-aware route comparison, with an optional
    fixed customs delay injected at any node in `border_nodes`.

    Respects the same override fallback hierarchy as effective_weight/baseline_weight:
    a segment with an active report/override uses its overridden cost, otherwise the
    tiered-imputation baseline. This is what makes it safe for the route_reporting
    pipeline and the live backend to both call this function against the same mutable
    graph and get consistent answers.
    """

    def edge_time_hr(d: dict) -> float:
        override = d.get("override")
        return override["base_time_mins"] / 60.0 if override is not None else d["travel_time"]

    def edge_spoilage(d: dict) -> float:
        override = d.get("override")
        if override is not None:
            return override["spoilage_cost"]
        return d["travel_time"] * d.get("roughness", 1.3)

    def time_weight(u, v, d):
        return edge_time_hr(d) + border_nodes.get(v, 0.0)

    def spoilage_weight(u, v, d):
        spoilage = edge_spoilage(d)
        if v in border_nodes:
            spoilage += border_nodes[v] * idle_heat_risk_factor
        return spoilage

    def evaluate(path: list[int]) -> dict:
        total_time, total_spoilage = 0.0, 0.0
        for u, v in zip(path[:-1], path[1:]):
            d = topology[u][v]
            total_time += edge_time_hr(d)
            total_spoilage += edge_spoilage(d)
            if v in border_nodes:
                delay = border_nodes[v]
                total_time += delay
                total_spoilage += delay * idle_heat_risk_factor
        spoilage_pct = min(1.0, total_spoilage / spoilage_threshold)
        return {
            "path": path,
            "total_time_hr": round(total_time, 2),
            "spoilage_index": round(total_spoilage, 2),
            "spoilage_risk_pct": round(spoilage_pct * 100, 1),
            "expected_loss_rand": round(spoilage_pct * shipment_value_rand, -2),
        }

    standard_path = nx.shortest_path(topology, origin_node, destination_node, weight=time_weight)
    optimized_path = nx.shortest_path(topology, origin_node, destination_node, weight=spoilage_weight)
    return evaluate(standard_path), evaluate(optimized_path)

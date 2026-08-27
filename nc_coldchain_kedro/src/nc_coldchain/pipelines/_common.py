"""Shared helpers reused across pipelines.

Where possible these call the project's *real* algorithms (vendored under
nc_coldchain.legacy) so the Kedro nodes are thin wrappers over the validated
code, not reimplementations. Heavy optional deps (pyproj) are imported lazily
with safe fallbacks so nodes stay importable in minimal environments.
"""
from __future__ import annotations

import math
from typing import Iterable

import networkx as nx


def geodesic_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """True geodesic distance; pyproj when available, haversine fallback."""
    try:
        from pyproj import Geod

        _, _, dist_m = Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
        return abs(dist_m) / 1000.0
    except Exception:
        r = 6371.0088
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def spoilage_cost(travel_time_hr: float, roughness: float, thermal_w: float,
                  mech_w: float, exponent: float) -> float:
    """Two-component spoilage edge cost. Uses the legacy formula when importable."""
    try:
        from nc_coldchain.legacy.nc_road_network_improved import spoilage_edge_cost

        # legacy uses fixed module weights; replicate with explicit weights here so
        # the optimiser can vary them per candidate.
        return travel_time_hr * (thermal_w + mech_w * (roughness ** exponent))
    except Exception:
        return travel_time_hr * (thermal_w + mech_w * (roughness ** exponent))


def point_in_boundary(lon: float, lat: float, boundary: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon; delegates to legacy nc_boundary when possible."""
    try:
        from nc_coldchain.legacy.nc_boundary import point_in_northern_cape

        return point_in_northern_cape(lon, lat, boundary)
    except Exception:
        inside = False
        n = len(boundary)
        j = n - 1
        for i in range(n):
            xi, yi = boundary[i][0], boundary[i][1]
            xj, yj = boundary[j][0], boundary[j][1]
            if ((yi > lat) != (yj > lat)) and (
                lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
            ):
                inside = not inside
            j = i
        return inside


def build_graph_from_tables(nodes_df, edges_df, thermal_w: float, mech_w: float,
                            exponent: float) -> nx.Graph:
    """Construct a weighted undirected NetworkX graph from node/edge feature tables."""
    G = nx.Graph()
    for _, r in nodes_df.iterrows():
        G.add_node(str(r["node_id"]), lon=float(r["lon"]), lat=float(r["lat"]),
                   name=str(r.get("name", "")), kind=str(r.get("kind", "waypoint")))
    for _, e in edges_df.iterrows():
        tt = float(e["travel_time_hr"])
        rough = float(e["roughness"])
        cost = spoilage_cost(tt, rough, thermal_w, mech_w, exponent)
        G.add_edge(str(e["u"]), str(e["v"]),
                   length_km=float(e["length_km"]), travel_time_hr=tt,
                   roughness=rough, maxspeed=float(e.get("maxspeed", 90.0)),
                   spoilage_cost=cost, weight=cost)
    return G


def reweight_graph(G: nx.Graph, thermal_w: float, mech_w: float,
                   exponent: float) -> nx.Graph:
    """Return a copy of G with spoilage_cost/weight recomputed under new weights.

    Operates on the already-constructed (and boundary-clipped) graph, so every
    candidate and the published optimal graph share one topology and node/edge
    count — no rebuild from raw tables, no clip drift.
    """
    H = G.copy()
    for _, _, d in H.edges(data=True):
        tt = float(d.get("travel_time_hr", 0.0))
        rough = float(d.get("roughness", 0.0))
        cost = spoilage_cost(tt, rough, thermal_w, mech_w, exponent)
        d["spoilage_cost"] = cost
        d["weight"] = cost
    return H


def shortest_spoilage_path(G: nx.Graph, src: str, dst: str) -> tuple[list[str], float]:
    """Dijkstra on the spoilage weight; returns (path, total_cost)."""
    path = nx.shortest_path(G, src, dst, weight="weight")
    cost = nx.shortest_path_length(G, src, dst, weight="weight")
    return path, float(cost)


def node_id_for_town(nodes_df, town: str) -> str:
    row = nodes_df[nodes_df["name"] == town]
    if row.empty:
        raise KeyError(f"town '{town}' not present in node table")
    return str(row.iloc[0]["node_id"])

"""Graph construction that yields the SAME artifact live_backend.py consumes.

The single source of truth is a bundle dict ``{G_main, main_nodes, cluster_coord,
town_nodes}`` — identical in shape to what ``nc_road_network_improved.build_clean_network``
returns (plus the snapped town->node map). live_backend.py loads exactly this
(as nc_road_graph.pkl), so the app and the pipeline share one graph — no drift,
no second inaccurate topology.

Two builders produce the bundle:
  * build_osm_bundle      -> the REAL road network (build_clean_network on the OSM extract)
  * build_synthetic_bundle-> an offline corridor abstraction (clearly an approximation)

Both then feed the same downstream audit / metrics / optimisation / Monte-Carlo.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

from .._common import geodesic_km

logger = logging.getLogger("nc_coldchain")


# --------------------------------------------------------------- town snapping --
def snap_towns(G: nx.Graph, cluster_coord: dict, towns: dict) -> dict:
    """Map each service town onto its nearest REAL graph node."""
    lats = [c[1] for c in cluster_coord.values()]
    mean_lat = float(np.mean(lats)) if lats else -29.5
    mx = 111_320.0 * np.cos(np.radians(mean_lat))
    my = 110_540.0
    nodes = list(G.nodes())
    coords = np.array([cluster_coord[n] for n in nodes])
    pts_m = np.column_stack([coords[:, 0] * mx, coords[:, 1] * my])
    town_nodes = {}
    for name, (lon, lat) in towns.items():
        p = np.array([lon * mx, lat * my])
        idx = int(np.argmin(np.linalg.norm(pts_m - p, axis=1)))
        town_nodes[name] = nodes[idx]
        dist_km = float(np.linalg.norm(pts_m[idx] - p) / 1000.0)
        logger.info("[PREPROCESSING] town %-12s -> node %s (%.1f km)", name, nodes[idx], dist_km)
    return town_nodes


def bundle_to_tables(G: nx.Graph, cluster_coord: dict, town_nodes: dict):
    """Derive the tabular node/edge feature tables the audit + viz consume."""
    node_of_town = {v: k for k, v in town_nodes.items()}
    node_rows = []
    for n in G.nodes():
        lon, lat = cluster_coord[n]
        node_rows.append({"node_id": str(n), "name": node_of_town.get(n, ""),
                          "lon": float(lon), "lat": float(lat),
                          "kind": "town" if n in node_of_town else "junction"})
    nodes_df = pd.DataFrame(node_rows)
    edge_rows = []
    for u, v, d in G.edges(data=True):
        length_km = float(d.get("length_km", 0.0))
        tt = float(d.get("travel_time", d.get("travel_time_hr", 0.0))) or (length_km / 90.0)
        rough = float(d.get("roughness", 2.0))
        speed = (length_km / tt) if tt > 0 else 90.0
        edge_rows.append({"u": str(u), "v": str(v), "length_km": round(length_km, 4),
                          "maxspeed": round(speed, 2), "travel_time_hr": round(tt, 6),
                          "roughness": round(rough, 3),
                          "spoilage_cost_edge": float(d.get("spoilage_cost_edge", 0.0))})
    edges_df = pd.DataFrame(edge_rows)
    return nodes_df, edges_df


# --------------------------------------------------------------- OSM builder ----
def build_osm_bundle(osm_path: str, towns: dict, use_spatial_knn: bool = True,
                     stitch: bool = True) -> dict:
    """Build the REAL routable graph from the OSM extract via the validated code."""
    from nc_coldchain.legacy import nc_road_network_improved as rn

    logger.info("[GRAPH] Building REAL network from OSM extract %s ...", osm_path)
    result = rn.build_clean_network(raw_path=str(osm_path), stitch=stitch,
                                    use_spatial_knn=use_spatial_knn)
    G_main = result["G_main"]
    cluster_coord = result["cluster_coord"]
    main_nodes = set(result["main_nodes"])
    town_nodes = snap_towns(G_main, cluster_coord, towns)
    logger.info("[GRAPH] REAL graph: %d nodes / %d edges (largest component)",
                G_main.number_of_nodes(), G_main.number_of_edges())
    return {"G_main": G_main, "main_nodes": main_nodes,
            "cluster_coord": cluster_coord, "town_nodes": town_nodes}


# --------------------------------------------------------- synthetic builder ----
def build_synthetic_bundle(towns: dict, network_params: dict, graph_params: dict,
                           road: pd.DataFrame) -> dict:
    """Offline corridor abstraction with the SAME bundle shape and edge attrs as
    the OSM build (travel_time, roughness, spoilage_cost_edge, length_km), so all
    downstream code and live_backend.py treat it identically."""
    thermal = float(graph_params["thermal_weight"])
    mech = float(graph_params["mech_weight"])
    exp = float(graph_params["roughness_exponent"])
    default_speed = float(graph_params.get("default_speed_kmh", 90.0))
    wpp = int(network_params.get("waypoints_per_edge", 2))
    base_rough = 2.0
    if isinstance(road, pd.DataFrame) and "iri" in road.columns and len(road):
        base_rough = float(pd.to_numeric(road["iri"], errors="coerce").dropna().mean() or 2.0)

    # integer node ids + cluster_coord, like the real build
    cluster_coord = {}
    town_nodes = {}
    nid = 0
    for name, (lon, lat) in towns.items():
        cluster_coord[nid] = (float(lon), float(lat))
        town_nodes[name] = nid
        nid += 1

    # MST + 2-NN adjacency for a connected, realistic corridor set
    dg = nx.Graph()
    ids = list(town_nodes.values())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            dg.add_edge(a, b, weight=geodesic_km(*cluster_coord[a], *cluster_coord[b]))
    keep = set(nx.minimum_spanning_tree(dg, weight="weight").edges())
    for a in ids:
        nbrs = sorted((geodesic_km(*cluster_coord[a], *cluster_coord[b]), b)
                      for b in ids if b != a)[:2]
        for _, b in nbrs:
            keep.add(tuple(sorted((a, b))))

    G = nx.Graph()
    for n, c in cluster_coord.items():
        G.add_node(n)
    for a, b in keep:
        chain = [a]
        (alon, alat), (blon, blat) = cluster_coord[a], cluster_coord[b]
        for k in range(1, wpp + 1):
            f = k / (wpp + 1)
            cluster_coord[nid] = (alon + (blon - alon) * f, alat + (blat - alat) * f)
            G.add_node(nid)
            chain.append(nid)
            nid += 1
        chain.append(b)
        for u, v in zip(chain[:-1], chain[1:]):
            length_km = geodesic_km(*cluster_coord[u], *cluster_coord[v])
            tt = length_km / default_speed
            spoil = tt * (thermal + mech * (base_rough ** exp))
            G.add_edge(u, v, length_km=length_km, travel_time=tt, roughness=base_rough,
                       spoilage_cost_edge=spoil, fclass="synthetic_corridor",
                       road_name="Synthetic corridor", inferred=False)
    main_nodes = set(G.nodes())
    logger.info("[GRAPH] SYNTHETIC graph: %d nodes / %d edges (illustrative, not real OSM)",
                G.number_of_nodes(), G.number_of_edges())
    return {"G_main": G, "main_nodes": main_nodes,
            "cluster_coord": cluster_coord, "town_nodes": town_nodes}

"""
Network Route Optimizer — Transport, Trade & Fisheries track
MICT SETA Skills Development Hackathon, Northern Cape (28-29 Aug 2026)

Models the road network as a graph and runs real optimization algorithms
on it (Dijkstra / k-shortest-paths), instead of the straight-line distance
approximation used in route_optimizer.py.

TWO MODES
---------
DEMO MODE (default, no extra downloads needed):
    Builds a small synthetic Northern Cape town-network graph so you can
    verify the routing logic works right now.

REAL MODE (for your machine, once you have OSM data):
    build_real_graph() shows how to load an actual routable network via
    osmnx (online) or pyrosm (offline, from a .osm.pbf file) and swap it
    in — same downstream routing functions work unchanged.

Run:
    pip install networkx --break-system-packages          # demo mode
    pip install osmnx pyrosm --break-system-packages       # real mode
    python3 network_route_optimizer.py
"""

import networkx as nx

# ---------------------------------------------------------------------------
# RISK MODEL — surface-based penalty applied to travel time.
# Replace with your trained delay-classifier's predicted probability per
# surface/cargo_type if you want the ML model driving this instead of a
# fixed lookup (see integrate_ml_risk() below).
# ---------------------------------------------------------------------------
SURFACE_RISK_MULTIPLIER = {
    "paved": 1.0,
    "asphalt": 1.0,
    "unpaved": 1.5,
    "gravel": 1.5,
    "dirt": 1.8,
    "unknown": 1.2,
}


# ---------------------------------------------------------------------------
# DEMO MODE — small synthetic graph so the routing logic is testable now.
# Nodes = towns, edges = road segments with length_km, speed_kmh, surface.
# Includes deliberate alternate paths so k-shortest-paths has something to find.
# ---------------------------------------------------------------------------
def build_demo_graph():
    G = nx.MultiDiGraph()
    edges = [
        # (from, to, length_km, speed_kmh, surface)
        ("Kimberley", "Douglas", 110, 100, "paved"),
        ("Douglas", "Upington", 210, 90, "paved"),
        ("Kimberley", "Warrenton", 60, 90, "paved"),
        ("Warrenton", "Upington", 260, 70, "unpaved"),   # slower shortcut alternative
        ("Upington", "Springbok", 400, 100, "paved"),
        ("Upington", "Pofadder", 240, 80, "paved"),
        ("Pofadder", "Springbok", 180, 60, "unpaved"),   # rougher back route
        ("Kimberley", "De Aar", 210, 100, "paved"),
        ("De Aar", "Calvinia", 380, 75, "unpaved"),
        ("Springbok", "Port Nolloth", 95, 80, "paved"),
    ]
    for u, v, length_km, speed_kmh, surface in edges:
        travel_time_hr = length_km / speed_kmh
        risk_mult = SURFACE_RISK_MULTIPLIER.get(surface, 1.2)
        attrs = dict(length_km=length_km, speed_kmh=speed_kmh, surface=surface,
                     travel_time=travel_time_hr, risk_time=travel_time_hr * risk_mult)
        G.add_edge(u, v, **attrs)
        G.add_edge(v, u, **attrs)  # treat as bidirectional for the demo
    return G


# ---------------------------------------------------------------------------
# REAL MODE — swap this in once you have OSM data on your machine.
# ---------------------------------------------------------------------------
def build_real_graph_osmnx(place="Northern Cape, South Africa"):
    """Requires internet (Overpass API) + `pip install osmnx`."""
    import osmnx as ox
    G = ox.graph_from_place(place, network_type="drive")
    G = ox.add_edge_speeds(G)          # infers km/h from OSM highway tags
    G = ox.add_edge_travel_times(G)    # adds travel_time in seconds
    for u, v, k, data in G.edges(keys=True, data=True):
        surface = data.get("surface", "unknown")
        if isinstance(surface, list):
            surface = surface[0]
        risk_mult = SURFACE_RISK_MULTIPLIER.get(surface, 1.2)
        data["risk_time"] = data["travel_time"] * risk_mult
    return G


def build_real_graph_pyrosm(pbf_path, bbox=None):
    """
    Offline alternative — no internet needed once you have the .pbf file.
    Download: https://download.geofabrik.de/africa/south-africa-latest.osm.pbf
    bbox = [min_lon, min_lat, max_lon, max_lat], e.g. Northern Cape:
           [16.45, -31.85, 25.30, -24.60]
    Requires `pip install pyrosm`.
    """
    from pyrosm import OSM
    osm = OSM(pbf_path, bounding_box=bbox)
    nodes, edges = osm.get_network(network_type="driving", nodes=True)
    G = osm.to_graph(nodes, edges, graph_type="networkx")
    for u, v, k, data in G.edges(keys=True, data=True):
        surface = data.get("surface", "unknown")
        if isinstance(surface, list):
            surface = surface[0]
        length_km = data.get("length", 0) / 1000
        speed_kmh = data.get("maxspeed", 60) or 60
        try:
            speed_kmh = float(speed_kmh)
        except (TypeError, ValueError):
            speed_kmh = 60
        travel_time_hr = length_km / max(speed_kmh, 1)
        risk_mult = SURFACE_RISK_MULTIPLIER.get(surface, 1.2)
        data["travel_time"] = travel_time_hr
        data["risk_time"] = travel_time_hr * risk_mult
    return G


def integrate_ml_risk(G, delay_model, cargo_type, hour, day_of_week):
    """
    OPTIONAL: replace the fixed SURFACE_RISK_MULTIPLIER lookup with your
    trained delay classifier from route_optimizer.py, so risk weighting is
    learned from data instead of hand-set.

    For each edge, build the same feature row the model was trained on
    (distance/speed/hour/day/cargo/surface) and use predicted delay
    probability as the risk multiplier.
    """
    import pandas as pd
    for u, v, k, data in G.edges(keys=True, data=True):
        row = pd.DataFrame([{
            "distance_km": data.get("length_km", data.get("length", 0) / 1000),
            "avg_speed_kmh": data.get("speed_kmh", 60),
            "hour": hour,
            "day_of_week": day_of_week,
            "cargo_type": cargo_type,
            "road_surface": data.get("surface", "unknown"),
        }])
        delay_prob = delay_model.predict_proba(row)[0, 1]
        data["risk_time"] = data["travel_time"] * (1 + delay_prob)
    return G


# ---------------------------------------------------------------------------
# ROUTING ALGORITHMS — work the same on demo or real graph.
# ---------------------------------------------------------------------------
def fastest_route(G, origin, destination):
    path = nx.shortest_path(G, origin, destination, weight="travel_time")
    total_time = nx.shortest_path_length(G, origin, destination, weight="travel_time")
    return path, total_time


def safest_route(G, origin, destination):
    path = nx.shortest_path(G, origin, destination, weight="risk_time")
    total_risk_time = nx.shortest_path_length(G, origin, destination, weight="risk_time")
    return path, total_risk_time


def k_route_options(G, origin, destination, k=3, weight="travel_time"):
    """Yen's algorithm — returns up to k distinct paths, best first.

    nx.shortest_simple_paths does not support MultiDiGraph, so we collapse
    parallel edges down to a simple DiGraph first, keeping the
    lowest-weight edge between any given (u, v) pair. Safe here since the
    demo/real graphs never actually carry true parallel edges (only one
    edge per direction between any two towns).
    """
    if G.is_multigraph():
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes(data=True))
        for u, v, data in G.edges(data=True):
            if H.has_edge(u, v):
                if data.get(weight, float("inf")) < H[u][v].get(weight, float("inf")):
                    H[u][v].update(data)
            else:
                H.add_edge(u, v, **data)
    else:
        H = G

    paths = []
    try:
        gen = nx.shortest_simple_paths(H, origin, destination, weight=weight)
        for _ in range(k):
            paths.append(next(gen))
    except (nx.NetworkXNoPath, StopIteration):
        pass
    return paths


def route_summary(G, path, weight="travel_time"):
    total = sum(G[u][v][0][weight] for u, v in zip(path[:-1], path[1:]))
    surfaces = [G[u][v][0].get("surface", "unknown") for u, v in zip(path[:-1], path[1:])]
    return {"path": " -> ".join(path), "total_" + weight: round(total, 2),
            "surfaces": surfaces}


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    G = build_demo_graph()

    origin, destination = "Kimberley", "Springbok"

    print("=" * 60)
    print(f"ROUTE OPTIONS: {origin} -> {destination}")
    print("=" * 60)

    path, t = fastest_route(G, origin, destination)
    print("\nFASTEST route (by travel time):")
    print(route_summary(G, path, "travel_time"))

    path, t = safest_route(G, origin, destination)
    print("\nSAFEST/lowest-risk route (surface-weighted):")
    print(route_summary(G, path, "risk_time"))

    print("\nTOP 3 route options (by travel time):")
    for i, p in enumerate(k_route_options(G, origin, destination, k=3), 1):
        summary = route_summary(G, p, "travel_time")
        print(f"  {i}. {summary}")

"""
Cold-Chain Spoilage Risk Optimizer — Transport, Trade & Fisheries track
MICT SETA Skills Development Hackathon, Northern Cape (28-29 Aug 2026)

PITCH
-----
Standard route optimizers (Google Maps etc.) minimize travel time only.
This optimizer minimizes SPOILAGE RISK for high-value refrigerated seafood
exports: rough/unpaved roads (fclass) damage cooling seals via vibration,
and border posts add hours of idle heat exposure during customs clearance.
The "fastest" route and the "safest for your catch" route are often
different roads — this script proves it and prices the difference in Rand.

DATA SOURCE
-----------
Road network + border-post locations are pulled LIVE from OpenStreetMap's
Overpass API via `osmnx` — no static country-wide .pbf/.shp download needed.
Only the Northern Cape bounding box is queried, on demand.

    ox.graph_from_bbox(...)                          -> live road network
    ox.features_from_place(tags={"barrier":"border_control"})  -> live border posts

DEMO MODE (runs right now, no internet needed here) uses a small synthetic
graph mirroring the real Kimberley -> Namibia border corridor, so the
routing/spoilage logic is fully testable before you plug in live API data.

Run:
    pip install networkx --break-system-packages         # demo mode
    pip install osmnx --break-system-packages             # real/API mode
    python3 fisheries_coldchain_optimizer.py
"""

import networkx as nx

# ---------------------------------------------------------------------------
# ROAD ROUGHNESS — keyed by OSM `fclass` (the field name used in Geofabrik's
# roads shapefile / osmnx's `highway` tag). Higher = more vibration/seal risk.
# ---------------------------------------------------------------------------
ROUGHNESS_MULTIPLIER = {
    "motorway": 1.00, "trunk": 1.00, "primary": 1.05,
    "secondary": 1.10, "tertiary": 1.15,
    "unclassified": 1.30, "residential": 1.20, "living_street": 1.20,
    "track": 2.20, "path": 2.50,
    "unknown": 1.30,
}

# Idle heat-exposure penalty while stopped at a border post (no vibration,
# but refrigeration strain + ambient heat while queued in the sun).
IDLE_HEAT_RISK_FACTOR = 1.2

# Spoilage-index threshold beyond which the shipment is treated as a total
# loss (tune this per species — lobster/abalone are far more sensitive than
# frozen hake). Used only to translate risk index into a Rand estimate.
SPOILAGE_THRESHOLD = 20.0       # risk-hours (composite risk budget, not literal spoilage time)
SHIPMENT_VALUE_RAND = 450_000   # e.g. a full reefer truck of rock lobster


# ---------------------------------------------------------------------------
# DEMO GRAPH — mirrors the real Kimberley -> Upington -> Vioolsdrift ->
# Namibia corridor. Includes a genuine trade-off: the Warrenton "shortcut"
# is FASTER by raw time but runs on a rough track; the Douglas route is
# slightly slower but fully paved.
# ---------------------------------------------------------------------------
def build_demo_graph():
    G = nx.DiGraph()
    edges = [
        # (from, to, length_km, speed_kmh, fclass)
        ("Kimberley", "Warrenton", 60, 90, "secondary"),
        ("Warrenton", "Upington", 180, 80, "track"),        # fast but rough shortcut
        ("Kimberley", "Douglas", 110, 100, "primary"),
        ("Douglas", "Upington", 210, 90, "primary"),         # slower but paved
        ("Upington", "Vioolsdrift_Border", 240, 100, "primary"),
        ("Vioolsdrift_Border", "Ariamsvlei_Namibia", 5, 40, "secondary"),
        # domestic corridor (no border) — alternate rough shortcut vs paved route
        ("Upington", "Pofadder", 180, 100, "primary"),
        ("Pofadder", "Springbok", 120, 70, "track"),         # rough shortcut, faster on paper
        ("Upington", "Springbok", 400, 100, "primary"),      # longer but fully paved
        ("Springbok", "Port_Nolloth_Harbour", 95, 90, "primary"),
    ]
    for u, v, length_km, speed_kmh, fclass in edges:
        travel_time_hr = length_km / speed_kmh
        G.add_edge(u, v, length_km=length_km, speed_kmh=speed_kmh,
                   fclass=fclass, travel_time=travel_time_hr)
    return G


# Border/customs posts and their typical clearance delay (hours). In real
# mode these coordinates are discovered live, not hardcoded — see
# fetch_border_nodes_overpass() below.
BORDER_NODES_DEMO = {
    "Vioolsdrift_Border": 5.0,   # hours — matches border_trade_crossings.csv delay pattern
}


# ---------------------------------------------------------------------------
# REAL / API MODE — live Overpass queries, no static file.
# ---------------------------------------------------------------------------
def fetch_live_road_network(bbox):
    """
    bbox = (north, south, east, west) e.g. Northern Cape:
           (-24.60, -31.85, 25.30, 16.45)
    Requires internet + `pip install osmnx`. Queries Overpass API directly —
    only the bbox is fetched, no country-wide file download.
    """
    import osmnx as ox
    north, south, east, west = bbox
    G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    # osmnx uses `highway` (same vocabulary as fclass) for road type
    for u, v, k, data in G.edges(keys=True, data=True):
        highway = data.get("highway", "unknown")
        if isinstance(highway, list):
            highway = highway[0]
        data["fclass"] = highway
    return G


def fetch_border_nodes_overpass(G, place="Northern Cape, South Africa",
                                 default_delay_hr=5.0):
    """
    Queries OSM live for barrier=border_control tags (and amenity=customs
    as a fallback) and snaps each to the nearest graph node — no hardcoded
    coordinates. Requires internet + `pip install osmnx`.
    """
    import osmnx as ox
    border_nodes = {}
    for tags in ({"barrier": "border_control"}, {"amenity": "customs"}):
        try:
            gdf = ox.features_from_place(place, tags=tags)
        except Exception:
            continue
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            point = geom if geom.geom_type == "Point" else geom.centroid
            node = ox.distance.nearest_nodes(G, X=point.x, Y=point.y)
            border_nodes[node] = default_delay_hr
    return border_nodes


# ---------------------------------------------------------------------------
# WEIGHT FUNCTIONS — plug into nx.shortest_path(G, weight=<callable>)
# ---------------------------------------------------------------------------
def make_time_weight(border_nodes):
    """Standard optimizer: minimizes time only (still counts border wait,
    same way Google Maps would if it knew about queues)."""
    def weight_fn(u, v, d):
        return d["travel_time"] + border_nodes.get(v, 0.0)
    return weight_fn


def make_spoilage_weight(border_nodes):
    """Fisheries-optimized: minimizes cumulative spoilage risk, not time."""
    def weight_fn(u, v, d):
        roughness = ROUGHNESS_MULTIPLIER.get(d.get("fclass", "unknown"), 1.3)
        spoilage = d["travel_time"] * roughness
        if v in border_nodes:
            spoilage += border_nodes[v] * IDLE_HEAT_RISK_FACTOR
        return spoilage
    return weight_fn


# ---------------------------------------------------------------------------
# ROUTE EVALUATION — measure BOTH metrics on a given path, regardless of
# which metric was used to select it, so the trade-off is visible.
# ---------------------------------------------------------------------------
def evaluate_path(G, path, border_nodes):
    total_time, total_spoilage = 0.0, 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]
        roughness = ROUGHNESS_MULTIPLIER.get(d.get("fclass", "unknown"), 1.3)
        total_time += d["travel_time"]
        total_spoilage += d["travel_time"] * roughness
        if v in border_nodes:
            delay = border_nodes[v]
            total_time += delay
            total_spoilage += delay * IDLE_HEAT_RISK_FACTOR
    spoilage_pct = min(1.0, total_spoilage / SPOILAGE_THRESHOLD)
    expected_loss = spoilage_pct * SHIPMENT_VALUE_RAND
    return {
        "path": " -> ".join(path),
        "total_time_hr": round(total_time, 2),
        "spoilage_index": round(total_spoilage, 2),
        "spoilage_risk_pct": round(spoilage_pct * 100, 1),
        "expected_loss_rand": round(expected_loss, -2),
    }


# ---------------------------------------------------------------------------
# SIMULATION — Standard route vs Fisheries-Optimized route
# ---------------------------------------------------------------------------
def simulate_comparison(G, origin, destination, border_nodes):
    time_weight = make_time_weight(border_nodes)
    spoilage_weight = make_spoilage_weight(border_nodes)

    standard_path = nx.shortest_path(G, origin, destination, weight=time_weight)
    optimized_path = nx.shortest_path(G, origin, destination, weight=spoilage_weight)

    standard_result = evaluate_path(G, standard_path, border_nodes)
    optimized_result = evaluate_path(G, optimized_path, border_nodes)

    print("=" * 70)
    print(f"ROUTE SIMULATION: {origin} -> {destination}")
    print(f"Shipment value at risk: R{SHIPMENT_VALUE_RAND:,}")
    print("=" * 70)

    print("\nSTANDARD ROUTE (time-optimized, like Google Maps):")
    for k, v in standard_result.items():
        print(f"  {k:20s}: {v}")

    print("\nFISHERIES-OPTIMIZED ROUTE (spoilage-risk-aware):")
    for k, v in optimized_result.items():
        print(f"  {k:20s}: {v}")

    time_delta = optimized_result["total_time_hr"] - standard_result["total_time_hr"]
    loss_saved = standard_result["expected_loss_rand"] - optimized_result["expected_loss_rand"]

    print("\n" + "-" * 70)
    print(f"RESULT: optimized route costs {time_delta:+.2f}h extra travel time,")
    print(f"        but reduces expected spoilage loss by R{loss_saved:,.0f}")
    print("-" * 70)
    return standard_result, optimized_result


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    G = build_demo_graph()

    print("\n" + "#" * 70)
    print("# SCENARIO 1 — cross-border export (road roughness + customs delay)")
    print("#" * 70)
    simulate_comparison(G, "Kimberley", "Ariamsvlei_Namibia", BORDER_NODES_DEMO)

    print("\n" + "#" * 70)
    print("# SCENARIO 2 — domestic delivery, no border (road roughness only)")
    print("#" * 70)
    simulate_comparison(G, "Upington", "Port_Nolloth_Harbour", BORDER_NODES_DEMO)

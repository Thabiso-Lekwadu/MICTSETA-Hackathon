"""
live_backend.py

FastAPI backend for the Northern Cape Fleet Dispatch system. Loads the pre-compiled
road network topology (nc_road_graph.pkl), ingests vehicle telemetry from TWO
interchangeable sources (a real Traccar Client mobile app feed, or a built-in
time-based simulator that moves smoothly along a computed route), computes BOTH
a standard (time-only) route and a spoilage-risk-optimized route so the trade-off
is visible, and accepts crowd-sourced driver ground-truth reports that override
routing impedance in real time.

Run standalone (this is a real server, not an in-process mock):
    uv run live_backend.py

Serves on http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree

import nc_road_network as ncr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRAPH_PATH = Path("nc_road_graph.pkl")
HOST = "0.0.0.0"  # 0.0.0.0 so a phone on the same network can reach /v1/telematics/incoming
PORT = 8000

VEHICLE_ID = "TRUCK-01"

FEED_SOURCE_REAL = "REAL-TIME TRACCAR HARDWARE"
FEED_SOURCE_SIMULATED = "SIMULATED TELEMETRY MATRIX"

# Simulated journeys run faster than real time so a demo trip finishes in
# minutes instead of hours. Position is a pure function of elapsed wall-clock
# time (not of how often the frontend polls), which is what makes movement
# smooth instead of twitchy.
SIM_TIME_ACCELERATION = 45.0

# Business-value constants (mirrors fisheries_coldchain_optimizer.py)
SPOILAGE_THRESHOLD = 20.0        # risk-hours budget before treated as total loss
SHIPMENT_VALUE_RAND = 450_000.0  # full reefer truck of rock lobster, at risk

# Documented, not fitted, roughness assumptions per reported road condition.
# Matches the roughness-multiplier scale established in nc_road_network.py.
ROAD_CONDITION_ROUGHNESS: dict[str, float] = {
    "Smooth Tarmac": 1.0,
    "Corrugated / Rough Gravel": 1.8,
    "Severe Potholes": 2.6,
    "Impassable / Washed Out": 6.0,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live_backend")

state_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class FieldReport(BaseModel):
    reporter_role: Literal["Driver", "Fisherman", "Cooperative Supervisor"]
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    road_condition: Literal[
        "Smooth Tarmac", "Corrugated / Rough Gravel", "Severe Potholes", "Impassable / Washed Out"
    ]
    actual_speed: float = Field(..., gt=0.0, le=160.0)


class TripConfig(BaseModel):
    mode: Literal["simulator", "hardware"]
    destination_town: str
    origin_town: str | None = None  # required for simulator mode only


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
tracking_state: dict = {
    "vehicle_id": VEHICLE_ID,
    "lat": None,
    "lon": None,
    "timestamp": int(time.time()),
    "cargo_temp_c": 3.5,
    "feed_source": FEED_SOURCE_SIMULATED,
    "speed_kmh": 0.0,
    "bearing": 0.0,
}

trip_state: dict = {
    "configured": False,
    "mode": "simulator",
    "origin_town": None,
    "destination_town": None,
    "origin_node": None,
    "destination_node": None,
    "sim_cumulative": None,   # list of (cum_hours, lon, lat) along optimized path, simulator only
    "sim_total_hours": 0.0,
    "trip_start_ts": None,
    "arrived": False,
}

active_driver_reports: dict[tuple[int, int], dict] = {}


# ---------------------------------------------------------------------------
# Topology loading and impedance matrix initialization
# ---------------------------------------------------------------------------
def load_topology() -> tuple[nx.Graph, set, dict[int, tuple[float, float]]]:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"'{GRAPH_PATH}' not found. Run Data_Audit.ipynb first to produce it, "
            f"or copy an existing nc_road_graph.pkl into this directory."
        )
    with GRAPH_PATH.open("rb") as f:
        artifacts = pickle.load(f)
    return artifacts["G_main"], artifacts["main_nodes"], artifacts["cluster_coord"]


def initialize_baseline_impedance(topology: nx.Graph) -> None:
    """Derives routing impedance fields from the raw edge attributes produced
    by nc_road_network.py. `spoilage_cost` is the primary (optimized) routing
    weight; `base_time_mins` alone drives the standard (time-only) route.
    Every edge gets an `override` slot (None until a driver report lands)."""
    for _u, _v, edge_attrs in topology.edges(data=True):
        travel_time_hr = edge_attrs["travel_time"]
        roughness = edge_attrs.get("roughness", 1.3)
        length_km = edge_attrs["length_km"]
        edge_attrs["imputed_speed_kmh"] = length_km / travel_time_hr if travel_time_hr > 0 else 0.0
        edge_attrs["base_time_mins"] = travel_time_hr * 60.0
        edge_attrs["spoilage_cost"] = travel_time_hr * roughness
        edge_attrs["override"] = None


def effective_time_mins(u: int, v: int, edge_attrs: dict) -> float:
    override = edge_attrs.get("override")
    return override["base_time_mins"] if override is not None else edge_attrs["base_time_mins"]


def effective_spoilage(u: int, v: int, edge_attrs: dict) -> float:
    override = edge_attrs.get("override")
    return override["spoilage_cost"] if override is not None else edge_attrs["spoilage_cost"]


def time_weight(u: int, v: int, edge_attrs: dict) -> float:
    """Standard optimizer: minutes only, like a generic maps app."""
    return effective_time_mins(u, v, edge_attrs)


def spoilage_weight(u: int, v: int, edge_attrs: dict) -> float:
    """Fisheries-optimized: minimizes cumulative spoilage risk, not time."""
    return effective_spoilage(u, v, edge_attrs)


# ---------------------------------------------------------------------------
# Spatial indices
# ---------------------------------------------------------------------------
class NodeSpatialIndex:
    """Nearest-neighbor snapping of raw GPS coordinates onto the closest node
    in the active routable topology (G_main)."""

    def __init__(self, topology: nx.Graph, cluster_coord: dict[int, tuple[float, float]]):
        self._node_ids: list[int] = list(topology.nodes())
        coordinates = np.array([cluster_coord[node_id] for node_id in self._node_ids])
        mean_latitude = float(coordinates[:, 1].mean())
        self._mx = 111_320.0 * np.cos(np.radians(mean_latitude))
        self._my = 110_540.0
        projected = np.column_stack([coordinates[:, 0] * self._mx, coordinates[:, 1] * self._my])
        self._tree = cKDTree(projected)

    def snap(self, latitude: float, longitude: float) -> tuple[int, float]:
        """Returns (node_id, distance_km)."""
        query_point = np.array([longitude * self._mx, latitude * self._my])
        distance_m, index = self._tree.query(query_point)
        return self._node_ids[int(index)], float(distance_m) / 1000.0


class EdgeSpatialIndex:
    """Nearest-neighbor snapping of a driver-reported GPS fix onto the closest
    road segment (edge), via each edge's midpoint coordinate."""

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


# ---------------------------------------------------------------------------
# Startup: load topology, build indices
# ---------------------------------------------------------------------------
logger.info("Loading network topology from %s", GRAPH_PATH)
live_topology, main_nodes, cluster_coord = load_topology()
initialize_baseline_impedance(live_topology)
logger.info(
    "Topology ready: %d nodes, %d edges, spoilage_cost impedance initialized",
    live_topology.number_of_nodes(),
    live_topology.number_of_edges(),
)

node_index = NodeSpatialIndex(live_topology, cluster_coord)
edge_index = EdgeSpatialIndex(live_topology, cluster_coord)

AVAILABLE_TOWNS = sorted(ncr.TOWNS.keys())

app = FastAPI(title="Northern Cape Fleet Telemetry and Routing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Route computation helpers
# ---------------------------------------------------------------------------
def resolve_town_node(town_name: str) -> int:
    if town_name not in ncr.TOWNS:
        raise HTTPException(status_code=422, detail=f"Unknown town '{town_name}'. Options: {AVAILABLE_TOWNS}")
    lon, lat = ncr.TOWNS[town_name]
    node_id, _snap_km = node_index.snap(lat, lon)
    return node_id


def build_route(origin_node: int, destination_node: int, weight_fn) -> dict:
    """Runs shortest-path with the given weight function and returns full
    per-segment detail (coordinates + road name, for hover tooltips) plus
    aggregate time/spoilage/business-value figures."""
    try:
        path = nx.shortest_path(live_topology, origin_node, destination_node, weight=weight_fn)
    except nx.NetworkXNoPath as exc:
        raise HTTPException(status_code=422, detail=f"No route found: {exc}") from exc
    except nx.NodeNotFound as exc:
        raise HTTPException(status_code=422, detail=f"Node not found in topology: {exc}") from exc

    segments = []
    total_time_mins = 0.0
    total_spoilage_cost = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_attrs = live_topology[u][v]
        seg_time = effective_time_mins(u, v, edge_attrs)
        seg_spoilage = effective_spoilage(u, v, edge_attrs)
        total_time_mins += seg_time
        total_spoilage_cost += seg_spoilage

        road_name = edge_attrs.get("road_name") or edge_attrs.get("road_ref")
        if not road_name:
            fclass_label = str(edge_attrs.get("fclass", "road")).replace("_", " ")
            road_name = f"Unnamed {fclass_label} road"
        elif edge_attrs.get("road_ref") and edge_attrs.get("road_name") and \
                edge_attrs["road_ref"] != edge_attrs["road_name"]:
            road_name = f"{edge_attrs['road_name']} ({edge_attrs['road_ref']})"

        segments.append({
            "coords": [list(cluster_coord[u]), list(cluster_coord[v])],
            "road_name": road_name,
            "fclass": edge_attrs.get("fclass"),
            "overridden": edge_attrs.get("override") is not None,
        })

    spoilage_pct = min(1.0, total_spoilage_cost / SPOILAGE_THRESHOLD)
    expected_loss_rand = round(spoilage_pct * SHIPMENT_VALUE_RAND, -2)

    return {
        "path_nodes": path,
        "coordinates": [list(cluster_coord[n]) for n in path],
        "segments": segments,
        "total_time_mins": round(total_time_mins, 1),
        "total_spoilage_cost": round(total_spoilage_cost, 3),
        "spoilage_risk_pct": round(spoilage_pct * 100, 1),
        "expected_loss_rand": expected_loss_rand,
    }


def build_cumulative_time_table(path_nodes: list[int]) -> tuple[list[tuple], float]:
    """Precomputes (cumulative_hours, lon, lat) at every node along a path,
    used to place the simulated truck at any elapsed-time fraction of the
    journey by simple interpolation instead of discrete jumps."""
    table = [(0.0, *cluster_coord[path_nodes[0]])]
    cum_hours = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge_attrs = live_topology[u][v]
        cum_hours += effective_time_mins(u, v, edge_attrs) / 60.0
        table.append((cum_hours, *cluster_coord[v]))
    return table, cum_hours


def interpolate_position(cumulative_table: list[tuple], elapsed_hours: float) -> tuple[float, float]:
    if elapsed_hours <= cumulative_table[0][0]:
        _, lon, lat = cumulative_table[0]
        return lon, lat
    if elapsed_hours >= cumulative_table[-1][0]:
        _, lon, lat = cumulative_table[-1]
        return lon, lat
    for i in range(1, len(cumulative_table)):
        t_hi, lon_hi, lat_hi = cumulative_table[i]
        if elapsed_hours <= t_hi:
            t_lo, lon_lo, lat_lo = cumulative_table[i - 1]
            span = t_hi - t_lo
            frac = (elapsed_hours - t_lo) / span if span > 0 else 0.0
            return lon_lo + (lon_hi - lon_lo) * frac, lat_lo + (lat_hi - lat_lo) * frac
    _, lon, lat = cumulative_table[-1]
    return lon, lat


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check() -> dict:
    return {"status": "online", "service": "Northern Cape Fleet Telemetry and Routing API"}


@app.get("/v1/towns")
def list_towns() -> dict:
    return {"towns": AVAILABLE_TOWNS}


@app.post("/v1/trip/configure")
def configure_trip(config: TripConfig) -> dict:
    """Sets the active destination (and, for simulator mode, the origin) and
    (re)computes both routes from scratch. Also resets the simulation clock
    so a newly configured trip starts moving from the origin, not wherever
    the previous trip left off."""
    destination_node = resolve_town_node(config.destination_town)

    with state_lock:
        trip_state["mode"] = config.mode
        trip_state["destination_town"] = config.destination_town
        trip_state["destination_node"] = destination_node
        trip_state["arrived"] = False

        if config.mode == "simulator":
            if not config.origin_town:
                raise HTTPException(status_code=422, detail="origin_town is required in simulator mode.")
            origin_node = resolve_town_node(config.origin_town)
            trip_state["origin_town"] = config.origin_town
            trip_state["origin_node"] = origin_node

            optimized = build_route(origin_node, destination_node, spoilage_weight)
            cumulative_table, total_hours = build_cumulative_time_table(optimized["path_nodes"])
            trip_state["sim_cumulative"] = cumulative_table
            trip_state["sim_total_hours"] = total_hours
            trip_state["trip_start_ts"] = time.monotonic()

            start_lon, start_lat = cluster_coord[origin_node]
            tracking_state["vehicle_id"] = VEHICLE_ID
            tracking_state["lat"] = start_lat
            tracking_state["lon"] = start_lon
            tracking_state["feed_source"] = FEED_SOURCE_SIMULATED
            tracking_state["cargo_temp_c"] = 3.5
            tracking_state["speed_kmh"] = 0.0
        else:
            trip_state["origin_town"] = None
            trip_state["origin_node"] = None
            trip_state["sim_cumulative"] = None
            trip_state["sim_total_hours"] = 0.0
            trip_state["trip_start_ts"] = None
            tracking_state["feed_source"] = FEED_SOURCE_REAL

        trip_state["configured"] = True

    logger.info(
        "TRIP CONFIGURED -> mode=%s origin=%s destination=%s",
        config.mode, config.origin_town, config.destination_town,
    )
    return {"status": "configured", **{k: v for k, v in trip_state.items() if k != "sim_cumulative"}}


@app.api_route("/v1/telematics/incoming", methods=["GET", "POST"])
async def telematics_incoming(request: Request) -> dict:
    """Real-hardware ingestion endpoint. Point Traccar Client's server URL at
    http://10.3.11.63:8000/v1/telematics/incoming — either GET query
    params or a form-urlencoded POST body work without reconfiguring the phone."""
    params: dict[str, str] = dict(request.query_params)

    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            try:
                form = await request.form()
                for key, value in form.items():
                    params.setdefault(key, str(value))
            except Exception as exc:
                logger.warning("TRACCAR INGEST -> could not parse POST body: %s", exc)

    device_id = params.get("id")
    raw_lat = params.get("lat")
    raw_lon = params.get("lon")
    raw_speed = params.get("speed")
    raw_bearing = params.get("bearing")

    if raw_lat is None or raw_lon is None:
        raise HTTPException(
            status_code=422,
            detail=f"Missing lat/lon. Received fields: {list(params.keys())}",
        )

    try:
        lat = float(raw_lat)
        lon = float(raw_lon)
        speed = float(raw_speed) if raw_speed not in (None, "") else None
        bearing = float(raw_bearing) if raw_bearing not in (None, "") else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse numeric field: {exc}") from exc

    with state_lock:
        tracking_state["vehicle_id"] = device_id or VEHICLE_ID
        tracking_state["lat"] = lat
        tracking_state["lon"] = lon
        tracking_state["timestamp"] = int(time.time())
        tracking_state["feed_source"] = FEED_SOURCE_REAL
        tracking_state["speed_kmh"] = speed if speed is not None else tracking_state.get("speed_kmh", 0.0)
        tracking_state["bearing"] = bearing if bearing is not None else tracking_state.get("bearing", 0.0)

    logger.info("TRACCAR INGEST -> device_id=%s lat=%.5f lon=%.5f", device_id, lat, lon)
    return {"status": "received", "feed_source": FEED_SOURCE_REAL}


@app.get("/v1/telematics/truck-01")
def get_telematics() -> dict:
    """Returns the current tracking state. In simulator mode, position is
    computed fresh on every call from elapsed wall-clock time — never mutated
    in place — so repeated polling never jumps or resets."""
    with state_lock:
        if trip_state["configured"] and trip_state["mode"] == "simulator" and trip_state["sim_cumulative"]:
            elapsed_real_s = time.monotonic() - trip_state["trip_start_ts"]
            elapsed_sim_hours = (elapsed_real_s * SIM_TIME_ACCELERATION) / 3600.0
            lon, lat = interpolate_position(trip_state["sim_cumulative"], elapsed_sim_hours)
            arrived = elapsed_sim_hours >= trip_state["sim_total_hours"]
            trip_state["arrived"] = arrived

            tracking_state["vehicle_id"] = VEHICLE_ID
            tracking_state["lat"] = lat
            tracking_state["lon"] = lon
            tracking_state["timestamp"] = int(time.time())
            tracking_state["feed_source"] = FEED_SOURCE_SIMULATED
            tracking_state["speed_kmh"] = 0.0 if arrived else 80.0
            progress_pct = (
                min(1.0, elapsed_sim_hours / trip_state["sim_total_hours"])
                if trip_state["sim_total_hours"] else 1.0
            )
            tracking_state["cargo_temp_c"] = round(3.5 + 1.5 * progress_pct, 2)

        payload = dict(tracking_state)
        if trip_state["configured"] and trip_state["mode"] == "simulator" and trip_state["sim_total_hours"]:
            elapsed_real_s = time.monotonic() - trip_state["trip_start_ts"]
            elapsed_sim_hours = (elapsed_real_s * SIM_TIME_ACCELERATION) / 3600.0
            payload["trip_progress_pct"] = round(min(1.0, elapsed_sim_hours / trip_state["sim_total_hours"]) * 100, 1)
        else:
            payload["trip_progress_pct"] = None
        payload["arrived"] = trip_state["arrived"]

    return payload


@app.get("/v1/routing/truck-01")
def get_routing() -> dict:
    """Computes BOTH the standard (time-only) and the spoilage-optimized
    route from the vehicle's current position to the configured destination,
    plus the business-value delta between them. In hardware mode this
    reroutes live from the phone's current GPS fix on every call."""
    with state_lock:
        if not trip_state["configured"]:
            raise HTTPException(status_code=409, detail="No trip configured yet. Call /v1/trip/configure first.")
        mode = trip_state["mode"]
        destination_node = trip_state["destination_node"]
        current_lat = tracking_state["lat"]
        current_lon = tracking_state["lon"]
        feed_source = tracking_state["feed_source"]
        origin_node_fixed = trip_state["origin_node"]

    if current_lat is None or current_lon is None:
        raise HTTPException(status_code=409, detail="No position available yet.")

    if mode == "simulator":
        # Fixed at the configured origin so the two routes being compared are
        # for the whole trip, not from the truck's current mid-journey
        # position (which would keep shrinking as it drives).
        origin_node = origin_node_fixed
    else:
        origin_node, _snap_km = node_index.snap(current_lat, current_lon)

    optimized = build_route(origin_node, destination_node, spoilage_weight)
    standard = build_route(origin_node, destination_node, time_weight)

    time_saved_mins = round(standard["total_time_mins"] - optimized["total_time_mins"], 1)
    rand_saved = round(standard["expected_loss_rand"] - optimized["expected_loss_rand"], -2)
    detour_active = optimized["path_nodes"] != standard["path_nodes"]

    current_node, snap_distance_km = node_index.snap(current_lat, current_lon)

    return {
        "vehicle_id": VEHICLE_ID,
        "feed_source": feed_source,
        "snapped_node_id": current_node,
        "snap_distance_km": round(snap_distance_km, 3),
        "destination_node_id": destination_node,
        "destination_town": trip_state["destination_town"],
        "origin_town": trip_state["origin_town"],
        "detour_active": detour_active,
        "optimized_route": optimized,
        "standard_route": standard,
        "business_value": {
            "time_saved_mins": time_saved_mins,
            "extra_travel_time_mins": round(-time_saved_mins, 1) if time_saved_mins < 0 else 0.0,
            "rand_saved": rand_saved,
            "standard_spoilage_risk_pct": standard["spoilage_risk_pct"],
            "optimized_spoilage_risk_pct": optimized["spoilage_risk_pct"],
            "shipment_value_rand": SHIPMENT_VALUE_RAND,
        },
        # kept for compatibility with older callers expecting a single "path"
        "path": optimized["coordinates"],
    }


@app.post("/v1/reports/submit")
def submit_report(report: FieldReport) -> dict:
    """Driver/fisherman ground-truth submission. Snaps the reported fix onto
    the nearest road segment and overrides that segment's spoilage_cost and
    time, triggering an immediate reroute on the next routing call."""
    try:
        matched_u, matched_v, distance_km = edge_index.snap(report.lat, report.lon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Nearest-segment lookup failed: {exc}") from exc

    roughness_override = ROAD_CONDITION_ROUGHNESS[report.road_condition]
    length_km = live_topology[matched_u][matched_v]["length_km"]
    travel_time_override_hr = length_km / report.actual_speed
    spoilage_cost_override = travel_time_override_hr * roughness_override

    override_record = {
        "reporter_role": report.reporter_role,
        "road_condition": report.road_condition,
        "roughness": roughness_override,
        "imputed_speed_kmh": report.actual_speed,
        "base_time_mins": travel_time_override_hr * 60.0,
        "spoilage_cost": spoilage_cost_override,
        "timestamp": int(time.time()),
    }

    with state_lock:
        live_topology[matched_u][matched_v]["override"] = override_record
        active_driver_reports[(matched_u, matched_v)] = override_record

    logger.warning(
        "FIELD REPORT RECEIVED -> reporter=%s condition=%s segment=(%s -> %s) "
        "distance_from_report_km=%.3f new_speed_kmh=%.1f new_spoilage_cost=%.3f",
        report.reporter_role, report.road_condition, matched_u, matched_v,
        distance_km, report.actual_speed, spoilage_cost_override,
    )

    return {
        "status": "success",
        "matched_segment": [matched_u, matched_v],
        "distance_from_report_km": round(distance_km, 3),
        "applied_override": override_record,
        "active_report_count": len(active_driver_reports),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
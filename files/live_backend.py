"""
live_backend.py

FastAPI backend for the Northern Cape Fleet Dispatch system. Loads the pre-compiled
road network topology (nc_road_graph.pkl), ingests vehicle telemetry from TWO
interchangeable sources (a real Traccar Client mobile app feed, or a built-in
simulator that cycles a hardcoded coordinate sequence), computes spoilage-risk-
optimized routes against a fixed destination hub, and accepts crowd-sourced driver
ground-truth reports that override routing impedance in real time.

Run standalone (this is a real server, not an in-process mock):
    uv run live_backend.py

Serves on http://127.0.0.1:8000

Traccar Client (Android/iOS "Traccar Client" app) should be pointed at:
    http://<this-machine-ip>:8000/v1/telematics/incoming
using its generic "OsmAnd" / query-string protocol, which sends telemetry as
GET query parameters (id, lat, lon, timestamp, speed, bearing, ...).
"""

from __future__ import annotations

import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Literal, Optional

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
DESTINATION_NAME = "Upington"

# Port Nolloth heading inland on the N7, then east on the N14 toward Upington.
# Used only by the SIMULATOR feed. Loops continuously so a live dashboard has
# something to watch indefinitely rather than stalling after one pass.
SIMULATOR_SEQUENCE: list[tuple[float, float]] = [
    (16.8667, -29.2500),   # Port Nolloth
    (17.8865, -29.6644),   # Springbok
    (19.4000, -29.1333),   # approaching Pofadder
    (21.1500, -29.3333),   # approaching Kenhardt
    (21.2561, -28.4478),   # Upington
]

FEED_SOURCE_REAL = "REAL-TIME TRACCAR HARDWARE"
FEED_SOURCE_SIMULATED = "SIMULATED TELEMETRY MATRIX"

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


# ---------------------------------------------------------------------------
# In-memory tracking state — single source of truth for whichever feed is
# currently driving the vehicle's position. Both ingestion endpoints write
# into this same dict, so the dashboard never needs to know which feed is
# active beyond reading `feed_source`.
# ---------------------------------------------------------------------------
tracking_state: dict = {
    "vehicle_id": VEHICLE_ID,
    "lat": SIMULATOR_SEQUENCE[0][1],
    "lon": SIMULATOR_SEQUENCE[0][0],
    "timestamp": int(time.time()),
    "cargo_temp_c": 3.5,
    "feed_source": FEED_SOURCE_SIMULATED,
    "speed_kmh": 0.0,
    "bearing": 0.0,
}
simulator_step = 0
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
    """
    Derives the routing impedance fields from the raw edge attributes produced by
    nc_road_network.py (length_km, travel_time, roughness). `spoilage_cost` is the
    primary routing weight throughout this system.

    Every edge also gets an `override` slot, initialized to None. Baseline fields are
    never mutated after this point, only read; driver reports write into `override`
    instead, so the original tiered-imputation baseline is always recoverable and the
    fallback hierarchy (report first, baseline second) is a simple None check rather
    than a destructive overwrite.
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
    """
    The fallback-hierarchy pathfinder weight function:
      1. If a driver report exists for this segment, use its community-validated
         spoilage_cost.
      2. Otherwise, fall back to the data-audit tiered-imputation baseline.
    """
    override = edge_attrs.get("override")
    if override is not None:
        return override["spoilage_cost"]
    return edge_attrs["spoilage_cost"]


def baseline_weight(u: int, v: int, edge_attrs: dict) -> float:
    """Ignores any active override; used only to detect whether a report has
    actually changed the optimal route (detour detection)."""
    return edge_attrs["spoilage_cost"]


# ---------------------------------------------------------------------------
# Spatial indices
# ---------------------------------------------------------------------------
class NodeSpatialIndex:
    """Nearest-neighbor snapping of raw GPS coordinates onto the closest node
    in main_nodes / the active routable topology (G_main)."""

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
# Startup: load topology, build indices, resolve destination
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

destination_lon, destination_lat = ncr.TOWNS[DESTINATION_NAME]
DESTINATION_NODE, destination_snap_km = node_index.snap(destination_lat, destination_lon)
logger.info(
    "Destination hub resolved: %s snapped to node_id=%s (%.2f km from nominal coordinate)",
    DESTINATION_NAME, DESTINATION_NODE, destination_snap_km,
)

app = FastAPI(title="Northern Cape Fleet Telemetry and Routing API")

# CORS: the dashboard's map/JS components fetch this API directly from the browser.
# Wide open on purpose since this only ever binds for local/demo/hackathon use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check() -> dict:
    return {"status": "online", "service": "Northern Cape Fleet Telemetry and Routing API"}


@app.api_route("/v1/telematics/incoming", methods=["GET", "POST"])
async def telematics_incoming(request: Request) -> dict:
    """
    Real-hardware ingestion endpoint. Point Traccar Client's server URL at
        http://<this-machine-ip>:8000/v1/telematics/incoming
    and it will call this endpoint on every GPS fix with `id`, `lat`, `lon`,
    `timestamp`, `speed`, `bearing`.

    Traccar Client's exact wire format varies by version/configuration: some
    send everything as URL query-string parameters (works with GET or POST),
    others send a POST with the same fields as a form-urlencoded body. This
    endpoint checks the query string first, then falls back to a parsed
    form body, so either mode works without reconfiguring the phone.

    Overwrites the in-memory tracking state, marks the feed source as real
    hardware, and snaps the raw phone coordinates onto the G_main topology.
    """
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
    timestamp = params.get("timestamp")
    raw_speed = params.get("speed")
    raw_bearing = params.get("bearing")

    if raw_lat is None or raw_lon is None:
        logger.error(
            "TRACCAR INGEST -> missing lat/lon. method=%s content_type=%s received_keys=%s",
            request.method, request.headers.get("content-type"), list(params.keys()),
        )
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

    try:
        snapped_node_id, snap_distance_km = node_index.snap(lat, lon)
    except Exception as exc:
        logger.error("TRACCAR INGEST -> snapping failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Snapping failed: {exc}") from exc

    with state_lock:
        tracking_state["vehicle_id"] = device_id or VEHICLE_ID
        tracking_state["lat"] = lat
        tracking_state["lon"] = lon
        tracking_state["timestamp"] = int(time.time())
        tracking_state["feed_source"] = FEED_SOURCE_REAL
        tracking_state["speed_kmh"] = speed if speed is not None else tracking_state.get("speed_kmh", 0.0)
        tracking_state["bearing"] = bearing if bearing is not None else tracking_state.get("bearing", 0.0)
        # cargo temperature isn't reported by the phone hardware feed; hold the
        # last known value steady rather than fabricating a new one.

    logger.info(
        "TRACCAR INGEST -> device_id=%s lat=%.5f lon=%.5f speed=%s bearing=%s "
        "snapped_node_id=%s snap_distance_km=%.3f",
        device_id, lat, lon, speed, bearing, snapped_node_id, snap_distance_km,
    )
    return {
        "status": "received",
        "feed_source": FEED_SOURCE_REAL,
        "snapped_node_id": snapped_node_id,
        "snap_distance_km": round(snap_distance_km, 3),
    }


@app.post("/v1/telematics/simulate-step")
def telematics_simulate_step() -> dict:
    """
    Fallback demo feed. Advances one step through a hardcoded 5-point sequence
    along a real Northern Cape transport lane (Port Nolloth -> Upington),
    looping continuously. Called by the dashboard on every polling tick when
    Simulator Mode is active.
    """
    global simulator_step
    with state_lock:
        sequence_length = len(SIMULATOR_SEQUENCE)
        lap_position = simulator_step % sequence_length
        longitude, latitude = SIMULATOR_SEQUENCE[lap_position]

        tracking_state["vehicle_id"] = VEHICLE_ID
        tracking_state["lat"] = latitude
        tracking_state["lon"] = longitude
        tracking_state["timestamp"] = int(time.time())
        tracking_state["cargo_temp_c"] = round(3.5 + 0.35 * lap_position, 2)
        tracking_state["feed_source"] = FEED_SOURCE_SIMULATED
        tracking_state["speed_kmh"] = 80.0
        tracking_state["bearing"] = 0.0
        simulator_step += 1

        payload = dict(tracking_state)
        payload["step"] = simulator_step

    logger.info(
        "SIMULATOR STEP -> lap_position=%d lat=%.4f lon=%.4f cargo_temp_c=%.2f",
        lap_position, latitude, latitude, payload["cargo_temp_c"],
    )
    return payload


@app.get("/v1/telematics/truck-01")
def get_telematics() -> dict:
    """Returns the unified active tracking state, regardless of which feed
    (real hardware or simulator) most recently wrote to it."""
    with state_lock:
        payload = dict(tracking_state)

    logger.info(
        "TELEMATICS -> vehicle=%s lat=%.4f lon=%.4f feed_source=%s",
        payload["vehicle_id"], payload["lat"], payload["lon"], payload["feed_source"],
    )
    return payload


@app.get("/v1/routing/truck-01")
def get_routing() -> dict:
    """Dynamically reads whichever location is currently active in memory
    (real or simulated) as the moving origin, and computes the optimal
    spoilage_cost path to the fixed destination hub."""
    with state_lock:
        current_lat = tracking_state["lat"]
        current_lon = tracking_state["lon"]
        feed_source = tracking_state["feed_source"]

    try:
        current_node, snap_distance_km = node_index.snap(current_lat, current_lon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Snapping failed: {exc}") from exc

    try:
        live_path = nx.shortest_path(
            live_topology, current_node, DESTINATION_NODE, weight=effective_weight
        )
        baseline_path = nx.shortest_path(
            live_topology, current_node, DESTINATION_NODE, weight=baseline_weight
        )
    except nx.NetworkXNoPath as exc:
        raise HTTPException(
            status_code=422,
            detail=f"No route between node {current_node} and destination node {DESTINATION_NODE}: {exc}",
        ) from exc
    except nx.NodeNotFound as exc:
        raise HTTPException(status_code=422, detail=f"Node not found in topology: {exc}") from exc

    total_time_mins = 0.0
    total_spoilage_cost = 0.0
    for u, v in zip(live_path[:-1], live_path[1:]):
        edge_attrs = live_topology[u][v]
        weight = effective_weight(u, v, edge_attrs)
        total_spoilage_cost += weight
        override = edge_attrs.get("override")
        total_time_mins += (
            override["base_time_mins"] if override is not None else edge_attrs["base_time_mins"]
        )

    detour_active = live_path != baseline_path
    coordinates = [[cluster_coord[node_id][0], cluster_coord[node_id][1]] for node_id in live_path]

    logger.info(
        "ROUTING -> feed_source=%s current_node=%s hops=%d total_time_mins=%.2f "
        "total_spoilage_cost=%.3f detour_active=%s",
        feed_source, current_node, len(live_path) - 1, total_time_mins,
        total_spoilage_cost, detour_active,
    )

    return {
        "vehicle_id": VEHICLE_ID,
        "feed_source": feed_source,
        "snapped_node_id": current_node,
        "snap_distance_km": round(snap_distance_km, 3),
        "destination_node_id": DESTINATION_NODE,
        "path": coordinates,
        "hop_count": len(live_path) - 1,
        "total_time_mins": round(total_time_mins, 2),
        "total_spoilage_cost": round(total_spoilage_cost, 3),
        "detour_active": detour_active,
    }


@app.post("/v1/reports/submit")
def submit_report(report: FieldReport) -> dict:
    """Driver/fisherman ground-truth submission. Snaps the reported fix onto
    the nearest road segment and overrides that segment's spoilage_cost,
    triggering an immediate reroute on the next /v1/routing/truck-01 call."""
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
        "distance_from_report_km=%.3f new_speed_kmh=%.1f new_roughness=%.2f "
        "new_spoilage_cost=%.3f",
        report.reporter_role, report.road_condition, matched_u, matched_v,
        distance_km, report.actual_speed, roughness_override, spoilage_cost_override,
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
"""
live_backend.py

FastAPI backend for the Northern Cape Fleet Dispatch system. Loads the pre-compiled
road network topology (nc_road_graph.pkl), streams simulated live vehicle telemetry,
computes spoilage-risk-optimized routes, and accepts crowd-sourced driver ground-truth
reports that override the routing impedance in real time.

Run standalone (this is a real server, not an in-process mock):
    uv run live_backend.py

Serves on http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import networkx as nx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree

import nc_road_network as ncr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRAPH_PATH = Path("nc_road_graph.pkl")
HOST = "127.0.0.1"
PORT = 8000

VEHICLE_ID = "TRUCK-01"
DESTINATION_NAME = "Upington"

# Port Nolloth heading inland on the N7, then east on the N14 toward Upington.
# The sequence loops continuously so a live dashboard has something to watch
# indefinitely rather than stalling after one pass.
GPS_TELEMETRY_SEQUENCE: list[tuple[float, float]] = [
    (16.8667, -29.2500),   # Port Nolloth
    (17.8865, -29.6644),   # Springbok
    (19.4000, -29.1333),   # approaching Pofadder
    (21.1500, -29.3333),   # approaching Kenhardt
    (21.2561, -28.4478),   # Upington
]

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


@dataclass
class VehicleState:
    step: int = 0
    latitude: float = GPS_TELEMETRY_SEQUENCE[0][1]
    longitude: float = GPS_TELEMETRY_SEQUENCE[0][0]
    cargo_temp_c: float = 3.5
    timestamp: int = 0


vehicle_state = VehicleState()
active_driver_reports: dict[tuple[int, int], dict] = {}


# ---------------------------------------------------------------------------
# Topology loading and impedance matrix initialization
# ---------------------------------------------------------------------------
def load_topology() -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"'{GRAPH_PATH}' not found. Run Data_Audit.ipynb first to produce it, "
            f"or copy an existing nc_road_graph.pkl into this directory."
        )
    with GRAPH_PATH.open("rb") as f:
        artifacts = pickle.load(f)
    return artifacts["G_main"], artifacts["cluster_coord"]


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
    in main_nodes / the active routable topology."""

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
live_topology, cluster_coord = load_topology()
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

# CORS: the dispatch map now polls this API directly from client-side JS running
# inside a components.html iframe (srcdoc origin), instead of via Streamlit's
# Python process. Without this, the browser blocks the fetch() calls entirely.
# Wide open on purpose since this only ever binds to 127.0.0.1 for local/demo use.
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


@app.get("/v1/telematics/truck-01")
def get_telematics() -> dict:
    with state_lock:
        sequence_length = len(GPS_TELEMETRY_SEQUENCE)
        lap_position = vehicle_state.step % sequence_length
        longitude, latitude = GPS_TELEMETRY_SEQUENCE[lap_position]

        vehicle_state.latitude = latitude
        vehicle_state.longitude = longitude
        vehicle_state.cargo_temp_c = round(3.5 + 0.35 * lap_position, 2)
        vehicle_state.timestamp = int(time.time())
        vehicle_state.step += 1

        payload = {
            "vehicle_id": VEHICLE_ID,
            "lat": vehicle_state.latitude,
            "lon": vehicle_state.longitude,
            "timestamp": vehicle_state.timestamp,
            "cargo_temp_c": vehicle_state.cargo_temp_c,
            "step": vehicle_state.step,
        }

    logger.info(
        "TELEMATICS -> vehicle=%s lat=%.4f lon=%.4f cargo_temp_c=%.2f step=%d",
        payload["vehicle_id"], payload["lat"], payload["lon"],
        payload["cargo_temp_c"], payload["step"],
    )
    return payload


@app.get("/v1/routing/truck-01")
def get_routing() -> dict:
    with state_lock:
        current_lat = vehicle_state.latitude
        current_lon = vehicle_state.longitude

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
        "ROUTING -> current_node=%s hops=%d total_time_mins=%.2f total_spoilage_cost=%.3f "
        "detour_active=%s",
        current_node, len(live_path) - 1, total_time_mins, total_spoilage_cost, detour_active,
    )

    return {
        "vehicle_id": VEHICLE_ID,
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
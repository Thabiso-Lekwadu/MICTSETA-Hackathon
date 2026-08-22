"""
live_pipeline.py

Dynamic moving-node tracking and real-time route optimization pipeline for the
Northern Cape Transport, Trade & Fisheries freight corridor (Port Nolloth -> Upington,
N7/N14 corridor).

Simulates a vehicle telematics feed over a mocked secure HTTPS REST API
(https://fleet-nc.co.za), snaps each live GPS fix onto the pre-compiled road network
topology (nc_road_graph.pkl), and continuously reassesses the lowest-spoilage-risk path
to destination. On a simulated incident event, the affected road segments' impedance is
penalized in place and the route is dynamically recomputed from the vehicle's current
snapped position.

Run:
    uv run live_pipeline.py
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*httpx.*starlette.testclient.*")

import networkx as nx
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy.spatial import cKDTree

logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRAPH_PATH = Path("nc_road_graph.pkl")

API_BASE_URL = "https://fleet-nc.co.za"
TELEMETRY_ENDPOINT = "/v1/telemetry/live"
VEHICLE_ID = "TRUCK-01"

POLL_INTERVAL_SECONDS = 2.0
INCIDENT_STEP = 3
INCIDENT_DELAY_MINS = 180.0

# Origin -> destination corridor: Port Nolloth heading inland on the N7, then east on the
# N14 toward Upington. Waypoints are approximate real-world coordinates for the corridor,
# not precise road-snapped geometry -- the snapping layer below handles that.
GPS_TELEMETRY_SEQUENCE: list[tuple[float, float]] = [
    (16.8667, -29.2500),   # Port Nolloth
    (17.8865, -29.6644),   # Springbok
    (19.4000, -29.1333),   # approaching Pofadder
    (21.1500, -29.3333),   # approaching Kenhardt
    (21.2561, -28.4478),   # Upington
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("live_pipeline")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TelemetryReading:
    vehicle_id: str
    latitude: float
    longitude: float
    timestamp: int
    cargo_temp_c: float


@dataclass(frozen=True)
class RouteAssessment:
    step: int
    snapped_node_id: int
    snap_distance_km: float
    hop_count: int
    total_time_mins: float
    total_spoilage_cost: float
    path_nodes: tuple[int, ...]


# ---------------------------------------------------------------------------
# Mock secure telemetry API
#
# Implemented as a real FastAPI application, exercised through Starlette's
# TestClient bound to the target base URL. This executes the full ASGI request/
# response cycle in-process -- no socket, no port binding, no external DNS -- which
# makes it bulletproof to run in any environment (CI runners, sandboxes, laptops
# behind a firewall) while still being a genuine HTTP interface, not a stub function.
# ---------------------------------------------------------------------------
class TelemetrySimulator:
    """Holds the moving-vehicle state and serves the next GPS fix on each request."""

    def __init__(self, coordinate_sequence: list[tuple[float, float]], vehicle_id: str):
        self._sequence = coordinate_sequence
        self._vehicle_id = vehicle_id
        self._cursor = 0
        self._base_cargo_temp_c = 3.5

    def next_reading(self) -> TelemetryReading:
        if self._cursor >= len(self._sequence):
            self._cursor = len(self._sequence) - 1  # hold at final fix
        longitude, latitude = self._sequence[self._cursor]
        # cargo warms slightly with each elapsed telemetry step, modelling cumulative
        # refrigeration strain over the trip
        cargo_temp_c = round(self._base_cargo_temp_c + 0.35 * self._cursor, 2)
        reading = TelemetryReading(
            vehicle_id=self._vehicle_id,
            latitude=latitude,
            longitude=longitude,
            timestamp=int(time.time()),
            cargo_temp_c=cargo_temp_c,
        )
        self._cursor += 1
        return reading


telemetry_simulator = TelemetrySimulator(GPS_TELEMETRY_SEQUENCE, VEHICLE_ID)
mock_api = FastAPI(title="Northern Cape Fleet Telemetry API")


@mock_api.get(TELEMETRY_ENDPOINT)
def get_live_telemetry() -> dict:
    reading = telemetry_simulator.next_reading()
    return {
        "vehicle_id": reading.vehicle_id,
        "lat": reading.latitude,
        "lon": reading.longitude,
        "timestamp": reading.timestamp,
        "cargo_temp_c": reading.cargo_temp_c,
    }


api_client = TestClient(mock_api, base_url=API_BASE_URL)


def fetch_live_telemetry() -> TelemetryReading:
    try:
        response = api_client.get(TELEMETRY_ENDPOINT)
        response.raise_for_status()
    except Exception as exc:  # network/transport failure against the mock HTTPS endpoint
        raise RuntimeError(f"Telemetry API request failed: {exc}") from exc

    payload = response.json()
    try:
        return TelemetryReading(
            vehicle_id=str(payload["vehicle_id"]),
            latitude=float(payload["lat"]),
            longitude=float(payload["lon"]),
            timestamp=int(payload["timestamp"]),
            cargo_temp_c=float(payload["cargo_temp_c"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed telemetry payload: {payload}") from exc


# ---------------------------------------------------------------------------
# Topology loading + impedance matrix initialization
# ---------------------------------------------------------------------------
def load_topology(graph_path: Path) -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
    if not graph_path.exists():
        raise FileNotFoundError(
            f"'{graph_path}' not found. Run Data_Audit.ipynb first to produce it, "
            f"or copy an existing nc_road_graph.pkl into this directory."
        )
    with graph_path.open("rb") as f:
        artifacts = pickle.load(f)

    try:
        main_graph: nx.Graph = artifacts["G_main"]
        cluster_coord: dict[int, tuple[float, float]] = artifacts["cluster_coord"]
    except KeyError as exc:
        raise RuntimeError(
            f"'{graph_path}' is missing an expected key: {exc}. "
            f"Expected keys: G_main, cluster_coord."
        ) from exc

    return main_graph, cluster_coord


def initialize_impedance_matrix(topology: nx.Graph) -> None:
    """
    Derives the two impedance fields this pipeline optimizes over, from the raw
    edge attributes produced by nc_road_network.py (length_km, travel_time, roughness):

        base_time_mins  = travel_time (hr) * 60        -- raw driving time
        spoilage_cost    = travel_time (hr) * roughness -- vibration/damage-risk-weighted
                                                            cold-chain exposure

    Mutates the topology's edge attribute dict in place so every downstream
    shortest-path call can weight on either field directly.
    """
    for _u, _v, edge_attrs in topology.edges(data=True):
        travel_time_hr = edge_attrs["travel_time"]
        roughness = edge_attrs.get("roughness", 1.3)
        edge_attrs["base_time_mins"] = travel_time_hr * 60.0
        edge_attrs["spoilage_cost"] = travel_time_hr * roughness


# ---------------------------------------------------------------------------
# Nearest-neighbor snapping layer
# ---------------------------------------------------------------------------
class SpatialSnapIndex:
    """KD-tree spatial index over every node in the active routable topology,
    projected to an approximate local metric plane for correct nearest-neighbor
    ordering (equirectangular scaling around the network's mean latitude)."""

    def __init__(self, topology: nx.Graph, cluster_coord: dict[int, tuple[float, float]]):
        self._node_ids: list[int] = list(topology.nodes())
        coordinates = np.array([cluster_coord[node_id] for node_id in self._node_ids])
        mean_latitude = float(coordinates[:, 1].mean())

        self._meters_per_deg_lon = 111_320.0 * np.cos(np.radians(mean_latitude))
        self._meters_per_deg_lat = 110_540.0

        projected_points = np.column_stack([
            coordinates[:, 0] * self._meters_per_deg_lon,
            coordinates[:, 1] * self._meters_per_deg_lat,
        ])
        self._tree = cKDTree(projected_points)

    def snap(self, latitude: float, longitude: float) -> tuple[int, float]:
        """Returns (nearest_node_id, snap_distance_km)."""
        query_point = np.array([
            longitude * self._meters_per_deg_lon,
            latitude * self._meters_per_deg_lat,
        ])
        distance_m, index = self._tree.query(query_point)
        nearest_node_id = self._node_ids[int(index)]
        return nearest_node_id, float(distance_m) / 1000.0


# ---------------------------------------------------------------------------
# Route assessment + dynamic rerouting
# ---------------------------------------------------------------------------
def assess_route(
    topology: nx.Graph,
    origin_node: int,
    destination_node: int,
    weight_field: str = "spoilage_cost",
) -> RouteAssessment:
    try:
        path_nodes = nx.shortest_path(topology, origin_node, destination_node, weight=weight_field)
    except nx.NetworkXNoPath as exc:
        raise RuntimeError(
            f"No route exists between node {origin_node} and node {destination_node} "
            f"in the current topology."
        ) from exc
    except nx.NodeNotFound as exc:
        raise RuntimeError(f"Node not present in topology: {exc}") from exc

    total_time_mins = 0.0
    total_spoilage_cost = 0.0
    for upstream_node, downstream_node in zip(path_nodes[:-1], path_nodes[1:]):
        segment = topology[upstream_node][downstream_node]
        total_time_mins += segment["base_time_mins"]
        total_spoilage_cost += segment["spoilage_cost"]

    return RouteAssessment(
        step=-1,  # filled in by the caller
        snapped_node_id=origin_node,
        snap_distance_km=0.0,  # filled in by the caller
        hop_count=len(path_nodes) - 1,
        total_time_mins=total_time_mins,
        total_spoilage_cost=total_spoilage_cost,
        path_nodes=tuple(path_nodes),
    )


def apply_incident_impedance(
    topology: nx.Graph,
    upstream_node: int,
    downstream_node: int,
    delay_mins: float,
) -> None:
    """
    Applies a critical-incident impedance penalty directly to the road segment
    immediately ahead of the vehicle's current position. Increases both the raw
    driving-time impedance and the spoilage-risk impedance -- a stalled/idling
    vehicle on a blocked segment accumulates cold-chain risk even though it isn't
    moving, so the penalty is not purely a time cost.
    """
    if not topology.has_edge(upstream_node, downstream_node):
        raise RuntimeError(
            f"Cannot apply incident: no direct segment between node {upstream_node} "
            f"and node {downstream_node} in the current topology."
        )
    segment = topology[upstream_node][downstream_node]
    segment["base_time_mins"] += delay_mins
    segment["spoilage_cost"] += (delay_mins / 60.0) * segment.get("roughness", 1.3)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
def run_pipeline() -> None:
    logger.info("=" * 78)
    logger.info("NORTHERN CAPE LIVE FREIGHT TRACKING PIPELINE -- INITIALIZING")
    logger.info("=" * 78)

    try:
        live_topology, cluster_coord = load_topology(GRAPH_PATH)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Topology load failed: %s", exc)
        sys.exit(1)

    initialize_impedance_matrix(live_topology)
    logger.info(
        "Topology loaded: %d nodes, %d edges | impedance fields initialized "
        "(base_time_mins, spoilage_cost)",
        live_topology.number_of_nodes(),
        live_topology.number_of_edges(),
    )

    snap_index = SpatialSnapIndex(live_topology, cluster_coord)

    destination_longitude, destination_latitude = GPS_TELEMETRY_SEQUENCE[-1]
    destination_node, destination_snap_km = snap_index.snap(destination_latitude, destination_longitude)
    logger.info(
        "Destination snapped -> node_id=%s (%.2f km from nominal coordinate)",
        destination_node,
        destination_snap_km,
    )
    logger.info("-" * 78)

    previous_assessment: Optional[RouteAssessment] = None

    for step in range(1, len(GPS_TELEMETRY_SEQUENCE) + 1):
        try:
            reading = fetch_live_telemetry()
        except RuntimeError as exc:
            logger.error("[STEP %d] TELEMETRY FETCH FAILED -> %s", step, exc)
            continue

        logger.info(
            "[STEP %d] RAW TELEMETRY    -> vehicle=%s lat=%.4f lon=%.4f "
            "cargo_temp_c=%.2f ts=%d",
            step, reading.vehicle_id, reading.latitude, reading.longitude,
            reading.cargo_temp_c, reading.timestamp,
        )

        try:
            snapped_node_id, snap_distance_km = snap_index.snap(reading.latitude, reading.longitude)
        except Exception as exc:
            logger.error("[STEP %d] SNAPPING FAILED -> %s", step, exc)
            continue

        logger.info(
            "[STEP %d] SNAPPED NODE     -> node_id=%s snap_distance_km=%.3f",
            step, snapped_node_id, snap_distance_km,
        )

        try:
            assessment = assess_route(live_topology, snapped_node_id, destination_node)
        except RuntimeError as exc:
            logger.error("[STEP %d] PATH ASSESSMENT FAILED -> %s", step, exc)
            continue

        logger.info(
            "[STEP %d] PATH ASSESSMENT  -> hops=%d total_time_mins=%.2f "
            "total_spoilage_cost=%.3f",
            step, assessment.hop_count, assessment.total_time_mins, assessment.total_spoilage_cost,
        )

        if step == INCIDENT_STEP:
            if assessment.hop_count == 0:
                logger.warning(
                    "[STEP %d] INCIDENT TRIGGER -> vehicle already at destination node; "
                    "skipping incident injection",
                    step,
                )
            else:
                incident_upstream = assessment.path_nodes[0]
                incident_downstream = assessment.path_nodes[1]
                logger.warning(
                    "[STEP %d] INCIDENT TRIGGER -> critical breakdown detected on segment "
                    "(%s -> %s) directly ahead; applying +%.2f min impedance penalty",
                    step, incident_upstream, incident_downstream, INCIDENT_DELAY_MINS,
                )

                try:
                    apply_incident_impedance(
                        live_topology, incident_upstream, incident_downstream, INCIDENT_DELAY_MINS
                    )
                except RuntimeError as exc:
                    logger.error("[STEP %d] INCIDENT APPLICATION FAILED -> %s", step, exc)
                else:
                    pre_incident_cost = assessment.total_spoilage_cost
                    pre_incident_time = assessment.total_time_mins

                    try:
                        rerouted_assessment = assess_route(
                            live_topology, snapped_node_id, destination_node
                        )
                    except RuntimeError as exc:
                        logger.error("[STEP %d] REROUTE FAILED -> %s", step, exc)
                    else:
                        path_changed = rerouted_assessment.path_nodes != assessment.path_nodes
                        logger.info(
                            "[STEP %d] REROUTE EXECUTED -> origin=node_id:%s "
                            "prev_time_mins=%.2f new_time_mins=%.2f "
                            "prev_spoilage_cost=%.3f new_spoilage_cost=%.3f path_changed=%s",
                            step, snapped_node_id, pre_incident_time,
                            rerouted_assessment.total_time_mins, pre_incident_cost,
                            rerouted_assessment.total_spoilage_cost, path_changed,
                        )
                        assessment = rerouted_assessment

        previous_assessment = assessment
        logger.info("-" * 78)
        time.sleep(POLL_INTERVAL_SECONDS)

    if previous_assessment is not None:
        logger.info("=" * 78)
        logger.info(
            "PIPELINE COMPLETE -> final_snapped_node=%s final_total_time_mins=%.2f "
            "final_total_spoilage_cost=%.3f",
            previous_assessment.snapped_node_id,
            previous_assessment.total_time_mins,
            previous_assessment.total_spoilage_cost,
        )
        logger.info("=" * 78)
    else:
        logger.error("PIPELINE COMPLETE -> no successful route assessments were produced")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
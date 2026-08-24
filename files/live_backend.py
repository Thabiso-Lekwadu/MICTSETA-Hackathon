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
import re
import threading
import time
from datetime import datetime
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
import weather_engine

# transformers is an optional dependency: the rest of the dispatch/routing
# system must keep working (telemetry ingest, driver reports, live map) even
# on a machine that never installed it. The /v1/analytics/strategy endpoint
# is the only thing that needs it, and it degrades to a clean JSON error
# instead of crashing the whole API process. The pipeline itself is loaded
# lazily (see get_ai_strategy_pipeline) so importing the package here never
# pays the multi-second/multi-GB model-load cost at server startup.
try:
    from transformers import pipeline as hf_pipeline
except ImportError:  # pragma: no cover - exercised only when the package is absent
    hf_pipeline = None

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

# --- Thermal spoilage model -------------------------------------------------
# Cold-chain spoilage has two genuinely different physical causes, and they're
# tracked separately rather than folded into one number:
#   1. Mechanical risk (above): road roughness x time -> vibration/seal damage.
#   2. Thermal risk (below): temperature exposure over time -> bacterial growth.
# Rock lobster / chilled seafood's safe carry range is documented, not fitted.
SAFE_TEMP_MAX_C_DEFAULT = 4.0    # default only — the live value is adjustable at runtime, see live_settings below
THERMAL_Q10 = 2.0                # food-science rule of thumb: spoilage rate ~doubles per 10C above threshold
THERMAL_SPOILAGE_THRESHOLD_HOURS = 24.0  # risk-hours budget before treated as a total thermal loss
MECHANICAL_RISK_WEIGHT = 0.5     # how road-damage risk and thermal risk are blended into one live figure
THERMAL_RISK_WEIGHT = 0.5

# --- Ambient-weather-driven idling warm-up model ----------------------------
# Live ambient temperature (via weather_engine.py / Open-Meteo) scales how
# fast an IDLING reefer chamber warms up — this models what happens while a
# real hardware feed has no dedicated cargo-interior sensor of its own yet.
# Deliberately scoped to HARDWARE mode only (see get_telematics): simulator
# mode already has its own tested, deterministic temperature curve
# (simulated_cargo_temp_c) and is left completely untouched by this.
EXTREME_HEAT_THRESHOLD_C = 38.0
COLD_CLIMATE_THRESHOLD_C = 20.0
IDLE_WARMING_TICK_EXTREME_C = 0.4    # °C per tracking tick, ambient >= 38.0°C
IDLE_WARMING_TICK_STANDARD_C = 0.1   # °C per tracking tick, 20.0°C <= ambient < 38.0°C
IDLE_WARMING_TICK_COLD_C = 0.02      # °C per tracking tick, ambient < 20.0°C
# Safety clamp: this is a per-tick idling drift model, not a physical
# simulation with its own equilibrium point — without a cap, a long idle
# period at a fast poll rate would run away well past any plausible trailer-
# interior temperature.
IDLE_WARMING_MAX_CARGO_TEMP_C = 60.0

# Live ambient weather at the vehicle's current position, refreshed on every
# /v1/telematics/truck-01 poll (see get_telematics) and read by the AI
# Strategy Layer and the /v1/routing/weather-profile endpoint. Plain dict
# read/write (no lock) for the same reason as live_settings above: a handful
# of GIL-atomic scalar fields, briefly reading mid-update has no consequence.
CURRENT_WEATHER: dict = {"temp_c": 25.0, "rain_mm": 0.0, "alert": "Normal"}

# Runtime-adjustable business thresholds. Cooperative dispatchers know their
# own cargo (species, packaging, ice quality) better than any hardcoded
# constant can — this is read by thermal_rate_multiplier/cargo_temp_status
# below and updated live via POST /v1/settings/thresholds, no restart needed.
# Plain dict read/write is fine here (no lock): it's a single float, GIL-atomic,
# and briefly reading a value mid-update has no meaningful consequence.
live_settings: dict = {
    "safe_temp_max_c": SAFE_TEMP_MAX_C_DEFAULT,
}

# Documented, not fitted, roughness assumptions per reported road condition.
# Matches the roughness-multiplier scale established in nc_road_network.py.
# The last three are storm-specific: active infrastructure failure during
# severe weather, distinct from routine wear — all pinned to IRI 6.0+ so a
# storm report always forces an immediate detour on the next reroute, same
# mechanism as any other driver-reported override.
ROAD_CONDITION_ROUGHNESS: dict[str, float] = {
    "Smooth Tarmac": 1.0,
    "Corrugated / Rough Gravel": 1.8,
    "Severe Potholes": 2.6,
    "Impassable / Washed Out": 6.0,
    "Flash Flood Mud Trap": 6.5,
    "Gravel Bed Erosion": 6.0,
    "Structural Road Washout": 7.5,
}

# ---------------------------------------------------------------------------
# AI Cognitive Strategy Layer — local Hugging Face model advisory config
# ---------------------------------------------------------------------------
# Small instruction-tuned model chosen specifically so it runs comfortably
# CPU-only on a laptop with no separate daemon to install/start (unlike
# Ollama) — `transformers` + this one line is the whole dependency.
HF_STRATEGY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

AI_STRATEGY_SYSTEM_PROMPT = (
    "You are an expert Logistics Econometrics Analyst and Cold-Chain Risk "
    "Specialist advising a commercial fishing cooperative in the Northern "
    "Cape, South Africa. Your goal is to convert raw routing and cargo-risk "
    "KPIs into sharp, high-value financial and operational business strategies.\n\n"
    "Ground every recommendation in the specific numbers and town names you are "
    "given — never speak in vague generalities, and never substitute a different "
    "origin, destination, or date than the ones stated in the request. If you are "
    "not given a figure or a name, do not invent one.\n\n"
    "This shipment's spoilage risk has two independently tracked physical causes, "
    "and your analysis must explicitly reason about both:\n"
    "  1. Mechanical risk: road-roughness x time damage from corrugated/unpaved "
    "     desert tracks versus paved corridors.\n"
    "  2. Thermal risk: temperature-exposure damage from the reefer unit "
    "     running above the safe carry threshold.\n\n"
    "If live ambient weather figures are provided in the request, factor them "
    "briefly into your thermal-risk discussion — e.g. extreme heat straining "
    "the reefer condenser, or heavy rain raising washout risk on unpaved "
    "segments. If no weather figures are provided, do not mention weather at all.\n\n"
    "Quantify the Rand-value trade-off between any extra travel time and the "
    "spoilage risk avoided by the optimized route.\n\n"
    "Do NOT write a title, a 'Date:' line, or a 'Route:' line — that header is "
    "generated separately and prepended for you. Begin your response directly "
    "with '## Introduction'. Structure the rest as concise markdown with clear "
    "section headers, and end with a short, prioritized action list the dispatch "
    "cooperative can act on today."
)


class StrategyRequest(BaseModel):
    origin: str
    destination: str
    standard_time_mins: float = Field(..., ge=0.0)
    optimized_time_mins: float = Field(..., ge=0.0)
    rand_saved: float
    mechanical_risk_reduction_pct: float
    thermal_risk_pct: float = Field(..., ge=0.0, le=100.0)
    cargo_temp_status: str
    surface_profile: str
    shipment_value_rand: float = Field(..., ge=0.0)
    # Optional and backward-compatible: any existing caller that doesn't
    # send these still works exactly as before (weather section just gets
    # omitted from the prompt — see generate_ai_strategy).
    ambient_temp_c: float | None = None
    rain_mm: float | None = None
    weather_alert: str | None = None


# Lazily loaded, then cached for the life of the process — loading the model
# takes real time (and a few GB of RAM/VRAM), so it must not happen on every
# request or block server startup. Guarded by a lock so two requests arriving
# before the first load finishes don't each try to load their own copy.
_ai_strategy_pipeline = None
_ai_strategy_pipeline_lock = threading.Lock()


def get_ai_strategy_pipeline():
    """Returns the cached text-generation pipeline, loading it on first use."""
    global _ai_strategy_pipeline
    if _ai_strategy_pipeline is not None:
        return _ai_strategy_pipeline
    with _ai_strategy_pipeline_lock:
        if _ai_strategy_pipeline is None:
            logger.info("AI STRATEGY -> loading local model '%s' (first call only)...", HF_STRATEGY_MODEL)
            _ai_strategy_pipeline = hf_pipeline("text-generation", model=HF_STRATEGY_MODEL)
            logger.info("AI STRATEGY -> model loaded and cached for subsequent calls.")
    return _ai_strategy_pipeline


def extract_strategy_text(pipeline_output) -> str:
    """The transformers text-generation pipeline, given chat-style `messages`
    input, returns generated_text as the full conversation (list of role/content
    dicts) rather than a single string in modern versions. Handles both that
    shape and the plain-string shape older versions return."""
    try:
        generated = pipeline_output[0]["generated_text"]
    except (IndexError, KeyError, TypeError):
        return ""

    if isinstance(generated, str):
        return generated.strip()

    if isinstance(generated, list):
        for turn in reversed(generated):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                return (turn.get("content") or "").strip()

    return ""


# A 1.5B model doesn't always follow "don't write a header" instructions
# reliably, and when it does write one it can hallucinate the wrong town
# names or an "[Insert Date]" placeholder. The real header is always built
# from the actual request data in code (see generate_ai_strategy) — this
# strips a handful of leading lines that look like a model-written title,
# byline, or "Date:"/"Route:" line so the two headers can't collide/duplicate.
_LEADING_HEADER_LINE_RE = re.compile(
    r"^\s*(#{1,3}\s*.*|(\*\*)?(date|route|to|memo|subject)(\*\*)?\s*:.*)\s*$",
    re.IGNORECASE,
)


def strip_hallucinated_header(text: str) -> str:
    lines = text.splitlines()
    idx = 0
    while idx < len(lines) and (not lines[idx].strip() or _LEADING_HEADER_LINE_RE.match(lines[idx])):
        idx += 1
    return "\n".join(lines[idx:]).strip()


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
        "Smooth Tarmac", "Corrugated / Rough Gravel", "Severe Potholes", "Impassable / Washed Out",
        "Flash Flood Mud Trap", "Gravel Bed Erosion", "Structural Road Washout",
    ]
    actual_speed: float = Field(..., gt=0.0, le=160.0)


class TripConfig(BaseModel):
    mode: Literal["simulator", "hardware"]
    destination_town: str
    origin_town: str | None = None  # required for simulator mode only


class ThresholdSettings(BaseModel):
    # Deliberately loose bounds — this is a business-domain call (species,
    # packaging, ice quality all shift what "safe" means), not a physics
    # constant, so the API shouldn't second-guess the dispatcher's number.
    safe_temp_max_c: float = Field(..., ge=-20.0, le=25.0)


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
    "sim_cumulative": None,          # list of (cum_hours, lon, lat) along optimized path, simulator only
    "sim_spoilage_cumulative": None, # parallel list of (cum_hours, cum_spoilage_cost), simulator only
    "sim_total_hours": 0.0,
    "trip_start_ts": None,
    "arrived": False,
    "trip_plan": None,       # static full origin->destination comparison, cached at configure time
    # Thermal accumulation for HARDWARE mode only (simulator mode integrates a
    # known closed-form curve fresh each call instead — see integrate_thermal_risk_hours).
    "thermal_risk_accum_hours": 0.0,
    "last_thermal_update_ts": None,
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


def interpolate_cumulative_scalar(table: list[tuple], elapsed_hours: float) -> float:
    """Same interpolation idea as interpolate_position but for a scalar
    running total (used for spoilage cost accumulated so far along the path)."""
    if not table:
        return 0.0
    if elapsed_hours <= table[0][0]:
        return table[0][1]
    if elapsed_hours >= table[-1][0]:
        return table[-1][1]
    for i in range(1, len(table)):
        t_hi, v_hi = table[i]
        if elapsed_hours <= t_hi:
            t_lo, v_lo = table[i - 1]
            span = t_hi - t_lo
            frac = (elapsed_hours - t_lo) / span if span > 0 else 0.0
            return v_lo + (v_hi - v_lo) * frac
    return table[-1][1]


def build_cumulative_spoilage_table(path_nodes: list[int]) -> list[tuple]:
    """Parallel structure to build_cumulative_time_table: (cum_hours,
    cum_spoilage_cost) at every node, so 'mechanical risk accumulated by the
    cargo so far' can be read off by interpolation exactly like position is."""
    table = [(0.0, 0.0)]
    cum_hours = 0.0
    cum_spoilage = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge_attrs = live_topology[u][v]
        cum_hours += effective_time_mins(u, v, edge_attrs) / 60.0
        cum_spoilage += effective_spoilage(u, v, edge_attrs)
        table.append((cum_hours, cum_spoilage))
    return table


def simulated_cargo_temp_c(progress_pct: float) -> float:
    """Deterministic simulated reefer-unit temperature curve: a gentle
    baseline drift from 3.0C to 4.5C over the trip (normal thermal creep),
    plus a simulated refrigeration/door-open event spiking to ~10.6C between
    30%-50% progress, so the thermal spoilage model has a real excursion to
    react to instead of a flat line. Replace this function's output with a
    real sensor reading (via /v1/telematics/incoming) once hardware is wired up."""
    progress_pct = max(0.0, min(1.0, progress_pct))
    baseline = 3.0 + 1.5 * progress_pct
    if 0.30 <= progress_pct <= 0.50:
        spike_frac = max(0.0, 1.0 - abs(progress_pct - 0.40) / 0.10)
        baseline += 7.0 * spike_frac
    return round(baseline, 2)


def thermal_rate_multiplier(temp_c: float) -> float:
    """Q10-style rate multiplier: spoilage accrues at the baseline rate at or
    below the live safe-temp threshold, and accelerates the further above it
    the cargo runs. Reads live_settings fresh every call, so a threshold
    change applies immediately to risk still accruing, not just new trips."""
    excess = max(0.0, temp_c - live_settings["safe_temp_max_c"])
    return THERMAL_Q10 ** (excess / 10.0)


def cargo_temp_status(temp_c: float) -> str:
    safe_max = live_settings["safe_temp_max_c"]
    if temp_c <= safe_max:
        return "Nominal"
    if temp_c <= safe_max + 5.0:
        return "Elevated"
    return "Critical"


def integrate_thermal_risk_hours(elapsed_hours: float, total_hours: float, steps: int = 200) -> float:
    """Numerically integrates the known simulated temperature curve from 0 to
    elapsed_hours, weighting each instant by its thermal_rate_multiplier.
    Recomputed fresh on every call (no incremental state, so it can never
    drift regardless of how often the frontend happens to poll)."""
    if elapsed_hours <= 0 or total_hours <= 0:
        return 0.0
    elapsed_hours = min(elapsed_hours, total_hours)
    dt = elapsed_hours / steps
    accum = 0.0
    for i in range(steps):
        t_mid = (i + 0.5) * dt
        progress = min(1.0, t_mid / total_hours)
        temp = simulated_cargo_temp_c(progress)
        accum += thermal_rate_multiplier(temp) * dt
    return accum


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check() -> dict:
    return {"status": "online", "service": "Northern Cape Fleet Telemetry and Routing API"}


@app.get("/v1/towns")
def list_towns() -> dict:
    return {"towns": AVAILABLE_TOWNS}


@app.get("/v1/settings/thresholds")
def get_thresholds() -> dict:
    """Current business-adjustable thresholds. Read by the dashboard on load
    so a fresh page/session shows whatever was last set, not the default."""
    return dict(live_settings)


@app.post("/v1/settings/thresholds")
def update_thresholds(settings: ThresholdSettings) -> dict:
    """Updates the safe cargo-temperature threshold immediately — it's read
    fresh on every thermal-risk calculation (see thermal_rate_multiplier /
    cargo_temp_status), so this takes effect on the current trip right away,
    no restart or trip reconfigure needed."""
    live_settings["safe_temp_max_c"] = settings.safe_temp_max_c
    logger.info("SETTINGS UPDATED -> safe_temp_max_c=%.1f", settings.safe_temp_max_c)
    return {"status": "updated", **live_settings}


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
            trip_state["sim_spoilage_cumulative"] = build_cumulative_spoilage_table(optimized["path_nodes"])
            trip_state["sim_total_hours"] = total_hours
            trip_state["trip_start_ts"] = time.monotonic()

            # Static full-journey reference figure, cached once here — this is
            # what makes it a stable baseline instead of recomputing (and thus
            # silently staying frozen at the same value) on every poll.
            standard = build_route(origin_node, destination_node, time_weight)
            trip_state["trip_plan"] = {
                "origin_town": config.origin_town,
                "destination_town": config.destination_town,
                "optimized_route": optimized,
                "standard_route": standard,
                "business_value": {
                    "time_saved_mins": round(standard["total_time_mins"] - optimized["total_time_mins"], 1),
                    "rand_saved": round(standard["expected_loss_rand"] - optimized["expected_loss_rand"], -2),
                    "standard_spoilage_risk_pct": standard["spoilage_risk_pct"],
                    "optimized_spoilage_risk_pct": optimized["spoilage_risk_pct"],
                    "shipment_value_rand": SHIPMENT_VALUE_RAND,
                },
            }

            start_lon, start_lat = cluster_coord[origin_node]
            tracking_state["vehicle_id"] = VEHICLE_ID
            tracking_state["lat"] = start_lat
            tracking_state["lon"] = start_lon
            tracking_state["feed_source"] = FEED_SOURCE_SIMULATED
            tracking_state["cargo_temp_c"] = simulated_cargo_temp_c(0.0)
            tracking_state["speed_kmh"] = 0.0
        else:
            trip_state["origin_town"] = None
            trip_state["origin_node"] = None
            trip_state["sim_cumulative"] = None
            trip_state["sim_spoilage_cumulative"] = None
            trip_state["sim_total_hours"] = 0.0
            trip_state["trip_start_ts"] = None
            trip_state["trip_plan"] = None
            tracking_state["feed_source"] = FEED_SOURCE_REAL

        # Reset cargo thermal accumulation for the new trip regardless of mode.
        trip_state["thermal_risk_accum_hours"] = 0.0
        trip_state["last_thermal_update_ts"] = None

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
    """Returns the current tracking state, plus live cargo-condition risk.

    Position is computed fresh on every call from elapsed wall-clock time —
    never mutated in place — so repeated polling never jumps or resets.

    Cargo risk has two independently tracked components, matching the two
    physical mechanisms that actually cause cold-chain spoilage:
      - mechanical_risk_pct: road-roughness x time damage ACTUALLY accrued so
        far on the path already driven (simulator mode only — hardware mode
        has no fixed planned route to measure "so far" against).
      - thermal_risk_pct: temperature-exposure damage accrued so far, from
        cargo_temp_c integrated over elapsed time. Simulator mode uses a
        closed-form integral of the known simulated curve (exact, no drift,
        unaffected by live weather — see simulated_cargo_temp_c). Hardware
        mode accumulates incrementally between polls using whatever the
        last real (or last-known) cargo_temp_c reading was, and — while
        idling — drifts that reading itself using live ambient weather (see
        CURRENT_WEATHER / weather_engine.py) as a stand-in for a real
        reefer-interior sensor.
    composite_cargo_risk_pct blends the two into one headline number.
    """
    # Snapshot just the position needed for a weather lookup, then release
    # the lock before making the (occasionally slow, first-call-per-cache-
    # window) external weather request — holding state_lock across a
    # blocking network call would stall every other endpoint's polling.
    with state_lock:
        snapshot_lat = tracking_state["lat"]
        snapshot_lon = tracking_state["lon"]

    weather_reading = (
        weather_engine.get_current_weather(snapshot_lat, snapshot_lon)
        if snapshot_lat is not None and snapshot_lon is not None
        else None
    )

    with state_lock:
        if weather_reading is not None:
            CURRENT_WEATHER["temp_c"] = weather_reading["temp_c"]
            CURRENT_WEATHER["rain_mm"] = weather_reading["rain_mm"]
            CURRENT_WEATHER["alert"] = weather_reading["alert"]

        mechanical_risk_pct: float | None = None
        thermal_risk_pct: float

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

            total_hours = trip_state["sim_total_hours"]
            progress_pct = min(1.0, elapsed_sim_hours / total_hours) if total_hours else 1.0
            tracking_state["cargo_temp_c"] = simulated_cargo_temp_c(progress_pct)

            mechanical_accum = interpolate_cumulative_scalar(trip_state["sim_spoilage_cumulative"], elapsed_sim_hours)
            mechanical_risk_pct = round(min(1.0, mechanical_accum / SPOILAGE_THRESHOLD) * 100, 1)

            thermal_accum_hours = integrate_thermal_risk_hours(elapsed_sim_hours, total_hours)
            thermal_risk_pct = round(min(1.0, thermal_accum_hours / THERMAL_SPOILAGE_THRESHOLD_HOURS) * 100, 1)

        else:
            # Hardware mode (or unconfigured): no known future curve to
            # integrate in closed form, so accumulate thermally in real time
            # between polls using whatever cargo_temp_c is currently known.
            now = time.monotonic()
            last_ts = trip_state["last_thermal_update_ts"]
            if last_ts is not None:
                dt_hours = max(0.0, (now - last_ts) / 3600.0)
                rate = thermal_rate_multiplier(tracking_state["cargo_temp_c"])
                trip_state["thermal_risk_accum_hours"] += dt_hours * rate

                # Ambient-weather-driven idling warm-up: while the vehicle is
                # stopped (e.g. at a border post or loading bay), the cargo
                # chamber's own reading drifts based on live outside air
                # temperature — this is what a real cargo sensor would show
                # once wired up; independent of the thermal-risk accumulation
                # above, which tracks cumulative spoilage exposure, not the
                # instantaneous temperature itself.
                is_idling = tracking_state.get("speed_kmh", 0.0) <= 0.5
                if is_idling:
                    ambient_temp_c = CURRENT_WEATHER["temp_c"]
                    if ambient_temp_c >= EXTREME_HEAT_THRESHOLD_C:
                        warming_tick_c = IDLE_WARMING_TICK_EXTREME_C
                    elif ambient_temp_c >= COLD_CLIMATE_THRESHOLD_C:
                        warming_tick_c = IDLE_WARMING_TICK_STANDARD_C
                    else:
                        warming_tick_c = IDLE_WARMING_TICK_COLD_C
                    tracking_state["cargo_temp_c"] = round(
                        min(IDLE_WARMING_MAX_CARGO_TEMP_C, tracking_state["cargo_temp_c"] + warming_tick_c), 2
                    )
            trip_state["last_thermal_update_ts"] = now

            thermal_risk_pct = round(
                min(1.0, trip_state["thermal_risk_accum_hours"] / THERMAL_SPOILAGE_THRESHOLD_HOURS) * 100, 1
            )

        if mechanical_risk_pct is not None:
            composite_cargo_risk_pct = round(
                MECHANICAL_RISK_WEIGHT * mechanical_risk_pct + THERMAL_RISK_WEIGHT * thermal_risk_pct, 1
            )
        else:
            composite_cargo_risk_pct = thermal_risk_pct

        expected_loss_rand_so_far = round(composite_cargo_risk_pct / 100.0 * SHIPMENT_VALUE_RAND, -2)

        payload = dict(tracking_state)
        payload["cargo_temp_status"] = cargo_temp_status(tracking_state["cargo_temp_c"])
        payload["mechanical_risk_pct"] = mechanical_risk_pct
        payload["thermal_risk_pct"] = thermal_risk_pct
        payload["composite_cargo_risk_pct"] = composite_cargo_risk_pct
        payload["expected_loss_rand_so_far"] = expected_loss_rand_so_far
        payload["ambient_weather"] = dict(CURRENT_WEATHER)

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
    """Computes the standard (time-only) and spoilage-optimized route from
    the vehicle's CURRENT position to the destination — in both modes — so
    these figures genuinely shrink as the truck drives instead of staying
    frozen at the full-trip value. (Simulator mode previously pinned the
    origin to the trip's start for every call, which made every routing
    figure identical from departure to arrival — that's fixed here.)

    For simulator mode, `trip_plan` is also included: the fixed, full
    origin-to-destination comparison cached once at /v1/trip/configure time,
    for a stable baseline to compare the live/remaining figures against."""
    with state_lock:
        if not trip_state["configured"]:
            raise HTTPException(status_code=409, detail="No trip configured yet. Call /v1/trip/configure first.")
        mode = trip_state["mode"]
        destination_node = trip_state["destination_node"]
        current_lat = tracking_state["lat"]
        current_lon = tracking_state["lon"]
        feed_source = tracking_state["feed_source"]
        trip_plan = trip_state["trip_plan"]

    if current_lat is None or current_lon is None:
        raise HTTPException(status_code=409, detail="No position available yet.")

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
        # Live/remaining: recomputed from the truck's actual current position
        # every call — these shrink toward zero as the truck approaches.
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
        # Static full-journey reference (simulator mode only) — the fixed
        # planning figures from when the trip was configured, unaffected by
        # how far the truck has since traveled.
        "trip_plan": trip_plan,
        # kept for compatibility with older callers expecting a single "path"
        "path": optimized["coordinates"],
    }


@app.get("/v1/routing/weather-profile")
def get_routing_weather_profile() -> dict:
    """Samples live ambient weather along the trip.

    Prefers the trip's FIXED full route — Origin/Midpoint/Destination from
    the actual departure-to-arrival plan, cached once at /v1/trip/configure
    time — whenever one exists. This is deliberate: sampling the shrinking
    current-position-to-destination "remaining route" instead (the previous
    behavior) meant "Origin" silently became wherever the truck currently
    was, and once little route remained, all three points could collapse
    onto nearly the same coordinate. The fixed plan never shrinks and never
    mislabels, so those three rows always mean what they say.

    Falls back to the remaining route only when no fixed plan exists
    (hardware mode has no known origin town ahead of time, so there's no
    fixed corridor to sample).

    Always ALSO includes a separate 'Current Position' reading from the
    vehicle's live coordinates — this is what tracks the truck in real
    time as it drives, distinct from the fixed route-corridor forecast rows.
    """
    with state_lock:
        if not trip_state["configured"]:
            raise HTTPException(status_code=409, detail="No trip configured yet. Call /v1/trip/configure first.")
        destination_node = trip_state["destination_node"]
        current_lat = tracking_state["lat"]
        current_lon = tracking_state["lon"]
        origin_town = trip_state["origin_town"]
        destination_town = trip_state["destination_town"]
        trip_plan = trip_state["trip_plan"]

    if current_lat is None or current_lon is None:
        raise HTTPException(status_code=409, detail="No position available yet.")

    if trip_plan is not None:
        coordinates = trip_plan["optimized_route"]["coordinates"]
        route_basis = "fixed_trip_plan"
    else:
        origin_node, _snap_km = node_index.snap(current_lat, current_lon)
        optimized = build_route(origin_node, destination_node, spoilage_weight)
        coordinates = optimized["coordinates"]
        route_basis = "remaining_route"

    if len(coordinates) == 1:
        sample_points = [coordinates[0], coordinates[0], coordinates[0]]
    else:
        midpoint_index = len(coordinates) // 2
        sample_points = [coordinates[0], coordinates[midpoint_index], coordinates[-1]]
    labels = ["Origin", "Midpoint", "Destination"]

    segments = []
    for label, (lon, lat) in zip(labels, sample_points):
        reading = weather_engine.get_current_weather(lat, lon)
        segments.append({
            "label": label,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "temp_c": reading["temp_c"],
            "rain_mm": reading["rain_mm"],
            "alert": reading["alert"],
            "source": reading["source"],
        })

    # Live reading at the truck's ACTUAL current position — always reflects
    # wherever it is right now, distinct from the fixed corridor rows above.
    # This is the one that visibly moves as the truck drives.
    current_reading = weather_engine.get_current_weather(current_lat, current_lon)
    segments.append({
        "label": "Current Position",
        "lat": round(current_lat, 5),
        "lon": round(current_lon, 5),
        "temp_c": current_reading["temp_c"],
        "rain_mm": current_reading["rain_mm"],
        "alert": current_reading["alert"],
        "source": current_reading["source"],
    })

    return {
        "vehicle_id": VEHICLE_ID,
        "origin_town": origin_town,
        "destination_town": destination_town,
        "route_basis": route_basis,
        "segments": segments,
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


@app.post("/v1/analytics/strategy")
def generate_ai_strategy(payload: StrategyRequest) -> dict:
    """AI Cognitive Strategy Layer. Takes the routing + cargo-risk KPIs
    already computed by /v1/routing/truck-01 and /v1/telematics/truck-01
    (never re-derives them) and passes them to a locally hosted Qwen2.5-1.5B
    model via `transformers`, asking it to turn the numbers into a business
    strategy memo. Runs entirely on-box — no external API calls, no data
    leaves the machine, no separate daemon (unlike Ollama) needs to be
    installed or started.

    Fails soft: if `transformers` isn't installed or the model can't be
    loaded/run, this returns a 200 with a clean {"status": "error", ...}
    JSON payload (never a raised exception), so a slow first-load or a
    missing dependency can never freeze or crash the rest of the dispatch API.
    """
    logger.info(
        "AI STRATEGY REQUEST -> origin=%s destination=%s standard_time_mins=%.1f "
        "optimized_time_mins=%.1f mechanical_risk_reduction_pct=%.1f thermal_risk_pct=%.1f "
        "surface_profile=%s",
        payload.origin, payload.destination, payload.standard_time_mins,
        payload.optimized_time_mins, payload.mechanical_risk_reduction_pct,
        payload.thermal_risk_pct, payload.surface_profile,
    )

    if hf_pipeline is None:
        logger.error("AI STRATEGY -> 'transformers' package is not installed.")
        return {
            "status": "error",
            "error_type": "transformers_not_installed",
            "message": (
                "The 'transformers' package is not installed on the backend host. "
                "Run `pip install transformers torch --break-system-packages` (or your "
                "project's equivalent) and restart live_backend.py."
            ),
        }

    try:
        generator = get_ai_strategy_pipeline()
    except Exception as exc:  # noqa: BLE001 - deliberately broad: model download/
        # load can fail for many reasons (no internet on first run, disk space,
        # corrupted cache, missing torch backend) and none of them should take
        # down the API process.
        logger.error("AI STRATEGY -> failed to load model '%s': %s", HF_STRATEGY_MODEL, exc)
        return {
            "status": "error",
            "error_type": "model_load_failed",
            "message": f"Could not load local model '{HF_STRATEGY_MODEL}': {exc}",
        }

    time_delta_mins = payload.standard_time_mins - payload.optimized_time_mins

    weather_section = ""
    if payload.ambient_temp_c is not None:
        weather_section = (
            f"- Live ambient temperature at the vehicle's position: {payload.ambient_temp_c:.1f}°C\n"
            f"- Live rain intensity: {(payload.rain_mm or 0.0):.1f} mm\n"
            f"- Live weather alert status: {payload.weather_alert or 'Normal'}\n"
        )

    user_prompt = (
        f"FIXED ROUTE FOR THIS MEMO: {payload.origin} -> {payload.destination}\n"
        f"(Use these exact two town names throughout. Do not mention any other "
        f"origin or destination town anywhere in your answer.)\n\n"
        f"Raw routing and cargo-risk KPIs for this shipment:\n"
        f"- Standard (time-only) route duration: {payload.standard_time_mins:.1f} minutes\n"
        f"- Spoilage-risk-optimized route duration: {payload.optimized_time_mins:.1f} minutes\n"
        f"- Extra/saved travel time from taking the optimized route: {time_delta_mins:+.1f} minutes\n"
        f"- Rand value saved by taking the optimized route: R {payload.rand_saved:,.0f}\n"
        f"- Mechanical (road-roughness) spoilage risk reduction from the optimized route: "
        f"{payload.mechanical_risk_reduction_pct:.1f} percentage points of shipment value\n"
        f"- Current thermal (temperature-exposure) risk accrued so far: {payload.thermal_risk_pct:.1f}%\n"
        f"- Current cargo temperature status: {payload.cargo_temp_status}\n"
        f"- Road surface profile along the optimized route: {payload.surface_profile}\n"
        f"{weather_section}"
        f"- Total shipment value at risk: R {payload.shipment_value_rand:,.0f}\n\n"
        "Using ONLY these figures, produce a strategic business memo for the cooperative's "
        "dispatch manager: quantify the Rand-value trade-off between extra travel time and "
        "spoilage avoided, address both the mechanical and thermal risk components separately, "
        "note the vehicle wear-and-tear implication of the surface profile, and recommend "
        "whether this shipment should take the standard or the optimized route.\n\n"
        f"Reminder: this memo is specifically about the {payload.origin} -> {payload.destination} "
        f"route. Do not substitute any other town names. Do not write your own title, date, or "
        f"route line — begin directly with '## Introduction'."
    )

    messages = [
        {"role": "system", "content": AI_STRATEGY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    logger.info(
        "AI STRATEGY -> dispatching payload to local model '%s' for inference (may take a while on CPU)",
        HF_STRATEGY_MODEL,
    )

    try:
        output = generator(messages, max_new_tokens=700, do_sample=True, temperature=0.4)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any local inference
        # failure mode must degrade to a clean JSON error, never propagate and
        # freeze the request thread or take down the API process.
        logger.error("AI STRATEGY -> inference failed: %s", exc)
        return {
            "status": "error",
            "error_type": "inference_failed",
            "message": f"Local model inference failed unexpectedly: {exc}",
        }

    strategy_markdown = strip_hallucinated_header(extract_strategy_text(output))

    if not strategy_markdown:
        logger.warning("AI STRATEGY -> model returned an empty response.")
        return {
            "status": "error",
            "error_type": "empty_response",
            "message": "The local model returned an empty response. Try again.",
        }

    # Built from the actual request data, not the model — this is what
    # guarantees the header always shows the real route and today's real
    # date, even though the (small, local) model can occasionally drift on
    # named entities or has no way to know the current date on its own.
    memo_header = (
        f"# Fleet Dispatch Strategy Memo\n\n"
        f"**Route:** {payload.origin} → {payload.destination}\n"
        f"**Date:** {datetime.now().strftime('%d %B %Y')}\n\n"
        f"---\n\n"
    )
    strategy_markdown = memo_header + strategy_markdown

    logger.info("AI STRATEGY -> inference complete, %d characters returned", len(strategy_markdown))

    return {
        "status": "success",
        "model": HF_STRATEGY_MODEL,
        "origin": payload.origin,
        "destination": payload.destination,
        "strategy_markdown": strategy_markdown,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
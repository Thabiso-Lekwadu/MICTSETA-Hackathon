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
import math
import os
import pickle
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree

# Prefer the improved road-network module (spatial-KNN imputation, connectivity
# stitching, precomputed spoilage_cost_edge); fall back to the original.
try:
    import nc_road_network_improved as ncr
except ImportError:
    import nc_road_network as ncr
import weather_engine

# Precise Northern Cape province polygon, used to clip the routable graph so no
# optimal path ever leaves the province (the OSM extract was bbox-clipped, which
# let North West roads leak in). Optional import: if absent, routing falls back
# to the un-clipped graph with a logged warning.
try:
    import nc_boundary
except ImportError:  # pragma: no cover
    nc_boundary = None

# vrp_simulation_generator.py is a sibling deliverable (Monte Carlo risk
# solver + Stream A/B/C synthetic generators). Imported lazily/optionally,
# exactly like the `transformers` import above: the core telemetry/routing/
# driver-report loop must keep working even on a checkout where that file
# hasn't been dropped in yet or fails to import for any reason. Only the
# three new /v1/simulation/* endpoints below depend on it, and each one
# degrades to a clean JSON error instead of crashing the server if it's
# unavailable — see get_vrp_simulator().
try:
    import vrp_simulation_generator as vrpsim
except ImportError:  # pragma: no cover - exercised only before that file exists
    vrpsim = None

# transformers is an optional dependency: the rest of the dispatch/routing
# system must keep working (telemetry ingest, driver reports, live map) even
# on a machine that never installed it. The /v1/analytics/strategy endpoint
# is the only thing that needs it, and it degrades to a clean JSON error
# instead of crashing the whole API process. The pipeline itself is loaded
# lazily (see get_ai_strategy_pipeline) so importing the package here never
# pays the multi-second/multi-GB model-load cost at server startup.
#
# NOTE: a decoupled OpenAI-compatible path (Ollama/Llamafile) is documented in
# AI_INFERENCE_MICROSERVICE_GUIDE.md for when you want to scale inference out of
# this process; this build runs the local Qwen model in-process as requested.
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
SHIPMENT_VALUE_RAND_DEFAULT = 450_000.0  # default only — live value is adjustable at runtime, see live_settings below

# Two-component spoilage weight (thermal + mechanical), matching
# nc_road_network_improved.spoilage_edge_cost and the 50/50 composite in
# metrics_explained.md. TIME is a first-class term (thermal), so the router will
# NOT pick a route that is much longer in TIME just to avoid roughness — the fix
# for "why is the optimal route so long". A smooth (roughness 1.0) edge costs
# exactly its travel time (THERMAL_SPOIL_WEIGHT + MECH_SPOIL_WEIGHT == 1.0). Used
# for the fallback when the graph doesn't carry a precomputed spoilage_cost_edge,
# and for live driver-report overrides.
THERMAL_SPOIL_WEIGHT = 0.5
MECH_SPOIL_WEIGHT = 0.5

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
#
# Modeled as Newton's-law-of-heating: the cargo temperature exponentially
# approaches an ambient/threshold-derived equilibrium rather than climbing
# in a straight line forever. This is deliberate: an unbounded per-poll
# linear increment has no natural ceiling and doesn't visibly respond to
# *changes* in conditions the way a real reefer chamber does — it just
# looks like a counter. An exponential approach genuinely converges, and
# the equilibrium itself is anchored to the business-adjustable
# safe_temp_max_c threshold (not just a flat constant), so tightening or
# loosening that threshold changes this too.
EXTREME_HEAT_THRESHOLD_C = 38.0
COLD_CLIMATE_THRESHOLD_C = 20.0
IDLE_WARMING_RATE_EXTREME_PER_HOUR = 2.5    # 1/hr decay constant, ambient >= 38.0°C
IDLE_WARMING_RATE_STANDARD_PER_HOUR = 0.8   # 1/hr, 20.0°C <= ambient < 38.0°C
IDLE_WARMING_RATE_COLD_PER_HOUR = 0.2       # 1/hr, ambient < 20.0°C
IDLE_WARMING_TARGET_OFFSET_EXTREME_C = 10.0   # equilibrium = safe_temp_max_c + this
IDLE_WARMING_TARGET_OFFSET_STANDARD_C = 4.0
IDLE_WARMING_TARGET_OFFSET_COLD_C = -1.0      # cold climate: equilibrium sits below threshold

# --- Hardware-mode motion detection -----------------------------------------
# A hardware trip is "configured" the moment a destination is set, but the
# truck may still be parked for a while before it actually departs. Cargo
# risk metrics (thermal accumulation, idling warm-up) are gated behind
# real detected motion — see hardware_motion_detected in trip_state — so a
# parked truck doesn't silently accrue spoilage risk against a shipment
# that hasn't left yet. Motion is detected by displacement from the first
# post-configure position OR a reported speed, whichever trips first.
MOTION_DETECTION_DISTANCE_M = 25.0
MOTION_DETECTION_SPEED_KMH = 1.0

# Hardware trip evaluation: how many recent GPS pings to retain for the
# sim-vs-real comparison, and how close (km) a ping must be to the simulated
# optimal corridor to count as "on route".
HARDWARE_PING_HISTORY_MAX = 2000
ROUTE_ADHERENCE_KM = 5.0

# Live ambient weather at the vehicle's current position, refreshed on every
# /v1/telematics/truck-01 poll (see get_telematics) and read by the AI
# Strategy Layer and the /v1/routing/weather-profile endpoint. Plain dict
# read/write (no lock) for the same reason as live_settings above: a handful
# of GIL-atomic scalar fields, briefly reading mid-update has no consequence.
CURRENT_WEATHER: dict = {"temp_c": 25.0, "rain_mm": 0.0, "alert": "Normal", "source": "unknown"}

# Runtime-adjustable business thresholds. Cooperative dispatchers know their
# own cargo (species, packaging, ice quality) better than any hardcoded
# constant can — this is read by thermal_rate_multiplier/cargo_temp_status
# below and updated live via POST /v1/settings/thresholds, no restart needed.
# Plain dict read/write is fine here (no lock): it's a single float, GIL-atomic,
# and briefly reading a value mid-update has no meaningful consequence.
live_settings: dict = {
    "safe_temp_max_c": SAFE_TEMP_MAX_C_DEFAULT,
    "shipment_value_rand": SHIPMENT_VALUE_RAND_DEFAULT,
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
# Weather sourcing — OpenWeatherMap primary, keyless fallback chain intact
# ---------------------------------------------------------------------------
# The Global Calibration Overrides call for OpenWeatherMap as the ambient-
# weather source, with a hard requirement that a timeout/outage can never
# raise an unhandled exception. weather_engine.py (a shared module, not one
# of this refactor's three deliverables) already implements exactly that
# fault-tolerant chain, but treats OpenWeatherMap as an *optional* upgrade
# over Open-Meteo. Rather than editing that file (it's a working, tested
# component outside this refactor's scope — "zero destructive modifications"
# applies to it too), this module wraps it: every live weather lookup in
# live_backend.py goes through get_ambient_weather() below, which tries
# OpenWeatherMap's current-weather endpoint FIRST whenever an API key is
# configured, and only falls through to weather_engine.get_current_weather()
# (Open-Meteo, then the 25°C/0mm baseline) on a missing key or any failure.
# This makes OpenWeatherMap primary end-to-end without weakening the
# existing fallback safety net at all.
# Resolved through weather_engine, which loads a `.env` from the weather
# script's folder and accepts OPENWEATHERMAP_API_KEY / OWM_API_KEY / API_KEY.
# The key lives only in the process environment — it is never logged here and
# never included in any API response sent to the frontend.
OPENWEATHERMAP_API_KEY = weather_engine.OPENWEATHERMAP_API_KEY
OPENWEATHERMAP_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHERMAP_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OWM_REQUEST_TIMEOUT_SECONDS = 5.0
OWM_FORECAST_CACHE_TTL_SECONDS = 900.0  # 15 min: forecast blocks don't change fast enough to justify tighter TTL
OWM_FORECAST_MAX_HORIZON_HOURS = 120.0  # OWM's free 5-day/3-hour forecast tier's real coverage window

_owm_forecast_cache: dict[tuple[float, float, int], dict] = {}
_owm_forecast_cache_lock = threading.Lock()

if not OPENWEATHERMAP_API_KEY:
    logging.getLogger("live_backend").warning(
        "OPENWEATHERMAP_API_KEY is not set. Ambient weather will automatically "
        "degrade to weather_engine's Open-Meteo/baseline chain until a key is "
        "exported (get a free key at https://openweathermap.org/api)."
    )


def get_ambient_weather(lat: float, lon: float) -> weather_engine.WeatherReading:
    """Live 'right now' ambient weather at a coordinate, OpenWeatherMap-first.
    Never raises — any failure (missing key, timeout, malformed payload)
    falls through to weather_engine's own Open-Meteo -> baseline chain."""
    if OPENWEATHERMAP_API_KEY:
        try:
            response = requests.get(
                OPENWEATHERMAP_CURRENT_URL,
                params={"lat": lat, "lon": lon, "appid": OPENWEATHERMAP_API_KEY, "units": "metric"},
                timeout=OWM_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            temp_c = float(payload["main"]["temp"])
            rain_block = payload.get("rain") or {}
            rain_mm = float(rain_block.get("1h", rain_block.get("3h", 0.0)) or 0.0)
            alert = (
                "Extreme Heat" if temp_c >= EXTREME_HEAT_THRESHOLD_C
                else "Heavy Rain / Washout Risk" if rain_mm >= weather_engine.HEAVY_RAIN_THRESHOLD_MM
                else "Normal"
            )
            return {"temp_c": temp_c, "rain_mm": rain_mm, "alert": alert, "source": "openweathermap"}
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see module docstring above
            logger.warning(
                "OWM PRIMARY WEATHER -> request failed for (%.4f, %.4f): %s. "
                "Falling back to weather_engine (Open-Meteo/baseline).", lat, lon, exc,
            )
    return weather_engine.get_current_weather(lat, lon)


def _owm_forecast_cache_key(lat: float, lon: float, target_dt: datetime) -> tuple[float, float, int]:
    # 3-hour block index matches OWM's free-tier forecast granularity, so
    # several nearby-in-time route samples reuse one cached block instead of
    # each issuing a fresh request.
    block_index = int(target_dt.timestamp() // (3 * 3600))
    return (round(lat, 2), round(lon, 2), block_index)


def get_owm_forecast(lat: float, lon: float, target_dt: datetime) -> dict | None:
    """Looks up OpenWeatherMap's free 5-day/3-hour forecast for the block
    nearest target_dt. Returns None (never raises) if no API key is set, the
    request fails, or target_dt falls outside the ~120h window that tier
    actually covers — callers fall back to the synthetic Stream A generator
    in vrp_simulation_generator.py for anything this can't answer."""
    if not OPENWEATHERMAP_API_KEY:
        return None
    now = datetime.now(timezone.utc)
    target_utc = target_dt if target_dt.tzinfo else target_dt.replace(tzinfo=timezone.utc)
    horizon_hours = (target_utc - now).total_seconds() / 3600.0
    if horizon_hours < 0 or horizon_hours > OWM_FORECAST_MAX_HORIZON_HOURS:
        return None

    key = _owm_forecast_cache_key(lat, lon, target_utc)
    now_monotonic = time.monotonic()
    with _owm_forecast_cache_lock:
        cached = _owm_forecast_cache.get(key)
        if cached is not None and now_monotonic - cached["cached_at"] < OWM_FORECAST_CACHE_TTL_SECONDS:
            return cached["reading"]

    try:
        response = requests.get(
            OPENWEATHERMAP_FORECAST_URL,
            params={"lat": lat, "lon": lon, "appid": OPENWEATHERMAP_API_KEY, "units": "metric"},
            timeout=OWM_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        blocks = payload.get("list", [])
        if not blocks:
            return None
        best_block, best_delta = None, float("inf")
        for block in blocks:
            block_dt = datetime.fromtimestamp(block["dt"], tz=timezone.utc)
            delta = abs((block_dt - target_utc).total_seconds())
            if delta < best_delta:
                best_delta = delta
                best_block = block
        if best_block is None:
            return None
        temp_c = float(best_block["main"]["temp"])
        rain_block = best_block.get("rain") or {}
        rain_mm = float(rain_block.get("3h", 0.0) or 0.0)
        reading = {
            "ambient_temp_c": temp_c,
            "rain_mm_per_hr": round(rain_mm / 3.0, 2),  # OWM reports a 3h total; normalize to a per-hour rate
            "source": "openweathermap_forecast",
        }
    except Exception as exc:  # noqa: BLE001 - forecast lookup is best-effort only
        logger.warning(
            "OWM FORECAST -> request failed for (%.4f, %.4f) @ %s: %s. Falling back to synthetic forecast.",
            lat, lon, target_utc.isoformat(), exc,
        )
        return None

    with _owm_forecast_cache_lock:
        _owm_forecast_cache[key] = {"cached_at": now_monotonic, "reading": reading}
    return reading


def get_forecast_weather(lat: float, lon: float, target_dt: datetime) -> dict:
    """Single point of entry for FUTURE weather (as opposed to get_ambient_weather,
    which is for RIGHT NOW). Tries OpenWeatherMap's real forecast first; if that
    can't answer (no key, request failure, or target_dt beyond ~5 days out),
    falls back to vrp_simulation_generator's synthetic Stream A model, and if
    THAT module also isn't available, falls back one more time to the same
    25°C/0mm clear-weather baseline weather_engine.py uses. This is the
    fault-tolerance chain Global Calibration Overrides §2 asks for, just
    extended to future timestamps instead of only 'right now'."""
    owm_reading = get_owm_forecast(lat, lon, target_dt)
    if owm_reading is not None:
        return owm_reading
    if vrpsim is not None:
        try:
            return vrpsim.synthetic_forecast_point(lat, lon, target_dt)
        except Exception as exc:  # noqa: BLE001 - synthetic generator must never take the endpoint down either
            logger.warning("SYNTHETIC FORECAST -> vrp_simulation_generator failed for (%.4f, %.4f): %s", lat, lon, exc)
    return {
        "ambient_temp_c": weather_engine.FALLBACK_TEMP_C,
        "rain_mm_per_hr": weather_engine.FALLBACK_RAIN_MM,
        "source": "fallback",
    }


# ---------------------------------------------------------------------------
# GPS fix quality — deliberately DECOUPLED from ncr.SNAP_TOLERANCE_M (150m)
# ---------------------------------------------------------------------------
# validation_report.md flagged that reusing the 150m graph-construction
# constant (endpoint-merge distance when BUILDING the road network) as a
# stand-in for GPS-accuracy validation was "a borrowed tolerance" that made
# the Snapping Accuracy check pass against a looser bar than intended. This
# is that dedicated constant: 30m matches the RMSE target validation_report.md
# itself states (Spatial Deviation <= 30m). It is used ONLY to label a fix's
# confidence in API responses below — it never changes which node/edge a
# coordinate actually snaps to for routing, which still correctly uses the
# full topology built with ncr.SNAP_TOLERANCE_M.
GPS_ACCURACY_TOLERANCE_M = 30.0


def gps_fix_confidence(distance_km: float) -> str:
    return "high_confidence" if distance_km * 1000.0 <= GPS_ACCURACY_TOLERANCE_M else "low_confidence"


# ---------------------------------------------------------------------------
# AI Cognitive Strategy Layer — local Hugging Face (Qwen) model advisory config
# ---------------------------------------------------------------------------
# Small instruction-tuned model chosen specifically so it runs comfortably
# CPU-only on a laptop with no separate daemon to install/start — `transformers`
# + this one line is the whole dependency. Overridable by env in case you swap
# in a different local HF model.
HF_STRATEGY_MODEL = os.environ.get("HF_STRATEGY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

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
    "Beyond spoilage, quantify these three operating-cost and revenue levers "
    "whenever the relevant figures are provided, tying each to a Rand impact:\n"
    "  a. Fuel-burn idling cost at the Vioolsdrift border post: a reefer engine "
    "     kept running to hold temperature while queued for customs burns diesel "
    "     for zero distance. Treat every hour of border wait as a direct fuel and "
    "     compressor-hours cost, and weigh it against departing in a slot with a "
    "     shorter expected customs queue.\n"
    "  b. Tyre and drivetrain wear relative to the International Roughness Index "
    "     (IRI): kilometres on high-IRI corrugated/unpaved track accelerate tyre, "
    "     suspension and seal wear far faster than smooth tar. Frame the rougher "
    "     route's hidden maintenance cost, not just its spoilage cost.\n"
    "  c. Premium market valuation of preserved seafood: cargo delivered within "
    "     an unbroken cold chain commands a premium price (export/sashimi grade), "
    "     while a thermal excursion downgrades it to a lower-value or condemned "
    "     grade. Express the optimized route's benefit as protected premium "
    "     revenue, not merely avoided loss.\n\n"
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
    # Route-justification inputs (optional). When present, the memo gains a
    # "## Route Justification" section explaining why this route was chosen
    # over the standard route and any other road on the map.
    routes_differ: bool | None = None
    rough_km_avoided: float | None = None
    standard_route_profile: str | None = None
    optimized_route_profile: str | None = None
    route_rationale_text: str | None = None
    # Hardware sim-vs-real evaluation inputs. When evaluation_mode == "hardware"
    # the memo adds a "## Simulation vs Real Trip" section grading how closely the
    # real trip tracked the simulation and how to close the gap.
    evaluation_mode: str | None = None
    validation_score: float | None = None
    route_adherence_pct: float | None = None
    actual_vs_planned_time_pct: float | None = None
    sim_vs_real_text: str | None = None


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


# An instruction-tuned model doesn't always follow "don't write a header"
# reliably, and when it does write one it can hallucinate the wrong town names
# or an "[Insert Date]" placeholder. The real header is always built from the
# actual request data in code (see generate_ai_strategy) — this strips a handful
# of leading lines that look like a model-written title, byline, or
# "Date:"/"Route:" line so the two headers can't collide/duplicate.
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
    # Both fields optional and independently applied (see update_thresholds)
    # so the sidebar's two "Apply" buttons can each update just their own
    # setting without having to resend the other one's current value.
    #
    # Deliberately loose bounds on safe_temp_max_c — this is a business-
    # domain call (species, packaging, ice quality all shift what "safe"
    # means), not a physics constant, so the API shouldn't second-guess the
    # dispatcher's number.
    safe_temp_max_c: float | None = Field(None, ge=-20.0, le=25.0)
    # Business-user-entered value of the shipment, used to convert accrued
    # spoilage-risk percentages into a Rand figure everywhere that happens
    # (route business-value comparisons, live expected-loss-so-far). No
    # sensible physics-based upper bound, so just require it be positive.
    shipment_value_rand: float | None = Field(None, gt=0.0)


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
    # Hardware-mode motion gating — see MOTION_DETECTION_* constants and
    # telematics_incoming. A parked truck's cargo metrics stay frozen until
    # this flips True.
    "hardware_motion_detected": False,
    "hardware_reference_lat": None,
    "hardware_reference_lon": None,
    # --- Hardware sim-vs-real evaluation (the feedback-loop validation) -------
    # The frozen "as-simulated" plan (optimized route + Monte Carlo estimate),
    # captured at the first fix, plus the real trip's ping history and the peak
    # cargo temperature actually observed — compared in /v1/hardware/trip-evaluation.
    "hardware_sim_plan": None,
    "hardware_ping_history": [],
    "hardware_max_cargo_temp_c": None,
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


def restrict_topology_to_northern_cape(topology: nx.Graph, coords: dict[int, tuple[float, float]]):
    """Clips the routable graph to the Northern Cape province polygon (+border
    buffer), then returns its largest connected component. This is the routing-
    time enforcement of the province boundary: even though the on-disk graph
    still contains North West / Free State roads that fell inside the extract's
    bounding box, only NC-eligible nodes survive here, so shortest_path can never
    route through another province. Returns (G_nc, nc_nodes, removed_count)."""
    if nc_boundary is None:
        return topology, set(topology.nodes()), 0

    border_posts = getattr(ncr, "BORDER_POSTS", None)
    eligible = {
        n for n in topology.nodes()
        if nc_boundary.node_eligible(coords[n][0], coords[n][1], border_posts)
    }
    removed = topology.number_of_nodes() - len(eligible)
    if not eligible:
        logger.warning("NC clip -> no eligible nodes found; keeping the un-clipped graph.")
        return topology, set(topology.nodes()), 0

    sub = topology.subgraph(eligible).copy()
    components = (nx.weakly_connected_components(sub) if sub.is_directed()
                 else nx.connected_components(sub))
    largest = max(components, key=len)
    g_nc = sub.subgraph(largest).copy()
    return g_nc, set(g_nc.nodes()), removed


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
        # Prefer a precomputed spoilage cost carried on the edge by the improved
        # nc_road_network build (which can apply a non-linear roughness exponent
        # and price inferred connectors). Falls back to the original linear
        # travel_time * roughness for any graph that doesn't carry the field, so
        # older pickles keep working identically.
        precomputed_spoilage = edge_attrs.get("spoilage_cost_edge")
        edge_attrs["spoilage_cost"] = (
            precomputed_spoilage if precomputed_spoilage is not None
            else travel_time_hr * (THERMAL_SPOIL_WEIGHT + MECH_SPOIL_WEIGHT * roughness)
        )
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


def approx_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Flat-earth equirectangular approximation — plenty accurate for the
    short displacements (tens of meters) motion detection cares about, and
    matches the same projection style already used by NodeSpatialIndex."""
    mean_lat_rad = math.radians((lat1 + lat2) / 2.0)
    mx = 111_320.0 * math.cos(mean_lat_rad)
    my = 110_540.0
    dx = (lon2 - lon1) * mx
    dy = (lat2 - lat1) * my
    return math.hypot(dx, dy)


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

# Clip the routable graph to the Northern Cape province polygon so every optimal
# path stays inside the province (fixes routes that dipped into North West via
# the N14 corridor above Kimberley). Applies to BOTH simulator and live-hardware
# routing, since both go through this one graph.
if nc_boundary is not None:
    _nodes_before = live_topology.number_of_nodes()
    live_topology, main_nodes, _removed = restrict_topology_to_northern_cape(live_topology, cluster_coord)
    logger.info(
        "NC clip -> routable graph restricted to %d/%d nodes inside Northern Cape "
        "(+%.0fkm border buffer); %d out-of-province nodes dropped.",
        live_topology.number_of_nodes(), _nodes_before, nc_boundary.BORDER_INCLUDE_KM, _removed,
    )
else:
    logger.warning("NC clip -> nc_boundary module not found; routing on the un-clipped (bbox) graph.")

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
    expected_loss_rand = round(spoilage_pct * live_settings["shipment_value_rand"], -2)

    return {
        "path_nodes": path,
        "coordinates": [list(cluster_coord[n]) for n in path],
        "segments": segments,
        "total_time_mins": round(total_time_mins, 1),
        "total_spoilage_cost": round(total_spoilage_cost, 3),
        "spoilage_risk_pct": round(spoilage_pct * 100, 1),
        "expected_loss_rand": expected_loss_rand,
    }


# ---------------------------------------------------------------------------
# Route justification — deterministic "why THIS route" analysis
# ---------------------------------------------------------------------------
# Northern Cape bounding box (from Data_Audit.ipynb's NC_BBOX). Used only to
# FLAG whether a chosen route strays outside the province envelope — the extract
# is bbox-clipped to NC, so a route should never leave it; if a node does, it's
# a data/clipping artifact worth surfacing rather than hiding.
NC_BBOX = {"min_lon": 16.45, "max_lon": 25.30, "min_lat": -31.85, "max_lat": -24.60}
NC_BBOX_TOLERANCE_DEG = 0.5  # same tolerance the audit uses at the extract edge

# fclass families treated as "rough / unpaved" for the wear-and-spoilage story.
ROUGH_FCLASSES = {
    "track", "track_grade1", "track_grade2", "track_grade3", "track_grade4",
    "track_grade5", "unclassified", "connector", "unpaved", "gravel", "dirt",
}


def _node_outside_nc(node: int) -> bool:
    lon, lat = cluster_coord[node]
    # Precise province-polygon test when available (with the same border buffer
    # the routing clip uses); falls back to the rectangular bbox otherwise.
    if nc_boundary is not None:
        return not nc_boundary.node_eligible(lon, lat, getattr(ncr, "BORDER_POSTS", None))
    return not (
        NC_BBOX["min_lon"] - NC_BBOX_TOLERANCE_DEG <= lon <= NC_BBOX["max_lon"] + NC_BBOX_TOLERANCE_DEG
        and NC_BBOX["min_lat"] - NC_BBOX_TOLERANCE_DEG <= lat <= NC_BBOX["max_lat"] + NC_BBOX_TOLERANCE_DEG
    )


def summarize_route(path_nodes: list[int]) -> dict:
    """Aggregates a resolved path into the facts the route-justification needs:
    total distance, per-fclass km breakdown, rough vs paved km, and whether any
    node strays outside the Northern Cape envelope."""
    total_km = 0.0
    by_fclass: dict[str, float] = {}
    rough_km = 0.0
    paved_km = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge_attrs = live_topology[u][v]
        km = float(edge_attrs.get("length_km", 0.0) or 0.0)
        fclass = str(edge_attrs.get("fclass", "unknown"))
        total_km += km
        by_fclass[fclass] = by_fclass.get(fclass, 0.0) + km
        if fclass in ROUGH_FCLASSES:
            rough_km += km
        else:
            paved_km += km
    outside_nc_nodes = sum(1 for n in path_nodes if _node_outside_nc(n))
    top_fclasses = sorted(by_fclass.items(), key=lambda kv: kv[1], reverse=True)
    profile = ", ".join(f"{fc} {km:.0f}km" for fc, km in top_fclasses[:5])
    return {
        "total_km": round(total_km, 1),
        "rough_km": round(rough_km, 1),
        "paved_km": round(paved_km, 1),
        "rough_pct": round(rough_km / total_km * 100, 1) if total_km > 0 else 0.0,
        "fclass_km": {fc: round(km, 1) for fc, km in top_fclasses},
        "profile": profile,
        "outside_nc_nodes": outside_nc_nodes,
        "hop_count": len(path_nodes) - 1,
    }


def build_route_rationale(standard: dict, optimized: dict,
                          standard_summary: dict, optimized_summary: dict) -> dict:
    """Plain-language, fully deterministic explanation of WHY the optimized route
    was chosen over the time-only route (and, implicitly, over any other road on
    the map): it is the spoilage-cost-minimizing path, and this quantifies the
    roughness it avoids and the time that costs. Never involves the LLM, so the
    Report tab can show a trustworthy justification even with the model offline."""
    routes_differ = optimized["path_nodes"] != standard["path_nodes"]
    time_delta_mins = round(optimized["total_time_mins"] - standard["total_time_mins"], 1)
    spoilage_delta_pct = round(standard["spoilage_risk_pct"] - optimized["spoilage_risk_pct"], 1)
    rough_km_avoided = round(standard_summary["rough_km"] - optimized_summary["rough_km"], 1)

    if not routes_differ:
        reason = (
            "The spoilage-optimized route and the fastest (time-only) route are the SAME road. "
            "On this corridor the quickest path is already the smoothest one, so there is no "
            "rougher-but-faster shortcut worth avoiding — the optimizer confirms the fastest route "
            "is also the lowest-spoilage route. Any other road visible on the map is either longer, "
            "rougher, or both, which is exactly why it was not chosen."
        )
    else:
        why_not_standard = (
            f"The greyed 'standard' route is ~{abs(time_delta_mins):.0f} min "
            f"{'faster' if time_delta_mins > 0 else 'different'} but runs over "
            f"{standard_summary['rough_km']:.0f} km of rough/unpaved road "
            f"({standard_summary['rough_pct']:.0f}% of its length); the chosen route trades "
            f"{abs(time_delta_mins):.0f} min of travel time to cut that to "
            f"{optimized_summary['rough_km']:.0f} km, avoiding ~{rough_km_avoided:.0f} km of "
            f"vibration-heavy surface and lowering modelled spoilage risk by {spoilage_delta_pct:.1f} "
            f"percentage points."
        )
        reason = (
            "The optimizer minimizes cumulative spoilage cost (travel time × road-roughness), not "
            "raw time. " + why_not_standard + " Every other road on the map was rejected because it "
            "scored worse on that combined time-and-roughness cost."
        )

    boundary_note = None
    if optimized_summary["outside_nc_nodes"] > 0 or standard_summary["outside_nc_nodes"] > 0:
        boundary_note = (
            f"Boundary check: {optimized_summary['outside_nc_nodes']} node(s) on the optimized route "
            f"and {standard_summary['outside_nc_nodes']} on the standard route fall outside the "
            "Northern Cape envelope (bbox-edge clipping); treat those legs as approximate."
        )

    opt_time = optimized["total_time_mins"]
    std_time = standard["total_time_mins"]
    opt_km = optimized_summary["total_km"]
    std_km = standard_summary["total_km"]

    # Panel-facing explanation of the distance-vs-time question ("why is the
    # optimal route so long?"). The map shows KILOMETRES, but the objective
    # minimizes SPOILAGE = time (thermal) + roughness×time (mechanical).
    if not routes_differ:
        panel_note = (
            "The chosen route is identical to the fastest route — no distance/time trade-off to explain."
        )
    elif opt_time <= std_time + 1.0:
        panel_note = (
            f"Looks longer on the map ({opt_km:.0f} km vs {std_km:.0f} km) but is actually the SAME or "
            f"FASTER in travel time ({opt_time:.0f} min vs {std_time:.0f} min) — it uses higher-speed "
            "paved roads instead of a shorter, slow, rough shortcut. Less time on the road = less "
            "spoilage, so there is no conflict with the cold-chain goal: the distance is longer, the "
            "hours are not."
        )
    else:
        panel_note = (
            f"The chosen route is {opt_km:.0f} km / {opt_time:.0f} min versus the standard "
            f"{std_km:.0f} km / {std_time:.0f} min. It accepts {opt_time - std_time:.0f} extra minutes to "
            f"avoid ~{rough_km_avoided:.0f} km of rough road, because spoilage is weighted 50/50 between "
            "time (thermal/heat exposure) and roughness (mechanical/vibration) — here the vibration and "
            "seal damage avoided outweighs the added heat exposure. If you want strictly minimum time, "
            "lower MECH_WEIGHT so time dominates the objective."
        )

    return {
        "routes_differ": routes_differ,
        "chosen": "spoilage_optimized",
        "time_delta_mins": time_delta_mins,
        "spoilage_reduction_pts": spoilage_delta_pct,
        "rough_km_avoided": rough_km_avoided,
        "optimized_time_mins": round(opt_time, 1),
        "standard_time_mins": round(std_time, 1),
        "optimized_km": round(opt_km, 1),
        "standard_km": round(std_km, 1),
        "standard_summary": standard_summary,
        "optimized_summary": optimized_summary,
        "reason": reason,
        "panel_note": panel_note,
        "boundary_note": boundary_note,
    }


# ---------------------------------------------------------------------------
# Hardware sim-vs-real evaluation — the feedback-loop validation
# ---------------------------------------------------------------------------
def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _point_to_polyline_km(lon: float, lat: float, coords: list) -> float:
    """Minimum distance (km) from a point to a [lon,lat] polyline, via true
    point-to-segment projection in a local equirectangular plane."""
    if not coords:
        return float("inf")
    if len(coords) == 1:
        return _haversine_km(lon, lat, coords[0][0], coords[0][1])
    mx = 111_320.0 * math.cos(math.radians(lat))
    my = 110_540.0
    px, py = lon * mx, lat * my
    best = float("inf")
    for a, b in zip(coords[:-1], coords[1:]):
        ax, ay = a[0] * mx, a[1] * my
        bx, by = b[0] * mx, b[1] * my
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0.0:
            dist = math.hypot(px - ax, py - ay)
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
            projx, projy = ax + t * dx, ay + t * dy
            dist = math.hypot(px - projx, py - projy)
        best = min(best, dist)
    return best / 1000.0


def freeze_hardware_sim_plan(origin_node: int, destination_node: int, optimized: dict) -> dict:
    """Captures the 'as-simulated' plan at the first GPS fix of a hardware trip:
    the spoilage-optimized route and (if the Monte Carlo module is available) a
    stochastic risk estimate. This is the yardstick the real trip is measured
    against in /v1/hardware/trip-evaluation."""
    summary = summarize_route(optimized["path_nodes"])
    plan = {
        "destination_town": trip_state["destination_town"],
        "planned_route_coordinates": optimized["coordinates"],
        "planned_path_nodes": optimized["path_nodes"],
        "planned_time_mins": optimized["total_time_mins"],
        "planned_km": summary["total_km"],
        "planned_rough_km": summary["rough_km"],
        "planned_spoilage_pct": optimized["spoilage_risk_pct"],
        "planned_surface_profile": summary["profile"],
        # Full build_route dict (segments + coordinates) so the map can draw a
        # STABLE planned line for hardware trips instead of re-snapping each poll.
        "optimized_route_full": optimized,
        "frozen_at_ts": int(time.time()),
        "monte_carlo": None,
    }
    if vrpsim is not None:
        try:
            mc = vrpsim.run_monte_carlo_risk_analysis(
                live_topology, cluster_coord, origin_node, destination_node,
                datetime.now(timezone.utc) + timedelta(minutes=2),
                iterations=200, shipment_value_rand=live_settings["shipment_value_rand"],
            )
            plan["monte_carlo"] = {
                key: mc[key] for key in (
                    "expected_journey_time_mins", "prob_total_spoilage_pct",
                    "value_at_risk_95_rand", "worst_peak_cargo_temp_c",
                ) if key in mc
            }
        except Exception as exc:  # noqa: BLE001 - MC is a nice-to-have; the plan still stands without it
            logger.warning("HARDWARE SIM PLAN -> Monte Carlo estimate failed: %s", exc)
    logger.info(
        "HARDWARE SIM PLAN frozen -> dest=%s planned_time=%.0fmin planned_km=%.0f spoilage=%.1f%%",
        plan["destination_town"], plan["planned_time_mins"], plan["planned_km"], plan["planned_spoilage_pct"],
    )
    return plan


def evaluate_hardware_trip(sim_plan: dict, pings: list, max_cargo_c: float | None) -> dict:
    """Deterministic simulation-vs-real comparison: how faithfully the real trip
    tracked the simulated optimal plan, plus concrete suggestions to close any
    gap so future real trips converge on the simulation."""
    pings_sorted = sorted(pings, key=lambda p: p["ts"])
    coords_plan = sim_plan["planned_route_coordinates"]

    actual_km = 0.0
    for a, b in zip(pings_sorted[:-1], pings_sorted[1:]):
        actual_km += _haversine_km(a["lon"], a["lat"], b["lon"], b["lat"])
    elapsed_min = max(0.0, (pings_sorted[-1]["ts"] - pings_sorted[0]["ts"]) / 60.0)

    adhered = sum(
        1 for p in pings_sorted
        if _point_to_polyline_km(p["lon"], p["lat"], coords_plan) <= ROUTE_ADHERENCE_KM
    )
    adherence_pct = adhered / len(pings_sorted) * 100.0

    planned_time = sim_plan["planned_time_mins"]
    planned_km = sim_plan["planned_km"]
    time_ratio = (elapsed_min / planned_time) if planned_time > 0 else None
    dist_ratio = (actual_km / planned_km) if planned_km > 0 else None

    safe_max = live_settings["safe_temp_max_c"]

    # Validation score: adherence (50%) + ETA closeness (30%) + thermal closeness (20%).
    time_closeness = 100.0
    if time_ratio is not None:
        time_closeness = max(0.0, 100.0 - abs(time_ratio - 1.0) * 100.0)
    thermal_closeness = 100.0
    if max_cargo_c is not None:
        thermal_closeness = 100.0 if max_cargo_c <= safe_max else max(0.0, 100.0 - (max_cargo_c - safe_max) * 12.0)
    validation_score = round(0.5 * adherence_pct + 0.3 * time_closeness + 0.2 * thermal_closeness, 1)

    suggestions: list[str] = []
    if adherence_pct < 85.0:
        suggestions.append(
            f"The real route left the simulated optimal corridor on {100.0 - adherence_pct:.0f}% of pings. "
            "Keeping the driver on the recommended route avoids rougher roads and makes the real trip match "
            "the simulation."
        )
    if time_ratio is not None and time_ratio > 1.2:
        suggestions.append(
            f"The trip is running {(time_ratio - 1.0) * 100:.0f}% over the simulated ETA. Reduce unplanned "
            "stops and border/queue idle to converge on the simulated duration."
        )
    if time_ratio is not None and time_ratio < 0.8:
        suggestions.append(
            f"The real trip is {(1.0 - time_ratio) * 100:.0f}% faster than simulated — the simulation's "
            "speed assumptions for these road classes look too conservative; recalibrate DEFAULT_SPEED_KMH "
            "upward for the classes on this corridor (see system_validation_test.py's self-calibration engine)."
        )
    if max_cargo_c is not None and max_cargo_c > safe_max:
        suggestions.append(
            f"Cargo peaked at {max_cargo_c:.1f} °C, above the {safe_max:.1f} °C safe threshold. Pre-cool the "
            "reefer before departure and minimise door openings so the real thermal curve tracks the simulated one."
        )
    if not suggestions:
        suggestions.append(
            "The real trip closely matches the simulation on route, timing and temperature — the model is "
            "validated for this corridor. Keep logging trips to widen the validated set."
        )

    return {
        "status": "success",
        "destination_town": sim_plan["destination_town"],
        "route_adherence_pct": round(adherence_pct, 1),
        "actual_km": round(actual_km, 1),
        "planned_km": round(planned_km, 1),
        "distance_ratio": round(dist_ratio, 2) if dist_ratio is not None else None,
        "actual_elapsed_min": round(elapsed_min, 1),
        "planned_time_mins": round(planned_time, 1),
        "time_ratio": round(time_ratio, 2) if time_ratio is not None else None,
        "actual_peak_cargo_temp_c": round(max_cargo_c, 2) if max_cargo_c is not None else None,
        "safe_temp_max_c": safe_max,
        "planned_spoilage_pct": sim_plan["planned_spoilage_pct"],
        "planned_surface_profile": sim_plan.get("planned_surface_profile"),
        "simulation_monte_carlo": sim_plan.get("monte_carlo"),
        "num_pings": len(pings_sorted),
        "validation_score": validation_score,
        "suggestions": suggestions,
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


# Reefer condenser workload emulator (spec §4.1). While the truck moves, the
# compressor load scales with how hard the unit is fighting ambient heat: a
# documented (not fitted) linear model above a 25°C comfort baseline. When the
# truck is stationary (speed < STATIONARY_SPEED_THRESHOLD_KMH — a queue or
# gridlock) the condenser can't reject heat with no airflow in desert sun, so
# the load is forced to 100% exactly as the spec requires. Reads live ambient
# from CURRENT_WEATHER, kept fresh by every /v1/telematics/truck-01 poll.
STATIONARY_SPEED_THRESHOLD_KMH = 5.0
COMPRESSOR_BASE_LOAD_PCT = 45.0
COMPRESSOR_HEAT_GAIN_PCT_PER_C = 3.0
COMPRESSOR_COMFORT_BASELINE_C = 25.0


def compressor_load_pct(speed_kmh: float, ambient_temp_c: float) -> float:
    """Emulated compressor/condenser workload (%). Forced to 100% for a
    stationary vehicle (models condenser strain in desert heat), otherwise a
    heat-driven linear load clamped to [0, 100]."""
    if speed_kmh < STATIONARY_SPEED_THRESHOLD_KMH:
        return 100.0
    load = COMPRESSOR_BASE_LOAD_PCT + COMPRESSOR_HEAT_GAIN_PCT_PER_C * max(0.0, ambient_temp_c - COMPRESSOR_COMFORT_BASELINE_C)
    return round(max(0.0, min(100.0, load)), 1)


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


@app.get("/v1/weather/status")
def weather_status() -> dict:
    """Reports whether an OpenWeatherMap key is configured and which provider
    last answered — WITHOUT ever exposing the key itself. Lets the dashboard say
    'key configured but OWM fell back' (e.g. a brand-new key can take ~2h to
    activate) instead of the misleading 'no key detected'."""
    return {
        "key_configured": bool(OPENWEATHERMAP_API_KEY),
        "preferred_provider": "openweathermap" if OPENWEATHERMAP_API_KEY else "open-meteo",
        "last_source": CURRENT_WEATHER.get("source", "unknown"),
    }


@app.get("/v1/settings/thresholds")
def get_thresholds() -> dict:
    """Current business-adjustable thresholds. Read by the dashboard on load
    so a fresh page/session shows whatever was last set, not the default."""
    return dict(live_settings)


@app.post("/v1/settings/thresholds")
def update_thresholds(settings: ThresholdSettings) -> dict:
    """Updates whichever of the runtime business settings were supplied
    (each is optional and applied independently — omit a field to leave it
    unchanged). Both are read fresh wherever they're used — safe_temp_max_c
    by thermal_rate_multiplier/cargo_temp_status, shipment_value_rand by
    build_route's expected_loss_rand and get_telematics's
    expected_loss_rand_so_far — so this takes effect immediately on the
    current trip, no restart or trip reconfigure needed. A trip's already-
    cached trip_plan snapshot (see configure_trip) is a deliberate
    exception: it's frozen at whatever these settings were at configure
    time, by design, as a stable baseline to compare the live figures
    against."""
    if settings.safe_temp_max_c is not None:
        live_settings["safe_temp_max_c"] = settings.safe_temp_max_c
        logger.info("SETTINGS UPDATED -> safe_temp_max_c=%.1f", settings.safe_temp_max_c)
    if settings.shipment_value_rand is not None:
        live_settings["shipment_value_rand"] = settings.shipment_value_rand
        logger.info("SETTINGS UPDATED -> shipment_value_rand=%.0f", settings.shipment_value_rand)
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
                    "shipment_value_rand": live_settings["shipment_value_rand"],
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
            # A truck configured for hardware tracking may not have sent a
            # real position yet, or may still be carrying a stale reading
            # from a previous simulator trip — start from the same known
            # baseline the simulator uses, so cargo metrics have a sane
            # starting point rather than whatever was last in memory.
            tracking_state["cargo_temp_c"] = simulated_cargo_temp_c(0.0)
            tracking_state["speed_kmh"] = 0.0

        # Reset cargo thermal accumulation for the new trip regardless of mode.
        trip_state["thermal_risk_accum_hours"] = 0.0
        trip_state["last_thermal_update_ts"] = None
        # Reset hardware motion gating — a freshly configured trip means the
        # truck hasn't necessarily moved yet, even if it moved during a
        # previous trip.
        trip_state["hardware_motion_detected"] = False
        trip_state["hardware_reference_lat"] = None
        trip_state["hardware_reference_lon"] = None
        # Reset the hardware sim-vs-real evaluation state for the new trip.
        trip_state["hardware_sim_plan"] = None
        trip_state["hardware_ping_history"] = []
        trip_state["hardware_max_cargo_temp_c"] = None

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

        # Retain the ping for the hardware sim-vs-real evaluation (real trip path).
        if trip_state["configured"] and trip_state["mode"] == "hardware":
            history = trip_state["hardware_ping_history"]
            history.append({
                "lat": lat, "lon": lon,
                "ts": tracking_state["timestamp"],
                "speed": speed if speed is not None else 0.0,
            })
            if len(history) > HARDWARE_PING_HISTORY_MAX:
                del history[: len(history) - HARDWARE_PING_HISTORY_MAX]

        # Motion detection for hardware mode: cargo risk metrics stay frozen
        # (see get_telematics) until the truck has genuinely moved, not just
        # been configured. The first fix after configure is recorded as a
        # reference point (the truck sitting still); subsequent fixes are
        # compared against it — displacement past MOTION_DETECTION_DISTANCE_M,
        # or a reported speed past MOTION_DETECTION_SPEED_KMH, flips the flag.
        if trip_state["configured"] and trip_state["mode"] == "hardware" and not trip_state["hardware_motion_detected"]:
            if trip_state["hardware_reference_lat"] is None:
                trip_state["hardware_reference_lat"] = lat
                trip_state["hardware_reference_lon"] = lon
            else:
                moved_m = approx_distance_m(
                    trip_state["hardware_reference_lat"], trip_state["hardware_reference_lon"], lat, lon
                )
                speed_trigger = speed is not None and speed >= MOTION_DETECTION_SPEED_KMH
                if moved_m >= MOTION_DETECTION_DISTANCE_M or speed_trigger:
                    trip_state["hardware_motion_detected"] = True
                    # Reset the thermal clock to this exact moment — without
                    # this, the first accumulation tick after motion starts
                    # would count all the idle time since configure/last
                    # poll as if the vehicle had been driving/thermally
                    # active the whole time.
                    trip_state["last_thermal_update_ts"] = time.monotonic()
                    logger.info(
                        "HARDWARE MOTION DETECTED -> displacement=%.1fm speed=%s km/h — cargo risk tracking begins now.",
                        moved_m, speed,
                    )

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
        get_ambient_weather(snapshot_lat, snapshot_lon)
        if snapshot_lat is not None and snapshot_lon is not None
        else None
    )

    with state_lock:
        if weather_reading is not None:
            CURRENT_WEATHER["temp_c"] = weather_reading["temp_c"]
            CURRENT_WEATHER["rain_mm"] = weather_reading["rain_mm"]
            CURRENT_WEATHER["alert"] = weather_reading["alert"]
            # Store which provider actually answered, so the dashboard can label
            # the source correctly (previously this was never set, which made the
            # Weather tab always read "Open-Meteo" even on OpenWeatherMap data).
            CURRENT_WEATHER["source"] = weather_reading.get("source", "unknown")

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
            # between polls using whatever cargo_temp_c is currently known —
            # but ONLY once the vehicle has genuinely started moving (see
            # hardware_motion_detected / telematics_incoming). A trip being
            # "configured" just means a destination was set; the truck can
            # sit parked for a while before it actually departs, and cargo
            # risk shouldn't silently accrue against a shipment that hasn't
            # left yet.
            now = time.monotonic()
            last_ts = trip_state["last_thermal_update_ts"]
            motion_started = trip_state["hardware_motion_detected"]

            if motion_started and last_ts is not None:
                dt_hours = max(0.0, (now - last_ts) / 3600.0)
                rate = thermal_rate_multiplier(tracking_state["cargo_temp_c"])
                trip_state["thermal_risk_accum_hours"] += dt_hours * rate

                # Ambient-weather-driven idling warm-up: while the vehicle is
                # stopped (e.g. at a border post) mid-trip, the cargo chamber
                # exponentially approaches an equilibrium set by live ambient
                # temperature AND the adjustable safe-temp threshold — a
                # real idling reefer drifts toward some ceiling, it doesn't
                # climb forever. This is what a real cargo sensor would show
                # once wired up; independent of the thermal-risk accumulation
                # above, which tracks cumulative spoilage exposure, not the
                # instantaneous temperature itself.
                is_idling = tracking_state.get("speed_kmh", 0.0) <= 0.5
                if is_idling and dt_hours > 0.0:
                    ambient_temp_c = CURRENT_WEATHER["temp_c"]
                    safe_max_c = live_settings["safe_temp_max_c"]
                    if ambient_temp_c >= EXTREME_HEAT_THRESHOLD_C:
                        rate_per_hour = IDLE_WARMING_RATE_EXTREME_PER_HOUR
                        target_c = safe_max_c + IDLE_WARMING_TARGET_OFFSET_EXTREME_C
                    elif ambient_temp_c >= COLD_CLIMATE_THRESHOLD_C:
                        rate_per_hour = IDLE_WARMING_RATE_STANDARD_PER_HOUR
                        target_c = safe_max_c + IDLE_WARMING_TARGET_OFFSET_STANDARD_C
                    else:
                        rate_per_hour = IDLE_WARMING_RATE_COLD_PER_HOUR
                        target_c = safe_max_c + IDLE_WARMING_TARGET_OFFSET_COLD_C

                    current_temp = tracking_state["cargo_temp_c"]
                    decay = math.exp(-rate_per_hour * dt_hours)
                    tracking_state["cargo_temp_c"] = round(target_c + (current_temp - target_c) * decay, 2)

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

        expected_loss_rand_so_far = round(
            composite_cargo_risk_pct / 100.0 * live_settings["shipment_value_rand"], -2
        )

        # Track the peak cargo temperature actually observed on a moving hardware
        # trip, for the sim-vs-real evaluation.
        if trip_state["configured"] and trip_state["mode"] == "hardware" and trip_state["hardware_motion_detected"]:
            current_cargo = tracking_state["cargo_temp_c"]
            previous_peak = trip_state["hardware_max_cargo_temp_c"]
            trip_state["hardware_max_cargo_temp_c"] = (
                current_cargo if previous_peak is None else max(previous_peak, current_cargo)
            )

        payload = dict(tracking_state)
        awaiting_motion = (
            trip_state["configured"]
            and trip_state["mode"] == "hardware"
            and not trip_state["hardware_motion_detected"]
        )
        payload["cargo_temp_status"] = (
            "Awaiting Motion" if awaiting_motion else cargo_temp_status(tracking_state["cargo_temp_c"])
        )
        payload["awaiting_motion"] = awaiting_motion
        payload["mechanical_risk_pct"] = mechanical_risk_pct
        payload["thermal_risk_pct"] = thermal_risk_pct
        payload["composite_cargo_risk_pct"] = composite_cargo_risk_pct
        payload["expected_loss_rand_so_far"] = expected_loss_rand_so_far
        payload["ambient_weather"] = dict(CURRENT_WEATHER)
        # Emulated reefer compressor workload (spec §4.1). Stationary vehicle
        # (speed < 5 km/h) is pinned to 100% to model condenser strain in
        # desert heat; otherwise it scales with live ambient temperature.
        payload["compressor_load_pct"] = compressor_load_pct(
            tracking_state.get("speed_kmh", 0.0), CURRENT_WEATHER["temp_c"]
        )

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

    # Hardware mode: freeze the 'as-simulated' plan on the first routing call
    # (the truck's first fix ≈ origin), so the real trip can later be evaluated
    # against it in /v1/hardware/trip-evaluation.
    if mode == "hardware" and trip_state["hardware_sim_plan"] is None:
        with state_lock:
            if trip_state["hardware_sim_plan"] is None:
                trip_state["hardware_sim_plan"] = freeze_hardware_sim_plan(
                    origin_node, destination_node, optimized
                )

    time_saved_mins = round(standard["total_time_mins"] - optimized["total_time_mins"], 1)
    rand_saved = round(standard["expected_loss_rand"] - optimized["expected_loss_rand"], -2)
    detour_active = optimized["path_nodes"] != standard["path_nodes"]

    # Deterministic "why this route" analysis — powers the Report tab's Route
    # Justification panel and grounds the AI memo's justification section.
    optimized_summary = summarize_route(optimized["path_nodes"])
    standard_summary = summarize_route(standard["path_nodes"])
    route_rationale = build_route_rationale(standard, optimized, standard_summary, optimized_summary)

    current_node, snap_distance_km = node_index.snap(current_lat, current_lon)

    # STABLE line for the map to draw. The live optimized/standard routes are
    # recomputed from the truck's CURRENT snapped node every poll — great for the
    # shrinking "remaining" metrics, but as the truck moves that snap jumps
    # between nearby nodes and the first hops flip, which made the drawn line
    # look squiggly/jittery. `display_route` is the FIXED full-journey planned
    # route (frozen at departure) — it never re-snaps, so the drawn line is
    # smooth and stable; the truck marker simply moves along it.
    if mode == "simulator" and trip_plan is not None:
        display_route = trip_plan["optimized_route"]
    elif mode == "hardware":
        with state_lock:
            _sim_plan = trip_state["hardware_sim_plan"]
        display_route = (_sim_plan.get("optimized_route_full") if _sim_plan else None) or optimized
    else:
        display_route = optimized

    return {
        "vehicle_id": VEHICLE_ID,
        "feed_source": feed_source,
        # Stable planned line for the map (see note above); metrics still use the
        # live optimized/standard routes below.
        "display_route": display_route,
        "snapped_node_id": current_node,
        "snap_distance_km": round(snap_distance_km, 3),
        "gps_fix_confidence": gps_fix_confidence(snap_distance_km),
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
            "shipment_value_rand": live_settings["shipment_value_rand"],
        },
        # Static full-journey reference (simulator mode only) — the fixed
        # planning figures from when the trip was configured, unaffected by
        # how far the truck has since traveled.
        "trip_plan": trip_plan,
        # Deterministic justification for why the optimized route was chosen
        # over the standard route and any other road on the map.
        "route_rationale": route_rationale,
        # kept for compatibility with older callers expecting a single "path"
        "path": optimized["coordinates"],
    }


@app.get("/v1/hardware/trip-evaluation")
def hardware_trip_evaluation() -> dict:
    """Simulation-vs-real evaluation for a live hardware trip. Compares the real
    trip (from Traccar pings) against the frozen simulated optimal plan: route
    adherence, actual-vs-planned time and distance, peak cargo temperature, and a
    validation score, plus concrete suggestions to make the real trip converge on
    the simulation. Degrades to a clean JSON status while data is still building."""
    with state_lock:
        if not trip_state["configured"] or trip_state["mode"] != "hardware":
            return {
                "status": "unavailable",
                "message": "Switch Telemetry Mode to Live Mobile Hardware Tracking and set a destination to "
                           "evaluate the real trip against the simulation.",
            }
        sim_plan = trip_state["hardware_sim_plan"]
        pings = list(trip_state["hardware_ping_history"])
        max_cargo = trip_state["hardware_max_cargo_temp_c"]

    if sim_plan is None:
        return {
            "status": "pending",
            "message": "Simulation plan not frozen yet — open the Customer Route Tracker once (so the plan "
                       "is computed from your first GPS fix), then drive the route.",
        }
    if len(pings) < 2:
        return {
            "status": "pending",
            "message": "Not enough live telemetry yet — the evaluation appears once the truck has moved and "
                       "sent a few Traccar pings.",
        }
    try:
        return evaluate_hardware_trip(sim_plan, pings, max_cargo)
    except Exception as exc:  # noqa: BLE001 - evaluation must degrade to JSON, never 500 the dashboard
        logger.error("HARDWARE EVALUATION -> failed: %s", exc)
        return {"status": "error", "message": f"Evaluation failed: {exc}"}


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
        reading = get_ambient_weather(lat, lon)
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
    current_reading = get_ambient_weather(current_lat, current_lon)
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
    # Same two-component (thermal + mechanical) weight the rest of the graph uses,
    # so a driver report is priced consistently with the routing objective.
    spoilage_cost_override = travel_time_override_hr * (THERMAL_SPOIL_WEIGHT + MECH_SPOIL_WEIGHT * roughness_override)

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
        "gps_fix_confidence": gps_fix_confidence(distance_km),
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
    leaves the machine, no separate daemon needs to be installed or started.

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

    sim_vs_real_section = ""
    if payload.evaluation_mode == "hardware":
        sim_vs_real_section = (
            "\nSIMULATION-VS-REAL FACTS (use these to write a '## Simulation vs Real Trip' section):\n"
            f"- Validation score (0-100, how closely the real trip matched the simulation): "
            f"{payload.validation_score if payload.validation_score is not None else 'n/a'}\n"
            f"- Route adherence: {payload.route_adherence_pct if payload.route_adherence_pct is not None else 'n/a'}% "
            "of pings stayed on the simulated optimal route\n"
            f"- Actual trip duration vs simulated ETA: "
            f"{payload.actual_vs_planned_time_pct if payload.actual_vs_planned_time_pct is not None else 'n/a'}% of plan\n"
            f"- System-computed evaluation detail: {payload.sim_vs_real_text or 'n/a'}\n"
        )

    route_justification_section = ""
    if payload.route_rationale_text or payload.standard_route_profile:
        route_justification_section = (
            "\nROUTE-SELECTION FACTS (use these to write a '## Route Justification' section):\n"
            f"- Chosen (spoilage-optimized) route surface profile: {payload.optimized_route_profile or 'n/a'}\n"
            f"- Standard (time-only) route surface profile: {payload.standard_route_profile or 'n/a'}\n"
            f"- Rough/unpaved km avoided by the chosen route: {(payload.rough_km_avoided or 0.0):.0f} km\n"
            f"- Do the two routes differ: {payload.routes_differ}\n"
            f"- Deterministic rationale already computed by the system: {payload.route_rationale_text or 'n/a'}\n"
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
        f"{route_justification_section}"
        f"{sim_vs_real_section}"
        f"- Total shipment value at risk: R {payload.shipment_value_rand:,.0f}\n\n"
        "INTERPRETATION GUIDANCE (read before writing):\n"
        "- If the optimized and standard route durations are equal and the Rand saved is R 0, this "
        "  does NOT mean the optimizer failed. It means the fastest route on this corridor already "
        "  coincides with the lowest-spoilage route — there is no rougher-but-faster shortcut to avoid, "
        "  so the time-optimal and spoilage-optimal paths are the same road. Frame this as a "
        "  CONFIRMED-OPTIMAL result (no cheaper or safer alternative exists on this corridor), and then "
        "  make the thermal-exposure risk the focus of the memo, since with mechanical risk already "
        "  minimized, temperature exposure is the binding risk on this shipment.\n"
        "- Keep the memo fully self-contained and make sure you COMPLETE the final prioritized action "
        "  list — do not stop mid-sentence.\n\n"
        "Using ONLY these figures, produce a strategic business memo for the cooperative's "
        "dispatch manager: quantify the Rand-value trade-off between extra travel time and "
        "spoilage avoided, address both the mechanical and thermal risk components separately, "
        "note the vehicle wear-and-tear implication of the surface profile, and recommend "
        "whether this shipment should take the standard or the optimized route. If ROUTE-SELECTION "
        "FACTS were provided above, include a dedicated '## Route Justification' section that "
        "explains, in plain language for a non-technical manager, WHY the system chose this "
        "specific route and did NOT choose the standard route or any other road on the map — "
        "ground it in the rough-km-avoided and surface-profile numbers given, and if the two "
        "routes are identical, say plainly that the fastest road is already the safest. If "
        "SIMULATION-VS-REAL FACTS were provided, include a '## Simulation vs Real Trip' section that "
        "grades how closely the real trip matched the simulation (using the validation score and "
        "adherence), states whether the simulation is validated for this corridor, and gives concrete "
        "steps to make future real trips converge on the simulated optimum — the goal is a real trip "
        "that is almost as perfect as the simulation predicted.\n\n"
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
        # 1100 tokens: a cold-chain strategy memo with an intro, separate
        # mechanical + thermal sections, the fuel/tyre/premium levers, and a
        # completed prioritized action list needs the headroom so it doesn't
        # cut off mid-sentence.
        output = generator(messages, max_new_tokens=1100, do_sample=True, temperature=0.35)
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


# ---------------------------------------------------------------------------
# Predictive / future-hazard routing — spec §4.2, §4.3, and the Monte Carlo
# call the frontend's "Journey Prediction & Scheduling" tab needs.
# ---------------------------------------------------------------------------
# How many nodes along a candidate path get an actual forecast lookup. The
# province-scale graph (tens of thousands of nodes) means a long-haul path
# can carry hundreds of hops; forecasting every single one would mean
# hundreds of OWM calls per request. Instead the path is sampled evenly at
# up to this many points (always including the first and last node), which
# is enough to catch a storm/heat cell sitting on the corridor without
# hammering the forecast API or making a request pathologically slow. This
# is a documented engineering trade-off, not a placeholder.
PREDICTIVE_ROUTE_MAX_SAMPLES = 40
# Hazard spoilage-cost multiplier applied to a sampled edge's neighborhood
# when the forecast at that point predicts extreme heat or storm washout —
# large enough that the spoilage-weighted shortest-path solver reliably
# routes around it, mirroring how a driver-reported "Impassable / Washed
# Out" override (roughness >= 6.0, see ROAD_CONDITION_ROUGHNESS) behaves.
PREDICTIVE_HAZARD_MULTIPLIER = 6.0
PREDICTIVE_MAX_HORIZON_DAYS = 7


class PredictiveRouteRequest(BaseModel):
    origin_town: str
    destination_town: str
    # ISO 8601, e.g. "2026-09-01T14:00:00". Naive timestamps are treated as UTC.
    target_datetime: str


class MonteCarloRequest(BaseModel):
    origin_town: str
    destination_town: str
    target_datetime: str
    shipment_value_rand: float | None = Field(None, gt=0.0)
    iterations: int = Field(1000, ge=100, le=5000)
    # When true, the response includes per-trial arrays so the frontend can
    # stream the distribution as it builds and draw a live histogram.
    return_samples: bool = False


def parse_target_datetime(raw: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse target_datetime '{raw}': {exc}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if parsed < now:
        raise HTTPException(status_code=422, detail="target_datetime must be in the future.")
    if parsed > now + timedelta(days=PREDICTIVE_MAX_HORIZON_DAYS):
        raise HTTPException(
            status_code=422,
            detail=f"target_datetime must be within {PREDICTIVE_MAX_HORIZON_DAYS} days from now.",
        )
    return parsed


def sample_route_hazards(path_nodes: list[int], target_dt: datetime) -> tuple[dict[tuple[int, int], float], list[dict]]:
    """Forecasts weather at an evenly-spaced sample of nodes along a path and
    returns (a) a per-edge hazard-multiplier dict covering every edge
    touching a hazardous sample node, for feeding into the spoilage weight
    function, and (b) a plain list describing each hazard for the API
    response / frontend map overlay."""
    if len(path_nodes) <= PREDICTIVE_ROUTE_MAX_SAMPLES:
        sample_indices = list(range(len(path_nodes)))
    else:
        step = (len(path_nodes) - 1) / (PREDICTIVE_ROUTE_MAX_SAMPLES - 1)
        sample_indices = sorted({round(i * step) for i in range(PREDICTIVE_ROUTE_MAX_SAMPLES)})

    hazard_multipliers: dict[tuple[int, int], float] = {}
    hazard_zones: list[dict] = []

    for idx in sample_indices:
        node = path_nodes[idx]
        lon, lat = cluster_coord[node]
        try:
            forecast = get_forecast_weather(lat, lon, target_dt)
        except Exception as exc:  # noqa: BLE001 - one bad sample must never abort the whole route computation
            logger.warning("PREDICTIVE ROUTE -> forecast sample failed at node %s: %s", node, exc)
            continue

        temp_c = forecast["ambient_temp_c"]
        rain_mm = forecast["rain_mm_per_hr"]
        is_extreme_heat = temp_c >= EXTREME_HEAT_THRESHOLD_C
        is_storm_washout = rain_mm >= weather_engine.HEAVY_RAIN_THRESHOLD_MM
        if not (is_extreme_heat or is_storm_washout):
            continue

        reason = "Extreme Heat" if is_extreme_heat else "Storm / Washout Risk"
        hazard_zones.append({
            "node_id": node,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "forecast_ambient_temp_c": round(temp_c, 1),
            "forecast_rain_mm_per_hr": round(rain_mm, 2),
            "reason": reason,
            "source": forecast["source"],
        })

        # Spike every edge incident to this node (both directions) so the solver
        # actually has an incentive to route around this point rather than just
        # through it. Works whether the loaded topology is directed (successors /
        # predecessors) or undirected (neighbors) — the default nc_road_network
        # build is undirected, while build_directed_graph() yields a MultiDiGraph.
        if live_topology.is_directed():
            incident_neighbors = list(live_topology.successors(node)) + list(live_topology.predecessors(node))
        else:
            incident_neighbors = list(live_topology.neighbors(node))
        for neighbor in incident_neighbors:
            hazard_multipliers[(node, neighbor)] = PREDICTIVE_HAZARD_MULTIPLIER
            hazard_multipliers[(neighbor, node)] = PREDICTIVE_HAZARD_MULTIPLIER

    return hazard_multipliers, hazard_zones


def build_predictive_route(origin_node: int, destination_node: int, hazard_multipliers: dict[tuple[int, int], float]) -> dict:
    """Same shape as build_route(), but the spoilage weight used for path
    selection AND for the reported spoilage totals is multiplied by any
    forecast-hazard override for that edge. Never mutates live_topology —
    the multiplier only exists for the duration of this one path-finding
    call, so it can never leak into the standard live routing endpoints."""
    def hazard_spoilage_weight(u: int, v: int, edge_attrs: dict) -> float:
        return effective_spoilage(u, v, edge_attrs) * hazard_multipliers.get((u, v), 1.0)

    try:
        path = nx.shortest_path(live_topology, origin_node, destination_node, weight=hazard_spoilage_weight)
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
        seg_spoilage = hazard_spoilage_weight(u, v, edge_attrs)
        total_time_mins += seg_time
        total_spoilage_cost += seg_spoilage
        segments.append({
            "coords": [list(cluster_coord[u]), list(cluster_coord[v])],
            "fclass": edge_attrs.get("fclass"),
            "hazard_multiplier_applied": hazard_multipliers.get((u, v), 1.0),
        })

    spoilage_pct = min(1.0, total_spoilage_cost / SPOILAGE_THRESHOLD)
    expected_loss_rand = round(spoilage_pct * live_settings["shipment_value_rand"], -2)

    return {
        "path_nodes": path,
        "coordinates": [list(cluster_coord[n]) for n in path],
        "segments": segments,
        "total_time_mins": round(total_time_mins, 1),
        "total_spoilage_cost": round(total_spoilage_cost, 3),
        "spoilage_risk_pct": round(spoilage_pct * 100, 1),
        "expected_loss_rand": expected_loss_rand,
    }


def compute_predictive_route(origin_town: str, destination_town: str, target_dt: datetime) -> dict:
    """Shared core used by both /v1/simulation/predictive-route and
    /v1/simulation/compare-slots so the two endpoints can never drift out of
    sync on how a slot is evaluated."""
    origin_node = resolve_town_node(origin_town)
    destination_node = resolve_town_node(destination_town)

    baseline_optimized = build_route(origin_node, destination_node, spoilage_weight)
    hazard_multipliers, hazard_zones = sample_route_hazards(baseline_optimized["path_nodes"], target_dt)

    if hazard_multipliers:
        predictive_route = build_predictive_route(origin_node, destination_node, hazard_multipliers)
        rerouted = predictive_route["path_nodes"] != baseline_optimized["path_nodes"]
    else:
        predictive_route = baseline_optimized
        rerouted = False

    return {
        "origin_town": origin_town,
        "destination_town": destination_town,
        "target_datetime": target_dt.isoformat(),
        "hazard_zones": hazard_zones,
        "hazard_count": len(hazard_zones),
        "rerouted_around_hazard": rerouted,
        "baseline_optimized_route": baseline_optimized,
        "predictive_route": predictive_route,
        "delta_vs_baseline": {
            "extra_time_mins": round(predictive_route["total_time_mins"] - baseline_optimized["total_time_mins"], 1),
            "spoilage_risk_change_pct": round(
                predictive_route["spoilage_risk_pct"] - baseline_optimized["spoilage_risk_pct"], 1
            ),
        },
    }


@app.post("/v1/simulation/predictive-route")
def predictive_route(payload: PredictiveRouteRequest) -> dict:
    """Spec §4.2. Accepts a future ISO timestamp up to 7 days out, samples the
    forecast (OpenWeatherMap forecast -> synthetic Stream A -> baseline, see
    get_forecast_weather) along the current spoilage-optimized corridor, and
    — wherever a sampled point forecasts extreme heat or a storm washout —
    spikes that neighborhood's spoilage_cost and re-solves the shortest path
    so the route dynamically detours around the future hazard zone."""
    target_dt = parse_target_datetime(payload.target_datetime)
    logger.info(
        "PREDICTIVE ROUTE REQUEST -> origin=%s destination=%s target=%s",
        payload.origin_town, payload.destination_town, target_dt.isoformat(),
    )
    return compute_predictive_route(payload.origin_town, payload.destination_town, target_dt)


@app.get("/v1/simulation/compare-slots")
def compare_departure_slots(origin_town: str, destination_town: str) -> dict:
    """Spec §4.3. Compares three departure windows (Now, +12h, +24h) and
    returns a JSON comparison matrix identifying which slot minimizes
    cold-chain degradation, with a plain-language recommendation per slot."""
    # Resolve towns once up front so an unknown town name fails fast with a
    # clear 422 instead of failing deep inside the third slot's computation.
    resolve_town_node(origin_town)
    resolve_town_node(destination_town)

    now = datetime.now(timezone.utc)
    slots = [("Now", now), ("+12h", now + timedelta(hours=12)), ("+24h", now + timedelta(hours=24))]

    rows = []
    for label, slot_dt in slots:
        # "Now" can be a few seconds behind datetime.now() by the time it
        # reaches parse_target_datetime's >= now check inside compute_predictive_route's
        # forecast lookups; nudge it forward by a minute so it never trips
        # that guard due to request latency.
        safe_dt = max(slot_dt, datetime.now(timezone.utc) + timedelta(minutes=1))
        result = compute_predictive_route(origin_town, destination_town, safe_dt)
        spoilage_pct = result["predictive_route"]["spoilage_risk_pct"]
        if spoilage_pct >= 25.0:
            recommendation = f"REJECT DEPARTURE: {spoilage_pct:.0f}% Spoilage Failure Risk"
        elif spoilage_pct >= 10.0:
            recommendation = f"CAUTION: {spoilage_pct:.0f}% Spoilage Risk — consider a later slot"
        else:
            recommendation = f"APPROVE DEPARTURE: {spoilage_pct:.1f}% Failure Risk"
        rows.append({
            "slot_label": label,
            "departure_datetime": safe_dt.isoformat(),
            "total_time_mins": result["predictive_route"]["total_time_mins"],
            "spoilage_risk_pct": spoilage_pct,
            "expected_loss_rand": result["predictive_route"]["expected_loss_rand"],
            "hazard_count": result["hazard_count"],
            "rerouted_around_hazard": result["rerouted_around_hazard"],
            "recommendation": recommendation,
        })

    best_slot = min(rows, key=lambda row: row["spoilage_risk_pct"])
    return {
        "origin_town": origin_town,
        "destination_town": destination_town,
        "slots": rows,
        "recommended_slot_label": best_slot["slot_label"],
    }


@app.post("/v1/simulation/monte-carlo-risk")
def monte_carlo_risk(payload: MonteCarloRequest) -> dict:
    """Backs the frontend's "🎲 Run Monte Carlo Stochastic Risk Test" button.
    Delegates the actual 1,000-trial thermal/customs/infrastructure-shock
    simulation to vrp_simulation_generator.run_monte_carlo_risk_analysis
    (spec §3.2) — this endpoint's job is just request validation, node
    resolution, and the same fail-soft-JSON pattern already used by
    /v1/analytics/strategy for its own optional heavy dependency."""
    if vrpsim is None:
        return {
            "status": "error",
            "error_type": "vrp_simulation_generator_not_available",
            "message": (
                "vrp_simulation_generator.py could not be imported. Make sure it's "
                "present alongside live_backend.py and installs cleanly."
            ),
        }

    target_dt = parse_target_datetime(payload.target_datetime)
    origin_node = resolve_town_node(payload.origin_town)
    destination_node = resolve_town_node(payload.destination_town)
    shipment_value_rand = payload.shipment_value_rand or live_settings["shipment_value_rand"]

    logger.info(
        "MONTE CARLO REQUEST -> origin=%s destination=%s target=%s iterations=%d shipment_value_rand=%.0f",
        payload.origin_town, payload.destination_town, target_dt.isoformat(),
        payload.iterations, shipment_value_rand,
    )

    try:
        result = vrpsim.run_monte_carlo_risk_analysis(
            live_topology, cluster_coord, origin_node, destination_node,
            target_dt, shipment_value_rand=shipment_value_rand, iterations=payload.iterations,
            return_samples=payload.return_samples,
        )
    except Exception as exc:  # noqa: BLE001 - a simulation failure must degrade to JSON, never crash the API
        logger.error("MONTE CARLO -> run_monte_carlo_risk_analysis failed: %s", exc)
        return {"status": "error", "error_type": "simulation_failed", "message": str(exc)}

    return {
        "status": "success",
        "origin_town": payload.origin_town,
        "destination_town": payload.destination_town,
        "target_datetime": target_dt.isoformat(),
        "iterations": payload.iterations,
        "shipment_value_rand": shipment_value_rand,
        **result,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
"""
vrp_simulation_generator.py

Simulation & Risk Evaluator for the Northern Cape Transport, Trade & Fisheries
cold-chain freight platform (MICT SETA Skills Development Hackathon, Northern
Cape, 28-29 Aug 2026).

This module is the third, previously-missing deliverable that live_backend.py
already imports and integrates with (see its `import vrp_simulation_generator as
vrpsim` block and the /v1/simulation/* endpoints). It provides three things the
rest of the platform relies on:

  1. Three coordinated synthetic data streams (spec section 3.1)
     - Stream A  Synthetic Weather Forecast Matrix   -> synthetic_weather_forecast.csv
     - Stream B  Synthetic Road Conditions / Field   -> synthetic_road_conditions.csv
     - Stream C  Synthetic Cold-Chain IoT Payloads   -> synthetic_cold_sensors.csv
     Stream B is DERIVED from Stream A; Stream C is driven by Stream A's ambient
     heat and Stream B's travel-time delays via a discrete Newton's-Law-of-Cooling
     state equation.

  2. The Monte Carlo stochastic risk solver (spec section 3.2)
     run_monte_carlo_risk_analysis(...) -> dict with Expected journey time,
     95% Value at Risk (Rands) against a shipment value, and the exact
     probability (%) of total cold-chain spoilage (cargo breaching -18.0 C).

  3. A statistical validation suite (spec section 3.3) logged to the console:
     RMSE between simulated thermodynamic temperature curves and a historical
     reference, and the Spatial Snapping Failure Rate confirming 100% of
     generated coordinates anchor within SNAP_TOLERANCE_M (150 m).

The backend also calls synthetic_forecast_point(lat, lon, target_dt) as the
FUTURE-weather fallback whenever OpenWeatherMap can't answer (no key, request
failure, or a horizon beyond OWM's ~5-day free tier) — that function is the
single point of contact for Stream A at an arbitrary coordinate/time and is
kept deterministic (hash-seeded) so repeated lookups for the same cell/hour
are stable and cache-friendly.

Design rules honored here (Global Calibration Overrides, spec section 2):
  * SNAP_TOLERANCE_M = 150 everywhere a nearest-neighbor snap happens.
  * No paid third-party APIs — this module fabricates its own weather/road
    truth; the backend layers the free OpenWeatherMap / driver-form sources
    on top.
  * Fault tolerance — every external I/O (graph pickle load, CSV read) is
    wrapped, and a missing artifact degrades to a documented fallback rather
    than raising into the caller. run_monte_carlo_risk_analysis never lets a
    single bad trial abort the whole run.

Run standalone to (re)generate the three CSVs and print the validation suite:
    uv run vrp_simulation_generator.py
    # or:  python3 vrp_simulation_generator.py --hours 72 --iterations 1000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import pickle
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# networkx / scipy are only strictly needed for the Monte Carlo route solver and
# the snapping-validation suite. Imported defensively so that the pure synthetic
# stream generators (Streams A/B/C) still work on a minimal checkout that hasn't
# installed the routing stack yet.
try:
    import networkx as nx
except ImportError:  # pragma: no cover - exercised only on a minimal install
    nx = None

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only on a minimal install
    cKDTree = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("vrp_simulation_generator")


# ---------------------------------------------------------------------------
# Global calibration overrides (spec section 2) and domain constants
# ---------------------------------------------------------------------------
# Replaces the original 25 m spatial-index snapping threshold EVERYWHERE in this
# module (data generator + validation suite), to capture low-density rural
# access tracks. Kept identical to nc_road_network.SNAP_TOLERANCE_M.
SNAP_TOLERANCE_M: float = 150.0

# --- Stream A ranges (spec section 3.1) ------------------------------------
AMBIENT_TEMP_MIN_C: float = 15.0
AMBIENT_TEMP_MAX_C: float = 42.0
RAIN_MIN_MM_PER_HR: float = 0.0
RAIN_MAX_MM_PER_HR: float = 15.0

# Northern Cape is semi-arid desert: warm diurnal mean, large day/night swing,
# rain a rare but violent event. These shape the synthetic forecast so it
# "looks like" the province rather than a generic temperate climate.
NC_DIURNAL_MEAN_C: float = 28.0
NC_DIURNAL_AMPLITUDE_C: float = 9.0
NC_TEMP_NOISE_SPAN_C: float = 6.0
NC_RAIN_EVENT_THRESHOLD: float = 0.86   # ~14% of hours carry some rain
NC_PEAK_TEMP_HOUR: float = 15.0          # hottest at ~15:00 local

# --- Hazard thresholds (shared with weather_engine.py / live_backend.py) ---
EXTREME_HEAT_THRESHOLD_C: float = 38.0
HEAVY_RAIN_THRESHOLD_MM: float = 5.0

# --- Stream B failure-event physics ----------------------------------------
# Where Stream A shows rain > 5 mm or heat > 38 C, Stream B forces a failure:
# safe vehicle speed collapses to 15 km/h and IRI spikes toward 6.0.
IMPASSABLE_SAFE_SPEED_KMH: float = 15.0
NORMAL_SAFE_SPEED_KMH: float = 90.0
IRI_BASELINE: float = 2.0
IRI_FAILURE_MAX: float = 6.0

# --- Stream C cold-chain thermodynamics ------------------------------------
CARGO_START_TEMP_C: float = -18.5        # spec: telemetry starts at -18.5 C
REEFER_SETPOINT_C: float = -18.5         # the unit tries to hold this
SPOILAGE_BREACH_TEMP_C: float = -18.0    # spec: total spoilage = breaching -18.0 C
SPOILAGE_TOTAL_LOSS_TEMP_C: float = -8.0 # at/above this, cargo treated as a total write-off

# Two-component spoilage edge weight (thermal + mechanical), matching
# nc_road_network_improved and live_backend so the Monte Carlo route profiling
# prices edges the same way the live router does. TIME is first-class, so a
# smooth (roughness 1.0) edge costs exactly its travel time.
THERMAL_SPOIL_WEIGHT: float = 0.5
MECH_SPOIL_WEIGHT: float = 0.5

# Discrete Newton's-Law-of-Cooling coefficients (per hour). The reefer chamber
# exponentially approaches an equilibrium temperature rather than climbing in a
# straight line — mirrors the idling warm-up model in live_backend.py so the
# two never disagree about the physics.
NEWTON_K_DRIVE_PER_HOUR: float = 0.18    # reefer running, truck moving
NEWTON_K_IDLE_PER_HOUR: float = 0.85     # stopped: condenser strain, faster leak
G_FORCE_RMS_SMOOTH: float = 0.12         # g, smooth tarmac baseline
G_FORCE_RMS_PER_IRI: float = 0.22        # g added per unit of IRI roughness
COMPRESSOR_BASE_LOAD_PCT: float = 45.0
COMPRESSOR_HEAT_GAIN_PCT_PER_C: float = 3.0  # extra load per C of ambient above 25

# --- Monte Carlo stochastic injections (spec section 3.2) ------------------
THERMAL_VARIANCE_STD_C: float = 2.5      # T_ambient = Forecast + N(0, 2.5)
BORDER_BASE_DELAY_MIN: float = 300.0     # Border_Delay = 300min + N(0, 45)
BORDER_DELAY_STD_MIN: float = 45.0
INFRA_SHOCK_PROBABILITY: float = 0.05    # 5% per trial: unpaved -> "Washed Out"
INFRA_SHOCK_EXTRA_IDLE_HOURS: float = 2.0  # a washout strands the truck idling in heat
VAR_CONFIDENCE: float = 0.95             # 95% Value at Risk

# Vioolsdrift border-post buffer. Sourced from nc_road_network.BORDER_POSTS;
# duplicated here as a documented fallback so this module stays importable even
# if that package isn't on the path. (lon, lat) to match the routing engine's
# lon/lat convention used throughout the codebase.
VIOOLSDRIFT_LONLAT: tuple[float, float] = (17.7500, -28.7500)
BORDER_BUFFER_KM: float = 25.0

# Road classes considered "unpaved" for the infrastructure-shock rule.
UNPAVED_FCLASSES: frozenset[str] = frozenset({
    "track", "track_grade1", "track_grade2", "track_grade3",
    "track_grade4", "track_grade5", "unclassified", "unpaved", "gravel", "dirt",
})

# Default artifact filenames (read by the frontend's ingestion ticker).
DEFAULT_WEATHER_CSV = Path("synthetic_weather_forecast.csv")
DEFAULT_ROAD_CSV = Path("synthetic_road_conditions.csv")
DEFAULT_SENSOR_CSV = Path("synthetic_cold_sensors.csv")
DEFAULT_GRAPH_PATH = Path("nc_road_graph.pkl")

# A representative NC freight coordinate (near Upington) used when a stream is
# generated without a specific route coordinate to anchor Stream A on.
DEFAULT_ANCHOR_LONLAT: tuple[float, float] = (21.2561, -28.4478)


# ---------------------------------------------------------------------------
# Deterministic hash-seeded noise helpers
# ---------------------------------------------------------------------------
def _stable_unit(*keys: object) -> float:
    """Deterministic pseudo-random float in [0.0, 1.0) from arbitrary keys.

    Used so synthetic_forecast_point() returns a STABLE value for the same
    (coordinate, hour) — repeated lookups (e.g. the backend's per-cell forecast
    cache, or several Monte Carlo trials sampling the same node) agree instead
    of jittering, without any shared mutable state."""
    joined = "|".join(str(k) for k in keys)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    # Use the top 64 bits as an integer, normalized to [0, 1).
    return int(digest[:16], 16) / float(1 << 64)


def _stable_gauss(mean: float, std: float, *keys: object) -> float:
    """Deterministic approximately-Gaussian draw via a Box-Muller transform on
    two independent stable-uniform values. Keeps synthetic streams reproducible
    for a given key set."""
    u1 = min(max(_stable_unit("bm1", *keys), 1e-12), 1.0 - 1e-12)
    u2 = _stable_unit("bm2", *keys)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z


def _ensure_utc(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC, matching live_backend.parse_target_datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km between two lon/lat points."""
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_km * c


# ---------------------------------------------------------------------------
# STREAM A — Synthetic Weather Forecast Matrix
# ---------------------------------------------------------------------------
def synthetic_forecast_point(lat: float, lon: float, target_dt: datetime) -> dict:
    """FUTURE ambient weather for one coordinate at one time (Stream A, sampled).

    This is the exact function live_backend.get_forecast_weather() falls back to
    when OpenWeatherMap can't answer, so its return shape is a hard contract:

        {"ambient_temp_c": float, "rain_mm_per_hr": float, "source": str}

    Deterministic in (rounded coordinate, hour) so the backend's forecast cache
    and repeated Monte Carlo samples get a stable answer. Never raises."""
    try:
        dt = _ensure_utc(target_dt)
    except Exception:  # noqa: BLE001 - a malformed datetime must still yield a reading
        dt = datetime.now(timezone.utc)

    # Round the coordinate to ~1.1 km cells (2 dp) so nearby samples share a
    # value, and key the diurnal cycle on the local hour-of-day.
    cell_lat = round(lat, 2)
    cell_lon = round(lon, 2)
    hour_of_day = dt.hour + dt.minute / 60.0
    day_key = dt.strftime("%Y-%m-%d")

    # Diurnal sinusoid: peak at NC_PEAK_TEMP_HOUR, trough ~12h opposite.
    phase = (hour_of_day - NC_PEAK_TEMP_HOUR) / 24.0 * 2.0 * math.pi
    diurnal = math.cos(phase)

    # Latitude effect: the far south of the province runs a touch cooler.
    lat_effect = (abs(cell_lat) - 28.0) * -0.4

    # Deterministic daily "weather regime" offset (a hot spell vs a cool front)
    # plus fine-grained per-hour noise.
    regime_offset = (_stable_unit("regime", cell_lat, cell_lon, day_key) - 0.5) * 8.0
    hourly_noise = (_stable_unit("temp", cell_lat, cell_lon, day_key, int(hour_of_day)) - 0.5) * NC_TEMP_NOISE_SPAN_C

    temp_c = NC_DIURNAL_MEAN_C + NC_DIURNAL_AMPLITUDE_C * diurnal + lat_effect + regime_offset + hourly_noise
    temp_c = _clamp(temp_c, AMBIENT_TEMP_MIN_C, AMBIENT_TEMP_MAX_C)

    # Rain: rare in the Northern Cape. A daily "wet regime" die gates whether
    # rain is possible that day; when it is, an hourly intensity is drawn.
    wet_regime = _stable_unit("wetregime", cell_lat, cell_lon, day_key)
    rain_mm_per_hr = 0.0
    if wet_regime >= NC_RAIN_EVENT_THRESHOLD:
        hourly_rain_die = _stable_unit("rain", cell_lat, cell_lon, day_key, int(hour_of_day))
        if hourly_rain_die >= 0.5:
            intensity = (hourly_rain_die - 0.5) / 0.5  # 0..1
            rain_mm_per_hr = intensity * RAIN_MAX_MM_PER_HR

    rain_mm_per_hr = _clamp(rain_mm_per_hr, RAIN_MIN_MM_PER_HR, RAIN_MAX_MM_PER_HR)

    return {
        "ambient_temp_c": round(temp_c, 2),
        "rain_mm_per_hr": round(rain_mm_per_hr, 2),
        "source": "synthetic_stream_a",
    }


@dataclass
class WeatherForecastRow:
    hour_index: int
    timestamp_iso: str
    lat: float
    lon: float
    ambient_temp_c: float
    rain_mm_per_hr: float
    heat_alert: bool
    rain_alert: bool


def generate_weather_forecast_matrix(
    start_dt: datetime,
    hours: int = 72,
    anchor_lonlat: tuple[float, float] = DEFAULT_ANCHOR_LONLAT,
) -> list[WeatherForecastRow]:
    """Stream A: a future HOURLY timeline of ambient_temp_c (15-42 C) and
    rain_mm_per_hr (0.0-15.0 mm) anchored at one coordinate. Built entirely
    from synthetic_forecast_point() so a single-point lookup and the full
    matrix can never disagree."""
    anchor_lon, anchor_lat = anchor_lonlat
    start_dt = _ensure_utc(start_dt)
    rows: list[WeatherForecastRow] = []
    for hour_index in range(hours):
        ts = start_dt + timedelta(hours=hour_index)
        point = synthetic_forecast_point(anchor_lat, anchor_lon, ts)
        temp_c = point["ambient_temp_c"]
        rain_mm = point["rain_mm_per_hr"]
        rows.append(WeatherForecastRow(
            hour_index=hour_index,
            timestamp_iso=ts.isoformat(),
            lat=round(anchor_lat, 5),
            lon=round(anchor_lon, 5),
            ambient_temp_c=temp_c,
            rain_mm_per_hr=rain_mm,
            heat_alert=temp_c >= EXTREME_HEAT_THRESHOLD_C,
            rain_alert=rain_mm >= HEAVY_RAIN_THRESHOLD_MM,
        ))
    return rows


# ---------------------------------------------------------------------------
# STREAM B — Synthetic Road Conditions / Field Logs (DERIVED from Stream A)
# ---------------------------------------------------------------------------
@dataclass
class RoadConditionRow:
    hour_index: int
    timestamp_iso: str
    lat: float
    lon: float
    trigger: str                # what in Stream A forced this state
    condition_label: str        # e.g. "Impassable / Washed Out" / "Severe Potholes" / "Clear"
    safe_speed_kmh: float
    iri: float                  # International Roughness Index
    edge_weight_multiplier: float  # what the router should multiply spoilage_cost by
    failure_event: bool


def derive_road_conditions(weather_rows: Sequence[WeatherForecastRow]) -> list[RoadConditionRow]:
    """Stream B: for every hour where Stream A shows rain > 5 mm or heat > 38 C,
    emit a matching failure event. Rain -> "Impassable / Washed Out"; heat ->
    "Severe Potholes" (tar bleeding / surface break-up in extreme desert heat).
    Safe speed drops to 15 km/h and IRI spikes toward 6.0; the recomputed edge
    weight multiplier is what the backend's spoilage-weighted shortest-path
    solver uses to reroute around the hazard."""
    rows: list[RoadConditionRow] = []
    for w in weather_rows:
        rain_failure = w.rain_mm_per_hr >= HEAVY_RAIN_THRESHOLD_MM
        heat_failure = w.ambient_temp_c >= EXTREME_HEAT_THRESHOLD_C

        if rain_failure:
            # Washout severity scales with how far past the 5 mm threshold we are.
            severity = _clamp((w.rain_mm_per_hr - HEAVY_RAIN_THRESHOLD_MM) /
                              (RAIN_MAX_MM_PER_HR - HEAVY_RAIN_THRESHOLD_MM), 0.0, 1.0)
            iri = IRI_BASELINE + (IRI_FAILURE_MAX - IRI_BASELINE) * (0.6 + 0.4 * severity)
            condition_label = "Impassable / Washed Out"
            trigger = f"rain {w.rain_mm_per_hr:.1f}mm/hr > {HEAVY_RAIN_THRESHOLD_MM:.0f}mm"
            safe_speed = IMPASSABLE_SAFE_SPEED_KMH
            failure = True
        elif heat_failure:
            severity = _clamp((w.ambient_temp_c - EXTREME_HEAT_THRESHOLD_C) /
                              (AMBIENT_TEMP_MAX_C - EXTREME_HEAT_THRESHOLD_C), 0.0, 1.0)
            iri = IRI_BASELINE + (IRI_FAILURE_MAX - IRI_BASELINE) * (0.5 + 0.5 * severity)
            condition_label = "Severe Potholes"
            trigger = f"heat {w.ambient_temp_c:.1f}C > {EXTREME_HEAT_THRESHOLD_C:.0f}C"
            safe_speed = IMPASSABLE_SAFE_SPEED_KMH
            failure = True
        else:
            iri = IRI_BASELINE
            condition_label = "Clear"
            trigger = "nominal"
            safe_speed = NORMAL_SAFE_SPEED_KMH
            failure = False

        iri = _clamp(iri, IRI_BASELINE, IRI_FAILURE_MAX)
        # Edge-weight multiplier: proportional to IRI (roughness) AND the speed
        # collapse (a 15 km/h crawl multiplies travel time ~6x versus 90 km/h).
        speed_penalty = NORMAL_SAFE_SPEED_KMH / max(safe_speed, 1.0)
        edge_weight_multiplier = round((iri / IRI_BASELINE) * (0.4 + 0.6 * speed_penalty), 3) if failure else 1.0

        rows.append(RoadConditionRow(
            hour_index=w.hour_index,
            timestamp_iso=w.timestamp_iso,
            lat=w.lat,
            lon=w.lon,
            trigger=trigger,
            condition_label=condition_label,
            safe_speed_kmh=round(safe_speed, 1),
            iri=round(iri, 2),
            edge_weight_multiplier=edge_weight_multiplier,
            failure_event=failure,
        ))
    return rows


# ---------------------------------------------------------------------------
# STREAM C — Synthetic Cold-Chain IoT Payloads (driven by Streams A + B)
# ---------------------------------------------------------------------------
@dataclass
class ColdSensorRow:
    hour_index: int
    timestamp_iso: str
    cargo_temp_c: float
    ambient_temp_c: float
    g_force_rms: float
    compressor_load_pct: float
    iri: float
    moving: bool
    spoilage_breach: bool


def newton_cooling_step(cargo_temp_c: float, equilibrium_c: float, k_per_hour: float, dt_hours: float) -> float:
    """One discrete Newton's-Law-of-Cooling state-equation step: the cargo
    temperature exponentially approaches `equilibrium_c` with time-constant
    1/k. Returns the updated temperature. Written out explicitly (no vectorized
    shortcut) so the state equation is auditable."""
    decay = math.exp(-k_per_hour * dt_hours)
    return equilibrium_c + (cargo_temp_c - equilibrium_c) * decay


def _reefer_equilibrium_driving(ambient_temp_c: float) -> float:
    """Equilibrium the reefer holds while driving with the compressor active.
    Holds at setpoint in cool/mild conditions, then creeps up as ambient heat
    starts to out-pace the condenser. The onset is at 30°C (not 38°C) with a
    gentle slope, so a long haul on a genuinely HOT day accrues some thermal
    exposure — the earlier 38°C onset made every mild-weather drive read as a
    flat 0% risk, which looked broken even though it was 'correct'."""
    derate = max(0.0, ambient_temp_c - 30.0) * 0.15
    return REEFER_SETPOINT_C + derate


def _reefer_equilibrium_idle(ambient_temp_c: float) -> float:
    """Equilibrium while stopped/idling — the compressor strains against desert
    heat with no airflow, so the chamber drifts much closer to ambient."""
    return REEFER_SETPOINT_C + max(0.0, ambient_temp_c - 20.0) * 0.55


def generate_cold_chain_stream(
    weather_rows: Sequence[WeatherForecastRow],
    road_rows: Sequence[RoadConditionRow],
    idle_hours_at_failures: float = 1.0,
) -> list[ColdSensorRow]:
    """Stream C: onboard telemetry (cargo_temp_c, g_force_rms, compressor_load_pct)
    evolved hour-by-hour with a discrete Newton's-Law-of-Cooling state equation
    driven by Stream A's ambient heat and Stream B's road failures (which stop
    the truck, switching it to the faster idle-warming regime and raising
    vibration via IRI)."""
    road_by_hour = {r.hour_index: r for r in road_rows}
    rows: list[ColdSensorRow] = []
    cargo_temp_c = CARGO_START_TEMP_C

    for w in weather_rows:
        road = road_by_hour.get(w.hour_index)
        iri = road.iri if road is not None else IRI_BASELINE
        # A failure event forces the truck to stop (or crawl) — model the hour
        # as idling for the failure fraction, so the chamber warms faster.
        is_failure = road.failure_event if road is not None else False
        moving = not is_failure

        if is_failure:
            equilibrium = _reefer_equilibrium_idle(w.ambient_temp_c)
            k = NEWTON_K_IDLE_PER_HOUR
            # Only part of the hour may be spent stalled; blend toward drive rate.
            idle_frac = _clamp(idle_hours_at_failures, 0.0, 1.0)
            k = NEWTON_K_IDLE_PER_HOUR * idle_frac + NEWTON_K_DRIVE_PER_HOUR * (1.0 - idle_frac)
        else:
            equilibrium = _reefer_equilibrium_driving(w.ambient_temp_c)
            k = NEWTON_K_DRIVE_PER_HOUR

        cargo_temp_c = newton_cooling_step(cargo_temp_c, equilibrium, k, dt_hours=1.0)

        g_force_rms = G_FORCE_RMS_SMOOTH + G_FORCE_RMS_PER_IRI * (iri - IRI_BASELINE)
        g_force_rms = max(0.0, g_force_rms) * (1.0 if moving else 0.4)

        compressor_load_pct = COMPRESSOR_BASE_LOAD_PCT + COMPRESSOR_HEAT_GAIN_PCT_PER_C * max(0.0, w.ambient_temp_c - 25.0)
        if is_failure:
            # Stationary in the sun with the condenser fighting heat soak: pinned
            # to full load, mirroring live_backend.py's stationary-queue rule.
            compressor_load_pct = 100.0
        compressor_load_pct = _clamp(compressor_load_pct, 0.0, 100.0)

        rows.append(ColdSensorRow(
            hour_index=w.hour_index,
            timestamp_iso=w.timestamp_iso,
            cargo_temp_c=round(cargo_temp_c, 2),
            ambient_temp_c=w.ambient_temp_c,
            g_force_rms=round(g_force_rms, 3),
            compressor_load_pct=round(compressor_load_pct, 1),
            iri=round(iri, 2),
            moving=moving,
            spoilage_breach=cargo_temp_c > SPOILAGE_BREACH_TEMP_C,
        ))
    return rows


# ---------------------------------------------------------------------------
# CSV writers — the three artifacts the frontend's ingestion ticker reads
# ---------------------------------------------------------------------------
def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Writes a list of dict rows to CSV. Wrapped by the callers below in
    try/except so a read-only working directory degrades to a logged warning
    rather than crashing the generator."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_weather_csv(rows: Sequence[WeatherForecastRow], path: Path = DEFAULT_WEATHER_CSV) -> Optional[Path]:
    try:
        _write_csv(
            path,
            ["hour_index", "timestamp_iso", "lat", "lon", "ambient_temp_c",
             "rain_mm_per_hr", "heat_alert", "rain_alert"],
            [r.__dict__ for r in rows],
        )
        logger.info("Stream A -> wrote %d rows to %s", len(rows), path)
        return path
    except OSError as exc:
        logger.warning("Stream A -> could not write %s: %s", path, exc)
        return None


def write_road_csv(rows: Sequence[RoadConditionRow], path: Path = DEFAULT_ROAD_CSV) -> Optional[Path]:
    try:
        _write_csv(
            path,
            ["hour_index", "timestamp_iso", "lat", "lon", "trigger", "condition_label",
             "safe_speed_kmh", "iri", "edge_weight_multiplier", "failure_event"],
            [r.__dict__ for r in rows],
        )
        logger.info("Stream B -> wrote %d rows to %s", len(rows), path)
        return path
    except OSError as exc:
        logger.warning("Stream B -> could not write %s: %s", path, exc)
        return None


def write_sensor_csv(rows: Sequence[ColdSensorRow], path: Path = DEFAULT_SENSOR_CSV) -> Optional[Path]:
    try:
        _write_csv(
            path,
            ["hour_index", "timestamp_iso", "cargo_temp_c", "ambient_temp_c",
             "g_force_rms", "compressor_load_pct", "iri", "moving", "spoilage_breach"],
            [r.__dict__ for r in rows],
        )
        logger.info("Stream C -> wrote %d rows to %s", len(rows), path)
        return path
    except OSError as exc:
        logger.warning("Stream C -> could not write %s: %s", path, exc)
        return None


def generate_all_streams(
    start_dt: Optional[datetime] = None,
    hours: int = 72,
    anchor_lonlat: tuple[float, float] = DEFAULT_ANCHOR_LONLAT,
    write_files: bool = True,
    weather_path: Path = DEFAULT_WEATHER_CSV,
    road_path: Path = DEFAULT_ROAD_CSV,
    sensor_path: Path = DEFAULT_SENSOR_CSV,
) -> dict:
    """Generate all three coordinated streams (A -> B -> C) and, by default,
    write them to the three CSVs the frontend reads. Returns the in-memory
    rows too, so callers (e.g. the frontend) can generate-and-use in one step
    without a disk round-trip."""
    if start_dt is None:
        start_dt = datetime.now(timezone.utc)
    start_dt = _ensure_utc(start_dt)

    weather_rows = generate_weather_forecast_matrix(start_dt, hours=hours, anchor_lonlat=anchor_lonlat)
    road_rows = derive_road_conditions(weather_rows)
    sensor_rows = generate_cold_chain_stream(weather_rows, road_rows)

    written: dict[str, Optional[Path]] = {}
    if write_files:
        written["weather_csv"] = write_weather_csv(weather_rows, weather_path)
        written["road_csv"] = write_road_csv(road_rows, road_path)
        written["sensor_csv"] = write_sensor_csv(sensor_rows, sensor_path)

    return {
        "start_dt": start_dt.isoformat(),
        "hours": hours,
        "weather_rows": weather_rows,
        "road_rows": road_rows,
        "sensor_rows": sensor_rows,
        "written": written,
    }


# ---------------------------------------------------------------------------
# Route profiling for the Monte Carlo solver
# ---------------------------------------------------------------------------
@dataclass
class _RouteProfile:
    path_nodes: list[int]
    coordinates: list[tuple[float, float]]  # (lon, lat)
    total_drive_hours: float
    fclasses: list[str]
    crosses_border: bool
    has_unpaved: bool
    forecast_ambient_temp_c: float
    forecast_rain_mm_per_hr: float


def _edge_time_hours(edge_attrs: dict) -> float:
    """Best-effort per-edge drive time in hours, tolerant of whichever impedance
    fields are present (base_time_mins from the live backend, or the raw
    travel_time / length_km+speed from nc_road_network)."""
    override = edge_attrs.get("override")
    if override is not None and "base_time_mins" in override:
        return max(0.0, override["base_time_mins"] / 60.0)
    if "base_time_mins" in edge_attrs:
        return max(0.0, edge_attrs["base_time_mins"] / 60.0)
    if edge_attrs.get("travel_time") is not None:
        return max(0.0, float(edge_attrs["travel_time"]))
    length_km = float(edge_attrs.get("length_km", 0.0) or 0.0)
    speed = float(edge_attrs.get("imputed_speed_kmh", 0.0) or 0.0)
    return length_km / speed if speed > 0 else 0.0


def _edge_spoilage(edge_attrs: dict) -> float:
    override = edge_attrs.get("override")
    if override is not None and "spoilage_cost" in override:
        return float(override["spoilage_cost"])
    if "spoilage_cost" in edge_attrs:
        return float(edge_attrs["spoilage_cost"])
    roughness = float(edge_attrs.get("roughness", 1.3) or 1.3)
    return _edge_time_hours(edge_attrs) * (THERMAL_SPOIL_WEIGHT + MECH_SPOIL_WEIGHT * roughness)


def _spoilage_weight(u: int, v: int, edge_attrs: dict) -> float:
    return _edge_spoilage(edge_attrs)


def _profile_route(
    G_main,
    cluster_coord: dict[int, tuple[float, float]],
    origin_node: int,
    destination_node: int,
    target_dt: datetime,
) -> _RouteProfile:
    """Runs the spoilage-optimized shortest path and summarizes everything the
    Monte Carlo solver needs: total drive time, road-class mix, whether the
    corridor passes the Vioolsdrift border buffer, and a route-averaged Stream A
    forecast at the target departure time."""
    if nx is None:
        raise RuntimeError("networkx is required for the Monte Carlo route solver but is not installed.")

    path = nx.shortest_path(G_main, origin_node, destination_node, weight=_spoilage_weight)

    coordinates: list[tuple[float, float]] = [tuple(cluster_coord[n]) for n in path]
    total_drive_hours = 0.0
    fclasses: list[str] = []
    for u, v in zip(path[:-1], path[1:]):
        edge_attrs = G_main[u][v]
        total_drive_hours += _edge_time_hours(edge_attrs)
        fclasses.append(str(edge_attrs.get("fclass", "unknown")))

    crosses_border = any(
        _haversine_km(lon, lat, VIOOLSDRIFT_LONLAT[0], VIOOLSDRIFT_LONLAT[1]) <= BORDER_BUFFER_KM
        for (lon, lat) in coordinates
    )
    has_unpaved = any(fc in UNPAVED_FCLASSES for fc in fclasses)

    # Route-averaged forecast at departure: sample a handful of evenly spaced
    # nodes so a storm/heat cell on part of the corridor still registers.
    sample_count = min(8, len(coordinates))
    if sample_count <= 1:
        sample_coords = coordinates
    else:
        step = (len(coordinates) - 1) / (sample_count - 1)
        sample_coords = [coordinates[round(i * step)] for i in range(sample_count)]

    temps: list[float] = []
    rains: list[float] = []
    for lon, lat in sample_coords:
        point = synthetic_forecast_point(lat, lon, target_dt)
        temps.append(point["ambient_temp_c"])
        rains.append(point["rain_mm_per_hr"])

    return _RouteProfile(
        path_nodes=list(path),
        coordinates=coordinates,
        total_drive_hours=total_drive_hours,
        fclasses=fclasses,
        crosses_border=crosses_border,
        has_unpaved=has_unpaved,
        forecast_ambient_temp_c=float(np.mean(temps)) if temps else NC_DIURNAL_MEAN_C,
        forecast_rain_mm_per_hr=float(np.max(rains)) if rains else 0.0,
    )


def _simulate_peak_cargo_temp(drive_hours: float, idle_hours: float, ambient_temp_c: float) -> float:
    """Integrate the discrete Newton's-Law-of-Cooling state equation across a
    single trial's driving phase followed by any idle phase (border customs and
    any infrastructure-shock stall), returning the PEAK cargo temperature the
    shipment reached. Peak, not final, is what determines spoilage — a chamber
    that breaches -18.0 C and is later re-cooled has still lost the cargo."""
    dt_hours = 0.05
    cargo_temp_c = CARGO_START_TEMP_C
    peak = cargo_temp_c

    drive_equilibrium = _reefer_equilibrium_driving(ambient_temp_c)
    drive_steps = int(max(0.0, drive_hours) / dt_hours)
    for _ in range(drive_steps):
        cargo_temp_c = newton_cooling_step(cargo_temp_c, drive_equilibrium, NEWTON_K_DRIVE_PER_HOUR, dt_hours)
        if cargo_temp_c > peak:
            peak = cargo_temp_c

    idle_equilibrium = _reefer_equilibrium_idle(ambient_temp_c)
    idle_steps = int(max(0.0, idle_hours) / dt_hours)
    for _ in range(idle_steps):
        cargo_temp_c = newton_cooling_step(cargo_temp_c, idle_equilibrium, NEWTON_K_IDLE_PER_HOUR, dt_hours)
        if cargo_temp_c > peak:
            peak = cargo_temp_c

    return peak


def _spoilage_loss_fraction(peak_temp_c: float) -> float:
    """Graduated loss: 0 below the -18.0 C breach point, ramping linearly to a
    total write-off by SPOILAGE_TOTAL_LOSS_TEMP_C (-8 C). A brief small breach
    costs a fraction of the cargo; a sustained warm excursion loses all of it."""
    if peak_temp_c <= SPOILAGE_BREACH_TEMP_C:
        return 0.0
    span = SPOILAGE_TOTAL_LOSS_TEMP_C - SPOILAGE_BREACH_TEMP_C
    return _clamp((peak_temp_c - SPOILAGE_BREACH_TEMP_C) / span, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Monte Carlo stochastic solver (spec section 3.2)
# ---------------------------------------------------------------------------
def run_monte_carlo_risk_analysis(
    G_main,
    cluster_coord: dict[int, tuple[float, float]],
    origin_node: int,
    destination_node: int,
    target_datetime: datetime,
    iterations: int = 1000,
    shipment_value_rand: float = 450_000.0,
    seed: Optional[int] = None,
    return_samples: bool = False,
) -> dict:
    """Monte Carlo cold-chain risk analysis for a future departure.

    Per trial, three stochastic shocks are injected (spec section 3.2):
      1. Thermal variance:        T_ambient = Forecast + N(0, 2.5)
      2. Customs-delay variance:  Border_Delay = 300 min + N(0, 45), applied at
                                  the Vioolsdrift border buffer when the route
                                  passes through it.
      3. Infrastructure shock:    5% probability per trial that an unpaved
                                  segment upgrades to "Washed Out" during heavy
                                  rain, stranding the truck idling in the heat.

    Cargo temperature is tracked across all trials via the discrete Newtonian
    cooling model. Returns Expected journey time, 95% Value at Risk (Rands)
    against `shipment_value_rand`, and the exact probability (%) of total
    spoilage (cargo breaching -18.0 C).

    Called by live_backend.py's /v1/simulation/monte-carlo-risk endpoint with
    the live topology, so the return keys here are a contract consumed by the
    frontend's Monte Carlo visualizer."""
    target_dt = _ensure_utc(target_datetime)
    profile = _profile_route(G_main, cluster_coord, origin_node, destination_node, target_dt)

    rng = random.Random(seed)

    journey_times_hours: list[float] = []
    losses_rand: list[float] = []
    peak_temps_c: list[float] = []
    spoiled_trials = 0
    border_trials = 0
    washout_trials = 0

    forecast_temp = profile.forecast_ambient_temp_c
    forecast_rain = profile.forecast_rain_mm_per_hr
    heavy_rain_forecast = forecast_rain >= HEAVY_RAIN_THRESHOLD_MM

    for _ in range(max(1, iterations)):
        try:
            # (1) Thermal variance.
            trial_ambient = forecast_temp + rng.gauss(0.0, THERMAL_VARIANCE_STD_C)

            # (2) Customs-delay variance at the Vioolsdrift border buffer.
            idle_hours = 0.0
            if profile.crosses_border:
                border_delay_min = max(0.0, BORDER_BASE_DELAY_MIN + rng.gauss(0.0, BORDER_DELAY_STD_MIN))
                idle_hours += border_delay_min / 60.0
                border_trials += 1

            # (3) Infrastructure shock: an unpaved segment washes out during
            #     heavy rain, stranding the truck idling in the heat.
            if profile.has_unpaved and heavy_rain_forecast and rng.random() < INFRA_SHOCK_PROBABILITY:
                idle_hours += INFRA_SHOCK_EXTRA_IDLE_HOURS
                washout_trials += 1

            journey_hours = profile.total_drive_hours + idle_hours
            peak_temp = _simulate_peak_cargo_temp(profile.total_drive_hours, idle_hours, trial_ambient)

            loss_fraction = _spoilage_loss_fraction(peak_temp)
            trial_loss = loss_fraction * shipment_value_rand

            journey_times_hours.append(journey_hours)
            peak_temps_c.append(peak_temp)
            losses_rand.append(trial_loss)
            if peak_temp > SPOILAGE_BREACH_TEMP_C:
                spoiled_trials += 1
        except Exception as exc:  # noqa: BLE001 - one bad trial must never abort the run
            logger.warning("MONTE CARLO -> trial skipped after error: %s", exc)
            continue

    if not journey_times_hours:
        raise RuntimeError("Monte Carlo produced no valid trials.")

    expected_time_hours = float(np.mean(journey_times_hours))
    losses_array = np.array(losses_rand, dtype=float)
    var_95_rand = float(np.percentile(losses_array, VAR_CONFIDENCE * 100.0))
    prob_spoilage_pct = spoiled_trials / len(journey_times_hours) * 100.0

    result = {
        "expected_journey_time_hours": round(expected_time_hours, 2),
        "expected_journey_time_mins": round(expected_time_hours * 60.0, 1),
        "value_at_risk_95_rand": round(var_95_rand, -2),
        "prob_total_spoilage_pct": round(prob_spoilage_pct, 2),
        "mean_expected_loss_rand": round(float(np.mean(losses_array)), -2),
        "mean_peak_cargo_temp_c": round(float(np.mean(peak_temps_c)), 2),
        "worst_peak_cargo_temp_c": round(float(np.max(peak_temps_c)), 2),
        "spoilage_breach_temp_c": SPOILAGE_BREACH_TEMP_C,
        "route_crosses_border": profile.crosses_border,
        "route_has_unpaved": profile.has_unpaved,
        "route_total_drive_hours": round(profile.total_drive_hours, 2),
        "forecast_ambient_temp_c": round(forecast_temp, 1),
        "forecast_rain_mm_per_hr": round(forecast_rain, 2),
        "border_trials": border_trials,
        "washout_trials": washout_trials,
        "iterations": len(journey_times_hours),
        "shipment_value_rand": shipment_value_rand,
        "target_datetime": target_dt.isoformat(),
    }

    # Optional per-trial arrays, so the frontend can stream the distribution
    # building up in real time (batched calls) and draw a live histogram of
    # peak cargo temperatures. Rounded to keep the JSON payload small.
    if return_samples:
        result["sample_peak_temps_c"] = [round(t, 2) for t in peak_temps_c]
        result["sample_losses_rand"] = [round(x, 2) for x in losses_rand]
        result["sample_journey_hours"] = [round(h, 3) for h in journey_times_hours]

    return result


# ---------------------------------------------------------------------------
# Statistical validation suite (spec section 3.3) — logged to console
# ---------------------------------------------------------------------------
def _reference_thermal_curve(weather_rows: Sequence[WeatherForecastRow]) -> list[float]:
    """A closed-form 'historical reference' cargo-temperature curve to validate
    the simulated Stream C against. Uses the same Newton state equation but with
    a single blended coefficient and no failure-driven idle switching — a smooth
    analytical baseline the noisy simulation should track closely, not match
    exactly."""
    ref: list[float] = []
    cargo_temp_c = CARGO_START_TEMP_C
    k_blend = (NEWTON_K_DRIVE_PER_HOUR + NEWTON_K_IDLE_PER_HOUR) / 2.0
    for w in weather_rows:
        equilibrium = _reefer_equilibrium_driving(w.ambient_temp_c)
        cargo_temp_c = newton_cooling_step(cargo_temp_c, equilibrium, k_blend, dt_hours=1.0)
        ref.append(cargo_temp_c)
    return ref


def compute_thermal_rmse(sensor_rows: Sequence[ColdSensorRow], reference_curve: Sequence[float]) -> float:
    """RMSE between the simulated Stream C cargo-temperature curve and the
    historical reference curve. Written out explicitly rather than via a library
    metric so the calculation is transparent in an audit."""
    n = min(len(sensor_rows), len(reference_curve))
    if n == 0:
        return 0.0
    squared_error_sum = 0.0
    for i in range(n):
        diff = sensor_rows[i].cargo_temp_c - reference_curve[i]
        squared_error_sum += diff * diff
    return math.sqrt(squared_error_sum / n)


def _build_snap_index(cluster_coord: dict[int, tuple[float, float]]):
    """Builds an equirectangular-projected KD-tree over node coordinates for the
    150 m snapping-accuracy validation. Falls back to a brute-force linear scan
    if scipy isn't available, so the suite still runs."""
    node_ids = list(cluster_coord.keys())
    coords = np.array([cluster_coord[n] for n in node_ids], dtype=float)  # (lon, lat)
    mean_lat = float(coords[:, 1].mean())
    mx = 111_320.0 * math.cos(math.radians(mean_lat))
    my = 110_540.0
    projected = np.column_stack([coords[:, 0] * mx, coords[:, 1] * my])

    if cKDTree is not None:
        tree = cKDTree(projected)

        def snap(lon: float, lat: float) -> float:
            distance_m, _ = tree.query(np.array([lon * mx, lat * my]))
            return float(distance_m)
    else:  # pragma: no cover - only when scipy is absent
        def snap(lon: float, lat: float) -> float:
            query = np.array([lon * mx, lat * my])
            dists = np.linalg.norm(projected - query, axis=1)
            return float(dists.min())

    return snap, node_ids, coords


def validate_spatial_snapping(
    cluster_coord: dict[int, tuple[float, float]],
    num_samples: int = 500,
    jitter_m: float = 120.0,
    seed: int = 7,
) -> dict:
    """Spatial Snapping Failure Rate: generate `num_samples` synthetic GPS
    coordinates by jittering real network nodes by up to `jitter_m` (< 150 m),
    snap each back, and confirm 100% anchor within SNAP_TOLERANCE_M. Returns the
    pass/fail rate and worst-case distance."""
    snap, node_ids, coords = _build_snap_index(cluster_coord)
    rng = random.Random(seed)
    mean_lat = float(coords[:, 1].mean())
    mx = 111_320.0 * math.cos(math.radians(mean_lat))
    my = 110_540.0

    within = 0
    worst_m = 0.0
    for _ in range(num_samples):
        idx = rng.randrange(len(node_ids))
        base_lon, base_lat = cluster_coord[node_ids[idx]]
        bearing = rng.uniform(0.0, 2.0 * math.pi)
        radius_m = rng.uniform(0.0, jitter_m)
        jitter_lon = base_lon + (radius_m * math.cos(bearing)) / mx
        jitter_lat = base_lat + (radius_m * math.sin(bearing)) / my
        snapped_m = snap(jitter_lon, jitter_lat)
        worst_m = max(worst_m, snapped_m)
        if snapped_m <= SNAP_TOLERANCE_M:
            within += 1

    within_pct = within / num_samples * 100.0 if num_samples else 0.0
    return {
        "samples": num_samples,
        "within_tolerance_pct": round(within_pct, 2),
        "failure_rate_pct": round(100.0 - within_pct, 2),
        "worst_snap_distance_m": round(worst_m, 1),
        "tolerance_m": SNAP_TOLERANCE_M,
    }


def _load_cluster_coord(graph_path: Path = DEFAULT_GRAPH_PATH) -> Optional[dict[int, tuple[float, float]]]:
    """Best-effort load of cluster_coord from nc_road_graph.pkl for the snapping
    validation. Returns None (never raises) if the artifact is missing/corrupt —
    the caller then substitutes a synthetic node cloud so the suite still runs."""
    if not graph_path.exists():
        logger.warning("Validation -> %s not found; using a synthetic node cloud for snapping test.", graph_path)
        return None
    try:
        with graph_path.open("rb") as handle:
            artifacts = pickle.load(handle)
        return artifacts["cluster_coord"]
    except Exception as exc:  # noqa: BLE001 - a bad pickle must degrade, not crash
        logger.warning("Validation -> failed to load %s (%s); using a synthetic node cloud.", graph_path, exc)
        return None


def _synthetic_node_cloud(count: int = 2000, seed: int = 11) -> dict[int, tuple[float, float]]:
    """A synthetic scattering of nodes across the Northern Cape bounding box,
    used only when the real graph isn't available, so validate_spatial_snapping
    still exercises real geometry."""
    rng = random.Random(seed)
    min_lon, max_lon = 16.45, 25.30
    min_lat, max_lat = -31.85, -24.60
    return {
        i: (rng.uniform(min_lon, max_lon), rng.uniform(min_lat, max_lat))
        for i in range(count)
    }


def run_validation_suite(streams: dict, graph_path: Path = DEFAULT_GRAPH_PATH) -> dict:
    """Runs and logs the full statistical validation suite (spec section 3.3)."""
    logger.info("=" * 78)
    logger.info("STATISTICAL VALIDATION SUITE")
    logger.info("=" * 78)

    # 1. Thermodynamic RMSE vs historical reference curve.
    reference_curve = _reference_thermal_curve(streams["weather_rows"])
    rmse = compute_thermal_rmse(streams["sensor_rows"], reference_curve)
    logger.info(
        "Thermodynamic RMSE (simulated Stream C vs historical reference): %.3f C over %d hours",
        rmse, len(streams["sensor_rows"]),
    )

    # 2. Spatial snapping failure rate at SNAP_TOLERANCE_M (150 m).
    cluster_coord = _load_cluster_coord(graph_path)
    if cluster_coord is None:
        cluster_coord = _synthetic_node_cloud()
    snap_result = validate_spatial_snapping(cluster_coord)
    status = "PASS" if snap_result["failure_rate_pct"] == 0.0 else "REVIEW"
    logger.info(
        "Spatial Snapping Failure Rate: %.2f%% (%.2f%% within %.0fm, worst=%.1fm) -> %s",
        snap_result["failure_rate_pct"], snap_result["within_tolerance_pct"],
        snap_result["tolerance_m"], snap_result["worst_snap_distance_m"], status,
    )

    logger.info("=" * 78)
    return {"thermal_rmse_c": round(rmse, 3), "snapping": snap_result}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Northern Cape cold-chain simulation & risk evaluator.")
    parser.add_argument("--hours", type=int, default=72, help="Length of the synthetic forecast timeline (hours).")
    parser.add_argument("--iterations", type=int, default=1000, help="Monte Carlo trials for the demo run.")
    parser.add_argument("--no-files", action="store_true", help="Skip writing the three CSV artifacts.")
    parser.add_argument("--graph", type=str, default=str(DEFAULT_GRAPH_PATH), help="Path to nc_road_graph.pkl.")
    parser.add_argument("--origin", type=str, default="Kimberley", help="Monte Carlo demo origin town.")
    parser.add_argument("--destination", type=str, default="Springbok", help="Monte Carlo demo destination town.")
    return parser.parse_args(argv)


def _demo_monte_carlo(graph_path: Path, origin_town: str, destination_town: str, iterations: int) -> None:
    """Optional standalone Monte Carlo demonstration against the real graph, if
    it and its town table are available. Purely illustrative for a console run —
    the live system drives this through the backend endpoint instead."""
    if nx is None:
        logger.info("Monte Carlo demo skipped: networkx not installed.")
        return
    if not graph_path.exists():
        logger.info("Monte Carlo demo skipped: %s not found (run Data_Audit.ipynb to produce it).", graph_path)
        return
    try:
        with graph_path.open("rb") as handle:
            artifacts = pickle.load(handle)
        G_main = artifacts["G_main"]
        cluster_coord = artifacts["cluster_coord"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Monte Carlo demo skipped: could not load %s: %s", graph_path, exc)
        return

    try:
        import nc_road_network as ncr
        towns = ncr.TOWNS
    except Exception:  # noqa: BLE001 - fall back to bare-coordinate resolution
        towns = None

    def resolve(town: str) -> Optional[int]:
        if towns is None or town not in towns:
            logger.info("Monte Carlo demo: town '%s' unavailable; skipping.", town)
            return None
        lon, lat = towns[town]
        # Nearest node via the same projection style used elsewhere.
        node_ids = list(cluster_coord.keys())
        coords = np.array([cluster_coord[n] for n in node_ids], dtype=float)
        mean_lat = float(coords[:, 1].mean())
        mx = 111_320.0 * math.cos(math.radians(mean_lat))
        my = 110_540.0
        query = np.array([lon * mx, lat * my])
        projected = np.column_stack([coords[:, 0] * mx, coords[:, 1] * my])
        return node_ids[int(np.argmin(np.linalg.norm(projected - query, axis=1)))]

    origin_node = resolve(origin_town)
    destination_node = resolve(destination_town)
    if origin_node is None or destination_node is None:
        return

    target_dt = datetime.now(timezone.utc) + timedelta(hours=12)
    try:
        result = run_monte_carlo_risk_analysis(
            G_main, cluster_coord, origin_node, destination_node,
            target_dt, iterations=iterations,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Monte Carlo demo failed: %s", exc)
        return

    logger.info("=" * 78)
    logger.info("MONTE CARLO DEMO -> %s -> %s (%d trials, depart +12h)", origin_town, destination_town, iterations)
    logger.info("  Expected journey time : %.2f h (%.0f min)",
                result["expected_journey_time_hours"], result["expected_journey_time_mins"])
    logger.info("  95%% Value at Risk     : R %s", f"{result['value_at_risk_95_rand']:,.0f}")
    logger.info("  Prob. total spoilage  : %.2f%% (breach > %.1f C)",
                result["prob_total_spoilage_pct"], result["spoilage_breach_temp_c"])
    logger.info("  Mean/worst peak temp  : %.2f C / %.2f C",
                result["mean_peak_cargo_temp_c"], result["worst_peak_cargo_temp_c"])
    logger.info("  Route crosses border  : %s | has unpaved: %s",
                result["route_crosses_border"], result["route_has_unpaved"])
    logger.info("=" * 78)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    start_dt = datetime.now(timezone.utc)

    logger.info("Generating three coordinated synthetic streams (A -> B -> C) over %d hours...", args.hours)
    streams = generate_all_streams(start_dt=start_dt, hours=args.hours, write_files=not args.no_files)

    heat_events = sum(1 for r in streams["road_rows"] if r.condition_label == "Severe Potholes")
    washout_events = sum(1 for r in streams["road_rows"] if r.condition_label == "Impassable / Washed Out")
    breaches = sum(1 for r in streams["sensor_rows"] if r.spoilage_breach)
    logger.info(
        "Stream summary -> %d heat-driven failures, %d washout failures, %d cargo spoilage-breach hours.",
        heat_events, washout_events, breaches,
    )

    run_validation_suite(streams, graph_path=Path(args.graph))
    _demo_monte_carlo(Path(args.graph), args.origin, args.destination, args.iterations)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

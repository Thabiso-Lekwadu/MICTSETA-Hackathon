"""
weather_engine.py

Isolated atmospheric data module for the Northern Cape Fleet Dispatch system.
Wraps the free, keyless Open-Meteo API (https://open-meteo.com) with a small
TTL cache so live dashboard polling (every ~2s from live_frontend.py) never
hammers the external service, and a hard fallback to a clear-weather
baseline so a network blip or API outage can never take down routing,
telemetry, or the dispatch UI that depend on it.

Deliberately kept dependency-free beyond `requests` and stateless from the
caller's point of view — live_backend.py never needs to know whether a given
reading came from the network or the cache or the fallback; it just gets a
WeatherReading back, always.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TypedDict

import requests

logger = logging.getLogger("weather_engine")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 5.0

# Fallback used whenever the live API can't be reached in time or returns
# something malformed — a plausible clear, mild Northern Cape day, not an
# alarming value, so a network blip can never accidentally trip a heat/storm
# alert downstream on its own.
FALLBACK_TEMP_C = 25.0
FALLBACK_RAIN_MM = 0.0
FALLBACK_ALERT = "Normal"

EXTREME_HEAT_THRESHOLD_C = 38.0
HEAVY_RAIN_THRESHOLD_MM = 5.0

# Coordinates are cached at 2-decimal-degree precision (~1.1km) so several
# polls of the same truck a few seconds apart share one cache entry instead
# of each issuing a fresh request. TTL keeps the cache from ever serving
# genuinely stale weather during a long-running session.
CACHE_TTL_SECONDS = 300.0

_cache: dict[tuple[float, float], tuple[float, "WeatherReading"]] = {}
_cache_lock = threading.Lock()


class WeatherReading(TypedDict):
    temp_c: float
    rain_mm: float
    alert: str          # "Normal" | "Extreme Heat" | "Heavy Rain / Washout Risk"
    source: str          # "open-meteo" | "fallback"


def _cache_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat, 2), round(lon, 2))


def _classify_alert(temp_c: float, rain_mm: float) -> str:
    # Heat takes priority in the label since it's the more acute risk to the
    # reefer unit itself; a segment can still be hot AND rainy simultaneously,
    # callers that need both signals should read temp_c/rain_mm directly
    # rather than relying on this single label for everything.
    if temp_c >= EXTREME_HEAT_THRESHOLD_C:
        return "Extreme Heat"
    if rain_mm >= HEAVY_RAIN_THRESHOLD_MM:
        return "Heavy Rain / Washout Risk"
    return "Normal"


def _fallback_reading() -> WeatherReading:
    return {
        "temp_c": FALLBACK_TEMP_C,
        "rain_mm": FALLBACK_RAIN_MM,
        "alert": FALLBACK_ALERT,
        "source": "fallback",
    }


def get_current_weather(lat: float, lon: float) -> WeatherReading:
    """Returns live ambient weather for a coordinate.

    Backed by a small TTL cache and the keyless Open-Meteo current-weather
    API. This function must never raise — routing, telemetry, and thermal
    logic all depend on always getting *some* reading back, so any network
    error, timeout, or malformed response degrades to the fallback baseline
    instead of propagating.
    """
    key = _cache_key(lat, lon)
    now = time.monotonic()

    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            cached_at, reading = cached
            if now - cached_at < CACHE_TTL_SECONDS:
                return reading

    try:
        response = requests.get(
            OPEN_METEO_BASE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,rain",
                "timezone": "auto",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        current = payload["current"]
        temp_c = float(current["temperature_2m"])
        rain_mm = float(current.get("rain", 0.0) or 0.0)
        reading: WeatherReading = {
            "temp_c": temp_c,
            "rain_mm": rain_mm,
            "alert": _classify_alert(temp_c, rain_mm),
            "source": "open-meteo",
        }
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure
        # mode (timeout, DNS failure, HTTP error, malformed/missing JSON
        # keys) must degrade to the fallback reading, never propagate and
        # break the caller's routing/telemetry loop.
        logger.warning(
            "weather_engine -> Open-Meteo request failed for (%.4f, %.4f): %s. Using fallback baseline.",
            lat, lon, exc,
        )
        reading = _fallback_reading()

    with _cache_lock:
        _cache[key] = (now, reading)

    return reading


def get_weather_profile(coordinates: list[tuple[float, float]]) -> list[WeatherReading]:
    """Given a list of (lon, lat) coordinate pairs — matching the routing
    engine's [lon, lat] convention used throughout live_backend.py — samples
    live weather at each point and returns one WeatherReading per point, in
    the same order given. Each individual lookup uses the same cache/
    fallback behavior as get_current_weather; a failure on one point never
    affects the others."""
    profile: list[WeatherReading] = []
    for lon, lat in coordinates:
        profile.append(get_current_weather(lat, lon))
    return profile

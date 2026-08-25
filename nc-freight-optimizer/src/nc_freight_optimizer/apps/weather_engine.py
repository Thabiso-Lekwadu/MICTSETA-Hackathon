"""
weather_engine.py

Isolated atmospheric data module for the Northern Cape Fleet Dispatch system.
Wraps the free, keyless Open-Meteo API (https://open-meteo.com) with a small
TTL cache so live dashboard polling (every ~2s from live_frontend.py) never
hammers the external service, and a hard fallback to a clear-weather
baseline so a network blip or API outage can never take down routing,
telemetry, or the dispatch UI that depend on it.

Optionally, if an OPENWEATHERMAP_API_KEY environment variable is set, current
readings are tried against OpenWeatherMap first — its "current weather"
endpoint blends in real nearby station observations rather than being a pure
forecast-model estimate, so it tracks a live thermometer (and Google's
weather numbers, which do the same blending) more closely. Open-Meteo is
still always the fallback if OpenWeatherMap isn't configured or a call to it
fails, so nothing about this app's behavior changes for anyone who doesn't
set the key.

Deliberately kept dependency-free beyond `requests` and stateless from the
caller's point of view — live_backend.py never needs to know whether a given
reading came from OpenWeatherMap, Open-Meteo, the cache, or the fallback; it
just gets a WeatherReading back, always.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TypedDict

import requests

logger = logging.getLogger("weather_engine")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
OPENWEATHERMAP_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
REQUEST_TIMEOUT_SECONDS = 5.0

# Free-tier station-blended provider. Optional — unset means "use Open-Meteo
# only", exactly as before. Get a free key at https://openweathermap.org/api
# if closer alignment with Google/ground-truth station readings matters more
# than staying fully keyless.
OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", "").strip()

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
    source: str          # "openweathermap" | "open-meteo" | "fallback"


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


def _fetch_openweathermap(lat: float, lon: float) -> WeatherReading | None:
    """Tries OpenWeatherMap's current-weather endpoint, which blends in
    real nearby station observations. Returns None (never raises) on any
    failure — timeout, network error, bad key, malformed response — so the
    caller can fall through to Open-Meteo exactly as if this provider
    weren't configured at all."""
    try:
        response = requests.get(
            OPENWEATHERMAP_BASE_URL,
            params={
                "lat": lat,
                "lon": lon,
                "appid": OPENWEATHERMAP_API_KEY,
                "units": "metric",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        temp_c = float(payload["main"]["temp"])
        # OWM reports rain volume for the last 1h (or 3h) under "rain", only
        # present in the payload when it's actually raining.
        rain_block = payload.get("rain") or {}
        rain_mm = float(rain_block.get("1h", rain_block.get("3h", 0.0)) or 0.0)
        return {
            "temp_c": temp_c,
            "rain_mm": rain_mm,
            "alert": _classify_alert(temp_c, rain_mm),
            "source": "openweathermap",
        }
    except Exception as exc:  # noqa: BLE001 - any failure just falls through
        logger.warning(
            "weather_engine -> OpenWeatherMap request failed for (%.4f, %.4f): %s. Falling back to Open-Meteo.",
            lat, lon, exc,
        )
        return None


def _fetch_open_meteo(lat: float, lon: float) -> WeatherReading | None:
    """Tries the free, keyless Open-Meteo forecast-model endpoint. Returns
    None (never raises) on any failure, so the caller falls through to the
    hard fallback baseline."""
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
        return {
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
        return None


def get_current_weather(lat: float, lon: float) -> WeatherReading:
    """Returns live ambient weather for a coordinate.

    Backed by a small TTL cache. Tries OpenWeatherMap first if
    OPENWEATHERMAP_API_KEY is set (station-blended, closer to Google's
    numbers); otherwise, or if that fails, falls through to the keyless
    Open-Meteo forecast-model estimate; and if that also fails, falls
    through to a hard fallback baseline. This function must never raise —
    routing, telemetry, and thermal logic all depend on always getting
    *some* reading back.
    """
    key = _cache_key(lat, lon)
    now = time.monotonic()

    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            cached_at, reading = cached
            if now - cached_at < CACHE_TTL_SECONDS:
                return reading

    reading: WeatherReading | None = None
    if OPENWEATHERMAP_API_KEY:
        reading = _fetch_openweathermap(lat, lon)
    if reading is None:
        reading = _fetch_open_meteo(lat, lon)
    if reading is None:
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
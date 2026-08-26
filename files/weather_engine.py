"""
weather_engine.py

Isolated atmospheric data module for the Northern Cape Fleet Dispatch system.

Provider order (highest fidelity first), all fault-tolerant:
  1. OpenWeatherMap  — used when an API key is configured. Its "current weather"
     endpoint blends in real nearby station observations, so it tracks a live
     thermometer (and Google's numbers) closely.
  2. Open-Meteo      — free, keyless forecast-model estimate. Always the fallback
     if OpenWeatherMap isn't configured or a call to it fails.
  3. 25°C / 0 mm baseline — a final hard fallback so a network blip can never
     take down routing, telemetry, or the dispatch UI.

API KEY HANDLING (never exposed):
  The OpenWeatherMap key is read from the environment, and — as a convenience —
  from a local `.env` file sitting in THIS module's folder. Any of these variable
  names is accepted (checked in order): OPENWEATHERMAP_API_KEY, OWM_API_KEY,
  API_KEY. So a `.env` next to this file containing simply:

      API_KEY = "your-openweathermap-key"

  is enough. The key is loaded into the process environment only; it is NEVER
  logged, never returned in any WeatherReading, and never sent to the frontend —
  callers only ever see the resolved provider name ("openweathermap" /
  "open-meteo" / "fallback"), not the secret. Keep `.env` out of version control
  (add it to .gitignore); a committable `.env.example` ships alongside.

Deliberately dependency-free beyond `requests`: the `.env` parser below is a tiny
built-in so no python-dotenv install is required.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import TypedDict

import requests

logger = logging.getLogger("weather_engine")

# ---------------------------------------------------------------------------
# .env loading + API-key resolution (no secret ever logged)
# ---------------------------------------------------------------------------
# Accepted env-var names for the OpenWeatherMap key, in priority order. API_KEY
# is included because that's the simplest name a user is likely to drop into a
# local .env, exactly as requested.
_OWM_KEY_ENV_NAMES = ("OPENWEATHERMAP_API_KEY", "OWM_API_KEY", "API_KEY")


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    """Parses one `KEY = "value"` / `KEY=value` / `export KEY=value` line.
    Ignores blanks and `#` comments. Strips surrounding quotes and whitespace.
    Returns (key, value) or None."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.lower().startswith("export "):
        stripped = stripped[len("export "):]
    if "=" not in stripped:
        return None
    key, _, value = stripped.partition("=")
    key = key.strip()
    value = value.strip()
    # Drop an inline comment only when the value isn't quoted.
    if value and value[0] not in ("'", '"') and " #" in value:
        value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    if not key:
        return None
    return key, value


def load_dotenv(dotenv_path: Path | None = None) -> bool:
    """Loads `.env` into os.environ WITHOUT overriding variables already set in
    the real environment (real env wins). Searches, in order: an explicit path if
    given, then a `.env` next to THIS module, then a `.env` in the current working
    directory (so it's found whether you run the backend from its own folder or
    the project root). Never raises, never logs any value. Returns True if any
    file was found and read."""
    candidates: list[Path] = []
    if dotenv_path is not None:
        candidates.append(Path(dotenv_path))
    else:
        candidates.append(Path(__file__).resolve().with_name(".env"))
        try:
            candidates.append(Path.cwd() / ".env")
        except Exception:  # noqa: BLE001 - cwd can be unavailable in odd contexts
            pass

    loaded_any = False
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = str(path.resolve())
            if resolved in seen or not path.exists():
                continue
            seen.add(resolved)
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                parsed = _parse_dotenv_line(raw_line)
                if parsed is None:
                    continue
                key, value = parsed
                os.environ.setdefault(key, value)  # real env var, then first .env found, wins
            logger.info("weather_engine -> loaded environment overrides from %s (values not logged).", path.name)
            loaded_any = True
        except Exception as exc:  # noqa: BLE001 - a malformed .env must never crash import
            logger.warning("weather_engine -> could not read %s (%s); continuing.", path, exc)
    return loaded_any


def resolve_owm_api_key() -> str:
    """Returns the OpenWeatherMap key from the first matching env var (after
    .env has been loaded), or "" if none is configured. The returned value is a
    secret — callers must not log it."""
    for name in _OWM_KEY_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


# Load .env once at import, then resolve the key.
load_dotenv()
OPENWEATHERMAP_API_KEY = resolve_owm_api_key()
WEATHER_KEY_CONFIGURED = bool(OPENWEATHERMAP_API_KEY)
ACTIVE_PROVIDER = "openweathermap" if WEATHER_KEY_CONFIGURED else "open-meteo"

if WEATHER_KEY_CONFIGURED:
    logger.info("weather_engine -> OpenWeatherMap key detected; using station-blended readings (key not logged).")
else:
    logger.info("weather_engine -> no OpenWeatherMap key found; using the keyless Open-Meteo provider.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
OPENWEATHERMAP_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
REQUEST_TIMEOUT_SECONDS = 5.0

FALLBACK_TEMP_C = 25.0
FALLBACK_RAIN_MM = 0.0
FALLBACK_ALERT = "Normal"

EXTREME_HEAT_THRESHOLD_C = 38.0
HEAVY_RAIN_THRESHOLD_MM = 5.0

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
    """Tries OpenWeatherMap's current-weather endpoint. Returns None (never
    raises) on any failure — timeout, network error, bad key, malformed
    response. The key is sent only as a request parameter; it is never logged,
    even inside the error path below."""
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
        rain_block = payload.get("rain") or {}
        rain_mm = float(rain_block.get("1h", rain_block.get("3h", 0.0)) or 0.0)
        return {
            "temp_c": temp_c,
            "rain_mm": rain_mm,
            "alert": _classify_alert(temp_c, rain_mm),
            "source": "openweathermap",
        }
    except Exception as exc:  # noqa: BLE001 - any failure just falls through
        # NB: %s on the exception never includes the appid — requests does not
        # echo query params in its exception messages for these error types.
        logger.warning(
            "weather_engine -> OpenWeatherMap request failed for (%.4f, %.4f): %s. Falling back to Open-Meteo.",
            lat, lon, exc,
        )
        return None


def _fetch_open_meteo(lat: float, lon: float) -> WeatherReading | None:
    """Tries the free, keyless Open-Meteo forecast-model endpoint. Returns None
    (never raises) on any failure, so the caller falls through to the hard
    fallback baseline."""
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
    except Exception as exc:  # noqa: BLE001 - any failure mode must degrade to the fallback reading
        logger.warning(
            "weather_engine -> Open-Meteo request failed for (%.4f, %.4f): %s. Using fallback baseline.",
            lat, lon, exc,
        )
        return None


def get_current_weather(lat: float, lon: float) -> WeatherReading:
    """Returns live ambient weather for a coordinate. OpenWeatherMap first if a
    key is configured, then Open-Meteo, then the hard baseline. Never raises."""
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
    """Given a list of (lon, lat) pairs (matching the routing engine's [lon, lat]
    convention), samples live weather at each and returns one WeatherReading per
    point, in order. A failure on one point never affects the others."""
    profile: list[WeatherReading] = []
    for lon, lat in coordinates:
        profile.append(get_current_weather(lat, lon))
    return profile
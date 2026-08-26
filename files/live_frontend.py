"""
live_frontend.py

Streamlit command dashboard for the Northern Cape Fleet Dispatch system.
Talks to live_backend.py (FastAPI, http://127.0.0.1:8000) over plain HTTP.

Reorganized into five tabs (spec §5):
  1. 🛰️  Customer Route Tracker        - live vehicle position, cargo temp,
                                          compressor workload, and marker on a map.
  2. 📝  Driver Reporting Desk          - mobile ground-truth form; a storm/washout
                                          report pins the segment to IRI ≥ 6.0 and
                                          forces a detour on the next reroute.
  3. 🔮  Journey Prediction & Scheduling - 7-day future planner: synthetic Stream
                                          A/B/C ingestion ticker, Monte Carlo
                                          stochastic risk test, and an executive
                                          departure-slot recommendation matrix.
  4. 📊  Report                          - the AI-generated business-value memo
                                          (local Qwen2.5-1.5B via the backend).
  5. 🌤️  Weather                         - live ambient climate + forecast along
                                          the route corridor.

Zero destructive modifications: the live Traccar hardware feed, the smooth
browser-side live map, and the driver manual-override loop are all preserved
exactly as they worked before — only reorganized and extended.

Run standalone (uv workspace, alongside a running live_backend.py):
    uv run streamlit run live_frontend.py
"""

from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import logging
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

# pandas is used for the synthetic-stream ingestion ticker and the executive
# matrix. Imported defensively so a missing install produces a clear message
# rather than a hard crash of the whole dashboard.
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# streamlit_folium + folium power the Journey Prediction tab's static hazard map
# (spec architecture note calls for streamlit_folium). Optional: if either isn't
# installed, that one map degrades to a caption and the rest of the tab still works.
try:
    import folium
    from streamlit_folium import st_folium
    HAVE_FOLIUM = True
except ImportError:  # pragma: no cover
    folium = None
    st_folium = None
    HAVE_FOLIUM = False

# The synthetic data generator is a sibling module. Imported lazily/optionally so
# the dashboard still loads if it's absent — only the ingestion ticker needs it,
# and it degrades to an explanatory message.
try:
    import vrp_simulation_generator as vrpsim
except ImportError:  # pragma: no cover
    vrpsim = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 5

MODE_HARDWARE = "Live Mobile Hardware Tracking (Traccar)"
MODE_SIMULATOR = "Automated Ingestion Simulator"

FEED_SOURCE_REAL = "REAL-TIME TRACCAR HARDWARE"
FEED_SOURCE_SIMULATED = "SIMULATED TELEMETRY MATRIX"

OPTIMIZED_COLOR = "#1f9e89"   # teal — spoilage-optimized route
STANDARD_COLOR = "#9a9a9a"    # grey, dashed — standard time-only route
HAZARD_COLOR = "#d64545"      # red — forecast hazard zones

# The three synthetic-stream artifacts written by vrp_simulation_generator.py.
WEATHER_CSV = Path("synthetic_weather_forecast.csv")
ROAD_CSV = Path("synthetic_road_conditions.csv")
SENSOR_CSV = Path("synthetic_cold_sensors.csv")

PREDICTION_MAX_HORIZON_DAYS = 7

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live_frontend")

st.set_page_config(
    page_title="Northern Cape Fleet Dispatch",
    page_icon="🚛",  # browser tab icon only — not UI body text
    layout="wide",
)

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3, h4, .stTabs [data-baseweb="tab"] p {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.75rem !important;
    }
    /* Kill the "stale element" fade/pulse during fragment reruns (see the
       long note in the original dispatch tab): forcing opacity back to 1 on
       any stale element removes the pulse; data still updates in place. */
    [data-testid="stElementContainer"][data-stale="true"] {
        opacity: 1 !important;
        transition: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Backend client helpers — every call is wrapped so an offline backend
# degrades gracefully instead of crashing the dashboard.
# ---------------------------------------------------------------------------
def backend_get(path: str, silent: bool = False) -> dict | None:
    try:
        response = requests.get(f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        if not silent:
            st.error(f"Cannot reach the backend server. Is live_backend.py running? Expected at {BACKEND_URL}")
        logger.error("GET %s failed: connection error", path)
        return None
    except requests.exceptions.Timeout:
        if not silent:
            st.error(f"Backend request to {path} timed out.")
        logger.error("GET %s failed: timeout", path)
        return None
    except requests.exceptions.HTTPError as exc:
        if not silent:
            st.error(f"Backend returned an error for {path}: {exc}")
        logger.error("GET %s failed: %s", path, exc)
        return None
    except requests.exceptions.RequestException as exc:
        if not silent:
            st.error(f"Unexpected error calling {path}: {exc}")
        logger.error("GET %s failed: %s", path, exc)
        return None


def backend_post(path: str, json_body: dict | None = None, timeout: float = REQUEST_TIMEOUT_SECONDS) -> dict | None:
    response = None
    try:
        response = requests.post(f"{BACKEND_URL}{path}", json=json_body, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the backend server. Is live_backend.py running? Expected at {BACKEND_URL}")
        logger.error("POST %s failed: connection error", path)
        return None
    except requests.exceptions.Timeout:
        st.error(f"Backend request to {path} timed out.")
        logger.error("POST %s failed: timeout", path)
        return None
    except requests.exceptions.HTTPError as exc:
        try:
            detail = response.json().get("detail", str(exc)) if response is not None else str(exc)
        except Exception:
            detail = str(exc)
        st.error(f"Backend rejected {path}: {detail}")
        logger.error("POST %s failed: %s", path, detail)
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Unexpected error calling {path}: {exc}")
        logger.error("POST %s failed: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "trip_configured" not in st.session_state:
    st.session_state.trip_configured = False
if "telemetry_mode" not in st.session_state:
    st.session_state.telemetry_mode = MODE_SIMULATOR
if "last_config" not in st.session_state:
    st.session_state.last_config = None
if "temp_history" not in st.session_state:
    st.session_state.temp_history = []  # list of {"poll": int, "Cargo Temp (C)": float}
if "montecarlo_result" not in st.session_state:
    st.session_state.montecarlo_result = None
if "slot_matrix" not in st.session_state:
    st.session_state.slot_matrix = None
if "predictive_route" not in st.session_state:
    st.session_state.predictive_route = None

towns_payload = backend_get("/v1/towns", silent=True)
AVAILABLE_TOWNS = towns_payload["towns"] if towns_payload else []

# Shared origin/destination selection — the single source of truth that keeps the
# sidebar Trip Setup and the Journey Prediction tab consistent BOTH ways. Every
# origin/destination selectbox (sidebar and prediction) reads its default from
# these and writes its choice back, so changing the trip in either place is
# reflected in the other on the next rerun. (Kept separate from widget keys to
# avoid Streamlit's duplicate-key restriction across the two locations.)
if AVAILABLE_TOWNS:
    st.session_state.setdefault("active_origin", AVAILABLE_TOWNS[0])
    st.session_state.setdefault(
        "active_destination", AVAILABLE_TOWNS[1] if len(AVAILABLE_TOWNS) > 1 else AVAILABLE_TOWNS[0]
    )


def _town_index(town: str) -> int:
    return AVAILABLE_TOWNS.index(town) if town in AVAILABLE_TOWNS else 0


# ---------------------------------------------------------------------------
# Sidebar — mode + trip setup + business settings
# ---------------------------------------------------------------------------
st.sidebar.title("Fleet Control Panel")

telemetry_mode = st.sidebar.selectbox(
    "Telemetry Mode",
    options=[MODE_HARDWARE, MODE_SIMULATOR],
    index=[MODE_HARDWARE, MODE_SIMULATOR].index(st.session_state.telemetry_mode),
    help=(
        "Hardware mode reads whatever the Traccar Client phone app most recently "
        "pushed to /v1/telematics/incoming — your live position is the trip's "
        "starting point. Simulator mode drives a truck along the computed route "
        "between two towns you choose, at accelerated speed."
    ),
)
if telemetry_mode != st.session_state.telemetry_mode:
    st.session_state.telemetry_mode = telemetry_mode
    st.session_state.trip_configured = False

st.sidebar.divider()
st.sidebar.subheader("Trip Setup")

if not AVAILABLE_TOWNS:
    st.sidebar.warning("Backend unreachable — cannot load destination list.")
else:
    if telemetry_mode == MODE_SIMULATOR:
        origin_town = st.sidebar.selectbox(
            "Starting point", options=AVAILABLE_TOWNS, index=_town_index(st.session_state.active_origin)
        )
        st.session_state.active_origin = origin_town
        destination_town = st.sidebar.selectbox(
            "Destination", options=AVAILABLE_TOWNS, index=_town_index(st.session_state.active_destination)
        )
        st.session_state.active_destination = destination_town
        same_town = origin_town == destination_town
        if same_town:
            st.sidebar.caption("Starting point and destination must differ.")
        if st.sidebar.button("Start Simulated Trip", disabled=same_town, use_container_width=True):
            result = backend_post(
                "/v1/trip/configure",
                {"mode": "simulator", "origin_town": origin_town, "destination_town": destination_town},
            )
            if result is not None:
                st.session_state.trip_configured = True
                st.session_state.last_config = {"origin": origin_town, "destination": destination_town}
                st.session_state.temp_history = []
                st.rerun()
    else:
        destination_town = st.sidebar.selectbox(
            "Destination", options=AVAILABLE_TOWNS, index=_town_index(st.session_state.active_destination)
        )
        st.session_state.active_destination = destination_town
        st.sidebar.caption(
            "Your current position (from Traccar Client) is used as the starting "
            "point automatically. Point Traccar Client's server URL at:\n\n"
            f"`{BACKEND_URL}/v1/telematics/incoming`\n\n"
            "using its OsmAnd / query-string protocol."
        )
        if st.sidebar.button("Set Destination", use_container_width=True):
            result = backend_post(
                "/v1/trip/configure", {"mode": "hardware", "destination_town": destination_town}
            )
            if result is not None:
                st.session_state.trip_configured = True
                st.session_state.last_config = {"origin": None, "destination": destination_town}
                st.session_state.temp_history = []
                st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Cargo Safety Thresholds")
st.sidebar.caption(
    "Set this from your own cold-chain domain knowledge (species, packaging, "
    "ice quality) — it overrides the system default immediately, no restart needed."
)
if "safe_temp_max_c" not in st.session_state or "shipment_value_rand" not in st.session_state:
    current_thresholds = backend_get("/v1/settings/thresholds", silent=True)
    st.session_state.safe_temp_max_c = (
        current_thresholds.get("safe_temp_max_c", 4.0) if current_thresholds else 4.0
    )
    st.session_state.shipment_value_rand = (
        current_thresholds.get("shipment_value_rand", 450_000.0) if current_thresholds else 450_000.0
    )
safe_temp_input = st.sidebar.number_input(
    "Safe Cargo Temp Max (°C)",
    min_value=-20.0,
    max_value=25.0,
    value=float(st.session_state.safe_temp_max_c),
    step=0.5,
    help="Above this temperature, thermal spoilage risk accrues faster than baseline.",
)
if st.sidebar.button("Apply Threshold", use_container_width=True):
    result = backend_post("/v1/settings/thresholds", {"safe_temp_max_c": safe_temp_input})
    if result is not None:
        st.session_state.safe_temp_max_c = result["safe_temp_max_c"]
        st.sidebar.success(f"Safe threshold updated to {result['safe_temp_max_c']:.1f} °C")

st.sidebar.divider()
st.sidebar.subheader("Shipment Value")
st.sidebar.caption(
    "The Rand value of this specific cargo — used to convert accrued spoilage-risk "
    "percentages into a Rand figure everywhere on the dashboard (route comparisons, "
    "\"Value at Risk So Far\", Monte Carlo VaR). Set this per shipment; it overrides "
    "the system default immediately, no restart needed."
)
shipment_value_input = st.sidebar.number_input(
    "Shipment Value (R)",
    min_value=0.01,
    value=float(st.session_state.shipment_value_rand),
    step=10_000.0,
    format="%.0f",
    help="Full value of the cargo on this truck. Expected-loss figures scale directly with this number.",
)
if st.sidebar.button("Apply Shipment Value", use_container_width=True):
    result = backend_post("/v1/settings/thresholds", {"shipment_value_rand": shipment_value_input})
    if result is not None:
        st.session_state.shipment_value_rand = result["shipment_value_rand"]
        st.sidebar.success(f"Shipment value updated to R {result['shipment_value_rand']:,.0f}")

st.sidebar.divider()
st.sidebar.caption(f"Backend: {BACKEND_URL}")
if st.session_state.trip_configured and st.session_state.last_config:
    cfg = st.session_state.last_config
    label = f"{cfg['origin']} -> {cfg['destination']}" if cfg["origin"] else f"Live position -> {cfg['destination']}"
    st.sidebar.success(f"Active trip: {label}")


# ---------------------------------------------------------------------------
# Live map — a single self-contained Leaflet component. It polls the backend
# directly from the browser (fetch, on its own JS timer) and moves the truck
# marker with a CSS transition, entirely independent of Streamlit reruns.
# Route lines are only redrawn when the path actually changes.
# ---------------------------------------------------------------------------
def render_live_map(initial_routing: dict | None, initial_telematics: dict) -> None:
    truck_lat = initial_telematics["lat"]
    truck_lon = initial_telematics["lon"]
    feed_source = initial_telematics.get("feed_source", "")

    # Draw the STABLE planned lines (display_route + the plan's standard route)
    # so the map doesn't re-snap and squiggle as the truck moves. Fall back to
    # the live routes if no plan is available yet.
    if initial_routing:
        _disp = initial_routing.get("display_route") or initial_routing.get("optimized_route")
        _plan = initial_routing.get("trip_plan") or {}
        _std = _plan.get("standard_route") or initial_routing.get("standard_route")
    else:
        _disp = None
        _std = None
    initial_standard = _std["segments"] if _std else []
    initial_optimized = _disp["segments"] if _disp else []
    initial_dest = _disp["coordinates"][-1] if _disp else None
    dest_label = initial_routing.get("destination_town", "") if initial_routing else ""

    html = f"""
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
      #fleet-map {{ height: 520px; border-radius: 8px; }}
      .truck-div-icon {{
        font-size: 22px;
        transition: transform {POLL_INTERVAL_SECONDS * 0.9}s linear;
        text-shadow: 0 0 3px #fff, 0 0 3px #fff;
      }}
    </style>
    <div id="fleet-map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
      const BACKEND = {json.dumps(BACKEND_URL)};
      const OPTIMIZED_COLOR = '{OPTIMIZED_COLOR}';
      const STANDARD_COLOR = '{STANDARD_COLOR}';
      const POLL_MS = {int(POLL_INTERVAL_SECONDS * 1000)};

      const map = L.map('fleet-map').setView([{truck_lat}, {truck_lon}], 7);
      // Keyless OpenStreetMap tiles. CartoDB's basemap CDN now stamps an
      // "API KEY REQUIRED" watermark across tiles for unkeyed use; OSM's
      // standard tiles need no key and never watermark.
      L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
      }}).addTo(map);

      let standardLayer = L.layerGroup().addTo(map);
      let optimizedLayer = L.layerGroup().addTo(map);
      let lastOptimizedKey = null;
      let destMarker = null;

      const truckIcon = L.divIcon({{className: 'truck-div-icon', html: '\\ud83d\\ude9a', iconSize: [26, 26], iconAnchor: [13, 13]}});
      let truckMarker = L.marker([{truck_lat}, {truck_lon}], {{icon: truckIcon}}).addTo(map);
      truckMarker.bindTooltip('{initial_telematics.get("vehicle_id", "Vehicle")} | {truck_lat:.5f}, {truck_lon:.5f} | {feed_source}');

      function drawSegments(segments, layerGroup, color, dashed, routeLabel) {{
        layerGroup.clearLayers();
        segments.forEach(function(seg) {{
          const coords = seg.coords.map(function(c) {{ return [c[1], c[0]]; }});
          const label = (seg.road_name || 'Road') + (seg.overridden ? ' (driver-reported condition)' : '') + ' (' + routeLabel + ')';
          L.polyline(coords, {{
            color: color, weight: dashed ? 4 : 5, opacity: dashed ? 0.6 : 0.9,
            dashArray: dashed ? '6,6' : null
          }}).bindTooltip(label).addTo(layerGroup);
        }});
      }}

      function routeKey(segments) {{
        return JSON.stringify(segments.map(function(s) {{ return s.coords; }}));
      }}

      const initialStandard = {json.dumps(initial_standard)};
      const initialOptimized = {json.dumps(initial_optimized)};
      const initialDest = {json.dumps(initial_dest)};
      const initialDestLabel = {json.dumps(dest_label)};

      if (initialStandard.length) drawSegments(initialStandard, standardLayer, STANDARD_COLOR, true, 'standard route');
      if (initialOptimized.length) {{
        drawSegments(initialOptimized, optimizedLayer, OPTIMIZED_COLOR, false, 'optimized route');
        lastOptimizedKey = routeKey(initialOptimized);
      }}
      if (initialDest) {{
        destMarker = L.marker([initialDest[1], initialDest[0]]).addTo(map);
        destMarker.bindTooltip('Destination: ' + initialDestLabel);
      }}

      async function pollRouting() {{
        try {{
          const res = await fetch(BACKEND + '/v1/routing/truck-01');
          if (!res.ok) return;
          const data = await res.json();
          // Use the STABLE planned lines so the drawn route never re-snaps/
          // squiggles as the truck moves; only a genuine reroute (driver report
          // or trip reconfigure) changes them and triggers a redraw.
          const opt = data.display_route || data.optimized_route;
          const plan = data.trip_plan || {{}};
          const std = plan.standard_route || data.standard_route;
          if (!opt || !std) return;
          const key = routeKey(opt.segments);
          if (key !== lastOptimizedKey) {{
            drawSegments(std.segments, standardLayer, STANDARD_COLOR, true, 'standard route');
            drawSegments(opt.segments, optimizedLayer, OPTIMIZED_COLOR, false, 'optimized route');
            lastOptimizedKey = key;
          }}
          const destCoords = opt.coordinates[opt.coordinates.length - 1];
          if (destMarker) {{
            destMarker.setLatLng([destCoords[1], destCoords[0]]);
            destMarker.setTooltipContent('Destination: ' + (data.destination_town || ''));
          }}
        }} catch (e) {{ console.warn('pollRouting failed:', e); }}
      }}

      async function pollTelemetry() {{
        try {{
          const res = await fetch(BACKEND + '/v1/telematics/truck-01');
          if (!res.ok) {{ console.warn('pollTelemetry non-OK response:', res.status); return; }}
          const data = await res.json();
          if (data.lat == null || data.lon == null) return;
          truckMarker.setLatLng([data.lat, data.lon]);
          truckMarker.setTooltipContent(
            (data.vehicle_id || 'Vehicle') + ' | ' + data.lat.toFixed(5) + ', ' + data.lon.toFixed(5) +
            ' | ' + (data.feed_source || '')
          );
        }} catch (e) {{ console.warn('pollTelemetry failed:', e); }}
      }}

      setInterval(pollTelemetry, POLL_MS);
      setInterval(pollRouting, POLL_MS);
    </script>
    """
    components.html(html, height=540, scrolling=False)


# ---------------------------------------------------------------------------
# Synthetic-stream ingestion — read the three CSVs, generating them on demand
# if they don't exist yet (via vrp_simulation_generator).
# ---------------------------------------------------------------------------
def ensure_synthetic_streams(force: bool = False) -> dict:
    """Returns {"weather": df, "road": df, "sensor": df, "status": str}. If the
    CSVs are missing (or force=True) and vrp_simulation_generator is importable,
    regenerates them first. Never raises — a failure returns empty frames and a
    status message."""
    if pd is None:
        return {"weather": None, "road": None, "sensor": None,
                "status": "pandas is not installed — cannot read the synthetic streams."}

    need_generate = force or not (WEATHER_CSV.exists() and ROAD_CSV.exists() and SENSOR_CSV.exists())
    if need_generate:
        if vrpsim is None:
            return {"weather": None, "road": None, "sensor": None,
                    "status": ("Synthetic stream CSVs not found and vrp_simulation_generator.py "
                               "could not be imported. Run `uv run vrp_simulation_generator.py` first.")}
        try:
            vrpsim.generate_all_streams(hours=72, write_files=True)
        except Exception as exc:  # noqa: BLE001
            return {"weather": None, "road": None, "sensor": None,
                    "status": f"Failed to generate synthetic streams: {exc}"}

    frames: dict = {"status": "ok"}
    for key, path in (("weather", WEATHER_CSV), ("road", ROAD_CSV), ("sensor", SENSOR_CSV)):
        try:
            frames[key] = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            frames[key] = None
            frames["status"] = f"Could not read {path.name}: {exc}"
    return frames


# ---------------------------------------------------------------------------
# Small pure-Python stats helpers for streaming the Monte Carlo distribution
# (no numpy dependency required on the dashboard host).
# ---------------------------------------------------------------------------
def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(xs: list, p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def _peak_temp_hist_data(temps: list, bins: int = 14):
    """Builds an ordered peak-cargo-temperature histogram for a live bar chart.
    Returns a pandas DataFrame (ordered by temperature bin) when pandas is
    available, else an ordered dict; None if there's nothing to plot."""
    if not temps:
        return None
    lo = min(temps)
    hi = max(temps)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    width = (hi - lo) / bins
    labels = [f"{lo + i * width:.1f}" for i in range(bins)]
    counts = [0] * bins
    for t in temps:
        idx = min(int((t - lo) / width), bins - 1)
        counts[idx] += 1
    if pd is not None:
        return pd.DataFrame({"Peak cargo °C": labels, "Trials": counts}).set_index("Peak cargo °C")
    return {labels[i]: counts[i] for i in range(bins)}


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_tracker, tab_report_desk, tab_prediction, tab_ai_report, tab_weather = st.tabs(
    [
        "🛰️ Customer Route Tracker",
        "📝 Driver Reporting Desk",
        "🔮 Journey Prediction & Scheduling",
        "📊 Report",
        "🌤️ Weather",
    ]
)


# ===========================================================================
# TAB 1 — Customer Route Tracker
# ===========================================================================
with tab_tracker:
    st.title("🛰️ Customer Route Tracker")
    st.caption("Live vehicle position, cargo condition, and compressor workload — updating as the truck moves.")

    if not st.session_state.trip_configured:
        st.info("Set a starting point and destination in the sidebar, then start the trip to see live tracking.")
    else:
        initial_routing = backend_get("/v1/routing/truck-01", silent=True)
        initial_telematics = backend_get("/v1/telematics/truck-01", silent=True)

        st.subheader("Live Vehicle Map")
        if initial_telematics is None or initial_telematics.get("lat") is None:
            st.warning("No position available yet. Make sure live_backend.py is running (and, in hardware mode, that Traccar Client has sent a fix).")
        else:
            render_live_map(initial_routing, initial_telematics)
            legend_cols = st.columns(2)
            legend_cols[0].markdown(
                f"<span style='color:{OPTIMIZED_COLOR}'>&#9644;</span> Optimized (spoilage-aware) route",
                unsafe_allow_html=True,
            )
            legend_cols[1].markdown(
                f"<span style='color:{STANDARD_COLOR}'>&#9644;</span> Standard (time-only) route",
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader("Live Vehicle Telemetry")

        # One-time layout scaffold (see the original dashboard's note): built
        # once per outer run, updated in place by the fragment so the metric
        # cards never flash on each poll.
        tracker_ui: dict = {"warning": st.empty()}
        card_cols = st.columns(4)
        tracker_ui["vehicle"] = card_cols[0].empty()
        tracker_ui["coords"] = card_cols[1].empty()
        tracker_ui["cargo_temp"] = card_cols[2].empty()
        tracker_ui["compressor"] = card_cols[3].empty()
        tracker_ui["status_banner"] = st.empty()
        tracker_ui["action_panel"] = st.empty()
        tracker_ui["chart_header"] = st.empty()
        tracker_ui["chart"] = st.empty()
        tracker_ui["chart_caption"] = st.empty()
        tracker_ui["feed"] = st.empty()
        tracker_ui["updated"] = st.empty()

        @st.fragment(run_every=POLL_INTERVAL_SECONDS)
        def tracker_metrics_view():
            telematics = backend_get("/v1/telematics/truck-01", silent=True)
            if telematics is None or telematics.get("lat") is None:
                tracker_ui["warning"].warning("No telemetry received yet.")
                # A placeholder created in the scaffold must be written on every
                # run (incl. the first) or a later fragment-only rerun raises
                # "container not written to during the initial run".
                tracker_ui["chart"].line_chart({"Cargo Temp (°C)": [float(st.session_state.safe_temp_max_c)]}, height=220)
                return
            tracker_ui["warning"].empty()

            vehicle_id = telematics.get("vehicle_id", "—")
            lat = telematics.get("lat", 0.0)
            lon = telematics.get("lon", 0.0)
            cargo_temp_c = telematics.get("cargo_temp_c", 0.0)
            compressor = telematics.get("compressor_load_pct")
            temp_status = telematics.get("cargo_temp_status", "Unknown")
            feed_source = telematics.get("feed_source", "UNKNOWN")

            safe_baseline = float(st.session_state.safe_temp_max_c)

            tracker_ui["vehicle"].metric("Active Vehicle ID", vehicle_id)
            tracker_ui["coords"].metric("Current GPS Coordinates", f"{lat:.4f}, {lon:.4f}")
            tracker_ui["cargo_temp"].metric(
                "Emulated Cargo Temp", f"{cargo_temp_c:.1f} °C",
                delta=f"{cargo_temp_c - safe_baseline:+.1f} °C vs baseline",
                delta_color="inverse",
                help=f"Safe baseline is {safe_baseline:.1f} °C (set it in the sidebar).",
            )
            tracker_ui["compressor"].metric(
                "Compressor Workload",
                f"{compressor:.0f}%" if compressor is not None else "n/a",
                help="Reefer condenser load. Pinned to 100% while the vehicle is stationary (queue/gridlock) to model condenser strain in desert heat.",
            )

            # Explicit baseline-crossing alert (spec: warn the user the moment
            # cargo passes the safe baseline they set). Critical/Elevated both
            # mean the cargo is above baseline and at risk; Nominal is at/below.
            if temp_status == "Awaiting Motion":
                tracker_ui["status_banner"].info(f"⏸️ Truck parked at {cargo_temp_c:.1f} °C — cargo metrics frozen until it departs.")
            elif temp_status == "Critical":
                tracker_ui["status_banner"].error(
                    f"🛑 CARGO AT RISK — temperature {cargo_temp_c:.1f} °C is CRITICALLY above your "
                    f"{safe_baseline:.1f} °C safe baseline (+{cargo_temp_c - safe_baseline:.1f} °C). "
                    f"Thermal spoilage risk is accelerating."
                )
            elif temp_status == "Elevated":
                tracker_ui["status_banner"].warning(
                    f"⚠️ CARGO AT RISK — temperature {cargo_temp_c:.1f} °C has passed your "
                    f"{safe_baseline:.1f} °C safe baseline (+{cargo_temp_c - safe_baseline:.1f} °C). "
                    f"Spoilage risk is now accruing faster than baseline."
                )
            else:
                tracker_ui["status_banner"].success(
                    f"✅ Cargo temperature nominal at {cargo_temp_c:.1f} °C (safe baseline {safe_baseline:.1f} °C)."
                )

            # --- DRIVER ACTION ITEMS (driver's view, on the road) ------------
            # When the load passes the safe baseline, show the concrete steps the
            # driver should take right now. This is the driver-facing half of the
            # cargo-risk trigger; the business office gets the fuller AI mitigation
            # memo in the Report tab off the same event.
            if telematics.get("cargo_at_risk"):
                items = telematics.get("driver_action_items", [])
                level = telematics.get("cargo_risk_level", 1)
                header = {
                    1: "⚠️ DRIVER ACTION REQUIRED — cargo just over the safe temperature",
                    2: "⚠️ DRIVER ACTION REQUIRED — cargo temperature RISING",
                    3: "🔴 DRIVER ACTION REQUIRED — SERIOUS cargo temperature breach",
                    4: "🛑 DRIVER ACTION — EMERGENCY: cargo may be compromised",
                }.get(level, "🚨 DRIVER ACTION REQUIRED — cargo above the safe temperature")
                # Render the whole list as ONE markdown block (not a per-item
                # loop) so the panel can never stack/duplicate on a fragment
                # rerun. The panel keeps showing until the cargo cools back to
                # the baseline, at which point cargo_at_risk clears.
                with tracker_ui["action_panel"].container():
                    (st.error if level >= 3 else st.warning)(header)
                    st.markdown("\n".join(f"- {it}" for it in items))
            else:
                tracker_ui["action_panel"].empty()

            # --- Live cold-chain temperature chart with the safe baseline -----
            # Appends one point per poll while the trip is genuinely underway
            # (not arrived, and — in hardware mode — actually moving), so a
            # parked or finished truck doesn't keep scrolling a flat line.
            arrived = telematics.get("arrived", False)
            awaiting_motion = telematics.get("awaiting_motion", False)
            if not arrived and not awaiting_motion:
                st.session_state.temp_history.append({
                    "poll": len(st.session_state.temp_history),
                    "Cargo Temp (°C)": cargo_temp_c,
                })
                st.session_state.temp_history = st.session_state.temp_history[-200:]

            tracker_ui["chart_header"].subheader("Cold-Chain Temperature vs Safe Baseline")
            history = st.session_state.temp_history
            if len(history) >= 1:
                if pd is not None:
                    chart_df = pd.DataFrame(history).set_index("poll")
                    chart_df["Safe Baseline (°C)"] = safe_baseline
                    tracker_ui["chart"].line_chart(
                        chart_df[["Cargo Temp (°C)", "Safe Baseline (°C)"]], height=220,
                    )
                else:
                    tracker_ui["chart"].line_chart(
                        {"Cargo Temp (°C)": [h["Cargo Temp (°C)"] for h in history]}, height=220,
                    )
                if arrived:
                    tracker_ui["chart_caption"].caption(
                        f"🏁 Trip complete — cargo temperature history frozen at arrival. "
                        f"Flat line = your {safe_baseline:.1f} °C safe baseline; any excursion above it is spoilage-risk exposure."
                    )
                else:
                    tracker_ui["chart_caption"].caption(
                        f"Live cargo temperature (updating as the truck moves) against your {safe_baseline:.1f} °C safe baseline. "
                        "When the cargo line rises above the baseline, the banner above turns to a risk warning."
                    )
            else:
                tracker_ui["chart"].line_chart({"Cargo Temp (°C)": [cargo_temp_c]}, height=220)
                tracker_ui["chart_caption"].caption("Collecting cargo temperature history…")

            if feed_source == FEED_SOURCE_REAL:
                tracker_ui["feed"].success(f"Data Feed Source: {feed_source}")
            elif feed_source == FEED_SOURCE_SIMULATED:
                arrived = telematics.get("arrived", False)
                tracker_ui["feed"].info(f"Data Feed Source: {feed_source}" + ("  |  Arrived at destination" if arrived else ""))
            else:
                tracker_ui["feed"].warning(f"Data Feed Source: {feed_source}")

            tracker_ui["updated"].caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

        tracker_metrics_view()


# ===========================================================================
# TAB 2 — Driver Reporting Desk
# ===========================================================================
with tab_report_desk:
    st.title("📝 Driver Reporting Desk")
    st.caption(
        "Submit a real-time road condition update. Reports override the routing "
        "impedance for the matched segment immediately — the next routing "
        "recalculation will reroute around bad conditions automatically."
    )

    default_lat = None
    default_lon = None
    last_telematics = backend_get("/v1/telematics/truck-01", silent=True)
    if last_telematics is not None:
        default_lat = last_telematics.get("lat")
        default_lon = last_telematics.get("lon")

    # Outside the form on purpose: a checkbox inside st.form wouldn't reveal the
    # storm dropdown until submit. Placed here, toggling it reruns immediately.
    is_storm_report = st.checkbox(
        "⛈️ Report Severe Storm / Active Washout Structural Damage",
        help=(
            "Use this for active infrastructure failure during severe weather — flash "
            "flooding, mud traps, washed-out road beds — distinct from routine wear. "
            "Storm reports are pinned to the highest roughness tier (IRI 6.0+), forcing "
            "an immediate detour on the dispatch map's next refresh."
        ),
    )

    with st.form("field_report_form", clear_on_submit=True):
        reporter_role = st.selectbox(
            "Who are you reporting as?",
            options=["Driver", "Fisherman", "Cooperative Supervisor"],
        )

        col_lat, col_lon = st.columns(2)
        report_lat = col_lat.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(default_lat) if default_lat is not None else -29.0,
            format="%.6f",
        )
        report_lon = col_lon.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(default_lon) if default_lon is not None else 20.0,
            format="%.6f",
        )
        st.caption(
            "Defaults to the truck's last known position. Adjust if you're "
            "reporting a different location along the route."
        )

        if is_storm_report:
            road_condition = st.selectbox(
                "Storm / Washout Condition",
                options=[
                    "Flash Flood Mud Trap",
                    "Gravel Bed Erosion",
                    "Structural Road Washout",
                ],
                help="Severe, storm-driven infrastructure damage — all three map to IRI 6.0+.",
            )
        else:
            road_condition = st.selectbox(
                "Road Condition",
                options=[
                    "Smooth Tarmac",
                    "Corrugated / Rough Gravel",
                    "Severe Potholes",
                    "Impassable / Washed Out",
                ],
            )

        actual_speed = st.number_input(
            "Actual speed you're able to travel (km/h)",
            min_value=1.0,
            max_value=160.0,
            value=60.0,
            step=5.0,
        )

        submitted = st.form_submit_button("Submit Field Report")

    if submitted:
        result = backend_post(
            "/v1/reports/submit",
            json_body={
                "reporter_role": reporter_role,
                "lat": report_lat,
                "lon": report_lon,
                "road_condition": road_condition,
                "actual_speed": actual_speed,
            },
        )
        if result is not None:
            st.success(
                f"Report received and matched to road segment {result['matched_segment']} "
                f"({result['distance_from_report_km']:.2f} km from your reported position)."
            )
            st.json(result["applied_override"])
            st.info(
                f"Total active field reports currently affecting routing: "
                f"{result['active_report_count']}"
            )
            logger.info(
                "Field report submitted: role=%s condition=%s segment=%s",
                reporter_role, road_condition, result["matched_segment"],
            )


# ===========================================================================
# TAB 3 — Journey Prediction & Scheduling
# ===========================================================================
with tab_prediction:
    st.title("🔮 Journey Prediction & Scheduling")
    st.caption(
        "Plan a future departure up to 7 days out: preview synthetic sensor streams "
        "frame-by-frame, run a Monte Carlo stochastic risk test, and compare departure "
        "slots on their cold-chain failure probability."
    )

    # --- Temporal controller (7-day future window) --------------------------
    st.subheader("Departure Window")
    today = date.today()
    # Contiguous 7-day-forward window rendered as an explicit day list — no
    # greyed-out calendar days, no month jumps: every option is selectable and
    # adjacent to the next. (The old calendar's min/max greying is what read as
    # "some dates not available / jumps".)
    day_options = [today + timedelta(days=i) for i in range(PREDICTION_MAX_HORIZON_DAYS + 1)]

    def _fmt_day(d):
        n = (d - today).days
        tag = "Today" if n == 0 else ("Tomorrow" if n == 1 else f"+{n} days")
        return f"{d.strftime('%a %d %b %Y')} · {tag}"

    ctrl_cols = st.columns([2, 1])
    dep_date = ctrl_cols[0].selectbox(
        "Departure date (next 7 days)", options=day_options, index=0, format_func=_fmt_day,
        help="Contiguous 7-day forward window (the OpenWeatherMap free-tier forecast horizon).",
    )
    dep_time = ctrl_cols[1].time_input("Departure time", value=dt_time(hour=6, minute=0))

    pred_origin = None
    pred_destination = None
    if AVAILABLE_TOWNS:
        od_cols = st.columns(2)
        pred_origin = od_cols[0].selectbox(
            "Prediction origin", options=AVAILABLE_TOWNS, index=_town_index(st.session_state.active_origin)
        )
        st.session_state.active_origin = pred_origin
        pred_destination = od_cols[1].selectbox(
            "Prediction destination", options=AVAILABLE_TOWNS, index=_town_index(st.session_state.active_destination)
        )
        st.session_state.active_destination = pred_destination
    else:
        st.warning("Backend unreachable — cannot load town list for prediction.")

    departure_dt = datetime.combine(dep_date, dep_time)
    # The backend rejects a target in the past or > 7 days out. If the chosen
    # local time has already passed today, clamp the API target a couple of
    # minutes into the future so the request still succeeds; the caption keeps
    # showing exactly what the user picked.
    _now_utc = datetime.utcnow()
    api_target_dt = departure_dt if departure_dt > _now_utc + timedelta(minutes=1) else _now_utc + timedelta(minutes=2)
    api_target_iso = api_target_dt.isoformat()
    st.caption(f"Planned departure: **{departure_dt.strftime('%A %d %B %Y, %H:%M')}** (interpreted as UTC by the backend).")
    if st.session_state.trip_configured and st.session_state.last_config:
        cfg = st.session_state.last_config
        active_od = f"{cfg['origin']} → {cfg['destination']}" if cfg["origin"] else f"Live position → {cfg['destination']}"
        st.caption(
            f"ℹ️ Defaults match your active trip (**{active_od}**). Prediction, the Customer Route Tracker, "
            "and the live hardware feed all run the **same** spoilage-optimized shortest-path engine on the "
            "same road graph, so the optimal route shown here is consistent with the one being tracked live "
            "(the live tracker simply re-solves it from the truck's current position instead of the origin town)."
        )

    st.divider()

    # --- Ingestion ticker over the synthetic Stream A/B/C CSVs --------------
    st.subheader("🎞️ Synthetic Telemetry Ingestion Ticker")
    st.caption(
        "Streamed frame-by-frame from synthetic_weather_forecast.csv, "
        "synthetic_road_conditions.csv and synthetic_cold_sensors.csv, presented as if "
        "arriving live from field hardware. Generated by vrp_simulation_generator.py."
    )

    regen = st.button("🔄 (Re)generate synthetic streams")
    streams = ensure_synthetic_streams(force=regen)
    if streams["status"] != "ok":
        st.warning(streams["status"])
    else:
        weather_df = streams["weather"]
        road_df = streams["road"]
        sensor_df = streams["sensor"]
        max_frame = min(
            len(weather_df) if weather_df is not None else 0,
            len(road_df) if road_df is not None else 0,
            len(sensor_df) if sensor_df is not None else 0,
        )
        if max_frame < 1:
            st.warning("Synthetic streams are empty — try regenerating them.")
        else:
            frame = st.slider(
                "🎞️ Advance Predictive Journey Frame",
                min_value=0, max_value=max_frame - 1, value=0,
                help="Scrub through the synthetic forecast hour-by-hour; the metric cards below update as if streaming live from hardware.",
            )
            w_row = weather_df.iloc[frame]
            r_row = road_df.iloc[frame]
            s_row = sensor_df.iloc[frame]

            st.caption(f"Frame {frame} · forecast timestamp {w_row['timestamp_iso']}")
            frame_cols = st.columns(4)
            frame_cols[0].metric("Ambient Temp", f"{float(w_row['ambient_temp_c']):.1f} °C")
            frame_cols[1].metric("Rain", f"{float(w_row['rain_mm_per_hr']):.1f} mm/hr")
            frame_cols[2].metric("Cargo Temp (Stream C)", f"{float(s_row['cargo_temp_c']):.1f} °C")
            frame_cols[3].metric("Compressor Load", f"{float(s_row['compressor_load_pct']):.0f}%")

            frame_cols2 = st.columns(4)
            frame_cols2[0].metric("Road Condition", str(r_row["condition_label"]))
            frame_cols2[1].metric("Safe Speed", f"{float(r_row['safe_speed_kmh']):.0f} km/h")
            frame_cols2[2].metric("IRI (Roughness)", f"{float(r_row['iri']):.2f}")
            frame_cols2[3].metric("G-Force RMS", f"{float(s_row['g_force_rms']):.2f} g")

            if bool(r_row["failure_event"]):
                st.error(f"🛑 Failure event this frame — {r_row['condition_label']} (trigger: {r_row['trigger']}). Edge weight ×{float(r_row['edge_weight_multiplier']):.2f}; router reroutes to an optimal path.")
            elif bool(s_row["spoilage_breach"]):
                st.warning(f"⚠️ Cargo has breached −18.0 °C at this frame (cargo temp {float(s_row['cargo_temp_c']):.1f} °C).")
            else:
                st.success("✅ Nominal conditions this frame.")

    st.divider()

    # --- Monte Carlo stochastic risk test ----------------------------------
    st.subheader("🎲 Monte Carlo Stochastic Risk Test")
    st.caption(
        "Runs the backend/vrp Monte Carlo solver: per-trial thermal variance N(0, 2.5 °C), "
        "Vioolsdrift customs delay 300 min + N(0, 45), and a 5% infrastructure-shock "
        "probability, tracking cargo temperature via a Newtonian cooling model."
    )
    mc_cols = st.columns([1, 2])
    mc_iterations = mc_cols[0].select_slider(
        "Trials", options=[200, 500, 1000, 2000], value=1000,
        help="More trials = tighter estimates but a slower run.",
    )

    if st.button("🎲 Run Monte Carlo Stochastic Risk Test", use_container_width=True):
        if not (pred_origin and pred_destination):
            st.error("Select a prediction origin and destination first.")
        elif pred_origin == pred_destination:
            st.error("Origin and destination must differ.")
        else:
            # Stream the simulation: split the run into batches and update a
            # progress bar, the metric cards, and a growing peak-temperature
            # histogram after each batch, so the user watches the distribution
            # build in real time instead of staring at a frozen button.
            total = int(mc_iterations)
            n_batches = max(1, min(10, total // 100))  # each batch ≥ 100 (backend minimum)
            base = total // n_batches
            batch_sizes = [base] * (n_batches - 1) + [total - base * (n_batches - 1)]

            st.markdown(f"**{pred_origin} → {pred_destination}** · streaming {total:,} trials")
            progress = st.progress(0.0, text="Starting Monte Carlo…")
            live_cols = st.columns(3)
            live_dur = live_cols[0].empty()
            live_var = live_cols[1].empty()
            live_prob = live_cols[2].empty()
            live_hist = st.empty()
            live_caption = st.empty()

            acc_peak: list = []
            acc_loss: list = []
            acc_journey: list = []
            last_res = None
            run_ok = True
            done = 0

            for bsize in batch_sizes:
                res = backend_post(
                    "/v1/simulation/monte-carlo-risk",
                    json_body={
                        "origin_town": pred_origin,
                        "destination_town": pred_destination,
                        "target_datetime": api_target_iso,
                        "shipment_value_rand": float(st.session_state.shipment_value_rand),
                        "iterations": int(bsize),
                        "return_samples": True,
                    },
                    timeout=120.0,
                )
                if res is None or res.get("status") != "success":
                    run_ok = False
                    st.error((res or {}).get("message", "Monte Carlo simulation failed."))
                    break
                last_res = res
                acc_peak += res.get("sample_peak_temps_c", [])
                acc_loss += res.get("sample_losses_rand", [])
                acc_journey += res.get("sample_journey_hours", [])
                done += bsize

                exp_min = _mean(acc_journey) * 60.0
                var95 = _percentile(acc_loss, 95)
                prob = (sum(1 for t in acc_peak if t > -18.0) / len(acc_peak) * 100.0) if acc_peak else 0.0
                progress.progress(done / total, text=f"Simulating cold-chain trials… {done:,}/{total:,}")
                live_dur.metric("Expected Journey Duration", f"{exp_min:.0f} min")
                live_var.metric("95% Value at Risk", f"R {var95:,.0f}")
                live_prob.metric("Prob. of Cold-Chain Failure", f"{prob:.1f}%")
                hist_data = _peak_temp_hist_data(acc_peak)
                if hist_data is not None:
                    live_hist.bar_chart(hist_data, height=200)
                live_caption.caption(
                    f"Peak cargo-temperature distribution over {len(acc_peak):,} trials so far — "
                    "anything to the right of −18.0 °C is a spoiled load."
                )

            if run_ok and last_res is not None:
                progress.progress(1.0, text=f"Done — {total:,} trials complete.")
                aggregated = dict(last_res)
                aggregated["origin_town"] = pred_origin
                aggregated["destination_town"] = pred_destination
                aggregated["expected_journey_time_mins"] = round(_mean(acc_journey) * 60.0, 1)
                aggregated["expected_journey_time_hours"] = round(_mean(acc_journey), 2)
                aggregated["value_at_risk_95_rand"] = round(_percentile(acc_loss, 95), -2) if acc_loss else 0.0
                aggregated["prob_total_spoilage_pct"] = (
                    round(sum(1 for t in acc_peak if t > -18.0) / len(acc_peak) * 100.0, 2) if acc_peak else 0.0
                )
                aggregated["mean_peak_cargo_temp_c"] = round(_mean(acc_peak), 2) if acc_peak else 0.0
                aggregated["worst_peak_cargo_temp_c"] = round(max(acc_peak), 2) if acc_peak else 0.0
                aggregated["iterations"] = len(acc_journey)
                aggregated["sample_peak_temps_c"] = acc_peak  # kept for the persistent histogram
                st.session_state.montecarlo_result = aggregated
                st.rerun()
            else:
                st.session_state.montecarlo_result = None

    mc = st.session_state.montecarlo_result
    if mc is not None:
        st.markdown(f"**{mc.get('origin_town','')} → {mc.get('destination_town','')}** · {mc.get('iterations','?')} trials")
        result_cols = st.columns(3)
        result_cols[0].metric("Expected Journey Duration", f"{mc['expected_journey_time_mins']:.0f} min",
                              help=f"{mc['expected_journey_time_hours']:.2f} hours (drive {mc.get('route_total_drive_hours', 0):.1f} h + idle).")
        result_cols[1].metric("95% Value at Risk", f"R {mc['value_at_risk_95_rand']:,.0f}",
                              help="95th-percentile Rand loss across all trials, against the shipment value.")
        result_cols[2].metric("Probability of Cold-Chain Failure", f"{mc['prob_total_spoilage_pct']:.1f}%",
                              help=f"Share of trials where peak cargo temp breached {mc.get('spoilage_breach_temp_c', -18.0):.1f} °C.")

        detail_cols = st.columns(3)
        detail_cols[0].metric("Mean / Worst Peak Cargo Temp", f"{mc['mean_peak_cargo_temp_c']:.1f} / {mc['worst_peak_cargo_temp_c']:.1f} °C")
        detail_cols[1].metric("Forecast Ambient", f"{mc['forecast_ambient_temp_c']:.1f} °C")
        detail_cols[2].metric("Route Crosses Border", "Yes" if mc.get("route_crosses_border") else "No")

        # Final peak-cargo-temperature distribution across all trials.
        final_samples = mc.get("sample_peak_temps_c")
        if final_samples:
            st.caption(
                "Peak cargo-temperature distribution across all trials — bars to the right of "
                "−18.0 °C are trials where the cold chain was breached:"
            )
            hist_data = _peak_temp_hist_data(final_samples)
            if hist_data is not None:
                st.bar_chart(hist_data, height=220)

        prob = mc["prob_total_spoilage_pct"]
        if prob >= 25.0:
            st.error(f"REJECT DEPARTURE: {prob:.0f}% Spoilage Failure Risk at this slot.")
        elif prob >= 10.0:
            st.warning(f"CAUTION: {prob:.0f}% Spoilage Risk — consider a cooler departure slot.")
        else:
            st.success(f"APPROVE DEPARTURE: {prob:.1f}% Failure Risk.")

        if prob <= 0.001:
            crosses = "Yes" if mc.get("route_crosses_border") else "No"
            st.info(
                f"ℹ️ **Why 0% here:** this route is driving-only (crosses Vioolsdrift border: {crosses}) in mild "
                f"ambient (~{mc.get('forecast_ambient_temp_c', '?')} °C), so the reefer holds the cargo below "
                "−18 °C the whole way — a genuinely low-risk shipment, not a broken result. To see risk accrue, "
                "pick a destination whose route crosses the border, choose a hotter departure slot, or route over "
                "unpaved segments (which raise the infrastructure-shock chance)."
            )

    st.divider()

    # --- Predictive route + hazard map (streamlit_folium) ------------------
    st.subheader("🗺️ Forecast-Aware Predictive Route")
    st.caption("Samples the 7-day forecast along the corridor and detours around any extreme-heat or storm-washout hazard zone.")
    if st.button("Compute forecast-aware route"):
        if not (pred_origin and pred_destination):
            st.error("Select a prediction origin and destination first.")
        elif pred_origin == pred_destination:
            st.error("Origin and destination must differ.")
        else:
            with st.spinner("Sampling forecast and re-solving the route..."):
                st.session_state.predictive_route = backend_post(
                    "/v1/simulation/predictive-route",
                    json_body={
                        "origin_town": pred_origin,
                        "destination_town": pred_destination,
                        "target_datetime": api_target_iso,
                    },
                    timeout=120.0,
                )

    pr = st.session_state.predictive_route
    if pr is not None and "predictive_route" in pr:
        pr_cols = st.columns(3)
        pr_cols[0].metric("Hazard Zones on Corridor", pr.get("hazard_count", 0))
        pr_cols[1].metric("Rerouted Around Hazard", "Yes" if pr.get("rerouted_around_hazard") else "No")
        pr_cols[2].metric("Extra Time vs Baseline", f"{pr['delta_vs_baseline']['extra_time_mins']:+.0f} min")

        if HAVE_FOLIUM:
            coords = pr["predictive_route"]["coordinates"]  # [lon, lat]
            if coords:
                mid_lat = sum(c[1] for c in coords) / len(coords)
                mid_lon = sum(c[0] for c in coords) / len(coords)
                # Keyless OpenStreetMap tiles — same basemap as the Customer
                # Route Tracker, and no CartoDB "API KEY REQUIRED" watermark.
                fmap = folium.Map(location=[mid_lat, mid_lon], zoom_start=7, tiles="OpenStreetMap")
                folium.PolyLine(
                    [[c[1], c[0]] for c in coords], color=OPTIMIZED_COLOR, weight=5, opacity=0.9,
                    tooltip="Forecast-aware optimized route",
                ).add_to(fmap)
                base_coords = pr["baseline_optimized_route"]["coordinates"]
                if base_coords and pr.get("rerouted_around_hazard"):
                    folium.PolyLine(
                        [[c[1], c[0]] for c in base_coords], color=STANDARD_COLOR, weight=4, opacity=0.6,
                        dash_array="6,6", tooltip="Baseline route (before hazard reroute)",
                    ).add_to(fmap)
                for hz in pr.get("hazard_zones", []):
                    folium.CircleMarker(
                        location=[hz["lat"], hz["lon"]], radius=8, color=HAZARD_COLOR, fill=True,
                        fill_opacity=0.7,
                        tooltip=f"{hz['reason']} — {hz['forecast_ambient_temp_c']:.0f}°C / {hz['forecast_rain_mm_per_hr']:.1f}mm",
                    ).add_to(fmap)
                # Origin/destination pins use a self-contained emoji DivIcon
                # instead of folium's default marker. The default marker loads
                # its PNG (marker-icon.png) from an external CDN, which is
                # blocked inside the streamlit_folium iframe and renders as a
                # broken-image square — a DivIcon is pure inline HTML, so it
                # always shows.
                # Icons consistent with the Customer Route Tracker: a 🚚 truck
                # emoji at the origin (same glyph as the live vehicle marker)
                # and a blue teardrop pin at the destination (an inline SVG that
                # matches Leaflet's default blue marker used on the tracker, but
                # self-contained so it can't break like an external-PNG marker).
                _truck_icon = folium.DivIcon(
                    html='<div style="font-size:26px;line-height:26px;text-shadow:0 0 3px #fff,0 0 3px #fff;">🚚</div>',
                    icon_size=(28, 28), icon_anchor=(14, 14),
                )
                _blue_pin_svg = (
                    '<svg xmlns="http://www.w3.org/2000/svg" width="26" height="38" viewBox="0 0 26 38">'
                    '<path d="M13 0C5.8 0 0 5.8 0 13c0 9.8 13 25 13 25s13-15.2 13-25C26 5.8 20.2 0 13 0z" '
                    'fill="#2A81CB" stroke="#ffffff" stroke-width="1.6"/>'
                    '<circle cx="13" cy="13" r="5.2" fill="#ffffff"/></svg>'
                )
                _dest_icon = folium.DivIcon(html=_blue_pin_svg, icon_size=(26, 38), icon_anchor=(13, 38))
                folium.Marker(
                    [coords[0][1], coords[0][0]],
                    tooltip=f"Origin: {pr.get('origin_town','')}",
                    icon=_truck_icon,
                ).add_to(fmap)
                folium.Marker(
                    [coords[-1][1], coords[-1][0]],
                    tooltip=f"Destination: {pr.get('destination_town','')}",
                    icon=_dest_icon,
                ).add_to(fmap)
                st_folium(fmap, height=460, use_container_width=True, returned_objects=[])
        else:
            st.info("Install `streamlit-folium` and `folium` to see the hazard map. Route stats are shown above.")

    st.divider()

    # --- Executive recommendation matrix (Monte Carlo per departure slot) ---
    st.subheader("🧮 Executive Departure-Slot Recommendation Matrix")
    st.caption(
        "Runs the Monte Carlo solver for three departure windows (Now, +12h, +24h) and "
        "ranks them by cold-chain failure probability, with a plain-language call per slot."
    )
    matrix_iters = st.select_slider(
        "Trials per slot", options=[200, 400, 800], value=400,
        help="Three slots are simulated, so a lower per-slot count keeps this responsive.",
    )
    if st.button("Build recommendation matrix", use_container_width=True):
        if not (pred_origin and pred_destination):
            st.error("Select a prediction origin and destination first.")
        elif pred_origin == pred_destination:
            st.error("Origin and destination must differ.")
        else:
            rows = []
            slots = [("Now", 5), ("+12h", 12 * 60), ("+24h", 24 * 60)]
            with st.spinner("Simulating each departure slot..."):
                for label, offset_min in slots:
                    slot_dt = datetime.utcnow() + timedelta(minutes=offset_min)
                    slot_result = backend_post(
                        "/v1/simulation/monte-carlo-risk",
                        json_body={
                            "origin_town": pred_origin,
                            "destination_town": pred_destination,
                            "target_datetime": slot_dt.isoformat(),
                            "shipment_value_rand": float(st.session_state.shipment_value_rand),
                            "iterations": int(matrix_iters),
                        },
                        timeout=120.0,
                    )
                    if slot_result is not None and slot_result.get("status") == "success":
                        prob = slot_result["prob_total_spoilage_pct"]
                        if prob >= 25.0:
                            rec = f"REJECT DEPARTURE: {prob:.0f}% Spoilage Failure Risk"
                        elif prob >= 10.0:
                            rec = f"CAUTION: {prob:.0f}% Spoilage Risk"
                        else:
                            rec = f"APPROVE DEPARTURE: {prob:.1f}% Failure Risk"
                        rows.append({
                            "Departure Slot": label,
                            "Failure Probability (%)": round(prob, 1),
                            "95% VaR (R)": round(slot_result["value_at_risk_95_rand"]),
                            "Expected Duration (min)": round(slot_result["expected_journey_time_mins"]),
                            "Forecast Ambient (°C)": round(slot_result["forecast_ambient_temp_c"], 1),
                            "Recommendation": rec,
                        })
            st.session_state.slot_matrix = rows

    if st.session_state.slot_matrix:
        rows = st.session_state.slot_matrix
        if pd is not None:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.table(rows)
        best = min(rows, key=lambda r: r["Failure Probability (%)"])
        st.success(f"✅ Recommended departure slot: **{best['Departure Slot']}** — lowest failure probability at {best['Failure Probability (%)']:.1f}%.")


# ===========================================================================
# TAB 4 — Report (AI-generated business-value memo)
# ===========================================================================
with tab_ai_report:
    st.title("📊 AI-Generated Business Value Report")
    st.caption(
        "Reads the live routing, cargo-risk, and weather metrics and asks a locally hosted "
        "Qwen2.5-1.5B model to turn them into a comprehensive business-strategy memo. Runs "
        "entirely on-box; the first report may take a minute or two while the model loads, then "
        "subsequent reports are fast (the model stays cached)."
    )

    strategy_routing = backend_get("/v1/routing/truck-01", silent=True)
    strategy_telematics = backend_get("/v1/telematics/truck-01", silent=True)

    is_hardware_mode = st.session_state.telemetry_mode == MODE_HARDWARE

    # --- Cargo-risk trigger (business/office view) --------------------------
    # Fires off the SAME event the driver sees on the Customer Route Tracker:
    # when the load passes its safe temperature, the office gets a red alert, the
    # exact action items the driver is seeing, and a one-click AI memo that leads
    # with immediate mitigation (driver + dispatch actions).
    if st.session_state.trip_configured and (strategy_telematics or {}).get("cargo_at_risk"):
        st.error(
            "🚨 CARGO AT RISK — the load has passed its safe temperature. The driver has the action "
            "items on their tracker; generate the mitigation memo for the office below."
        )
        risk_action_items = (strategy_telematics or {}).get("driver_action_items", [])
        with st.expander("Action items the driver is seeing right now (Customer Route Tracker)", expanded=True):
            st.markdown("\n".join(f"- {_item}" for _item in risk_action_items))
        if st.button("🚨 Generate Cargo-Risk Mitigation Report (AI)", use_container_width=True, key="cargo_mitigation_btn"):
            _amb = (strategy_telematics or {}).get("ambient_weather") or {}
            _cfg = st.session_state.last_config or {}
            alert_payload = {
                "origin": _cfg.get("origin") or "Live position",
                "destination": _cfg.get("destination") or "Destination",
                "standard_time_mins": 0.0,
                "optimized_time_mins": 0.0,
                "rand_saved": 0.0,
                "mechanical_risk_reduction_pct": 0.0,
                "thermal_risk_pct": (strategy_telematics or {}).get("thermal_risk_pct", 0.0),
                "cargo_temp_status": (strategy_telematics or {}).get("cargo_temp_status", "Elevated"),
                "surface_profile": "n/a",
                "shipment_value_rand": float(st.session_state.shipment_value_rand),
                "ambient_temp_c": _amb.get("temp_c"),
                "rain_mm": _amb.get("rain_mm"),
                "weather_alert": _amb.get("alert"),
                "alert_mode": True,
                "cargo_temp_c": (strategy_telematics or {}).get("cargo_temp_c"),
                "safe_temp_max_c": float(st.session_state.safe_temp_max_c),
                "cargo_alert_text": " ".join(risk_action_items),
            }
            with st.spinner("🤖 Generating immediate cargo-risk mitigation memo…"):
                alert_result = backend_post("/v1/analytics/strategy", json_body=alert_payload, timeout=600)
            if alert_result is not None:
                if alert_result.get("status") == "success":
                    st.markdown(alert_result.get("strategy_markdown", ""))
                    st.caption(f"Generated on-box by `{alert_result.get('model', 'local model')}`.")
                else:
                    st.error(alert_result.get("message", "Mitigation report failed."))
        st.divider()

    if not st.session_state.trip_configured:
        st.info("Start a trip in the sidebar first — the report is built from that trip's live metrics.")
    elif is_hardware_mode:
        # ---- HARDWARE MODE: simulation-vs-real trip evaluation -------------
        st.subheader("🛰️ Simulation vs Real Trip — Live Hardware Evaluation")
        st.caption(
            "Compares your live Traccar trip against the simulated optimal plan frozen at your first GPS "
            "fix, grades how closely reality matched the simulation, and suggests how to make the real "
            "trip converge on the simulated optimum. Success = a real trip almost as perfect as the plan."
        )
        ev = backend_get("/v1/hardware/trip-evaluation", silent=True)
        if ev is None:
            st.warning("Backend unreachable — cannot evaluate the trip.")
        elif ev.get("status") != "success":
            st.info(ev.get("message", "Evaluation not ready yet — drive the route so telemetry accumulates."))
        else:
            score = ev["validation_score"]
            cells = st.columns(4)
            cells[0].metric("Validation Score", f"{score:.0f}/100",
                            help="How closely the real trip matched the simulation across route, timing and temperature.")
            cells[1].metric("Route Adherence", f"{ev['route_adherence_pct']:.0f}%",
                            help="Share of GPS pings that stayed on the simulated optimal corridor.")
            cells[2].metric("Actual vs Planned Time", f"{ev['actual_elapsed_min']:.0f} / {ev['planned_time_mins']:.0f} min")
            peak = ev.get("actual_peak_cargo_temp_c")
            cells[3].metric("Peak Cargo Temp", f"{peak:.1f} °C" if peak is not None else "n/a")

            cells2 = st.columns(3)
            cells2[0].metric("Actual Distance", f"{ev['actual_km']:.0f} km")
            cells2[1].metric("Planned Distance", f"{ev['planned_km']:.0f} km")
            cells2[2].metric("Pings Evaluated", ev["num_pings"])

            if score >= 85:
                st.success(f"✅ Simulation VALIDATED for this corridor — the real trip matched the plan ({score:.0f}/100).")
            elif score >= 60:
                st.warning(f"🟡 Partial match ({score:.0f}/100) — see the steps below to close the gap.")
            else:
                st.error(f"🔴 The real trip diverged from the simulation ({score:.0f}/100) — see the steps below.")

            st.markdown("**How to make the real trip match the simulation:**")
            for suggestion in ev["suggestions"]:
                st.markdown(f"- {suggestion}")

            if st.button("💡 Generate AI Simulation-vs-Real Report", use_container_width=True):
                amb = (strategy_telematics or {}).get("ambient_weather") or {}
                time_pct = round(ev["time_ratio"] * 100.0, 0) if ev.get("time_ratio") is not None else None
                sim_text = (
                    f"Route adherence {ev['route_adherence_pct']:.0f}%; actual {ev['actual_elapsed_min']:.0f} min "
                    f"vs planned {ev['planned_time_mins']:.0f} min; peak cargo "
                    f"{ev.get('actual_peak_cargo_temp_c')} °C; validation {score:.0f}/100. "
                    + " ".join(ev["suggestions"])
                )
                eval_payload = {
                    "origin": "Live start position",
                    "destination": ev.get("destination_town", ""),
                    "standard_time_mins": ev["planned_time_mins"],
                    "optimized_time_mins": ev["actual_elapsed_min"],
                    "rand_saved": 0.0,
                    "mechanical_risk_reduction_pct": 0.0,
                    "thermal_risk_pct": (strategy_telematics or {}).get("thermal_risk_pct", 0.0),
                    "cargo_temp_status": (strategy_telematics or {}).get("cargo_temp_status", "Unknown"),
                    "surface_profile": ev.get("planned_surface_profile") or "unknown",
                    "shipment_value_rand": float(st.session_state.shipment_value_rand),
                    "ambient_temp_c": amb.get("temp_c"),
                    "rain_mm": amb.get("rain_mm"),
                    "weather_alert": amb.get("alert"),
                    "evaluation_mode": "hardware",
                    "validation_score": score,
                    "route_adherence_pct": ev["route_adherence_pct"],
                    "actual_vs_planned_time_pct": time_pct,
                    "sim_vs_real_text": sim_text,
                }
                with st.spinner("🤖 Local model compiling the simulation-vs-real evaluation... (typically 3-6 minutes on CPU)"):
                    eval_result = backend_post("/v1/analytics/strategy", json_body=eval_payload, timeout=600)
                if eval_result is not None:
                    if eval_result.get("status") == "success":
                        st.markdown(eval_result.get("strategy_markdown", ""))
                        st.caption(f"Generated on-box by `{eval_result.get('model', 'local model')}`.")
                    else:
                        st.error(eval_result.get("message", "AI evaluation report failed."))
    elif strategy_routing is None or strategy_routing.get("trip_plan") is None:
        st.caption("Trip plan unavailable yet — start a simulator trip to generate a strategy report.")
    else:
        # Snapshot of the metrics feeding the report, so the dispatcher sees what
        # the model is reasoning over.
        trip_plan = strategy_routing["trip_plan"]
        plan_bv = trip_plan["business_value"]
        optimized_route = trip_plan["optimized_route"]

        preview_cols = st.columns(4)
        preview_cols[0].metric("Planned ETA (optimized)", f"{optimized_route['total_time_mins']:.0f} min")
        preview_cols[1].metric("Rand Saved vs Standard", f"R {plan_bv['rand_saved']:,.0f}")
        preview_cols[2].metric("Thermal Risk So Far", f"{(strategy_telematics or {}).get('thermal_risk_pct', 0.0):.1f}%")
        amb = (strategy_telematics or {}).get("ambient_weather") or {}
        preview_cols[3].metric("Ambient At Vehicle", f"{amb.get('temp_c', 0.0):.1f} °C")

        # --- Deterministic Route Justification (always accurate, no LLM) -------
        # Explains WHY the optimized route was chosen over the standard route and
        # any other road on the map. Computed by the backend from the actual
        # graph, so it's shown even if the local model is offline.
        rationale = strategy_routing.get("route_rationale")
        if rationale:
            st.divider()
            st.subheader("🧭 Route Justification — why this route, not the others")
            rj_cols = st.columns(3)
            rj_cols[0].metric("Rough km avoided", f"{rationale['rough_km_avoided']:.0f} km")
            rj_cols[1].metric(
                "Time cost vs standard",
                f"{rationale['time_delta_mins']:+.0f} min" if rationale["routes_differ"] else "0 min",
            )
            rj_cols[2].metric("Spoilage risk cut", f"{rationale['spoilage_reduction_pts']:.1f} pts")

            # Distance AND time side by side — the key to the "why so long?"
            # question: the map shows km, but the objective minimizes time-driven
            # spoilage, so a longer-in-km route is fine when it's faster in hours.
            dt_cols = st.columns(2)
            dt_cols[0].metric(
                "Optimized route",
                f"{rationale.get('optimized_km', 0):.0f} km",
                delta=f"{rationale.get('optimized_time_mins', 0):.0f} min", delta_color="off",
            )
            dt_cols[1].metric(
                "Standard route",
                f"{rationale.get('standard_km', 0):.0f} km",
                delta=f"{rationale.get('standard_time_mins', 0):.0f} min", delta_color="off",
            )
            if rationale.get("panel_note"):
                st.info("📋 **For the panel — why the optimal route can look long:** " + rationale["panel_note"])
            st.markdown(rationale["reason"])
            opt_s = rationale["optimized_summary"]
            std_s = rationale["standard_summary"]
            st.caption(
                f"**Chosen (optimized) route surface:** {opt_s['profile']} — "
                f"{opt_s['rough_km']:.0f} km rough of {opt_s['total_km']:.0f} km total."
            )
            st.caption(
                f"**Standard (time-only) route surface:** {std_s['profile']} — "
                f"{std_s['rough_km']:.0f} km rough of {std_s['total_km']:.0f} km total."
            )
            if rationale.get("boundary_note"):
                st.warning(rationale["boundary_note"])

        st.divider()

        if st.button("💡 Generate AI Business Strategy Report", use_container_width=True):
            surface_classes = sorted({
                (seg.get("fclass") or "unknown") for seg in optimized_route.get("segments", [])
            })
            surface_profile = ", ".join(surface_classes) if surface_classes else "unknown"

            mechanical_risk_reduction_pct = (
                plan_bv["standard_spoilage_risk_pct"] - plan_bv["optimized_spoilage_risk_pct"]
            )
            thermal_risk_pct = (strategy_telematics or {}).get("thermal_risk_pct", 0.0)
            cargo_temp_status = (strategy_telematics or {}).get("cargo_temp_status", "Unknown")
            strategy_ambient = (strategy_telematics or {}).get("ambient_weather") or {}

            strategy_payload = {
                "origin": trip_plan.get("origin_town") or "Current live position",
                "destination": trip_plan.get("destination_town", ""),
                "standard_time_mins": trip_plan["standard_route"]["total_time_mins"],
                "optimized_time_mins": optimized_route["total_time_mins"],
                "rand_saved": plan_bv["rand_saved"],
                "mechanical_risk_reduction_pct": round(mechanical_risk_reduction_pct, 1),
                "thermal_risk_pct": thermal_risk_pct,
                "cargo_temp_status": cargo_temp_status,
                "surface_profile": surface_profile,
                "shipment_value_rand": plan_bv["shipment_value_rand"],
                "ambient_temp_c": strategy_ambient.get("temp_c"),
                "rain_mm": strategy_ambient.get("rain_mm"),
                "weather_alert": strategy_ambient.get("alert"),
                # Route-justification inputs so the memo gains a grounded
                # "## Route Justification" section (see backend generate_ai_strategy).
                "routes_differ": rationale.get("routes_differ") if rationale else None,
                "rough_km_avoided": rationale.get("rough_km_avoided") if rationale else None,
                "standard_route_profile": rationale["standard_summary"]["profile"] if rationale else None,
                "optimized_route_profile": rationale["optimized_summary"]["profile"] if rationale else None,
                "route_rationale_text": rationale.get("reason") if rationale else None,
            }

            logger.info("AI STRATEGY REQUEST -> dispatching trip-plan metrics to backend: %s", strategy_payload)

            # Generous timeout: the local daemon keeps the model warm, so calls
            # are usually quick, but a first cold-start pull or a slow CPU box can
            # still take a while — better to wait than to false-timeout.
            STRATEGY_TIMEOUT_SECONDS = 600
            with st.spinner("🤖 Local model compiling econometric recommendations... (first run ~2-4 min on CPU while it loads, then fast)"):
                strategy_result = backend_post(
                    "/v1/analytics/strategy", json_body=strategy_payload, timeout=STRATEGY_TIMEOUT_SECONDS
                )

            if strategy_result is not None:
                if strategy_result.get("status") == "success":
                    strategy_markdown = strategy_result.get("strategy_markdown", "")
                    st.markdown(strategy_markdown)
                    st.caption(f"Generated on-box by `{strategy_result.get('model', 'local model')}`.")
                    logger.info(
                        "AI STRATEGY REPORT -> received %d characters from model '%s'",
                        len(strategy_markdown), strategy_result.get("model", "?"),
                    )
                else:
                    st.error(strategy_result.get("message", "AI strategy generation failed."))
                    logger.warning("AI STRATEGY REPORT -> backend returned error_type=%s", strategy_result.get("error_type"))


# ===========================================================================
# TAB 5 — Weather (live ambient climate + forecast along the route)
# ===========================================================================
with tab_weather:
    st.title("🌤️ Route Weather Analytics")

    wui: dict = {"source_info": st.empty()}
    wui["climate_header"] = st.empty()
    weather_cols = st.columns(3)
    wui["temp_metric"] = weather_cols[0].empty()
    wui["rain_metric"] = weather_cols[1].empty()
    wui["warm_metric"] = weather_cols[2].empty()
    wui["climate_alert"] = st.empty()

    st.divider()
    st.subheader("Road Condition Forecast — Along the Optimized Route")
    st.caption(
        "Origin/Midpoint/Destination are fixed to the trip's actual planned route and stay "
        "put as the truck drives. Current Position is separate and always reflects wherever "
        "the truck is right now — that's the one that moves."
    )
    wui["forecast_empty_state"] = st.empty()
    wui["forecast_table"] = st.empty()
    wui["forecast_alert"] = st.empty()
    wui["hardware_note"] = st.empty()
    wui["source_footer"] = st.empty()

    @st.fragment(run_every=POLL_INTERVAL_SECONDS)
    def weather_tab_view():
        weather_telematics = backend_get("/v1/telematics/truck-01", silent=True)
        active_weather_source = ((weather_telematics or {}).get("ambient_weather") or {}).get("source")
        wstatus = backend_get("/v1/weather/status", silent=True) or {}
        key_configured = bool(wstatus.get("key_configured", False))

        with wui["source_info"].container():
            if active_weather_source == "openweathermap":
                st.caption(
                    "🟢 Source: **OpenWeatherMap** (your API key was detected and is answering). Readings "
                    "blend in nearby station observations, so they track closely with sites like Google "
                    "Weather. The key is loaded from your local `.env` and is never shown here."
                )
            elif key_configured:
                # Key IS configured, but this reading came from Open-Meteo / the
                # baseline. The usual cause is a brand-new OWM key that hasn't
                # activated yet (can take up to ~2 hours), or a transient timeout.
                shown = "baseline fallback" if active_weather_source == "fallback" else "Open-Meteo"
                st.caption(
                    f"🟡 Your **OpenWeatherMap key IS configured**, but the reading right now came from "
                    f"**{shown}**. A newly-created OWM key can take up to ~2 hours to activate (or the last "
                    "call timed out) — the dashboard switches to OpenWeatherMap automatically once its API "
                    "starts responding. Nothing else to do."
                )
            elif active_weather_source == "fallback":
                st.caption(
                    "🟠 Source: **baseline fallback** (25 °C / 0 mm) — the live weather providers could not "
                    "be reached just now. Routing and telemetry keep working; readings refresh automatically "
                    "when a provider responds again."
                )
            else:
                st.caption(
                    "🔵 Source: **Open-Meteo** (keyless forecast-model estimate). No OpenWeatherMap key was "
                    "detected on the backend."
                )
                st.caption(
                    "ℹ️ These are a live forecast-model estimate for the exact coordinate, not a nearby "
                    "weather-station observation. To switch to station-blended OpenWeatherMap readings, drop "
                    "`API_KEY = \"...\"` into a `.env` file next to weather_engine.py (free tier at "
                    "openweathermap.org) and **restart the backend** so it reloads the key."
                )

        ambient = (weather_telematics or {}).get("ambient_weather") or {
            "temp_c": 25.0, "rain_mm": 0.0, "alert": "Normal",
        }
        ambient_temp_c = ambient.get("temp_c", 25.0)

        if ambient_temp_c >= 38.0:
            warming_rate_status = "🔥 Accelerated"
        elif ambient_temp_c < 20.0:
            warming_rate_status = "🧊 Suppressed"
        else:
            warming_rate_status = "➖ Normal"

        wui["climate_header"].subheader("Active Vehicle Climate Context")
        wui["temp_metric"].metric("Outside Ambient Temperature", f"{ambient_temp_c:.1f} °C")
        wui["rain_metric"].metric("Rain Index", f"{ambient.get('rain_mm', 0.0):.1f} mm")
        wui["warm_metric"].metric(
            "Dynamic Thermodynamic Warming Rate",
            warming_rate_status,
            help=(
                "How fast an IDLING reefer chamber is modeled to warm up right now, driven by "
                "live ambient temperature: Accelerated ≥38°C, Normal 20-38°C, Suppressed <20°C."
            ),
        )

        if ambient.get("alert") == "Extreme Heat":
            wui["climate_alert"].error(
                "🔥 Extreme heat at the vehicle's current position — refrigeration compressor "
                "under significant thermal stress."
            )
        elif ambient.get("alert") == "Heavy Rain / Washout Risk":
            wui["climate_alert"].warning(
                "🌧️ Heavy rain at the vehicle's current position — unpaved segments at "
                "elevated mud-washout risk."
            )
        else:
            wui["climate_alert"].success(
                f"✅ No weather alert at the vehicle's current position ({ambient_temp_c:.1f} °C)."
            )

        weather_profile = backend_get("/v1/routing/weather-profile", silent=True)
        if weather_profile is None or not weather_profile.get("segments"):
            wui["forecast_empty_state"].info(
                "Start a trip in the Customer Route Tracker to see a weather forecast along the route."
            )
            # Claim the table slot with a lightweight caption instead of an empty
            # st.dataframe. The empty dataframe rendered a boxy "empty" widget that
            # the 2-second fragment refresh made flash in and out; a caption is
            # stable and never shows that box.
            wui["forecast_table"].caption("— no route forecast yet —")
            wui["forecast_alert"].empty()
            wui["hardware_note"].empty()
            wui["source_footer"].empty()
            return
        wui["forecast_empty_state"].empty()

        forecast_rows = []
        any_heat_alert = False
        any_rain_alert = False
        for seg in weather_profile["segments"]:
            if seg["label"] == "Current Position":
                row_temp_c = ambient_temp_c
                row_rain_mm = ambient.get("rain_mm", 0.0)
                row_alert = ambient.get("alert", "Normal")
            else:
                row_temp_c = seg["temp_c"]
                row_rain_mm = seg["rain_mm"]
                row_alert = seg["alert"]

            forecast_rows.append({
                "Segment": seg["label"],
                "Latitude": f"{seg['lat']:.4f}",
                "Longitude": f"{seg['lon']:.4f}",
                "Temperature (°C)": f"{row_temp_c:.1f}",
                "Rain (mm)": f"{row_rain_mm:.1f}",
                "Alert": row_alert,
            })
            if row_alert == "Extreme Heat":
                any_heat_alert = True
            elif row_alert == "Heavy Rain / Washout Risk":
                any_rain_alert = True

        wui["forecast_table"].dataframe(forecast_rows, use_container_width=True, hide_index=True)

        if any_heat_alert:
            wui["forecast_alert"].error(
                "🔥 One or more segments ahead report extreme heat — refrigeration "
                "compressors along this route are under significant strain."
            )
        elif any_rain_alert:
            wui["forecast_alert"].warning(
                "🌧️ One or more segments ahead report heavy rain — unpaved/gravel "
                "sections at elevated risk of mud washouts."
            )
        else:
            wui["forecast_alert"].success("✅ No extreme weather alerts along the route right now.")

        if weather_profile.get("route_basis") == "remaining_route":
            wui["hardware_note"].caption(
                "⚠️ No fixed trip plan exists yet, so Origin/Midpoint/Destination above are "
                "sampled from the truck's current position onward instead — they will converge "
                "as it nears the destination. This resolves automatically once a departure "
                "baseline is established (first GPS fix in Hardware mode, or the moment a "
                "Simulator trip starts)."
            )
        else:
            wui["hardware_note"].empty()

        source_label = weather_profile["segments"][0].get("source", "open-meteo")
        source_note = (
            "station-blended, requires OPENWEATHERMAP_API_KEY"
            if source_label == "openweathermap"
            else "forecast-model estimate, no API key required"
        )
        wui["source_footer"].caption(
            f"Weather data source: {source_label} ({source_note}) · "
            "cached up to 5 minutes per ~1.1km grid cell."
        )

    weather_tab_view()
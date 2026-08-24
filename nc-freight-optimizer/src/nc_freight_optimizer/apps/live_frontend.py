"""
live_frontend.py

Streamlit command dashboard for the Northern Cape Fleet Dispatch system.
Talks to live_backend.py (FastAPI, http://127.0.0.1:8000) over plain HTTP.

Two tabs:
  Fleet Dispatch Hub       - trip setup, live map with the standard route and
                              the spoilage-optimized route side by side, and
                              the business-value numbers behind the choice.
  Driver Ground-Truth Form - crowd-sourced road-condition reports that trigger
                              live reroutes on the backend.

Live polling is isolated inside a Streamlit fragment, so it refreshes only the
map/metrics block on its own timer instead of rerunning the whole page. That
is what stops the flicker/reset that used to hit the Ground-Truth tab too.

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
from datetime import datetime

import requests
import streamlit as st
import streamlit.components.v1 as components

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
# Typography — a distinct, deliberate font pairing instead of the Streamlit
# default. Headings in Space Grotesk (a geometric, slightly technical
# display face), body/metrics in IBM Plex Sans (clean, highly legible at
# small sizes for dense dashboard numbers).
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
    st.session_state.temp_history = []  # list of {"poll": int, "cargo_temp_c": float} for the live chart

towns_payload = backend_get("/v1/towns", silent=True)
AVAILABLE_TOWNS = towns_payload["towns"] if towns_payload else []


# ---------------------------------------------------------------------------
# Sidebar — mode + trip setup
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
        origin_town = st.sidebar.selectbox("Starting point", options=AVAILABLE_TOWNS, index=0)
        destination_default = 1 if len(AVAILABLE_TOWNS) > 1 else 0
        destination_town = st.sidebar.selectbox(
            "Destination", options=AVAILABLE_TOWNS, index=destination_default
        )
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
        destination_town = st.sidebar.selectbox("Destination", options=AVAILABLE_TOWNS, index=0)
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
    "\"Value at Risk So Far\"). Set this per shipment; it overrides the system default "
    "immediately, no restart needed."
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
# Route lines are only redrawn when the path actually changes (e.g. a driver
# report triggers a reroute), not on every poll — that plus the CSS
# transition is what makes the movement smooth instead of twitchy.
# ---------------------------------------------------------------------------
def render_live_map(initial_routing: dict | None, initial_telematics: dict) -> None:
    truck_lat = initial_telematics["lat"]
    truck_lon = initial_telematics["lon"]
    feed_source = initial_telematics.get("feed_source", "")

    initial_standard = initial_routing["standard_route"]["segments"] if initial_routing else []
    initial_optimized = initial_routing["optimized_route"]["segments"] if initial_routing else []
    initial_dest = initial_routing["optimized_route"]["coordinates"][-1] if initial_routing else None
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
      // Use the same backend URL the Python side already talks to
      // successfully, instead of re-deriving it from window.location.
      // Guessing it from the browser's own protocol/hostname breaks as
      // soon as the dashboard is served from anywhere other than
      // exactly "http://<same-host>:8000" (a tunnel, a different LAN
      // IP, HTTPS, etc.) — the fetch then silently fails and the truck
      // marker never receives new coordinates, even though the
      // server-side polling (which isn't subject to browser CORS/
      // mixed-content rules) keeps working and the metric numbers keep
      // updating. That mismatch is exactly what caused the icon to sit
      // still while the coordinates kept changing.
      const BACKEND = {json.dumps(BACKEND_URL)};
      const OPTIMIZED_COLOR = '{OPTIMIZED_COLOR}';
      const STANDARD_COLOR = '{STANDARD_COLOR}';
      const POLL_MS = {int(POLL_INTERVAL_SECONDS * 1000)};

      const map = L.map('fleet-map').setView([{truck_lat}, {truck_lon}], 7);
      L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
        attribution: 'CartoDB'
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
          const key = routeKey(data.optimized_route.segments);
          if (key !== lastOptimizedKey) {{
            drawSegments(data.standard_route.segments, standardLayer, STANDARD_COLOR, true, 'standard route');
            drawSegments(data.optimized_route.segments, optimizedLayer, OPTIMIZED_COLOR, false, 'optimized route');
            lastOptimizedKey = key;
          }}
          const destCoords = data.optimized_route.coordinates[data.optimized_route.coordinates.length - 1];
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
      setInterval(pollRouting, POLL_MS * 2);
    </script>
    """
    components.html(html, height=540, scrolling=False)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dispatch, tab_field_report, tab_weather = st.tabs(
    ["🚛 Fleet Dispatch Hub", "📝 Driver Ground-Truth Form", "🌤️ Route Weather Analytics"]
)


# ---------------------------------------------------------------------------
# TAB A — Fleet Dispatch Hub (fragment-isolated live polling)
# ---------------------------------------------------------------------------
with tab_dispatch:
    st.title("Northern Cape Fleet Dispatch Hub")
    st.caption("Live spoilage-risk-optimized routing for cold-chain fisheries transport.")

    if not st.session_state.trip_configured:
        st.info("Set a starting point and destination in the sidebar, then start the trip to see live routing.")
    else:

        # --- Live map: rendered ONCE as a self-contained Leaflet component
        # that polls the backend directly from the browser and moves the
        # truck marker with a CSS transition. This is what was glitching:
        # a folium map gets fully rebuilt into a fresh iframe on every poll,
        # which flashes/reloads visibly. A plain Streamlit rerun (trip
        # reconfigured, tab reopened) rebuilds this once; the fragment below
        # never touches it, so it never flickers again.
        initial_routing = backend_get("/v1/routing/truck-01", silent=True)
        initial_telematics = backend_get("/v1/telematics/truck-01", silent=True)
        st.subheader("Live Route Map")
        if initial_telematics is None or initial_telematics.get("lat") is None:
            st.warning("No position available yet.")
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

        # --- Embedded AI Cognitive Strategy Layer ---------------------------
        # Sits directly beneath the live map. Purely on-demand: it never
        # fires automatically and never touches the polling fragment below,
        # so it can't add latency or flicker to the live map/metrics. The
        # first click is slow (the local model loads then); every click
        # after that is fast, since the backend caches it in memory.
        st.markdown("---")
        st.subheader("🧠 Embedded AI Cognitive Strategy Layer")

        strategy_routing = backend_get("/v1/routing/truck-01", silent=True)
        strategy_telematics = backend_get("/v1/telematics/truck-01", silent=True)

        if strategy_routing is None or strategy_routing.get("trip_plan") is None:
            st.caption("Trip plan unavailable yet — start a trip to generate a strategy report.")
        else:
            st.caption(
                "Sends this trip's fixed planning KPIs to a locally hosted Qwen2.5-1.5B model "
                "and returns a business-strategy memo. Runs entirely on-box; the first report "
                "may take a minute or two while the model loads."
            )
            if st.button("💡 Generate AI Business Strategy Report", use_container_width=True):
                trip_plan = strategy_routing["trip_plan"]
                plan_bv = trip_plan["business_value"]
                optimized_route = trip_plan["optimized_route"]

                # The road classes (fclass) actually present along the
                # optimized route, for the wear-and-tear discussion.
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
                }

                logger.info(
                    "AI STRATEGY REQUEST -> dispatching trip-plan metrics to backend: %s",
                    strategy_payload,
                )

                # CPU inference on a 1.5B model generating up to 700 tokens took
                # ~3m12s in testing — 180s wasn't enough headroom and cut it off
                # right before it finished. Generous margin here on purpose.
                STRATEGY_TIMEOUT_SECONDS = 600
                with st.spinner("🤖 Local model compiling econometric recommendations... (typically 2-4 minutes on CPU)"):
                    strategy_result = backend_post(
                        "/v1/analytics/strategy", json_body=strategy_payload, timeout=STRATEGY_TIMEOUT_SECONDS
                    )

                if strategy_result is not None:
                    if strategy_result.get("status") == "success":
                        st.markdown(strategy_result["strategy_markdown"])
                        logger.info(
                            "AI STRATEGY REPORT -> received %d characters from model '%s'",
                            len(strategy_result["strategy_markdown"]), strategy_result.get("model", "?"),
                        )
                    else:
                        st.error(strategy_result.get("message", "AI strategy generation failed."))
                        logger.warning(
                            "AI STRATEGY REPORT -> backend returned error_type=%s",
                            strategy_result.get("error_type"),
                        )

        st.divider()

        # --- One-time layout scaffold ----------------------------------
        # Built once per OUTER script run (page load, trip (re)configured,
        # threshold changed, etc. — all infrequent) and captured by closure
        # in dispatch_metrics_view() below. The fragment's run_every ticks
        # only re-execute that inner function, never this scaffold, so it
        # never calls st.columns()/st.subheader()/st.metric() at the top
        # level on every poll — it only calls .metric()/.markdown()/etc. on
        # these SAME placeholder objects. That's the actual fix: calling
        # st.columns() or st.metric() fresh every ~2s poll rebuilds those
        # DOM nodes from scratch each cycle, which is what reads as a
        # blink. Updating an existing placeholder in place just patches its
        # content, so the layout never flashes. (Deliberately NOT stashed
        # in session_state — these DeltaGenerator handles are only valid
        # for this script run, so they're rebuilt fresh, cheaply, every
        # time this outer code runs, and simply closed over by the fragment.)
        ui: dict = {"warning": st.empty()}

        ui["plan_wrap"] = st.empty()

        ui["live_header"] = st.empty()
        live_cols = st.columns(2)
        ui["live_eta"] = live_cols[0].empty()
        ui["live_risk"] = live_cols[1].empty()
        st.divider()

        ui["cargo_header"] = st.empty()
        cargo_cols = st.columns(4)
        ui["cargo_temp"] = cargo_cols[0].empty()
        ui["cargo_mech"] = cargo_cols[1].empty()
        ui["cargo_thermal"] = cargo_cols[2].empty()
        ui["cargo_value"] = cargo_cols[3].empty()
        ui["cargo_alert"] = st.empty()

        ui["chart"] = st.empty()
        ui["chart_caption"] = st.empty()
        st.divider()

        tel_cols = st.columns(4)
        ui["tel_vehicle"] = tel_cols[0].empty()
        ui["tel_coords"] = tel_cols[1].empty()
        ui["tel_temp"] = tel_cols[2].empty()
        ui["tel_fourth"] = tel_cols[3].empty()

        ui["feed"] = st.empty()
        ui["updated"] = st.empty()

        @st.fragment(run_every=POLL_INTERVAL_SECONDS)
        def dispatch_metrics_view():
            telematics = backend_get("/v1/telematics/truck-01", silent=True)
            routing = backend_get("/v1/routing/truck-01", silent=True)

            if telematics is None:
                ui["warning"].warning("No telemetry received yet. Make sure live_backend.py is running.")
                return
            ui["warning"].empty()

            feed_source = telematics.get("feed_source", "UNKNOWN")
            arrived = telematics.get("arrived", False)
            progress_pct = telematics.get("trip_progress_pct")
            cargo_temp_c = telematics.get("cargo_temp_c", 0.0)
            temp_status = telematics.get("cargo_temp_status", "Unknown")

            # --- Trip Plan: fixed full-journey baseline, cached once when the
            # trip was configured. Doesn't change as the truck drives — that's
            # the point, it's what "Live Remaining" below is measured against.
            # Rendered into a single placeholder container, so switching it
            # on/off (or refreshing identical values) never reshapes the
            # rest of the layout.
            trip_plan = routing.get("trip_plan") if routing is not None else None
            if trip_plan is not None:
                with ui["plan_wrap"].container():
                    st.subheader("Trip Plan (fixed at departure)")
                    plan_bv = trip_plan["business_value"]
                    plan_cols = st.columns(3)
                    plan_cols[0].metric(
                        "Planned ETA (optimized)", f"{trip_plan['optimized_route']['total_time_mins']:.0f} min"
                    )
                    plan_cols[1].metric("Time Cost vs Standard", f"{-plan_bv['time_saved_mins']:+.0f} min")
                    plan_cols[2].metric(
                        "Planned Spoilage Risk Avoided",
                        f"{plan_bv['standard_spoilage_risk_pct'] - plan_bv['optimized_spoilage_risk_pct']:.1f} pts",
                    )
                    st.divider()
            else:
                ui["plan_wrap"].empty()

            # --- Live Remaining: recomputed from the truck's CURRENT position
            # on every poll — this is what used to be frozen at the full-trip
            # value all the way through the drive; it now genuinely shrinks.
            # Same "must write every run" rule as the chart below: give the
            # else branch something to write too, so these three placeholders
            # are never skipped on a run (e.g. a transient routing hiccup)
            # only to be written for the first time on a later fragment-only
            # rerun.
            if routing is not None:
                bv = routing["business_value"]
                optimized = routing["optimized_route"]

                ui["live_header"].subheader("Live Remaining Route")
                ui["live_eta"].metric("Remaining ETA (optimized)", f"{optimized['total_time_mins']:.0f} min")
                ui["live_risk"].metric(
                    "Remaining Spoilage Risk Avoided",
                    f"{bv['standard_spoilage_risk_pct'] - bv['optimized_spoilage_risk_pct']:.1f} pts",
                    help=f"Optimized: {bv['optimized_spoilage_risk_pct']:.1f}% risk vs "
                         f"Standard: {bv['standard_spoilage_risk_pct']:.1f}% risk, for the route "
                         f"from where the truck is right now to the destination.",
                )
            else:
                ui["live_header"].subheader("Live Remaining Route")
                ui["live_eta"].metric("Remaining ETA (optimized)", "n/a")
                ui["live_risk"].metric("Remaining Spoilage Risk Avoided", "n/a")

            # --- Cargo Condition: what's actually happening to the shipment,
            # driven by cargo_temp_c (mechanical risk = road damage already
            # driven over; thermal risk = temperature exposure accumulated so
            # far). Replace cargo_temp_c's source with a real reefer sensor
            # feed via /v1/telematics/incoming once hardware is wired up —
            # everything downstream of that one number already works.
            ui["cargo_header"].subheader("Cargo Condition")
            ui["cargo_temp"].metric("Cargo Temperature", f"{cargo_temp_c:.1f} C")
            mech_risk = telematics.get("mechanical_risk_pct")
            ui["cargo_mech"].metric(
                "Mechanical Risk (roads so far)",
                f"{mech_risk:.1f}%" if mech_risk is not None else "n/a",
                help="Road-roughness damage accrued on the distance already driven. "
                     "Only tracked in Simulator mode, which has a fixed planned route.",
            )
            ui["cargo_thermal"].metric(
                "Thermal Risk (heat so far)",
                f"{telematics.get('thermal_risk_pct', 0):.1f}%",
                help="Temperature-exposure damage accrued so far, integrated over elapsed time.",
            )
            ui["cargo_value"].metric(
                "Value at Risk So Far",
                f"R {telematics.get('expected_loss_rand_so_far', 0):,.0f}",
                help="Composite (mechanical + thermal) risk applied to the shipment value.",
            )

            if temp_status == "Awaiting Motion":
                ui["cargo_alert"].info(
                    f"⏸️ Truck parked at {cargo_temp_c:.1f} C — cargo risk metrics are frozen until it "
                    f"starts moving, so a stationary shipment doesn't silently accrue spoilage risk "
                    f"before it's actually departed."
                )
            elif temp_status == "Critical":
                ui["cargo_alert"].error(
                    f"Cargo temperature CRITICAL at {cargo_temp_c:.1f} C — thermal spoilage risk is "
                    f"accelerating (composite risk {telematics.get('composite_cargo_risk_pct', 0):.1f}%)."
                )
            elif temp_status == "Elevated":
                ui["cargo_alert"].warning(
                    f"Cargo temperature elevated at {cargo_temp_c:.1f} C — above the "
                    f"{st.session_state.safe_temp_max_c:.1f} C safe threshold, "
                    f"spoilage risk is accruing faster than baseline."
                )
            else:
                ui["cargo_alert"].success(f"Cargo temperature nominal at {cargo_temp_c:.1f} C.")

            # --- Live temperature history chart -----------------------------
            # Only appends while the trip is genuinely underway: not yet
            # arrived, AND (for hardware mode) actually moving. Without the
            # "arrived" check it kept appending forever after the trip
            # finished; without the "awaiting motion" check it did the same
            # for a truck that was simply parked and hadn't departed yet —
            # in both cases the chart never settled and kept scrolling even
            # though nothing was actually changing.
            awaiting_motion = telematics.get("awaiting_motion", False)
            if not arrived and not awaiting_motion:
                st.session_state.temp_history.append({
                    "poll": len(st.session_state.temp_history),
                    "Cargo Temp (C)": cargo_temp_c,
                })
                st.session_state.temp_history = st.session_state.temp_history[-200:]  # cap buffer length

            # ui["chart"] MUST be written to on every single run of this
            # fragment, including the very first one — a placeholder created
            # outside a fragment (st.empty(), in the scaffold above) has to
            # be claimed by an actual call during the run that creates it,
            # or Streamlit can't reserve a stable position for it. Gating
            # this behind "len(temp_history) >= 2" meant that on the very
            # first poll (0 or 1 points collected so far), the chart
            # placeholder was left untouched; the first time a LATER,
            # fragment-only rerun tried to draw into it, Streamlit raised
            # "container was not written to during the initial run". Always
            # drawing something — even a single-point placeholder chart —
            # keeps the slot claimed from run one, so later reruns can just
            # update it in place.
            if len(st.session_state.temp_history) >= 2:
                chart_data = {
                    row["poll"]: row["Cargo Temp (C)"] for row in st.session_state.temp_history
                }
                ui["chart"].line_chart(chart_data, height=180)
                if arrived:
                    ui["chart_caption"].caption(
                        f"🏁 Trip complete — cargo temperature history frozen at arrival "
                        f"(safe threshold: {st.session_state.safe_temp_max_c:.1f} C)."
                    )
                else:
                    ui["chart_caption"].caption(
                        f"Live cargo temperature over the session "
                        f"(safe threshold: {st.session_state.safe_temp_max_c:.1f} C). "
                        "This is exactly where a real reefer-unit sensor feed would plug in."
                    )
            else:
                ui["chart"].line_chart({0: cargo_temp_c}, height=180)
                ui["chart_caption"].caption("Collecting cargo temperature history…")

            # --- Telemetry metrics -----------------------------------------
            ui["tel_vehicle"].metric("Vehicle ID", telematics.get("vehicle_id", "—"))
            ui["tel_coords"].metric(
                "Current Coordinates",
                f"{telematics.get('lat', 0):.4f}, {telematics.get('lon', 0):.4f}",
            )
            ui["tel_temp"].metric("Cargo Temperature", f"{cargo_temp_c:.1f} C")
            if progress_pct is not None:
                ui["tel_fourth"].metric("Trip Progress", f"{progress_pct:.0f}%")
            elif routing is not None:
                ui["tel_fourth"].metric(
                    "Spoilage Cost Index", f"{routing['optimized_route']['total_spoilage_cost']:.2f}",
                )
            else:
                ui["tel_fourth"].metric("Trip Progress", "n/a")

            if feed_source == FEED_SOURCE_REAL:
                ui["feed"].success(f"Data Feed Source: {feed_source}")
            elif feed_source == FEED_SOURCE_SIMULATED:
                ui["feed"].info(f"Data Feed Source: {feed_source}" + ("  |  Arrived at destination" if arrived else ""))
            else:
                ui["feed"].warning(f"Data Feed Source: {feed_source}")

            ui["updated"].caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

        dispatch_metrics_view()


# ---------------------------------------------------------------------------
# TAB B — Driver Ground-Truth Form (outside the fragment, never rerun by polling)
# ---------------------------------------------------------------------------
with tab_field_report:
    st.title("Driver Ground-Truth Report")
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

    # Outside the form on purpose: widgets inside st.form don't trigger a
    # rerun until submit, so a checkbox in there couldn't reveal the storm
    # dropdown live. Placed here, toggling it reruns the script immediately
    # and the form below picks up its current value on that same rerun.
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


# ---------------------------------------------------------------------------
# TAB C — Route Weather Analytics (live ambient climate + forecast along path)
# ---------------------------------------------------------------------------
with tab_weather:
    st.title("🌤️ Route Weather Analytics")
    st.caption(
        "Live ambient climate data at the vehicle's position and along the active route, "
        "via the free Open-Meteo API — no API key required."
    )
    st.caption(
        "ℹ️ Readings are a live forecast-model estimate for the exact coordinate, not a "
        "nearby weather station observation — expect roughly a 1-3°C margin versus sites "
        "like Google Weather that blend in local station data."
    )

    weather_telematics = backend_get("/v1/telematics/truck-01", silent=True)
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

    st.subheader("Active Vehicle Climate Context")
    weather_cols = st.columns(3)
    weather_cols[0].metric("Outside Ambient Temperature", f"{ambient_temp_c:.1f} °C")
    weather_cols[1].metric("Rain Index", f"{ambient.get('rain_mm', 0.0):.1f} mm")
    weather_cols[2].metric(
        "Dynamic Thermodynamic Warming Rate",
        warming_rate_status,
        help=(
            "How fast an IDLING reefer chamber is modeled to warm up right now, driven by "
            "live ambient temperature: Accelerated ≥38°C, Normal 20-38°C, Suppressed <20°C."
        ),
    )

    if ambient.get("alert") == "Extreme Heat":
        st.error(
            "🔥 Extreme heat at the vehicle's current position — refrigeration compressor "
            "under significant thermal stress."
        )
    elif ambient.get("alert") == "Heavy Rain / Washout Risk":
        st.warning(
            "🌧️ Heavy rain at the vehicle's current position — unpaved segments at "
            "elevated mud-washout risk."
        )
    else:
        st.success(f"✅ No weather alert at the vehicle's current position ({ambient_temp_c:.1f} °C).")

    st.divider()
    st.subheader("Road Condition Forecast — Along the Optimized Route")
    st.caption(
        "Origin/Midpoint/Destination are fixed to the trip's actual planned route and stay "
        "put as the truck drives. Current Position is separate and always reflects wherever "
        "the truck is right now — that's the one that moves."
    )

    weather_profile = backend_get("/v1/routing/weather-profile", silent=True)
    if weather_profile is None or not weather_profile.get("segments"):
        st.info("Start a trip in the Fleet Dispatch Hub to see a weather forecast along the route.")
    else:
        forecast_rows = []
        any_heat_alert = False
        any_rain_alert = False
        for seg in weather_profile["segments"]:
            forecast_rows.append({
                "Segment": seg["label"],
                "Latitude": f"{seg['lat']:.4f}",
                "Longitude": f"{seg['lon']:.4f}",
                "Temperature (°C)": f"{seg['temp_c']:.1f}",
                "Rain (mm)": f"{seg['rain_mm']:.1f}",
                "Alert": seg["alert"],
            })
            if seg["alert"] == "Extreme Heat":
                any_heat_alert = True
            elif seg["alert"] == "Heavy Rain / Washout Risk":
                any_rain_alert = True

        st.dataframe(forecast_rows, use_container_width=True, hide_index=True)

        if any_heat_alert:
            st.error(
                "🔥 One or more segments ahead report extreme heat — refrigeration "
                "compressors along this route are under significant strain."
            )
        if any_rain_alert:
            st.warning(
                "🌧️ One or more segments ahead report heavy rain — unpaved/gravel "
                "sections at elevated risk of mud washouts."
            )
        if not any_heat_alert and not any_rain_alert:
            st.success("✅ No extreme weather alerts along the route right now.")

        if weather_profile.get("route_basis") == "remaining_route":
            st.caption(
                "⚠️ Hardware mode has no fixed origin town, so Origin/Midpoint/Destination "
                "above are sampled from the truck's current position onward — they will "
                "converge as it nears the destination. That's expected, not a bug."
            )

        source_label = weather_profile["segments"][0].get("source", "open-meteo")
        st.caption(
            f"Weather data source: {source_label} · via Open-Meteo (no API key required) · "
            "cached up to 5 minutes per ~1.1km grid cell."
        )
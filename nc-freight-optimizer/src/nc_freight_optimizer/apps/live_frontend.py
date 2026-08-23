"""
live_frontend.py

Streamlit command dashboard for the Northern Cape Fleet Dispatch system.
Talks to live_backend.py (FastAPI, http://127.0.0.1:8000) over plain HTTP.

Two tabs:
  Fleet Dispatch Hub       - live map + metrics, switchable between a real
                              Traccar Client phone feed and a built-in simulator.
  Driver Ground-Truth Form - crowd-sourced road-condition reports that trigger
                              live reroutes on the backend.

Run standalone (uv workspace, alongside a running live_backend.py):
    uv run streamlit run live_frontend.py
"""


from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
import time
from datetime import datetime

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 5

MODE_HARDWARE = "📡 Live Mobile Hardware Tracking (Traccar)"
MODE_SIMULATOR = "🤖 Automated Ingestion Simulator"

FEED_SOURCE_REAL = "REAL-TIME TRACCAR HARDWARE"
FEED_SOURCE_SIMULATED = "SIMULATED TELEMETRY MATRIX"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live_frontend")

st.set_page_config(
    page_title="Northern Cape Fleet Dispatch",
    page_icon="🚛",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Backend client helpers — every call is wrapped so an offline backend
# degrades gracefully instead of crashing the dashboard.
# ---------------------------------------------------------------------------
def backend_get(path: str) -> dict | None:
    try:
        response = requests.get(f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "🔌 Cannot reach the backend server. Is `live_backend.py` running? "
            f"Expected at {BACKEND_URL}"
        )
        logger.error("GET %s failed: connection error", path)
        return None
    except requests.exceptions.Timeout:
        st.error(f"⏱️ Backend request to `{path}` timed out.")
        logger.error("GET %s failed: timeout", path)
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"⚠️ Backend returned an error for `{path}`: {exc}")
        logger.error("GET %s failed: %s", path, exc)
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"⚠️ Unexpected error calling `{path}`: {exc}")
        logger.error("GET %s failed: %s", path, exc)
        return None


def backend_post(path: str, json_body: dict | None = None) -> dict | None:
    try:
        response = requests.post(
            f"{BACKEND_URL}{path}", json=json_body, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "🔌 Cannot reach the backend server. Is `live_backend.py` running? "
            f"Expected at {BACKEND_URL}"
        )
        logger.error("POST %s failed: connection error", path)
        return None
    except requests.exceptions.Timeout:
        st.error(f"⏱️ Backend request to `{path}` timed out.")
        logger.error("POST %s failed: timeout", path)
        return None
    except requests.exceptions.HTTPError as exc:
        try:
            detail = response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"⚠️ Backend rejected `{path}`: {detail}")
        logger.error("POST %s failed: %s", path, detail)
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"⚠️ Unexpected error calling `{path}`: {exc}")
        logger.error("POST %s failed: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Session state initialization — persists the last known telemetry/routing
# payloads across Streamlit reruns so the dashboard still shows something
# sensible even when the polling loop is paused.
# ---------------------------------------------------------------------------
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "last_telematics" not in st.session_state:
    st.session_state.last_telematics = None
if "last_routing" not in st.session_state:
    st.session_state.last_routing = None
if "last_poll_time" not in st.session_state:
    st.session_state.last_poll_time = None


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("🚛 Fleet Control Panel")

telemetry_mode = st.sidebar.selectbox(
    "Telemetry Mode",
    options=[MODE_HARDWARE, MODE_SIMULATOR],
    index=1,
    help=(
        "Hardware mode reads whatever the Traccar Client phone app most recently "
        "pushed to /v1/telematics/incoming. Simulator mode advances a hardcoded "
        "5-point route on every poll."
    ),
)

st.sidebar.divider()

pipeline_toggle = st.sidebar.toggle(
    "🚦 Launch Ingestion Pipeline",
    value=st.session_state.pipeline_running,
    help="When ON, the dashboard polls the backend every "
         f"{POLL_INTERVAL_SECONDS}s and auto-refreshes.",
)
st.session_state.pipeline_running = pipeline_toggle

if pipeline_toggle:
    st.sidebar.success("Pipeline running — polling every "
                        f"{POLL_INTERVAL_SECONDS}s.")
else:
    st.sidebar.info("Pipeline paused. Toggle on to start live polling.")

if telemetry_mode == MODE_HARDWARE:
    st.sidebar.caption(
        "📱 Point Traccar Client's server URL at:\n\n"
        f"`{BACKEND_URL}/v1/telematics/incoming`\n\n"
        "using its OsmAnd / query-string protocol."
    )

st.sidebar.divider()
st.sidebar.caption(f"Backend: `{BACKEND_URL}`")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dispatch, tab_field_report = st.tabs(
    ["🚛 Fleet Dispatch Hub", "📝 Driver Ground-Truth Form"]
)


# ---------------------------------------------------------------------------
# TAB A — Fleet Dispatch Hub
# ---------------------------------------------------------------------------
with tab_dispatch:
    st.title("Northern Cape Fleet Dispatch Hub")
    st.caption("Live spoilage-risk-optimized routing for cold-chain fisheries transport.")

    # One polling tick, only performed while the pipeline is toggled on.
    if st.session_state.pipeline_running:
        if telemetry_mode == MODE_SIMULATOR:
            backend_post("/v1/telematics/simulate-step")
        # In hardware mode we deliberately skip triggering anything here —
        # the phone pushes to /v1/telematics/incoming on its own schedule.

        telematics_payload = backend_get("/v1/telematics/truck-01")
        if telematics_payload is not None:
            st.session_state.last_telematics = telematics_payload

        routing_payload = backend_get("/v1/routing/truck-01")
        if routing_payload is not None:
            st.session_state.last_routing = routing_payload

        st.session_state.last_poll_time = datetime.now().strftime("%H:%M:%S")

    telematics = st.session_state.last_telematics
    routing = st.session_state.last_routing

    if telematics is None:
        st.warning(
            "No telemetry received yet. Toggle **Launch Ingestion Pipeline** in the "
            "sidebar to start polling, and make sure `live_backend.py` is running."
        )
    else:
        feed_source = telematics.get("feed_source", "UNKNOWN")

        # --- Metrics panel ---------------------------------------------------
        metric_cols = st.columns(5)
        metric_cols[0].metric("Vehicle ID", telematics.get("vehicle_id", "—"))
        metric_cols[1].metric(
            "Current Coordinates",
            f"{telematics.get('lat', 0):.4f}, {telematics.get('lon', 0):.4f}",
        )
        metric_cols[2].metric(
            "Cargo Temperature",
            f"{telematics.get('cargo_temp_c', 0):.1f} °C",
        )
        if routing is not None:
            metric_cols[3].metric(
                "ETA to Upington",
                f"{routing.get('total_time_mins', 0):.0f} min",
            )
            metric_cols[4].metric(
                "Spoilage Cost Index",
                f"{routing.get('total_spoilage_cost', 0):.2f}",
                delta="Detour active" if routing.get("detour_active") else "On baseline route",
                delta_color="inverse" if routing.get("detour_active") else "off",
            )

        # --- Feed source status card ------------------------------------------
        if feed_source == FEED_SOURCE_REAL:
            st.success(f"🟢 Data Feed Source: **{feed_source}**")
        elif feed_source == FEED_SOURCE_SIMULATED:
            st.info(f"🔵 Data Feed Source: **{feed_source}**")
        else:
            st.warning(f"⚪ Data Feed Source: **{feed_source}**")

        if st.session_state.last_poll_time:
            st.caption(f"Last updated: {st.session_state.last_poll_time}")

        # --- Map ---------------------------------------------------------------
        st.subheader("Live GPS Map")
        truck_lat = telematics.get("lat")
        truck_lon = telematics.get("lon")

        fleet_map = folium.Map(location=[truck_lat, truck_lon], zoom_start=8, tiles="CartoDB positron")

        folium.Marker(
            location=[truck_lat, truck_lon],
            popup=f"{telematics.get('vehicle_id', 'Vehicle')} — {feed_source}",
            tooltip="Current truck position",
            icon=folium.Icon(color="green" if feed_source == FEED_SOURCE_REAL else "blue", icon="truck", prefix="fa"),
        ).add_to(fleet_map)

        if routing is not None and routing.get("path"):
            route_coords = [[lat, lon] for lon, lat in routing["path"]]
            folium.PolyLine(
                locations=route_coords,
                color="red" if routing.get("detour_active") else "#2c7fb8",
                weight=5,
                opacity=0.8,
                tooltip="Spoilage-optimized route to Upington",
            ).add_to(fleet_map)
            if route_coords:
                folium.Marker(
                    location=route_coords[-1],
                    popup="Destination Hub: Upington",
                    icon=folium.Icon(color="darkred", icon="flag-checkered", prefix="fa"),
                ).add_to(fleet_map)

        st_folium(fleet_map, width=None, height=500, key="fleet_dispatch_map")

    # --- Auto-refresh loop ------------------------------------------------
    if st.session_state.pipeline_running:
        time.sleep(POLL_INTERVAL_SECONDS)
        st.rerun()


# ---------------------------------------------------------------------------
# TAB B — Driver Ground-Truth Form
# ---------------------------------------------------------------------------
with tab_field_report:
    st.title("📝 Driver Ground-Truth Report")
    st.caption(
        "Submit a real-time road condition update. Reports override the routing "
        "impedance for the matched segment immediately — the next routing "
        "recalculation will reroute around bad conditions automatically."
    )

    default_lat = None
    default_lon = None
    if st.session_state.last_telematics is not None:
        default_lat = st.session_state.last_telematics.get("lat")
        default_lon = st.session_state.last_telematics.get("lon")

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

        submitted = st.form_submit_button("🚨 Submit Field Report")

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
                f"✅ Report received and matched to road segment "
                f"{result['matched_segment']} "
                f"({result['distance_from_report_km']:.2f} km from your reported position)."
            )
            st.json(result["applied_override"])
            st.info(
                f"Total active field reports currently affecting routing: "
                f"**{result['active_report_count']}**"
            )
            logger.info(
                "Field report submitted: role=%s condition=%s segment=%s",
                reporter_role, road_condition, result["matched_segment"],
            )
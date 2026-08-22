"""
live_frontend.py

Streamlit dashboard for the Northern Cape Fleet Dispatch system. Polls live_backend.py
for vehicle telemetry and routing, renders the moving vehicle and its optimized route on
an interactive Folium map, and provides a mobile-friendly field reporting form that lets
drivers, fishermen, and cooperative supervisors submit ground-truth road condition
reports that immediately affect routing.

Requires live_backend.py to already be running on http://127.0.0.1:8000.

Run:
    uv run streamlit run live_frontend.py
"""

from __future__ import annotations

import time

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

BACKEND_BASE_URL = "http://127.0.0.1:8000"
TELEMATICS_ENDPOINT = f"{BACKEND_BASE_URL}/v1/telematics/truck-01"
ROUTING_ENDPOINT = f"{BACKEND_BASE_URL}/v1/routing/truck-01"
REPORTS_ENDPOINT = f"{BACKEND_BASE_URL}/v1/reports/submit"

REQUEST_TIMEOUT_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 2.0

# Sample coordinates for the field report form, spanning the Port Nolloth to Upington
# corridor, so a presenter can pick a realistic location without typing raw decimals.
# The Springbok corridor entry is a verified reroute-triggering location: reporting it
# as Impassable will visibly shift the dispatch map's route.
SAMPLE_LOCATIONS: dict[str, tuple[float, float]] = {
    "N7 near Springbok (verified reroute trigger)": (-29.307839, 17.138515),
    "N14 near Pofadder": (-29.1333, 19.4000),
    "Coastal track near Port Nolloth": (-29.2100, 16.9200),
    "N14 near Kenhardt": (-29.3333, 21.1500),
    "Custom coordinates": None,
}

ROAD_CONDITIONS = [
    "Smooth Tarmac",
    "Corrugated / Rough Gravel",
    "Severe Potholes",
    "Impassable / Washed Out",
]

REPORTER_ROLES = ["Driver", "Fisherman", "Cooperative Supervisor"]

st.set_page_config(page_title="Northern Cape Fleet Dispatch", layout="wide")


# ---------------------------------------------------------------------------
# Backend client helpers, each wrapped so a dead backend never crashes the app
# ---------------------------------------------------------------------------
def fetch_telemetry() -> tuple[dict | None, str | None]:
    try:
        response = requests.get(TELEMATICS_ENDPOINT, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def fetch_routing() -> tuple[dict | None, str | None]:
    try:
        response = requests.get(ROUTING_ENDPOINT, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def submit_field_report(payload: dict) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(REPORTS_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "live_tracking_enabled" not in st.session_state:
    st.session_state.live_tracking_enabled = True
if "last_report_result" not in st.session_state:
    st.session_state.last_report_result = None
if "last_report_error" not in st.session_state:
    st.session_state.last_report_error = None


st.title("Northern Cape Fleet Dispatch")
st.caption("Transport, Trade and Fisheries corridor: Port Nolloth to Upington")

tab_dispatch, tab_form = st.tabs(["Fleet Dispatch Hub", "Driver Ground-Truth Form"])


# ---------------------------------------------------------------------------
# TAB A: Fleet Dispatch Hub
# ---------------------------------------------------------------------------
with tab_dispatch:
    st.session_state.live_tracking_enabled = st.checkbox(
        "Enable live auto-refresh (every 2 seconds)",
        value=st.session_state.live_tracking_enabled,
    )

    telemetry, telemetry_error = fetch_telemetry()
    routing, routing_error = fetch_routing()

    if telemetry_error is not None or routing_error is not None:
        st.error(
            "Backend server is offline or unreachable at "
            f"{BACKEND_BASE_URL}. Start it with: uv run live_backend.py"
        )
        if telemetry_error:
            st.caption(f"Telemetry error detail: {telemetry_error}")
        if routing_error:
            st.caption(f"Routing error detail: {routing_error}")
    else:
        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
        with metric_col_1:
            st.metric("Active Vehicle ID", telemetry["vehicle_id"])
        with metric_col_2:
            st.metric("Current Cargo Temperature (C)", f"{telemetry['cargo_temp_c']:.2f}")
        with metric_col_3:
            st.metric(
                "Current Coordinates",
                f"{telemetry['lat']:.4f}, {telemetry['lon']:.4f}",
            )
        with metric_col_4:
            detour_label = "DETOUR ACTIVE" if routing["detour_active"] else "Standard route"
            st.metric("Route Status", detour_label)

        route_coordinates = routing["path"]
        route_latlon = [[lat, lon] for lon, lat in route_coordinates]
        vehicle_latlon = [telemetry["lat"], telemetry["lon"]]

        map_center = route_latlon[len(route_latlon) // 2] if route_latlon else vehicle_latlon
        dispatch_map = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB Positron")

        route_color = "red" if routing["detour_active"] else "green"
        route_label = (
            "Detour route (avoiding a reported hazard)"
            if routing["detour_active"]
            else "Standard optimized route"
        )
        folium.PolyLine(
            locations=route_latlon,
            color=route_color,
            weight=5,
            tooltip=route_label,
        ).add_to(dispatch_map)

        folium.Marker(
            location=vehicle_latlon,
            popup=f"{telemetry['vehicle_id']} (cargo {telemetry['cargo_temp_c']:.1f}C)",
            icon=folium.Icon(icon="truck", prefix="fa", color="blue"),
        ).add_to(dispatch_map)

        if route_latlon:
            folium.Marker(
                location=route_latlon[-1],
                popup="Destination hub: Upington",
                icon=folium.Icon(icon="flag-checkered", prefix="fa", color="darkgreen"),
            ).add_to(dispatch_map)

        st_folium(dispatch_map, use_container_width=True, height=560, key="dispatch_map")

        assessment_col_1, assessment_col_2, assessment_col_3 = st.columns(3)
        with assessment_col_1:
            st.metric("Total Time (mins)", f"{routing['total_time_mins']:.1f}")
        with assessment_col_2:
            st.metric("Total Spoilage Cost", f"{routing['total_spoilage_cost']:.2f}")
        with assessment_col_3:
            st.metric("Route Hop Count", routing["hop_count"])


# ---------------------------------------------------------------------------
# TAB B: Driver Ground-Truth Form
# ---------------------------------------------------------------------------
with tab_form:
    st.subheader("Field Report")
    st.caption(
        "Simulates the mobile form a driver, fisherman, or cooperative supervisor would "
        "use from a loading bay or truck stop to report current road conditions."
    )

    reporter_role = st.selectbox("Select Role", REPORTER_ROLES)

    location_choice = st.selectbox("Location", list(SAMPLE_LOCATIONS.keys()))
    if SAMPLE_LOCATIONS[location_choice] is not None:
        default_lat, default_lon = SAMPLE_LOCATIONS[location_choice]
    else:
        default_lat, default_lon = -29.3078, 17.1385

    coordinate_col_1, coordinate_col_2 = st.columns(2)
    with coordinate_col_1:
        report_lat = st.number_input(
            "Latitude", value=default_lat, min_value=-90.0, max_value=90.0, format="%.6f"
        )
    with coordinate_col_2:
        report_lon = st.number_input(
            "Longitude", value=default_lon, min_value=-180.0, max_value=180.0, format="%.6f"
        )

    road_condition = st.selectbox("Current Road Condition", ROAD_CONDITIONS)
    actual_speed = st.number_input(
        "Actual Traffic Speed (km/h)", min_value=1.0, max_value=160.0, value=40.0, step=1.0
    )

    if st.button("Submit Field Report", type="primary"):
        payload = {
            "reporter_role": reporter_role,
            "lat": report_lat,
            "lon": report_lon,
            "road_condition": road_condition,
            "actual_speed": actual_speed,
        }
        result, error = submit_field_report(payload)
        st.session_state.last_report_result = result
        st.session_state.last_report_error = error

    if st.session_state.last_report_error is not None:
        st.error(
            "Could not reach the backend server to submit this report. "
            f"Detail: {st.session_state.last_report_error}"
        )
    elif st.session_state.last_report_result is not None:
        result = st.session_state.last_report_result
        if road_condition in ("Severe Potholes", "Impassable / Washed Out"):
            st.warning(
                f"Report submitted: {road_condition} confirmed on segment "
                f"{result['matched_segment']} ({result['distance_from_report_km']:.2f} km "
                f"from your reported location). The routing engine has been updated. "
                f"Switch to the Fleet Dispatch Hub tab to see the vehicle reroute."
            )
        else:
            st.success(
                f"Report submitted: {road_condition} recorded on segment "
                f"{result['matched_segment']} ({result['distance_from_report_km']:.2f} km "
                f"from your reported location)."
            )
        with st.expander("Applied override details"):
            st.json(result["applied_override"])


# ---------------------------------------------------------------------------
# Auto-refresh loop
# ---------------------------------------------------------------------------
if st.session_state.live_tracking_enabled:
    time.sleep(POLL_INTERVAL_SECONDS)
    st.rerun()

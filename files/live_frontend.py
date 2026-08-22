"""
live_frontend.py

Streamlit dashboard for the Northern Cape Fleet Dispatch system. The Fleet Dispatch Hub
tab renders a persistent Leaflet map driven entirely by client-side JavaScript: a
setInterval loop polls live_backend.py directly from the browser and a
requestAnimationFrame loop tweens the truck marker between fixes, so the map is never
torn down and rebuilt by a Streamlit rerun. Tab B is an ordinary Streamlit form that
lets drivers, fishermen, and cooperative supervisors submit ground-truth road condition
reports that immediately affect routing.

Requires live_backend.py to already be running on http://127.0.0.1:8000.

Run:
    uv run streamlit run live_frontend.py
"""

from __future__ import annotations

import json

import requests
import streamlit as st

BACKEND_BASE_URL = "http://127.0.0.1:8000"
TELEMATICS_ENDPOINT = f"{BACKEND_BASE_URL}/v1/telematics/truck-01"
ROUTING_ENDPOINT = f"{BACKEND_BASE_URL}/v1/routing/truck-01"
REPORTS_ENDPOINT = f"{BACKEND_BASE_URL}/v1/reports/submit"

REQUEST_TIMEOUT_SECONDS = 5.0

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

# Fixed map center (the corridor midpoint). Never recalculated from the current
# route, so the camera doesn't jump or recentre as the vehicle moves.
MAP_CENTER = [-29.3, 19.0]
MAP_ZOOM = 6

st.set_page_config(page_title="Northern Cape Fleet Dispatch", layout="wide")


# ---------------------------------------------------------------------------
# Backend client helpers (used only by the Tab B form, which is a normal
# Streamlit POST-on-click interaction; the animated map in Tab A talks to the
# backend directly from JS instead, see render_dispatch_component below)
# ---------------------------------------------------------------------------
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
#
# Why the old approach flickered: time.sleep() + st.rerun() (and later the
# st.fragment(run_every=...) version) re-executed the whole render function on
# every cycle. st_folium tears the Folium map down and rebuilds it from scratch
# each time it re-runs, so the map flashed white and the marker "teleported"
# from wherever it last was straight to the new fix, instead of gliding.
#
# Fix: render the map exactly once as a components.html iframe containing a
# plain Leaflet map (not Folium/st_folium — Folium has no notion of "update
# this marker in place", it only knows how to draw a fresh map). The iframe's
# own JavaScript then:
#   1. polls the backend directly with fetch(), independent of Streamlit's
#      script-run cycle entirely, and
#   2. tweens the marker from its last position to the newly-fetched position
#      over the poll interval using requestAnimationFrame, instead of
#      snapping to it.
# Streamlit only re-renders this component when refresh_interval_seconds
# changes (a deliberate user action on the slider), never on a timer.
# ---------------------------------------------------------------------------
def render_dispatch_component(refresh_interval_seconds: int) -> None:
    config = {
        "telematicsUrl": TELEMATICS_ENDPOINT,
        "routingUrl": ROUTING_ENDPOINT,
        "healthUrl": BACKEND_BASE_URL + "/",
        "pollMs": refresh_interval_seconds * 1000,
        "mapCenter": MAP_CENTER,
        "mapZoom": MAP_ZOOM,
    }
    config_json = json.dumps(config)

    component_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" />
<style>
  html, body {{
    margin: 0; padding: 0; background: #0e1117;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  #status-bar {{
    display: flex; flex-wrap: wrap; gap: 10px;
    padding: 10px 4px 14px 4px;
  }}
  .metric {{
    background: #161b22; border: 1px solid #262b33; border-radius: 8px;
    padding: 10px 16px; min-width: 150px; flex: 1;
  }}
  .metric-label {{
    color: #8b949e; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.04em; margin-bottom: 4px;
  }}
  .metric-value {{
    color: #e6edf3; font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums;
  }}
  .metric-value.route-standard {{ color: #3fb950; }}
  .metric-value.route-detour {{ color: #f85149; }}
  #map {{
    height: 560px; width: 100%; border-radius: 8px; border: 1px solid #262b33;
    background: #161b22;
  }}
  #connection-banner {{
    display: none; background: #4a1414; color: #ffb4b4; border: 1px solid #f85149;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; font-size: 13px;
  }}
  #connection-banner.visible {{ display: block; }}
</style>
</head>
<body>
  <div id="connection-banner">
    Backend server is offline or unreachable. Start it with: uv run live_backend.py
  </div>
  <div id="status-bar">
    <div class="metric"><div class="metric-label">Active Vehicle ID</div><div class="metric-value" id="m-vehicle">&mdash;</div></div>
    <div class="metric"><div class="metric-label">Cargo Temperature (&deg;C)</div><div class="metric-value" id="m-temp">&mdash;</div></div>
    <div class="metric"><div class="metric-label">Current Coordinates</div><div class="metric-value" id="m-coords">&mdash;</div></div>
    <div class="metric"><div class="metric-label">Route Status</div><div class="metric-value" id="m-status">&mdash;</div></div>
  </div>
  <div id="map"></div>
  <div id="status-bar">
    <div class="metric"><div class="metric-label">Total Time (mins)</div><div class="metric-value" id="m-time">&mdash;</div></div>
    <div class="metric"><div class="metric-label">Total Spoilage Cost</div><div class="metric-value" id="m-spoilage">&mdash;</div></div>
    <div class="metric"><div class="metric-label">Route Hop Count</div><div class="metric-value" id="m-hops">&mdash;</div></div>
  </div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script>
(function() {{
  const CONFIG = {config_json};

  const map = L.map('map', {{ zoomControl: true }}).setView(CONFIG.mapCenter, CONFIG.mapZoom);
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    subdomains: 'abcd', maxZoom: 19,
  }}).addTo(map);

  // Leaflet sizes its tile grid off the #map div's dimensions at the instant
  // L.map() runs. Inside an iframe that Streamlit is still laying out, that
  // can be a stale/partial size, which throws off tile alignment and centering
  // until something forces a recalculation. invalidateSize() does that; call
  // it once shortly after load and again on any later resize of the iframe.
  setTimeout(function() {{ map.invalidateSize(); map.setView(CONFIG.mapCenter, CONFIG.mapZoom); }}, 200);
  window.addEventListener('resize', function() {{ map.invalidateSize(); }});

  const truckIcon = L.divIcon({{
    className: '', html: '<div style="font-size:22px;transform:translate(-50%,-50%);">&#128666;</div>',
    iconSize: [0, 0],
  }});
  const flagIcon = L.divIcon({{
    className: '', html: '<div style="font-size:20px;transform:translate(-50%,-90%);">&#127937;</div>',
    iconSize: [0, 0],
  }});

  let routeLine = null;
  let vehicleMarker = null;
  let destinationMarker = null;

  // Animation state: we always animate from `displayLatLng` (where the marker
  // visually is right now) to `targetLatLng` (the most recently fetched fix),
  // over `pollMs` milliseconds, using requestAnimationFrame. A new fetch simply
  // retargets the animation instead of snapping the marker.
  let displayLatLng = null;
  let targetLatLng = null;
  let animStartTime = null;
  let animStartLatLng = null;
  let animFrameHandle = null;

  function lerp(a, b, t) {{ return a + (b - a) * t; }}

  function animationStep(timestamp) {{
    if (!targetLatLng) {{ animFrameHandle = requestAnimationFrame(animationStep); return; }}
    if (animStartTime === null) {{ animStartTime = timestamp; animStartLatLng = displayLatLng || targetLatLng; }}

    const elapsed = timestamp - animStartTime;
    const t = Math.min(1, elapsed / CONFIG.pollMs);
    // ease-in-out so the truck doesn't jerk to a stop right as the next fetch lands
    const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

    const lat = lerp(animStartLatLng[0], targetLatLng[0], eased);
    const lon = lerp(animStartLatLng[1], targetLatLng[1], eased);
    displayLatLng = [lat, lon];

    if (!vehicleMarker) {{
      vehicleMarker = L.marker(displayLatLng, {{ icon: truckIcon }}).addTo(map);
    }} else {{
      vehicleMarker.setLatLng(displayLatLng);
    }}

    animFrameHandle = requestAnimationFrame(animationStep);
  }}
  animFrameHandle = requestAnimationFrame(animationStep);

  function setConnectionOk(isOk) {{
    document.getElementById('connection-banner').classList.toggle('visible', !isOk);
  }}

  function updateRoute(routing) {{
    const latlngs = routing.path.map(function(coord) {{ return [coord[1], coord[0]]; }});
    const color = routing.detour_active ? '#f85149' : '#3fb950';

    if (!routeLine) {{
      routeLine = L.polyline(latlngs, {{ color: color, weight: 5 }}).addTo(map);
    }} else {{
      // Update the existing layer in place rather than removing/re-adding it —
      // this is what avoids the map-wide redraw flash.
      routeLine.setLatLngs(latlngs);
      routeLine.setStyle({{ color: color }});
    }}

    if (latlngs.length) {{
      const destination = latlngs[latlngs.length - 1];
      if (!destinationMarker) {{
        destinationMarker = L.marker(destination, {{ icon: flagIcon }})
          .bindPopup('Destination hub: Upington')
          .addTo(map);
      }} else {{
        destinationMarker.setLatLng(destination);
      }}
    }}

    document.getElementById('m-status').textContent = routing.detour_active ? 'DETOUR ACTIVE' : 'Standard route';
    document.getElementById('m-status').className = 'metric-value ' + (routing.detour_active ? 'route-detour' : 'route-standard');
    document.getElementById('m-time').textContent = routing.total_time_mins.toFixed(1);
    document.getElementById('m-spoilage').textContent = routing.total_spoilage_cost.toFixed(2);
    document.getElementById('m-hops').textContent = routing.hop_count;
  }}

  function updateTelemetry(telemetry) {{
    document.getElementById('m-vehicle').textContent = telemetry.vehicle_id;
    document.getElementById('m-temp').textContent = telemetry.cargo_temp_c.toFixed(2);
    document.getElementById('m-coords').textContent = telemetry.lat.toFixed(4) + ', ' + telemetry.lon.toFixed(4);

    const newTarget = [telemetry.lat, telemetry.lon];
    if (!displayLatLng) {{ displayLatLng = newTarget; }}
    targetLatLng = newTarget;
    animStartTime = null; // retarget: next animation frame restarts the tween from wherever we are now
  }}

  async function pollOnce() {{
    try {{
      const [telemetryResponse, routingResponse] = await Promise.all([
        fetch(CONFIG.telematicsUrl), fetch(CONFIG.routingUrl),
      ]);
      if (!telemetryResponse.ok || !routingResponse.ok) throw new Error('non-200 response');
      const telemetry = await telemetryResponse.json();
      const routing = await routingResponse.json();
      updateTelemetry(telemetry);
      updateRoute(routing);
      setConnectionOk(true);
    }} catch (err) {{
      setConnectionOk(false);
    }}
  }}

  pollOnce();
  setInterval(pollOnce, CONFIG.pollMs);
}})();
</script>
</body>
</html>
"""
    # components.v1.html is deprecated in current Streamlit versions in favour of
    # st.iframe, which auto-detects an HTML string and embeds it the same way
    # (srcdoc, JS execution allowed) — no import beyond `streamlit as st` needed.
    # height="content" lets Streamlit measure the actual rendered height instead
    # of a hardcoded guess, so the map isn't clipped with an internal scrollbar.
    st.iframe(component_html, height="content")


with tab_dispatch:
    st.session_state.live_tracking_enabled = st.checkbox(
        "Enable live auto-refresh",
        value=st.session_state.live_tracking_enabled,
    )

    # Defensive guard: recent Streamlit versions reconnect an interrupted
    # WebSocket (e.g. the Windows ConnectionResetError noise you saw) into the
    # *existing* session instead of restarting it, which can replay a
    # widget's session_state value from before a code edit. If that stored
    # value no longer belongs to REFRESH_OPTIONS, select_slider raises
    # "X is not in iterable" instead of just falling back to the default.
    # Clearing the stale entry before instantiating the widget self-heals it.
    REFRESH_OPTIONS = [2, 3, 5, 8]
    REFRESH_KEY = "refresh_interval_seconds"
    if REFRESH_KEY in st.session_state and st.session_state[REFRESH_KEY] not in REFRESH_OPTIONS:
        del st.session_state[REFRESH_KEY]

    refresh_seconds = st.select_slider(
        "Refresh interval (seconds)", options=REFRESH_OPTIONS, value=3, key=REFRESH_KEY,
    )

    if st.session_state.live_tracking_enabled:
        render_dispatch_component(refresh_seconds)
    else:
        st.info("Live tracking paused. Enable it above to resume the dispatch map.")


# ---------------------------------------------------------------------------
# TAB B: Driver Ground-Truth Form (unchanged — a normal Streamlit form, so its
# occasional reruns on submit are fine; it never touches the animated map)
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
                f"The Fleet Dispatch Hub tab will pick up the reroute on its next poll."
            )
        else:
            st.success(
                f"Report submitted: {road_condition} recorded on segment "
                f"{result['matched_segment']} ({result['distance_from_report_km']:.2f} km "
                f"from your reported location)."
            )
        with st.expander("Applied override details"):
            st.json(result["applied_override"])
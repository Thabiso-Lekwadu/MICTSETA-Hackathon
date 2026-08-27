"""data_extraction nodes: pull map, boundary and driver-report sources into raw."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("nc_coldchain")


def extract_town_registry(network_params: dict) -> dict:
    """Publish the service-town + border-post registry into the raw layer."""
    towns = network_params["towns"]
    posts = network_params.get("border_posts", {})
    logger.info("[DATA EXTRACTION] Loading map data: %d towns, %d border posts",
                len(towns), len(posts))
    return {"towns": towns, "border_posts": posts}


def extract_boundary(network_params: dict) -> dict:
    """Load the Northern Cape boundary polygon from GeoJSON into the raw layer."""
    path = Path(network_params["boundary_geojson"])
    coords: list[list[float]] = []
    if path.exists():
        gj = json.loads(path.read_text(encoding="utf-8"))
        coords = _first_polygon_ring(gj)
    if not coords:
        # fall back to the vendored authoritative polygon
        try:
            from nc_coldchain.legacy.nc_boundary import NORTHERN_CAPE_BOUNDARY  # type: ignore
            coords = [list(pt) for pt in NORTHERN_CAPE_BOUNDARY]
        except Exception:
            coords = []
    logger.info("[DATA EXTRACTION] Boundary polygon: %d vertices", len(coords))
    return {"boundary": coords, "vertices": len(coords)}


def _first_polygon_ring(geojson: dict) -> list[list[float]]:
    def rings_from_geom(geom):
        t = geom.get("type")
        c = geom.get("coordinates")
        if t == "Polygon":
            return c[0]
        if t == "MultiPolygon":
            return c[0][0]
        return []
    if geojson.get("type") == "FeatureCollection":
        for feat in geojson.get("features", []):
            ring = rings_from_geom(feat.get("geometry", {}))
            if ring:
                return [[float(x), float(y)] for x, y, *_ in ring]
    elif geojson.get("type") == "Feature":
        ring = rings_from_geom(geojson.get("geometry", {}))
        return [[float(x), float(y)] for x, y, *_ in ring]
    elif geojson.get("type") in ("Polygon", "MultiPolygon"):
        ring = rings_from_geom(geojson)
        return [[float(x), float(y)] for x, y, *_ in ring]
    return []


def extract_osm_roads(network_params: dict) -> dict:
    """Pull the Northern Cape road network from OpenStreetMap (Geofabrik) and
    preprocess it into the extract the graph build reads — the Kedro-native port
    of fast_nc_roads.py.

    Idempotent: if the target file already exists it is reused. Gated by
    network.download_osm so an offline/synthetic run never triggers the large
    download. Returns a marker {'path', 'exists', 'downloaded'} that the
    preprocessing graph node depends on (so extraction is ordered first).
    """
    if str(network_params.get("road_source", "osm")).lower() != "osm":
        logger.info("[DATA EXTRACTION] road_source is synthetic — skipping OSM extract.")
        return {"path": None, "exists": False, "downloaded": False, "skipped": True}

    target = Path(network_params["osm_road_path"])
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        logger.info("[DATA EXTRACTION] OSM extract already present: %s (reusing)", target)
        return {"path": str(target), "exists": True, "downloaded": False}

    if not network_params.get("download_osm", False):
        logger.info("[DATA EXTRACTION] OSM extract missing and download_osm=false — "
                    "graph build will decide (fail-loud or synthetic).")
        return {"path": str(target), "exists": False, "downloaded": False}

    # ---- download + slice (ported from fast_nc_roads.py) ---------------------
    import os
    import zipfile

    import geopandas as gpd
    import requests

    url = network_params.get(
        "osm_download_url",
        "https://download.geofabrik.de/africa/south-africa-latest-free.shp.zip")
    bbox = network_params.get("osm_bbox", [16.45, -31.85, 25.30, -24.60])  # lon/lat
    work = target.parent / "_osm_download"
    work.mkdir(parents=True, exist_ok=True)
    local_zip = work / "south-africa-latest-free.shp.zip"
    extract_dir = work / "unzipped"

    logger.info("[DATA EXTRACTION] Downloading OSM roads from %s ...", url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    with requests.get(url, headers=headers, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(local_zip, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)

    logger.info("[DATA EXTRACTION] Unzipping road vector layer ...")
    extract_dir.mkdir(exist_ok=True)
    wanted = ("gis_osm_roads_free_1.shp", "gis_osm_roads_free_1.shx",
              "gis_osm_roads_free_1.dbf", "gis_osm_roads_free_1.prj")
    with zipfile.ZipFile(local_zip, "r") as zf:
        for f in zf.namelist():
            if any(f.endswith(w) for w in wanted):
                zf.extract(f, extract_dir)
    shp = None
    for root, _dirs, files in os.walk(extract_dir):
        for name in files:
            if name.endswith("gis_osm_roads_free_1.shp"):
                shp = os.path.join(root, name)
    if shp is None:
        raise FileNotFoundError("gis_osm_roads_free_1.shp not found in the archive.")

    logger.info("[DATA EXTRACTION] Loading shapefile + slicing to Northern Cape bbox ...")
    roads = gpd.read_file(shp, engine="pyogrio")
    min_lon, min_lat, max_lon, max_lat = bbox
    nc = roads.cx[min_lon:max_lon, min_lat:max_lat]
    logger.info("[DATA EXTRACTION] %d NC road segments (of %d SA). Writing %s",
                len(nc), len(roads), target)
    if str(target).endswith(".gpkg"):
        nc.to_file(target, driver="GPKG", layer="edges", engine="pyogrio")
    elif str(target).endswith(".parquet"):
        nc.to_parquet(target)
    else:
        nc.to_file(target, driver="GeoJSON")

    # cleanup temp download
    try:
        local_zip.unlink(missing_ok=True)
        for root, dirs, files in os.walk(extract_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        extract_dir.rmdir()
    except OSError:
        pass
    logger.info("[DATA EXTRACTION] OSM extract ready: %s", target)
    return {"path": str(target), "exists": True, "downloaded": True}


def fetch_live_weather(network_params: dict, runtime_params: dict) -> dict:
    """Call the REAL OpenWeatherMap API (via weather_engine + the injected key)
    for current ambient at the anchor town. Gated by runtime.use_live_weather so
    the default run stays offline/deterministic. Fail-safe: any error or missing
    key degrades to Open-Meteo / fallback inside weather_engine, and a network
    failure here returns an 'enabled: false' record rather than crashing the run.
    """
    if not runtime_params.get("use_live_weather", False):
        logger.info("[WEATHER] Live weather disabled (runtime.use_live_weather=false).")
        return {"enabled": False, "reason": "disabled"}
    try:
        from nc_coldchain.legacy import weather_engine as we

        name, (lon, lat) = next(iter(network_params["towns"].items()))
        reading = we.get_current_weather(float(lat), float(lon))
        logger.info("[WEATHER] Live reading for %s: %.1f C (source=%s)",
                    name, reading["temp_c"], reading["source"])
        return {"enabled": True, "anchor_town": name, "ambient_temp_c": reading["temp_c"],
                "source": reading["source"], "alert": reading.get("alert")}
    except Exception as exc:  # never let a weather call break the pipeline
        logger.warning("[WEATHER] Live fetch failed (%s) — continuing without it.", exc)
        return {"enabled": False, "reason": str(exc)}


def extract_drivers_reports(network_params: dict, repro_params: dict) -> pd.DataFrame:
    """Deterministically synthesise a small driver hazard-report table (raw source).

    In production this node would read the live Driver Reporting Desk submissions;
    here it emits a reproducible seed set so the downstream lifecycle has driver
    data to audit and fold into edge overrides.
    """
    import random

    rng = random.Random(repro_params["random_seed"])
    towns = list(network_params["towns"].items())
    hazards = [
        ("flooding", 0.35), ("rough_surface", 0.15),
        ("closure", 1.00), ("stock_on_road", 0.10), ("pothole", 0.20),
    ]
    rows = []
    for i in range(6):
        name, (lon, lat) = towns[rng.randrange(len(towns))]
        hz, mult_bump = hazards[rng.randrange(len(hazards))]
        rows.append({
            "report_id": f"DR-{i+1:03d}",
            "town_near": name,
            "lat": round(lat + rng.uniform(-0.2, 0.2), 5),
            "lon": round(lon + rng.uniform(-0.2, 0.2), 5),
            "hazard_type": hz,
            "severity": rng.choice(["low", "medium", "high"]),
            "edge_weight_multiplier": round(1.0 + mult_bump, 3),
            "timestamp_iso": f"2026-08-2{rng.randint(0,6)}T0{rng.randint(6,9)}:00:00+00:00",
        })
    df = pd.DataFrame(rows)
    logger.info("[DATA EXTRACTION] Loading drivers data: %d hazard reports", len(df))
    return df

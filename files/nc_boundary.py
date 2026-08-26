"""
nc_boundary.py

Precise Northern Cape province boundary + geometry helpers, shared by
live_backend.py (routing-time province clipping) and nc_road_network_improved.py
(preprocessing-time clipping). No shapely/geopandas dependency — a pure-Python
ray-casting point-in-polygon test, so it works everywhere the rest of the stack
does, including inside the FastAPI process.

WHY THIS EXISTS
---------------
The raw OSM extract (fast_nc_roads.py) was clipped to a rectangular BOUNDING BOX
around the Northern Cape. But the province is not a rectangle — the North West /
Free State border cuts in above Kimberley, so paved North West roads (the N14
corridor toward Vryburg) sit INSIDE that bbox and leaked into the routable graph.
The spoilage-optimizer then happily used them, producing "optimal" routes that
physically leave the province. Clipping to the true province POLYGON (below)
removes those roads so every optimal path stays in the Northern Cape.

The polygon was sourced from an open dataset of South African provincial
boundaries and validated: all seven project towns fall inside it, and known
North West / Free State / Western Cape points (incl. the exact Kimberley→Vryburg
N14 dip) fall outside.

For production you can drop a higher-resolution `northern_cape_boundary.geojson`
(a single Polygon/MultiPolygon feature, WGS84 lon/lat) next to this file and it
will be loaded in preference to the embedded ring — see load_boundary_override().
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger("nc_boundary")

# Northern Cape outer boundary ring, [lon, lat], WGS84. ~185 vertices — accurate
# enough to separate the province from its neighbours along every border,
# including the concave North West notch above Kimberley.
NORTHERN_CAPE_BOUNDARY: list[list[float]] = [
    [16.4871, -28.5729], [16.6731, -28.4598], [16.7932, -28.3743], [16.8411, -28.2099],
    [16.8687, -28.1679], [16.893, -28.0826], [16.9738, -28.055], [17.012, -28.0583],
    [17.0454, -28.0363], [17.1118, -28.046], [17.1558, -28.0925], [17.1898, -28.1394],
    [17.192, -28.2088], [17.2131, -28.2321], [17.3086, -28.2241], [17.3641, -28.2733],
    [17.4096, -28.5714], [17.4409, -28.7096], [17.4852, -28.7002], [17.6034, -28.7108],
    [17.7141, -28.7511], [17.9133, -28.7813], [18.0824, -28.876], [18.1665, -28.9019],
    [18.3976, -28.8989], [18.5177, -28.8823], [18.7457, -28.8399], [18.9545, -28.8667],
    [19.0817, -28.9594], [19.1615, -28.9454], [19.2766, -28.8895], [19.4346, -28.7135],
    [19.5117, -28.598], [19.5792, -28.5252], [19.6883, -28.516], [19.7963, -28.4842],
    [19.8967, -28.4277], [19.9832, -28.3927], [19.983, -27.6804], [19.9827, -27.0637],
    [19.9826, -26.7273], [19.9823, -26.0545], [19.9821, -25.6621], [19.9819, -24.9893],
    [19.9864, -24.7688], [20.1189, -24.8874], [20.3649, -25.0332], [20.4625, -25.2237],
    [20.5251, -25.3322], [20.6114, -25.4394], [20.6965, -25.7071], [20.7941, -25.8939],
    [20.8414, -26.1313], [20.7947, -26.2019], [20.6955, -26.342], [20.6225, -26.4275],
    [20.6387, -26.817], [20.7794, -26.8481], [20.9077, -26.8], [21.1225, -26.8653],
    [21.2829, -26.845], [21.4495, -26.8264], [21.6645, -26.8631], [21.7438, -26.8205],
    [21.9358, -26.6639], [22.0577, -26.6176], [22.1783, -26.4399], [22.2573, -26.3409],
    [22.3429, -26.3174], [22.4826, -26.2048], [22.6248, -26.1115], [22.7024, -26.4132],
    [22.8538, -26.5947], [22.9637, -26.628], [23.031, -26.6728], [23.0737, -26.9997],
    [23.0813, -27.2118], [23.2119, -27.334], [23.4486, -27.4164], [23.571, -27.4749],
    [23.7263, -27.5369], [23.8911, -27.4836], [24.0134, -27.3362], [24.4293, -27.7901],
    [24.5578, -28.0525], [24.6479, -27.8656], [24.694, -27.7898], [24.7382, -27.5854],
    [24.972, -27.6855], [25.0267, -27.7077], [24.9489, -28.0982], [25.0051, -28.0701],
    [25.0135, -28.0682], [24.9812, -28.1884], [24.9235, -28.2593], [24.916, -28.3174],
    [24.8918, -28.4202], [24.8637, -28.466], [24.7618, -28.8654], [24.6551, -29.0693],
    [24.417, -29.5099], [24.4746, -29.7856], [24.5104, -29.8116], [24.5754, -29.866],
    [24.6462, -29.888], [24.6792, -29.9608], [24.7072, -29.9742], [24.8073, -30.0217],
    [24.9069, -30.2012], [24.9345, -30.2143], [24.9782, -30.2579], [25.0435, -30.3302],
    [25.1313, -30.3996], [25.2579, -30.4969], [25.3146, -30.5465], [25.3968, -30.5636],
    [25.4675, -30.6131], [25.483, -30.7558], [25.5066, -30.9344], [25.4723, -30.9957],
    [25.4455, -31.0952], [25.3872, -31.1607], [25.2151, -31.2016], [25.0633, -31.2644],
    [24.9122, -31.3202], [24.809, -31.3692], [24.6959, -31.3979], [24.5579, -31.423],
    [24.3346, -31.7193], [24.2417, -31.7492], [24.147, -31.7899], [24.0232, -31.7195],
    [23.8088, -31.7276], [23.5512, -31.6779], [23.2009, -31.8696], [23.0378, -31.9325],
    [22.943, -31.8564], [22.6055, -31.7766], [22.1444, -31.8542], [22.0873, -31.9102],
    [22.0781, -32.0086], [22.08, -32.0955], [21.8527, -32.243], [21.6764, -32.2228],
    [21.4231, -32.3485], [21.1178, -32.6072], [20.7051, -32.9181], [20.4212, -32.9369],
    [20.3084, -32.84], [20.1858, -32.7372], [20.0951, -32.5219], [20.0166, -32.2946],
    [19.8013, -32.3728], [19.6729, -32.4499], [19.5336, -32.6376], [19.4729, -32.5835],
    [19.4456, -32.077], [19.3192, -32.0223], [19.1956, -31.9202], [19.0951, -31.8996],
    [19.0286, -31.8932], [19.0606, -31.5534], [19.0085, -31.43], [18.9965, -31.3051],
    [18.9283, -30.9422], [18.9379, -30.8442], [18.7717, -30.6168], [18.5071, -30.4757],
    [18.3622, -30.5775], [18.2472, -30.7828], [18.1569, -30.8217], [18.024, -30.7751],
    [17.8284, -31.063], [17.7503, -31.1319], [17.5482, -30.7872], [17.4295, -30.5768],
    [17.3735, -30.4872], [17.2744, -30.3117], [17.1664, -30.0129], [17.0972, -29.8508],
    [17.0563, -29.6764], [16.9783, -29.4679], [16.8849, -29.299], [16.8477, -29.2199],
    [16.7205, -28.9776], [16.5895, -28.8359], [16.5412, -28.7053], [16.4803, -28.6135],
    [16.4871, -28.5729],
]

# Distance a node may sit OUTSIDE the polygon and still count as eligible — keeps
# legitimate border-approach roads (e.g. the Orange River / Vioolsdrift crossing,
# which lies right on the Namibia line) without re-admitting the far North West
# dip, which is ~100 km from the boundary.
BORDER_INCLUDE_KM = 25.0


def _active_boundary() -> list[list[float]]:
    override = load_boundary_override()
    return override if override else NORTHERN_CAPE_BOUNDARY


def point_in_northern_cape(lon: float, lat: float, boundary: list[list[float]] | None = None) -> bool:
    """Ray-casting point-in-polygon test (WGS84 lon/lat). True if the point is
    inside the Northern Cape province ring."""
    poly = boundary if boundary is not None else _active_boundary()
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def node_eligible(lon: float, lat: float, border_posts: dict | None = None,
                  border_km: float = BORDER_INCLUDE_KM, boundary: list[list[float]] | None = None) -> bool:
    """Whether a graph node at (lon, lat) may be used for Northern Cape routing:
    inside the province polygon, OR within `border_km` of a known border post
    (so cross-border legs to Namibia via Vioolsdrift still work). `border_posts`
    is a {name: (lon, lat)} dict (e.g. nc_road_network.BORDER_POSTS)."""
    if point_in_northern_cape(lon, lat, boundary):
        return True
    if border_posts:
        for _name, (blon, blat) in border_posts.items():
            if _haversine_km(lon, lat, blon, blat) <= border_km:
                return True
    return False


def load_boundary_override(path: Path | None = None) -> list[list[float]] | None:
    """Loads a higher-resolution boundary from `northern_cape_boundary.geojson`
    next to this module if present (a single Polygon/MultiPolygon feature, WGS84).
    Returns the largest outer ring as [[lon,lat],...], or None. Never raises."""
    geojson_path = path or Path(__file__).resolve().with_name("northern_cape_boundary.geojson")
    try:
        if not geojson_path.exists():
            return None
        data = json.loads(geojson_path.read_text(encoding="utf-8"))
        # Accept a FeatureCollection, a Feature, or a bare geometry.
        geom = data
        if data.get("type") == "FeatureCollection":
            geom = data["features"][0]["geometry"]
        elif data.get("type") == "Feature":
            geom = data["geometry"]
        if geom["type"] == "Polygon":
            rings = [geom["coordinates"][0]]
        elif geom["type"] == "MultiPolygon":
            rings = [poly[0] for poly in geom["coordinates"]]
        else:
            return None
        largest = max(rings, key=len)
        return [[float(pt[0]), float(pt[1])] for pt in largest]
    except Exception as exc:  # noqa: BLE001 - a bad override file must never break routing
        logger.warning("nc_boundary -> could not load override %s (%s); using embedded ring.", geojson_path, exc)
        return None
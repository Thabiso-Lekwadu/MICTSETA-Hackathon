"""
data_ingestion.py

Fully automated raw-data extraction: downloads the South Africa OSM road network
extract from Geofabrik, validates the download, extracts the roads shapefile
components, and clips to the Northern Cape bounding box. This is the pipeline's
real "extract" step -- no manually-placed file, no dependency on a previously
recovered dataset. `kedro run` on a machine with internet access reproduces
data/01_raw/ from nothing.

The original hand-run version of this download (fast_nc_roads.py) had a real bug
worth remembering: it pointed at the bare Geofabrik homepage instead of the actual
file URL, silently downloaded an HTML error page, and only failed later at
`zipfile.ZipFile()` with a confusing "not a zip file" error. This version validates
the response Content-Type immediately after the request, so a wrong URL or a
temporary Geofabrik outage fails fast with a clear message instead of producing a
corrupt local file that passes silently until the next pipeline step.
"""

from __future__ import annotations

import logging
import tempfile
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import requests

logger = logging.getLogger(__name__)

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DOWNLOAD_CHUNK_SIZE = 8192
DOWNLOAD_TIMEOUT_SECONDS = 120
MAX_DOWNLOAD_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


def _download_with_retries(url: str, dest_path: Path) -> None:
    last_exception: Exception | None = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)...", url, attempt, MAX_DOWNLOAD_RETRIES)
            response = requests.get(
                url, headers=REQUEST_HEADERS, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "zip" not in content_type and "octet-stream" not in content_type:
                raise ValueError(
                    f"Expected a zip download but got Content-Type '{content_type}'. "
                    f"This usually means the URL points at an HTML page (a redirect, "
                    f"an error page, or a mistyped path) rather than the actual file. "
                    f"URL used: {url}"
                )

            bytes_written = 0
            with dest_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

            if bytes_written == 0:
                raise ValueError("Download completed but the file is empty (0 bytes).")
            if not zipfile.is_zipfile(dest_path):
                raise ValueError(
                    f"Downloaded {bytes_written:,} bytes but the result is not a valid "
                    f"zip file. The server may have returned a truncated response."
                )

            logger.info("Download OK: %.1f MB", bytes_written / (1024 * 1024))
            return

        except (requests.RequestException, ValueError) as exc:
            last_exception = exc
            logger.warning("Attempt %d/%d failed: %s", attempt, MAX_DOWNLOAD_RETRIES, exc)
            if dest_path.exists():
                dest_path.unlink()
            if attempt < MAX_DOWNLOAD_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)

    raise RuntimeError(
        f"Failed to download a valid zip from {url} after {MAX_DOWNLOAD_RETRIES} attempts. "
        f"Last error: {last_exception}"
    )


def download_and_extract_raw_roads(
    geofabrik_url: str,
    shapefile_prefix: str,
    clip_bbox: dict[str, float],
) -> gpd.GeoDataFrame:
    """
    The pipeline's real extract step.

    geofabrik_url:    direct URL to the country-level *-free.shp.zip archive
                       (must be the actual file URL, not a Geofabrik listing page --
                       see the module docstring for why that distinction matters).
    shapefile_prefix: the target shapefile's basename inside the archive, without
                       extension, e.g. "gis_osm_roads_free_1".
    clip_bbox:         {min_lon, max_lon, min_lat, max_lat} to clip the country-wide
                       extract down to the Northern Cape.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_path = tmp_path / "country_extract.shp.zip"
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        _download_with_retries(geofabrik_url, zip_path)

        target_suffixes = [f"{shapefile_prefix}{ext}" for ext in (".shp", ".shx", ".dbf", ".prj")]
        logger.info("Extracting road shapefile components: %s", target_suffixes)
        with zipfile.ZipFile(zip_path, "r") as zf:
            matched = [name for name in zf.namelist() if any(name.endswith(s) for s in target_suffixes)]
            if not matched:
                raise ValueError(
                    f"None of {target_suffixes} were found inside the downloaded archive. "
                    f"Archive contents: {zf.namelist()[:20]}{'...' if len(zf.namelist()) > 20 else ''}"
                )
            for name in matched:
                zf.extract(name, extract_dir)

        shp_path = extract_dir / f"{shapefile_prefix}.shp"
        if not shp_path.exists():
            # Geofabrik archives sometimes nest contents one directory deep.
            candidates = list(extract_dir.rglob(f"{shapefile_prefix}.shp"))
            if not candidates:
                raise FileNotFoundError(f"Could not locate {shapefile_prefix}.shp after extraction.")
            shp_path = candidates[0]

        logger.info("Loading shapefile: %s", shp_path)
        roads_gdf = gpd.read_file(shp_path, engine="pyogrio")
        logger.info("Loaded %d road segments for the full country extract.", len(roads_gdf))

        clipped = roads_gdf.cx[
            clip_bbox["min_lon"]:clip_bbox["max_lon"],
            clip_bbox["min_lat"]:clip_bbox["max_lat"],
        ].copy()
        logger.info(
            "Clipped to Northern Cape bounding box: %d / %d segments retained.",
            len(clipped), len(roads_gdf),
        )

        if len(clipped) == 0:
            raise ValueError(
                f"Clipping to {clip_bbox} produced zero road segments. Check that the "
                f"bounding box coordinates and the source extract's CRS actually overlap."
            )

        return clipped

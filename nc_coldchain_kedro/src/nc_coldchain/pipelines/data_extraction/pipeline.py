"""data_extraction pipeline: sources -> data/01_raw."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    extract_boundary,
    extract_drivers_reports,
    extract_osm_roads,
    extract_town_registry,
    fetch_live_weather,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(extract_town_registry, "params:network", "raw_town_registry",
             name="extract_town_registry"),
        node(extract_boundary, "params:network", "raw_boundary_polygon",
             name="extract_boundary"),
        node(extract_osm_roads, "params:network", "raw_osm_ready",
             name="extract_osm_roads"),
        node(extract_drivers_reports, ["params:network", "params:reproducibility"],
             "raw_drivers_reports", name="extract_drivers_reports"),
        node(fetch_live_weather, ["params:network", "params:runtime"],
             "raw_live_weather", name="extract_live_weather"),
    ])

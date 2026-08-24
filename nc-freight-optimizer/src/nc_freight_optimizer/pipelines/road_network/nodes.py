"""Kedro nodes for the road_network pipeline.

Each function here is a thin wrapper: real logic lives in road_network_core.py,
these just adapt it to Kedro's catalog-in / catalog-out node signature. This mirrors
the extract / transform / load split: validate_raw_roads is the extract-time gate,
clean_and_enrich_roads and build_road_graph are transforms, bundle_graph_artifacts
is the load step that produces the final artifact the live backend consumes.
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx

from nc_freight_optimizer import road_network_core as core
from nc_freight_optimizer.data_ingestion import download_and_extract_raw_roads


def extract_raw_roads(geofabrik_url: str, shapefile_prefix: str, clip_bbox: dict) -> gpd.GeoDataFrame:
    """The pipeline's true first node: downloads and clips the raw OSM extract
    from Geofabrik. No manually-placed file, no dependency on previously
    recovered data -- `kedro run` reproduces data/01_raw/ from nothing, given
    internet access."""
    return download_and_extract_raw_roads(geofabrik_url, shapefile_prefix, clip_bbox)


def validate_raw_roads(raw_roads: gpd.GeoDataFrame, expected_bbox: dict, bbox_tolerance_deg: float) -> gpd.GeoDataFrame:
    return core.validate_raw_roads(raw_roads, expected_bbox, bbox_tolerance_deg)


def clean_and_enrich_roads(
    validated_roads: gpd.GeoDataFrame,
    exclude_fclass: list[str],
    simplify_tolerance_deg: float,
    default_speed_kmh: dict[str, float],
    roughness_multiplier: dict[str, float],
) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    return core.clean_and_enrich(
        validated_roads, exclude_fclass, simplify_tolerance_deg,
        default_speed_kmh, roughness_multiplier,
    )


def build_road_graph(
    cleaned_roads: gpd.GeoDataFrame, snap_tolerance_m: float
) -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
    return core.build_graph(cleaned_roads, snap_tolerance_m)


def extract_main_component(graph_full: nx.Graph) -> tuple[nx.Graph, list[int]]:
    graph_main, main_nodes = core.largest_component_subgraph(graph_full)
    return graph_main, sorted(main_nodes)


def bundle_graph_artifacts(
    graph_full: nx.Graph,
    graph_main: nx.Graph,
    main_nodes: list[int],
    cluster_coord: dict[int, tuple[float, float]],
) -> dict:
    """Produces the single combined artifact live_backend.py loads. Kept as one
    bundled pickle (rather than four separate catalog entries) because the live
    app needs all four pieces together at startup and this avoids four separate
    file reads on every server boot."""
    return {
        "G_full": graph_full,
        "G_main": graph_main,
        "main_nodes": set(main_nodes),
        "cluster_coord": cluster_coord,
    }

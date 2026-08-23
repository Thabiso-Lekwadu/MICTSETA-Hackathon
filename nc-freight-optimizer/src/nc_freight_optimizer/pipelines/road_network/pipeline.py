from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline as modular_pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return modular_pipeline([
        node(
            func=nodes.validate_raw_roads,
            inputs=["raw_roads", "params:road_network.expected_bbox", "params:road_network.bbox_tolerance_deg"],
            outputs="validated_roads",
            name="validate_raw_roads_node",
        ),
        node(
            func=nodes.clean_and_enrich_roads,
            inputs=[
                "validated_roads",
                "params:road_network.exclude_fclass",
                "params:road_network.simplify_tolerance_deg",
                "params:road_network.default_speed_kmh",
                "params:road_network.roughness_multiplier",
            ],
            outputs=["cleaned_roads", "speed_lookup"],
            name="clean_and_enrich_roads_node",
        ),
        node(
            func=nodes.build_road_graph,
            inputs=["cleaned_roads", "params:road_network.snap_tolerance_m"],
            outputs=["road_graph_full", "cluster_coord"],
            name="build_road_graph_node",
        ),
        node(
            func=nodes.extract_main_component,
            inputs="road_graph_full",
            outputs=["road_graph_main", "main_nodes"],
            name="extract_main_component_node",
        ),
        node(
            func=nodes.bundle_graph_artifacts,
            inputs=["road_graph_full", "road_graph_main", "main_nodes", "cluster_coord"],
            outputs="nc_road_graph_bundle",
            name="bundle_graph_artifacts_node",
        ),
    ])

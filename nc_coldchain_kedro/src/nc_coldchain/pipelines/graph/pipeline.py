"""graph pipeline: validate + measure the real road-graph bundle."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import calculate_graph_metrics, validate_graph


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(validate_graph, ["road_graph_bundle", "params:graph"],
             "graph_validation", name="graph_validate"),
        node(calculate_graph_metrics, "road_graph_bundle",
             "graph_metrics", name="graph_calculate_metrics"),
    ])

"""graph_optimization pipeline: real-graph candidates -> nc_road_graph + publish."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_optimal_graph, evaluate_candidates, publish_graph


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            evaluate_candidates,
            ["road_graph_bundle", "graph_validation", "params:optimization",
             "params:graph"],
            "candidate_graph_scores",
            name="optimize_evaluate_candidates",
        ),
        node(
            build_optimal_graph,
            ["road_graph_bundle", "candidate_graph_scores", "params:graph",
             "audit_feedback", "params:reproducibility"],
            ["nc_road_graph", "optimal_graph_metadata"],
            name="optimize_build_optimal_graph",
        ),
        node(
            publish_graph,
            ["nc_road_graph", "params:optimization"],
            ["optimal_graph_published", "optimal_graph_geojson"],
            name="optimize_publish_graph",
        ),
    ])

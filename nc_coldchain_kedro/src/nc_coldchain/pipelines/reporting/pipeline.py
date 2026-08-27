"""reporting pipeline: consolidate outputs -> data/09_reporting."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_data_quality_report,
    build_graph_change_report,
    build_simulation_report,
    build_system_status,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(build_data_quality_report, ["audit_results", "audit_feedback"],
             "data_quality_report", name="report_data_quality"),
        node(build_graph_change_report,
             ["optimal_graph_metadata", "audit_feedback", "graph_metrics"],
             "graph_change_report", name="report_graph_change"),
        node(build_simulation_report,
             ["monte_carlo_results", "optimal_graph_metadata"],
             "simulation_report", name="report_simulation"),
        node(build_system_status,
             ["optimal_graph_metadata", "graph_metrics", "audit_feedback",
              "monte_carlo_results"],
             "system_status", name="report_system_status"),
    ])

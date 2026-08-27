"""data_audit pipeline: primary/feature -> audit results, report, feedback."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import audit_tables, generate_audit_feedback, generate_audit_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            audit_tables,
            ["prm_weather", "prm_road", "prm_sensors", "prm_drivers",
             "feat_graph_nodes", "feat_graph_edges", "params:audit"],
            "audit_results",
            name="audit_run_checks",
        ),
        node(generate_audit_report, "audit_results", "audit_report",
             name="audit_generate_report"),
        node(
            generate_audit_feedback,
            ["audit_results", "feat_graph_nodes", "feat_graph_edges", "params:audit"],
            "audit_feedback",
            name="audit_generate_feedback",
        ),
    ])

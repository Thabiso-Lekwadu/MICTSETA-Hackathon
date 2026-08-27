"""Shared artifact paths. The backend reads ONLY these Kedro-produced artifacts;
it never re-runs pipeline logic. Path root is overridable via DATA_PATH.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_PATH", "data"))

OPTIMAL_GRAPH_JSON = DATA_ROOT / "08_graph" / "optimized" / "optimal_graph.json"
OPTIMAL_GRAPH_GEOJSON = DATA_ROOT / "08_graph" / "visualizations" / "optimal_graph.geojson"
OPTIMAL_GRAPH_METADATA = DATA_ROOT / "08_graph" / "optimized" / "metadata.json"
GRAPH_METRICS = DATA_ROOT / "08_graph" / "processed" / "graph_metrics.json"
CANDIDATE_SCORES = DATA_ROOT / "08_graph" / "optimized" / "candidate_scores.json"

AUDIT_RESULTS = DATA_ROOT / "09_reporting" / "audit" / "audit_results.json"
AUDIT_FEEDBACK = DATA_ROOT / "09_reporting" / "audit" / "audit_feedback.json"

MONTE_CARLO_RESULTS = DATA_ROOT / "07_model_output" / "monte_carlo_results.json"

DATA_QUALITY_REPORT = DATA_ROOT / "09_reporting" / "data_quality_report.json"
GRAPH_CHANGE_REPORT = DATA_ROOT / "09_reporting" / "graph_change_report.json"
SIMULATION_REPORT = DATA_ROOT / "09_reporting" / "simulation_report.json"
SYSTEM_STATUS = DATA_ROOT / "09_reporting" / "system_status.json"

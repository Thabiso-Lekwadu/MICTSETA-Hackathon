"""reporting nodes: consolidate audit, graph, optimisation and simulation outputs."""
from __future__ import annotations

import logging

logger = logging.getLogger("nc_coldchain")


def build_data_quality_report(audit_results: dict, audit_feedback: dict) -> dict:
    tables = {n: {"rows": t["rows"], "duplicate_rows": t["duplicate_rows"],
                  "max_missing_fraction": round(max(t["missing_fraction"].values() or [0.0]), 4),
                  "range_violations": int(sum(t["range_violations"].values()))}
              for n, t in audit_results["tables"].items()}
    return {"passed": audit_results["passed"],
            "quality_issues": audit_results["quality_issues"],
            "tables": tables,
            "feedback": {"data_valid": audit_feedback["data_valid"],
                         "graph_rebuild_required": audit_feedback["graph_rebuild_required"],
                         "reason": audit_feedback["reason"]}}


def build_graph_change_report(metadata: dict, audit_feedback: dict,
                              graph_metrics: dict) -> dict:
    prev = audit_feedback.get("previous", {})
    report = {
        "current": {"graph_version": metadata["graph_version"],
                    "node_count": metadata["node_count"],
                    "edge_count": metadata["edge_count"],
                    "objective_score": metadata["objective_score"],
                    "mean_edge_weight": metadata["mean_edge_weight"]},
        "previous": {"graph_version": prev.get("graph_version"),
                     "node_count": prev.get("node_count"),
                     "edge_count": prev.get("edge_count"),
                     "mean_edge_weight": prev.get("mean_edge_weight")},
        "changes": {
            "node_count": f"{prev.get('node_count')} -> {metadata['node_count']}",
            "edge_count": f"{prev.get('edge_count')} -> {metadata['edge_count']}",
            "objective_score_new": metadata["objective_score"],
        },
        "rebuild_reason": audit_feedback.get("reason"),
        "connectivity": graph_metrics.get("is_connected"),
    }
    logger.info("[REPORTING] Graph change: %s -> %s",
                prev.get("graph_version"), metadata["graph_version"])
    return report


def build_simulation_report(monte_carlo_results: dict, metadata: dict) -> dict:
    routes = monte_carlo_results.get("routes", {})
    worst = max(routes.items(), key=lambda kv: kv[1]["prob_breach"], default=(None, {}))
    return {"graph_version": metadata["graph_version"],
            "ambient_temp_c": monte_carlo_results.get("ambient_temp_c"),
            "routes": routes,
            "highest_risk_route": worst[0],
            "highest_risk_prob_breach": worst[1].get("prob_breach")}


def build_system_status(metadata: dict, graph_metrics: dict, audit_feedback: dict,
                        monte_carlo_results: dict) -> dict:
    """Single observability snapshot the backend serves at /health and /status."""
    status = {
        "pipeline_complete": True,
        "graph_available": True,
        "graph_version": metadata["graph_version"],
        "graph_created_at": metadata["created_at"],
        "node_count": metadata["node_count"],
        "edge_count": metadata["edge_count"],
        "graph_connected": graph_metrics.get("is_connected"),
        "objective_score": metadata["objective_score"],
        "audit_passed": audit_feedback["data_valid"],
        "rebuild_reason": audit_feedback.get("reason"),
        "monte_carlo_complete": bool(monte_carlo_results.get("routes")),
    }
    logger.info("[REPORTING] System status: graph=%s audit_passed=%s mc=%s",
                status["graph_version"], status["audit_passed"],
                status["monte_carlo_complete"])
    return status

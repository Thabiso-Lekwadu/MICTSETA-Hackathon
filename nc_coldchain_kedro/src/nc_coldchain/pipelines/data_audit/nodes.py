"""data_audit nodes: first-class quality checks + a real audit->feedback signal.

The audit does NOT silently fix data; it measures it, writes machine-readable
results, and produces a feedback object that downstream graph nodes consult to
decide whether a rebuild / re-optimisation is actually required.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("nc_coldchain")

_PREV_META = Path("data/08_graph/optimized/metadata.json")


# --------------------------------------------------------------- table checks ---
def _audit_table(name: str, df: pd.DataFrame, ranges: dict, audit_params: dict) -> dict:
    z_thr = float(audit_params.get("outlier_z_threshold", 4.0))
    n = len(df)
    missing = {c: float(df[c].isna().mean()) for c in df.columns}
    dup = int(df.duplicated().sum())
    outliers, range_viol, dist = {}, {}, {}
    for c in df.select_dtypes(include="number").columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if not len(s):
            continue
        std = s.std(ddof=0)
        z = ((s - s.mean()).abs() / std) if std > 0 else pd.Series(0.0, index=s.index)
        outliers[c] = int((z > z_thr).sum())
        dist[c] = {"min": float(s.min()), "max": float(s.max()),
                   "mean": round(float(s.mean()), 4), "std": round(float(std), 4)}
        if c in ranges:
            lo, hi = ranges[c]["min"], ranges[c]["max"]
            range_viol[c] = int(((s < lo) | (s > hi)).sum())
    return {
        "table": name, "rows": n,
        "columns": list(df.columns),
        "missing_fraction": missing,
        "duplicate_rows": dup,
        "outliers": outliers,
        "range_violations": range_viol,
        "distributions": dist,
    }


def audit_tables(weather, road, sensors, drivers, graph_nodes, graph_edges,
                 audit_params: dict) -> dict:
    logger.info("[DATA AUDIT] Running quality checks on 4 data tables + graph inputs ...")
    ranges = {k: v for k, v in audit_params.items() if isinstance(v, dict) and "min" in v}
    results = {
        "tables": {
            "weather": _audit_table("weather", weather, ranges, audit_params),
            "road": _audit_table("road", road, ranges, audit_params),
            "sensors": _audit_table("sensors", sensors, ranges, audit_params),
            "drivers": _audit_table("drivers", drivers, ranges, audit_params),
        },
        "graph_inputs": {
            "node_count": int(len(graph_nodes)),
            "edge_count": int(len(graph_edges)),
            "town_count": int((graph_nodes["kind"] == "town").sum()),
            "mean_edge_weight": round(float(pd.to_numeric(
                graph_edges["travel_time_hr"], errors="coerce").mean()), 6),
            "negative_lengths": int((pd.to_numeric(
                graph_edges["length_km"], errors="coerce") < 0).sum()),
        },
    }
    # overall pass/fail
    max_missing = float(audit_params.get("max_missing_fraction", 0.30))
    dup_tol = int(audit_params.get("duplicate_tolerance", 0))
    issues = []
    for tname, t in results["tables"].items():
        for col, frac in t["missing_fraction"].items():
            if frac > max_missing:
                issues.append(f"{tname}.{col} missing {frac:.0%} > {max_missing:.0%}")
        if t["duplicate_rows"] > dup_tol:
            issues.append(f"{tname} has {t['duplicate_rows']} duplicate rows")
        for col, cnt in t["range_violations"].items():
            if cnt:
                issues.append(f"{tname}.{col} has {cnt} out-of-range values")
    results["quality_issues"] = issues
    results["passed"] = len(issues) == 0
    logger.info("[DATA AUDIT] Audit %s (%d issue(s))",
                "passed" if results["passed"] else "found issues", len(issues))
    return results


def generate_audit_report(audit_results: dict) -> pd.DataFrame:
    """Flatten audit results into a tabular, human-readable report."""
    rows = []
    for tname, t in audit_results["tables"].items():
        rows.append({
            "table": tname, "rows": t["rows"], "columns": len(t["columns"]),
            "duplicate_rows": t["duplicate_rows"],
            "total_outliers": int(sum(t["outliers"].values())),
            "total_range_violations": int(sum(t["range_violations"].values())),
            "max_missing_fraction": round(max(t["missing_fraction"].values() or [0.0]), 4),
        })
    gi = audit_results["graph_inputs"]
    rows.append({"table": "graph_inputs", "rows": gi["edge_count"],
                 "columns": gi["node_count"], "duplicate_rows": 0,
                 "total_outliers": 0, "total_range_violations": gi["negative_lengths"],
                 "max_missing_fraction": 0.0})
    return pd.DataFrame(rows)


def generate_audit_feedback(audit_results: dict, graph_nodes, graph_edges,
                            audit_params: dict) -> dict:
    """Compare current graph inputs to the previously published graph and decide
    whether a rebuild / re-optimisation is genuinely required. This is a REAL
    feedback signal driven by measured change, not an echo of the input."""
    gi = audit_results["graph_inputs"]
    prev = {}
    if _PREV_META.exists():
        try:
            prev = json.loads(_PREV_META.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    prev_nodes = int(prev.get("node_count", -1))
    prev_edges = int(prev.get("edge_count", -1))
    prev_meanw = float(prev.get("mean_edge_weight", -1.0))

    trig = audit_params.get("rebuild_triggers", {})
    node_delta = abs(gi["node_count"] - prev_nodes) if prev_nodes >= 0 else gi["node_count"]
    edge_delta = abs(gi["edge_count"] - prev_edges) if prev_edges >= 0 else gi["edge_count"]
    if prev_meanw > 0:
        weight_pct = abs(gi["mean_edge_weight"] - prev_meanw) / prev_meanw * 100.0
    else:
        weight_pct = 100.0

    first_build = not prev
    rebuild = (
        first_build
        or node_delta > int(trig.get("node_count_delta", 0))
        or edge_delta > int(trig.get("edge_count_delta", 0))
        or weight_pct > float(trig.get("mean_edge_weight_pct", 5.0))
    )
    changed_nodes = []
    if prev_nodes >= 0 and prev_nodes != gi["node_count"]:
        changed_nodes.append(f"node_count {prev_nodes} -> {gi['node_count']}")
    changed_edges = []
    if prev_edges >= 0 and prev_edges != gi["edge_count"]:
        changed_edges.append(f"edge_count {prev_edges} -> {gi['edge_count']}")

    feedback = {
        "data_valid": bool(audit_results["passed"]),
        "first_build": first_build,
        "graph_rebuild_required": bool(rebuild),
        "optimization_required": bool(rebuild),
        "reason": ("first build" if first_build else
                   ("graph-relevant change detected" if rebuild else "no material change")),
        "deltas": {"node_delta": node_delta, "edge_delta": edge_delta,
                   "mean_edge_weight_pct": round(weight_pct, 3)},
        "changed_nodes": changed_nodes,
        "changed_edges": changed_edges,
        "quality_issues": audit_results["quality_issues"],
        "previous": {"node_count": prev_nodes, "edge_count": prev_edges,
                     "mean_edge_weight": prev_meanw,
                     "graph_version": prev.get("graph_version")},
    }
    logger.info("[AUDIT FEEDBACK] rebuild_required=%s (%s)",
                feedback["graph_rebuild_required"], feedback["reason"])
    return feedback

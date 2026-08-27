"""FastAPI backend: serves the LATEST Kedro-generated artifacts.

Architectural rule (from the master spec): the backend contains NO data-processing
or graph-construction logic. It reads well-defined artifacts the Kedro pipeline
produced and exposes them over HTTP. If an artifact does not yet exist, the
endpoint returns 503 rather than fabricating data — the frontend must never show a
graph the pipeline has not produced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make 'shared' importable
from shared import paths  # noqa: E402

app = FastAPI(title="NC Cold-Chain Artifact API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def _read(path: Path, label: str):
    if not path.exists():
        raise HTTPException(status_code=503,
                            detail=f"{label} not available yet — run the pipeline first")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"failed to read {label}: {exc}")


@app.get("/health")
def health():
    if paths.SYSTEM_STATUS.exists():
        status = _read(paths.SYSTEM_STATUS, "system status")
        return {"status": "ok", **status}
    return {"status": "starting", "pipeline_complete": False,
            "graph_available": paths.OPTIMAL_GRAPH_JSON.exists()}


@app.get("/status")
def status():
    return _read(paths.SYSTEM_STATUS, "system status")


@app.get("/graph")
def graph():
    """The latest optimal graph as node-link JSON for the frontend map."""
    return _read(paths.OPTIMAL_GRAPH_JSON, "optimal graph")


@app.get("/graph/geojson")
def graph_geojson():
    return _read(paths.OPTIMAL_GRAPH_GEOJSON, "optimal graph geojson")


@app.get("/graph/metadata")
def graph_metadata():
    return _read(paths.OPTIMAL_GRAPH_METADATA, "graph metadata")


@app.get("/graph/metrics")
def graph_metrics():
    return _read(paths.GRAPH_METRICS, "graph metrics")


@app.get("/graph/candidates")
def graph_candidates():
    return _read(paths.CANDIDATE_SCORES, "candidate scores")


@app.get("/audit")
def audit():
    return {"results": _read(paths.AUDIT_RESULTS, "audit results"),
            "feedback": _read(paths.AUDIT_FEEDBACK, "audit feedback")}


@app.get("/monte-carlo")
def monte_carlo():
    return _read(paths.MONTE_CARLO_RESULTS, "monte carlo results")


@app.get("/reports")
def reports():
    return {
        "data_quality": _read(paths.DATA_QUALITY_REPORT, "data quality report"),
        "graph_change": _read(paths.GRAPH_CHANGE_REPORT, "graph change report"),
        "simulation": _read(paths.SIMULATION_REPORT, "simulation report"),
    }


@app.get("/reports/graph-change")
def report_graph_change():
    return _read(paths.GRAPH_CHANGE_REPORT, "graph change report")

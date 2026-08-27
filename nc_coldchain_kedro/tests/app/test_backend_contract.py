"""API-contract test for the backend. Skips cleanly when FastAPI is absent."""
import json
import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))


def _client(tmp_path, monkeypatch):
    # point the backend at a temp DATA_PATH with a minimal system_status + graph
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    (tmp_path / "09_reporting").mkdir(parents=True)
    (tmp_path / "08_graph" / "optimized").mkdir(parents=True)
    (tmp_path / "09_reporting" / "system_status.json").write_text(json.dumps(
        {"graph_version": "g-test", "graph_available": True,
         "audit_passed": True, "monte_carlo_complete": True}))
    (tmp_path / "08_graph" / "optimized" / "optimal_graph.json").write_text(json.dumps(
        {"nodes": [], "links": []}))
    # import after env is set so shared.paths picks up DATA_PATH
    for m in list(sys.modules):
        if m.startswith("shared") or m == "backend.main" or m.startswith("app.backend"):
            sys.modules.pop(m, None)
    from backend import main  # noqa: WPS433
    return TestClient(main.app)


def test_health_and_graph(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["graph_version"] == "g-test"
    g = client.get("/graph")
    assert g.status_code == 200
    assert "nodes" in g.json()


def test_missing_artifact_returns_503(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    # monte-carlo artifact was never written -> 503, not a fabricated result
    assert client.get("/monte-carlo").status_code == 503

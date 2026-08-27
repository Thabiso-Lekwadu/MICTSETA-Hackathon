"""End-to-end logic test: runs every pipeline's NODE functions in dependency
order, using the real conf/base/parameters.yml, WITHOUT the Kedro runtime.

This proves the connected data product produces a versioned optimal graph and a
Monte-Carlo result from synthetic data alone — the CRITICAL acceptance test, in
pure-Python form so it runs in CI without Docker.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _params():
    return yaml.safe_load((ROOT / "conf" / "base" / "parameters.yml").read_text())


def run_full_flow():
    import os
    os.chdir(ROOT)  # boundary geojson path in params is project-relative

    from nc_coldchain.pipelines.data_extraction import nodes as ex
    from nc_coldchain.pipelines.synthetic_data import nodes as sy
    from nc_coldchain.pipelines.preprocessing import nodes as pp
    from nc_coldchain.pipelines.data_audit import nodes as au
    from nc_coldchain.pipelines.graph import nodes as gr
    from nc_coldchain.pipelines.graph_optimization import nodes as go
    from nc_coldchain.pipelines.monte_carlo import nodes as mc
    from nc_coldchain.pipelines.reporting import nodes as rp

    P = _params()
    # This offline logic test has no OSM extract available, so it exercises the
    # SYNTHETIC dispatch path explicitly. Production runs use road_source='osm'.
    P["network"]["road_source"] = "synthetic"

    # --- data_extraction
    towns = ex.extract_town_registry(P["network"])
    boundary = ex.extract_boundary(P["network"])
    drivers = ex.extract_drivers_reports(P["network"], P["reproducibility"])

    # --- synthetic_data
    weather, road, sensors = sy.generate_synthetic_streams(
        P["network"], P["runtime"], P["reproducibility"])

    # --- preprocessing
    w2, r2, s2, d2 = pp.standardize_schema(weather, road, sensors, drivers)
    w3, r3, s3, d3 = pp.clean_data(w2, r2, s2, d2, P["preprocessing"])
    osm_ready = ex.extract_osm_roads(P["network"])  # download_osm ignored: synthetic mode
    bundle, edges, nodes = pp.assemble_graph_inputs(
        towns, osm_ready, r3, P["network"], P["graph"])
    live_weather = ex.fetch_live_weather(P["network"], P["runtime"])  # disabled -> offline
    mc_params = pp.prepare_model_data(s3, live_weather, P["monte_carlo"])

    # --- data_audit
    audit_results = au.audit_tables(w3, r3, s3, d3, nodes, edges, P["audit"])
    au.generate_audit_report(audit_results)
    feedback = au.generate_audit_feedback(audit_results, nodes, edges, P["audit"])

    # --- graph (validate + measure the real bundle)
    validation = gr.validate_graph(bundle, P["graph"])
    metrics = gr.calculate_graph_metrics(bundle)

    # --- graph_optimization
    scores = go.evaluate_candidates(bundle, validation, P["optimization"], P["graph"])
    nc_road_graph, metadata = go.build_optimal_graph(
        bundle, scores, P["graph"], feedback, P["reproducibility"])
    _, geojson = go.publish_graph(nc_road_graph, P["optimization"])

    # --- monte_carlo
    prepared = mc.prepare_monte_carlo_inputs(nc_road_graph, mc_params)
    mc_results = mc.run_monte_carlo(prepared, P["reproducibility"])

    # --- reporting
    dq = rp.build_data_quality_report(audit_results, feedback)
    change = rp.build_graph_change_report(metadata, feedback, metrics)
    sim = rp.build_simulation_report(mc_results, metadata)
    status = rp.build_system_status(metadata, metrics, feedback, mc_results)

    return dict(metrics=metrics, validation=validation, metadata=metadata,
                geojson=geojson, mc_results=mc_results, status=status,
                feedback=feedback, dq=dq, change=change, sim=sim, scores=scores)


def test_end_to_end():
    out = run_full_flow()
    assert out["validation"]["valid"] is True
    assert out["metrics"]["is_connected"] is True
    assert out["metrics"]["town_count"] == 7
    assert out["metadata"]["graph_version"].startswith("g-")
    assert out["metadata"]["edge_count"] == out["metrics"]["edge_count"]
    assert out["mc_results"]["routes"], "monte carlo produced no routes"
    for r in out["mc_results"]["routes"].values():
        assert 0.0 <= r["prob_breach"] <= 1.0
    assert out["status"]["graph_available"] is True
    assert out["status"]["monte_carlo_complete"] is True


if __name__ == "__main__":
    import json
    o = run_full_flow()
    print("VALIDATION:", o["validation"])
    print("METRICS:", json.dumps(o["metrics"]))
    print("METADATA:", json.dumps(o["metadata"]))
    print("SCORES:", json.dumps(o["scores"]["candidates"]))
    print("MC:", json.dumps(o["mc_results"]["routes"]))
    print("STATUS:", json.dumps(o["status"]))
    print("FEEDBACK:", json.dumps({k: o["feedback"][k] for k in
          ("first_build", "graph_rebuild_required", "reason", "data_valid")}))
    print("OK — end-to-end logic flow passed")

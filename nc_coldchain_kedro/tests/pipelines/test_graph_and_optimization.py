"""Unit tests for the bundle-based graph validation, metrics and optimisation."""
from nc_coldchain.pipelines.preprocessing import graph_build as gb
from nc_coldchain.pipelines.graph import nodes as gr
from nc_coldchain.pipelines.graph_optimization import nodes as go


def _bundle(params):
    towns = {k: v for k, v in list(params["network"]["towns"].items())[:4]}
    import pandas as pd
    return gb.build_synthetic_bundle(towns, params["network"], params["graph"],
                                     pd.DataFrame({"iri": [2.0]}))


def test_validate_and_metrics(params):
    bundle = _bundle(params)
    v = gr.validate_graph(bundle, params["graph"])
    assert v["valid"] and v["connected"]
    m = gr.calculate_graph_metrics(bundle)
    assert m["is_connected"] and m["town_count"] == 4
    # every town is a real node in the graph
    for n in bundle["town_nodes"].values():
        assert n in bundle["G_main"]


def test_optimization_selects_lowest_and_bakes_weights(params):
    bundle = _bundle(params)
    opt = {"candidate_weights": [{"thermal": 0.7, "mech": 0.3},
                                 {"thermal": 0.3, "mech": 0.7}],
           "evaluation_pairs": [list(bundle["town_nodes"])[:2]],
           "objective": "min_mean_spoilage_cost"}
    scores = go.evaluate_candidates(bundle, {"valid": True}, opt, params["graph"])
    assert scores["candidates"][0]["objective_score"] <= scores["candidates"][1]["objective_score"]
    nc_graph, meta = go.build_optimal_graph(
        bundle, scores, params["graph"],
        {"reason": "test", "deltas": {}, "previous": {}}, {"deterministic": True})
    # the app artifact has the exact keys live_backend.py reads
    assert {"G_main", "main_nodes", "cluster_coord"}.issubset(nc_graph.keys())
    assert meta["node_count"] == nc_graph["G_main"].number_of_nodes()
    # optimal weights are baked into spoilage_cost_edge
    for _, _, d in nc_graph["G_main"].edges(data=True):
        assert "spoilage_cost_edge" in d
        break

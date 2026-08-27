"""Unit tests for preprocessing nodes."""
import pandas as pd

from nc_coldchain.pipelines.preprocessing import nodes as pp


def test_clean_drops_duplicates(params):
    df = pd.DataFrame({
        "hour_index": [0, 0, 1],
        "timestamp_iso": ["t0", "t0", "t1"],
        "ambient_temp_c": [30.0, 30.0, None],
    })
    out = pp._clean(df, params["preprocessing"], ["hour_index", "timestamp_iso"])
    assert len(out) == 2                       # one duplicate removed
    assert out["ambient_temp_c"].isna().sum() == 0  # missing imputed


def test_build_graph_nodes_seven_towns(params):
    reg = {"towns": params["network"]["towns"]}
    boundary = {"boundary": []}
    nodes = pp.build_graph_nodes(reg, boundary, params["network"])
    assert len(nodes) == 7
    assert set(nodes["kind"]) == {"town"}


def test_build_graph_edges_connected(params):
    reg = {"towns": params["network"]["towns"]}
    nodes = pp.build_graph_nodes(reg, {"boundary": []}, params["network"])
    road = pd.DataFrame({"iri": [2.0, 3.0]})
    edges, aug = pp.build_graph_edges(nodes, road, params["network"], params["graph"])
    assert len(edges) > 0
    assert (aug["kind"] == "waypoint").sum() > 0
    # every edge endpoint exists in the augmented node table
    ids = set(aug["node_id"].astype(str))
    assert set(edges["u"]).issubset(ids) and set(edges["v"]).issubset(ids)

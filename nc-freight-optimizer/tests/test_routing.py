"""Unit tests for routing.py -- specifically the fallback-hierarchy weight
functions, since that's the mechanism the whole "driver report overrides the
baseline" demo depends on. A bug here would silently ignore reports."""

import networkx as nx
import pytest

from nc_freight_optimizer.routing import (
    NodeSpatialIndex,
    baseline_weight,
    compare_routes,
    effective_weight,
    initialize_baseline_impedance,
)


def _make_test_graph() -> nx.Graph:
    """A -> B -> C (direct, short) and A -> D -> C (longer detour)."""
    G = nx.Graph()
    G.add_edge("A", "B", length_km=10, travel_time=0.1, roughness=1.0)
    G.add_edge("B", "C", length_km=10, travel_time=0.1, roughness=1.0)
    G.add_edge("A", "D", length_km=15, travel_time=0.2, roughness=1.2)
    G.add_edge("D", "C", length_km=15, travel_time=0.2, roughness=1.2)
    return G


class TestEffectiveWeight:
    def test_uses_baseline_when_no_override_present(self):
        G = _make_test_graph()
        initialize_baseline_impedance(G)
        weight = effective_weight("A", "B", G["A"]["B"])
        assert weight == pytest.approx(0.1 * 1.0)

    def test_override_takes_priority_over_baseline(self):
        G = _make_test_graph()
        initialize_baseline_impedance(G)
        G["A"]["B"]["override"] = {"spoilage_cost": 999.0}
        assert effective_weight("A", "B", G["A"]["B"]) == 999.0

    def test_baseline_weight_ignores_override(self):
        """This is what makes detour detection possible: baseline_weight must
        see through an active override to compute what the route WOULD have
        been without it."""
        G = _make_test_graph()
        initialize_baseline_impedance(G)
        original = G["A"]["B"]["spoilage_cost"]
        G["A"]["B"]["override"] = {"spoilage_cost": 999.0}
        assert baseline_weight("A", "B", G["A"]["B"]) == original


class TestCompareRoutes:
    def test_no_override_standard_and_optimized_agree_on_short_uniform_graph(self):
        G = _make_test_graph()
        initialize_baseline_impedance(G)
        standard, optimized = compare_routes(G, "A", "C", border_nodes={})
        assert standard["path"] == ["A", "B", "C"]
        assert optimized["path"] == ["A", "B", "C"]

    def test_severe_override_on_direct_path_forces_detour(self):
        G = _make_test_graph()
        initialize_baseline_impedance(G)
        # Report the direct A-B segment as impassable: spoilage cost spikes.
        G["A"]["B"]["override"] = {"spoilage_cost": 1000.0, "base_time_mins": 1000.0}
        _, optimized = compare_routes(G, "A", "C", border_nodes={})
        assert optimized["path"] == ["A", "D", "C"], (
            "Expected the optimizer to route around the reported segment"
        )

    def test_border_node_penalty_is_added_to_total_time(self):
        G = _make_test_graph()
        initialize_baseline_impedance(G)
        standard_no_border, _ = compare_routes(G, "A", "C", border_nodes={})
        standard_with_border, _ = compare_routes(G, "A", "C", border_nodes={"B": 5.0})
        assert standard_with_border["total_time_hr"] > standard_no_border["total_time_hr"]


class TestNodeSpatialIndex:
    def test_snaps_to_nearest_node(self):
        G = nx.Graph()
        G.add_node(1)
        G.add_node(2)
        cluster_coord = {1: (17.0, -29.0), 2: (20.0, -29.0)}
        index = NodeSpatialIndex(G, cluster_coord)
        node_id, distance_km = index.snap(latitude=-29.0, longitude=17.1)
        assert node_id == 1
        assert distance_km < 15  # well within a plausible snap distance for this offset

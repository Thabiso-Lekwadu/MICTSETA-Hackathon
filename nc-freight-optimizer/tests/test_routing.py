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

    def test_prefers_well_connected_node_over_closer_dead_end(self):
        """Regression test for a real bug: De Aar's coordinate snapped to a
        degree-1 dead-end stub 270m away instead of a degree-2 junction 420m
        away, forcing a ~400km detour to escape the dead end. The nearest node
        overall should never win over a well-connected node just slightly
        farther away."""
        G = nx.Graph()
        G.add_edge("dead_end_stub", "isolated_tail", length_km=0.5, travel_time=0.01, roughness=1.0)
        G.add_edge("junction", "highway_a", length_km=10, travel_time=0.1, roughness=1.0)
        G.add_edge("junction", "highway_b", length_km=10, travel_time=0.1, roughness=1.0)

        cluster_coord = {
            "dead_end_stub": (24.0, -30.65),   # 0.0 km from query point -- closest overall
            "isolated_tail": (24.0, -30.66),
            "junction": (24.005, -30.65),       # ~0.4km away, but degree 2 (well-connected)
            "highway_a": (24.05, -30.65),
            "highway_b": (23.95, -30.65),
        }

        index = NodeSpatialIndex(G, cluster_coord, min_degree=2, degree_search_radius_km=3.0)
        node_id, _ = index.snap(latitude=-30.65, longitude=24.0)
        assert node_id == "junction", (
            f"Expected snap to prefer the well-connected junction over the closer "
            f"dead-end stub, got '{node_id}'"
        )

    def test_falls_back_to_nearest_overall_when_nothing_well_connected_is_near(self):
        G = nx.Graph()
        G.add_node("only_option")  # degree 0, no well-connected candidates exist at all
        cluster_coord = {"only_option": (20.0, -29.0)}
        index = NodeSpatialIndex(G, cluster_coord, min_degree=2)
        node_id, _ = index.snap(latitude=-29.0, longitude=20.0)
        assert node_id == "only_option"

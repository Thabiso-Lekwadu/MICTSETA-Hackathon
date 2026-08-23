from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline as modular_pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return modular_pipeline([
        node(
            func=nodes.run_route_sensitivity_scan,
            inputs=[
                "nc_road_graph_bundle",
                "params:routing.towns",
                "params:routing.border_posts",
                "params:routing.border_delay_hr",
            ],
            outputs="route_sensitivity_scan",
            name="run_route_sensitivity_scan_node",
        ),
    ])

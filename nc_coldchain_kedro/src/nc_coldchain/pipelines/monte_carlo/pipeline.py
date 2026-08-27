"""monte_carlo pipeline: optimal graph + MC params -> simulation outputs."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_monte_carlo_inputs, run_monte_carlo


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            prepare_monte_carlo_inputs,
            ["nc_road_graph", "model_input_mc_params"],
            "monte_carlo_prepared",
            name="monte_carlo_prepare_inputs",
        ),
        node(
            run_monte_carlo,
            ["monte_carlo_prepared", "params:reproducibility"],
            "monte_carlo_results",
            name="monte_carlo_run",
        ),
    ])

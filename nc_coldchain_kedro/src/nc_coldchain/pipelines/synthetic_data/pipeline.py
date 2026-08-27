"""synthetic_data pipeline: generator -> data/01_raw/synthetic."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_synthetic_streams


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            generate_synthetic_streams,
            ["params:network", "params:runtime", "params:reproducibility"],
            ["raw_synthetic_weather", "raw_synthetic_road", "raw_synthetic_sensors"],
            name="generate_synthetic_streams",
        ),
    ])

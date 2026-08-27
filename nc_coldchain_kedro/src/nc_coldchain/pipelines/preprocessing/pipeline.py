"""preprocessing pipeline: raw -> intermediate -> primary -> feature/model_input."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    assemble_graph_inputs,
    clean_data,
    prepare_model_data,
    standardize_schema,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            standardize_schema,
            ["raw_synthetic_weather", "raw_synthetic_road", "raw_synthetic_sensors",
             "raw_drivers_reports"],
            ["int_weather", "int_road", "int_sensors", "int_drivers"],
            name="preprocess_standardize_schema",
        ),
        node(
            clean_data,
            ["int_weather", "int_road", "int_sensors", "int_drivers",
             "params:preprocessing"],
            ["prm_weather", "prm_road", "prm_sensors", "prm_drivers"],
            name="preprocess_clean_data",
        ),
        node(
            assemble_graph_inputs,
            ["raw_town_registry", "raw_osm_ready", "prm_road",
             "params:network", "params:graph"],
            ["road_graph_bundle", "feat_graph_edges", "feat_graph_nodes"],
            name="preprocess_assemble_graph_inputs",
        ),
        node(
            prepare_model_data,
            ["prm_sensors", "raw_live_weather", "params:monte_carlo"],
            "model_input_mc_params",
            name="preprocess_prepare_model_data",
        ),
    ])

"""synthetic_data nodes: generate Streams A/B/C via the real generator, land in raw.

Synthetic data is a first-class citizen of the lifecycle: it is produced here,
written to data/01_raw/synthetic, and then flows through exactly the same
preprocessing -> audit -> graph -> optimisation -> monte_carlo -> reporting path
as any real data would.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger("nc_coldchain")


def _start_dt(repro_params: dict) -> datetime:
    """Reproducible anchor when deterministic, wall-clock otherwise."""
    if repro_params.get("deterministic", True):
        # fixed anchor so a dev run is byte-identical across machines
        return datetime(2026, 8, 26, 6, 0, 0, tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def generate_synthetic_streams(network_params: dict, runtime_params: dict,
                               repro_params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (weather, road, sensors) DataFrames — Streams A, B and C."""
    from nc_coldchain.legacy import vrp_simulation_generator as gen

    hours = int(runtime_params.get("n_synthetic_hours", 72))
    # anchor the weather matrix on the first town so single-point and matrix agree
    first_town = next(iter(network_params["towns"].values()))
    anchor = (float(first_town[0]), float(first_town[1]))
    start = _start_dt(repro_params)

    logger.info("[SYNTHETIC DATA] Generating %d hours of Streams A/B/C ...", hours)
    weather_rows = gen.generate_weather_forecast_matrix(start, hours=hours, anchor_lonlat=anchor)
    road_rows = gen.derive_road_conditions(weather_rows)
    sensor_rows = gen.generate_cold_chain_stream(weather_rows, road_rows)

    weather_df = pd.DataFrame([r.__dict__ for r in weather_rows])
    road_df = pd.DataFrame([r.__dict__ for r in road_rows])
    sensor_df = pd.DataFrame([r.__dict__ for r in sensor_rows])
    logger.info("[SYNTHETIC DATA] weather=%d road=%d sensors=%d rows",
                len(weather_df), len(road_df), len(sensor_df))
    return weather_df, road_df, sensor_df

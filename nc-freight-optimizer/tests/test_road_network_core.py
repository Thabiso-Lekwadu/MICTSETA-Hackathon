"""Unit tests for road_network_core.py -- the tiered maxspeed imputation is the
piece most worth protecting against silent regression, since a bug there
produces plausible-looking but wrong travel times rather than a crash."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from nc_freight_optimizer.road_network_core import (
    audit_structure,
    impute_maxspeed,
    validate_raw_roads,
)

DEFAULT_SPEED_KMH = {
    "trunk": 100, "primary": 100, "secondary": 80, "track": 30,
    "track_grade1": 40, "track_grade5": 15,
}


def _make_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    for row in rows:
        row.setdefault("geometry", LineString([(0, 0), (0, 1)]))
        row.setdefault("ref", None)
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


class TestImputeMaxspeed:
    def test_zero_is_treated_as_missing_not_zero_kmh(self):
        gdf = _make_gdf([{"fclass": "trunk", "maxspeed": 0}])
        result = impute_maxspeed(gdf, DEFAULT_SPEED_KMH)
        assert result["imputed_speed_kmh"].iloc[0] > 0
        assert result["speed_source"].iloc[0] != "observed"

    def test_observed_value_is_trusted_as_is(self):
        gdf = _make_gdf([{"fclass": "trunk", "maxspeed": 120}])
        result = impute_maxspeed(gdf, DEFAULT_SPEED_KMH)
        assert result["imputed_speed_kmh"].iloc[0] == 120
        assert result["speed_source"].iloc[0] == "observed"

    def test_falls_back_to_domain_default_with_no_real_signal(self):
        gdf = _make_gdf([{"fclass": "track_grade5", "maxspeed": 0}])
        result = impute_maxspeed(gdf, DEFAULT_SPEED_KMH)
        assert result["imputed_speed_kmh"].iloc[0] == DEFAULT_SPEED_KMH["track_grade5"]
        assert result["speed_source"].iloc[0] == "domain_default"

    def test_fclass_median_used_when_enough_real_samples(self):
        rows = [{"fclass": "secondary", "maxspeed": 80} for _ in range(35)]
        rows.append({"fclass": "secondary", "maxspeed": 0})  # the one to impute
        gdf = _make_gdf(rows)
        result = impute_maxspeed(gdf, DEFAULT_SPEED_KMH, min_fclass_n=30)
        imputed_row = result[result["maxspeed"] == 0].iloc[0]
        assert imputed_row["imputed_speed_kmh"] == 80
        assert imputed_row["speed_source"] == "fclass_median"

    def test_speed_floor_prevents_near_zero_travel_speed(self):
        gdf = _make_gdf([{"fclass": "trunk", "maxspeed": 2}])
        result = impute_maxspeed(gdf, DEFAULT_SPEED_KMH, speed_floor_kmh=5.0)
        assert result["imputed_speed_kmh"].iloc[0] >= 5.0

    def test_track_grades_do_not_collapse_to_the_same_speed(self):
        # Regression test for the bug found during development: grade1-5 all
        # borrowing from a combined "track" real-data pool erased the whole
        # point of grading. With no real samples for any grade, each must
        # fall through independently to its own domain default.
        gdf = _make_gdf([
            {"fclass": "track_grade1", "maxspeed": 0},
            {"fclass": "track_grade5", "maxspeed": 0},
        ])
        result = impute_maxspeed(gdf, DEFAULT_SPEED_KMH)
        grade1_speed = result[result["fclass"] == "track_grade1"]["imputed_speed_kmh"].iloc[0]
        grade5_speed = result[result["fclass"] == "track_grade5"]["imputed_speed_kmh"].iloc[0]
        assert grade1_speed > grade5_speed


class TestValidateRawRoads:
    def test_passes_through_valid_data_unchanged(self):
        gdf = _make_gdf([{"fclass": "trunk", "maxspeed": 100, "osm_id": "1"}])
        bbox = {"min_lon": -1, "max_lon": 1, "min_lat": -1, "max_lat": 1}
        result = validate_raw_roads(gdf, bbox)
        assert len(result) == 1

    def test_raises_on_empty_input(self):
        gdf = gpd.GeoDataFrame({"fclass": [], "maxspeed": [], "osm_id": [], "geometry": []}, crs="EPSG:4326")
        bbox = {"min_lon": -1, "max_lon": 1, "min_lat": -1, "max_lat": 1}
        with pytest.raises(ValueError, match="empty"):
            validate_raw_roads(gdf, bbox)

    def test_raises_when_bbox_is_wildly_wrong(self):
        gdf = _make_gdf([{"fclass": "trunk", "maxspeed": 100, "osm_id": "1"}])
        # geometry sits at (0,0)-(0,1); this expected bbox is nowhere near it
        bbox = {"min_lon": 100, "max_lon": 101, "min_lat": 40, "max_lat": 41}
        with pytest.raises(ValueError, match="extent"):
            validate_raw_roads(gdf, bbox, bbox_tolerance_deg=0.5)

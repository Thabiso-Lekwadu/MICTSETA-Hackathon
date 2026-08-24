"""
Integration test: the full pipeline runs end to end via a real KedroSession,
exactly as `kedro run` does, and produces the graph bundle live_backend.py
depends on.

The one thing this test mocks is the actual Geofabrik network call: hitting a
real external service on every test run would make the suite slow, flaky, and
dependent on network access being available wherever tests run (a CI runner
with no egress, for instance). Everything downstream of that single boundary --
validation, cleaning, imputation, graph construction, connectivity extraction,
bundling, and the reporting pipeline -- runs for real, unmocked, exactly as it
would in production. `kedro run` itself (not this test) is what exercises the
real download.
"""

from pathlib import Path

import geopandas as gpd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from shapely.geometry import LineString


def _synthetic_raw_roads() -> gpd.GeoDataFrame:
    """A tiny but structurally valid stand-in for the real Geofabrik extract --
    same columns, same fclass vocabulary, small enough to build a graph from in
    milliseconds. Deliberately includes a few connected roads spanning a small
    area within the expected Northern Cape bounding box."""
    rows = [
        {"osm_id": "1", "fclass": "trunk", "maxspeed": 100, "ref": "N12", "name": "Test Trunk",
         "geometry": LineString([(20.0, -28.0), (20.1, -28.05), (20.2, -28.1)])},
        {"osm_id": "2", "fclass": "primary", "maxspeed": 0, "ref": "R31", "name": None,
         "geometry": LineString([(20.2, -28.1), (20.3, -28.15)])},
        {"osm_id": "3", "fclass": "track", "maxspeed": 0, "ref": None, "name": None,
         "geometry": LineString([(20.3, -28.15), (20.35, -28.2)])},
        {"osm_id": "4", "fclass": "secondary", "maxspeed": 80, "ref": None, "name": None,
         "geometry": LineString([(20.0, -28.0), (19.9, -28.0)])},
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


class TestKedroRun:
    def test_kedro_run_produces_graph_bundle(self, mocker):
        mocker.patch(
            "nc_freight_optimizer.pipelines.road_network.nodes.download_and_extract_raw_roads",
            return_value=_synthetic_raw_roads(),
        )

        project_path = Path.cwd()
        bootstrap_project(project_path)

        with KedroSession.create(project_path=project_path) as session:
            session.run()

        bundle_path = project_path / "data" / "03_primary" / "nc_road_graph.pkl"
        assert bundle_path.exists(), (
            "Pipeline ran but did not produce data/03_primary/nc_road_graph.pkl "
            "-- live_backend.py would fail to start."
        )

        report_path = project_path / "data" / "08_reporting" / "route_sensitivity_scan.csv"
        assert report_path.exists(), "route_reporting pipeline did not produce its CSV output."

    def test_extract_raw_roads_node_delegates_to_data_ingestion(self):
        """Confirms the pipeline node is a thin wrapper around the real download
        logic in data_ingestion.py, not a reimplementation -- so the mock in the
        test above (which patches the imported reference inside nodes.py) is
        actually intercepting the real call path used in production."""
        import inspect

        from nc_freight_optimizer.pipelines.road_network import nodes

        assert callable(nodes.extract_raw_roads)
        source = inspect.getsource(nodes.extract_raw_roads)
        assert "requests.get" not in source, (
            "extract_raw_roads should delegate to data_ingestion.download_and_extract_raw_roads "
            "rather than calling requests directly"
        )
        assert "download_and_extract_raw_roads" in source

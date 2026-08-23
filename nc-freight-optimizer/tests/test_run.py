"""
Integration test: the full pipeline runs end to end via a real KedroSession,
exactly as `kedro run` does, and produces the graph bundle live_backend.py
depends on. This replaces the scaffold's default test, which only checked that
an *empty* project fails to run -- not applicable once real pipelines exist.
"""

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


class TestKedroRun:
    def test_kedro_run_produces_graph_bundle(self):
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

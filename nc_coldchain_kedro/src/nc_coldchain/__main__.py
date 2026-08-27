"""Entry point so `python -m nc_coldchain` runs the Kedro project."""
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession


def main() -> None:
    package_name = "nc_coldchain"
    configure_project(package_name)
    project_path = Path(__file__).resolve().parents[2]
    with KedroSession.create(project_path=project_path) as session:
        session.run()


if __name__ == "__main__":
    main()

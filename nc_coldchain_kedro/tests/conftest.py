"""Shared pytest fixtures."""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def params():
    return yaml.safe_load((ROOT / "conf" / "base" / "parameters.yml").read_text())


@pytest.fixture(scope="session")
def project_root():
    return ROOT

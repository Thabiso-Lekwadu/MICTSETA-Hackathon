"""Vendored domain modules. These use FLAT imports of each other
(e.g. `import nc_boundary` inside nc_road_network_improved), so add this folder
to sys.path at package-import time — otherwise nc_boundary loads as None and the
Northern Cape province clip is silently skipped (routes can cross the border)."""
import os
import sys

_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
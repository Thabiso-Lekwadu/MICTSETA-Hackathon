"""Kedro hooks: readable stage logging + data-lineage / graph-version recording.

These hooks give the operator the "single connected data product" experience the
architecture demands: every node emits a human-readable [STAGE] line into the
Docker logs, and every dataset save is appended to a lineage log so the path from
raw source to optimal graph is traceable without a separate lineage system.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from kedro.framework.hooks import hook_impl

logger = logging.getLogger("nc_coldchain")

_PLACEHOLDERS = {"", ".........", "REPLACE_IN_conf_local_credentials_yml",
                 "REPLACE_WITH_YOUR_KEY_OR_SET_IN_conf_local",
                 "paste-your-openweathermap-key-here",
                 "your-real-key", "REPLACE_ME"}


class CredentialsToEnvHook:
    """Inject the OpenWeatherMap key from Kedro credentials into the environment
    so weather_engine.resolve_owm_api_key() uses it. The key is read from
    conf/local/credentials.yml (git-ignored), never logged or echoed."""

    @hook_impl
    def after_context_created(self, context: Any) -> None:
        try:
            creds = context.config_loader["credentials"]
        except Exception:
            return
        key = ((creds or {}).get("openweathermap", {}) or {}).get("API_KEY")
        if key and str(key).strip() not in _PLACEHOLDERS:
            # weather_engine accepts API_KEY / OPENWEATHERMAP_API_KEY / OWM_API_KEY
            os.environ.setdefault("OPENWEATHERMAP_API_KEY", str(key).strip())
            os.environ.setdefault("API_KEY", str(key).strip())
            logger.info("[WEATHER] OpenWeatherMap key loaded from credentials "
                        "(provider=openweathermap).")
        else:
            logger.info("[WEATHER] No real OpenWeatherMap key configured — "
                        "weather falls back to Open-Meteo.")

# Map pipeline/node name fragments to the friendly stage banners in the spec.
_STAGE_BANNERS = {
    "extract": "DATA EXTRACTION",
    "synthetic": "SYNTHETIC DATA",
    "preprocess": "PREPROCESSING",
    "clean": "PREPROCESSING",
    "audit": "DATA AUDIT",
    "feedback": "AUDIT FEEDBACK",
    "graph_construct": "GRAPH",
    "construct_graph": "GRAPH",
    "optimi": "GRAPH OPTIMIZATION",
    "monte": "MONTE CARLO",
    "report": "REPORTING",
}

_LINEAGE_PATH = Path("data/09_reporting/data_lineage.log")


def _banner_for(node_name: str) -> str:
    low = node_name.lower()
    for frag, banner in _STAGE_BANNERS.items():
        if frag in low:
            return banner
    return "PIPELINE"


class PipelineTimerHook:
    """Emit readable per-node banners and timings."""

    def __init__(self) -> None:
        self._t0: dict[str, float] = {}

    @hook_impl
    def before_node_run(self, node: Any) -> None:
        self._t0[node.name] = time.time()
        logger.info("[%s] %s ...", _banner_for(node.name), node.name)

    @hook_impl
    def after_node_run(self, node: Any, outputs: dict[str, Any]) -> None:
        dt = time.time() - self._t0.get(node.name, time.time())
        produced = ", ".join(outputs.keys()) if outputs else "-"
        logger.info("[%s] %s done in %.2fs -> %s", _banner_for(node.name), node.name, dt, produced)


class LineageHook:
    """Append every dataset save to a lineage log and surface graph versions."""

    @hook_impl
    def after_dataset_saved(self, dataset_name: str, data: Any) -> None:
        try:
            _LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            record = {"dataset": dataset_name, "ts": time.time()}
            if dataset_name == "optimal_graph_metadata" and isinstance(data, dict):
                record["graph_version"] = data.get("graph_version")
                record["objective_score"] = data.get("objective_score")
                logger.info(
                    "[GRAPH OPTIMIZATION] Optimal graph %s (score=%.4f, nodes=%s, edges=%s)",
                    data.get("graph_version"),
                    float(data.get("objective_score", 0.0)),
                    data.get("node_count"),
                    data.get("edge_count"),
                )
            with _LINEAGE_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:  # lineage is best-effort, never fail a run over it
            logger.debug("lineage record skipped for %s", dataset_name, exc_info=True)

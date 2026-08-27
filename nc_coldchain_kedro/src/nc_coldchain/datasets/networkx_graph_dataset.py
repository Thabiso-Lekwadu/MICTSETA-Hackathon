"""Custom Kedro dataset for serialising a NetworkX graph as node-link JSON.

Graph artifacts do not fit the built-in tabular dataset types, so we implement a
small AbstractDataset. Node-link JSON is chosen (over pickle) for the *published*
graph because it is language-neutral, human-inspectable, and directly consumable
by the frontend map layer, satisfying the requirement that the frontend visualise
the latest optimal graph from a well-defined artifact rather than a static file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from kedro.io import AbstractDataset


class NetworkXGraphDataset(AbstractDataset):
    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def _load(self) -> nx.Graph:
        with self._filepath.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return nx.node_link_graph(data, edges="links")

    def _save(self, graph: nx.Graph) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(graph, edges="links")
        with self._filepath.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, default=str)

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath), "format": "node-link-json"}

    def _exists(self) -> bool:
        return self._filepath.exists()

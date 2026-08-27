"""Pipeline registry: assemble the end-to-end connected data product.

Dependency order (enforced by Kedro dataset dependencies, not manual ordering):

    data_extraction ─┐
                     ├─> preprocessing ─> data_audit ─> graph ─> graph_optimization ─┐
    synthetic_data ──┘                                                               │
                                                              ┌──────────────────────┤
                                                              ▼                       ▼
                                                         monte_carlo             reporting
"""
from __future__ import annotations

from kedro.pipeline import Pipeline

from nc_coldchain.pipelines import (
    data_audit,
    data_extraction,
    graph,
    graph_optimization,
    monte_carlo,
    preprocessing,
    reporting,
    synthetic_data,
)


def register_pipelines() -> dict[str, Pipeline]:
    p_extract = data_extraction.create_pipeline()
    p_synth = synthetic_data.create_pipeline()
    p_prep = preprocessing.create_pipeline()
    p_audit = data_audit.create_pipeline()
    p_graph = graph.create_pipeline()
    p_opt = graph_optimization.create_pipeline()
    p_mc = monte_carlo.create_pipeline()
    p_report = reporting.create_pipeline()

    full = (p_extract + p_synth + p_prep + p_audit + p_graph + p_opt + p_mc + p_report)

    return {
        "data_extraction": p_extract,
        "synthetic_data": p_synth,
        "preprocessing": p_prep,
        "data_audit": p_audit,
        "graph": p_graph,
        "graph_optimization": p_opt,
        "monte_carlo": p_mc,
        "reporting": p_report,
        # graph-relevant rebuild slice: everything from audit onward
        "rebuild": (p_audit + p_graph + p_opt + p_mc + p_report),
        "__default__": full,
    }

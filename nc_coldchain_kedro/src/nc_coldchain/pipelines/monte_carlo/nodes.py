"""monte_carlo: stochastic spoilage risk over the OPTIMAL real graph's routes.

Reuses the project's validated pure thermodynamic functions
(_simulate_peak_cargo_temp, _spoilage_loss_fraction) so the Kedro Monte-Carlo and
the live app share one risk model.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np

logger = logging.getLogger("nc_coldchain")


def _peak_and_loss():
    from nc_coldchain.legacy.vrp_simulation_generator import (
        _simulate_peak_cargo_temp, _spoilage_loss_fraction,
    )
    return _simulate_peak_cargo_temp, _spoilage_loss_fraction


def prepare_monte_carlo_inputs(nc_road_graph: dict, mc_params: dict) -> dict:
    """Resolve each OD pair to a route on the optimal real graph + its drive hours."""
    G = nc_road_graph["G_main"]
    town_nodes = nc_road_graph["town_nodes"]
    pairs = mc_params.get("evaluation_pairs", [])
    routes = {}
    for a, b in pairs:
        na, nb = town_nodes.get(a), town_nodes.get(b)
        if na is None or nb is None:
            logger.warning("[MONTE CARLO] town missing for %s->%s", a, b)
            continue
        try:
            path = nx.shortest_path(G, na, nb, weight="spoilage_cost_edge")
            cost = nx.shortest_path_length(G, na, nb, weight="spoilage_cost_edge")
            drive_hours = sum(G[u][v].get("travel_time", G[u][v].get("travel_time_hr", 0.0))
                              for u, v in zip(path[:-1], path[1:]))
            routes[f"{a}->{b}"] = {"n_hops": len(path) - 1, "spoilage_cost": round(cost, 4),
                                   "drive_hours": round(float(drive_hours), 3)}
        except Exception as exc:
            logger.warning("[MONTE CARLO] cannot prepare %s->%s: %s", a, b, exc)
    logger.info("[MONTE CARLO] Prepared %d route(s) for simulation", len(routes))
    return {"routes": routes, "params": mc_params}


def run_monte_carlo(mc_inputs: dict, repro_params: dict) -> dict:
    peak_fn, loss_fn = _peak_and_loss()
    p = mc_inputs["params"]
    trials = int(p.get("trials", 1000))
    ambient = float(p.get("ambient_temp_c", 32.0))
    breach_c = float(p.get("breach_threshold_c", -18.0))
    var_pct = float(p.get("var_percentile", 95))
    value = float(p.get("shipment_value_rand", 450000))
    seed = int(repro_params.get("random_seed", 42))

    out_routes = {}
    for name, r in mc_inputs["routes"].items():
        rng = np.random.default_rng(seed)
        drive_hours = float(r["drive_hours"])
        peaks = np.empty(trials)
        losses = np.empty(trials)
        for i in range(trials):
            thermal_noise = rng.normal(0.0, float(p.get("thermal_noise_std_c", 2.5)))
            delay_s = float(p.get("border_delay_mean_s", 300)) + rng.normal(
                0.0, float(p.get("border_delay_std_s", 45)))
            infra = rng.random() < float(p.get("infra_shock_prob", 0.05))
            idle_hours = max(0.0, delay_s) / 3600.0 + (1.5 if infra else 0.0)
            peak = peak_fn(drive_hours, idle_hours, ambient + thermal_noise)
            peaks[i] = peak
            losses[i] = loss_fn(peak)
        breach_prob = float(np.mean(peaks > breach_c))
        var_loss_frac = float(np.percentile(losses, var_pct))
        out_routes[name] = {
            "trials": trials, "drive_hours": round(drive_hours, 3),
            "prob_breach": round(breach_prob, 4),
            "mean_peak_temp_c": round(float(peaks.mean()), 3),
            "p95_peak_temp_c": round(float(np.percentile(peaks, 95)), 3),
            "expected_loss_fraction": round(float(losses.mean()), 4),
            f"var{int(var_pct)}_loss_fraction": round(var_loss_frac, 4),
            f"var{int(var_pct)}_rand": round(var_loss_frac * value, 2),
        }
        logger.info("[MONTE CARLO] %s: breach=%.1f%% VaR%d=R%.0f",
                    name, breach_prob * 100, int(var_pct), var_loss_frac * value)
    return {"ambient_temp_c": ambient, "breach_threshold_c": breach_c,
            "shipment_value_rand": value, "routes": out_routes}

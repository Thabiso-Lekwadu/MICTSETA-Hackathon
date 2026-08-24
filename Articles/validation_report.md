# Northern Cape Fleet Routing — System Validation Report

**Date:** 2026-08-23
**Network tested:** `nc_road_graph.pkl` — real topology (63,346 nodes, 68,974 edges)
**Result:** 3 PASS | 1 WARN | 2 FAIL (of 6 checks)

## Summary

| Suite | Metric | Score | Tolerance | Status |
|---|---|---|---|---|
| Resilience / Incident | Node Overlap After Incident | 94.6% | Reroute expected (< 100%) | **PASS** |
| Cold-Chain Optimizer | Mean Spoilage Cost Reduction | 0.4% | ≥ 5% | **FAIL** |
| Telemetry Cross-Validation | Spatial Deviation (RMSE) | 126.1 m | ≤ 30 m | **FAIL** |
| Telemetry Cross-Validation | Snapping Accuracy Rate | 86.7% | ≥ 80% (within 150 m) | **PASS** |
| Telemetry Cross-Validation | Temporal Convergence | +5.3% | within ±20% | **PASS** |
| Self-Calibration Engine | Speed Model Drift Detected | 5 classes | ≥ 10% deviation | **WARN** |

## Findings

**Resilience — PASS.** A simulated incident worth 50% of the baseline route's own cost, injected mid-route on the Port Nolloth → Upington path, forces a reroute that shares 94.6% of nodes with the original (240 → 265 hops). The network has genuine redundancy here.

**Cold-Chain Optimizer — FAIL.** Across 20 long-haul O/D pairs (mean separation 369 km, all ≥ 50 km apart), the spoilage-optimized router only beats the time-only router by 0.4% on average (range 0.0–2.2%), against a 5% target. This is a real shortfall, not a sampling artifact — the long-haul-biased sample removed the previous false-FAIL cause (adjacent-node pairs diluting the mean toward zero).

**Spatial Deviation (RMSE) — FAIL.** Telemetry pings deviate 126.1 m on average (point-to-segment) from their matched road, well outside the 30 m target — worth investigating whether this is real GPS/road-geometry noise or an artifact of the synthetic telemetry generator.

**Snapping Accuracy — PASS, but against a borrowed tolerance.** The 150 m tolerance used here is `nc_road_network.py`'s `SNAP_TOLERANCE_M`, a graph-construction constant (endpoint-merge distance when building the graph), not a purpose-built GPS-accuracy threshold. The 86.7% pass rate is real, but the bar is looser than intended — recommend a dedicated tolerance constant for this check.

**Temporal Convergence — PASS (after fix).** Previously failed at +231.9%/+284.7% due to a bug in the synthetic telemetry generator (elapsed time was back-calculated from a single segment's speed rather than integrated hop-by-hop). Fixed; now reads +5.3%, consistent with the intentional 15% real-world speed slowdown built into the simulator.

**Speed Model Drift — WARN (expected).** 5 of the road classes the synthetic session touched deviate ≥10% from `DEFAULT_SPEED_KMH`'s assumptions, producing a calibrated speed vector. This is the self-calibration engine doing its job, not a defect.

## Fixes applied this session
1. **Temporal Convergence bug** — synthetic telemetry now integrates elapsed time hop-by-hop instead of back-calculating from one segment's speed.
2. **Cold-Chain sampling bias** — O/D pairs are now filtered to ≥50 km apart, so the test reflects long-haul trucking rather than adjacent intersections.
3. **Incident penalty scaling** — the resilience test's injected incident now scales with the baseline route's own cost (50%) instead of a fixed +3h, so it stays meaningful at any network size.

## Open items
- Cold-Chain Optimizer still fails its 5% target — needs investigation into why the spoilage-aware router isn't finding meaningfully better routes on real long-haul pairs.
- Spatial RMSE still fails its 30 m target — needs investigation into GPS noise model or road geometry resolution.
- Snapping Accuracy's 150 m tolerance should be decoupled from `nc_road_network.py`'s graph-building constant.

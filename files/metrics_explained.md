# Northern Cape Fleet Dispatch — Metrics & KPI Guide

This explains every number on the dashboard: what it measures, how it's
calculated under the hood, and what it means for a dispatcher making a
decision. Grouped by where you see it on screen.

---

## 1. How the system thinks about risk (read this first)

Cold-chain spoilage for the cargo (chilled/frozen seafood) happens for
**two separate physical reasons**, and the whole dashboard is built around
tracking them independently before blending them into money terms:

- **Mechanical risk** — vibration and jostling damage from rough roads,
  building up the longer and rougher the drive.
- **Thermal risk** — heat exposure damage, building up the longer the cargo
  sits above its safe temperature.

Both are expressed as a **% of a "total loss" budget** (a number of
risk-hours before the model treats the cargo as a write-off), then the two
percentages are blended 50/50 into one **composite risk %**, which is
finally converted into a **Rand value** using the shipment's value. That
Rand-conversion step is what "Value at Risk So Far" is.

---

## 2. Trip Plan (fixed at departure)

Calculated once, the moment you click "Start Trip," and then **frozen** —
it doesn't change as the truck drives. It's the fixed yardstick everything
"live" is compared against.

| Metric | What it means | How it's calculated |
|---|---|---|
| **Planned ETA (optimized)** | How long the *spoilage-aware* route is expected to take, start to finish. | Sum of drive-time across every road segment on the route the system chose to minimize spoilage risk (not necessarily the fastest route). |
| **Time Cost vs Standard** | How many extra minutes the spoilage-safe route costs you compared to the fastest possible route (a normal maps app). | `optimized route time − standard (fastest) route time`. A positive number means the safer route is slower — that's often the trade-off for lower spoilage risk. |
| **Planned Spoilage Risk Avoided** | How many percentage points of spoilage risk you save by taking the optimized route instead of the fastest one. | `standard route's spoilage risk % − optimized route's spoilage risk %`. |

**Business translation:** this block answers *"was it worth taking the
slower route?"* — a few extra minutes on the road in exchange for
meaningfully lower spoilage risk on a high-value shipment is usually a good
trade.

---

## 3. Live Remaining Route

Unlike the Trip Plan above, this block **recalculates on every poll** from
the truck's *actual current position* — so it genuinely shrinks toward zero
as the truck approaches the destination.

| Metric | What it means | How it's calculated |
|---|---|---|
| **Remaining ETA (optimized)** | Minutes left to arrive, from right now. | Same route-optimization logic as the Trip Plan, but re-run from the truck's live GPS position to the destination. |
| **Remaining Spoilage Risk Avoided** | How much spoilage risk is still being avoided on the *rest* of the journey by staying on the optimized route. | `standard route's remaining spoilage risk % − optimized route's remaining spoilage risk %`, computed from current position onward. |

**Business translation:** if this number keeps shrinking as expected, the
truck is tracking the planned route. If it jumps unexpectedly, the truck
may have deviated or a road condition report changed the optimal path.

---

## 4. Cargo Condition

This is the heart of the spoilage-monitoring system — what's actually
happening to the shipment right now.

| Metric | What it means | How it's calculated |
|---|---|---|
| **Cargo Temperature** | The current reefer-chamber temperature reading. | Comes from the live sensor feed (hardware mode) or a realistic simulated curve (simulator mode — a gentle warm-up from 3.0°C to 4.5°C over the trip, plus a simulated door-open/refrigeration-hiccup spike to ~10.6°C partway through, so the model has something realistic to react to). |
| **Mechanical Risk (roads so far)** | % of the "total mechanical loss" budget used up by road roughness on the distance *already driven*. Simulator mode only — hardware mode has no fixed planned route to measure "so far" against. | Each road segment has a roughness cost (see the table in §7). These accumulate as the truck drives, and the running total is expressed as `min(100%, accumulated cost ÷ 20.0-risk-hour budget)`. |
| **Thermal Risk (heat so far)** | % of the "total thermal loss" budget used up by heat exposure so far. | The cargo's temperature history is integrated over elapsed time. Any time spent *above* the Safe Cargo Temp threshold accrues risk faster — the rate roughly **doubles for every 10°C over the threshold** (a standard food-science rule of thumb, called a Q10 model). Expressed as `min(100%, accumulated hours ÷ 24-hour budget)`. |
| **Value at Risk So Far** | The estimated Rand loss if the shipment were declared spoiled right now, given everything that's happened to it so far. | `composite risk % × Shipment Value`, where `composite risk % = 0.5 × Mechanical Risk + 0.5 × Thermal Risk` (or just Thermal Risk alone in hardware mode, since mechanical isn't tracked there). Rounded to the nearest R100. |
| **Cargo temperature status banner** (✅ Nominal / ⚠️ Elevated / 🛑 Critical) | A quick traffic-light read on the current reading. | **Nominal**: at or below your Safe Temp Max. **Elevated**: up to 5°C above it. **Critical**: more than 5°C above it. |

**Business translation:** "Value at Risk So Far" is the number a manager
actually cares about — it turns two abstract percentages into a Rand
figure that scales directly with what you set as the shipment's value.
Raise the shipment value in the sidebar and this figure (and the trip-plan
Rand-saved figure) scale up proportionally; it's not a hardcoded truck-load
value anymore, you control it per shipment.

---

## 5. Live Temperature History Chart

A rolling line chart of cargo temperature over the session, one point
added per poll while the trip is moving (frozen once the truck arrives).
Not a KPI by itself, but it's what "Thermal Risk (heat so far)" is
effectively the area-under-the-curve of, above the safe threshold.

---

## 6. Telemetry Row (bottom of Dispatch tab)

| Metric | What it means |
|---|---|
| **Vehicle ID** | Which truck this is (currently a single simulated/tracked vehicle, `TRUCK-01`). |
| **Current Coordinates** | The truck's live latitude/longitude. |
| **Cargo Temperature** | Same reading as above, repeated here for a quick glance alongside position. |
| **Trip Progress** | % of the planned route completed (simulator mode). Falls back to **Spoilage Cost Index** in hardware mode (no fixed route to measure progress against), which is the raw accumulated spoilage cost of the optimized route — lower is better, but it's a relative index, not a percentage. |
| **Data Feed Source** | Whether the position/telemetry is coming from real hardware (Traccar GPS tracker) or the built-in simulator. |

---

## 7. Route Weather Analytics tab

| Metric | What it means | How it's calculated |
|---|---|---|
| **Outside Ambient Temperature** | Live outside air temperature at the truck's current position. | Pulled from the free Open-Meteo weather API for the truck's live coordinates (cached briefly so it isn't re-fetched on every 2-second poll). |
| **Rain Index** | Live rainfall at the truck's current position, in mm. | Same Open-Meteo source. |
| **Dynamic Thermodynamic Warming Rate** | How fast an *idling* (stopped) reefer chamber is modeled to warm up right now, given the outside temperature. | 🔥 Accelerated ≥38°C outside · ➖ Normal 20–38°C · 🧊 Suppressed <20°C. This only matters while the truck is stopped — it's a stand-in for what a real cargo-interior sensor would show once hardware is wired up, since without one, ambient weather is used to model how the chamber drifts while idling. |
| **Weather alert banner** | Flags conditions that raise spoilage or road risk. | 🔥 *Extreme Heat* if ambient ≥38°C (compressor strain). 🌧️ *Heavy Rain / Washout Risk* if rainfall ≥5mm (unpaved-road washout risk). Otherwise ✅ Normal. |
| **Road Condition Forecast table** | Ambient temperature, rain, and alert level at the Origin, Midpoint, and Destination of the *planned* route, plus the truck's live Current Position. | Same Open-Meteo lookup, run for each of those fixed reference points along the route (Origin/Midpoint/Destination don't move; Current Position tracks the truck). Useful for spotting a storm cell sitting on the road ahead before the truck reaches it. |

---

## 8. Business Settings (sidebar — inputs, not outputs)

These aren't metrics themselves, but they directly control several of the
calculations above, live, with no restart needed:

| Setting | What it controls |
|---|---|
| **Safe Cargo Temp Max (°C)** | The line above which Thermal Risk starts accruing faster than baseline, and above which the status banner flips from Nominal to Elevated/Critical. Set this from your own domain knowledge of the species, packaging, and ice quality involved. |
| **Shipment Value (R)** | The Rand value of *this specific* cargo. Drives "Value at Risk So Far," and the Rand-saved figures in the Trip Plan and Live Remaining sections — all of them scale linearly with this number. A trip already in progress keeps its *frozen* Trip Plan Rand-saved figure at whatever the shipment value was when the trip started; only the live/remaining figures pick up a change immediately. |

---

## Appendix: the constants behind the model

For reference, the fixed assumptions baked into the risk model:

- **Total mechanical-loss budget:** 20.0 risk-hours (100% mechanical risk = this much accumulated road-roughness cost).
- **Total thermal-loss budget:** 24.0 risk-hours (100% thermal risk = this much accumulated heat-exposure cost).
- **Mechanical/Thermal blend weights:** 50% / 50% into the composite risk figure.
- **Q10 thermal rate factor:** 2.0 — spoilage rate roughly doubles for every 10°C the cargo sits above the safe threshold.
- **Road roughness assumptions** (used to compute mechanical risk per segment):

  | Road Condition | Roughness Multiplier |
  |---|---|
  | Smooth Tarmac | 1.0 |
  | Corrugated / Rough Gravel | 1.8 |
  | Severe Potholes | 2.6 |
  | Gravel Bed Erosion | 6.0 |
  | Impassable / Washed Out | 6.0 |
  | Flash Flood Mud Trap | 6.5 |
  | Structural Road Washout | 7.5 |

  (The last three are storm-specific conditions, deliberately set high
  enough that a driver's storm report always forces an immediate reroute.)

- **Simulated time acceleration:** in Simulator mode, trip time runs 45×
  faster than real time, so a demo trip that would take hours on the road
  finishes in minutes on screen — this only affects the *pace* of the
  simulation, not any of the risk math itself.

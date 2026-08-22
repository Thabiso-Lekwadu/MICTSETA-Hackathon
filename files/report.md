# Engineering and Architectural Justification Report

## A Cold-Chain-Aware Dynamic Routing System for Northern Cape Fisheries and Freight Logistics

**MICT SETA Skills Development Hackathon, Northern Cape, 28-29 August 2026**
**Track: Transport, Trade and Fisheries**

---

## 1. Title and Executive Overview

### 1.1 Solution Title

**Spoilage-Aware Dynamic Routing for Cold-Chain Freight in the Northern Cape: A Decoupled Topology, Impedance, and Real-Time Ground-Truth Architecture**

### 1.2 Why Standard Routing Engines Fail This Use Case

Commercial routing engines such as Google Maps, and the OpenStreetMap-based routing stacks most of them are built on, optimize a single objective: minimum travel time. For a passenger car in an urban network, this is a reasonable proxy for what the driver wants. For a refrigerated freight vehicle (a reefer) carrying wild-caught fish or abalone across the Northern Cape, it is not.

Three structural properties of this geography and this cargo type break the single-objective assumption:

1. **Road surface quality varies enormously and is decoupled from distance.** A route that saves twenty minutes by using an ungraded farm track instead of a paved secondary road is, for a reefer, very often the worse choice: the vibration load on the vehicle and the thermal load on the cargo can outweigh the time saved many times over. A time-only optimizer cannot see this trade-off because it has no representation of road surface as a cost variable.

2. **Administrative delay is a fixed cost that a distance-and-speed model cannot represent.** A border crossing such as Vioolsdrift does not slow a vehicle down over distance; it stops it entirely for a period that is largely independent of how the vehicle got there. This is a node-level cost, not an edge-level one, and most routing engines have no primitive for it.

3. **The road network itself is not fully connected in the underlying data.** In sparsely mapped rural regions, OpenStreetMap extracts contain large numbers of digitized-but-unconnected track fragments. A naive router that does not account for this will either silently choose a physically nonexistent shortcut or throw a runtime path-finding exception. This is addressed at the data layer, not the routing layer, in this system.

This project's response is a decoupled, layered architecture in which data ingestion, network topology construction, impedance (cost) modeling, path optimization, live telemetry, and the user interface are five independent components connected by well-defined data contracts. A time-only baseline can still be computed at any point (it is, in fact, computed on every routing request as a comparison baseline), but the system's primary output is a spoilage-risk-weighted route that trades a small amount of time for a large reduction in cold-chain exposure.

### 1.3 Scope of This Report

This report documents the system exactly as implemented, not as originally specified. Where an implementation detail differs from an earlier design brief (for example, the graph is an undirected simple graph rather than a directed multigraph, and the cross-border delay is a node-arrival penalty of five hours rather than a spatial buffer check with a three-hour penalty), this report states the actual behavior and explains why, rather than restating an assumption that does not match the code. Section 5 is dedicated entirely to separating real, measured data from documented assumptions, because a hackathon panel's first good question will be how much of this is real.

---

## 2. Comprehensive System Architecture and Layer Mechanics

The system is organized as seven layers. Data flows strictly downward through the first five; the sixth (telemetry) and seventh (interface) layers sit alongside the optimizer and consume its output on every request cycle.

```
Data Ingestion Layer (Geofabrik OSM extract, parquet/GeoJSON)
        |
Data Audit and Repair Layer (25m snapping, component pruning, tiered imputation)
        |
Topology Grid Engine (networkx.Graph: nodes = junction coordinates, edges = road segments)
        |
Impedance Matrix Modeling Component (spoilage_cost, base_time_mins per edge)
        |
Core Optimizer Router (Dijkstra shortest path, override-aware weight function)
        |                                   \
Dynamic Telemetry and Snapping Layer          Visual Interface Component
(live GPS, nearest-neighbor node/edge lookup)  (Streamlit + Folium, FastAPI client)
```

### 2.1 Data Ingestion Layer

The source dataset is a Geofabrik OpenStreetMap extract for South Africa, spatially filtered to the Northern Cape bounding box (longitude 16.45 to 25.30, latitude -31.85 to -24.60). The working copy of this extract used throughout development contains **107,323 raw road-line segments**, each carrying OSM-native attributes: `osm_id`, `fclass` (road classification), `name`, `ref` (route reference such as N7, N14, or R355), `oneway`, `maxspeed`, `bridge`, `tunnel`, and `layer`.

Storage uses Apache Parquet via `pyarrow`/`geopandas.read_parquet`, chosen over the original GeoJSON and Shapefile formats for three concrete reasons: columnar storage allows sub-second full-column reads without deserializing geometry that is not needed for a given query; the compressed file size for this dataset is roughly 2.6 times smaller than the equivalent GeoJSON (32.7 MB versus 86.6 MB for the raw extract used in this project); and Parquet preserves exact dtypes across save and load cycles, which is not guaranteed by GeoJSON's text-based number representation.

### 2.2 Data Audit and Repair Layer

Raw OSM extracts cannot be routed on directly. Three integrity problems must be resolved before the data becomes a usable network, and this system resolves all three explicitly and auditably rather than silently.

**Class filtering.** Eight OSM classes are excluded before any further processing: `residential`, `service`, `footway`, `path`, `steps`, `pedestrian`, `cycleway`, and `bridleway`. These represent in-town local streets and non-vehicle infrastructure that are irrelevant to province-scale freight routing and would otherwise inflate the graph by tens of thousands of edges with no routing value. This reduces the working set from 107,323 to **43,424 segments**.

**Geometric simplification.** Each retained line geometry is simplified using the Douglas-Peucker algorithm at a tolerance of 0.0005 degrees (approximately 55 meters), with `preserve_topology=True` so that endpoint coordinates are never altered, only intermediate vertices are thinned. This reduced the total vertex count across the filtered dataset from approximately 1,243,580 to approximately 169,433, a reduction of roughly 7.3 times, with no change to the routing topology.

**Endpoint snapping and component pruning.** This is the most consequential repair step and is documented in detail because it materially changes what the system can and cannot route between. Raw OSM line endpoints that represent the same real-world junction are not guaranteed to share exactly identical floating-point coordinates; digitization by different contributors, at different times, using different source imagery, routinely produces junctions that are a few meters apart in the data but zero meters apart in reality. Left unresolved, this produces a graph that looks connected on a map but is not connected as a data structure, which causes `networkx.exception.NetworkXNoPath` at query time for perfectly reasonable-looking routes.

The repair is a coordinate-clustering snap: every line endpoint is projected into an approximate local metric plane (an equirectangular approximation scaled by the network's mean latitude, which is accurate enough at the tolerance in use), a `scipy.spatial.cKDTree` is built over all endpoints, and any pair of endpoints within **25 meters** of each other is merged into a single graph node via a union-find structure. This is a standard technique in OSM-derived network construction, and the 25 meter tolerance was chosen as a value large enough to absorb realistic digitization error while small enough to avoid incorrectly merging two genuinely distinct, closely spaced road junctions.

After snapping, the full graph contains 134,096 nodes and 121,552 edges. Critically, this graph is **not** fully connected: it decomposes into 14,830 separate connected components. The largest of these, designated `G_main`, contains **34,394 nodes and 35,263 edges**, representing only **25.6 percent** of all nodes in the snapped graph. The remaining 74.4 percent is spread across thousands of small, isolated fragments, which in this rural, low-density region are overwhelmingly individual farm tracks and dead-end rural roads that were digitized from satellite imagery without a mapper ever confirming or drawing the connecting link to the rest of the network. This is treated as a genuine characteristic of the source data, not a bug to be silently worked around, and Section 3.1 documents the specific engineering justification for isolating `G_main` as the sole routable graph.

### 2.3 Topology Grid Engine

The routable network is represented as a `networkx.Graph`: an **undirected, simple graph**, not a directed multigraph. This is stated explicitly because it is a real simplification relative to a fully general road-network model, and the report would be incomplete if it did not say so plainly. Each node is a coordinate-cluster representative (a single longitude and latitude pair after snapping); each edge carries a metadata dictionary of `length_km`, `travel_time` (hours), `fclass`, `roughness`, `imputed_speed_kmh`, `base_time_mins`, `spoilage_cost`, and an `override` slot described in Section 2.4.

The choice of an undirected simple graph over a directed multigraph has two consequences that are worth stating for completeness. First, the `oneway` attribute present in the raw OSM data is not currently propagated into edge directionality; a one-way street is represented identically to a two-way street. Second, where two OSM ways exist between the same pair of snapped endpoints (which occurs occasionally, for example where a service road runs parallel to a short section of the trunk road), only the first-encountered edge is retained, since adding an edge to an existing node pair overwrites rather than duplicates. Both of these are acceptable simplifications for a province-scale freight-corridor demonstration where one-way restrictions are rare outside town centers, which have already been excluded from the routable graph, but both are explicitly flagged here as a direction for future hardening in Section 6.1 rather than presented as already solved.

### 2.4 Impedance Matrix Modeling Component

Every edge carries two independently computable cost fields, both derived, never measured directly:

```
base_time_mins  = travel_time_hr x 60
spoilage_cost    = travel_time_hr x roughness
```

where `travel_time_hr = length_km / imputed_speed_kmh`, and `roughness` is a per-`fclass` multiplier documented in Section 4.2. `spoilage_cost` is the primary routing weight used throughout the system; `base_time_mins` is retained and reported alongside it so that every route comparison can show both what the time cost is and what the risk cost is, letting the trade-off be seen directly rather than hidden inside a single blended number.

Two additive, node-level penalties are layered on top of this per-edge cost during path evaluation, rather than being baked into the edges themselves, because they are properties of arriving at a specific location, not of traversing a specific road segment:

- **Cross-border customs delay**, detailed in Section 4.1.
- **Live incident delay**, injected dynamically by the telemetry pipeline described in Section 2.6, and separately by the driver ground-truth override layer described in Section 6.2.

### 2.5 Core Optimizer Router

Path optimization uses `networkx.shortest_path` with Dijkstra's algorithm (the default for non-negative edge weights, which `spoilage_cost` always is by construction), invoked with a **callable weight function** rather than a static attribute name. This is a deliberate design choice: a callable weight function can inspect the `override` field on each edge at query time and apply the fallback-hierarchy logic described in Section 6.2 (driver report first, tiered-imputation baseline second) without ever mutating the graph's baseline data. The baseline `spoilage_cost` field is written once at initialization and never overwritten; overrides live in a separate field and are checked first. This means the original data-audit baseline is always recoverable, and a route computed while ignoring any active override can always be compared against a route computed while respecting active overrides, to detect precisely whether a given field report has actually changed the optimal route.

### 2.6 Dynamic Telemetry and Snapping Layer

Live vehicle position is not assumed to land on a graph node. Each incoming GPS fix (latitude, longitude) is snapped to the nearest node in the active topology using a `scipy.spatial.cKDTree` built once at service startup over every node in `G_main`, projected into the same approximate local metric plane used for endpoint snapping in Section 2.2. This is a Euclidean nearest-neighbor lookup, not a point-to-line-segment projection; it is accurate to within the spacing of graph vertices along a road, which after the simplification in Section 2.2 is on the order of tens of meters on straight sections and finer on curves, and is sufficient for the routing decision this system needs to make. A parallel index, an edge spatial index, performs the equivalent nearest-neighbor lookup against edge midpoints rather than node coordinates, and is used specifically by the driver field-report endpoint to identify which road segment, rather than which junction, a report refers to.

### 2.7 Visual Interface Component

The user interface is a Streamlit application consuming a FastAPI backend over plain HTTP on localhost port 8000, with the map rendered via Folium inside the streamlit-folium component. The interface is organized as two tabs: a Fleet Dispatch Hub showing the live vehicle position, live metrics, and the currently optimal route on a Folium canvas, and a Driver Ground-Truth Form for field reporting, described in Section 6.2.

One implementation detail is worth recording here as a genuine engineering lesson rather than omitting it. The auto-refresh mechanism was initially implemented as an unconditional sleep followed by a full-script rerun at the top level of the application. This causes the entire page, including the map component, to be torn down and rebuilt on every refresh cycle, producing visible flicker and a map that recentres itself every cycle. The corrected implementation scopes the auto-refresh to a fragment containing only the metrics and map, leaving the tab structure, the field-report form's input state, and the map's own pan and zoom state untouched between refreshes. This is a standard performance pattern in this class of dashboard framework, and is documented here because it materially affects how the system should be demonstrated to a judging panel: an unscoped rerun loop looks like an unstable system even when the underlying routing logic is correct.

---

## 3. Granular Modeling Decisions and Technical Justifications

### 3.1 Topological Pruning: Isolating the Largest Connected Component

**Decision.** Route computation is restricted exclusively to `G_main`, the largest connected component (34,394 of 134,096 nodes, 25.6 percent). The remaining 74.4 percent of the snapped graph, spread across 14,830 disconnected fragments, is excluded from routing entirely.

**Alternative considered and rejected.** The alternative was to route across the full snapped graph and handle path failures reactively, catching a no-path exception at query time and returning an error to the caller. This was rejected because it converts a data-quality issue, known and quantifiable at build time, into a runtime failure mode that could surface unpredictably during a live demonstration, depending entirely on which two points a judge or a script happened to pick.

**Justification.** By construction, any two nodes within `G_main` are guaranteed to have at least one path between them; the no-path exception becomes structurally unreachable for any query where both endpoints have been snapped to `G_main`, rather than merely unlikely. This is a stronger guarantee than exception handling can provide, because it is enforced at the data layer rather than defended against at the query layer. The cost of this decision is that any location whose nearest node happens to fall in one of the smaller fragments will be snapped to the nearest node that is in `G_main`, which may be several kilometers away rather than a near-exact match. This is exactly the mechanism that produces the Vioolsdrift border-post snap distance of approximately 9.6 kilometers documented in Section 5, and it is treated as an honest, visible limitation rather than a hidden one.

### 3.2 Tiered Speed Imputation Hierarchy

**Decision.** Every road segment's `maxspeed` value of exactly zero is treated as missing data, not as a literal zero-kilometer-per-hour speed limit, and is replaced through a five-tier fallback hierarchy before any travel-time calculation is performed.

**Why `maxspeed == 0` must be treated as missing, explicitly.** OpenStreetMap's tagging convention uses zero (or an absent tag, normalized to zero in this Geofabrik shapefile extract) to mean speed limit unknown or unmapped, not that vehicles are forbidden to move and not that the road carries a genuine zero-kilometer-per-hour limit. A vehicle cannot physically travel at zero kilometers per hour while still making progress along a route. If this value were used directly in the travel-time calculation, the result is a division-by-zero error for any segment with an unmapped speed limit, or, if guarded naively by clamping the denominator to a small value, a segment that appears to take an effectively infinite amount of time to traverse, which the optimizer would then always route around regardless of whether that road is actually usable. Given that **105,630 of 107,323 raw segments (98.4 percent)** have `maxspeed == 0`, either failure mode would make the vast majority of the rural network either uncomputable or falsely appear catastrophically slow.

**The five-tier hierarchy, in the order actually applied:**

| Tier | Method | Trigger condition | Segments (of 43,424 filtered) | Share |
|---|---|---|---|---|
| 0 | Observed | Real, non-zero `maxspeed` value present | 1,319 | 3.0% |
| 1 | Class-and-reference median | Median of real values sharing the same road class and route-reference prefix, for example primary roads carrying an R-route reference | Group has 10 or more real samples | 2,410 | 5.5% |
| 2 | Class median | Median of real values for the road class alone | Class has 30 or more real samples | 11,287 | 26.0% |
| 3 | Parent-class median | Link classes borrow their base class's median | Parent has 5 or more real samples | 172 | 0.4% |
| 4 | Domain default | Documented, not fitted, South African road-design-speed assumption table | Last resort | 28,236 | 65.0% |

Every segment retains a source label recording exactly which tier produced its value, which makes the imputation auditable: any figure downstream of this table can be traced back to whether it rests on real observation or on a documented assumption.

**A specific finding that justifies the tiered design over a flat fallback.** An earlier, simpler version of this hierarchy allowed the five graded track classes (progressively degraded farm-track categories) to borrow from a single combined all-grades real-data median when their own individual sample sizes were too small to trust directly. Testing this revealed a defect: because the grade-3 track class alone happened to have 33 real samples, enough to clear its own class-median threshold, at a relatively high median of 50 kilometers per hour, the combined-grade average was pulled upward, and the remaining four grades, which individually had only 2 to 21 real samples each, all collapsed to approximately the same borrowed speed. This defeats the entire purpose of grading: a well-maintained grade-1 track and a barely passable grade-5 track were being assigned nearly identical travel times. The fix removes the graded track classes from the parent-borrowing tier entirely; when a graded track class's own sample size is not met, it falls straight through to the grade-differentiated domain-default table, which correctly orders grade 5 at 15 kilometers per hour below grade 4 at 20 below grade 2 at 30 below grade 1 at 40. One residual, honestly reported anomaly remains: grade 3's own real-data median, 50 kilometers per hour, is higher than the assumption-driven grade-1 and grade-2 values, because it earned its own class-median tier on real data while they did not. This is not corrected to force a clean monotonic pattern; the real data is reported as measured, with the inconsistency in OpenStreetMap's community-assigned track grading noted as the explanation rather than smoothed over.

### 3.3 Multi-Criteria Impedance Modeling

**Decision.** Routing cost is not a single blended scalar computed once and then treated as opaque. It is a small number of independently interpretable components, combined by a stated, simple rule, so that the operator can always decompose why the router chose a given path into how much of the cost is time, how much is roughness, and how much is a fixed delay.

**Formulation.**

```
per-edge:       spoilage_cost(edge) = travel_time_hr(edge) x roughness(edge)
node arrival:    spoilage_cost(node) += border_delay_hr(node) x idle_heat_risk_factor   [if node is a border post]
path total:      spoilage_cost(path) = sum over edges in path of spoilage_cost(edge)
                                         + sum over border nodes traversed of spoilage_cost(node)
```

Roughness is a multiplicative term because vibration-driven cold-chain risk compounds with exposure duration on a given surface type; a longer segment of the same surface should carry proportionally more risk, not a flat penalty regardless of length. Border and incident delays are additive, node-level terms because they represent a fixed administrative or operational stoppage largely independent of the distance traveled to reach that point. This hybrid multiplicative-and-additive structure was chosen over a single learned or hand-tuned linear weighting of raw features specifically because it keeps each term separately auditable and separately calibratable, which matters directly for the calibration strategy described in Section 6.1.

---

## 4. Operational Risk Engineering: Customs, Roughness, and Spoilage

### 4.1 Cross-Border Customs Bottleneck: Actual Mechanism

The cross-border delay is implemented as a **node-arrival penalty applied during path-cost evaluation**, not as a spatial buffer or proximity-radius check performed against every edge in the graph. Concretely: the border post's approximate coordinate is snapped once, at service startup, to the nearest node in `G_main` using the same nearest-neighbor mechanism described in Section 2.6. Whenever a computed path passes through that specific snapped node, a fixed penalty is added to the running total: **5.0 hours** added to the time cost, and that same 5.0 hours multiplied by an idle-heat risk factor of **1.2** added to the spoilage cost, reflecting that a stationary vehicle in a customs queue accumulates thermal risk to its cargo even though it is not experiencing vibration damage during that period.

Because the extracted road network is clipped to the Northern Cape provincial bounding box, and the Vioolsdrift crossing sits close to that boundary, the nearest routable node to the border post's nominal coordinate is approximately **9.6 kilometers** away rather than an exact match. This is reported honestly as a real limitation of a bounding-box-clipped, line-only OSM extract, which contains no explicit border-control point feature to snap to directly. A production-grade version of this system would query the Overpass API live for features tagged as border-control barriers, and snap to that discovered point instead of a hardcoded approximate coordinate; this mechanism is implemented and demonstrated separately in this project's cold-chain optimizer module as a proof of concept, but is not yet the default data source used by the production backend service.

A related but distinct mechanism exists in this system's live-telemetry demonstration script: a simulated ad hoc infrastructure breakdown, injected onto whichever road segment sits immediately ahead of the vehicle's live-snapped position at a specific telemetry step, carrying a separate, configurable penalty of **180 minutes (3 hours)**. This is not the border-customs mechanism; it demonstrates a different capability, dynamic re-optimization from a moving origin in response to an unplanned, location-specific event, as opposed to the border delay's fixed, location-known event. Conflating these two mechanisms would misrepresent the system to a technical reviewer, so this report keeps them explicitly separate.

### 4.2 Road Roughness and Its Relationship to the International Roughness Index

The International Roughness Index is a real, standardized pavement-quality metric, typically expressed in meters of vertical displacement per kilometer traveled, measured by laser profilometer equipment mounted on a survey vehicle. Reference ranges commonly cited in pavement engineering literature place newly constructed asphalt around 1.0 to 2.0 meters per kilometer, aged but maintained tarmac around 2.0 to 4.0, and severely degraded or unpaved surfaces well above 6.0, occasionally exceeding 15 to 20 for the worst eroded tracks.

This system does **not** use measured International Roughness Index data, because no such survey exists for the Northern Cape's rural and farm-track network at the resolution this project requires, and none was collected as part of this hackathon build. What this system uses instead is a **documented, hand-set ordinal roughness multiplier**, keyed to the OpenStreetMap road classification, intentionally described throughout this project's code and documentation as roughness rather than as the International Roughness Index, to avoid implying a precision the data does not have:

| Road class | Roughness multiplier | Conceptual comparison |
|---|---|---|
| Trunk and trunk link | 1.00 | Smooth sealed national road |
| Primary and primary link | 1.05 | Well-maintained sealed provincial road |
| Secondary and secondary link | 1.10 | Sealed secondary road |
| Tertiary and tertiary link | 1.15 | Sealed or well-graded tertiary road |
| Unclassified | 1.30 | Ungraded but regularly used rural road |
| Living street | 1.20 | Low-speed local access road |
| Track, grade 1 | 1.40 | Well-maintained farm track |
| Track, grade 2 | 1.60 | Moderately maintained farm track |
| Track, grade 3 | 1.90 | Rough farm track |
| Track, grade 4 | 2.20 | Poorly maintained farm track |
| Track, grade 5 | 2.60 | Barely passable track |

This table should be read as a relative risk ranking, ordinally consistent with the direction the International Roughness Index concept predicts, paved before graded before ungraded before eroded, not as a set of calibrated index values in meters per kilometer. Section 6.1 describes the concrete path to replacing this table with regression-fitted coefficients once real trip-time and cargo-condition data is available.

### 4.3 Road Roughness and the Physics of Cold-Chain Failure

The roughness multiplier exists because road surface condition affects refrigerated cargo through two distinct physical failure mechanisms, both of which motivate this project's design even though neither is directly instrumented or measured by the current system; they are the operational rationale for treating roughness as a first-class cost variable rather than an artifact of the model itself.

**Kinetic structural stress.** Sustained vibration from an unpaved or degraded road surface subjects a refrigerated trailer to continuous low-amplitude mechanical stress. Over an extended trip, this stress manifests as micro-cracking around door seals and gasket surfaces, and progressive loosening of compression fittings and couplings in the refrigeration unit's coolant piping. Either failure mode allows warm ambient air to intrude into an otherwise sealed cold compartment, or allows refrigerant to leak, both of which directly compromise the temperature-controlled environment the cargo depends on.

**Condenser thermal suffocation.** A refrigeration unit's condenser depends on airflow, generated partly by the vehicle's forward motion, to reject heat from the refrigerant cycle. On a corrugated or heavily rutted surface, a driver is forced to reduce speed substantially to maintain control and avoid further vehicle damage. Reduced forward speed reduces the natural airflow across the condenser coils at precisely the moment ambient desert temperatures, which in the Northern Cape's interior routinely exceed 40 degrees Celsius in summer, are placing maximum thermal load on the refrigeration system. The combined effect of reduced cooling airflow and elevated ambient heat load is a well-documented failure pattern in refrigerated transport: engine and compressor units operating outside their designed thermal envelope for extended periods, leading to compressor overheating and, in the worst case, complete refrigeration unit shutdown.

Both mechanisms justify why a spoilage-aware routing model must treat slowness caused by poor road surface as categorically different from slowness caused by traffic, even though a pure time-minimization objective cannot distinguish the two.

---

## 5. Data Integrity: Real Versus Synthetic Matrix

This section exists specifically so that a judge or technical reviewer can determine, for any number this system produces, whether it is grounded in observed data or in a documented modeling assumption.

### 5.1 Real Data

| Element | Source | Notes |
|---|---|---|
| Road line geometries | OpenStreetMap, via Geofabrik South Africa extract | 107,323 raw segments in the Northern Cape bounding box |
| OSM object identifiers | OpenStreetMap | Verified zero duplicates in the working dataset |
| Road classification (`fclass`) | OpenStreetMap community tagging | 23 distinct classes present in the raw extract |
| Route reference tags, for example N7, N14, R355 | OpenStreetMap community tagging | Used as a real signal in tier-1 speed imputation |
| One-way, bridge, and tunnel flags | OpenStreetMap community tagging | Captured in the raw data; one-way status not yet propagated into graph edge directionality, Section 2.3 |
| Observed maximum-speed values | OpenStreetMap community tagging | 1,693 of 107,323 raw segments, 1.6 percent; 1,319 of 43,424 filtered segments, 3.0 percent, carry a genuine non-zero value |
| Snapped node connectivity structure | Derived directly from geometry, not assumed | The 25.6 percent largest-component figure is a measured, not assumed, property of the real extract |

### 5.2 Synthetic, Simulated, or Imputed Data

| Element | Status | Documented in |
|---|---|---|
| 97.0 percent of speed values on the filtered network | Imputed via the five-tier hierarchy in Section 3.2 | Data audit and enrichment pipeline |
| Roughness multiplier table | Hand-set ordinal assumption, not regression-fitted or index-calibrated | Section 4.2 |
| Border customs delay, 5.0 hours | Documented assumption, not sourced from customs authority data | Section 4.1 |
| Idle heat risk factor, 1.2 multiplier | Documented assumption | Section 4.1 |
| Live incident delay, 180 minutes, in the telemetry demonstration | Arbitrary demonstration value, not tied to any real incident dataset | Live telemetry pipeline script |
| Mock vehicle telemetry stream | Five hardcoded waypoint coordinates along the Port Nolloth to Upington corridor, looped continuously | Live backend service |
| Simulated cargo temperature curve | Linear warming assumption, 0.35 degrees Celsius per telemetry step, with no basis in a measured thermal model | Live backend service |
| Driver-reported road-condition-to-roughness mapping | Documented assumption table: smooth tarmac 1.0, corrugated or rough gravel 1.8, severe potholes 2.6, impassable or washed out 6.0 | Live backend service |
| Reference town coordinates | Manually specified approximate town-center coordinates, not sourced from OpenStreetMap place nodes | Shared network module |
| Illustrative shipment value and spoilage threshold used in monetary loss estimates | Calibration constants chosen to produce an interpretable demonstration, not derived from actual catch values or cold-chain break-point research | Cold-chain optimizer module |

---

## 6. Validation, Evaluation Metrics, and Ground-Truthing Layer

### 6.1 Arbitrary Parameter Alignment and Calibration Strategy

The following parameters are currently documented assumptions, listed here explicitly alongside the concrete operational strategy that would replace each with a fitted or measured value.

| Assumed parameter | Current value | Calibration strategy |
|---|---|---|
| Border customs delay | 5.0 hours, flat | Replace with a distribution fitted to timestamped queue-time reports submitted through a future extension of the driver ground-truth form for border crossings, segmented by time of day and day of week |
| Roughness multiplier table | Hand-set, 1.00 to 2.60 | Replace with coefficients regressed from paired predicted-versus-actual travel time observations per road class, collected through the actual-speed field already present in the field-report submission endpoint |
| Domain-default speed table, tier 4 of the imputation hierarchy | Hand-set, South African road-design-speed convention | Progressively displaced by real data as tiers 0 through 2 sample sizes grow with continued fleet operation; the tiered structure means this migration happens automatically and per-segment as more real speed observations accumulate, with no code change required |
| Idle heat risk factor | 1.2, flat multiplier | Replace with a temperature-and-duration-dependent function once cargo-temperature telemetry from real refrigerated units, rather than the simulated linear warming curve, is available |

### 6.2 The Driver Ingestion Feedback Loop

The mobile-friendly field-reporting form allows any of three defined roles, Driver, Fisherman, or Cooperative Supervisor, to submit a structured report consisting of a reported location, a categorical road condition (smooth tarmac, corrugated or rough gravel, severe potholes, or impassable or washed out), and a directly measured actual traffic speed in kilometers per hour.

On submission, the backend performs a nearest-edge lookup via the edge spatial index described in Section 2.6, translates the reported road condition into a roughness override using the table in Section 5.2, computes an overridden spoilage cost for that specific segment using the driver's own reported speed rather than the tiered-imputation baseline, and writes this into the edge's override field. This override is then consulted first by every subsequent routing query's weight function, described in Section 2.5, ahead of the data-audit tiered-imputation baseline; the baseline itself is never overwritten and remains available for direct comparison at any time. This is, precisely, an in-memory priority-override layer sitting in front of the tiered-imputation baseline, not a permanent modification of the underlying dataset.

This mechanism was verified end to end against the real Northern Cape network, not only against synthetic test data. A baseline route from Port Nolloth to Upington on the live system computed 284 hops, a total time cost of 264.41 minutes, and a total spoilage cost of 4.491. A field report submitted at a verified location on the N7 corridor near Springbok, latitude -29.307839, longitude 17.138515, marked impassable or washed out with a reported actual speed of 5 kilometers per hour, caused the very next routing query to return a materially different path: 555 hops, a total time cost of 537.56 minutes, and a total spoilage cost of 11.08, with the system's detour-active flag correctly changing to true. This specific location is retained as a documented, verified demonstration point precisely because a full pairwise scan across the network's major towns found that not every road segment sits on a genuine alternative route; the underlying network is sparse enough that many individual segments are the only way through, and reporting them as impassable increases cost without changing the chosen path, which is itself an honest finding about the real network's structure rather than a flaw in the override mechanism.

### 6.3 Statistical Evaluation Metrics

Three metrics are proposed as the formal evaluation methodology for this system going forward. None of the three has yet been computed against a real historical fleet dataset, because no such dataset currently exists for this network; each is defined precisely enough below to be computed as soon as one does.

**1. Betweenness centrality deviation.** For a given node or edge in the main routable graph, betweenness centrality is the fraction of all-pairs shortest paths in the graph that pass through it. This metric quantifies graph adaptability under an incident: compute betweenness centrality for every edge under the baseline-only weighting, inject a driver-reported or simulated incident penalty on a target edge, recompute betweenness centrality for every edge under the override-aware weighting, and measure the resulting shift in centrality mass onto the edges that absorbed the rerouted traffic. A large, spatially coherent shift, traffic concentrating onto a specific alternate corridor rather than dispersing arbitrarily, is evidence that the routing engine is making a structurally sound detour decision rather than an unstable one.

**2. Impedance-to-time variance ratio.** Defined as the normalized error between this system's modeled total time in minutes for a given route and the actual elapsed trip time reported by a real driver traversing that same route:

```
variance_ratio = |modeled_time_mins - actual_reported_time_mins| / actual_reported_time_mins
```

computed per trip and aggregated as a rolling mean over the most recent set of trips. The explicit target is for this ratio to trend toward zero as the volume of real driver-reported speed data accumulates and progressively displaces the tier-4 domain-default assumptions described in Section 3.2, since a smaller ratio directly indicates that the tiered imputation hierarchy is converging on real operating conditions rather than remaining dependent on the initial assumption table.

**3. Graph validation rate.** Defined as the proportion of submitted GPS coordinates, whether from the live telemetry stream or the driver field-report form, that snap successfully to a node or edge within the main routable graph inside a defined distance threshold:

```
validation_rate = (count of snaps with distance <= threshold_km) / (total snap attempts)
```

A reasonable operational threshold, based on the town-connectivity table already measured in this project, all seven core towns snapped within 0.6 kilometers of the routable network, the boundary-clipped Vioolsdrift border post snapped at 9.6 kilometers, is 2 kilometers. Tracking this rate over time and by geographic region directly measures where the underlying OpenStreetMap extract's routable coverage is strong versus where it is thin, and provides a concrete, falsifiable signal for where future data-collection or manual road-network correction effort should be directed.

---

*This report describes the system as built and independently verified during this project's development, including direct application-programming-interface-level and automated-testing-level execution of the routing, telemetry, and field-reporting endpoints against the real Northern Cape road network extract. Figures presented without a stated source are the author's own measurements against that implementation.*
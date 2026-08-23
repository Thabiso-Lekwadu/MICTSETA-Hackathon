# Data Audit Findings

Summary of what `notebooks/Data_Audit.ipynb` and the `road_network` pipeline found
in the raw Northern Cape OSM road extract. Read this before quoting any number from
this project in a pitch -- know which ones are measured and which are assumptions.

## Scale

- 107,323 raw road segments extracted for the province.
- 43,424 kept after excluding non-freight-relevant classes (residential, service,
  footway, path, steps, pedestrian, cycleway, bridleway).

## Speed data completeness (the headline finding)

OSM's `maxspeed == 0` means "unknown," not "0 km/h." After correcting for that:

| Tier | Method | Share of segments |
|---|---|---|
| observed | real, non-zero maxspeed | ~1.2% |
| fclass_ref_median | median for (fclass, route-ref prefix), e.g. R-road primaries | ~2.2% |
| fclass_median | median for the fclass alone | ~26.0% |
| parent_class_median | `*_link` classes borrow their base class | ~0.4% |
| domain_default | documented South African road-speed assumption | ~65.0% |

Two-thirds of the network's travel-time estimate rests on a documented assumption
table, not measured data. State this plainly if asked.

`track_grade3` is a specific anomaly worth knowing: its own real-data median
(50 km/h, n=33 -- enough to earn its own tier) is *higher* than the assumption-driven
`track_grade1`/`track_grade2` values. That's not a bug; OSM's community-mapped
grading isn't strictly monotonic in every region, and real data overrides the
assumption table whenever there's enough of it, even when the result looks
counter-intuitive.

## Graph connectivity

Even after snapping endpoints within 25m of each other (to fix floating-point/
digitization mismatches), only ~26% of the network's nodes form one connected
routable component. The rest are ~15,000 small fragments -- mostly isolated farm
tracks and dead-end rural roads mapped in OSM but never connected to the formal
network. This is a known characteristic of crowd-sourced OSM data in remote,
low-density regions, not a processing bug. All routing in this project runs on the
largest connected component only.

## Bounding box

The raw extract was clipped with a rectangular bounding box, not the true
(irregular) provincial polygon, so it legitimately extends slightly into North West
Province near the northern corner (observed max latitude ~-24.10, about 0.5 degrees
north of the nominal Northern Cape extent). `conf/base/parameters.yml` documents
this with a 1.0 degree validation tolerance rather than silently narrowing it.

## Practical implication for routing demos

Divergence between the "Standard" (time-only) and "Fisheries-Optimized"
(spoilage-aware) route is real but modest at province scale on this network --
paved trunk/primary roads already dominate on raw time in most cases, so the two
routes often coincide. The strongest, most reliable demonstration is a
**regional/local leg without a paved bypass option**. A verified example used
throughout this project's live demo: reporting the segment near
**lat -29.307839, lon 17.138515** (N7 near Springbok) as "Impassable / Washed Out"
reliably triggers a visible reroute (hop count roughly 284 -> 555 on the
Kimberley-area corridor test). Don't assume every random coordinate along the
corridor will trigger a visible detour -- many won't, because that segment may be
the only way through.

# Sim-to-Real and Scenario Construction

## Static Environmental Data — Road Network Conversion

The Kuala Lumpur road network used in all experiments was sourced from **OpenStreetMap (OSM)** and converted to SUMO's native network format (`.net.xml`) using the **`netconvert`** command-line tool bundled with Eclipse SUMO 1.26.0. The resulting compressed network file is `sumo_files/osm.net.xml.gz`.

Key `netconvert` operations applied during conversion:

| Operation                     | Effect                                                                    |
| ----------------------------- | ------------------------------------------------------------------------- |
| OSM → SUMO edge/lane topology | Maps OSM `<way>` elements to directed SUMO edges                          |
| Signal-phase discovery        | Identifies controlled intersections and generates initial TLS definitions |
| Geometry clean-up             | Removes internal junctions, short edges, and redundant nodes              |

The compressed `.gz` format is read directly by SUMO at runtime and also parsed by `gen_fringe_supplement.py` via Python's `gzip` module to load the junction topology for fringe trip generation.

## Static Environmental Data — Traffic Demand Route File Generation

All `.rou.xml` route files were produced by a **five-step automated pipeline** implemented in `sumo_tools/run_pipeline.py`. The pipeline is invoked once and generates the complete set of scenario route files reproducibly.

The two rainy OOD route files are derived from `osm.peak.rou.xml` by modifying per-vType kinematic parameters to simulate reduced traction and increased following distances (`generate_rainy_xml.py`):

| Parameter              | Moderate Rain | Heavy Rain |
| ---------------------- | ------------- | ---------- |
| `accel` factor         | ×0.80         | ×0.70      |
| `decel` factor         | ×0.90         | ×0.80      |
| `maxSpeed` factor      | ×0.85         | ×0.75      |
| Reaction time `tau`    | 1.5 s         | 1.8 s      |
| Speed variance `sigma` | 0.65          | 0.75       |
| Target vehicle count   | 2 000         | 2 000      |

The four accident OOD route files are derived from `osm.moderate.rou.xml` by inserting a **statically parked vehicle** (`accident_veh`) on a specific lane at simulation time `t = 900 s`. Aggressive lane-change parameters are applied network-wide to simulate the realistic driver response to an obstruction (`gen_accidents_routes.py`):

| Parameter       | Value | Rationale                                                           |
| --------------- | ----- | ------------------------------------------------------------------- |
| `lcStrategic`   | 0.5   | Drivers begin lane-change planning later (limited urban sightlines) |
| `lcCooperative` | 0.25  | Open-lane traffic yields reluctantly                                |
| `lcAssertive`   | 1.5   | Blocked drivers accept tighter merge gaps                           |

The four accident locations and their blocked lane identifiers are:

| File               | Location                       | Blocked Lane    | Road Type       |
| ------------------ | ------------------------------ | --------------- | --------------- |
| `osm.acc1.rou.xml` | Jalan Tun Tan Cheng Lock       | `135488882#0_1` | 3-lane arterial |
| `osm.acc2.rou.xml` | Jalan Yap Ah Loy               | `771688494#0_1` | 3-lane arterial |
| `osm.acc3.rou.xml` | Jalan Petaling                 | `135623338#0_1` | 2-lane road     |
| `osm.acc4.rou.xml` | Jalan Tun Tan Cheng Lock seg-2 | `28215290#0_1`  | 2-lane road     |

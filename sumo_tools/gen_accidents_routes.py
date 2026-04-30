"""
Generates the 4 accident route files under sumo_files/.
Each file is derived from osm.moderate.rou.xml with:
  - Aggressive lane-change vType parameters to simulate realistic driver behaviour
  - One statically parked accident vehicle on a single lane
    (3-lane roads: one middle lane blocked; 2-lane roads: one right lane blocked)
Run once: python sumo_tools/gen_accidents_routes.py
"""

import re
from pathlib import Path

# Resolve paths relative to this file's location so the script works regardless
# of which directory it is invoked from.
_HERE = Path(__file__).resolve().parent
_SUMO_FILES = _HERE.parent / "sumo_files"

# Aggressive lane-change parameters injected into every vType (accident files only).
#   lcStrategic  0.5  – drivers begin lane-change planning much later than normal
#                       (realistic: dense urban streetscape limits forward visibility)
#   lcCooperative 0.25 – open-lane traffic yields reluctantly but not absolutely
#                        (0.1 was too extreme; produced near-deterministic gridlock
#                         regardless of signal timing, poisoning the reward signal)
#   lcAssertive  1.5  – blocked vehicles force into tighter gaps to escape the queue
#                       (realistic: anxious drivers accept smaller merge gaps)
_LC_PARAMS = ' lcStrategic="0.5" lcCooperative="0.25" lcAssertive="1.5"'

accidents = [
    {
        "suffix":           "acc1",
        "stop_lane":        "135488882#0_1",
        "start_pos":        35.0,
        "end_pos":          55.0,
        "route_edges":      "771731732 1159554013#0 135488882#0 780210981#0 780210980 28215290#0 436198228#0",
        "desc":             "Jalan Tun Tan Cheng Lock (3-lane arterial, middle lane blocked)",
        "three_lane":       False,
    },
    {
        "suffix":           "acc2",
        "stop_lane":        "771688494#0_1",
        "start_pos":        20.0,
        "end_pos":          40.0,
        "route_edges":      (
            "771754621 771731732 1159554013#0 135488882#0 780210981#0 "
            "780210980 137366598#0 135623338#0 771688494#0 135623660#0"
        ),
        "desc":             "Jalan Yap Ah Loy (3-lane 80 kph road, middle lane blocked)",
        "three_lane":       False,
    },
    {
        "suffix":           "acc3",
        "stop_lane":        "135623338#0_1",
        "start_pos":        30.0,
        "end_pos":          52.0,
        "route_edges":      (
            "771754621 771731732 1159554013#0 135488882#0 780210981#0 "
            "780210980 137366598#0 135623338#0 771688494#0 135623660#0"
        ),
        "desc":             "Jalan Petaling (2-lane, right lane blocked)",
        "three_lane":       False,
    },
    {
        "suffix":           "acc4",
        "stop_lane":        "28215290#0_1",
        "start_pos":        20.0,
        "end_pos":          43.0,
        "route_edges":      "771731732 1159554013#0 135488882#0 780210981#0 780210980 28215290#0 436198228#0",
        "desc":             "Jalan Tun Tan Cheng Lock seg-2 (2-lane, right lane blocked)",
        "three_lane":       False,
    },
]

# Phase A accident hardness control:
# Delay the static blocking vehicle insertion so the network first experiences
# normal flow, then a disruption shock mid-episode.
ACCIDENT_DEPART_TIME = 900.0

base_path = _SUMO_FILES / "osm.moderate.rou.xml"
with open(base_path, "r", encoding="utf-8") as fh:
    base_content = fh.read()

# Find insertion point: right after the last <vType .../> line, before the first <vehicle>
# vType tags are self-closing (e.g. <vType ... />)
last_vtype_start = base_content.rfind("<vType")
if last_vtype_start == -1:
    raise ValueError("No <vType> found in base route file")
close_pos = base_content.index("/>", last_vtype_start)
insert_pos = base_content.index("\n", close_pos) + 1


def _find_insert_pos_by_depart(content: str, depart_time: float, fallback_pos: int) -> int:
    """Return insertion index so <vehicle> blocks remain sorted by depart time."""
    vehicle_open_pattern = re.compile(r'(\n\s*<vehicle\b[^>]*\bdepart="([0-9.]+)"[^>]*>)')
    for match in vehicle_open_pattern.finditer(content):
        depart_str = match.group(2)
        try:
            depart_value = float(depart_str)
        except ValueError:
            continue
        if depart_value > depart_time:
            return match.start(1) + 1

    # If no later departure exists, append before closing </routes>.
    routes_close = content.rfind("</routes>")
    if routes_close != -1:
        return routes_close
    return fallback_pos


def _stop_xml(lane: str, start_pos: float, end_pos: float) -> str:
    return (
        f'        <stop lane="{lane}"'
        f' startPos="{start_pos:.1f}"'
        f' endPos="{end_pos:.1f}"'
        f' duration="99999" parking="false"/>\n'
    )


for acc in accidents:
    accident_block = (
        f'    <!-- ===== ACCIDENT VEHICLES: {acc["desc"]} ===== -->\n'
        f'    <vehicle id="accident_veh" depart="{ACCIDENT_DEPART_TIME:.2f}" type="truck"'
        f' departLane="best" departSpeed="0">\n'
        f'        <route edges="{acc["route_edges"]}"/>\n'
        + _stop_xml(acc["stop_lane"], acc["start_pos"], acc["end_pos"])
        + f'    </vehicle>\n'
    )
    if acc.get("three_lane"):
        accident_block += (
            f'    <vehicle id="accident_veh2" depart="1.0" type="passenger"'
            f' departLane="best" departSpeed="0">\n'
            f'        <route edges="{acc["route_edges"]}"/>\n'
            + _stop_xml(acc["second_stop_lane"], acc["start_pos"], acc["end_pos"])
            + f'    </vehicle>\n'
        )

    ordered_insert_pos = _find_insert_pos_by_depart(
        base_content,
        ACCIDENT_DEPART_TIME,
        insert_pos,
    )
    output = base_content[:ordered_insert_pos] + accident_block + base_content[ordered_insert_pos:]

    # Inject aggressive LC params into every vType definition in this output file.
    # The lambda strips trailing whitespace from the tag body before appending params.
    output = re.sub(
        r'(<vType\b[^>]*?)\s*/>',
        lambda m: m.group(1).rstrip() + _LC_PARAMS + ' />',
        output,
    )

    out_path = _SUMO_FILES / f'osm.{acc["suffix"]}.rou.xml'
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(output)
    print(f"Created {out_path}  ({len(output):,} bytes)")

#!/usr/bin/env python3
"""
generate_rainy_xml.py

Generates a rainy-weather SUMO route file from the base peak-hour route file.
Supports Curriculum Learning for the Rainy Weather Expert in the
Context-Aware Mixture of Experts architecture.

Callable API:
    from generate_rainy_xml import generate_rainy_route
    rain_condition = generate_rainy_route(num_vehicles=500,
                                          output_file="sumo_files/osm.rainy.rou.xml")

CLI:
    python generate_rainy_xml.py --num-vehicles 2000 --output-file sumo_files/osm.rainy.rou.xml
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_FILE = Path(__file__).parent.parent / "sumo_files" / "osm.peak.rou.xml"

# vType base attributes (including original accel) preserved from osm.peak.rou.xml
_VTYPE_BASE_ATTRS: dict[str, dict] = {
    "passenger": {
        "vClass": "passenger", "length": "4.5",
        "accel": "2.6",  "decel": "4.5",  "sigma": "0.5", "maxSpeed": "15.0",
        "color": "0,1,1",
    },
    "motorcycle": {
        "vClass": "motorcycle", "length": "2.0",
        "accel": "3.5",  "decel": "5.0",  "sigma": "0.7", "maxSpeed": "20.0",
        "minGap": "1.0", "color": "0,1,0",
    },
    "truck": {
        "vClass": "truck",       "length": "10.0",
        "accel": "1.0",  "decel": "2.5",  "sigma": "0.4", "maxSpeed": "10.0",
        "minGap": "3.5", "color": "0,0,1",
    },
    "bus": {
        "vClass": "bus",         "length": "12.0",
        "accel": "1.2",  "decel": "3.0",  "sigma": "0.4", "maxSpeed": "12.0",
        "minGap": "3.0", "color": "1,0,0",
    },
}

# Domain randomization presets
_RAIN_CONDITIONS: dict[str, dict] = {
    "Moderate Rain": {
        "accel_factor":  0.8,
        "decel_factor":  0.9,
        "speed_factor":  0.85,
        "tau_val":       "1.5",
        "sigma_val":     "0.65",
    },
    "Heavy Rain": {
        "accel_factor":  0.7,
        "decel_factor":  0.8,
        "speed_factor":  0.75,
        "tau_val":       "1.8",
        "sigma_val":     "0.75",
    },
}

# Traffic mix weights for rainy conditions
_MIX_TYPES:   list[str] = ["passenger", "motorcycle", "truck", "bus"]
_MIX_WEIGHTS: list[int] = [88,          2,            7,       3]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Sensible SUMO defaults used when a base attribute is absent
_ATTR_DEFAULTS: dict[str, float] = {
    "accel":    2.6,
    "decel":    3.0,
    "maxSpeed": 15.0,
}


def _build_vtype_elements(rain_name: str) -> list[ET.Element]:
    """Return a list of <vType> Elements for the chosen rain condition."""
    condition    = _RAIN_CONDITIONS[rain_name]
    accel_factor = condition["accel_factor"]
    decel_factor = condition["decel_factor"]
    speed_factor = condition["speed_factor"]
    tau_val      = condition["tau_val"]
    sigma_val    = condition["sigma_val"]

    elements: list[ET.Element] = []
    for vtype_id, base_attrs in _VTYPE_BASE_ATTRS.items():
        elem = ET.Element("vType")
        elem.set("id", vtype_id)
        for attr, value in base_attrs.items():
            elem.set(attr, value)

        # Apply weather degradation with fallback defaults
        orig_accel    = float(base_attrs.get("accel",    _ATTR_DEFAULTS["accel"]))
        orig_decel    = float(base_attrs.get("decel",    _ATTR_DEFAULTS["decel"]))
        orig_maxspeed = float(base_attrs.get("maxSpeed", _ATTR_DEFAULTS["maxSpeed"]))

        elem.set("accel",    f"{orig_accel    * accel_factor:.2f}")
        elem.set("decel",    f"{orig_decel    * decel_factor:.2f}")
        elem.set("maxSpeed", f"{orig_maxspeed * speed_factor:.2f}")
        elem.set("tau",      tau_val)
        elem.set("sigma",    sigma_val)
        elements.append(elem)

    return elements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_rainy_route(
    num_vehicles: int,
    output_file: str,
    rain_condition: str | None = None,
) -> str:
    """
    Generate a rainy-weather SUMO route XML file.

    Parameters
    ----------
    num_vehicles : int
        Number of vehicles to sample from the base file (max 1300).
    output_file : str
        Destination path for the generated XML file.
    rain_condition : str or None, optional
        Force a specific rain condition: "Moderate Rain" or "Heavy Rain".
        If None (default), the condition is chosen randomly.

    Returns
    -------
    str
        The rain condition used: "Moderate Rain" or "Heavy Rain".

    Raises
    ------
    ValueError
        If num_vehicles exceeds the number of vehicles in the base file,
        or if rain_condition is not a recognised condition key.
    """
    # 1. Parse base file and extract all <vehicle> nodes
    tree = ET.parse(BASE_FILE)
    root = tree.getroot()
    all_vehicles = root.findall("vehicle")

    if num_vehicles > len(all_vehicles):
        raise ValueError(
            f"Requested {num_vehicles} vehicles but the base file only contains "
            f"{len(all_vehicles)}."
        )

    # 2. Sample & Sort (CRITICAL: ascending depart time prevents SUMO temporal crashes)
    sampled = random.sample(all_vehicles, num_vehicles)
    sampled.sort(key=lambda v: float(v.get("depart", 0.0)))

    # 3. Domain Randomization: use provided condition or pick randomly
    if rain_condition is not None:
        if rain_condition not in _RAIN_CONDITIONS:
            raise ValueError(
                f"rain_condition must be one of {list(_RAIN_CONDITIONS.keys())}, "
                f"got {rain_condition!r}."
            )
        rain_name = rain_condition
    else:
        rain_name = random.choice(list(_RAIN_CONDITIONS.keys()))

    # 4. Traffic Mix Reassignment
    new_types = random.choices(_MIX_TYPES, weights=_MIX_WEIGHTS, k=num_vehicles)
    for vehicle, new_type in zip(sampled, new_types):
        vehicle.set("type", new_type)

    # 5. Build output tree: inject vType definitions, then append sorted vehicles
    out_root = ET.Element("routes")

    for vtype_elem in _build_vtype_elements(rain_name):
        out_root.append(vtype_elem)

    for vehicle in sampled:
        out_root.append(vehicle)

    # Pretty-print (requires Python >= 3.9)
    ET.indent(out_root, space="  ")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ET.ElementTree(out_root).write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(f"[generate_rainy_xml] Rain condition  : {rain_name}")
    print(f"[generate_rainy_xml] Vehicles written : {num_vehicles}")
    print(f"[generate_rainy_xml] Output file      : {output_path.resolve()}")

    return rain_name


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a rainy-weather SUMO route file for curriculum learning."
    )
    parser.add_argument(
        "--num-vehicles", type=int, default=2000,
        help="Number of vehicles to sample from the base file (default: 500).",
    )
    parser.add_argument(
        "--output-file", type=str,
        default=str(Path(__file__).parent.parent / "sumo_files" / "osm.rainy.rou.xml"),
        help="Output XML file path (default: ../sumo_files/osm.rainy.rou.xml).",
    )
    parser.add_argument(
        "--rain-condition", type=str, default=None,
        choices=list(_RAIN_CONDITIONS.keys()),
        help="Force a specific rain condition. Chosen randomly if omitted.",
    )
    args = parser.parse_args()

    generate_rainy_route(args.num_vehicles, args.output_file, args.rain_condition)

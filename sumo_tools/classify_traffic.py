#!/usr/bin/env python3
"""Redistribute SUMO vehicle types in a .rou.xml file.

This script preserves each vehicle's existing id, depart time, and nested route data,
while reassigning only the vehicle type according to a fixed weighted distribution.
"""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path


VTYPES = [
    {
        "id": "passenger",
        "vClass": "passenger",
        "length": "4.5",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "maxSpeed": "15.0",
        "color": "0,1,1",
    },
    {
        "id": "motorcycle",
        "vClass": "motorcycle",
        "length": "2.0",
        "accel": "3.5",
        "decel": "5.0",
        "sigma": "0.7",
        "maxSpeed": "20.0",
        "minGap": "1.0",
        "color": "0,1,0",
    },
    {
        "id": "truck",
        "vClass": "truck",
        "length": "10.0",
        "accel": "1.0",
        "decel": "2.5",
        "sigma": "0.4",
        "maxSpeed": "10.0",
        "minGap": "3.5",
        "color": "0,0,1",
    },
    {
        "id": "bus",
        "vClass": "bus",
        "length": "12.0",
        "accel": "1.2",
        "decel": "3.0",
        "sigma": "0.4",
        "maxSpeed": "12.0",
        "minGap": "3.0",
        "color": "1,0,0",
    },
]

TYPE_IDS = ["passenger", "motorcycle", "truck", "bus"]
TYPE_WEIGHTS = [0.70, 0.20, 0.07, 0.03]

# Edges that restrict vehicle types to light vehicles only (passenger/motorcycle)
LIGHT_VEHICLE_ONLY_EDGES = {"E0"}
LIGHT_TYPE_IDS = ["passenger", "motorcycle"]
LIGHT_TYPE_WEIGHTS = [0.70, 0.20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reassign SUMO vehicle types in a route file using fixed probabilities."
    )
    parser.add_argument("--input", required=True, help="Path to input .rou.xml file")
    parser.add_argument("--output", required=True, help="Path to output .rou.xml file")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible assignments",
    )
    return parser.parse_args()


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """In-place pretty printer for ElementTree output."""
    indent = "\n" + ("  " * level)
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def replace_vtype_definitions(root: ET.Element) -> None:
    """Remove existing top-level vType tags and insert required definitions first."""
    for child in list(root):
        if child.tag == "vType":
            root.remove(child)

    for index, attrs in enumerate(VTYPES):
        root.insert(index, ET.Element("vType", attrs))


def _route_uses_restricted_edge(vehicle: ET.Element) -> bool:
    """Return True if the vehicle's inline route passes through any light-vehicle-only edge."""
    route_elem = vehicle.find("route")
    if route_elem is None:
        return False
    edges = set(route_elem.get("edges", "").split())
    return bool(edges & LIGHT_VEHICLE_ONLY_EDGES)


def reassign_vehicle_types(root: ET.Element) -> int:
    """Assign each vehicle a new type using weighted random selection.

    Vehicles whose route passes through a restricted edge (e.g. E0) are
    limited to passenger/motorcycle only.
    """
    vehicles = root.findall("vehicle")
    total_vehicles = len(vehicles)
    if total_vehicles == 0:
        return 0

    for vehicle in vehicles:
        if _route_uses_restricted_edge(vehicle):
            vehicle_type = random.choices(LIGHT_TYPE_IDS, weights=LIGHT_TYPE_WEIGHTS, k=1)[0]
        else:
            vehicle_type = random.choices(TYPE_IDS, weights=TYPE_WEIGHTS, k=1)[0]
        vehicle.set("type", vehicle_type)

    return total_vehicles


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    tree = ET.parse(input_path)
    root = tree.getroot()

    total_vehicles = reassign_vehicle_types(root)
    replace_vtype_definitions(root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent_xml(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(f"Processed {total_vehicles} vehicles.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()

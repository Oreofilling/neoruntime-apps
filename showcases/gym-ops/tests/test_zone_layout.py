"""Config-layout invariants for the zone division in config.example.yaml."""
from itertools import combinations
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parent.parent / "config.example.yaml"
EPS = 1e-6


def _load():
    with open(CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _signed_area(poly):
    total = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) <= EPS:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4)
          - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4)
          - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return px, py


def _clip_convex(subject, clip):
    """Sutherland-Hodgman clip. Zones are kept convex by design."""
    if _signed_area(clip) < 0:
        clip = list(reversed(clip))

    output = list(subject)
    for i, edge_start in enumerate(clip):
        edge_end = clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break

        def inside(pt):
            return ((edge_end[0] - edge_start[0]) * (pt[1] - edge_start[1])
                    - (edge_end[1] - edge_start[1]) * (pt[0] - edge_start[0])) >= -EPS

        prev = input_list[-1]
        for cur in input_list:
            cur_inside = inside(cur)
            prev_inside = inside(prev)
            if cur_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, cur, edge_start, edge_end))
                output.append(cur)
            elif prev_inside:
                output.append(_line_intersection(prev, cur, edge_start, edge_end))
            prev = cur
    return output


def _overlap_area(a, b):
    clipped = _clip_convex(a, b)
    if len(clipped) < 3:
        return 0.0
    return abs(_signed_area(clipped))


def test_zone_polygons_stay_inside_frame():
    for z in _load()["zones"]:
        for x, y in z["polygon"]:
            assert -EPS <= x <= 1 + EPS, (z["id"], x)
            assert -EPS <= y <= 1 + EPS, (z["id"], y)


def test_zones_have_no_overlapping_interiors():
    zones = _load()["zones"]
    for a, b in combinations(zones, 2):
        area = _overlap_area(a["polygon"], b["polygon"])
        assert area <= EPS, (
            f"zones {a['id']} and {b['id']} overlap area={area:.6f}"
        )


def test_equipment_zone_refs_exist():
    cfg = _load()
    zone_ids = {z["id"] for z in cfg["zones"]}
    for e in cfg.get("equipment", []):
        assert e["zone_id"] in zone_ids, (
            f"equipment {e['id']} references unknown zone {e['zone_id']}"
        )


def test_zone_ids_unique():
    ids = [z["id"] for z in _load()["zones"]]
    assert len(ids) == len(set(ids))

"""Unit tests for zone palette assignment and overlay zone rendering.

Covers: ZONE_PALETTE cycling + config color override, forbidden flag
passthrough, and the draw_zones visual contract — per-zone tint fill, crowd
red override, forbidden amber style regardless of configured color, and the
rounded label badge anchored at the polygon bbox top-left.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import overlay  # noqa: E402
from zones import ZONE_PALETTE, Zone, ZoneManager  # noqa: E402

W, H = 1280, 720
GRAY = 128

# normalized polygons tiling a 16:9 frame (mirrors config.example.yaml)
TOP = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.6], [0.0, 0.6]]
RIGHT = [[0.5, 0.0], [1.0, 0.0], [1.0, 0.6], [0.5, 0.6]]
BOTTOM = [[0.0, 0.6], [1.0, 0.6], [1.0, 1.0], [0.0, 1.0]]


def make_manager(zone_cfgs: list[dict]) -> ZoneManager:
    return ZoneManager(zone_cfgs, [], occupancy_seconds=5,
                       long_occupation_seconds=1800)


def make_zone(zid: str, name: str, polygon: list, color: str = "",
              capacity: int = 0, forbidden: bool = False) -> Zone:
    return Zone(id=zid, name=name, polygon=polygon, capacity=capacity,
                color=color, forbidden=forbidden)


def render(zones: list[Zone], crowded=None, counts=None) -> np.ndarray:
    frame = np.full((H, W, 3), GRAY, dtype=np.uint8)
    overlay.draw_zones(frame, zones, crowded=crowded, counts=counts)
    return frame


# ---- palette assignment ----

def test_palette_assigned_by_index_and_cycles() -> None:
    n = len(ZONE_PALETTE) + 2
    mgr = make_manager([{"id": f"z{i}", "name": f"Z{i}",
                         "polygon": [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]]}
                        for i in range(n)])
    colors = [z.color for z in mgr.zones]
    assert colors[:len(ZONE_PALETTE)] == ZONE_PALETTE
    assert colors[len(ZONE_PALETTE)] == ZONE_PALETTE[0]  # wraps around
    assert colors[-1] == ZONE_PALETTE[1]


def test_config_color_overrides_palette() -> None:
    mgr = make_manager([{"id": "z1", "name": "Z1", "color": "#123456",
                         "polygon": [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]]}])
    assert mgr.zones[0].color == "#123456"


def test_invalid_color_falls_back_to_palette() -> None:
    mgr = make_manager([{"id": "z1", "name": "Z1", "color": "not-a-hex",
                         "polygon": [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]]}])
    assert mgr.zones[0].color == ZONE_PALETTE[0]


def test_forbidden_flag_passthrough() -> None:
    mgr = make_manager([
        {"id": "z1", "name": "Z1", "forbidden": True,
         "polygon": [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]]},
        {"id": "z2", "name": "Z2",
         "polygon": [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]]},
    ])
    assert mgr.zones[0].forbidden is True
    assert mgr.zones[1].forbidden is False


# ---- draw_zones rendering ----

def test_normal_zone_fill_tints_toward_zone_color() -> None:
    z = make_zone("z1", "Free Weight", TOP, color="#6ea8ff", capacity=8)
    frame = render([z], counts={"z1": 3})
    b, g, r = frame[300, 300]  # deep inside TOP, far from badge/border
    assert (b, g, r) != (GRAY, GRAY, GRAY)  # something drawn
    assert b > r                            # blue-tinted (#6ea8ff)


def test_crowded_zone_overridden_red() -> None:
    z = make_zone("z2", "Cardio", RIGHT, color="#3ecf8e", capacity=10)
    frame = render([z], crowded={"z2"}, counts={"z2": 12})
    b, g, r = frame[300, 900]  # inside RIGHT
    assert r > g > GRAY - 40   # red dominates, not the green base color


def test_forbidden_zone_rendered_amber_regardless_of_color() -> None:
    # violet base (#b48cff) would tint b > r; amber style must give r > b
    z = make_zone("z3", "Staff Only", BOTTOM, color="#b48cff", forbidden=True)
    frame = render([z])
    b, g, r = frame[600, 640]  # inside BOTTOM
    assert r > b


def test_badge_dark_chip_at_bbox_top_left() -> None:
    z = make_zone("z1", "Free Weight Area", TOP, color="#6ea8ff", capacity=8)
    frame = render([z], counts={"z1": 3})
    window = frame[8:52, 8:300]  # bbox top-left + inset region
    assert int(window.min()) < 80  # dark chip present


def test_draw_zones_without_counts_degrades_to_name_only() -> None:
    z = make_zone("z1", "Z", TOP, color="#6ea8ff")
    frame = render([z])  # counts=None — must not raise
    assert frame.shape == (H, W, 3)


def test_all_three_states_together_smoke() -> None:
    zones = [
        make_zone("z1", "Free Weight", TOP, capacity=8),
        make_zone("z2", "Cardio", RIGHT, capacity=10),
        make_zone("z3", "Staff Only", BOTTOM, forbidden=True),
    ]
    frame = render(zones, crowded={"z2"},
                   counts={"z1": 3, "z2": 12, "z3": 0})
    assert frame.shape == (H, W, 3)

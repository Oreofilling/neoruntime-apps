"""Unit tests for slots.py (per-slot occupancy state machine + grid mode)."""
import pytest

from slots import (
    EMPTY_CODE,
    SlotManager,
    build_grid_slots,
    classify,
    goods_summary,
    items_by_code,
    normalize_polygon,
)

POLY = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]  # left half
POLY2 = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]  # right half


def _slots_cfg() -> list[dict]:
    return [
        {"id": "s1", "name": "货架区1", "expected_code": "A", "capacity": 1,
         "polygon": POLY},
        {"id": "s2", "name": "货架区2", "expected_code": "E", "capacity": 1,
         "polygon": POLY2},
    ]


def test_classify_pure():
    assert classify(0, 1) == "EMPTY"
    assert classify(1, 1) == "FULL"
    assert classify(2, 1) == "FULL"
    assert classify(1, 0) == "PARTIAL"  # no capacity cap
    assert classify(1, 4) == "PARTIAL"


def test_normalize_polygon_validates():
    assert len(normalize_polygon(POLY)) == 4
    with pytest.raises(ValueError):   # out of [0,1]
        normalize_polygon([[0.0, 0.0], [1.2, 0.0], [0.5, 0.5]])
    with pytest.raises(ValueError):   # fewer than 3 vertices
        normalize_polygon([[0.0, 0.0], [1.0, 1.0]])


def test_apply_product_code_reads_full():
    mgr = SlotManager(_slots_cfg())
    snaps, trans = mgr.apply({"s1": {"code": "A", "score": 0.31}})
    assert trans == []                       # first tick seeds the baseline
    by_id = {s.slot_id: s for s in snaps}
    assert by_id["s1"].state == "FULL" and by_id["s1"].count == 1
    assert by_id["s1"].by_code == {"A": 1}
    assert by_id["s1"].code == "A" and by_id["s1"].score == pytest.approx(0.31)
    assert by_id["s2"].state == "EMPTY"      # no observation -> empty start


def test_apply_empty_code_reads_vacant():
    mgr = SlotManager(_slots_cfg())
    _, _ = mgr.apply({"s1": {"code": "B", "score": 0.4}})   # stocked baseline
    snaps, trans = mgr.apply({"s1": {"code": EMPTY_CODE, "score": 0.5}})
    by_id = {s.slot_id: s for s in snaps}
    assert by_id["s1"].state == "EMPTY" and by_id["s1"].count == 0
    assert by_id["s1"].by_code == {}
    assert any(t.prev == "FULL" and t.next == "EMPTY" for t in trans)
    assert by_id["s1"].code == EMPTY_CODE    # vacancy code passes through


def test_apply_missing_entry_holds_last_observation():
    """A dark/uncertain frame must not fire a stockout."""
    mgr = SlotManager(_slots_cfg())
    _, _ = mgr.apply({"s1": {"code": "A", "score": 0.3}})
    snaps, trans = mgr.apply({})             # s1 absent this tick
    by_id = {s.slot_id: s for s in snaps}
    assert by_id["s1"].state == "FULL"       # held
    assert by_id["s1"].by_code == {"A": 1}
    assert trans == []
    assert by_id["s1"].code == "" and by_id["s1"].score == 0.0


def test_apply_none_and_blank_code_hold():
    mgr = SlotManager(_slots_cfg())
    _, _ = mgr.apply({"s1": {"code": "A", "score": 0.3}})
    for per_slot in (None, {"s1": {}}):
        snaps, _ = mgr.apply(per_slot)
        assert {s.slot_id: s for s in snaps}["s1"].state == "FULL"


def test_transition_refill_cycle():
    mgr = SlotManager(_slots_cfg())
    mgr.apply({"s1": {"code": EMPTY_CODE, "score": 0.5}})   # baseline empty
    _, trans = mgr.apply({"s1": {"code": "A", "score": 0.3}})
    assert any(t.prev == "EMPTY" and t.next == "FULL" for t in trans)
    _, trans = mgr.apply({"s1": {"code": "A", "score": 0.31}})  # no change
    assert trans == []
    _, trans = mgr.apply({"s1": {"code": EMPTY_CODE, "score": 0.5}})
    assert any(t.prev == "FULL" and t.next == "EMPTY" for t in trans)


def test_tilted_flag():
    mgr = SlotManager(_slots_cfg())
    snaps, _ = mgr.apply({"s1": {"code": "E", "score": 0.9},   # E-item in A-slot
                          "s2": {"code": "E", "score": 0.9}})
    by_id = {s.slot_id: s for s in snaps}
    assert by_id["s1"].tilted == ["E"]
    assert by_id["s2"].tilted == []


def test_capacity_gt1_reads_partial_while_stocked():
    cfg = _slots_cfg()
    cfg[0]["capacity"] = 4
    mgr = SlotManager(cfg)
    snaps, _ = mgr.apply({"s1": {"code": "A", "score": 0.3}})
    assert {s.slot_id: s for s in snaps}["s1"].state == "PARTIAL"


def test_slot_lookup_and_empty_config():
    mgr = SlotManager(_slots_cfg())
    assert mgr.slot("s2") is not None
    assert mgr.slot("s9") is None
    snaps, trans = SlotManager([]).apply({"s1": {"code": "A", "score": 0.3}})
    assert snaps == [] and trans == []


def test_slot_requires_id():
    with pytest.raises(ValueError):
        SlotManager([{"id": "", "name": "x", "capacity": 1, "polygon": POLY}])


# ---- grid mode (path A: big region + virtual cell slots) ------------------

def test_build_grid_slots_shape():
    slots = build_grid_slots((0.0, 0.0, 1.0, 1.0), rows=3, cols=4)
    assert len(slots) == 12
    assert len({s["id"] for s in slots}) == 12          # unique ids
    for s in slots:
        assert s["capacity"] == 1
        assert s["expected_code"] == ""                 # no planogram in grid
        poly = s["polygon"]
        assert len(poly) == 4                           # axis-aligned rect
        for x, y in poly:
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_build_grid_slots_tile_region():
    """Cells must tile the region exactly: 2x2 over a quarter box."""
    slots = build_grid_slots((0.1, 0.2, 0.9, 0.8), rows=2, cols=2)
    rects = sorted(
        (s["polygon"][0][0], s["polygon"][0][1],        # (x1, y1)
         s["polygon"][2][0], s["polygon"][2][1])        # (x2, y2)
        for s in slots
    )
    assert rects == [(0.1, 0.2, 0.5, 0.5), (0.1, 0.5, 0.5, 0.8),
                     (0.5, 0.2, 0.9, 0.5), (0.5, 0.5, 0.9, 0.8)]


def test_build_grid_slots_validates():
    with pytest.raises(ValueError):                     # degenerate region
        build_grid_slots((0.5, 0.0, 0.5, 1.0), 2, 2)
    with pytest.raises(ValueError):                     # region outside [0,1]
        build_grid_slots((0.0, 0.0, 1.2, 1.0), 2, 2)
    with pytest.raises(ValueError):                     # zero rows
        build_grid_slots((0.0, 0.0, 1.0, 1.0), 0, 2)
    with pytest.raises(ValueError):                     # too many cells
        build_grid_slots((0.0, 0.0, 1.0, 1.0), 13, 13)


def test_grid_slots_flow_through_manager():
    """Grid cells are ordinary slots: stocked -> FULL, no planogram -> no tilt."""
    cfgs = build_grid_slots((0.0, 0.0, 1.0, 1.0), rows=1, cols=2)
    mgr = SlotManager(cfgs)
    snaps, _ = mgr.apply({cfgs[0]["id"]: {"code": "A", "score": 0.3}})
    by_id = {s.slot_id: s for s in snaps}
    assert by_id[cfgs[0]["id"]].state == "FULL"
    assert by_id[cfgs[0]["id"]].tilted == []            # expected_code "" -> no tilt
    assert by_id[cfgs[1]["id"]].state == "EMPTY"


def test_goods_summary_counts_occupied_cells():
    cfgs = build_grid_slots((0.0, 0.0, 1.0, 1.0), rows=2, cols=2)
    mgr = SlotManager(cfgs)
    ids = [c["id"] for c in cfgs]
    mgr.apply({ids[0]: {"code": "A", "score": 0.3},
               ids[1]: {"code": "D", "score": 0.28}})   # baseline
    snaps, _ = mgr.apply({ids[0]: {"code": "A", "score": 0.31},
                          ids[1]: {"code": "D", "score": 0.29},
                          ids[2]: {"code": EMPTY_CODE, "score": 0.4}})
    summary = goods_summary(snaps)
    assert summary["cells"] == 4
    assert summary["occupied"] == 2
    assert summary["by_code"] == {"A": 1, "D": 1}


def test_items_by_code_joins_detections_to_cell_codes():
    """件数 breakdown: each goods detection inherits the CLIP code of the
    grid cell it sits in; unnamed cells (EMPTY / missing) bucket 'other'."""
    cfgs = build_grid_slots((0.0, 0.0, 1.0, 1.0), rows=2, cols=2)
    mgr = SlotManager(cfgs)
    ids = [c["id"] for c in cfgs]
    mgr.apply({ids[0]: {"code": "D", "score": 0.3},    # bottled water cell
               ids[1]: {"code": "A", "score": 0.3},    # canned drink cell
               ids[3]: {"code": EMPTY_CODE, "score": 0.4}})  # unnamed cell
    snaps, _ = mgr.apply({ids[0]: {"code": "D", "score": 0.31},
                          ids[1]: {"code": "A", "score": 0.31},
                          ids[3]: {"code": EMPTY_CODE, "score": 0.41}})
    dets = [
        {"label": "bottle", "goods": True, "cell": ids[0]},
        {"label": "bottle", "goods": True, "cell": ids[0]},   # 2 items, same cell
        {"label": "can", "goods": True, "cell": ids[1]},
        {"label": "box", "goods": True, "cell": ids[3]},      # CLIP unnamed -> other
        {"label": "box", "goods": True, "cell": "g9-9"},      # no such cell -> other
        {"label": "hand", "goods": False, "cell": ids[0]},    # not goods
        {"label": "bottle", "goods": True, "cell": None},     # no cell: skipped
    ]
    assert items_by_code(dets, snaps) == {"D": 2, "A": 1, "other": 2}


def test_items_by_code_empty_inputs():
    assert items_by_code([], []) == {}


def test_grid_show_cells_false_hides_cell_rendering():
    """One-shelf view: show_cells=False drops cell borders/chips; the region
    rect and GOODS·ITEMS summary stay (render-only knob — counting cells)."""
    import numpy as np
    from overlay import CODE_COLORS, _GRID_EMPTY_COLOR, draw_slots

    cfgs = build_grid_slots((0.0, 0.0, 1.0, 1.0), rows=2, cols=2)
    mgr = SlotManager(cfgs)
    ids = [c["id"] for c in cfgs]
    snaps, _ = mgr.apply({ids[0]: {"code": "A", "score": 0.3},
                          ids[1]: {"code": "D", "score": 0.28}})

    def render(show):
        frame = np.zeros((240, 320, 3), np.uint8)
        draw_slots(frame, snaps, mgr.slots, grid_region=(0.0, 0.0, 1.0, 1.0),
                   items=2, show_cells=show)
        return frame

    # only colors the fixture draws: occupied cells read A and D, the other
    # two cells use the vacant/gray border (B/C/E never render here).
    drawn = (CODE_COLORS["A"], CODE_COLORS["D"], _GRID_EMPTY_COLOR)
    for color in drawn:
        shown = int((render(True) == color).all(axis=2).sum())
        hidden = int((render(False) == color).all(axis=2).sum())
        assert shown > 0, f"expected {color} cell borders when shown"
        assert hidden == 0, f"cell border {color} must vanish when hidden"

    # summary chip survives the hidden view (dark chip bg + light text)
    hidden = render(False)
    assert (hidden == (24, 26, 32)).all(axis=2).sum() > 50

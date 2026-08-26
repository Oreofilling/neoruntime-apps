"""Shelf slot occupancy state machine (slot-anchored classification edition).

Each configured slot polygon is classified independently (see clip_classify);
SlotManager.apply() turns per-slot classification results into the same
(snapshots, transitions) stream the analytics and alert layers already
consume. A slot holds at most one recognized product per tick (count is 0 or
1); per-item goods counting now lives in detector.py (yolo_world chain A,
`ITEMS m`) alongside this per-cell CLIP chain, so capacity > 1 slots still
read as PARTIAL while stocked — provision slots with capacity 1 for binary
FULL/EMPTY semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SLOT_STATES = ("EMPTY", "PARTIAL", "FULL")
EMPTY_CODE = "EMPTY"  # vocabulary code meaning "vacant slot"


def normalize_polygon(raw) -> list[tuple[float, float]]:
    """Coerce a config polygon into a list of normalized (x, y) tuples."""
    pts: list[tuple[float, float]] = []
    for p in raw or []:
        x, y = float(p[0]), float(p[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"polygon point out of [0,1]: {(x, y)}")
        pts.append((x, y))
    if len(pts) < 3:
        raise ValueError(f"polygon needs >= 3 vertices, got {len(pts)}")
    return pts


def classify(count: int, capacity: int) -> str:
    """Occupancy state from item count vs slot capacity."""
    if count <= 0:
        return "EMPTY"
    if capacity > 0 and count >= capacity:
        return "FULL"
    return "PARTIAL"


# Grid mode caps: cells are classified serially at ~48 ms each on device, so
# beyond this a scan round takes > ~7 s and the occupancy picture goes stale.
GRID_MAX_CELLS = 144


def build_grid_slots(region, rows: int, cols: int) -> list[dict]:
    """Generate virtual cell slots tiling ``region`` (path A: full-frame count).

    ``region`` is a normalized (x1, y1, x2, y2) box; ``rows`` x ``cols`` cells
    tile it exactly with no gap or overlap. Cells are ordinary slots (capacity
    1, no expected_code) so the whole CLIP scan / state-machine / analytics
    pipeline runs on them unchanged — grid counting = occupied-cell count.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in region)
    except (TypeError, ValueError) as e:
        raise ValueError(f"grid region must be 4 numbers: {region!r}") from e
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"grid region needs x2>x1 and y2>y1: {region!r}")
    if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
        raise ValueError(f"grid region outside [0,1]: {region!r}")
    rows, cols = int(rows), int(cols)
    if rows < 1 or cols < 1:
        raise ValueError(f"grid rows/cols must be >= 1: {rows}x{cols}")
    if rows * cols > GRID_MAX_CELLS:
        raise ValueError(
            f"grid {rows}x{cols} = {rows * cols} cells exceeds "
            f"GRID_MAX_CELLS={GRID_MAX_CELLS}")

    cw, ch = (x2 - x1) / cols, (y2 - y1) / rows
    slots: list[dict] = []
    for r in range(rows):
        for c in range(cols):
            cx1, cy1 = x1 + c * cw, y1 + r * ch
            slots.append({
                "id": f"g{r + 1}-{c + 1}",
                "name": f"Grid {r + 1}·{c + 1}",
                "expected_code": "",
                "capacity": 1,
                "polygon": [[cx1, cy1], [cx1 + cw, cy1],
                            [cx1 + cw, cy1 + ch], [cx1, cy1 + ch]],
            })
    return slots


def goods_summary(snapshots: list) -> dict:
    """Aggregate cell snapshots into the big-frame goods count.

    "occupied" = cells whose held observation is a product code (state FULL/
    PARTIAL); cells never confidently read hold EMPTY and do not count, so a
    dark frame cannot inflate the number. ``by_code`` counts occupied cells
    per recognized code.
    """
    by_code: dict[str, int] = {}
    occupied = 0
    for s in snapshots:
        if s.count <= 0:
            continue
        occupied += 1
        for code, n in (s.by_code or {}).items():
            by_code[code] = by_code.get(code, 0) + n
    return {"cells": len(snapshots), "occupied": occupied,
            "by_code": by_code}


def items_by_code(dets: list[dict], snapshots: list) -> dict[str, int]:
    """Per-category ITEM counts: join each per-item detection (detector.py)
    to the CLIP code held by the grid cell it sits in.

    Mirrors count_items() gating (goods detections with a cell only). An
    item inside a cell CLIP could not name (EMPTY / no snapshot) buckets
    under "other" — the detector saw goods the classifier could not."""
    code_by_slot = {s.slot_id: s.code for s in snapshots}
    out: dict[str, int] = {}
    for d in dets:
        if not (d.get("goods") and d.get("cell")):
            continue
        code = code_by_slot.get(d["cell"])
        code = code if code and code != EMPTY_CODE else "other"
        out[code] = out.get(code, 0) + 1
    return out


@dataclass
class ShelfSlot:
    id: str
    name: str
    expected_code: str
    capacity: int
    polygon: list[tuple[float, float]]


@dataclass
class SlotSnapshot:
    slot_id: str
    state: str
    count: int
    by_code: dict[str, int]
    expected_code: str
    capacity: int
    tilted: list[str] = field(default_factory=list)
    code: str = ""        # argmax vocabulary code for this tick ("" = unseen)
    score: float = 0.0    # cosine of that argmax match

    def ratio(self) -> float:
        if self.capacity <= 0:
            return 0.0 if self.count <= 0 else 1.0
        return min(1.0, self.count / self.capacity)


@dataclass
class SlotTransition:
    slot_id: str
    prev: str
    next: str
    count: int


class SlotManager:
    """Tracks per-slot occupancy across classification ticks."""

    def __init__(self, slots_cfg: list[dict] | None) -> None:
        self.slots: list[ShelfSlot] = []
        for cfg in slots_cfg or []:
            slot = ShelfSlot(
                id=str(cfg.get("id", "")),
                name=str(cfg.get("name", "")),
                expected_code=str(cfg.get("expected_code", "")).upper(),
                capacity=int(cfg.get("capacity", 1) or 0),
                polygon=normalize_polygon(cfg.get("polygon")),
            )
            if not slot.id:
                raise ValueError("slot entry needs a non-empty 'id'")
            self.slots.append(slot)
        self._prev_state: dict[str, str] = {}
        self._last_counts: dict[str, dict[str, int]] = {}

    def slot(self, slot_id: str) -> ShelfSlot | None:
        for s in self.slots:
            if s.id == slot_id:
                return s
        return None

    def apply(
        self, per_slot: dict[str, dict] | None
    ) -> tuple[list[SlotSnapshot], list[SlotTransition]]:
        """Compute snapshots + transitions from per-slot classification.

        ``per_slot``: slot_id -> {"code": str, "score": float}, the argmax
        vocabulary code for the slot crop and its cosine.

        - code is a product code      -> count 1 of that code
        - code == EMPTY_CODE          -> vacant, count 0
        - entry missing / code ""     -> no observation this tick (infer
          failed or below the cosine gate): the previous counts are held so
          one dark or uncertain frame cannot fire a stockout.
        """
        snapshots: list[SlotSnapshot] = []
        transitions: list[SlotTransition] = []
        for s in self.slots:
            res = (per_slot or {}).get(s.id) or {}
            code = str(res.get("code", "")).upper()
            score = float(res.get("score", 0.0))
            if code and code != EMPTY_CODE:
                counts: dict[str, int] = {code: 1}
            elif not code:
                counts = self._last_counts.get(s.id, {})
            else:
                counts = {}
            self._last_counts[s.id] = counts

            total = sum(counts.values())
            state = classify(total, s.capacity)
            prev = self._prev_state.get(s.id)
            if prev is None:
                self._prev_state[s.id] = state  # first tick seeds baseline
            elif state != prev:
                transitions.append(SlotTransition(s.id, prev, state, total))
                self._prev_state[s.id] = state

            snapshots.append(SlotSnapshot(
                slot_id=s.id, state=state, count=total,
                by_code=dict(counts), expected_code=s.expected_code,
                capacity=s.capacity,
                tilted=[c for c in counts
                        if s.expected_code and c != s.expected_code],
                code=code, score=score,
            ))
        return snapshots, transitions

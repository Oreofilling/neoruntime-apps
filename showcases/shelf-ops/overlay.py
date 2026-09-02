"""Overlay painting for the shelf preview (CLIP slots + per-item boxes).

Slot polygons are tinted by occupancy state; each slot's chip shows the
recognized vocabulary code (or EMPTY) with its cosine confidence. On top of
that, draw_detections() paints the per-item yolo_world boxes (detector.py):
goods bold in their item-category color (bottle/can/...) with label+score,
negative hits (person/hand/shelf) in thinner gray — plus a legend line when
both kinds are present.

Grid mode (path A) draws instead: one thick region rectangle, thin dim cell
borders, a code chip only on occupied cells, and a `GOODS n/N · ITEMS m`
summary at the region's top-left — the big frame IS the shelf, so empty
cells stay quiet. ITEMS m counts deduped goods detection boxes in-region
(chain A); GOODS n/N keeps its cell-CLIP semantics (chain B).
"""
from __future__ import annotations

import cv2
import numpy as np

from slots import EMPTY_CODE

# Product code -> BGR accent (chips / UI parity with static/js/app.js).
CODE_COLORS: dict[str, tuple[int, int, int]] = {
    "A": (76, 201, 78),
    "B": (71, 169, 238),
    "C": (82, 123, 253),
    "D": (103, 176, 232),
    "E": (209, 156, 76),
}
STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "EMPTY": (48, 48, 220),
    "PARTIAL": (60, 190, 250),
    "FULL": (76, 201, 86),
}
_CHIP_BG = (24, 26, 32)


def _code_color(code: str) -> tuple[int, int, int]:
    return CODE_COLORS.get(code, (160, 160, 160))


_GRID_REGION_COLOR = (235, 235, 235)   # big-frame rectangle (light)
_GRID_EMPTY_COLOR = (105, 105, 105)    # vacant / unseen cell border (dim)

# Per-detection boxes (detector.py): goods colored by their item category
# (bottle/can/...), negatives gray — the palette mirrors CATEGORY_COLOR in
# static/js/app.js so on-frame boxes match the UI chips.
DET_GOODS_COLOR = (76, 201, 86)          # fallback when category is unknown
DET_NEG_COLOR = (160, 160, 160)
CATEGORY_BGR: dict[str, tuple[int, int, int]] = {
    "bottle": (209, 156, 76),
    "can": (71, 169, 238),
    "carton": (82, 123, 253),
    "bag": (103, 176, 232),
    "jar": (209, 108, 156),
    "box": (76, 201, 78),
    "other": (167, 149, 139),
}
_DET_DARK = (16, 18, 24)                 # under-stroke so boxes read on any bg


def _chip(frame: np.ndarray, x: int, y: int, text: str, color,
          scale: float = 0.55, bold: bool = False) -> None:
    """Opaque-bg + text chip with its top-left at (x, y)."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(frame, (x, y), (x + tw + 10, y + th + baseline + 6),
                  _CHIP_BG, -1)
    cv2.putText(frame, text, (x + 5, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if bold else 1, cv2.LINE_AA)


def _draw_grid(frame: np.ndarray, snapshots: list, slots: list, region,
               items: int | None = None, show_cells: bool = True,
               highlight_empty: bool = False) -> None:
    """Path A rendering: thick region rect, per-cell borders, GOODS n/N.

    Occupied cells (held count > 0) get their code color + chip — the chip
    shows the *held* code (from by_code) so a missed read this tick doesn't
    blank the label; vacant / never-read cells stay thin and dim.
    show_cells=False is the whole-frame-as-one-shelf view: the region rect
    + summary stay, cell borders/chips are skipped (cells still count).
    highlight_empty (demo video mode) additionally marks cells CLIP has
    CONFIRMED empty (state EMPTY, count 0) with a thick red border + EMPTY
    chip — the visual hook for "this slot needs restocking"; never-read
    cells stay dim gray so confidence reads at a glance.
    """
    h, w = frame.shape[:2]
    rx1, ry1 = int(region[0] * w), int(region[1] * h)
    rx2, ry2 = int(region[2] * w), int(region[3] * h)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), _GRID_REGION_COLOR, 3)

    # summary chip rect first: when the region touches the top edge there is
    # no room above, and top-left cell chips must dodge below it.
    occupied_n = sum(1 for s in snapshots if s.count > 0)
    summary = f"GOODS {occupied_n}/{len(slots)}"
    if items is not None:
        summary += f" · ITEMS {items}"
    (sw, sh), sbase = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    sx, sy = rx1 + 2, max(0, ry1 - sh - sbase - 8)
    s_right, s_bottom = sx + sw + 10, sy + sh + sbase + 6

    by_id = {s.slot_id: s for s in snapshots}
    if show_cells:
        for slot in slots:
            snap = by_id.get(slot.id)
            occupied = snap is not None and snap.count > 0
            confirmed_empty = (highlight_empty and snap is not None
                               and snap.state == "EMPTY" and snap.count == 0)
            held = next(iter(snap.by_code), "") if occupied else ""
            poly = np.array([(p[0] * w, p[1] * h) for p in slot.polygon],
                            dtype=np.int32)
            if confirmed_empty:
                cv2.polylines(frame, [poly], True, STATE_COLORS["EMPTY"], 4)
            else:
                cv2.polylines(frame, [poly], True,
                              _code_color(held) if occupied
                              else _GRID_EMPTY_COLOR,
                              2 if occupied else 1)
            if occupied:
                label = f"{slot.id} {held}"
                if snap.code:                      # confident this tick -> cos
                    label += f" {snap.score:.2f}"
                cx, cy = int(poly[0][0]) + 2, int(poly[0][1]) + 2
                if cx < s_right and cy < s_bottom:
                    cy = s_bottom + 2              # dodge the summary chip
                _chip(frame, cx, cy, label, _code_color(held), 0.5)
            elif confirmed_empty:
                cx, cy = int(poly[0][0]) + 2, int(poly[0][1]) + 2
                if cx < s_right and cy < s_bottom:
                    cy = s_bottom + 2
                _chip(frame, cx, cy, f"{slot.id} EMPTY",
                      STATE_COLORS["EMPTY"], 0.5, bold=True)

    _chip(frame, sx, sy, summary, _GRID_REGION_COLOR, 0.8)


def draw_slots(frame: np.ndarray, snapshots: list, slots: list,
               grid_region=None, items: int | None = None,
               show_cells: bool = True,
               highlight_empty: bool = False) -> None:
    """Paint slot overlays; grid mode (path A) renders the big-frame grid.

    items (ITEMS m) extends the grid summary chip only; None keeps the
    plain `GOODS n/N` text. show_cells=False renders the region as one
    shelf (no cell borders/chips) — see _draw_grid. highlight_empty marks
    CLIP-confirmed empty cells red (demo video mode).
    """
    if grid_region is not None:
        _draw_grid(frame, snapshots, slots, grid_region, items=items,
                   show_cells=show_cells, highlight_empty=highlight_empty)
        return
    h, w = frame.shape[:2]
    by_id = {s.slot_id: s for s in snapshots}
    for slot in slots:
        snap = by_id.get(slot.id)
        state = snap.state if snap else "PARTIAL"
        code = (snap.code if snap else "") or ""
        poly = np.array([(p[0] * w, p[1] * h) for p in slot.polygon],
                        dtype=np.int32)
        is_empty = state == "EMPTY"
        cv2.polylines(frame, [poly], True,
                      STATE_COLORS.get(state, (160, 160, 160)),
                      5 if is_empty else 2)

        shown = code if code and code != EMPTY_CODE else EMPTY_CODE
        chip = f"{slot.id} {shown}"
        if snap and code:
            chip += f" {snap.score:.2f}"
        x, y = int(poly[0][0]), int(poly[0][1])
        (tw, th), baseline = cv2.getTextSize(
            chip, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y = max(0, y - th - baseline - 8)
        cv2.rectangle(frame, (x, y), (x + tw + 10, y + th + baseline + 6),
                      _CHIP_BG, -1)
        cv2.putText(frame, chip, (x + 5, y + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    _code_color(code if code != EMPTY_CODE else ""),
                    1, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, dets: list,
                    goods_only: bool = False) -> None:
    """Paint per-item detection boxes on top of the slot/grid layers.

    Boxes come from detector.py detect()/detect_tiled(): normalized
    [x1, y1, x2, y2] in original-frame coords. Goods boxes are drawn bold in
    their item-category color over a dark under-stroke (readable on any
    background) with a `label score [cell]` chip above the box; negative hits
    (person/hand/shelf) are thinner gray boxes — visible but not counted.
    Stroke/chip sizes scale with the frame resolution, and a mixed frame gets
    a one-line legend at the bottom so viewers know what is counted.
    goods_only=True (demo video mode) drops the negatives entirely so the
    video shows nothing but the counted item boxes.
    """
    h, w = frame.shape[:2]
    if goods_only:                 # demo view: shelf/price-tag hits stay hidden
        dets = [d for d in dets if d.get("goods")]
    s = max(1.0, min(2.2, min(h, w) / 640.0))    # boldness ~ frame size
    t_goods = max(3, round(2.5 * s))
    t_neg = max(2, round(1.1 * s))
    chip_scale = min(0.72, 0.55 * s)
    any_goods = any_neg = False
    # negatives first so counted goods boxes paint over any overlap
    for d in sorted(dets, key=lambda d: bool(d.get("goods"))):
        x1, y1, x2, y2 = d.get("box", (0, 0, 0, 0))
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        goods = bool(d.get("goods"))
        if goods:
            any_goods = True
            color = CATEGORY_BGR.get(str(d.get("category") or ""),
                                     DET_GOODS_COLOR)
            t = t_goods
        else:
            any_neg = True
            color = DET_NEG_COLOR
            t = t_neg
        cv2.rectangle(frame, (px1 - t, py1 - t), (px2 + t, py2 + t),
                      _DET_DARK, t + 2)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, t)
        label = f"{d.get('label', '?')} {float(d.get('score', 0)):.2f}"
        cell = d.get("cell")
        if cell:
            label += f" [{cell}]"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, chip_scale, 1)
        cy = py1 - th - baseline - 8
        if cy < 0:                       # box touches the top -> chip inside
            cy = py1 + 2
        cx = min(px1, w - tw - 12)
        _chip(frame, max(0, cx), cy, label, color, chip_scale, bold=True)

    if any_goods and any_neg:
        # bottom-center legend (corners are taken: summary top-left, mode
        # buttons top-right, live pill bottom-right, note bottom-left)
        text = "colored = counted item / gray = not counted"
        ls = min(0.58, 0.45 * s)
        (tw2, th2), b2 = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ls, 1)
        _chip(frame, max(0, (w - tw2 - 10) // 2), h - th2 - b2 - 14,
              text, (235, 235, 235), ls)

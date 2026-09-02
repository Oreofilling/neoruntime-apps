"""draw_detections visual polish: bold category-colored boxes + legend.

Boxes must be readable at a glance (the user-facing ask): goods strokes use
the item-category palette over a dark under-stroke, negatives stay thinner
gray, and a mixed frame carries a bottom legend explaining what is counted.
"""
import numpy as np

from overlay import (CATEGORY_BGR, _CHIP_BG, _DET_DARK, DET_GOODS_COLOR,
                     DET_NEG_COLOR, draw_detections)


def frame():
    # mid-gray background: neither the dark under-stroke nor any palette
    # color exists in it, so exact-match pixel counts are unambiguous
    return np.full((480, 640, 3), 120, np.uint8)


def px(frame, color):
    return int((frame == color).all(axis=2).sum())


GOOD_BOTTLE = {"label": "juice bottle", "goods": True, "score": 0.31,
               "category": "bottle", "cell": "g1-1",
               "box": (0.1, 0.1, 0.4, 0.5)}
NEG_PERSON = {"label": "person", "goods": False, "score": 0.28,
              "category": "person", "cell": None,
              "box": (0.6, 0.2, 0.9, 0.9)}


def test_goods_box_uses_category_color_with_dark_understroke():
    f = frame()
    draw_detections(f, [GOOD_BOTTLE])
    assert px(f, CATEGORY_BGR["bottle"]) > 0      # box stroke / chip text
    assert px(f, _DET_DARK) > 0                   # contrast under-stroke
    assert px(f, _CHIP_BG) > 0                    # label chip background


def test_goods_without_known_category_falls_back_to_green():
    f = frame()
    draw_detections(f, [{**GOOD_BOTTLE, "category": None}])
    assert px(f, DET_GOODS_COLOR) > 0


def test_negative_box_is_gray():
    f = frame()
    draw_detections(f, [NEG_PERSON])
    assert px(f, DET_NEG_COLOR) > 0
    assert px(f, DET_GOODS_COLOR) == 0


def test_goods_stroke_is_bolder_than_negative_stroke():
    fg, fn = frame(), frame()
    draw_detections(fg, [GOOD_BOTTLE])
    draw_detections(fn, [NEG_PERSON])
    # same box geometry -> the thicker goods stroke paints more edge pixels
    # (chip text adds some, but the margin covers that)
    assert px(fg, CATEGORY_BGR["bottle"]) > px(fn, DET_NEG_COLOR)


def test_mixed_frame_draws_counted_legend_goods_only_does_not():
    mixed, only = frame(), frame()
    draw_detections(mixed, [GOOD_BOTTLE, NEG_PERSON])
    draw_detections(only, [GOOD_BOTTLE])
    # legend text is white-ish; LINE_AA blends it, so count near-white
    legendish = lambda f: int((f >= 220).all(axis=2).sum())
    assert legendish(mixed) > 0
    assert legendish(only) == 0


def test_chip_inside_frame_when_box_touches_top():
    f = frame()
    draw_detections(f, [{**GOOD_BOTTLE, "box": (0.1, 0.0, 0.4, 0.5)}])
    # chip background exists and nothing was drawn above the frame
    assert px(f, _CHIP_BG) > 0
    assert f[0].max() > 0

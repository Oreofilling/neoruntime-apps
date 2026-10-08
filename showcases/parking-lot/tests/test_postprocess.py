"""Unit tests for parking-lot post-processing logic.

Tests anti-spoofing depth analysis, CTC decoder (incl. the real 18385-class
PaddleOCR v5 dictionary), plate validation, per-track temporal voting,
NMS parsing, and YOLO grid decoding with synthetic numpy arrays.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parking_lot.config import LPR_CHARSET, LPR_CTC_BLANK
from parking_lot.postprocess import (
    ctc_greedy_decode,
    PlateTracker,
    center_in_boxes,
    decode_recognition,
    iou_xywh,
    parse_nms_raw,
    parse_yolo_grid,
    iou,
    letterbox_crop,
    plate_crop_rects,
    validate_plate_text,
    analyze_depth_spoof,
    SpoofResult,
)


# ---------------------------------------------------------------------------
# Anti-spoofing depth analysis
# ---------------------------------------------------------------------------

class TestAnalyzeDepthSpoof:
    def _make_depth(self, h: int, w: int, value: float,
                    noise: float = 0.0) -> np.ndarray:
        base = np.full((h, w), value, dtype=np.float32)
        if noise > 0:
            rng = np.random.default_rng(42)
            base += rng.normal(0, noise, (h, w)).astype(np.float32)
        return base

    def test_uniform_depth_is_spoof(self) -> None:
        depth = self._make_depth(256, 320, 0.5)
        result = analyze_depth_spoof(depth, (0.2, 0.3, 0.4, 0.4), 256, 320)
        assert result.is_spoof is True
        assert result.depth_variance < 0.001
        assert result.confidence > 0.6

    def test_varied_depth_is_not_spoof(self) -> None:
        rng = np.random.default_rng(42)
        depth = rng.uniform(0.1, 0.9, (256, 320)).astype(np.float32)
        result = analyze_depth_spoof(depth, (0.2, 0.3, 0.4, 0.4), 256, 320)
        assert result.is_spoof is False
        assert result.depth_variance > 0.01

    def test_slight_noise_is_spoof(self) -> None:
        depth = self._make_depth(256, 320, 0.5, noise=0.005)
        result = analyze_depth_spoof(depth, (0.2, 0.3, 0.4, 0.4), 256, 320)
        assert result.is_spoof is True

    def test_moderate_noise_is_not_spoof(self) -> None:
        depth = self._make_depth(256, 320, 0.5, noise=0.1)
        result = analyze_depth_spoof(depth, (0.2, 0.3, 0.4, 0.4), 256, 320)
        assert result.is_spoof is False

    def test_empty_bbox_returns_not_spoof(self) -> None:
        depth = self._make_depth(256, 320, 0.5)
        result = analyze_depth_spoof(depth, (0.0, 0.0, 0.0, 0.0), 256, 320)
        assert result.is_spoof is False
        assert result.confidence == 0.0

    def test_different_depth_and_frame_size(self) -> None:
        depth = self._make_depth(128, 160, 0.5)
        result = analyze_depth_spoof(depth, (0.2, 0.3, 0.4, 0.4), 720, 1280)
        assert isinstance(result, SpoofResult)

    def test_custom_threshold(self) -> None:
        depth = self._make_depth(256, 320, 0.5, noise=0.05)
        # High threshold: anything under 0.5 variance is considered spoof
        high = analyze_depth_spoof(depth, (0.2, 0.3, 0.4, 0.4), 256, 320, threshold=0.5)
        # Very low threshold: only near-zero variance is spoof
        low = analyze_depth_spoof(depth, (0.2, 0.3, 0.4, 0.4), 256, 320, threshold=0.0001)
        assert high.is_spoof is True
        assert low.is_spoof is False


# ---------------------------------------------------------------------------
# CTC Greedy Decoder
# ---------------------------------------------------------------------------

class TestCTCGreedyDecode:
    CHARSET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-"

    def test_simple_decode(self) -> None:
        logits = np.zeros((5, len(self.CHARSET)), dtype=np.float32)
        for t, char in enumerate("A1B2C"):
            logits[t, self.CHARSET.index(char)] = 1.0
        text, conf = ctc_greedy_decode(logits, self.CHARSET)
        assert text == "A1B2C"
        assert conf > 0.9

    def test_blank_collapse(self) -> None:
        # CTC collapses consecutive repeats, not blank-separated repeats.
        # A_blank_A produces "AA" (two separate A emissions), not "A".
        logits = np.zeros((6, len(self.CHARSET)), dtype=np.float32)
        logits[0, self.CHARSET.index("A")] = 1.0
        logits[1, 0] = 1.0  # blank
        logits[2, self.CHARSET.index("A")] = 1.0  # separate emission -> "AA"
        logits[3, self.CHARSET.index("B")] = 1.0
        logits[4, 0] = 1.0
        logits[5, self.CHARSET.index("C")] = 1.0
        text, conf = ctc_greedy_decode(logits, self.CHARSET)
        assert text == "AABC"

    def test_consecutive_repeat_collapse(self) -> None:
        # Consecutive same-character emissions collapse into one
        logits = np.zeros((4, len(self.CHARSET)), dtype=np.float32)
        logits[0, self.CHARSET.index("A")] = 1.0
        logits[1, self.CHARSET.index("A")] = 1.0  # consecutive -> collapse
        logits[2, self.CHARSET.index("B")] = 1.0
        logits[3, self.CHARSET.index("C")] = 1.0
        text, conf = ctc_greedy_decode(logits, self.CHARSET)
        assert text == "ABC"

    def test_empty_input(self) -> None:
        logits = np.zeros((0, len(self.CHARSET)), dtype=np.float32)
        text, conf = ctc_greedy_decode(logits, self.CHARSET)
        assert text == ""
        assert conf == 0.0

    def test_all_blanks(self) -> None:
        logits = np.zeros((4, len(self.CHARSET)), dtype=np.float32)
        logits[:, 0] = 1.0
        text, conf = ctc_greedy_decode(logits, self.CHARSET)
        assert text == ""


# ---------------------------------------------------------------------------
# Plate text validation (structural sanity before a read can confirm)
# ---------------------------------------------------------------------------

class TestValidatePlateText:
    def test_standard_cn_plate(self) -> None:
        assert validate_plate_text("京A12345") is True

    def test_new_energy_8_char(self) -> None:
        assert validate_plate_text("粤B678901") is True

    def test_middot_and_spaces_stripped(self) -> None:
        assert validate_plate_text("京A·12345") is True
        assert validate_plate_text("京A 12345") is True

    def test_lenient_ascii_fallback(self) -> None:
        # No province char recognized — plain alphanumeric still passes.
        assert validate_plate_text("A12345") is True

    def test_reject_empty_and_question_marks(self) -> None:
        assert validate_plate_text("") is False
        assert validate_plate_text("京A?2345") is False

    def test_reject_too_short_and_too_long(self) -> None:
        assert validate_plate_text("京A12") is False    # 4 chars
        assert validate_plate_text("A123456789") is False  # 10 chars

    def test_reject_bad_body_after_province(self) -> None:
        # Second char must be an ascii letter, not a digit.
        assert validate_plate_text("京12345") is False

    def test_reject_non_alnum_body(self) -> None:
        assert validate_plate_text("京A1234!") is False


# ---------------------------------------------------------------------------
# Geometry helpers for candidate gating / tracking
# ---------------------------------------------------------------------------

class TestIouXywh:
    def test_perfect_overlap(self) -> None:
        assert iou_xywh((0.1, 0.1, 0.2, 0.1), (0.1, 0.1, 0.2, 0.1)) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert iou_xywh((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        v = iou_xywh((0.0, 0.0, 0.2, 0.1), (0.1, 0.0, 0.2, 0.1))
        assert 0.3 < v < 0.7


class TestCenterInBoxes:
    def test_inside(self) -> None:
        assert center_in_boxes(0.15, 0.15, [(0.1, 0.1, 0.2, 0.2)]) is True

    def test_outside_all(self) -> None:
        assert center_in_boxes(0.9, 0.9, [(0.1, 0.1, 0.2, 0.2)]) is False

    def test_empty_boxes(self) -> None:
        assert center_in_boxes(0.5, 0.5, []) is False


# ---------------------------------------------------------------------------
# PlateTracker — per-track IoU temporal voting (replaces the 1.2.4-era
# single global accumulator whose shared history cross-polluted plates)
# ---------------------------------------------------------------------------

class TestPlateTracker:
    @staticmethod
    def _tracker(**kw) -> PlateTracker:
        defaults = dict(
            iou_threshold=0.25, confirm_votes=2, vote_window=5,
            min_vote_conf=0.7, track_ttl=2.0, emit_ttl=30.0,
        )
        defaults.update(kw)
        return PlateTracker(**defaults)

    def test_two_plates_do_not_cross_pollute(self) -> None:
        """The headline 1.2.4 bug: global history returned plate A's text
        for plate B's box (and kept returning stale text on empty frames)."""
        tr = self._tracker()
        a, b = (0.10, 0.60, 0.12, 0.05), (0.60, 0.62, 0.12, 0.05)
        for t in range(3):
            states, _ = tr.update(
                [(a, "京A12345", 0.9), (b, "粤B67890", 0.9)], now=float(t))
        texts = {s.text for s in states}
        assert texts == {"京A12345", "粤B67890"}
        # Empty frame must NOT resurrect either plate (old accumulator did).
        # now jumps past track_ttl -> both tracks expire.
        states, events = tr.update([], now=10.0)
        assert states == [] and events == []

    def test_two_consistent_votes_confirm_and_emit(self) -> None:
        tr = self._tracker()
        box = (0.1, 0.1, 0.1, 0.05)
        states, events = tr.update([(box, "京A12345", 0.9)], now=0.0)
        assert states == [] and events == []  # 1 vote is not enough
        states, events = tr.update([(box, "京A12345", 0.9)], now=0.1)
        assert [s.text for s in states] == ["京A12345"]
        assert [e.text for e in events] == ["京A12345"]

    def test_low_confidence_votes_ignored(self) -> None:
        tr = self._tracker()
        box = (0.1, 0.1, 0.1, 0.05)
        for t in range(5):
            states, _ = tr.update([(box, "京A12345", 0.5)], now=float(t))
        assert states == []  # never confirmed without >=0.7 conf votes

    def test_invalid_text_never_confirms(self) -> None:
        tr = self._tracker()
        box = (0.1, 0.1, 0.1, 0.05)
        for t in range(5):
            states, _ = tr.update([(box, "AB", 0.99)], now=float(t))
        assert states == []  # fails validate_plate_text -> no confirm

    def test_track_expires_after_ttl(self) -> None:
        tr = self._tracker()
        box = (0.1, 0.1, 0.1, 0.05)
        tr.update([(box, "京A12345", 0.9)], now=0.0)
        states, _ = tr.update([(box, "京A12345", 0.9)], now=0.5)
        assert states  # confirmed while alive
        # > track_ttl (2.0s) since last_seen with no detection -> gone.
        states, _ = tr.update([], now=5.0)
        assert states == []

    def test_emit_ttl_gates_duplicate_events(self) -> None:
        tr = self._tracker()
        box = (0.1, 0.1, 0.1, 0.05)
        tr.update([(box, "京A12345", 0.9)], now=0.0)
        _, events = tr.update([(box, "京A12345", 0.9)], now=1.0)
        assert len(events) == 1           # confirmation emits once
        # Keep the track alive (1 s cadence << track_ttl) with the same
        # text: no re-emit until emit_ttl (30 s) since last_emit elapses.
        for t in range(2, 31):
            _, events = tr.update([(box, "京A12345", 0.9)], now=float(t))
            assert events == [], f"unexpected re-emit at t={t}"
        _, events = tr.update([(box, "京A12345", 0.9)], now=31.0)
        assert len(events) == 1           # 31-1 == 30 >= emit_ttl: refresh

    def test_corrected_read_reemits_immediately(self) -> None:
        tr = self._tracker()
        box = (0.1, 0.1, 0.1, 0.05)
        for t in range(5):
            tr.update([(box, "京A12345", 0.9)], now=float(t))
        # Window slides until the new votes win the majority; the frame the
        # confirmed text corrects must emit an event right away.
        emitted_correction = False
        for t in range(5, 10):
            states, events = tr.update([(box, "京A12346", 0.9)], now=float(t))
            if events:
                assert states[0].text == events[0].text == "京A12346"
                emitted_correction = True
        assert emitted_correction
        assert states and states[0].text == "京A12346"

    def test_same_box_new_track_after_gap(self) -> None:
        """A car leaves and another parks in the same spot: the stale
        track must not force the old plate onto the new one."""
        tr = self._tracker()
        box = (0.1, 0.1, 0.1, 0.05)
        tr.update([(box, "京A12345", 0.9)], now=0.0)
        tr.update([(box, "京A12345", 0.9)], now=0.5)
        # Gap > ttl, then a different plate in the same place.
        states, events = tr.update([(box, "沪C88888", 0.9)], now=10.0)
        assert states == []  # fresh track: single vote not confirmed
        states, _ = tr.update([(box, "沪C88888", 0.9)], now=10.5)
        assert [s.text for s in states] == ["沪C88888"]

    def test_track_ids_are_stable_and_distinct(self) -> None:
        tr = self._tracker()
        a, b = (0.10, 0.60, 0.12, 0.05), (0.60, 0.62, 0.12, 0.05)
        tr.update([(a, "京A12345", 0.9), (b, "粤B67890", 0.9)], now=0.0)
        states, _ = tr.update([(a, "京A12345", 0.9), (b, "粤B67890", 0.9)], now=0.1)
        ids = sorted(s.track_id for s in states)
        assert ids == [1, 2]
        # B disappears while A keeps being observed: once B is unseen past
        # track_ttl it is dropped, and A keeps its original id (IoU match,
        # not a fresh track).
        states = []
        for t in (0.2, 0.5, 1.0, 1.5, 2.0, 2.5):
            states, _ = tr.update([(a, "京A12345", 0.9)], now=float(t))
        assert [s.track_id for s in states] == [1]

    def test_reset_forgets_everything(self) -> None:
        tr = self._tracker()
        box = (0.1, 0.1, 0.1, 0.05)
        tr.update([(box, "京A12345", 0.9)], now=0.0)
        tr.update([(box, "京A12345", 0.9)], now=0.1)
        tr.reset()
        states, _ = tr.update([(box, "京A12345", 0.9)], now=0.2)
        assert states == []  # history wiped; needs 2 fresh votes again


# ---------------------------------------------------------------------------
# Real PaddleOCR v5 dictionary (18385 classes, blank=0 layout)
# ---------------------------------------------------------------------------

class TestRealDictCharset:
    def test_full_table_layout(self) -> None:
        # PaddleOCR CTCLabelDecode with use_space_char=True:
        # ["blank"] + dict(18383) + [" "] == 18385 entries, blank at 0.
        assert len(LPR_CHARSET) == 18385
        assert LPR_CTC_BLANK == 0
        assert LPR_CHARSET[0] == ""       # blank placeholder, never emitted
        assert LPR_CHARSET[-1] == " "
        assert LPR_CHARSET[1] == "　"  # dict's first char (ideographic space)

    def test_decode_cn_plate_from_full_table(self) -> None:
        text = "京A12345"
        idx = [LPR_CHARSET.index(ch) for ch in text]
        assert all(i > 0 for i in idx)  # no blanks in the path
        logits = np.full((len(text), len(LPR_CHARSET)), -1.0, dtype=np.float32)
        for t, i in enumerate(idx):
            logits[t, i] = 1.0
        out, conf = ctc_greedy_decode(logits, LPR_CHARSET, blank=LPR_CTC_BLANK)
        assert out == text
        assert conf > 0.9

    def test_flatten_path_num_classes_from_charset(self) -> None:
        """decode_recognition on a FLAT raw output: the reshape must use
        len(charset) (18385), not blank+1 — the 1.2.4 bug collapsed every
        timestep into one class column and decoded garbage."""
        text = "京A12345"
        idx = [LPR_CHARSET.index(ch) for ch in text]
        logits = np.full((len(text), len(LPR_CHARSET)), -1.0, dtype=np.float32)
        for t, i in enumerate(idx):
            logits[t, i] = 1.0

        class RawOnly:
            ocr_lines = []
            raw_outputs = [logits.flatten()]

        out, conf = decode_recognition(RawOnly(), LPR_CHARSET, blank=LPR_CTC_BLANK)
        assert out == text

    def test_ocr_lines_preferred_over_raw(self) -> None:
        class WithLines:
            class _Line:
                text = "沪B99999"
                confidence = 0.95
            ocr_lines = [_Line()]
            raw_outputs = [np.zeros((4, len(LPR_CHARSET)), dtype=np.float32)]

        out, conf = decode_recognition(WithLines(), LPR_CHARSET, blank=LPR_CTC_BLANK)
        assert (out, conf) == ("沪B99999", 0.95)


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

class TestIoU:
    def test_perfect_overlap(self) -> None:
        assert iou(0, 0, 1, 1, 0, 0, 1, 1) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert iou(0, 0, 1, 1, 2, 2, 1, 1) == pytest.approx(0.0)

    def test_half_overlap(self) -> None:
        result = iou(0, 0, 1, 1, 0.5, 0, 1, 1)
        assert 0.2 < result < 0.5

    def test_zero_area(self) -> None:
        assert iou(0, 0, 0, 0, 0, 0, 1, 1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Letterbox crop
# ---------------------------------------------------------------------------

class TestLetterboxCrop:
    def test_basic_crop_shape(self) -> None:
        bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        crop = letterbox_crop(bgr, 0.3, 0.3, 0.2, 0.2, 300, 75)
        assert crop.shape == (75, 300, 3)

    def test_edge_bbox(self) -> None:
        bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        crop = letterbox_crop(bgr, 0.0, 0.0, 0.1, 0.1, 300, 75)
        assert crop.shape == (75, 300, 3)

    def test_zero_bbox_returns_filled_canvas(self) -> None:
        bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        crop = letterbox_crop(bgr, 0.5, 0.5, 0.0, 0.0, 300, 75)
        assert crop.shape == (75, 300, 3)
        assert np.all(crop == 0)


# ---------------------------------------------------------------------------
# DSP plate-crop rects (mirror of letterbox_crop margins for multi_crop_hw)
# ---------------------------------------------------------------------------

class TestPlateCropRects:
    FRAME_W, FRAME_H = 1920, 1080
    TARGET_W, TARGET_H = 320, 48

    def test_matches_letterbox_crop_region(self) -> None:
        """rect region must equal letterbox_crop's crop window (±1px even-align)."""
        bbox = (0.30, 0.40, 0.20, 0.10)
        rect = plate_crop_rects(
            [bbox], self.FRAME_W, self.FRAME_H, self.TARGET_W, self.TARGET_H,
        )[0]
        assert rect is not None
        rx, ry, rw, rh, dw, dh = rect
        # letterbox_crop clamps in BGR (h, w) order; same math on the frame
        x, y, w, h = bbox
        mx = 0.10
        my = 0.35
        exp_x1 = max(0, int((x - w * mx) * self.FRAME_W))
        exp_y1 = max(0, int((y - h * my) * self.FRAME_H))
        exp_x2 = min(self.FRAME_W, int((x + w + w * mx) * self.FRAME_W))
        exp_y2 = min(self.FRAME_H, int((y + h + h * my) * self.FRAME_H))
        assert abs(rx - exp_x1) <= 1
        assert abs(ry - exp_y1) <= 1
        assert abs((rx + rw) - exp_x2) <= 1
        assert abs((ry + rh) - exp_y2) <= 1
        assert (dw, dh) == (self.TARGET_W, self.TARGET_H)

    def test_all_coordinates_even(self) -> None:
        bboxes = [(0.31, 0.43, 0.17, 0.09), (0.05, 0.05, 0.13, 0.07)]
        rects = plate_crop_rects(
            bboxes, self.FRAME_W, self.FRAME_H, self.TARGET_W, self.TARGET_H,
        )
        for rect in rects:
            assert rect is not None
            assert all(v % 2 == 0 for v in rect)

    def test_clamped_at_frame_edges(self) -> None:
        # Box straddling the top-left corner: region clamps to the frame.
        rect = plate_crop_rects(
            [(0.0, 0.0, 0.1, 0.1)], self.FRAME_W, self.FRAME_H,
            self.TARGET_W, self.TARGET_H,
        )[0]
        assert rect is not None
        x, y, w, h, _, _ = rect
        assert x == 0 and y == 0
        assert x + w <= self.FRAME_W and y + h <= self.FRAME_H

    def test_degenerate_box_maps_to_none(self) -> None:
        # Zero-size box after clamping cannot form a valid crop region.
        rects = plate_crop_rects(
            [(0.5, 0.5, 0.0, 0.0)], self.FRAME_W, self.FRAME_H,
            self.TARGET_W, self.TARGET_H,
        )
        assert rects == [None]

    def test_order_matches_input_and_mixed_validity(self) -> None:
        bboxes = [(0.3, 0.3, 0.2, 0.1), (0.5, 0.5, 0.0, 0.0), (0.6, 0.2, 0.1, 0.1)]
        rects = plate_crop_rects(
            bboxes, self.FRAME_W, self.FRAME_H, self.TARGET_W, self.TARGET_H,
        )
        assert len(rects) == 3
        assert rects[1] is None
        assert rects[0] is not None and rects[2] is not None

    def test_accepts_parse_yolo_grid_5_tuples(self) -> None:
        # Since 1.2.5 parse_yolo_grid returns (x, y, w, h, conf); the
        # trailing conf must be ignored by geometry, not unpacked.
        rect = plate_crop_rects(
            [(0.30, 0.40, 0.20, 0.10, 0.192)], self.FRAME_W, self.FRAME_H,
            self.TARGET_W, self.TARGET_H,
        )[0]
        assert rect is not None
        assert rect[4:] == (self.TARGET_W, self.TARGET_H)


# ---------------------------------------------------------------------------
# NMS raw parsing
# ---------------------------------------------------------------------------

class TestParseNmsRaw:
    def _mock_result(self, tensor: np.ndarray) -> object:
        class MockResult:
            raw_outputs = [tensor]
            objects = []
        return MockResult()

    def test_uint8_nms_format(self) -> None:
        header = np.array([2], dtype=np.float32)
        dets = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.9],
            [0.5, 0.5, 0.7, 0.8, 0.7],
        ], dtype=np.float32).flatten()
        buf = np.frombuffer(
            np.concatenate([header.view(np.uint8), dets.view(np.uint8)]),
            dtype=np.uint8,
        )
        result = self._mock_result(buf)
        vehicles = parse_nms_raw(result)
        assert len(vehicles) == 2
        assert vehicles[0].confidence == pytest.approx(0.9)

    def test_no_raw_outputs(self) -> None:
        class EmptyResult:
            raw_outputs = []
            objects = []
        assert parse_nms_raw(EmptyResult()) == []

    def test_low_confidence_filtered(self) -> None:
        header = np.array([1], dtype=np.float32)
        det = np.array([[0.1, 0.2, 0.3, 0.4, 0.1]], dtype=np.float32).flatten()
        buf = np.frombuffer(
            np.concatenate([header.view(np.uint8), det.view(np.uint8)]),
            dtype=np.uint8,
        )
        result = self._mock_result(buf)
        assert parse_nms_raw(result) == []


# ---------------------------------------------------------------------------
# YOLOv4 plate-grid decoding (client-side, license_plate_det)
# ---------------------------------------------------------------------------

class TestParseYoloGrid:
    """Regression coverage for the tiny_yolov4 raw-grid decoder.

    The conf gate was recalibrated 0.2 -> 0.12 after on-device measurement
    (93.72, 2026-09-03): the HEF's obj*cls tops out ~0.20 on ground-truth
    plates, so the old gate rejected every plate, synthetic or real.
    """

    @staticmethod
    def _make_raw_outputs(cells_by_grid):
        """Build the two raw tensors from {(grid_idx): [(gy,gx,a,vals)]}."""
        shapes = [(13, 13, 3, 6), (26, 26, 3, 6)]
        tensors = []
        for gi, shape in enumerate(shapes):
            t = np.zeros(shape, dtype=np.uint16)
            for gy, gx, a, vals in cells_by_grid.get(gi, []):
                t[gy, gx, a] = np.round(
                    np.asarray(vals, np.float32) * 65535.0
                ).astype(np.uint16)
            tensors.append(t.flatten())
        return tensors

    @staticmethod
    def _result(tensors):
        class GridResult:
            raw_outputs = tensors
            objects = []
        return GridResult()

    def test_measured_plate_signature_passes_gate(self) -> None:
        # obj=0.40 x cls=0.48 -> conf 0.192: the empirical on-device
        # signature of a real plate; rejected by the old 0.2 gate.
        tensors = self._make_raw_outputs({
            1: [(16, 10, 1, (0.5, 0.5, 0.5, 0.5, 0.40, 0.48))],
        })
        boxes = parse_yolo_grid(self._result(tensors), "license_plate_det")
        assert len(boxes) == 1
        # 5-tuples since 1.2.5: conf rides along for candidate gating.
        x, y, w, h, conf = boxes[0]
        assert conf == pytest.approx(0.40 * 0.48, abs=1e-3)
        # grid1 anchor 1 = (37, 58): w = 37*0.5/416, cx = 10.5/26.
        assert x == pytest.approx(10.5 / 26 - (37 * 0.5 / 416) / 2, abs=1e-3)
        assert y == pytest.approx(16.5 / 26 - (58 * 0.5 / 416) / 2, abs=1e-3)

    def test_boxes_carry_conf_sorted_desc(self) -> None:
        # Two non-overlapping cells: higher conf first, conf preserved per box.
        tensors = self._make_raw_outputs({
            0: [(6, 6, 0, (0.5, 0.5, 0.5, 0.5, 0.40, 0.60))],   # conf .24
            1: [(16, 10, 1, (0.5, 0.5, 0.5, 0.5, 0.40, 0.48))],  # conf .192
        })
        boxes = parse_yolo_grid(self._result(tensors), "license_plate_det")
        assert len(boxes) == 2
        assert boxes[0][4] > boxes[1][4]

    def test_background_obj_suppressed(self) -> None:
        # Below the obj>=0.3 pre-gate: never a detection, whatever cls.
        tensors = self._make_raw_outputs({
            0: [(6, 6, 0, (0.5, 0.5, 0.5, 0.5, 0.25, 0.95))],
            1: [(16, 10, 1, (0.5, 0.5, 0.5, 0.5, 0.25, 0.95))],
        })
        assert parse_yolo_grid(self._result(tensors), "license_plate_det") == []

    def test_weak_conf_suppressed(self) -> None:
        # obj passes but conf 0.07 < 0.12 gate.
        tensors = self._make_raw_outputs({
            1: [(16, 10, 1, (0.5, 0.5, 0.5, 0.5, 0.35, 0.20))],
        })
        assert parse_yolo_grid(self._result(tensors), "license_plate_det") == []

    def test_uint8_tensor_view_path(self) -> None:
        tensors = self._make_raw_outputs({
            1: [(16, 10, 1, (0.5, 0.5, 0.5, 0.5, 0.40, 0.48))],
        })
        as_u8 = [t.view(np.uint8) for t in tensors]
        boxes = parse_yolo_grid(self._result(as_u8), "license_plate_det")
        assert len(boxes) == 1

    def test_overlapping_detections_nms_keeps_one(self) -> None:
        # Same cell, anchors 1 and 2, tw/th chosen so the boxes nest with
        # IoU ~0.79 > 0.45 -> NMS keeps only the higher-conf one.
        tensors = self._make_raw_outputs({
            1: [
                (16, 10, 1, (0.5, 0.5, 1.0, 1.0, 0.40, 0.48)),
                (16, 10, 2, (0.5, 0.5, 0.55, 0.74, 0.45, 0.50)),
            ],
        })
        boxes = parse_yolo_grid(self._result(tensors), "license_plate_det")
        assert len(boxes) == 1

    def test_unknown_model_id_returns_empty(self) -> None:
        tensors = self._make_raw_outputs({
            1: [(16, 10, 1, (0.5, 0.5, 0.5, 0.5, 0.40, 0.48))],
        })
        assert parse_yolo_grid(self._result(tensors), "not_a_model") == []

    def test_missing_second_tensor_returns_empty(self) -> None:
        # The decoder's contract requires both scale tensors.
        tensors = self._make_raw_outputs({
            0: [(6, 6, 0, (0.5, 0.5, 0.5, 0.5, 0.40, 0.48))],
        })[:1]
        assert parse_yolo_grid(self._result(tensors), "license_plate_det") == []

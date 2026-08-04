"""Unit tests for parking-lot post-processing logic.

Tests anti-spoofing depth analysis, CTC decoder, OCR accumulator,
NMS parsing, and YOLO grid decoding with synthetic numpy arrays.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parking_lot.postprocess import (
    ctc_greedy_decode,
    PlateAccumulator,
    parse_nms_raw,
    iou,
    letterbox_crop,
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
# Plate Accumulator (temporal smoothing)
# ---------------------------------------------------------------------------

class TestPlateAccumulator:
    def test_single_read_passes_through(self) -> None:
        acc = PlateAccumulator(window_size=5, min_confidence=0.7)
        text, conf = acc.update("A12345", 0.9)
        assert text == "A12345"

    def test_majority_vote(self) -> None:
        acc = PlateAccumulator(window_size=5, min_confidence=0.7)
        acc.update("A12345", 0.8)
        acc.update("A12345", 0.85)
        acc.update("B67890", 0.9)
        acc.update("A12345", 0.75)
        text, conf = acc.update("A12345", 0.8)
        assert text == "A12345"

    def test_low_confidence_ignored(self) -> None:
        acc = PlateAccumulator(window_size=5, min_confidence=0.7)
        acc.update("A12345", 0.9)
        text, conf = acc.update("", 0.3)
        assert text == "A12345"

    def test_window_size_limit(self) -> None:
        acc = PlateAccumulator(window_size=3, min_confidence=0.7)
        acc.update("A11111", 0.9)
        acc.update("B22222", 0.9)
        acc.update("C33333", 0.9)
        acc.update("D44444", 0.9)
        text, conf = acc.update("D44444", 0.9)
        assert text == "D44444"


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

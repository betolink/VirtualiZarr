"""Tests for _is_regular_coord and _build_union_coord helpers."""

from __future__ import annotations

import numpy as np
import pytest

from virtualizarr.xarray import _build_union_coord, _is_regular_coord


class TestIsRegularCoord:
    def test_regular_ascending(self):
        arr = np.array([0.0, 120.0, 240.0, 360.0])
        assert _is_regular_coord(arr) is True

    def test_regular_descending(self):
        arr = np.array([360.0, 240.0, 120.0, 0.0])
        assert _is_regular_coord(arr) is True

    def test_single_element(self):
        assert _is_regular_coord(np.array([5.0])) is True

    def test_two_elements(self):
        assert _is_regular_coord(np.array([0.0, 10.0])) is True

    def test_irregular(self):
        arr = np.array([0.0, 120.0, 250.0, 360.0])  # third step is 130, not 120
        assert _is_regular_coord(arr) is False

    def test_all_same(self):
        # zero spacing - not useful but regular
        arr = np.array([5.0, 5.0, 5.0])
        assert _is_regular_coord(arr) is False  # zero step is not valid


class TestBuildUnionCoord:
    """Tests for _build_union_coord(arrays) -> np.ndarray"""

    def _make_coords(self, starts_ns, length, step=120.0):
        """Return a list of 1-D float64 coordinate arrays."""
        return [
            np.arange(s, s + length * step, step, dtype="float64")
            for s in starts_ns
        ]

    def test_non_overlapping_ascending(self):
        c1 = np.array([0.0, 120.0, 240.0])
        c2 = np.array([360.0, 480.0, 600.0])
        result = _build_union_coord([c1, c2])
        expected = np.arange(0.0, 720.0, 120.0)
        np.testing.assert_array_equal(result, expected)

    def test_non_overlapping_descending(self):
        c1 = np.array([600.0, 480.0, 360.0])
        c2 = np.array([240.0, 120.0, 0.0])
        result = _build_union_coord([c1, c2])
        expected = np.arange(600.0, -120.0, -120.0)
        np.testing.assert_array_equal(result, expected)

    def test_overlapping_coords(self):
        """Union of overlapping ranges should deduplicate."""
        c1 = np.array([0.0, 120.0, 240.0, 360.0])
        c2 = np.array([240.0, 360.0, 480.0, 600.0])
        result = _build_union_coord([c1, c2])
        expected = np.arange(0.0, 720.0, 120.0)
        np.testing.assert_array_equal(result, expected)

    def test_single_array(self):
        c = np.array([0.0, 120.0, 240.0])
        result = _build_union_coord([c])
        np.testing.assert_array_equal(result, c)

    def test_mismatched_spacing_raises(self):
        c1 = np.array([0.0, 120.0, 240.0])
        c2 = np.array([0.0, 240.0, 480.0])  # step=240 != 120
        with pytest.raises(ValueError, match="spacing"):
            _build_union_coord([c1, c2])

    def test_irregular_array_raises(self):
        c1 = np.array([0.0, 120.0, 240.0])
        c2 = np.array([360.0, 500.0, 620.0])  # irregular
        with pytest.raises(ValueError, match="regular"):
            _build_union_coord([c1, c2])

    def test_grid_misalignment_raises(self):
        """c2 starts at 60 — not on the 120-step grid that starts at 0."""
        c1 = np.array([0.0, 120.0, 240.0])
        c2 = np.array([60.0, 180.0, 300.0])
        with pytest.raises(ValueError, match="align"):
            _build_union_coord([c1, c2])

    def test_itslive_like(self):
        """Two ITS_LIVE-like tiles, 120 m grid, partially overlapping."""
        # g1: x from -122828 to some positive value, step=120
        g1_x = np.arange(-122828.0, -122828.0 + 2048 * 120, 120.0)
        # g2: x from -307148, step=120, length=2560
        g2_x = np.arange(-307148.0, -307148.0 + 2560 * 120, 120.0)
        result = _build_union_coord([g1_x, g2_x])
        union_min = min(g1_x[0], g2_x[0])
        union_max = max(g1_x[-1], g2_x[-1])
        assert result[0] == union_min
        assert result[-1] == union_max
        # spacing preserved
        assert np.allclose(np.diff(result), 120.0)

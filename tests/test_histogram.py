"""Tests for gelsa.histogram — serial and parallel variants."""
import numpy as np
import pytest
from numba import typed, types

from gelsa.histogram import (
    _histogram1d,
    _histogram2d,
    _histogram2d_accumulate,
    _histogram2d_accumulate_mask,
    histogram1d,
    histogram2d,
    histogram2d_accumulate,
    histogram2d_accumulate_mask,
)

RNG = np.random.default_rng(0)

BINS_Y = np.arange(0, 11, dtype=np.float64)   # 10 bins [0, 10)
BINS_X = np.arange(0, 11, dtype=np.float64)
BINS_1D = np.arange(0, 11, dtype=np.float64)

# Small (exercises serial fallback) and large (exercises parallel path)
N_SMALL = 500
N_LARGE = 50_000


@pytest.fixture(scope="module", autouse=True)
def warmup_jit():
    """Compile all JIT functions once before any test runs."""
    y = np.array([0.5], dtype=np.float64)
    x = np.array([0.5], dtype=np.float64)
    w = np.array([1.0], dtype=np.float64)
    _histogram2d(y, x, w, BINS_Y, BINS_X)
    histogram2d(y, x, w, BINS_Y, BINS_X)
    _histogram1d(x, w, BINS_1D)
    histogram1d(x, w, BINS_1D)
    h = np.zeros((10, 10), dtype=np.float64)
    _histogram2d_accumulate(y, x, w, BINS_Y, BINS_X, h)
    h2 = np.zeros((10, 10), dtype=np.float64)
    histogram2d_accumulate(y, x, w, BINS_Y, BINS_X, h2)
    md = _make_mask_dict(1, 1)
    h3 = np.zeros(1, dtype=np.float64)
    b = np.array([0.0, 1.0])
    _histogram2d_accumulate_mask(y[:1], x[:1], w[:1], b, b, h3, md)
    h4 = np.zeros(1, dtype=np.float64)
    histogram2d_accumulate_mask(y[:1], x[:1], w[:1], b, b, h4, md)


def _make_2d(n):
    y = RNG.uniform(0, 10, n).astype(np.float64)
    x = RNG.uniform(0, 10, n).astype(np.float64)
    w = RNG.uniform(0.5, 1.5, n).astype(np.float64)
    return y, x, w


def _make_1d(n):
    x = RNG.uniform(0, 10, n).astype(np.float64)
    w = RNG.uniform(0.5, 1.5, n).astype(np.float64)
    return x, w


def _np2d(y, x, w):
    ref, _, _ = np.histogram2d(y, x, bins=[BINS_Y, BINS_X], weights=w)
    return ref


def _np1d(x, w):
    ref, _ = np.histogram(x, bins=BINS_1D, weights=w)
    return ref


# ---------------------------------------------------------------------------
# _histogram2d (serial)
# ---------------------------------------------------------------------------

class TestHistogram2dSerial:
    def test_against_numpy_small(self):
        y, x, w = _make_2d(N_SMALL)
        np.testing.assert_allclose(_histogram2d(y, x, w, BINS_Y, BINS_X), _np2d(y, x, w))

    def test_against_numpy_large(self):
        y, x, w = _make_2d(N_LARGE)
        np.testing.assert_allclose(_histogram2d(y, x, w, BINS_Y, BINS_X), _np2d(y, x, w))

    def test_out_of_range_ignored(self):
        y = np.array([-1.0, 11.0, 5.0], dtype=np.float64)
        x = np.array([5.0,   5.0, 5.0], dtype=np.float64)
        w = np.ones(3, dtype=np.float64)
        result = _histogram2d(y, x, w, BINS_Y, BINS_X)
        assert result.sum() == pytest.approx(1.0)

    def test_unit_weight_sums_to_n(self):
        y, x, _ = _make_2d(N_SMALL)
        w = np.ones(N_SMALL, dtype=np.float64)
        assert _histogram2d(y, x, w, BINS_Y, BINS_X).sum() == pytest.approx(N_SMALL)

    def test_output_shape(self):
        y, x, w = _make_2d(10)
        result = _histogram2d(y, x, w, BINS_Y, BINS_X)
        assert result.shape == (10, 10)


# ---------------------------------------------------------------------------
# histogram2d (parallel)
# ---------------------------------------------------------------------------

class TestHistogram2dParallel:
    def test_matches_serial_small(self):
        y, x, w = _make_2d(N_SMALL)
        np.testing.assert_allclose(
            histogram2d(y, x, w, BINS_Y, BINS_X),
            _histogram2d(y, x, w, BINS_Y, BINS_X),
        )

    def test_matches_serial_large(self):
        y, x, w = _make_2d(N_LARGE)
        np.testing.assert_allclose(
            histogram2d(y, x, w, BINS_Y, BINS_X),
            _histogram2d(y, x, w, BINS_Y, BINS_X),
        )

    def test_against_numpy_large(self):
        y, x, w = _make_2d(N_LARGE)
        np.testing.assert_allclose(histogram2d(y, x, w, BINS_Y, BINS_X), _np2d(y, x, w))


# ---------------------------------------------------------------------------
# _histogram1d (serial)
# ---------------------------------------------------------------------------

class TestHistogram1dSerial:
    def test_against_numpy_small(self):
        x, w = _make_1d(N_SMALL)
        np.testing.assert_allclose(_histogram1d(x, w, BINS_1D), _np1d(x, w))

    def test_against_numpy_large(self):
        x, w = _make_1d(N_LARGE)
        np.testing.assert_allclose(_histogram1d(x, w, BINS_1D), _np1d(x, w))

    def test_out_of_range_ignored(self):
        x = np.array([-1.0, 11.0, 5.0], dtype=np.float64)
        w = np.ones(3, dtype=np.float64)
        result = _histogram1d(x, w, BINS_1D)
        assert result.sum() == pytest.approx(1.0)

    def test_output_shape(self):
        x, w = _make_1d(10)
        assert _histogram1d(x, w, BINS_1D).shape == (10,)


# ---------------------------------------------------------------------------
# histogram1d (parallel)
# ---------------------------------------------------------------------------

class TestHistogram1dParallel:
    def test_matches_serial_small(self):
        x, w = _make_1d(N_SMALL)
        np.testing.assert_allclose(
            histogram1d(x, w, BINS_1D),
            _histogram1d(x, w, BINS_1D),
        )

    def test_matches_serial_large(self):
        x, w = _make_1d(N_LARGE)
        np.testing.assert_allclose(
            histogram1d(x, w, BINS_1D),
            _histogram1d(x, w, BINS_1D),
        )

    def test_against_numpy_large(self):
        x, w = _make_1d(N_LARGE)
        np.testing.assert_allclose(histogram1d(x, w, BINS_1D), _np1d(x, w))


# ---------------------------------------------------------------------------
# histogram2d_accumulate (serial and parallel)
# ---------------------------------------------------------------------------

def _make_mask_dict(ny, nx):
    """Map every (iy, ix) bin to flat index iy*nx + ix.  hist must be 1D of size ny*nx."""
    d = typed.Dict.empty(
        key_type=types.UniTuple(types.int64, 2),
        value_type=types.int64,
    )
    for iy in range(ny):
        for ix in range(nx):
            d[(iy, ix)] = iy * nx + ix
    return d


class TestHistogram2dAccumulateSerial:
    def test_no_mask_matches_numpy(self):
        y, x, w = _make_2d(N_SMALL)
        hist = np.zeros((10, 10), dtype=np.float64)
        _histogram2d_accumulate(y, x, w, BINS_Y, BINS_X, hist)
        np.testing.assert_allclose(hist, _np2d(y, x, w))

    def test_no_mask_accumulates(self):
        """Calling twice should double the result."""
        y, x, w = _make_2d(N_SMALL)
        hist = np.zeros((10, 10), dtype=np.float64)
        _histogram2d_accumulate(y, x, w, BINS_Y, BINS_X, hist)
        _histogram2d_accumulate(y, x, w, BINS_Y, BINS_X, hist)
        np.testing.assert_allclose(hist, 2 * _np2d(y, x, w))

    def test_empty_input_leaves_hist_unchanged(self):
        y = np.empty(0, dtype=np.float64)
        x = np.empty(0, dtype=np.float64)
        w = np.empty(0, dtype=np.float64)
        hist = np.ones((10, 10), dtype=np.float64)
        expected = hist.copy()
        _histogram2d_accumulate(y, x, w, BINS_Y, BINS_X, hist)
        np.testing.assert_array_equal(hist, expected)

    def test_with_mask_dict(self):
        """Points land in the 1D hist at the index given by mask_dict."""
        bins = np.arange(0, 4, dtype=np.float64)  # 3 bins per axis
        y = np.array([0.5, 1.5, 2.5], dtype=np.float64)
        x = np.array([0.5, 1.5, 2.5], dtype=np.float64)
        w = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        mask_dict = _make_mask_dict(3, 3)
        hist = np.zeros(9, dtype=np.float64)  # 1D, size ny*nx
        _histogram2d_accumulate_mask(y, x, w, bins, bins, hist, mask_dict)

        assert hist[0] == pytest.approx(1.0)  # bin (0,0) -> j=0
        assert hist[4] == pytest.approx(2.0)  # bin (1,1) -> j=4
        assert hist[8] == pytest.approx(3.0)  # bin (2,2) -> j=8

    def test_with_mask_dict_minus1_skips_bin(self):
        """Bins mapped to j=-1 must be excluded from the output."""
        bins = np.arange(0, 3, dtype=np.float64)  # 2 bins per axis
        d = typed.Dict.empty(
            key_type=types.UniTuple(types.int64, 2),
            value_type=types.int64,
        )
        d[(0, 0)] = 0    # keep
        d[(0, 1)] = -1   # skip
        d[(1, 0)] = 1    # keep
        d[(1, 1)] = -1   # skip

        y = np.array([0.5, 0.5, 1.5, 1.5], dtype=np.float64)
        x = np.array([0.5, 1.5, 0.5, 1.5], dtype=np.float64)
        w = np.ones(4, dtype=np.float64)

        hist = np.zeros(2, dtype=np.float64)  # 1D, only 2 kept bins
        _histogram2d_accumulate_mask(y, x, w, bins, bins, hist, d)
        assert hist[0] == pytest.approx(1.0)  # bin (0,0)
        assert hist[1] == pytest.approx(1.0)  # bin (1,0)


class TestHistogram2dAccumulateParallel:
    def test_no_mask_matches_serial_small(self):
        y, x, w = _make_2d(N_SMALL)
        h_serial   = np.zeros((10, 10), dtype=np.float64)
        h_parallel = np.zeros((10, 10), dtype=np.float64)
        _histogram2d_accumulate(y, x, w, BINS_Y, BINS_X, h_serial)
        histogram2d_accumulate( y, x, w, BINS_Y, BINS_X, h_parallel)
        np.testing.assert_allclose(h_parallel, h_serial)

    def test_no_mask_matches_serial_large(self):
        y, x, w = _make_2d(N_LARGE)
        h_serial   = np.zeros((10, 10), dtype=np.float64)
        h_parallel = np.zeros((10, 10), dtype=np.float64)
        _histogram2d_accumulate(y, x, w, BINS_Y, BINS_X, h_serial)
        histogram2d_accumulate( y, x, w, BINS_Y, BINS_X, h_parallel)
        np.testing.assert_allclose(h_parallel, h_serial)

    def test_with_mask_dict_matches_serial(self):
        bins = np.arange(0, 6, dtype=np.float64)  # 5 bins per axis
        y, x, w = _make_2d(N_LARGE)
        y = (y % 5).astype(np.float64)
        x = (x % 5).astype(np.float64)

        mask_dict = _make_mask_dict(5, 5)
        h_serial   = np.zeros(25, dtype=np.float64)  # 1D, size ny*nx
        h_parallel = np.zeros(25, dtype=np.float64)
        _histogram2d_accumulate_mask(y, x, w, bins, bins, h_serial,   mask_dict)
        histogram2d_accumulate_mask( y, x, w, bins, bins, h_parallel, mask_dict)
        np.testing.assert_allclose(h_parallel, h_serial)

    def test_empty_input_leaves_hist_unchanged(self):
        y = np.empty(0, dtype=np.float64)
        x = np.empty(0, dtype=np.float64)
        w = np.empty(0, dtype=np.float64)
        hist = np.ones((10, 10), dtype=np.float64)
        expected = hist.copy()
        histogram2d_accumulate(y, x, w, BINS_Y, BINS_X, hist)
        np.testing.assert_array_equal(hist, expected)

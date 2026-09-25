"""Integration tests for SpecFrame coordinate transforms.

These tests require the SGS calibration files and are skipped automatically
when they are not available (CI without calib data, fresh checkouts, etc.).
"""
import numpy as np
import pytest
from .conftest import needs_calib


RNG = np.random.default_rng(42)
N = 500
RA0, DEC0 = 10.0, 0.0

RA         = RNG.uniform(RA0 - 1, RA0 + 1, N)
DEC        = RNG.uniform(DEC0 - 1, DEC0 + 1, N)
WAVELENGTH = RNG.uniform(12500, 18500, N)


@needs_calib
class TestRadecToPixel:
    def test_returns_three_arrays(self, gelsa_frame):
        _, frame = gelsa_frame
        x, y, det = frame.radec_to_pixel(RA, DEC, WAVELENGTH)
        assert len(x) == N
        assert len(y) == N
        assert len(det) == N

    def test_on_detector_pixels_in_range(self, gelsa_frame):
        _, frame = gelsa_frame
        x, y, det = frame.radec_to_pixel(RA, DEC, WAVELENGTH)
        det = np.asarray(det, dtype=int)
        on = det >= 0
        assert on.sum() > 0, "expected some points to land on a detector"
        assert np.all(x[on] >= 0) and np.all(x[on] < 2048)
        assert np.all(y[on] >= 0) and np.all(y[on] < 2048)

    def test_deterministic(self, gelsa_frame):
        _, frame = gelsa_frame
        x1, y1, det1 = frame.radec_to_pixel(RA, DEC, WAVELENGTH)
        x2, y2, det2 = frame.radec_to_pixel(RA, DEC, WAVELENGTH)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(det1, det2)


@needs_calib
class TestRoundtrip:
    def test_pixel_to_radec_recovers_input(self, gelsa_frame):
        _, frame = gelsa_frame
        x, y, det = frame.radec_to_pixel(RA, DEC, WAVELENGTH)
        det = np.asarray(det, dtype=int)
        on = det >= 0

        ra_rt, dec_rt = frame.pixel_to_radec(x[on], y[on], det[on], WAVELENGTH[on])

        dra  = np.abs((ra_rt - RA[on] + 180) % 360 - 180) * 3600
        ddec = np.abs(dec_rt - DEC[on]) * 3600
        assert np.max(dra)  < 0.01, f"max RA  error {np.max(dra):.4f} arcsec"
        assert np.max(ddec) < 0.01, f"max Dec error {np.max(ddec):.4f} arcsec"

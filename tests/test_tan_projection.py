"""Tests for the TAN projection Numba kernels."""
import numpy as np
import pytest
from astropy.wcs import WCS
from gelsa.sgs.tan_projection import _tan_projection_jit, _tan_projection_inverse_jit


RNG = np.random.default_rng(42)

RA0, DEC0 = 10.0, 45.0
PLATE_SCALE = 0.3 / 3600.0
CD = np.array([[-PLATE_SCALE, 0.0], [0.0, PLATE_SCALE]])
CDINV = np.linalg.inv(CD)
CRPIX0, CRPIX1 = 1024.5, 1024.5


def _astropy_world2pix(ra, dec, ra0, dec0, cdinv, crpix0, crpix1):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "RA---TAN", "DEC--TAN"
    wcs.wcs.crval = [ra0, dec0]
    wcs.wcs.crpix = [crpix0, crpix1]
    wcs.wcs.cd = np.linalg.inv(cdinv)
    return wcs.all_world2pix(ra, dec, 0)


@pytest.fixture(scope="module", autouse=True)
def warmup_jit():
    """Trigger Numba JIT compilation once for the whole module."""
    ra = RNG.uniform(RA0 - 0.1, RA0 + 0.1, 10)
    dec = RNG.uniform(DEC0 - 0.1, DEC0 + 0.1, 10)
    _tan_projection_jit(ra, dec, RA0, DEC0, CDINV, CRPIX0, CRPIX1)
    x = RNG.uniform(900, 1100, 10)
    y = RNG.uniform(900, 1100, 10)
    _tan_projection_inverse_jit(x, y, RA0, DEC0, CD, CRPIX0, CRPIX1)


class TestForwardAgainstAstropy:
    def test_max_pixel_error(self):
        ra  = RNG.uniform(RA0 - 0.5, RA0 + 0.5, 1000)
        dec = RNG.uniform(DEC0 - 0.5, DEC0 + 0.5, 1000)

        x_jit, y_jit = _tan_projection_jit(ra, dec, RA0, DEC0, CDINV, CRPIX0, CRPIX1)
        x_wcs, y_wcs = _astropy_world2pix(ra, dec, RA0, DEC0, CDINV, CRPIX0, CRPIX1)

        np.testing.assert_allclose(x_jit, x_wcs, atol=1e-6)
        np.testing.assert_allclose(y_jit, y_wcs, atol=1e-6)

    def test_reference_point_maps_to_crpix(self):
        x, y = _tan_projection_jit(
            np.array([RA0]), np.array([DEC0]),
            RA0, DEC0, CDINV, CRPIX0, CRPIX1,
        )
        assert x[0] == pytest.approx(CRPIX0 - 1, abs=1e-10)
        assert y[0] == pytest.approx(CRPIX1 - 1, abs=1e-10)


class TestRoundtrip:
    def test_sky_to_pixel_to_sky(self):
        ra  = RNG.uniform(RA0 - 0.4, RA0 + 0.4, 500)
        dec = RNG.uniform(DEC0 - 0.4, DEC0 + 0.4, 500)

        x, y = _tan_projection_jit(ra, dec, RA0, DEC0, CDINV, CRPIX0, CRPIX1)
        ra_rt, dec_rt = _tan_projection_inverse_jit(x, y, RA0, DEC0, CD, CRPIX0, CRPIX1)

        np.testing.assert_allclose(ra_rt, ra, atol=1e-10)
        np.testing.assert_allclose(dec_rt, dec, atol=1e-10)

    def test_pixel_to_sky_to_pixel(self):
        x = RNG.uniform(512, 1536, 500)
        y = RNG.uniform(512, 1536, 500)

        ra, dec = _tan_projection_inverse_jit(x, y, RA0, DEC0, CD, CRPIX0, CRPIX1)
        x_rt, y_rt = _tan_projection_jit(ra, dec, RA0, DEC0, CDINV, CRPIX0, CRPIX1)

        np.testing.assert_allclose(x_rt, x, atol=1e-10)
        np.testing.assert_allclose(y_rt, y, atol=1e-10)

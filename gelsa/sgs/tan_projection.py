import numpy as np
from numba import njit, float64


@njit(fastmath=True, cache=True)
def _tan_projection_jit(ra, dec, ra0, dec0, cdinv, crpix0, crpix1):
    """
    TAN projection + CD-inverse transform for N sky positions.

    Parameters
    ----------
    ra, dec : (N,) float64 arrays, degrees
    ra0, dec0 : float64, reference sky point (CRVAL), degrees
    cdinv : (2, 2) float64, inverse of the WCS CD matrix
    crpix0, crpix1 : float64, reference pixel (1-indexed, CRPIX)

    Returns
    -------
    x_out, y_out : (N,) float64 arrays, 0-indexed pixel / mm coordinates
    """
    n = ra.shape[0]
    x_out = np.empty(n, dtype=float64)
    y_out = np.empty(n, dtype=float64)

    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi

    ra0_rad  = ra0  * deg2rad
    dec0_rad = dec0 * deg2rad
    sin_dec0 = np.sin(dec0_rad)
    cos_dec0 = np.cos(dec0_rad)

    cpx = crpix0 - 1.0   # convert to 0-indexed
    cpy = crpix1 - 1.0

    c00 = cdinv[0, 0]
    c01 = cdinv[0, 1]
    c10 = cdinv[1, 0]
    c11 = cdinv[1, 1]

    for i in range(n):
        delta_ra = (ra[i] - ra0) * deg2rad
        dec_rad  = dec[i] * deg2rad

        sin_dec = np.sin(dec_rad)
        cos_dec = np.cos(dec_rad)
        cos_dra = np.cos(delta_ra)
        sin_dra = np.sin(delta_ra)

        denom = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra

        x_iwc = cos_dec * sin_dra / denom * rad2deg
        y_iwc = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / denom * rad2deg

        x_out[i] = c00 * x_iwc + c01 * y_iwc + cpx
        y_out[i] = c10 * x_iwc + c11 * y_iwc + cpy

    return x_out, y_out


@njit(fastmath=True, cache=True)
def _tan_projection_inverse_jit(x, y, ra0, dec0, cd, crpix0, crpix1):
    """
    Inverse TAN projection + CD transform for N pixel positions.

    Mirrors _tan_projection_jit exactly: pixel (0-indexed) → sky (degrees).

    Steps per point:
      1. IWC (degrees) = CD · (pixel − (CRPIX − 1))
      2. u = x_iwc · π/180,  v = y_iwc · π/180
      3. dec = arcsin((sin(dec0) + v·cos(dec0)) / √(1 + u² + v²))
      4. ra  = ra0 + atan2(u, cos(dec0) − v·sin(dec0))

    Parameters
    ----------
    x, y   : (N,) float64, 0-indexed pixel / mm coordinates
    ra0, dec0 : float64, reference sky point (CRVAL), degrees
    cd     : (2, 2) float64, the WCS CD matrix
    crpix0, crpix1 : float64, reference pixel (1-indexed, CRPIX)

    Returns
    -------
    ra_out, dec_out : (N,) float64 arrays, degrees
    """
    n = x.shape[0]
    ra_out  = np.empty(n, dtype=float64)
    dec_out = np.empty(n, dtype=float64)

    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi

    ra0_rad  = ra0  * deg2rad
    dec0_rad = dec0 * deg2rad
    sin_dec0 = np.sin(dec0_rad)
    cos_dec0 = np.cos(dec0_rad)

    cpx = crpix0 - 1.0
    cpy = crpix1 - 1.0

    c00 = cd[0, 0]
    c01 = cd[0, 1]
    c10 = cd[1, 0]
    c11 = cd[1, 1]

    for i in range(n):
        # pixel → IWC (degrees)
        px = x[i] - cpx
        py = y[i] - cpy
        x_iwc = c00 * px + c01 * py
        y_iwc = c10 * px + c11 * py

        # IWC → tangent-plane radians
        u = x_iwc * deg2rad
        v = y_iwc * deg2rad

        # inverse TAN
        norm = np.sqrt(1.0 + u * u + v * v)
        dec_out[i] = np.arcsin((sin_dec0 + v * cos_dec0) / norm) * rad2deg
        ra_out[i]  = ra0 + np.arctan2(u, cos_dec0 - v * sin_dec0) * rad2deg

    return ra_out, dec_out
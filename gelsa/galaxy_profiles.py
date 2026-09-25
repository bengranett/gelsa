import math
import numpy as np
import numba


DISK_R50 = 1.67834699
DV_COEF = 7.67
DV_NORM = 7.67**8/40320
DV_INDX = 0.25


@numba.jit(
    "float64[:](float64[:],float64)",
    nopython=True,
    parallel=True,
    cache=True
)
def disk(r, r50):
    """Disk profile

    Parameters
    ----------
    r : numpy.ndarray
        radius coordinate
    r50 : float
        half light radius of disk

    Returns
    -------
    double:
        profile

    """
    y = np.zeros(len(r), dtype='d')
    if r50 <= 0:
        return y

    scale = r50 / DISK_R50

    for i in numba.prange(len(r)):
        x = r[i] * 1./scale
        y[i] = 1./(2 * math.pi * scale * scale) * math.exp(-x)

    return y


@numba.jit(
    "float64[:](float64[:],float64)",
    nopython=True,
    parallel=True,
    cache=True
)
def disk_integrated(r, r50):
    """ Cumulative integral of exponential disk profile

    Notes
    -----
    The integrated profile is

    .. math::
       F(x) = 1 - \left(1+x\\right)e^{-x}

    Parameters
    ----------
    r : numpy.ndarray
        radius coordinate
    r50 : float
        half light radius of disk

    Returns
    -------
    double:
        profile
    """
    y = np.zeros(len(r), dtype='d')
    if r50 <= 0:
        return y

    scale = r50 / DISK_R50

    for i in numba.prange(len(r)):
        x = r[i] * 1. / scale
        y[i] = 1 - (1 + x) * math.exp(-x)

    return y


@numba.jit(
    "float64[:](float64[:],float64)",
    nopython=True,
    parallel=True,
    cache=True
)
def bulge(r, r50):
    """De Vaucouleurs profile

    Parameters
    ----------
    r : numpy.ndarray
        radius coordinate
    r50 : float
        half-light radius of bulge profile

    Returns
    -------
    numpy.ndarray:
        profile
    """
    y = np.zeros_like(r)
    if r50 <= 0:
        return y

    for i in numba.prange(len(r)):

        x = DV_COEF * math.pow(r[i] * 1. / r50, DV_INDX)

        norm = DV_NORM / math.pi / (r50 * r50)

        y[i] = norm * math.exp(-x)

    return y

@numba.jit(
    "float64[:](float64[:],float64)",
    nopython=True,
    parallel=True,
    cache=True
)
def bulge_integrated(r, r50):
    """ Cumulative integral of De Vaucouleurs profile

    Notes
    -----
    The integrated profile is

    .. math::
       F(x) = 1 - e^{-x} \left(1 + \sum_{i=1}^{7} \\frac{x^{i}}{i!} \\right)

    Parameters
    ----------
    r : numpy.ndarray
        radius coordinate
    r50 : float
        half-light radius of bulge profile

    Returns
    -------
    numpy.ndarray:
        profile

    """
    y = np.zeros_like(r)
    if r50 <= 0:
        return y

    for i in numba.prange(len(r)):
        if r[i] <= 0:
            continue
        x = DV_COEF * math.pow(r[i] / r50, DV_INDX)

        sum = 1 + x                      \
                + 1./2 * x*x             \
                + 1./6 * x*x*x           \
                + 1./24 * x*x*x*x        \
                + 1./120 * x*x*x*x*x     \
                + 1./720 * x*x*x*x*x*x   \
                + 1./5040 * x*x*x*x*x*x*x  # pow 7

        y[i] = 1 - sum * math.exp(-x)

    return y

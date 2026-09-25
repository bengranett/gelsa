import sys
import numpy as np
from scipy import optimize
from astropy.wcs import WCS


def ensurelist(x):
    """
    Parameters
    ----------
    x : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    if isinstance(x, str):
        return [x]

    try:
        len(x)
    except TypeError:
        return [x]

    return x


def asarray(x):
    """
    """
    return np.array(ensurelist(x))


def is_number(s):
    """
    Parameters
    ----------
    s : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    try:
        float(s)
        return True
    except TypeError:
        return False
    except ValueError:
        return False


def intrange(low, high, step, limit=3):
    """ """
    if low > high:
        low, high = high, low
    n = int((high - low)/step)
    n = max(limit, n)
    return np.linspace(low, high, n)


def inverse2d(func, x, y, wavelength, *args, guess=None, tol=1e-4, **kwargs):
    """ """
    points = np.transpose([x, y])
    scalar = False
    if len(points.shape) == 1:
        scalar = True
        points = points[np.newaxis, :]
    if guess is None:
        guess = points
    else:
        guess = np.ones_like(points)*guess

    out = np.zeros(points.shape, dtype='d')
    for i in range(points.shape[0]):
        r = optimize.root(
            lambda x_: points[i] - func(x_[0], x_[1], wavelength[i], *args, **kwargs),
            guess[i],
            tol=tol
        )
        out[i] = r.x
        if not r.success:
            print("Inverse error:", r.message, file=sys.stderr)
    if scalar:
        return out[0, :]
    return np.transpose(out)


def rotate_around(wcs, x, y, theta_rad):
    """ """
    ny, nx = wcs.array_shape
    x_ = x - nx/2.
    y_ = y - ny/2.
    costheta = np.cos(theta_rad)
    sintheta = np.sin(theta_rad)
    dx = x_ * costheta + y_ * sintheta
    dy = -x_ * sintheta + y_ * costheta
    return dx + nx/2, dy + ny/2


def make_wcs(ra, dec, pa=0, rotate=0, pixel_size_arcsec=1, shape=None, width_deg=0.9):
    """ """
    wcs_out = WCS()
    res = pixel_size_arcsec/3600
    if shape is None:
        n = int(width_deg / res)
        shape = (n, n)
        print(f"{shape=}")

    orientation = np.deg2rad(-pa + 90 + rotate)
    cd_matrix = np.array([[np.cos(orientation), -np.sin(orientation)],
                          [np.sin(orientation), np.cos(orientation)]])
    rot_matrix = np.array([[0, -1], [1, 0]])
    cd_matrix = rot_matrix.dot(cd_matrix)
    cd_matrix *= pixel_size_arcsec / 3600


    wcs_out.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs_out.wcs.crval = (ra, dec)
    wcs_out.wcs.crpix = (shape[0]//2,shape[1]//2)
    wcs_out.wcs.cd = cd_matrix
    wcs_out.array_shape = shape
    return wcs_out
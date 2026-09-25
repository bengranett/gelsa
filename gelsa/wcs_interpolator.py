import numpy as np
from ._vendor.fast_interp import interp2d


class WCSInterpolator:
    """Fast drop-in replacement for astropy WCS all_pix2world / all_world2pix.

    Evaluates the true WCS on a coarse pixel grid at setup time, then answers
    coordinate queries via JIT-compiled bilinear (or higher-order) interpolation.

    Usage
    -----
    wi = WCSInterpolator(wcs)
    wi.setup()           # uses wcs.array_shape
    wi.setup(step=32)    # explicit step, shape from wcs
    wi.setup(shape=(nrow, ncol), step=32)  # fully explicit
    ra, dec = wi.all_pix2world(x, y, 0)
    x, y    = wi.all_world2pix(ra, dec, 0)
    """

    def __init__(self, wcs):
        self._wcs = wcs

    def __getattr__(self, name):
        return getattr(self._wcs, name)

    def setup(self, shape=None, step=32, k=1):
        """Build interpolation grids.

        Parameters
        ----------
        shape : (nrow, ncol), optional — defaults to wcs.array_shape
        step  : grid sampling step in pixels
        k     : interpolation order (1, 3, or 5)
        """
        if shape is None:
            shape = self._wcs.array_shape
        nrow, ncol = shape
        xs = np.arange(0, ncol, step, dtype=float)
        ys = np.arange(0, nrow, step, dtype=float)
        h  = float(step)

        # Evaluate true WCS on pixel grid — meshgrid gives shape (n_y, n_x)
        xg, yg = np.meshgrid(xs, ys)
        ra, dec = self._wcs.all_pix2world(xg.ravel(), yg.ravel(), 0)

        # interp2d: f[ix, iy] — first dim = x (col), second dim = y (row)
        ra_f  = np.ascontiguousarray(ra.reshape(xg.shape).T.astype(np.float64))
        dec_f = np.ascontiguousarray(dec.reshape(xg.shape).T.astype(np.float64))

        e = [1, 1]
        self._pix2ra  = interp2d([xs[0], ys[0]], [xs[-1], ys[-1]], [h, h], ra_f,  k=k, e=e)
        self._pix2dec = interp2d([xs[0], ys[0]], [xs[-1], ys[-1]], [h, h], dec_f, k=k, e=e)

        # Inverse grid: regular RA/Dec grid, same number of points
        ras  = np.linspace(ra_f.min(),  ra_f.max(),  len(xs))
        decs = np.linspace(dec_f.min(), dec_f.max(), len(ys))
        h_ra  = float(ras[1]  - ras[0])
        h_dec = float(decs[1] - decs[0])

        rag, decg = np.meshgrid(ras, decs)
        xp, yp = self._wcs.all_world2pix(rag.ravel(), decg.ravel(), 0)

        xp_f = np.ascontiguousarray(xp.reshape(rag.shape).T.astype(np.float64))
        yp_f = np.ascontiguousarray(yp.reshape(rag.shape).T.astype(np.float64))

        self._world2x = interp2d([ras[0], decs[0]], [ras[-1], decs[-1]], [h_ra, h_dec], xp_f, k=k, e=e)
        self._world2y = interp2d([ras[0], decs[0]], [ras[-1], decs[-1]], [h_ra, h_dec], yp_f, k=k, e=e)

    def all_pix2world(self, x, y, origin=0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xr = (x - origin).ravel()
        yr = (y - origin).ravel()
        ra  = self._pix2ra(xr,  yr).reshape(x.shape)
        dec = self._pix2dec(xr, yr).reshape(x.shape)
        return ra, dec

    def all_world2pix(self, ra, dec, origin=0):
        ra  = np.asarray(ra,  dtype=float)
        dec = np.asarray(dec, dtype=float)
        x = self._world2x(ra.ravel(), dec.ravel()).reshape(ra.shape) + origin
        y = self._world2y(ra.ravel(), dec.ravel()).reshape(ra.shape) + origin
        return x, y

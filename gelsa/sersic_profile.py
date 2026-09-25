import sys
import numpy as np
from scipy.special import gamma, gammainc
from scipy.interpolate import interp1d


def sersic_b(n):
    """ Compute the sersic b parameter given the index n

    Formula from Ciotti & Bertin 1999

    Parameters
    ----------
    n : double
      Sersic index between 0 and 10

    Returns
    -------
    double : b parameter in sersic profile
    """
    return 2.*n - 1./3. + 4./405*1./n + 46./25515.*1./n**2


def sersic_profile(r, scale, n):
    """ """
    if scale <= 0:
        return 0*r
    if n <= 0:
        return 0*r
    if n > 50:
        return 0*r

    b = sersic_b(n)

    y = np.zeros(len(r))

    x = b * (r * 1./scale)**(1./n)
    norm = scale**2 * 2*np.pi * n / b**(2*n)
    norm *= gamma(2*n)
    y = 1./norm * np.exp(-x)
    return y


def sersic_integrated(r, scale, n):
    """ Return the light integrated to radius r

    Parameters
    ----------
    r : double
        radius
    scale : double
        scale radius
    n : double
        sersic index

    Returns
    -------
    double
    """
    if scale <= 0:
        return 0

    if n <= 0:
        return 0

    b = sersic_b(n)

    x = b * (r * 1./scale)**(1./n)

    incgam = gammainc(2*n, x)

    return incgam


class SersicProfile:
    """ """
    def __init__(self, res=9999, rmax=200., n_min=0.3, n_max=10, n_num=100, tol=1e-2):
        """ """
        assert rmax > 0
        assert rmax < 10000
        assert res >= 1
        assert res < 10000
        assert tol > 0
        assert tol < 1

        self.res = int(res)
        self.rmax = rmax
        self.tol = tol

        self._r = np.linspace(0, rmax, res)

        self.sersic_grid = np.linspace(n_min, n_max, n_num)
        self.setup_interp()

    def make_interpolator(self, sersic_index):
        """ """
        y = sersic_integrated(self._r, 1.0, sersic_index)

        if np.abs(1-y[-1]) > self.tol:
            print(f"Sersic profile does not converge to 1: with Sersic index={sersic_index} rmax={self._r[-1]}, y={1-y[-1]}>{self.tol}, but should be 1.  You can increase rmax.", file=sys.stderr)

        return interp1d(y, self._r, fill_value=(0, self._r[-1]), bounds_error=False)

    def setup_interp(self):
        """Setup the interpolator grid
        """
        n = self.sersic_grid.shape[0]

        self.interpolators = []
        for i in range(n):
            self.interpolators.append(self.make_interpolator(self.sersic_grid[i]))

        self.n_interpolators = n
        self.n_step = self.sersic_grid[1] - self.sersic_grid[0]
        self.n_min = self.sersic_grid[0]

    def get_interpolator(self, sersic_n):
        """Return the interpolator for the inverse cumulative function for the
          Sersic index

        Parameters
        ----------
        sersic_n : float
          sersic index

        Returns:
        --------
        interpolator : scipy.interpolate.interp1d
        """
        ngrid = self.sersic_grid
        i = int((sersic_n - self.n_min) / self.n_step)
        if i < 0:
            i = 0
        if i > ngrid.shape[0]-1:
            i = ngrid.shape[0]-1

        return self.interpolators[i]


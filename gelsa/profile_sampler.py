import numpy as np
from scipy import interpolate

from . import galaxy_profiles
from . import consts
from . sersic_profile import SersicProfile


class ProfileSampler:
    def __init__(self, integrated_profile_func, rmax=10, step=0.01):
        """ """
        r = np.arange(0, rmax, step)
        y = integrated_profile_func(r, 1.)
        # truncate flux to rmax
        y /= y[-1]
        self.cumu_inv_func = interpolate.interp1d(
            y, r,
        )

    def __call__(self, n):
        """ """
        n = int(n)
        u = np.random.uniform(0, 1, n)
        r = self.cumu_inv_func(u)
        theta = np.random.uniform(0, 2*np.pi, n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.transpose([x, y])


class FwhmGaussianSampler:
    def __call__(self, n):
        n = int(n)
        return np.random.normal(0, consts.fwhm_to_sigma, (n, 2))
    #def __call__(self, n,FWHM):
    #    n = int(n)
    #    r = np.abs(np.random.normal(loc=0.0, scale=consts.fwhm_to_sigma*FWHM, size=n))
    #    theta = np.random.uniform(low=0.0, high=2*np.pi, size=n)
    #    x = r * np.cos(theta)
    #    y = r * np.sin(theta)   
#
    #    return np.transpose([x, y])

class SersicSampler:
    def __init__(self, **kwargs):
        """ """
        self._profile = SersicProfile(**kwargs)

    def __call__(self, count, sersic_index):
        """ """
        cumu_inv_func = self._profile.get_interpolator(sersic_index)
        count = int(count)
        u = np.random.uniform(0, 1, count)
        r = cumu_inv_func(u)
        theta = np.random.uniform(0, 2*np.pi, count)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.transpose([x, y])


# initiate global samplers
disk_sampler = ProfileSampler(galaxy_profiles.disk_integrated)
bulge_sampler = ProfileSampler(galaxy_profiles.bulge_integrated, rmax=50)
fwhm_gaussian_sampler = FwhmGaussianSampler()
sersic_sampler = SersicSampler()

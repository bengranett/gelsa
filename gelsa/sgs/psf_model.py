import json
import numpy as np
from scipy import interpolate


def interpolator(x, y):
    """Utility function to set up interpolator"""
    return interpolate.interp1d(x, y,
                                bounds_error=False,
                                fill_value=(y[0], y[-1]))


class DefaultPSFModel:
    _default_params = {
        'psf_amp': 0.781749,
        'psf_scale1': 0.84454,  # pixel coordinates
        'psf_scale2': 3.64980,  # pixel coordinates
    }

    def __init__(self, **kwargs):
        """ """
        self.params = self._default_params.copy()
        self.params.update(kwargs)

    def get_model(self, *args, **kwargs):
        """ """
        return self.params['psf_amp'], self.params['psf_scale1'], self.params['psf_scale2']


class PSFModel:
    def __init__(self, path):
        """ """
        self._load(path)

    def _load(self, path):
        """ """
        print(f"loading {path}")
        with open(path, 'r') as inp:
            pack_list = json.load(inp)

        self.models = self._make_interpolators(pack_list)

    def _make_interpolators(self, pack_list):
        """ """
        pack_out = {}
        for key, pack in pack_list.items():
            x = pack['Wavelength']
            amp = pack['Amp']
            sig1 = pack['Sig1']
            sig2 = pack['Sig2']
            func_amp = interpolator(x, amp)
            func_sig1 = interpolator(x, sig1)
            func_sig2 = interpolator(x, sig2)
            pack_out[key] = (func_amp, func_sig1, func_sig2)
        return pack_out

    def get_model(self, grism_name='BGS000', **args):
        """ """
        for grism, pack in self.models.items():
            if (grism == grism_name):
                return pack
        raise ValueError(f"No PSF model found for {grism_name=}")


_default_rng = np.random.default_rng()


def sample_psf(psf_params, count=None, pos_fov=None, wavelength_ang=None, seed=None):
    """Draw samples from the PSF in pixel coordinates.

    The chromatic PSF can be simulated by passing in an array `wavelength_ang`
    with the same length as count. This tags each sample with a wavelength and
    applies the chromatic PSF.

    The dependence on focal plane position is input with `pos_fov`. This
    is a single tuple (x_fov, y_fov) in focal plane coordinates.

    Parameters
    ----------
    count : int
        number of samples to draw
    pos_fov : tuple
        tuple with focal plane coordinates (fov_x, fov_y)
    wavelength_ang : numpy.array, float
        array of wavelength values, or single float (angstroms)

    Returns
    -------
    numpy.ndarray
    """
    rng = np.random.default_rng(seed) if seed is not None else _default_rng

    func_amp, func_sig1, func_sig2 = psf_params

    if wavelength_ang is not None:
        count = len(wavelength_ang)

    try:
        # compute the PSF parameters at wavelength
        amp = func_amp(wavelength_ang)
        sig1 = func_sig1(wavelength_ang)
        sig2 = func_sig2(wavelength_ang)
    except TypeError:
        # otherwise numbers were passed in
        amp = func_amp*np.ones(count)
        sig1 = func_sig1*np.ones(count)
        sig2 = func_sig2*np.ones(count)

    # draw samples from unit gaussian
    coord = np.random.normal(0, 1, (2, count))

    # Draw from a binomial distribution with p set to amp
    # to select the first or second gaussian component
    mode = rng.binomial(1, 1-amp)
    mode1 = mode == 0
    mode2 = mode == 1

    coord[:, mode1] *= sig1[mode1]
    coord[:, mode2] *= sig2[mode2]

    return coord

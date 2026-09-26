import os
import numpy as np
from astropy.io import fits
import xml.etree.ElementTree as ET
from .._vendor.fast_interp import interp3d


class RelativeFluxCalibration:
    """ """
    datadir = "data"

    def __init__(self, path, datadir=".", workdir="."):
        """ """
        self.workdir = workdir
        self.datadir = datadir

        if self.workdir is None:
            self.workdir = os.path.dirname(path)
        else:
            path = os.path.join(workdir, path)
        self._load(path)

    def _load(self, path):
        """ """
        # print(f"loading {path}")
        if path.endswith("xml"):
            root = ET.parse(path).getroot()
            fits_filename = root.find("Data/DataStorage/DataContainer/FileName").text
            data_path = os.path.join(self.workdir, self.datadir, fits_filename)
        else:
            data_path = os.path.join(self.workdir, path)

        with fits.open(data_path) as hdul:
            pack_list = {}
            for hdu in hdul[1:]:
                key = hdu.header['EXTNAME'].split(".")[0]
                if key in pack_list:
                    continue
                pack = {
                    'Grism': hdu.header['GWA_POS'],
                    'GWATilt': hdu.header['GWA_TILT'],
                    'key': key
                }
                pack_list[key] = pack

            for key in pack_list.keys():
                header = hdul[f"{key}.SCI"].header
                data = hdul[f"{key}.SCI"].data[:]
                dq = hdul[f"{key}.DQ"].data[:]
                wstart = header['CRVAL3']
                wstep = header['CDELT3']
                corr_interp, dq_interp, limits = self._get_interpolators(data, dq, wstart, wstep)
                pack_list[key]['Map'] = corr_interp
                pack_list[key]['MapDQ'] = dq_interp
                pack_list[key]['Limits'] = limits

        self.pack_list = list(pack_list.values())

    def get_model(self, grism_name='BGS000', tilt=0):
        """ """
        for pack in self.pack_list:
            if (pack['Grism'] == grism_name) and (pack['GWATilt'] == tilt):
                return pack
        raise ValueError(f"No relative flux loss model found for {grism_name=} {tilt=}")

    def _get_interpolators(self, data, dq, wstart, wstep):
        """ """
        XX = np.linspace(-1., 1., data.shape[2])
        YY = np.linspace(-1., 1., data.shape[1])
        ZZ = np.arange(data.shape[0]) * wstep + wstart
        data[np.isnan(data)] = 0.0
        dq[np.isnan(dq)] = 1.0
        corr_interp = interp3d(
            [wstart, -1, -1],
            [ZZ[-1], 1, 1],
            [wstep, YY[1]-YY[0], XX[1]-XX[0]],
            data.astype(float),
            k=1
        )
        dq_interp = interp3d(
            [wstart, -1, -1],
            [ZZ[-1], 1, 1],
            [wstep, YY[1]-YY[0], XX[1]-XX[0]],
            dq.astype(float),
            k=1
        )
        limits = (
            (ZZ.min(), ZZ.max()),
            (YY.min(), YY.max()),
            (XX.min(), XX.max())
        )
        return corr_interp, dq_interp, limits


def get_flux_loss_fov(detector_model, pack, xfov, yfov, wavelength_ang=15000):
    """ """
    scalar_out = np.ndim(xfov) == 0
    xfov = np.atleast_1d(xfov)
    yfov = np.atleast_1d(yfov)

    wavelength_ang = np.ones(len(xfov)) * wavelength_ang

    xfov_s, yfov_s = detector_model.getScaledFOV(xfov, yfov)

    sel = (xfov_s > -1) & (xfov_s < 1) & (yfov_s > -1) & (yfov_s < 1)

    # There is no bounds check on wavelength.
    # The interpolator will extrapolate beyond the wavelength limits,
    # which is perfectly fine when there is no wavelength dependence!

    mag = pack['Map'](wavelength_ang[sel], yfov_s[sel], xfov_s[sel])

    fluxloss = np.zeros(len(xfov), dtype='d')
    fluxloss[sel] = 10**(-0.4 * mag)

    if scalar_out:
        return fluxloss[0]

    return fluxloss



def get_flux_loss(detector_model, pack, x, y, det_index, wavelength_ang=15000):
    """Return the relative flux loss

    Parameters
    ----------
    detector_model :
    pack :
    x : float, list
        detector pixel coordinate
    y : float, list
        detector pixel coordinate
    det_id : int, list
        detector index (from 0)
    wavelength_ang : float, list
        wavelength in angstroms
    """
    scalar_out = np.ndim(x) == 0
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    wavelength_ang = np.ones(len(x)) * wavelength_ang

    xfov, yfov = detector_model.getFOVPosition(x, y, det_index)

    xfov_s, yfov_s = detector_model.getScaledFOV(xfov, yfov)

    sel = (xfov_s > -1) & (xfov_s < 1) & (yfov_s > -1) & (yfov_s < 1)

    # There is no bounds check on wavelength.
    # The interpolator will extrapolate beyond the wavelength limits,
    # which is perfectly fine when there is no wavelength dependence!

    mag = pack['Map'](wavelength_ang[sel], yfov_s[sel], xfov_s[sel])

    fluxloss = np.zeros(len(x), dtype='d')
    fluxloss[sel] = 10**(-0.4 * mag)

    if scalar_out:
        return fluxloss[0]

    return fluxloss

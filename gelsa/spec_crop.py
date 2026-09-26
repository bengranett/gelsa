import numpy as np
from scipy.interpolate import interpolate
import astropy
import numba
from ._vendor.fast_interp import interp3d

from .utils import intrange
from .histogram import _histogram2d


@numba.njit(cache=True)
def _linear_interp(xp, fp, x):
    """Linear interpolation returning NaN outside [xp[0], xp[-1]]."""
    n = len(xp)
    if x < xp[0] or x > xp[n - 1]:
        return np.nan
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp[lo] * (1.0 - t) + fp[hi] * t


@numba.njit(cache=True)
def _resample_drops(xx, yy, signal, var,
                    cos_t, sin_t, x_mid, y_mid,
                    trace_par, wave_knots, perp_knots,
                    inv_sens_bins, area_weight_bins,
                    extraction_window_pix,
                    perp_bins, wave_bins,
                    offs_x, offs_y,
                    ndrops):
    """Inner drop loop for resample_on_wavelength — serial."""
    n_pix = len(xx)
    n_perp = len(perp_bins) - 1
    n_wave = len(wave_bins) - 1
    image_out = np.zeros((n_perp, n_wave))
    var_out = np.zeros((n_perp, n_wave))
    area_out = np.zeros((n_perp, n_wave))

    half_window = extraction_window_pix * 0.5
    perp_step = perp_bins[1] - perp_bins[0]
    wave_step_inv = 1.0 / (wave_bins[1] - wave_bins[0])
    inv_ndrops = 1.0 / ndrops

    for drop in range(ndrops):
        for i in range(n_pix):
            pix_dx = xx[i] + offs_x[drop, i] - x_mid
            pix_dy = yy[i] + offs_y[drop, i] - y_mid

            t_par = pix_dx * cos_t + pix_dy * sin_t
            t_perp = -pix_dx * sin_t + pix_dy * cos_t

            wavelength = _linear_interp(trace_par, wave_knots, t_par)
            if wavelength != wavelength:  # NaN check
                continue

            curve = _linear_interp(trace_par, perp_knots, t_par)
            t_perp -= curve

            if t_perp < -half_window or t_perp > half_window:
                continue

            iw = int((wavelength - wave_bins[0]) * wave_step_inv)
            if iw < 0 or iw >= n_wave:
                continue

            ip = int((t_perp - perp_bins[0]) / perp_step)
            if ip < 0 or ip >= n_perp:
                continue

            inv_s = inv_sens_bins[iw]
            image_out[ip, iw] += signal[i] * inv_s * inv_ndrops
            var_out[ip, iw] += var[i] * inv_s * inv_s * inv_ndrops
            area_out[ip, iw] += area_weight_bins[iw] * inv_ndrops

    return image_out, var_out, area_out


@numba.njit(parallel=True, cache=True)
def resample_drops(xx, yy, signal, var,
                   cos_t, sin_t, x_mid, y_mid,
                   trace_par, wave_knots, perp_knots,
                   inv_sens_bins, area_weight_bins,
                   extraction_window_pix,
                   perp_bins, wave_bins,
                   offs_x, offs_y,
                   ndrops, n_threads=1):
    """Inner drop loop for resample_on_wavelength — parallel over drops."""
    n_threads = max(1, n_threads)

    if (n_threads == 1) or (ndrops < 4):
        return _resample_drops(
            xx, yy, signal, var,
            cos_t, sin_t, x_mid, y_mid,
            trace_par, wave_knots, perp_knots,
            inv_sens_bins, area_weight_bins,
            extraction_window_pix,
            perp_bins, wave_bins,
            offs_x, offs_y,
            ndrops,
        )

    n_pix = len(xx)
    n_perp = len(perp_bins) - 1
    n_wave = len(wave_bins) - 1

    half_window = extraction_window_pix * 0.5
    perp_step = perp_bins[1] - perp_bins[0]
    wave_step_inv = 1.0 / (wave_bins[1] - wave_bins[0])
    inv_ndrops = 1.0 / ndrops

    local_image = np.zeros((n_threads, n_perp, n_wave))
    local_var   = np.zeros((n_threads, n_perp, n_wave))
    local_area  = np.zeros((n_threads, n_perp, n_wave))

    chunk = (ndrops + n_threads - 1) // n_threads

    for t in numba.prange(n_threads):
        start = t * chunk
        end = min(start + chunk, ndrops)
        for drop in range(start, end):
            for i in range(n_pix):
                pix_dx = xx[i] + offs_x[drop, i] - x_mid
                pix_dy = yy[i] + offs_y[drop, i] - y_mid

                t_par = pix_dx * cos_t + pix_dy * sin_t
                t_perp = -pix_dx * sin_t + pix_dy * cos_t

                wavelength = _linear_interp(trace_par, wave_knots, t_par)
                if wavelength != wavelength:  # NaN check
                    continue

                curve = _linear_interp(trace_par, perp_knots, t_par)
                t_perp -= curve

                if t_perp < -half_window or t_perp > half_window:
                    continue

                iw = int((wavelength - wave_bins[0]) * wave_step_inv)
                if iw < 0 or iw >= n_wave:
                    continue

                ip = int((t_perp - perp_bins[0]) / perp_step)
                if ip < 0 or ip >= n_perp:
                    continue

                inv_s = inv_sens_bins[iw]
                local_image[t, ip, iw] += signal[i] * inv_s * inv_ndrops
                local_var[t, ip, iw]   += var[i] * inv_s * inv_s * inv_ndrops
                local_area[t, ip, iw]  += area_weight_bins[iw] * inv_ndrops

    image_out = np.zeros((n_perp, n_wave))
    var_out   = np.zeros((n_perp, n_wave))
    area_out  = np.zeros((n_perp, n_wave))
    for t in range(n_threads):
        image_out += local_image[t]
        var_out   += local_var[t]
        area_out  += local_area[t]

    return image_out, var_out, area_out


class SpecCrop:
    deg_step = 0.01
    wave_step = 200.
    x_step = 200
    y_step = 200

    def __init__(self, image, mask, var,
                 detector=None, bbox=None, frame=None,
                 continuum_subtraction=None):
        """ """
        self.detector = detector
        self.bbox = bbox
        self.image = image
        self.mask = mask
        self.var = var
        self.shape = image.shape
        self.interper_ra = None
        self.interper_dec = None
        self.interper_x = None
        self.interper_y = None
        self._cs = continuum_subtraction

        self.frame = frame

    def copy(self):
        """ """
        S = SpecCrop(
            self.image.copy(),
            self.mask.copy(),
            self.var.copy(),
            detector=self.detector,
            bbox=self.bbox,
            continuum_subtraction=self._cs
        )
        S.frame = self.frame
        S.interper_ra = self.interper_ra
        S.interper_dec = self.interper_dec
        S.interper_x = self.interper_x
        S.interper_y = self.interper_y
        return S

    @staticmethod
    def crop(frame, detector, bbox, continuum_subtraction=None):
        """ """
        x_0, x_1, y_0, y_1 = bbox
        image, mask, var = frame.get_detector(detector)
        image = image[y_0:y_1, x_0:x_1]
        mask = mask[y_0:y_1, x_0:x_1]
        var = var[y_0:y_1, x_0:x_1]

        if continuum_subtraction is not None:
            image, mask = continuum_subtraction.process_cutout(frame, image, mask)

        return SpecCrop(image, mask, var,
                        detector=detector, bbox=bbox, frame=frame,
                        continuum_subtraction=continuum_subtraction)

    @staticmethod
    def crop_trace(frame, detector, x, y, padx=100, pady=10, continuum_subtraction=None):
        """ """
        image, mask, var = frame.get_detector(detector)
        nrow, ncol = image.shape

        x_0 = max(0, int(x.min() - padx))
        x_1 = min(ncol, int(x.max() + padx))
        y_0 = max(0, int(y.min() - pady))
        y_1 = min(nrow, int(y.max() + pady))

        bbox = (x_0, x_1, y_0, y_1)

        image = image[y_0:y_1, x_0:x_1]
        mask = mask[y_0:y_1, x_0:x_1]
        var = var[y_0:y_1, x_0:x_1]

        if continuum_subtraction is not None:
            image, mask = continuum_subtraction.process_cutout(frame, image, mask)

        return SpecCrop(image, mask, var,
                        detector=detector, bbox=bbox, frame=frame)

    def _setup_pixel_interpolator(self, frame):
        """ """
        x_0, x_1, y_0, y_1 = self.bbox
        w_0, w_1 = frame.params['wavelength_range']

        xx = np.array([x_0, x_1, x_0, x_1]).astype(float)
        yy = np.array([y_0, y_1, y_0, y_1]).astype(float)
        wave = np.array([w_0, w_0, w_1, w_1]).astype(float)

        ra_, dec_ = frame.pixel_to_radec(xx, yy, self.detector, wave)

        ra0 = np.min(ra_)
        dec0 = np.min(dec_)
        ra1 = np.max(ra_)
        dec1 = np.max(dec_)

        mu = np.cos(np.radians(dec0))
        if mu > 0:
            ra_step = self.deg_step/mu
        else:
            print(f"Warning! at pole, declination hit 90: {dec0}")
            ra_step = self.deg_step

        ra_grid = intrange(ra0, ra1, ra_step)
        dec_grid = intrange(dec0, dec1, self.deg_step)
        wave_grid = intrange(w_0, w_1, self.wave_step)

        ra_, dec_, wave_ = np.meshgrid(ra_grid, dec_grid, wave_grid, indexing='ij')
        x, y, det_ = frame.framecoord.radec_to_pixel(ra_.flatten(), dec_.flatten(), wave_.flatten())
        off_det = det_ != self.detector
        x[off_det] = -1000
        y[off_det] = -1000

        x = x.reshape(ra_.shape)
        y = y.reshape(ra_.shape)

        x -= x_0
        y -= y_0

        self.interper_x = interp3d(
            [ra_grid[0], dec_grid[0], wave_grid[0]],
            [ra_grid[-1], dec_grid[-1], wave_grid[-1]],
            [ra_grid[1]-ra_grid[0], dec_grid[1]-dec_grid[0], wave_grid[1]-wave_grid[0]],
            x,
            k=1
        )
        self.interper_y = interp3d(
            [ra_grid[0], dec_grid[0], wave_grid[0]],
            [ra_grid[-1], dec_grid[-1], wave_grid[-1]],
            [ra_grid[1]-ra_grid[0], dec_grid[1]-dec_grid[0], wave_grid[1]-wave_grid[0]],
            y,
            k=1
        )
        # self.interper_x = RegularGridInterpolator(
        #     (ra_grid, dec_grid, wave_grid), x,
        #     bounds_error=False,
        #     fill_value=None
        # )
        # self.interper_y = RegularGridInterpolator(
        #     (ra_grid, dec_grid, wave_grid), y,
        #     bounds_error=False,
        #     fill_value=None
        # )

    def _setup_radec_interpolator(self, frame):
        """ """
        x_0, x_1, y_0, y_1 = self.bbox
        w_0, w_1 = frame.params['wavelength_range']

        x_grid = intrange(x_0, x_1, self.x_step).astype(float)
        y_grid = intrange(y_0, y_1, self.y_step).astype(float)
        wave_grid = intrange(w_0, w_1, self.wave_step).astype(float)

        x_, y_, wave_ = np.meshgrid(x_grid, y_grid, wave_grid, indexing='ij')

        det_ = np.ones(x_.shape, dtype=int) * self.detector
        ra, dec = frame.framecoord.pixel_to_radec(
            x_.flatten(), y_.flatten(), det_.flatten(), wave_.flatten())

        ra = ra.reshape(x_.shape)
        dec = dec.reshape(x_.shape)

        x_grid -= x_0
        y_grid -= y_0

        self.interper_ra = interp3d(
            [x_grid[0], y_grid[0], wave_grid[0]],
            [x_grid[-1], y_grid[-1], wave_grid[-1]],
            [x_grid[1]-x_grid[0], y_grid[1]-y_grid[0], wave_grid[1]-wave_grid[0]],
            ra,
            k=1
        )
        self.interper_dec = interp3d(
            [x_grid[0], y_grid[0], wave_grid[0]],
            [x_grid[-1], y_grid[-1], wave_grid[-1]],
            [x_grid[1]-x_grid[0], y_grid[1]-y_grid[0], wave_grid[1]-wave_grid[0]],
            dec,
            k=1
        )
        # self.interper_ra = RegularGridInterpolator(
        #     (x_grid, y_grid, wave_grid),
        #     ra,
        #     bounds_error=False,
        #     fill_value=None
        # )
        # self.interper_dec = RegularGridInterpolator(
        #     (x_grid, y_grid, wave_grid),
        #     dec,
        #     bounds_error=False,
        #     fill_value=None
        # )

    def radec_to_pixel(self, ra, dec, wavelength):
        """ """
        # points = np.transpose([ra, dec, wavelength])
        x = self.interper_x(ra, dec, wavelength)
        y = self.interper_y(ra, dec, wavelength)
        return x, y

    def pixel_to_radec(self, x, y, wavelength):
        """ """
        # points = np.transpose([x, y, wavelength])
        ra = self.interper_ra(x, y, wavelength)
        dec = self.interper_dec(x, y, wavelength)
        return ra, dec

    def pixel_to_radec_(self, x, y, wavelength):
        """ """
        ra, dec = self.frame.pixel_to_radec(
            x+self.bbox[0], y+self.bbox[2],
            self.detector, wavelength)
        return ra, dec

    def radec_to_pixel_(self, ra, dec, wavelength, dispersion_order=1):
        """ """
        scalar = np.ndim(ra) == 0
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)

        x, y, det = self.frame.radec_to_pixel(ra, dec, wavelength,
                                              dispersion_order=dispersion_order)
        x -= self.bbox[0]
        y -= self.bbox[2]
        invalid = det != self.detector
        x[invalid] = -1
        y[invalid] = -1

        if scalar:
            return x[0], y[0]
        return x, y

    def get_dispersion_direction_on_wcs(self, wcs, step=50):
        """Compute dispersion direction on the sky.
        The angle is return using the position angle convention:
        angle from North, increasing toward East.

        Angle is returned in radians.

        Parameters
        ----------
        wcs
        ra
        dec
        wavelength

        Returns
        -------
        theta : float
        dispersion direction in radians
        """
        ra, dec = self.center
        wavelength = (self.wavelength_trace.min()+self.wavelength_trace.max())/2.
        x, y = self.radec_to_pixel_(ra, dec, wavelength+step)
        sign = 1
        if x < 0:
            x, y = self.radec_to_pixel_(ra, dec, wavelength-step)
            sign = -1
            if x < 0:
                raise Exception

        ra1, dec1 = self.pixel_to_radec_(x, y, wavelength)
        im_x, im_y = wcs.all_world2pix(ra, dec, 0)
        im_x1, im_y1 = wcs.all_world2pix(ra1, dec1, 0)
        dy = im_y1 - im_y
        dx = im_x1 - im_x
        dx *= sign
        dy *= sign
        return np.arctan2(dx, dy)

    def get_relative_flux_loss(self, x, y, wavelength):
        """Get the relative flux loss at pixel coordinate"""
        x_frame = x + self.bbox[0]
        y_frame = y + self.bbox[2]
        return self.frame.get_relative_flux_loss(
            x_frame,
            y_frame,
            self.detector,
            wavelength
        )

    def get_relative_flux_loss_radec(self, ra, dec, wavelength_ang=None):
        """Get the relative flux loss at pixel coordinate"""
        return self.frame.get_relative_flux_loss_radec(
            ra,
            dec,
            wavelength_ang
        )

    def get_sensitivity(self, wavelength):
        """ """
        return self.frame.lineflux_to_counts(
            1, wavelength
        )

    def resample_on_radec(self, ra, dec, wavelength,
                          width=None,
                          wcs=None,
                          ndrops=500,
                          pixel_shrink=1,
                          use_rel_flux_calib=True,
                          use_abs_flux_calib=True,
                          seed=3):
        """ """
        if self.interper_x is None:
            self._setup_radec_interpolator(self.frame)

        rng = np.random.default_rng(seed)

        shape_out = wcs.array_shape
        pix_scale_out = astropy.wcs.utils.proj_plane_pixel_scales(wcs)[0]*3600

        pix_scale_in = self.frame.params['PIXSCALE']

        if width is None:
            width = int(pix_scale_in/pix_scale_out * \
                        np.sqrt(np.sum(np.square(shape_out))))

        x, y = self.radec_to_pixel_(np.array([ra]), np.array([dec]),
                                    np.array([wavelength]))
        x = x[0]
        y = y[0]
        if x < 0 or y < 0:
            raise NoExtraction

        if x > self.shape[1] or y > self.shape[0]:
            raise NoExtraction

        x0 = int(x - width//2)
        x1 = int(x0 + width)
        y0 = int(y - width//2)
        y1 = int(y0 + width)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.shape[1]-1, x1)
        y1 = min(self.shape[0]-1, y1)

        pix_bins = (
            np.arange(wcs.array_shape[0]+1),
            np.arange(wcs.array_shape[1]+1),
        )

        image = self.image[y0:y1+1, x0:x1+1]
        mask = self.mask[y0:y1+1, x0:x1+1]
        var = self.var[y0:y1+1, x0:x1+1]
        valid = (mask == 0) & (var > 0) & np.isfinite(
            var) & np.isfinite(image) & np.isfinite(mask)

        if np.sum(valid)==0:
            raise NoExtraction

        signal = image[valid]/ndrops
        signal_var = var[valid]/ndrops

        xgrid = np.arange(x0, x1+1)
        ygrid = np.arange(y0, y1+1)
        xx, yy = np.meshgrid(xgrid, ygrid, indexing='xy')
        xx = xx[valid]
        yy = yy[valid]

        if use_rel_flux_calib:
            flux_loss = self.get_relative_flux_loss_radec(ra, dec, wavelength)
            if flux_loss > 0:
                signal /= flux_loss
                signal_var /= flux_loss**2

        if use_abs_flux_calib:
            sens = self.get_sensitivity(np.array([wavelength]))[0]
            signal /= sens
            signal_var /= sens**2

        n_pix = len(xx)
        wl_pix = np.full(n_pix, wavelength)

        # Pre-generate all offsets at once: (ndrops, n_pix)
        offs = rng.uniform(-0.5, 0.5, (2, ndrops, n_pix)) * pixel_shrink + 0.5

        # Batch all drops into one coordinate call: (ndrops * n_pix,)
        xx_all = (xx[np.newaxis, :] + offs[0]).ravel()
        yy_all = (yy[np.newaxis, :] + offs[1]).ravel()
        wl_all = np.tile(wl_pix, ndrops)

        ra_all, dec_all = self.pixel_to_radec(xx_all, yy_all, wavelength=wl_all)
        pix_x_all, pix_y_all = wcs.all_world2pix(ra_all, dec_all, 0)

        signal_all = np.tile(signal, ndrops)
        var_all    = np.tile(signal_var, ndrops)
        ones_all   = np.full(ndrops * n_pix, 1.0 / ndrops)

        image_out = _histogram2d(pix_y_all, pix_x_all, weight=signal_all,
                                bins_y=pix_bins[0], bins_x=pix_bins[1])
        var_image  = _histogram2d(pix_y_all, pix_x_all, weight=var_all,
                                 bins_y=pix_bins[0], bins_x=pix_bins[1])
        area_out   = _histogram2d(pix_y_all, pix_x_all, weight=ones_all,
                                 bins_y=pix_bins[0], bins_x=pix_bins[1])

        image_out *= area_out
        var_image *= area_out**4

        return image_out, var_image, area_out, pix_bins

    def resample_on_wavelength(self, ra, dec,
                               extraction_window_pix=11,
                               wave_range=None,
                               wave_step=10.,
                               ndrops=100,
                               pixel_shrink=1,
                               seed=3,
                               use_abs_flux_calib=True,
                               use_rel_flux_calib=True,
                               super_sample=3,
                               contamination_model=None,
                               n_threads=1
                              ):
        """Resample 2D spectrum on linear wavelength grid.

        Parameters
        ----------
        ra
        dec
        wave_range
        wave_step
        wcs
        ndrops
        pixel_shrink : float
        seed :

        Returns
        -------
        image, bins
        """
        rng = np.random.default_rng(seed)

        image = self.image.astype(float)

        if contamination_model is not None:
            image -= contamination_model

        mask = self.mask.astype(float)
        var = self.var.astype(float)
        valid = (mask == 0) & (var > 0) & np.isfinite(
            var) & np.isfinite(image) & np.isfinite(mask)

        var = var[valid]
        signal = image[valid]

        shape = self.image.shape
        ygrid = np.arange(0, shape[0])
        xgrid = np.arange(0, shape[1])
        xx, yy = np.meshgrid(xgrid, ygrid, indexing='xy')
        xx = xx[valid].astype(np.float64)
        yy = yy[valid].astype(np.float64)

        trace = np.linspace(*wave_range, 100)
        trace_x, trace_y = self.radec_to_pixel_(
            ra*np.ones(len(trace)), dec*np.ones(len(trace)), trace)
        valid_trace = (trace_x > 0) & (trace_y > 0)
        if np.sum(valid_trace) <= 1:
            raise NoExtraction

        trace = trace[valid_trace]
        trace_x = trace_x[valid_trace]
        trace_y = trace_y[valid_trace]

        mid_i = len(trace_x)//2

        dx = trace_x - trace_x[mid_i]
        dy = trace_y - trace_y[mid_i]

        nonzero = dx != 0
        theta = np.arctan2(dy[nonzero], dx[nonzero])
        positive = theta > 0
        theta[positive] -= np.pi

        theta = np.median(theta)
        cos_trace = np.cos(theta)
        sin_trace = np.sin(theta)

        trace_par = dx * cos_trace + dy * sin_trace
        trace_perp = -dx * sin_trace + dy * cos_trace

        if use_rel_flux_calib:
            flux_loss = self.get_relative_flux_loss_radec(
                ra, dec,
                (wave_range[0]+wave_range[1])/2.
            )
            if flux_loss > 0:
                signal /= flux_loss
                var /= flux_loss

        pix_bin_ = np.arange(extraction_window_pix*super_sample + 1) / super_sample - extraction_window_pix/2.
        pix_bins = (
            pix_bin_,
            np.arange(*wave_range, wave_step),
        )
        n_wave_bins = len(pix_bins[1]) - 1

        if use_abs_flux_calib:
            wave_centers = 0.5 * (pix_bins[1][:-1] + pix_bins[1][1:])
            sens = self.get_sensitivity(wave_centers) * wave_step
            inv_sens_bins = np.where(sens > 0, 1.0 / np.where(sens > 0, sens, 1.0), 0.0)
            area_weight_bins = (sens > 0).astype(np.float64)
        else:
            inv_sens_bins = np.ones(n_wave_bins)
            area_weight_bins = np.ones(n_wave_bins)

        offs_x = rng.uniform(-0.5, 0.5, (ndrops, len(xx))) * pixel_shrink + 0.5
        offs_y = rng.uniform(-0.5, 0.5, (ndrops, len(xx))) * pixel_shrink + 0.5

        image_out, var_out, area_out = resample_drops(
            xx, yy, signal, var,
            cos_trace, sin_trace, trace_x[mid_i], trace_y[mid_i],
            trace_par, trace, trace_perp,
            inv_sens_bins, area_weight_bins,
            float(extraction_window_pix),
            pix_bins[0], pix_bins[1],
            offs_x, offs_y,
            ndrops,
            n_threads=n_threads
        )

        image_out *= area_out
        var_out *= area_out**4 * super_sample**2

        return image_out, var_out, area_out, pix_bins


class NoExtraction(Exception):
    pass

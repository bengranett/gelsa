import logging
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from . import consts

logger = logging.getLogger(__name__)


def _rolling_nanmedian(image, filt_n):
    """Row-wise sliding-window nanmedian, equivalent to
    ``ndimage.generic_filter(image, ..., size=(1, filt_n), mode='reflect')``
    but vectorized. Note scipy's 'reflect' mode repeats the edge pixel,
    matching numpy's 'symmetric' pad mode (not numpy's 'reflect').
    """
    pad_left = filt_n // 2
    pad_right = filt_n - 1 - pad_left
    padded = np.pad(image, ((0, 0), (pad_left, pad_right)), mode='symmetric')
    windows = sliding_window_view(padded, filt_n, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmedian(windows, axis=-1)


def make_line_mask(crop, redshift, width_pix=6, wavelength_list=None):
    """Construct a mask with rectangles around emission lines.

    Parameters
    ----------
    crop : speccrop.Crop
    redshift : float
    width_pix : int
    wavelength_list : list or None

    Returns
    -------
    mask : ndarray
    """
    ny, nx = crop.image.shape
    line_mask = np.zeros((ny, nx), dtype=bool)

    if (redshift is None) or (redshift <= 0):
        return line_mask

    if width_pix <= 0:
        return line_mask

    ra, dec = crop.center

    wave_min, wave_max = crop.frame.params['wavelength_range']

    half = int(width_pix // 2)

    if not wavelength_list:
        wavelength_list = consts.extended_wavelength_list

    for lam_rest in wavelength_list:
        lam_obs = lam_rest * (1 + redshift)
        if not (wave_min <= lam_obs <= wave_max):
            continue
        if len(crop.wavelength_trace) == 0:
            continue
        x, y = crop.radec_to_pixel_(ra, dec, lam_obs)
        if x < 0 or y < 0:
            # off the detector
            continue
        x = int(round(x))
        y = int(round(y))

        x0 = max(0, x - half)
        x1 = min(x + half, nx)
        y0 = max(0, y - half)
        y1 = min(y + half, ny)

        line_mask[y0:y1, x0:x1] = 1
    return line_mask


def _rotate_image(image, tilt):
    """Rotate by tilt degrees with linear interpolation.

    NaN pixels spread only to their immediate neighbours, and the
    uncovered corners are filled with NaN.
    """
    image = np.asarray(image, dtype=float)
    if tilt == 0:
        return image
    return ndimage.rotate(
        image, tilt,
        reshape=False,
        order=1,
        mode='constant',
        cval=np.nan
    )


def median_filter(image, mask, filter_size_pix=0, tilt=0, line_mask=None):
    """Subtract a running median along the dispersion direction.

    Pixels that are non-finite, flagged in mask, or flagged in line_mask
    are excluded from the median. Returns the residual image and a boolean
    array of pixels where no residual could be computed.
    """
    invalid = ~np.isfinite(image) | (mask > 0)
    if filter_size_pix <= 0:
        return image, invalid

    exclude = invalid
    if line_mask is not None:
        exclude = exclude | (line_mask > 0)

    # excluded pixels become NaN, which _rolling_nanmedian skips
    image_ = np.where(exclude, np.nan, image)

    rotated_image = _rotate_image(image_, tilt)
    filtered_rotated_image = _rolling_nanmedian(rotated_image, filter_size_pix)
    rotated_back = _rotate_image(filtered_rotated_image, -tilt)
    resid = image - rotated_back
    invalid = invalid | ~np.isfinite(resid)
    resid[invalid] = 0
    return resid, invalid


def cosmetic_filter(resid_image, threshold=80, n=5):
    """ """
    kernel = np.array([np.ones(n), -1*np.ones(n), np.ones(n)])
    kernel = kernel / np.sum(kernel)

    filtered = ndimage.median_filter(resid_image, 15, axes=1)
    filtered = ndimage.convolve(filtered, kernel)

    invalid = np.abs(filtered) > threshold
    resid_image[invalid] = 0

    return resid_image, invalid


class MedianFilterBase:
    _default_params = {
        'median_filter_size_pix': 50,
        'cosmetic_threshold': 80,
        'cosmetic_n': 5,
        'redshift': None,
        'mask_lines': False,
        'line_mask_width_pix': 6
    }

    def __init__(self, median_filter_size_pix=50, redshift=None,
                 mask_lines=False,
                 **kwargs):
        """ """
        self.params = self._default_params.copy()
        if median_filter_size_pix is not None:
            self.params['median_filter_size_pix'] = median_filter_size_pix
        if redshift is not None:
            self.params['redshift'] = redshift
        if self.params['mask_lines'] is not None:
            self.params['mask_lines'] = mask_lines
        for key, value in kwargs.items():
            if key not in self.params:
                logger.warning(f"Unknown parameter {key}")
        self.params.update(kwargs)
        self._standardize_kernel_size()

    def _standardize_kernel_size(self):
        """ """
        filter_size = self.params['median_filter_size_pix']
        filter_size = int(filter_size)
        if filter_size <= 0:
            # filtering disabled
            self.params['median_filter_size_pix'] = 0
            return
        if filter_size < 3:
            filter_size = 3
        if filter_size % 2 == 0:
            filter_size += 1
        self.params['median_filter_size_pix'] = filter_size

    def process_frame(self, frame, image, mask):
        """ """
        return image, mask

    def process_cutout(self, crop, image, mask):
        """ """
        """ """
        return image, mask


class MedianFilterFrame(MedianFilterBase):
    def process_frame(self, frame, image, mask):
        """ """
        tilt = frame.params['tilt']
        image_out, invalid = median_filter(
            image,
            mask,
            self.params['median_filter_size_pix'],
            tilt
        )
        image_out, invalid2 = cosmetic_filter(
            image_out,
            self.params['cosmetic_threshold'],
            self.params['cosmetic_n']
        )

        invalid = invalid | invalid2

        mask[invalid] = 1

        return image_out, mask


class MedianFilterCutout(MedianFilterBase):
    def process_cutout(self, crop, image, mask):
        """ """
        line_mask = None
        if self.params['mask_lines']:
            redshift = self.params['redshift']
            if redshift is None:
                redshift = crop.redshift
            line_mask = make_line_mask(
                crop, redshift,
                width_pix=self.params['line_mask_width_pix']
            )
        tilt = crop.frame.params['tilt']
        image_out, invalid = median_filter(
            image,
            mask,
            filter_size_pix=self.params['median_filter_size_pix'],
            tilt=tilt,
            line_mask=line_mask
        )
        mask[invalid] = 1
        return image_out, mask

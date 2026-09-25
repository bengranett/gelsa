import numpy as np
from scipy import ndimage


def median_filter(image, filter_size_pix=0, tilt=0):
    """ """
    if filter_size_pix <= 0:
        return
    invalid = np.logical_not(np.isfinite(image))
    image[invalid] = 0
    rotated_image = ndimage.rotate(image, tilt, reshape=False)
    filtered_rotated_image = ndimage.median_filter(rotated_image, filter_size_pix, axes=1)
    rotated_back = ndimage.rotate(filtered_rotated_image, -tilt, reshape=False, cval=np.nan)
    resid = image - rotated_back
    invalid = np.logical_not(np.isfinite(resid))
    resid[invalid] = 0

    return resid, invalid


def cosmetic_filter(resid_image, threshold=80, n=5, ):
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
        'redshift': None
    }
    def __init__(self, median_filter_size_pix=50, redshift=None, **kwargs):
        """ """
        self.params = self._default_params.copy()
        if median_filter_size_pix is not None:
            self.params['median_filter_size_pix'] = median_filter_size_pix
        if redshift is not None:
            self.params['redshift'] = redshift
        self.params.update(kwargs)

    def process_frame(self, frame, image, mask):
        """ """
        return image, mask

    def process_cutout(self, frame, image, mask):
        """ """
        """ """
        return image, mask


class MedianFilterFrame(MedianFilterBase):
    def process_frame(self, frame, image, mask):
        """ """
        tilt = frame.params['tilt']
        image_out, invalid = median_filter(
            image,
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
    def process_cutout(self, frame, image, mask):
        """ """
        tilt = frame.params['tilt']
        image_out, invalid = median_filter(
            image,
            self.params['median_filter_size_pix'],
            tilt
        )
        mask[invalid] = 1

        return image_out, mask

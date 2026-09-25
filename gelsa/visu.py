import numpy as np
from scipy import ndimage, interpolate
from matplotlib import pyplot as plt
import skimage
from scipy.ndimage import rotate, median_filter

from astropy.visualization import make_lupton_rgb

from . import spec_crop


_defaults = {
    'cmap': plt.cm.plasma.with_extremes(bad='grey'),
    'origin': 'lower',
    'interpolation': 'Nearest'
}


def filter_image(image, masked_pix, iter=1, median_window=3):
    """Process the image with an iterative median filter.

    Bad pixels are repaired iteratively by replacing them
    with the values from the median filtered image.

    The kernel size is set by the median_window class variable.

    Parameters
    ----------
    image : numpy.ndarray
        input image to process
    masked_pix : numpy.ndarray
        selection array indicating bad pixels
    iter : int

    Returns
    -------
    numpy.ndarray
        processed image
    """
    im_m = None
    im_copy = image.copy()
    for loop in range(iter):
        if loop > 0:
            im_copy[masked_pix] = im_m[masked_pix]
        im_m = ndimage.median_filter(im_copy, size=median_window)
    return im_m, im_copy

def show(image=None, mask=None, var=None, crop=None, axes=None, levels=(10, 80),
         vmin_offset=200,
         infill=False, label_wavelength_step=500, colorbar=False, slice=None,
         extent=None, **params):
    """Display an image with matplotlib """
    # copy in default plot parameters
    for key, value in _defaults.items():
        if key not in params:
            params[key] = value

    if isinstance(image, spec_crop.SpecCrop):
        crop = image
        image = crop.image
        var = crop.var
        mask = crop.mask

    if crop is not None:
        if image is None:
            image = crop.image
        if mask is None:
            mask = crop.mask
        if var is None:
            var = crop.var

    if slice is None:
        slice = np.s_[:,:]

    image_ = image[slice]
    valid = np.isfinite(image_)
    if mask is not None:
        valid &= mask[slice] == 0
    if var is not None:
        valid &= np.isfinite(var[slice]) & (var[slice] > 0)
    if ('vmin' not in params) or ('vmax' not in params):
        vmin, vmax = np.percentile(
            image_[valid].flatten(),
            (levels[0], levels[1])
        )
        params['vmin'] = vmin
        if vmin_offset > 0:
            params['vmax'] = vmin + vmin_offset
        else:
            params['vmax'] = vmax

    if infill:
        _, image_ = filter_image(image[slice], valid==False, iter=10, median_window=5)
    else:
        image_ = np.ma.array(image[slice], mask=valid==False)

    if axes is None:
        axes = plt.gca()

    aspect = 'equal'
    if crop is not None:
        start = crop.wavelength_trace.min()
        end = crop.wavelength_trace.max()
        print(start, end)

        w = np.linspace(start, end, 100)

        ra, dec = crop.center
        xx, yy = crop.radec_to_pixel_(
            ra*np.ones(len(w)), dec*np.ones(len(w)), w)
        valid = xx > 0
        if np.sum(valid)>1:
            # axes.plot(w[valid], yy[valid]-5, c='c', zorder=10)
            # axes.plot(w[valid], yy[valid]+5, c='c', zorder=10)
            func = interpolate.interp1d(xx[valid], w[valid],
                                        bounds_error=False, fill_value='extrapolate')
            ny, nx = image_.shape
            extent = (func(0), func(nx-1), 0, ny-1)
            aspect = np.abs(extent[1]-extent[0])/nx

    im = axes.imshow(image_, extent=extent, aspect=aspect, **params)

    if extent is not None:
        if extent[1] < extent[0]:
            axes.invert_xaxis()

    # fig = plt.gcf()
    if colorbar:
        plt.colorbar(im, ax=axes, pad=0, aspect=40, location='top')
    # hide y tick labels
    # axes.yaxis.set_ticklabels([])

    return image_


def trace1d(image, mask=None, var=None, axis=0, **params):
    """Make simple 1D trace of the 2D image"""
    # copy in default plot parameters
    image_ = image.copy()
    valid = np.isfinite(image_)
    if mask is not None:
        valid &= mask == 0
    if var is not None:
        valid &= np.isfinite(var) & (var > 0)
    mask_ = np.zeros(image_.shape)
    mask_[valid] = 1
    norm = np.sum(mask_, axis=axis)
    trace = np.sum(image_, axis=axis) / norm
    return trace


def plot_detectors_focal_plane(frame, **plotparams):
    """ """
    from matplotlib import pyplot as plt
    ax = plt.subplot(111, aspect='equal')
    for x, y in frame.detector_model.detector_poly_list:
        x = np.concatenate([x, [x[0]]])
        y = np.concatenate([y, [y[0]]])
        ax.plot(x, y, **plotparams)
    low, high = frame.detector_model.envelope
    plt.plot(
        [low[0], high[0], high[0], low[0], low[0]],
        [low[1], low[1], high[1], high[1], low[1]],
        dashes=[4, 1], c='k'
    )


def plot_detectors_sky(frame, **plotparams):
    """ """
    from matplotlib import pyplot as plt
    ax = plt.subplot(111, aspect=1./np.cos(np.radians(frame.params['DEC'])))
    for x, y in frame.detector_model.detector_poly_list:
        x = np.concatenate([x, [x[0]]])
        y = np.concatenate([y, [y[0]]])
        ra, dec = frame.framecoord.fov_to_radec(x, y, 10000)
        ax.plot(ra, dec, **plotparams)


def rank_transform(image):
    """ """
    image_ = np.zeros(image.shape, dtype=int)
    values = np.unique(image)
    values.sort()
    for i, v in enumerate(values):
        image_[image==v] = i
    return image_
        
        
def normalize_image(x, var=None, mask=None, levels=(10, 99.95),
                    vmin=None, vmax=None,
                    power=0.5,
                    return_levels=False):
    """Normalize an image with power. Applies percentile clipping.

    Parameters
    ----------
    x
    levels
    power

    Returns
    -------
    numpy.ndarray
    """
    valid = np.isfinite(x)
    try:
        valid &= ~ valid.mask
    except AttributeError:
        pass
    if var is not None:
        valid &= (var > 0) & (np.isfinite(var))
    if mask is not None:
        valid &= (mask > 0) & (np.isfinite(mask))
    if np.sum(valid) == 0:
        print("no valid pixels!")
        return y
    if (vmin is None) or (vmax is None):
        low, high = np.percentile(x[valid].data, levels)
    else:
        low, high = vmin, vmax
    delta = high - low
    if delta == 0:
        print(f"warning, high and low are equal {high} - {low}")
        delta = 1
    y = np.zeros_like(x)
    y[valid] = (x[valid] - low) / delta
    y[y < 0] = 0
    y[y > 1] = 1
    y = np.ma.array(y**power, mask=1-valid)
    if return_levels:
        return y, dict(vmin=low, vmax=high, power=power)
    else:
        return y

def make_color_image(r, g, b, brightness=None,
                     **kwargs):
    """Creates a color composite with rgb channels and optional brigtness value

    The color image is made with astropy.visualization.make_lupton_rgb.

    If a brightness image is passed, it is used to set the value
    in the hsv color space. It should be normalized to the range [0, 1].

    Parameters
    ----------
    r : numpy.array
    g : numpy.array
    b : numpy.array
    brightness : numpy.array
    stretch : float
        argument for astropy make_lupton_rgb
    Q : float
        argument for astropy make_lupton_rgb

    Returns
    -------
    numpy.array
    """
    im_rgb = make_lupton_rgb(r, g, b, **kwargs)
    if brightness is not None:
        im_rgb = im_rgb / 255
        im_hsv = skimage.color.rgb2hsv(im_rgb)
        im_hsv[:, :, 2] = brightness
        im_rgb = skimage.color.hsv2rgb(im_hsv)
    return im_rgb


def colorize_image(image, hue=0, saturation=1):
    """Colorize an image map. The image array should be normalized in the range [0, 1]

    Parameters
    ----------
    image : numpy.ndarray
        image map (2D array)
    hue : float
        Hue value, in the range [0, 1]
    saturation : float
        Saturation value, in the range [0, 1]

    Returns
    -------
    numpy.ndarray
    """
    im_rgb = skimage.color.gray2rgb(image)
    hsv = skimage.color.rgb2hsv(im_rgb)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return skimage.color.hsv2rgb(hsv)


def make_image_composite(image_rgb, image_overlay, alpha_power=1, alpha=0.7):
    """Overlay two rgb images.
    The brightness of image_overlay is used for the mask.

    Parameters
    ----------
    image_rgb : numpy.ndarray
        base rgb image
    image_overlay : numpy.ndarray
        rgb image to overlay
    alpha_power : float
        exponent for the alpha map
    alpha : float
        opacity value in the range 0 transparent to 1 opaque.

    Returns
    -------
    numpy.ndarray
    """
    brightness = skimage.color.rgb2hsv(image_overlay)[:, :, 2]
    # set transparency alpha channel
    alpha_mask = np.ones_like(image_rgb)*brightness[:, :, np.newaxis]
    alpha_mask = alpha * alpha_mask**alpha_power
    return image_rgb*(1-alpha_mask) + image_overlay*alpha_mask


def median_filter_image(image, filter_size=40, tilt=0):
    """Apply median filter to the image"""
    invalid = np.logical_not(np.isfinite(image))
    image[invalid] = 0
    if tilt != 0:
        rotated_image = rotate(image, tilt, reshape=False)
    else:
        rotated_image = image
    filtered_rotated_image = median_filter(rotated_image, filter_size, axes=1)
    if tilt != 0:
        rotated_back = rotate(filtered_rotated_image, -tilt, reshape=False, cval=np.nan)
    else:
        rotated_back = filtered_rotated_image
    resid = image - rotated_back
    invalid = np.logical_not(np.isfinite(resid))
    resid[invalid] = 0

    n = 5
    kernel = np.array([np.ones(n), -1*np.ones(n), np.ones(n)])
    kernel = kernel / np.sum(kernel)

    filtered = median_filter(resid, 15, axes=1)
    filtered = ndimage.convolve(filtered, kernel)

    cut = np.percentile(np.abs(filtered), 95)
    invalid2 = np.abs(filtered) > cut
    resid[invalid2] = 0

    invalid[invalid2] = 0

    return resid, invalid

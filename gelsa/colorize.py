import math
import numba
import numpy as np
from astropy.wcs import WCS
from scipy import ndimage
import skimage
from matplotlib import cm, colors

import reproject
from reproject import mosaicking


from . import visu
from . import spec_crop


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


def make_wcs_from_specframe(frame, ra=None, dec=None, pixel_size_arcsec=1, shape=None, width_deg=0.9, rotate=0):
    """ """
    pa = frame.params['PA']
    if ra is None:
        ra = frame.params['RA']
        dec = frame.params['DEC']

    return make_wcs(ra, dec, pa,
                    rotate=rotate,
                    pixel_size_arcsec=pixel_size_arcsec,
                    shape=shape,
                    width_deg=width_deg)

#     orientation = np.deg2rad(-pa + 90 + rotate)
#     cd_matrix = np.array([[np.cos(orientation), -np.sin(orientation)],
#                           [np.sin(orientation), np.cos(orientation)]])
#     rot_matrix = np.array([[0, -1], [1, 0]])
#     cd_matrix = rot_matrix.dot(cd_matrix)
#     cd_matrix *= pixel_size_arcsec / 3600

#     if shape is None:
#         res = pixel_size_arcsec/3600
#         n = int(width_deg / res)
#         shape = (n, n)

#     wcs_out = WCS()
#     wcs_out.wcs.ctype = ('RA---TAN', 'DEC--TAN')
#     wcs_out.wcs.cd = cd_matrix
#     wcs_out.wcs.crval = np.array([ra, dec], dtype=np.float64)
#     wcs_out.wcs.crpix = (shape[0]//2, shape[1]//2)
#     wcs_out.array_shape = shape
#     return wcs_out


def composite(im_rgb, brightness, mask=None, levels=(20, 99), normalize=True):
    """ """

    im_hsv = skimage.color.rgb2hsv(im_rgb)
    if normalize:
        brightness_norm = visu.normalize_image(brightness, levels=levels)
    else:
        brightness_norm = brightness
    im_hsv[:, :, 2] = brightness
    im_rgb = skimage.color.hsv2rgb(im_hsv)
    if mask is not None:
        valid = mask > 0
        im_rgb[:,:,0] *= valid
        im_rgb[:,:,1] *= valid
        im_rgb[:,:,2] *= valid
    # return (im_rgb * 255.9999999).astype('uint8')
    return im_rgb


def mosaic_image(image_list, wcs_out):
    """ """
    image, foot = mosaicking.reproject_and_coadd(
        image_list,
        wcs_out,
        shape_out=wcs_out.array_shape,
        reproject_function=reproject.reproject_interp
    )
    return image, foot




@numba.njit(cache=True)
def histogram_rgb(y, x, weight, bins_y, bins_x, hist):
    """Histogram 2D"""
    n = len(x)
    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    for i in range(n):
        if y[i] < bins_y[0]:
            continue
        if y[i] > bins_y[-1]:
            continue
        if x[i] < bins_x[0]:
            continue
        if x[i] > bins_x[-1]:
            continue
        ind_y = int((y[i] - bins_y[0]) / step_y)
        ind_x = int((x[i] - bins_x[0]) / step_x)
        hist[ind_y, ind_x, 0] += weight[i, 0]
        hist[ind_y, ind_x, 1] += weight[i, 1]
        hist[ind_y, ind_x, 2] += weight[i, 2]


@numba.njit(cache=True, nogil=True)
def _histogram2d_accumulate(y, x, weight0, bins_y, bins_x, hist):
    """Histogram 2D accumulate into a 2D array — serial."""
    n = len(x)

    if n == 0:
        return

    ny = len(bins_y) - 1
    nx = len(bins_x) - 1

    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    for i in range(n):
        ind_y = math.floor((y[i] - bins_y[0]) / step_y)
        if ind_y < 0 or ind_y >= ny:
            continue
        ind_x = math.floor((x[i] - bins_x[0]) / step_x)
        if ind_x < 0 or ind_x >= nx:
            continue
        hist[ind_y, ind_x] += weight0


# def draw_line(image, x0, y0, x1, y1, rgb0, rgb1):
    # """ """


def colorize(specframe, image, wcs, fixed_wavelength, wcs_out=None, 
             level=99, alpha=2, wave_step=10, wavelength_range=None,
             cmap='rainbow'):
    """ """
    if wcs_out is None:
        wcs_out = wcs

    image_ = image**alpha

    thresh = np.percentile(image_, level)

    xx = np.arange(image.shape[1])
    yy = np.arange(image.shape[0])
    xx, yy = np.meshgrid(xx, yy)

    sel = image_ >= thresh
    xx = xx[sel]
    yy = yy[sel]
    im_sel = image_[sel]

    ra_, dec_ = wcs.all_pix2world(xx, yy, 0)

    wave_start, wave_end = specframe.params['wavelength_range']
    nbins = int((wave_end - wave_start) / wave_step + 1)
    wavelength_trace = np.linspace(*specframe.params['wavelength_range'], nbins)
    
    if wavelength_range is None:
        wavelength_range = specframe.params['wavelength_range']

    norm = colors.Normalize(wavelength_range[0], wavelength_range[-1])
    smap = cm.ScalarMappable(norm, cmap=cmap)
    rgba = smap.to_rgba(wavelength_trace)

    out_shape = wcs_out.array_shape

    im_rgb = np.zeros((out_shape[0], out_shape[1], 3), dtype=float)
    im_norm = np.zeros(out_shape, dtype=float)

    bin_y = np.arange(out_shape[0]+1)
    bin_x = np.arange(out_shape[1]+1)

    for i in range(len(ra_)):
        flux_weight = im_sel[i]

        x, y = specframe.framecoord.radec_to_fov(
            ra_[i]*np.ones(len(wavelength_trace)),
            dec_[i]*np.ones(len(wavelength_trace)),
            wavelength_trace
        )

        rgba_sel = rgba

        # map to RA, Dec at fixed wavelength
        ra_tmp, dec_tmp = specframe.framecoord.fov_to_radec(x, y, fixed_wavelength)
        x, y = wcs_out.wcs_world2pix(ra_tmp, dec_tmp, 0)

        valid = (x > 0) & (y > 0) & (x < out_shape[1]) & (y < out_shape[0])

        x = x[valid]
        if len(x) == 0:
            continue
        y = y[valid]
        rgba_sel = rgba_sel[valid]

        histogram_rgb(y, x, rgba_sel*flux_weight, bin_y, bin_x, im_rgb)
        _histogram2d_accumulate(y, x, flux_weight, bin_y, bin_x, im_norm)

    valid = im_norm > 0
    im_rgb[valid, 0] = im_rgb[valid, 0] / im_norm[valid]
    im_rgb[valid, 1] = im_rgb[valid, 1] / im_norm[valid]
    im_rgb[valid, 2] = im_rgb[valid, 2] / im_norm[valid]

    return im_rgb


def reproject_specframe(frame_list, wcs_out, fixed_wavelength=15500, width=2000, ndrops=1,
                        subtract_continuum=False,
                        median_filter_size_pix=50):
    """ """
    image_stack = np.zeros(wcs_out.array_shape, dtype='d')
    norm_stack = np.zeros(wcs_out.array_shape, dtype='d')

    mask_stack = np.zeros(wcs_out.array_shape, dtype=int)

    x = np.random.uniform(0, 1, 10000)*wcs_out.array_shape[1]
    y = np.random.uniform(0, 1, 10000)*wcs_out.array_shape[0]
    ra_, dec_ = wcs_out.all_pix2world(x, y, 0)

    for frame in frame_list:

        frame.params['apply_median_filter'] = subtract_continuum
        frame.params['median_filter_size_pix'] = median_filter_size_pix

        x, y, detectors = frame.radec_to_pixel(ra_, dec_, fixed_wavelength)
        detector_list = np.unique(detectors)

        for det in detector_list:
            if det < 0:
                continue

            det_sel = detectors == det
            ra_c = np.mean(ra_[det_sel])
            dec_c = np.mean(dec_[det_sel])

            # print(f"Extracting detector {det}")
            image, mask, var = frame.get_detector(det)
            shape = image.shape
            crop = spec_crop.SpecCrop(
                image, mask, var,
                detector=det,
                bbox=[0, shape[1], 0, shape[0]],
                frame=frame
            )
            try:
                image_out, var_image, norm_image, pix_bins = crop.resample_on_radec(
                    ra_c, dec_c,
                    fixed_wavelength,
                    width=width,
                    wcs=wcs_out,
                    ndrops=ndrops,
                    pixel_shrink=1,
                    use_rel_flux_calib=False,
                    use_abs_flux_calib=False,
                )
            except:
                continue
            valid = norm_image > 0
            spec = image_out * 0
            spec[valid] = image_out[valid]

            image_stack[valid] += image_out[valid]
            norm_stack[valid] += norm_image[valid]

    valid = norm_stack > 0
    image_stack[valid] = image_stack[valid] / norm_stack[valid]

    # valid = mask_stack > 0
    # image_stack = image_stack.astype('f4')
    # image_stack[~valid] = -99

    return image_stack, norm_stack > 0


def spectro_composite(spec_im, im_rgb, wcs_in, wcs_out, levels=(5,98), power=1):
    """ """
    import reproject
    mask = spec_im > 0

    source_im = visu.normalize_image(spec_im, levels=levels, power=power)

    shape = wcs_out.array_shape
    image_rgb_big = np.zeros((shape[0], shape[1], 3))

    for i in range(3):
        im_proj = reproject.reproject_interp(
            (im_rgb[:, :, i], wcs_in), wcs_out, wcs_out.array_shape,)
        image_rgb_big[:, :, i], _ = ndimage.gaussian_filter(im_proj, 1)

    return composite(image_rgb_big, source_im, mask, normalize=False)



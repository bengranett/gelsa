import sys
import numpy as np
from scipy import interpolate, ndimage
from astropy.io import fits
from astropy.table import Table

from . import utils
from . import sample_dist
from . import consts
from . import histogram
from . import overlaps

from numba import int64
from numba.typed import Dict


class SpectralImager:

    params = {
        'wavelength_range': (9000, 19000),
        'wavelength_step': 1.,
        'nphot_max': 100000,
        'workdir': '.',
        'datadir': '.',
        'mask_zero_orders': True,
        'zero_order_cat_path': 'zero_order_cat.fits',
    }

    def __init__(self, specframe, **kwargs):
        """ """
        self.specframe = specframe
        self.framecoord = specframe.framecoord

        self.params = self.__class__.params.copy()
        self.params.update(kwargs)

        self.params.update(specframe.params)
        self.params['wavelength_range'] = specframe.params['wavelength_range']
        # print(f"spec imager wavelength range {self.params['wavelength_range']}")

        self.wave = np.arange(
            self.params['wavelength_range'][0],
            self.params['wavelength_range'][1],
            self.params['wavelength_step']
        )
        self.detector_shape = specframe.detector_shape

        self.sigma = np.sqrt(specframe.params['sigma2_det'] * specframe.params['exptime_sec'])

        self.init_image()

    def init_image(self):
        """ """
        shape = self.detector_shape
        bin_y = np.arange(0, shape[1]+1, 1, dtype='d')
        bin_x = np.arange(0, shape[0]+1, 1, dtype='d')
        self.pixel_grid = (bin_y, bin_x)

    def sample_spectrum(self, galaxy):
        """Generate samples of wavelength"""

        flux_spectrum = galaxy.sed(self.wave)# + galaxy.emline(self.wave)

        flux_spectrum[flux_spectrum<0] = 0

#         print(flux_spectrum)
        counts_spectrum = self.specframe.flux_to_counts(flux_spectrum, self.wave)


        func = sample_dist.SampleDistribution(self.wave, counts_spectrum)


        counts = np.random.poisson(np.sum(counts_spectrum))

        # print('total counts:', counts)

        if counts == 0:
            return np.array([]), 0
#         print(f"counts {counts}")

        weight = 1
        if counts > self.params['nphot_max']:
            weight = counts / self.params['nphot_max']
            counts = self.params['nphot_max']
            # print(f"alert {weight=}", file=sys.stderr)

        try:
            samples = func.sample(counts)

        except ValueError:
            return np.array([]), 0

        return samples, weight

    def sample_emline(self, galaxy, line):

        redshift = galaxy.params['redshift']
        wavelength_obs = (1 + redshift) * consts.lines[line]

       # else:
        sigma_size = galaxy.params['velocity_disp']/consts.c*wavelength_obs

        if wavelength_obs<galaxy.params['obs_wavelength_range'][0] or wavelength_obs>galaxy.params['obs_wavelength_range'][1]:
            counts_emline = 0
        else:
            counts_emline = self.specframe.lineflux_to_counts(galaxy.params['fluxes_emlines'][line], wavelength_obs)
            counts_emline = np.random.poisson(counts_emline)

        weight = 1
        if counts_emline > self.params['nphot_max']:
            weight = counts_emline / self.params['nphot_max']
            counts_emline = self.params['nphot_max']
            # print(f"alert {weight=}", file=sys.stderr)

        samples_emline = np.random.normal(wavelength_obs, sigma_size, counts_emline)

        return samples_emline, weight

    def sample(self, galaxy):
        """ """
        wavelength, weight = self.sample_spectrum(galaxy)

        if len(wavelength) == 0:
            return np.array([]), np.array([]), np.array([]), 0

        ra, dec = galaxy.sample_image(len(wavelength))
        # print('Number of samples is: {}'.format(ra.shape[0]))
        x, y, detector = self.specframe.radec_to_pixel(ra, dec, wavelength)

        # xpsf, ypsf = self.specframe.sample_psf(
        #     x, y, detector,
        #     wavelength_ang=wavelength
        # )

        # x += xpsf
        # y += ypsf

        # get the flux calibration at the first sample
        flux_loss = self.specframe.get_relative_flux_loss_radec(
            ra[0], dec[0]
        )
        weight = weight * flux_loss

        return x, y, detector, weight

    def sample_line(self, galaxy, line):
        """ """
        wavelength, weight = self.sample_emline(galaxy, line)

        if len(wavelength) == 0:
            return np.array([]), np.array([]), np.array([]), 0

        ra, dec = galaxy.sample_image(len(wavelength))

        x, y, detector = self.specframe.radec_to_pixel(ra, dec, wavelength)

        xpsf, ypsf = self.specframe.sample_psf(
            x, y, detector,
            wavelength_ang=wavelength
        )

        x += xpsf
        y += ypsf

        flux_loss = self.specframe.get_relative_flux_loss_radec(
            ra[0], dec[0]
        )
        weight *= flux_loss

        return x, y, detector, weight

    def make_image_from_samples(self, ra, dec, wavelength, weight=None, masks=None, noise=True, return_var=True):
        """ """
        images = {}
        if masks is not None:
            for det, (npix, _) in masks.items():
                images[det] = np.zeros(npix, dtype='d')

        if len(ra) == 0:
            return images

        x, y, detector = self.specframe.radec_to_pixel(ra, dec, wavelength)
        if weight is None:
            weight = np.ones(len(x))

        ra_mean = np.mean(ra)
        dec_mean = np.mean(dec)

        flux_loss = self.specframe.get_relative_flux_loss_radec(
            ra_mean, dec_mean
        )
        weight = weight * flux_loss

        for d in np.unique(detector):
            if d < 0:
                continue
            if masks is not None:
                if d not in masks:
                    continue
            sel = detector == d
            if d not in images:
                images[d] = np.zeros(self.specframe.detector_shape, dtype='d')

            if masks is not None:
                histogram.histogram2d_accumulate_mask(
                    y[sel], x[sel], weight[sel],
                    bins_y=self.pixel_grid[0],
                    bins_x=self.pixel_grid[1],
                    hist=images[d],
                    mask_dict=masks[d][1],
                )
            else:
                histogram.histogram2d_accumulate(
                    y[sel], x[sel], weight[sel],
                    bins_y=self.pixel_grid[0],
                    bins_x=self.pixel_grid[1],
                    hist=images[d],
                )
        # remove axes with length 1
        for d, image in images.items():
            images[d] = np.squeeze(image)

        return images

    def make_image(self, galaxy_list, masks=None, noise=True, return_var=True, show_progress=False):
        """Builds image and variance image"""
        images = {}
        if masks is not None:
            for det, (npix, _) in masks.items():
                images[det] = np.zeros(npix, dtype='d')

        for i, g in enumerate(galaxy_list):
            x, y, detector, weight = self.sample(g)
            for d in np.unique(detector):
                if d < 0:
                    continue
                if masks is not None:
                    if d not in masks:
                        continue
                sel = detector == d
                if d not in images:
                    images[d] = np.zeros(self.specframe.detector_shape, dtype='d')

                # weight is a scalar, convert to vector
                weight_ = weight * np.ones(np.sum(sel))

                if masks is not None:
                    histogram.histogram2d_accumulate_mask(
                        y[sel], x[sel], weight_,
                        bins_y=self.pixel_grid[0],
                        bins_x=self.pixel_grid[1],
                        hist=images[d],
                        mask_dict=masks[d][1],
                    )
                else:
                    histogram.histogram2d_accumulate(
                        y[sel], x[sel], weight_,
                        bins_y=self.pixel_grid[0],
                        bins_x=self.pixel_grid[1],
                        hist=images[d],
                    )

            for l in range(len(consts.lines)):
                x, y, detector, weight = self.sample_line(g, l)
                for d in np.unique(detector):
                    if d < 0:
                        continue
                    if masks is not None:
                        if d not in masks:
                            continue
                    sel = detector == d
                    if d not in images:
                        images[d] = np.zeros(self.specframe.detector_shape, dtype='d')

                    # print(f"counts {np.sum(sel)}")
                    # weight is a scalar, convert to vector
                    weight_ = weight * np.ones(np.sum(sel))

                    if masks is not None:
                        histogram.histogram2d_accumulate_mask(
                            y[sel], x[sel], weight_,
                            bins_y=self.pixel_grid[0],
                            bins_x=self.pixel_grid[1],
                            hist=images[d],
                            mask_dict=masks[d][1],
                        )
                    else:
                        histogram.histogram2d_accumulate(
                            y[sel], x[sel], weight_,
                            bins_y=self.pixel_grid[0],
                            bins_x=self.pixel_grid[1],
                            hist=images[d],
                        )
            if show_progress:
                print(f"\r spectrum {i}/{len(galaxy_list)}", end="", flush=True)

        # remove axes with length 1
        for d, image in images.items():
            images[d] = np.squeeze(image)

        if return_var:
            var_images = {}
            # poisson variance is equal to mean
            for d, image in images.items():
                var_images[d] = image + self.sigma**2

        if noise:
            # add detector noise background
            for d, image in images.items():
                image += np.random.normal(0, self.sigma, image.shape)

        if return_var:
            return images, var_images
        else:
            return images

    def make_mask(self, galaxy_list, input_mask=None, width=2, iterations_min=3):
        """ """
        galaxy_list = utils.ensurelist(galaxy_list)

        mask_images = {}

        for gal_i, gal in enumerate(galaxy_list):

            ra, dec = gal.sample_image(100000)
            wavelength = np.random.uniform(*self.params['wavelength_range'], len(ra))
            x, y, detector = self.specframe.radec_to_pixel(ra, dec, wavelength)

            weight=np.ones(len(x))

            for d in np.unique(detector):
                if d < 0:
                    continue
                if d not in input_mask:
                    continue
                sel = detector == d

                im = histogram.histogram2d(y[sel], x[sel], weight[sel],
                                           bins_y=self.pixel_grid[0], bins_x=self.pixel_grid[1])
                im = im > 0
                im = ndimage.gaussian_filter(im.astype(float), 1)

                im = im > 0.1
                # radius = self.specframe.arcsec_to_pixel(
                    # galaxy_list[gal_i].halflight_radius
                # )

                # iterations = max(iterations_min, int(np.round(width*radius)))

                # im = ndimage.binary_dilation(
                    # im,
                    # structure=ndimage.generate_binary_structure(2, 2),
                    # iterations=iterations
                # )
                if d not in mask_images:
                    mask_images[d] = im
                else:
                    mask_images[d] += im

        if input_mask is not None:
            for d in mask_images.keys():
                mask_images[d] *= input_mask[d]

        index_mask_list = {}
        for d in mask_images.keys():
            # values outside of mask are -1
            index_mask = np.zeros(mask_images[d].shape, dtype=int) - 1
            # values inside of mask are set to an index 0,1,2,3...
            sel = mask_images[d] > 0
            num_unmasked_pix = np.sum(sel)
            index_mask[sel] = np.arange(np.sum(sel))
            # store a tuple with the number of unmasked pixels and the indices
            index_mask_list[d] = num_unmasked_pix, index_mask

        return index_mask_list

    def check_overlap(self, gal1, gal2, wavelength_step=20):
        """Determine if two sources overlap in the dispersed image.

        Parameters
        ----------
        ra1
        dec1
        ra2
        dec2

        Returns
        -------
        bool
        """
        wave_trace = np.arange(
            self.params['wavelength_range'][0],
            self.params['wavelength_range'][1],
            wavelength_step
        )
        x1, y1, detector1 = self.specframe.radec_to_pixel(
            gal1.params['ra']*np.ones(len(wave_trace)),
            gal1.params['dec']*np.ones(len(wave_trace)),
            wave_trace,
            objid=gal1.params['id']
        )
        x2, y2, detector2 = self.specframe.radec_to_pixel(
            gal2.params['ra']*np.ones(len(wave_trace)),
            gal2.params['dec']*np.ones(len(wave_trace)),
            wave_trace,
            objid=gal2.params['id']
        )

        separation = (gal1.halflight_radius + gal2.halflight_radius) / 0.3 * 2
        separation = max(10, separation)
        # print(f"separation {separation} pixels")
        # print(np.unique(detector1), np.unique(detector2))
        for d1 in np.unique(detector1):
            if d1 < 0:
                continue
            sel1 = detector1 == d1
            sel2 = detector2 == d1
            if np.sum(sel2) == 0:
                # obj2 not on same detector
                continue
            # print(f"on same detector {np.sum(sel1)} {np.sum(sel2)}")
            overlap = overlaps.check_segment_overlap(
                x1[sel1], y1[sel1],
                x2[sel2], y2[sel2],
                separation=separation
            )
            if overlap:
                return True
        return False


    # def write(self, filename, **kwargs):
    #     """Write image to a FITS file"""
    #     hdu = fits.PrimaryHDU()
    #     image_hdu = fits.ImageHDU(data=self.image, header=self.specframe.wcs.to_header())
    #     hdul = fits.HDUList([hdu, image_hdu])
    #     hdul.writeto(filename, **kwargs)

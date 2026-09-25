import sys
import numpy as np
from scipy.ndimage import gaussian_filter

from . import sample_dist
from . import profile_sampler
from . import consts

# from . import gravlens

class Galaxy:
    _default_params = {
        'object_id': 0,
        'ra': 0,
        'dec': 0,
        'redshift': 1.0,
        'fwhm_arcsec': 1.,
        'profile': 'gaussian',
        'bulge_semimajor_axis': 1.,
        'disk_semimajor_axis': 1.,
        'bulge_fraction': 0.2,
        # 'bulge_r50': 0,
        # 'disk_r50': 0,
        'bulge_sersic_index': 4,
        'pa': 0,
        'axis_ratio': None,
        'bulge_axis_ratio': 1,
        'disk_axis_ratio': 1,
        'obs_wavelength_step': 1,
        'obs_wavelength_range': (9200., 19000.),
        'continuum_params': (15000, -1e-5, -18),
        'fluxes_emlines' : None,
        'velocity_disp': 1.2e15, #angstrom/s
        'nlines': 11,
        'id': 0,
        'lens': False,
        'wavelength': None,
    }

    def __init__(self, **kwargs):
        """ """
        self.params = self._default_params.copy()
        self.params['fluxes_emlines'] = np.zeros(self.params['nlines'])

        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise ValueError(f"Unknown galaxy parameter {key}")

        if self.params['axis_ratio'] is not None:
            # if 'bulge_axis_ratio' in kwargs:
            #     print("Warning bulge_axis_ratio and axis_ratio cannot be set simultaneously. Using bulge_axis_ratio=axis_ratio.", file=sys.stderr)
            # if 'disk_axis_ratio' in kwargs:
            #     print("Warning disk_axis_ratio and axis_ratio cannot be set simultaneously. Using disk_axis_ratio=axis_ratio.", file=sys.stderr)
            self.params['bulge_axis_ratio'] = self.params['axis_ratio']
            self.params['disk_axis_ratio'] = self.params['axis_ratio']

        # self.correct_radius()

        if self.params['wavelength'] is None:
            self.wavelength = np.arange(
                self.params['obs_wavelength_range'][0],
                self.params['obs_wavelength_range'][1],
                self.params['obs_wavelength_step']
            )
            self.params['wavelength'] = self.wavelength
        else:
            self.wavelength = self.params['wavelength']
            if len(self.wavelength) < 2:
                raise ValueError
            self.params['obs_wavelength_range'] = self.params['wavelength'][0], self.params['wavelength'][-1]
            self.params['obs_wavelength_step'] = self.params['wavelength'][1] - self.params['wavelength'][0]

        self.init_sed()

        # if self.params['lens']:
            # self.lens_imager = gravlens.LensImager()


    def copy(self):
        params = self.params.copy()
        clone = type(self)(**params)
        try:
            clone._stamp = self._stamp
            clone._segmap = self._segmap
            clone._wcs = self._wcs
        except AttributeError:
            pass
        return clone

    def _correct_radius(self, semimajor_axis, axis_ratio):
        """
        Correct the axis such that semimajor_axis is the simulated semimajor
        axis and the semiminor axis is semimajor_axis*axis_ratio.
        This is the convenction used in flagship for bulge_r50 and disk_r50.
        Check https://gitlab.euclid-sgs.uk/fpassala/euclidtools/-/wikis/morphology
        for more info.
        """
        return semimajor_axis*np.sqrt(axis_ratio)

    def correct_radius(self):
        if self.params['axis_ratio'] is not None:
            self.params['bulge_r50'] = self._correct_radius(self.params['bulge_semimajor_axis'], self.params['axis_ratio'])
            self.params['disk_r50'] = self._correct_radius(self.params['disk_semimajor_axis'], self.params['axis_ratio'])
        else:
            self.params['bulge_r50'] = self._correct_radius(self.params['bulge_semimajor_axis'], self.params['bulge_axis_ratio'])
            self.params['disk_r50'] = self._correct_radius(self.params['disk_semimajor_axis'], self.params['disk_axis_ratio'])




 #   def init_emline(self):

 #       redshift = self.params['redshift']
 #       wavelength_obs = (1 + self.params['redshift']) * consts.rest_wavelength_ha
 #       sigma_size = self.params['velocity_disp']/consts.c*wavelength_obs
 #       ampl = self.params['fha']/np.sqrt(2 * np.pi * sigma_size**2)

 #       emline_range=(wavelength_obs-5*sigma_size, wavelength_obs+5*sigma_size)
 #       self.emline_wave = np.arange(
 #           *emline_range,
 #           sigma_size/10.)

 #       n=500

 #       self.emline = ampl * np.random.normal(wavelength_obs, sigma_size, n)
        #self.emline = ampl * np.exp(-(self.emline_wave - wavelength_obs)**2/(2 * sigma_size**2))

    def init_sed(self):
        """Take care of units"""
        x, a, b = self.params['continuum_params']
        # print(x,a,b)
        # print(self.wavelength)
        self.sed = 10**((self.wavelength - x) * a + b)
        # print(10**((self.wavelength - x) * a + b))

    @property
    def profile_sampler(self):
        try:
            return self._profile_sampler
        except AttributeError:

            if self.params['profile'][0].lower() == 'g': # profile name starts with g for gaussian
                # profile is gaussian
                # print("initalizing gaussian profile")
                self._profile_sampler = self._sample_gaussian
            elif self.params['profile'][0].lower() == 's':  # s for sersic
                # profile is sersic + disk
                # print("initalizing sersic profile")
                self._profile_sampler = self._sample_sersic_bulge_and_disk
            elif self.params['profile'][0].lower() == 'i': # image
                self._profile_sampler = self.sample_from_segmap
            else:
                # profile is bulgydisk
                # print("initalizing bulgy disk profile")
                self._profile_sampler = self._sample_bulgy_disk
            return self._profile_sampler

    @property
    def sed(self):
        return self._sed

    @sed.setter
    def sed(self, y):
        self.sed_params = y
        self._sed = sample_dist.SampleDistribution(self.wavelength, y)

    @property
    def halflight_radius(self):
        """Return the half-light radius in arcsec"""
        try:
            return self._halflight_radius
        except AttributeError:
            pass
        # sample the profile to compute the half light radius
        coord_xy = self.profile_sampler(1e4)
        r = np.sqrt(coord_xy[:, 0]**2 + coord_xy[:, 1]**2)
        self._halflight_radius = np.median(r)

        return self._halflight_radius

    def set_sed(self, wavelength, flux):
        """ """
        wave_obs = wavelength * (1 + self.params['redshift'])
        self.sed = np.interp(self.wavelength, wave_obs, flux)

    def set_flux_line(self, **kwargs):
        """ """
        for name, flux in kwargs.items():
            i = consts.line_indices[name]
            self.params['fluxes_emlines'][i] = flux

    def set_flux_Ha(self, flux_Ha=1e-15, NII_Ha_ratio=0.2):
        """ """
        self.set_flux_line(Ha=flux_Ha)
        flux_N2 = flux_Ha * NII_Ha_ratio
        flux_N2a = flux_N2 * 1./4
        flux_N2b = flux_N2 * 3./4
        self.set_flux_line(N2_6549=flux_N2a)
        self.set_flux_line(N2_6585=flux_N2b)

    def set_flux_HbO3(self, flux_Hb=1e-16, flux_O3_5008=1e-16):
        """ """
        self.set_flux_line(Hb=flux_Hb)
        flux_O3_4960 = flux_O3_5008 / 3.
        self.set_flux_line(O3_5008=flux_O3_5008)
        self.set_flux_line(O3_4960=flux_O3_4960)

 #   @property
 #   def emline(self):
 #       return self._emline

 #   @emline.setter
 #   def emline(self,y):

 #       redshift = self.params['redshift']
 #       wavelength_obs = (1 + self.params['redshift']) * consts.rest_wavelength_ha
 #       sigma_size = self.params['velocity_disp']/consts.c*wavelength_obs
 #       ampl = self.params['fha']/np.sqrt(2 * np.pi * sigma_size**2)

 #       emline_range=(wavelength_obs-5*sigma_size, wavelength_obs+5*sigma_size)
 #       self.emline_wave = np.arange(
 #           *emline_range,
 #           sigma_size/10.)

 #       n=500

 #       self.emline = ampl * np.random.normal(wavelength_obs, sigma_size, n)
 #       self._emline = interpolate.interp1d(self.emline_wave,y)

    def _shear_coord(self, coord, axis_ratio):
        """ """
        root_axis_ratio = np.sqrt(axis_ratio)
        coord *= np.array([root_axis_ratio, 1./root_axis_ratio])
        return coord

    def _sample_gaussian(self, n):
        """ """
        if n == 0:
            return np.zeros((n, 2))
        coord = profile_sampler.fwhm_gaussian_sampler(n) * self.params['fwhm_arcsec']
        #coord = profile_sampler.fwhm_gaussian_sampler(n,self.params['fwhm_arcsec'])
        #print('FWHM used in sampler:', self.params['fwhm_arcsec'])
        return self._shear_coord(coord, self.params['axis_ratio'])

    def _sample_bulge(self, n, axis_ratio):
        """ """
        if n == 0:
            return np.zeros((n, 2))
        coord = profile_sampler.bulge_sampler(n) * self.params['bulge_r50']
        return self._shear_coord(coord, axis_ratio)

    def _sample_disk(self, n, axis_ratio):
        """ """
        if n == 0:
            return np.zeros((n, 2))
        coord = profile_sampler.disk_sampler(n) * self.params['disk_r50']
        return self._shear_coord(coord, axis_ratio)

    def _sample_bulgy_disk(self, n):
        """ """
        if n == 0:
            return np.zeros((n, 2))

        n_bulge = int(np.round(self.params['bulge_fraction'] * n))
        n_disk = n - n_bulge
        samples = []
        if n_bulge > 0:
            samples.append(self._sample_bulge(n_bulge, self.params['bulge_axis_ratio']))
        if n_disk > 0:
            samples.append(self._sample_disk(n_disk, self.params['disk_axis_ratio']))

        return np.vstack(samples)

    def _sample_sersic_bulge(self, n, axis_ratio):
        """ """
        if n == 0:
            return np.zeros((n, 2))
        coord = profile_sampler.sersic_sampler(n, self.params['bulge_sersic_index']) * self.params['bulge_r50']
        return self._shear_coord(coord, axis_ratio)

    def _sample_sersic_bulge_and_disk(self, n):
        """ """
        if n == 0:
            return np.zeros((n, 2))

        n_bulge = int(np.round(self.params['bulge_fraction'] * n))
        n_disk = n - n_bulge
        samples = []
        if n_bulge > 0:
            samples.append(self._sample_sersic_bulge(n_bulge, self.params['bulge_axis_ratio']))
        if n_disk > 0:
            samples.append(self._sample_disk(n_disk, self.params['disk_axis_ratio']))

        return np.vstack(samples)

    def sample_image(self, n):
        """Draw samples from the galaxy image"""
        if self.params['profile'].lower()[0] == 'i':
            return self.sample_from_segmap(n, rotate_center=False).transpose()
        coord_xy = self.profile_sampler(n)

        # rotate to position angle
        angle = np.deg2rad(self.params['pa'])
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        rotation_mat = np.array([[cosangle, -sinangle], [sinangle, cosangle]])
        coord_xy = coord_xy @ rotation_mat

        # convert arcsec to deg
        coord_xy /= 3600

        x, y = coord_xy.transpose()

        if self.params['lens']:
            print("Running lens imager...", len(x))
            x, y = self.lens_imager(x, y)
            print("Done", len(x))


        # rotate to RA, Dec on sky
        x = self.params['ra'] + x / np.cos(np.deg2rad(self.params['dec']))
        y = self.params['dec'] + y

        return x, y

    def set_image(self, stamp, segmap, wcs):
        """ """
        self._stamp = stamp
        self._segmap = segmap
        self._wcs = wcs

    def sample_from_segmap(self, n, rotate_center=True):
        """ """
        n = int(n)
        # print("sampling from segmap", n)
        # print("offset 1")
        if self.params['object_id'] == 0:
            ra, dec = self.params['ra'], self.params['dec']
            x, y = self._wcs.all_world2pix(ra, dec, 0)
            x = x.astype(int)
            y = y.astype(int)
            object_id = self._segmap[y, x]
        else:
            object_id = self.params['object_id']
        if object_id <= 0:
            raise ValueError(f"segmentation map return invalid object ID for these coordinates {ra=}, {dec=}; {x=}, {y=} {object_id=}")
        obj_sel = self._segmap == object_id
        if np.sum(obj_sel) == 0:
            raise ValueError(f"no pixels selected from segmentation map at {object_id=}")
        y_obj, x_obj = np.where(obj_sel)
        v_obj = self._stamp[obj_sel].copy()
        v_obj[v_obj < 0] = 0
        if np.sum(v_obj) == 0:
            raise ValueError(f"image is 0")
        ind = np.random.choice(len(v_obj), p=v_obj/np.sum(v_obj), size=n, replace=True)
        x_obj_s = x_obj[ind]
        y_obj_s = y_obj[ind]
        # print(f"w=0.5")
        offx, offy = np.random.uniform(-0.5, 0.5, (2, n))
        ra_obj, dec_obj = self._wcs.all_pix2world(x_obj_s + offx, y_obj_s+offy, 0)
        if rotate_center:
            ra_obj = (ra_obj - self.params['ra']) * np.cos(np.radians(self.params['dec']))
            dec_obj = dec_obj - self.params['dec']
        return np.transpose([ra_obj, dec_obj])

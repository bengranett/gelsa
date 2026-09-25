import numpy as np
from astropy.table import vstack, unique

from . import optimize, decontam_utils


class Decontamination:

    _default_params = {
        'wavelength_step': 13.,
        'contaminant_separation_arcsec': 5,
        'minimum_separation_arcsec': 0.3,
        'reg_lambda_list': [0.1, 1, 10, 100, 1000],
        'reg_floor': 1e-4,
        'nsamples': 10000,
        'sigma_clip': 10,
        'nthreads': 4,
        'smooth': True,
        'masking_iterations': 1,
        'compute_var': False,
    }

    def __init__(self, **kwargs):
        """ """
        self.params = self._default_params.copy()
        self.params.update(kwargs)
        self._id_counter = 0
        self.target_list = []
        self.contaminants = None
        self.cat = None
        self.image_list = []
        self.specimager_list = []
        self.frame_list = []
        self._pointing_id_list = []

    def add_target(self, ra, dec, segmap_id=None, object_id=None, image=None, segmap=None, wcs=None, tile=None):
        """ """
        if segmap_id is None:
            if image is None:
                segmap_id = None
            else:
                self._id_counter += 1
                segmap_id = self._id_counter

        if segmap is not None:
            if not np.isin(segmap_id, segmap):
                raise ValueError(f"segmap id {segmap_id} is not in the segmap array")

        target = dict(
            segmap_id=segmap_id,
            object_id=object_id,
            ra=ra,
            dec=dec,
            tile=tile,
            image=image,
            segmap=segmap,
            wcs=wcs
        )
        self.target_list.append(target)

    def add_catalog(self, cat):
        """ """
        if self.cat is None:
            self.cat = cat
        else:
            self.cat = vstack([self.cat, cat])
        self.cat = unique(self.cat, keys='OBJECT_ID')
        self.cat.add_index('OBJECT_ID')
        self.cat.add_index('SEGMENTATION_MAP_ID')

    def add_image(self, image, segmap, wcs, tile=None):
        """ """
        self.image_list.append(
            dict(
                image=image,
                segmap=segmap,
                wcs=wcs,
                tile=tile
            )
        )

    def add_frame(self, *frame_list):
        """ """
        for frame in frame_list:
            pointing_id = frame.params['PTGID']
            if pointing_id in self._pointing_id_list:
                print(f"skip frame with duplicate pointing_id {pointing_id}")
                continue
            self.frame_list.append(frame)
            self._pointing_id_list.append(pointing_id)

    def build_contaminant_list(self):
        """ """
        self.contaminants = decontam_utils.build_contaminant_list(
                self.frame_list,
                self.target_list,
                self.cat,
                separation_arcsec=self.params['contaminant_separation_arcsec'],
                minimum_separation_arcsec=self.params['minimum_separation_arcsec']
        )
        return self.contaminants

    def _get_wavelength_range(self):
        """ """
        wavelength_min = 1e6
        wavelength_max = 0
        for frame in self.frame_list:
            low, high = frame.params['wavelength_range']
            wavelength_min = min(wavelength_min, low)
            wavelength_max = max(wavelength_max, high)
        return wavelength_min, wavelength_max

    def fit(self, **params):
        """ """
        for key, val in params.items():
            if key in self.params:
                self.params[key] = val

        wavelength_range = self._get_wavelength_range()

        contaminants = self.build_contaminant_list()

        target_gal_list = decontam_utils.build_target_gal_list(
            self.target_list, self.image_list,
            wavelength_range=wavelength_range,
            wavelength_step=self.params['wavelength_step']
        )

        contam_gal_list = decontam_utils.build_gal_list(
            contaminants, self.image_list,
            wavelength_range=wavelength_range,
            wavelength_step=self.params['wavelength_step']
        )

        gal_list = target_gal_list + contam_gal_list

        specimager_list, images, var_images, pix_mask = decontam_utils.pack_images(
            self.frame_list, gal_list)

        self.specimager_list = specimager_list

        fits  = optimize.measure_everything(
            images,
            var_images,
            specimager_list,
            target_galaxies=target_gal_list,
            contaminant_galaxies=contam_gal_list,
            pixmask_list=pix_mask,
            reg_lambda_list=self.params['reg_lambda_list'],
            dx=self.params['nsamples'],
            n_jobs=self.params['nthreads'],
            masking_iterations=self.params['masking_iterations'],
            sigma_clip=self.params['sigma_clip'],
            reg_floor=self.params['reg_floor'],
            smooth=self.params['smooth'],
            compute_var=self.params['compute_var']
        )
        self._fits = fits
        return fits

    def apply(self, frame):
        """ """
        pass

import numpy as np
from scipy import ndimage

from . import consts
from . import specframe
from .spec_crop import NoExtraction


import time

class Analysis:
    GRISMS = ('RGS000', 'RGS180', 'BGS000')

    def __init__(self, G, datastore=None, **args):
        """ """
        self.G = G
        self.DS = datastore
        self.sir_pack = []

    def __len__(self):
        """ """
        try:
            return len(self.crop_list)
        except AttributeError:
            return len(self.sir_pack)
        
    def read_science_frames(self, DataSetRelease, NirDataSetRelease=None,
                            pointing_id_list=None,
                            grisms=None, **args):
        """Read available SIR science frames

        Parameters
        ----------
        DataSetRelease
        pointing_id_list
        grisms
        """
        if grisms is None:
            grisms = self.GRISMS
        self.sir_pack = self.DS.load_sir_pack(
            DataSetRelease=DataSetRelease,
            NirDataSetRelease=NirDataSetRelease,
            pointing_id_list=pointing_id_list,
            grisms=grisms, **args
        )
        print(f"Loaded {len(self.sir_pack)} SIR science frames")

    def read_fits(self, data_path):
        # Grab all .fits.gz files
        import glob
        import os
        fits_files = glob.glob(os.path.join(data_path, "*.fits"))
        # Construct sir_pack as a list of FITS
        self.sir_pack = []
        for file in fits_files:
            self.sir_pack.append({'frame_path': file})
        print(f"Loaded {len(self.sir_pack)} SIR science frames")

    def read_file_list(self, fits_files):
        """ """
        # Construct sir_pack as a list of FITS
        self.sir_pack = []
        for file in fits_files:
            self.sir_pack.append({'frame_path': file})
        print(f"Loaded {len(self.sir_pack)} SIR science frames")

    def add_frame(self, frame, **params):
        """ """
        self.sir_pack.append({'frame': frame})

    def crop(self, ra, dec, redshift=None, padx=5, pady=25, index_select=None, median_filter_size=0, select_detector=None, continuum_subtraction=None, progress_hook=None):
        """Process the crops

        Parameters
        ----------
        ra
        dec
        redshift
        """
        self.ra = ra
        self.dec = dec
        self.redshift = redshift
        self.padx = padx
        self.pady = pady

        self.crop_list = []

        for i, pack in enumerate(self.sir_pack):
            if index_select is not None:
                if i not in index_select:
                    if progress_hook is not None:
                        progress_hook(1)
                    continue
            if 'frame' in pack:
                # load a frame instance
                frame = pack['frame']
            else:
                # load a frame from a file
                frame = self.G.load_frame(**pack, continuum_subtraction=continuum_subtraction)

            if median_filter_size > 0:
                frame.params['apply_median_filter'] = True
                frame.params['median_filter_size_pix'] = median_filter_size

            try:
                det_crops = frame.cutout(ra, dec, redshift, padx=padx, pady=pady,
                                         select_detector=select_detector)
            except ValueError:
                if progress_hook is not None:
                    progress_hook(1)
                continue

            frame.close_frame()

            self.crop_list.append(det_crops)
            if progress_hook is not None:
                progress_hook(1)

    def build_stacked_2D_spectrum(self, extraction_window_pix=11,
                                  wave_range=(8800, 19100),
                                  wave_step=10,
                                  pixel_shrink=1,
                                  ndrops=100,
                                  use_abs_flux_calib=True,
                                  use_rel_flux_calib=True,
                                  use_inv_var_weights=False,
                                  super_sample=3,
                                  contamination_galaxy_list=None,
                                  progress_hook=None,
                                  **extraction_params):
        """Build a stacked 2D spectrum resampled on a wavelength grid.

        Parameters
        ----------
        wcs
        wave_range
        ndrops
        """
        do_decontamination = False
        if contamination_galaxy_list:
            sim_crop_list = self.build_decontamination_model(contamination_galaxy_list)
            do_decontamination = True

        resampled_spectra_n = 0
        resampled_spectra = 0
        resampled_spectra_var = 0
        counts = 0
        got_extraction = False

        for i, det_crop_list in enumerate(self.crop_list):
            for detector in det_crop_list.keys():
                crop = det_crop_list[detector]
                sim_crop = None
                if do_decontamination:
                    sim_crop = sim_crop_list[i][detector].image

                try:
                    image_out, var_out, norm_out, pix_bins = crop.resample_on_wavelength(
                        self.ra, self.dec,
                        extraction_window_pix=extraction_window_pix,
                        wave_range=wave_range,
                        wave_step=wave_step,
                        pixel_shrink=pixel_shrink,
                        ndrops=ndrops,
                        use_abs_flux_calib=use_abs_flux_calib,
                        use_rel_flux_calib=use_rel_flux_calib,
                        super_sample=super_sample,
                        contamination_model=sim_crop
                    )
                except NoExtraction:
                    continue

                got_extraction = True

                if use_inv_var_weights:
                    weight_ = np.zeros(var_out.shape, dtype='d')
                    nonzero = var_out > 0
                    weight_[nonzero] = norm_out[nonzero]*1./var_out[nonzero]
                    image_out *= weight_
                    norm_out *= weight_
                    var_out *= weight_**2

                resampled_spectra = resampled_spectra + image_out
                resampled_spectra_var = resampled_spectra_var + var_out

                resampled_spectra_n = resampled_spectra_n + norm_out

                nonzero = norm_out > 0
                counts = counts + nonzero

                pix_scale = crop.frame.params['PIXSCALE']
            if progress_hook is not None:
                progress_hook(1)

        if not got_extraction:
            raise NoExtraction

        valid = resampled_spectra_n > 0

        out = np.zeros(resampled_spectra.shape, dtype='d')
        out_var = np.zeros(resampled_spectra.shape, dtype='d')
        out[valid] = resampled_spectra[valid] / resampled_spectra_n[valid]
        out_var[valid] = resampled_spectra_var[valid] / resampled_spectra_n[valid]**2

        pix_perp = (pix_bins[0][1:] + pix_bins[0][:-1])/2.
        wave = (pix_bins[1][1:] + pix_bins[1][:-1])/2.

        spec_pack = {
            'spec2d': out,
            'spec2d_var': out_var,
            'spec2d_norm': resampled_spectra_n,
            'spec2d_counts': counts,
            'pixel_perp': pix_perp,
            'wavelength': wave,
            'extraction_info': {'pix_scale': pix_scale}
        }

        pack_ = self.extract_spec2d_to_1d(
            spec_pack,
            pix_scale=pix_scale,
            extraction_window_pix=extraction_window_pix,
            **extraction_params
        )

        spec_pack.update(pack_)

        return spec_pack

    @staticmethod
    def extract_spec2d_to_1d(spec_pack,
                             extraction_profile='gaussian',
                             extraction_sigma_arcsec=1,
                             extraction_fwhm_arcsec=None,
                             extraction_window_pix=5,
                             pix_scale=None):
        """ """
        if not extraction_profile.lower()[0] in ['g', 'f']:
            print(f"Don't understand extraction profile `{extraction_profile}` Recognized extraction profiles include 'gaussian' or 'flat'. Will use flat.")

        spec2d = spec_pack['spec2d']
        spec2d_var = spec_pack['spec2d_var']
        spec2d_area = spec_pack['spec2d_norm']
        spec2d_counts = spec_pack['spec2d_counts']
        pix_bins = spec_pack['pixel_perp']

        if 'extraction_info' in spec_pack:
            extraction_info = spec_pack['extraction_info']
        else:
            extraction_info = {}

        if pix_scale is None:
            pix_scale = spec_pack['extraction_info']['pix_scale']

        extraction_info['window_pix'] = extraction_window_pix

        if extraction_profile.lower().startswith("g"):
            if extraction_fwhm_arcsec is not None:
                extraction_sigma_arcsec = extraction_fwhm_arcsec * consts.fwhm_to_sigma
            else:
                extraction_fwhm_arcsec = extraction_sigma_arcsec / consts.fwhm_to_sigma
            sigma_kernel = pix_scale/extraction_sigma_arcsec
            kernel = np.exp(-0.5*(pix_bins*sigma_kernel)**2)
            # set normalization so sum to infinity is 1
            kernel *= sigma_kernel / np.sqrt(2*np.pi) * (pix_bins[1] - pix_bins[0])
            extraction_info['profile'] = 'gaussian'
            extraction_info['fwhm_arcsec'] = extraction_fwhm_arcsec
        else:
            kernel = (np.abs(pix_bins) <= (extraction_window_pix+1)).astype(float)
            kernel = kernel / np.sum(kernel)

        norm_2 = np.sum(kernel**2)

        spec1d = np.sum(spec2d.T*kernel, axis=1) / norm_2

        spec1d_var = np.sum(spec2d_var.T*kernel**2, axis=1) / norm_2**2

        counts1d = np.mean(spec2d_counts, axis=0)

        area1d = np.mean(spec2d_area)

        pack = {
            'spec1d': spec1d,
            'spec1d_var': spec1d_var,
            'counts1d': counts1d,
            'area1d': area1d,
            'extraction_info': extraction_info
        }

        return pack

    def build_stacked_line_map(self, wavelength, wcs,
                                  median_filter_size=0,
                                  width=50,
                                  pixel_shrink=1,
                                  ndrops=500,
                                  use_abs_flux_calib=True,
                                  use_rel_flux_calib=True,
                                  progress_hook=None
                                  ):
        """Build a stacked 2D spectrum resampled on a wavelength grid.

        Parameters
        ----------
        wcs
        wave_range
        median_filter_size
        ndrops
        """
        resampled_spectra_n = 0
        resampled_spectra = 0
        resampled_spectra_var = 0
        counts = 0

        for i, det_crop_list in enumerate(self.crop_list):
            for crop in det_crop_list.values():
                if median_filter_size > 0:
                    tilt = crop.frame.params['tilt']
                    filtered_crop_data, mask_ = median_filter_image(crop.image, median_filter_size,
                                                             tilt=tilt)
                    crop_filtered = crop.copy()
                    crop_filtered.image = filtered_crop_data
                    crop_filtered.mask[mask_ > 0] = 1
                else:
                    crop_filtered = crop

                try:
                    image_out, image_var, image_norm, pix_bins = crop_filtered.resample_on_radec(
                        self.ra, self.dec, wavelength,
                        wcs=wcs,
                        width=width,
                        pixel_shrink=pixel_shrink,
                        ndrops=ndrops,
                        use_rel_flux_calib=use_rel_flux_calib,
                        use_abs_flux_calib=use_abs_flux_calib,
                    )
                except NoExtraction:
                    continue

                resampled_spectra = resampled_spectra + image_out
                resampled_spectra_var = resampled_spectra_var + image_var

                if image_norm.max() > 0:
                    resampled_spectra_n = resampled_spectra_n + image_norm

                nonzero = image_norm > 0
                counts = counts + nonzero
            if progress_hook is not None:
                progress_hook(1)


        valid = resampled_spectra_n > 0
        out = np.zeros(resampled_spectra.shape, dtype='d')
        out_var = np.zeros(resampled_spectra.shape, dtype='d')
        out[valid] = resampled_spectra[valid] / resampled_spectra_n[valid]
        out_var[valid] = resampled_spectra_var[valid] / resampled_spectra_n[valid]**2

        return out, out_var, resampled_spectra_n, counts

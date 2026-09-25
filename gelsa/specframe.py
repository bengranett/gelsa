import logging
import os

import numpy as np

from . import continuum_subtraction as csb
from . import photframe, spec_imager, utils
from .sgs import dmutils, frame_coordinates, psf_model, relative_flux
from .spec_crop import NoExtraction, SpecCrop

logger = logging.getLogger(__name__)


class SpecFrame:
    """
    Represents a spectral frame (observation) in the Gelsa simulation.

    This class handles the parameters, loading, and coordinate transformations
    for a specific spectral observation frame (e.g., a NISP slitless
    spectroscopy exposure). It interfaces with various calibration models
    (detector, optical, dispersion, sensitivity, PSF) to provide accurate
    coordinate conversions and flux calculations. It can load existing
    FITS frames or be used to define and simulate new ones.
    """

    _default_params = {
        'grism_name': 'RGS000',
        'RA': 0,
        'DEC': 0,
        'PA': 0,
        'tilt': 0,
        'PIXSCALE': 0.3,
        'exptime_sec': 550.,
        'sigma2_det': 2.33,
        'apply_median_filter': False,
        'median_filter_size_pix': 40,
        'median_filter_mask_threshold': 80,
        'apply_persistence_correction': False,
        'persistence_mask_threshold': 30,
        'use_cache': False,
    }

    dispersion_angles = {
        'RGS000': 0.,
        'RGS180': 180.,
        'BGS000': 0.
    }

    detectors = [11,21,31,41,
                12,22,32,42,
                13,23,33,43,
                14,24,34,44]

    def __init__(self, continuum_subtraction=None, **kwargs):
        """
        Initialize a SpecFrame object.

        Sets up default parameters and updates them with any provided keyword
        arguments. Initializes the frame coordinate system.

        Args:
            **kwargs: Keyword arguments to override default parameters like
                      'RA', 'DEC', 'PA', 'grism_name', 'tilt', etc.
        """
        self._detector_cache = {}

        self.params = self._default_params.copy()
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
        self.hdu_loaded = False
        self._setup()
        self._setup_continuum_subtraction(continuum_subtraction)

    def __str__(self):
        """
        Return a string representation of the SpecFrame object.

        Includes the class name and the current parameter dictionary.

        Returns:
            str: String representation of the object.
        """
        s = f"<{self.__class__}\n"
        s += "params = {"
        for key, value in self.params.items():
            s += f"  '{key}': {value},\n"
        s += "} >"
        return s

    def copy(self):
        """ """
        newframe = SpecFrame(**self.params)
        newframe.framecoord = self.framecoord
        newframe.set_detector_model(self.detector_model)
        newframe.sensitivity_params = self.sensitivity_params
        newframe.optical_params = self.framecoord.optical_params
        newframe.params['wavelength_range'] = self.params['wavelength_range']
        newframe.relative_flux_loss_params = self.relative_flux_loss_params
        newframe.psf_params = self.psf_params
        try:
            newframe.zero_order_mask = self.zero_order_mask
        except AttributeError:
            newframe.zero_order_mask = None
        try:
            newframe._persistence_frame = self._persistence_frame
        except AttributeError:
            pass
        return newframe

    def _setup(self):
        """
        Internal setup method.

        Calculates the total dispersion angle based on grism and tilt.
        Initializes the FrameCoordinates object used for transformations.
        """
        self.params['angle'] = self.params['tilt'] + \
            self.dispersion_angles[self.params['grism_name']]

        self.framecoord = frame_coordinates.FrameCoordinates(
            self.params,
        )

    def _setup_continuum_subtraction(self, continuum_subtraction=None):
        """ """
        if continuum_subtraction is None:
            if self.params['apply_median_filter'] and \
                (self.params['median_filter_size_pix'] > 0):
                continuum_subtraction = csb.MedianFilterFrame(
                    median_filter_size_pix=self.params['median_filter_size_pix']
                )
        self._cs = continuum_subtraction

    def load_frame(self, frame_path=None, loctable_path=None, persistence_path=None):
        """
        Load frame data and metadata from a FITS file.

        Reads the primary header and potentially updates pointing information
        (RA, DEC, PA) either from specific SIR headers or by using a
        location table if provided or found automatically.

        Args:
            frame_path (str, optional): Path to the FITS science frame file.
                                        Can be .gz compressed. Defaults to None.
            loctable_path (str, optional): Path to the corresponding location
                                           table FITS file. If None, attempts
                                           to find it automatically based on
                                           the science frame path. Defaults to None.
        """

        if frame_path.endswith(".gz"):
            # check if decompressed file exists
            path_ = frame_path[:-3]
            if os.path.exists(path_):
                frame_path = path_
                logger.info(f"found {frame_path}")

        self._hdul, self._metadata = dmutils.load_scienceframe(frame_path)

        self._header = self._hdul[0].header
        self.params['frame_path'] = frame_path
        self.params['loctable_path'] = loctable_path
        self.params['grism_name'] = self._header['GWA_POS'].strip()
        self.params['tilt'] = self._header['GWA_TILT']
        self.params['DATE-OBS'] = self._header['DATE-OBS']
        try:
            # Look for SIR adjusted pointing center
            self.params['RA'] = self._header['SIR_RA']
            self.params['DEC'] = self._header['SIR_DEC']
            self.params['PA'] = self._header['SIR_PA']
            got_adjusted_pointing = True
        except KeyError:
            # Fall back to commanded pointing with location table
            self.params['RA'] = self._header['RA']
            self.params['DEC'] = self._header['DEC']
            self.params['PA'] = self._header['PA']
            got_adjusted_pointing = False
        self.params['PIXSCALE'] = self._header['PIXSCALE']
        self.params['PTGID'] = self._header['PTGID']
        self.params['OBS_ID'] = self._header['OBS_ID']
        self.params['DITHOBS'] = self._header['DITHOBS']
        # Read with get: these are not in every frame, and a missing keyword
        # should leave the value empty rather than stop the frame loading.
        self.params['CALBLKID'] = self._header.get('CALBLKID')
        self.params['OBSMODE'] = self._header.get('OBSMODE')
        self.params['PATCH_ID'] = self._header.get('PATCH_ID')

        self._setup()
        if not got_adjusted_pointing:
            if loctable_path is None:
                try:
                    loctable_path = dmutils.find_location_table(frame_path)
                except ValueError:
                    logger.warning("could not find location table automatically")
            try:
                self.framecoord.update_optical_model_from_location_table(loctable_path)
            except Exception:
                pass
        self.hdu_loaded = True

    def set_pointing_center(self, ra, dec, pa):
        """ """
        self.framecoord.set_pointing_center(ra, dec, pa)

    @property
    def hdul(self):
        """
        Access the FITS HDUList object for the frame.

        Loads the FITS file on first access if not already loaded.

        Returns:
            astropy.io.fits.HDUList: The HDUList object.
        """
        try:
            return self._hdul
        except AttributeError:
            self._hdul, self._metadata = dmutils.load_scienceframe(self.params['frame_path'])
            # close_frame() drops the header along with the HDU list, so restore
            # it here: a reopened frame must be as closeable as a freshly
            # loaded one.
            self._header = self._hdul[0].header
            self.hdu_loaded = True
        return self._hdul

    def clear_cache(self):
        """ """
        self._detector_cache = {}

    def close_frame(self):
        """
        Close the associated FITS file handle.

        Deletes internal references to the HDUList and header. Closing a frame
        that is already closed does nothing.
        """
        if self.hdu_loaded:
            hdul = self.__dict__.pop('_hdul', None)
            if hdul is not None:
                hdul.close()
            self.__dict__.pop('_header', None)
            self.zero_order_mask = None
            self.clear_cache()

    def set_detector_model(self, detector_model):
        """
        Set the detector model for coordinate transformations.

        Args:
            detector_model (detector_model.DetectorModel): The detector model instance.
        """
        self.framecoord.detector_model = detector_model
        self.detector_model = detector_model

    def set_optical_model(self, optical_model):
        """
        Set the optical model based on the current grism and tilt.

        Args:
            optical_model (optical_model.OpticalModel): The optical model instance.
        """
        self.framecoord.optical_params = optical_model.get_model(
            self.params['grism_name'], tilt=self.params['tilt']
        )
        self.optical_params = self.framecoord.optical_params

    def set_dispersion_model(self, dispersion_model):
        """
        Set the dispersion models (IDS and CRV) based on grism and tilt.

        Args:
            dispersion_model (dict): A dictionary containing 'ids' and 'crv'
                                     DisplacementModel instances.
        """
        ids_model = dispersion_model['ids']
        crv_model = dispersion_model['crv']
        self.framecoord.ids_params = ids_model.get_model(
            self.params['grism_name'], tilt=self.params['tilt'])
        self.framecoord.crv_params = crv_model.get_model(
            self.params['grism_name'], tilt=self.params['tilt'])

    def set_sensitivity(self, sensitivity_model):
        """
        Set the sensitivity model and update the frame's wavelength range.

        Args:
            sensitivity_model (sensitivity_model.SensitivityModel): The sensitivity
                                                                    model instance.
        """
        self.sensitivity_params = sensitivity_model.get_model(
            self.params['grism_name'], self.params['tilt'])
        self.params['wavelength_range'] = self.sensitivity_params['bounds']

    def set_relative_flux_model(self, relative_flux_model):
        """
        Set the relative flux loss calibration model.

        Args:
            relative_flux_model (relative_flux.RelativeFluxCalibration or None):
                The relative flux model instance, or None if not used.
        """
        self.relative_flux_loss_params = None
        if relative_flux_model is not None:
            self.relative_flux_loss_params = relative_flux_model.get_model(
                self.params['grism_name'], self.params['tilt']
            )

    def set_psf_model(self, psf_model):
        """
        Set the Point Spread Function (PSF) model.

        Args:
            psf_model (psf_model.PSFModel or psf_model.DefaultPSFModel or None):
                The PSF model instance, or None if not used.
        """
        self.psf_params = None
        if psf_model is not None:
            self.psf_params = psf_model.get_model(
                self.params['grism_name']
            )

    def set_zero_order_mask(self, zero_order_mask):
        """
        Set and apply the zero-order contamination mask.

        Args:
            zero_order_mask (zero_order_mask.ZeroOrderMask or None):
                The zero-order mask instance, or None if not used.
        """
        if zero_order_mask is None:
            self.zero_order_mask = None
        else:
            self.zero_order_mask = zero_order_mask.create_zero_order_mask(self)

    def set_persistence_mask(self, path):
        """ """
        if path is None:
            self._persistence_frame = None
            return
        self._persistence_frame = photframe.PhotFrame(sir_layout=True)
        self._persistence_frame.load_frame(path)

    @property
    def persistence_frame(self):
        """ """
        try:
            return self._persistence_frame
        except AttributeError:
            return None

    @property
    def detector_shape(self):
        """
        Return detector size in pixels (ny, nx).

        Note the order (ny, nx) consistent with numpy array indexing.

        Returns:
            tuple: (height, width) in pixels.
        """
        nx = self.detector_model.params['nx_pixels']
        ny = self.detector_model.params['ny_pixels']
        return ny, nx # Note: numpy shape order (rows, columns)

    def arcsec_to_pixel(self, x_arcsec):
        """Convert units from arsec to pixels

        x_pix = x_arcsec / pixscale
        with pixscale in arcsec/pixel
        """
        return x_arcsec / self.params['PIXSCALE']

    def radec_to_pixel_(self, ra, dec, wavelength, dispersion_order=1, **kwargs):
        """
        Compute detector pixel coordinates from RA, Dec sky position and wavelength.

        Args:
            ra (float or numpy.ndarray): Sky position RA in degrees.
            dec (float or numpy.ndarray): Sky position Dec in degrees.
            wavelength (float or numpy.ndarray): Wavelength in angstrom.
            dispersion_order (int, optional): Dispersion order (0 or 1). Defaults to 1.
            **kwargs: Additional arguments passed to framecoord.radec_to_pixel.

        Returns:
            tuple: (x, y, detector_index)
                   x (float or ndarray): Pixel x-coordinate(s).
                   y (float or ndarray): Pixel y-coordinate(s).
                   detector_index (int or ndarray): Detector index/indices (0-15), -1 if off detector.
        """
        return self.framecoord.radec_to_pixel_(ra, dec, wavelength,dispersion_order)

    def radec_to_pixel(self, ra, dec, wavelength, dispersion_order=1, **kwargs):
        """
        Compute detector pixel coordinates from RA, Dec sky position and wavelength.

        Args:
            ra (float or numpy.ndarray): Sky position RA in degrees.
            dec (float or numpy.ndarray): Sky position Dec in degrees.
            wavelength (float or numpy.ndarray): Wavelength in angstrom.
            dispersion_order (int, optional): Dispersion order (0 or 1). Defaults to 1.
            **kwargs: Additional arguments passed to framecoord.radec_to_pixel.

        Returns:
            tuple: (x, y, detector_index)
                   x (float or ndarray): Pixel x-coordinate(s).
                   y (float or ndarray): Pixel y-coordinate(s).
                   detector_index (int or ndarray): Detector index/indices (0-15), -1 if off detector.
        """
        return self.framecoord.radec_to_pixel(ra, dec, wavelength,dispersion_order)

    def radec_to_fov(self, ra, dec, wavelength, dispersion_order=1, **kwargs):
        """
        Compute detector pixel coordinates from RA, Dec sky position and wavelength.

        Args:
            ra (float or numpy.ndarray): Sky position RA in degrees.
            dec (float or numpy.ndarray): Sky position Dec in degrees.
            wavelength (float or numpy.ndarray): Wavelength in angstrom.
            dispersion_order (int, optional): Dispersion order (0 or 1). Defaults to 1.
            **kwargs: Additional arguments passed to framecoord.radec_to_pixel.

        Returns:
            tuple: (x, y, detector_index)
                   x (float or ndarray): Pixel x-coordinate(s).
                   y (float or ndarray): Pixel y-coordinate(s).
                   detector_index (int or ndarray): Detector index/indices (0-15), -1 if off detector.
        """


        return self.framecoord.radec_to_fov(ra, dec, wavelength, dispersion_order=dispersion_order)



    def fov_to_pixel(self, xfov,yfov, **kwargs):
        """
        Compute detector pixel coordinates from RA, Dec sky position and wavelength.

        Args:
            ra (float or numpy.ndarray): Sky position RA in degrees.
            dec (float or numpy.ndarray): Sky position Dec in degrees.
            wavelength (float or numpy.ndarray): Wavelength in angstrom.
            dispersion_order (int, optional): Dispersion order (0 or 1). Defaults to 1.
            **kwargs: Additional arguments passed to framecoord.radec_to_pixel.

        Returns:
            tuple: (x, y, detector_index)
                   x (float or ndarray): Pixel x-coordinate(s).
                   y (float or ndarray): Pixel y-coordinate(s).
                   detector_index (int or ndarray): Detector index/indices (0-15), -1 if off detector.
        """

        return self.detector_model.getPixel(xfov, yfov)


    def pixel_to_radec(self, x, y, det, wavelength, dispersion_order=1):
        """
        Compute sky position RA, Dec from detector pixel coordinates and wavelength.

        Args:
            x (float or numpy.ndarray): Detector pixel x-coordinate(s).
            y (float or numpy.ndarray): Detector pixel y-coordinate(s).
            det (int or numpy.ndarray): NISP detector index/indices (0-15).
            wavelength (float or numpy.ndarray): Wavelength in angstrom.
            dispersion_order (int, optional): Dispersion order (0 or 1). Defaults to 1.

        Returns:
            tuple: (RA, Dec)
                   RA (float or ndarray): Right Ascension coordinate(s) in degrees.
                   Dec (float or ndarray): Declination coordinate(s) in degrees.
        """
        return self.framecoord.pixel_to_radec( x, y, det, wavelength,
                                              dispersion_order)

    def sample_psf(self, x=None, y=None, detector=None, wavelength_ang=None,
                   **args):
        """
        Draw samples representing the PSF offset at given coordinates/wavelength.

        Args:
            x (float or ndarray, optional): Detector x-coordinate(s). Not currently used
                                            for position dependence. Defaults to None.
            y (float or ndarray, optional): Detector y-coordinate(s). Not currently used
                                            for position dependence. Defaults to None.
            detector (int or ndarray, optional): Detector index/indices. Not currently used.
                                                 Defaults to None.
            wavelength_ang (float or ndarray, optional): Wavelength(s) in angstroms.
                                                         Used for chromatic PSF. Defaults to None.
            **args: Additional arguments passed to psf_model.sample_psf.

        Returns:
            tuple: (dx, dy) offsets in pixel coordinates, or (0, 0) if no PSF model is set.
        """
        if self.psf_params is None:
            return 0, 0
        pos_fov = None
        samples = psf_model.sample_psf(
            self.psf_params,
            pos_fov=pos_fov, wavelength_ang=wavelength_ang,
            **args
        )
        return samples

    def lineflux_to_counts(self, flux, wavelength):
        """
        Convert line flux in erg/s/cm^2 to photon counts.

        Uses the sensitivity model and exposure time.

        Args:
            flux (float or ndarray): Line flux in erg/s/cm^2.
            wavelength (float or ndarray): Wavelength(s) in angstroms.

        Returns:
            float or ndarray: Corresponding photon counts.
        """
        return flux * self.sensitivity_params['func'](wavelength) * self.params['exptime_sec']

    def flux_to_counts(self, flux, wavelength):
        """
        Convert flux density in erg/s/cm^2/A to photon counts.

        Uses the sensitivity model, wavelength step, and exposure time.

        Args:
            flux (float or ndarray): Flux density in erg/s/cm^2/A.
            wavelength (float or ndarray): Wavelength array in angstroms. Assumes
                                           uniform spacing to calculate step.

        Returns:
            float or ndarray: Corresponding photon counts.
        """
        step = wavelength[1] - wavelength[0]
        return flux * step * self.sensitivity_params['func'](wavelength) * self.params['exptime_sec']

    def counts_to_flux(self, counts, wavelength):
        """
        Convert photon counts to flux density in erg/s/cm^2/A.

        Inverse of flux_to_counts. Uses sensitivity, wavelength step, and exposure time.

        Args:
            counts (float or ndarray): Photon counts.
            wavelength (float or ndarray): Wavelength array in angstroms. Assumes
                                           uniform spacing to calculate step.

        Returns:
            float or ndarray: Corresponding flux density in erg/s/cm^2/A.
        """
        step = wavelength[1] - wavelength[0]
        return counts / (step * self.sensitivity_params['func'](wavelength) * self.params['exptime_sec'])

    def get_sensitivity(self, wavelength):
        """
        Return sensitivity (response) in units counts/(erg/cm^2/s).

        Note: This is the sensitivity integrated over the pixel wavelength step.

        Args:
            wavelength (float or ndarray): Wavelength array in angstroms. Assumes
                                           uniform spacing to calculate step.

        Returns:
            float or ndarray: Sensitivity value(s).
        """
        step = wavelength[1] - wavelength[0]
        return step * self.sensitivity_params['func'](wavelength) * self.params['exptime_sec']

    # alias
    sensitivity = get_sensitivity

    def get_relative_flux_loss_radec(self, ra, dec, wavelength_ang=None):
        """Calculates the relative flux loss factor at given detector coordinates and wavelength.

        Uses the relative flux calibration model (`relative_flux_loss_params`)
        if it has been set via `set_relative_flux_model`.

        Args:
            x (float | np.ndarray): Detector x-coordinate(s).
            y (float | np.ndarray): Detector y-coordinate(s).
            det_index (int | np.ndarray): NISP detector index/indices (0-15).
            wavelength_ang (float | np.ndarray): Wavelength(s) in Angstroms.

        Returns:
            float | np.ndarray: Flux loss factor. This is a multiplicative factor
                (<= 1.0) representing the fraction of flux remaining after
                accounting for effects like vignetting or detector gaps modeled
                by the relative flux calibration. Returns 1.0 if no model is
                set or if the model doesn't apply to the given inputs. The
                shape matches the input coordinate/wavelength arrays.

        Raises:
            AttributeError: If the detector model has not been set (needed by
                the underlying `relative_flux.get_flux_loss`).
        """
        try:
            self.relative_flux_loss_params
        except AttributeError:
            return 1

        if self.relative_flux_loss_params is None:
            return 1

        if wavelength_ang is None:
            wavelength_ang = (self.params['wavelength_range'][0] + self.params['wavelength_range'][1])/2.

        scalar = np.ndim(ra) == 0
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)

        wavelength_ang = np.ones(len(ra)) * wavelength_ang

        xmm_undist, ymm_undist = self.framecoord.getUndistortedObjectPosition(ra, dec)
        xfov, yfov = self.framecoord.getReferencePosition_jit(
            xmm_undist=xmm_undist, ymm_undist=ymm_undist,
            order=1
        )

        y = relative_flux.get_flux_loss_fov(
            self.detector_model,
            self.relative_flux_loss_params,
            xfov, yfov, wavelength_ang=wavelength_ang
        )
        if scalar:
            return y[0]
        return y

    def get_detector(self, detector_index):
        """
        Get the pixel data arrays for a specific detector.

        Retrieves the science image, data quality (mask), and variance arrays.
        Applies the zero-order mask if it has been set.

        Args:
            detector_index (int): NISP detector index (0-15).

        Returns:
            tuple: (image, mask, variance)
                   image (ndarray): Science image data.
                   mask (ndarray): Data quality mask (0=good, 1=bad).
                   variance (ndarray): Variance data.
        """
        d = self.detectors[detector_index]
        cache_key = detector_index
        if cache_key in self._detector_cache:
            return self._detector_cache[cache_key]

        if self.hdu_loaded:
            extname = f'DET{d}.SCI'
            maskname = f'DET{d}.DQ'
            varname = f'DET{d}.VAR'
            fullimage = self.hdul[extname].data
            fullmask = self.hdul[maskname].data
            fullvar = self.hdul[varname].data
        else:
            # defaults if simulating
            fullimage = 0
            fullmask = 0
            fullvar = 0

        # Handle case where data is simulated in memory
        if hasattr(self, '_data') and detector_index in self._data:
            fullimage += fullimage + self._data[detector_index]
            fullmask = (fullmask + self._mask[detector_index]) > 0
            fullvar = fullvar + self._var[detector_index]
        elif not self.hdu_loaded:
            # Nothing loaded or simulated. In this case, image is zero
            nx = self.framecoord.detector_model.params['nx_pixels']
            ny = self.framecoord.detector_model.params['ny_pixels']
            fullimage = np.zeros((ny, nx), dtype='d')
            fullmask = np.zeros((ny, nx), dtype=bool)
            fullvar = np.zeros((ny, nx), dtype='d')

        try:
            if self.zero_order_mask is not None:
                invalid = self.zero_order_mask[detector_index] == 0
                fullimage[invalid] = 0
                fullmask[invalid] = 1
        except AttributeError:
            pass

        if self._cs is not None:
            fullimage, fullmask = self._cs.process_frame(self, fullimage, fullmask)

        if self.persistence_frame is not None:
            persistance_image = self.persistence_frame.get_detector(detector_index)
            if self.params['apply_persistence_correction']:
                # apply persistence correction
                fullimage -= persistance_image
            if self.params['persistence_mask_threshold'] > 0:
                # mask persistence above threshold
                invalid = persistance_image > self.params['persistence_mask_threshold']
                fullimage[invalid] = 0
                fullmask[invalid] = 1
                logger.info(f"Masking persistence {self.params['persistence_mask_threshold']} " \
                      f"fraction of pixels: {np.sum(invalid)/invalid.size}")

        # store in cache
        if self.params['use_cache']:
            self._detector_cache[cache_key] = (fullimage, fullmask, fullvar)

        return fullimage, fullmask, fullvar

    def cutout(self, ra, dec, redshift=0, lam_step=20, wavelength_range=None,
               select_detector=None, **crop_args):
        """
        Create spectral cutouts (SpecCrop objects) around a sky position.

        Calculates the trace of the object across the detectors for the given
        wavelength range and creates SpecCrop instances for each detector
        the trace falls on.

        Args:
            ra (float): Right Ascension of the target center (degrees).
            dec (float): Declination of the target center (degrees).
            redshift (float, optional): Redshift of the target (used for metadata). Defaults to 0.
            lam_step (float, optional): Wavelength step for calculating the trace (Angstroms).
                                        Defaults to 20.
            wavelength_range (tuple, optional): (min, max) wavelength range for the trace.
                                                Defaults to the frame's sensitivity range.
            **crop_args: Additional keyword arguments passed to SpecCrop.crop_trace
                         (e.g., padx, pady for padding around the trace).

        Returns:
            dict: A dictionary where keys are detector indices (int) and values
                  are the corresponding SpecCrop objects.

        Raises:
            ValueError: If the target's trace does not fall on any detector within
                        the specified wavelength range.
        """
        if wavelength_range is None:
            wavelength_range = self.params['wavelength_range']
        wave_trace = utils.intrange(*wavelength_range, lam_step)
        n = len(wave_trace)

        detx, dety, detid = self.radec_to_pixel(
            ra*np.ones(n),
            dec*np.ones(n),
            wave_trace
        )
        valid = detid >= 0

        if select_detector is not None:
            logger.info(f"selecting detector {select_detector}")
            valid &= detid == select_detector

        if np.sum(valid) == 0:
            raise ValueError(f"RA, Dec not on detector within wavelength range {wavelength_range}: {(ra, dec)}")

        detx = detx[valid]
        dety = dety[valid]
        detid = detid[valid]
        wave_trace = wave_trace[valid]

        pack_list = {}

        for det in np.unique(detid):
            sel = detid == det
            detx_ = detx[sel]
            dety_ = dety[sel]
            crop = SpecCrop.crop_trace(self, det, detx_, dety_, **crop_args)
            crop.center = (ra, dec)
            crop.wavelength_trace = wave_trace[sel]
            crop.redshift = redshift
            crop.apply_continuum_subtraction(self._cs)

            pack_list[det] = crop

        return pack_list

    def add_sources(self, galaxy_list, **kwargs):
        """
        Simulate and add sources to the frame's internal data arrays.

        This method is used when the SpecFrame is not loaded from an existing
        FITS file but is being used for simulation. It uses SpectralImager
        to generate the image of the provided galaxies.

        Args:
            galaxy_list (list): A list of Galaxy objects to simulate.
            **kwargs: Additional keyword arguments passed to
                      SpectralImager.make_image.
        """
        self._data = {}
        self._var = {}
        self._mask = {}
        S = spec_imager.SpectralImager(self)
        images, var_images = S.make_image(galaxy_list, return_var=True, **kwargs)

        for det, image in images.items():
            if det not in self._data:
                self._data[det] = 0
                self._var[det] = 0
                self._mask[det] = np.zeros(image.shape, dtype=bool)
            self._data[det] = self._data[det] + image
            self._var[det] = self._var[det] + var_images[det]

    def resample_on_wavelength(self, ra, dec, **args):
        """Extracts and combines 2D spectra onto a uniform wavelength grid.

        Creates cutouts around the target RA/Dec using `cutout`, then calls
        `resample_on_wavelength` on each `SpecCrop` object. The results from
        all detectors are summed.

        Args:
            ra (float): Right Ascension of the target center (degrees).
            dec (float): Declination of the target center (degrees).
            **args: Additional keyword arguments passed directly to
                    `SpecCrop.resample_on_wavelength` (e.g., `wave_range`,
                    `wave_step`, `extraction_window`, `ndrops`).

        Returns:
            tuple:
                - im (np.ndarray): The combined, resampled 2D spectrum (flux).
                - var (np.ndarray): The combined variance of the resampled spectrum.
                - norm (np.ndarray): The combined normalization map (sum of weights).
                - pix_bins (tuple): The pixel and wavelength bin edges used for
                  resampling, taken from the last processed crop (assumes they
                  are consistent).

        Raises:
            ValueError: If `cutout` fails to find the target on any detector.
            # Add other potential exceptions from SpecCrop.resample_on_wavelength
        """
        try:
            crop = self.cutout(ra, dec)
        except ValueError:
            #print("Source not on detector")
            raise

        im = 0
        var = 0
        norm = 0
        got_extraction = False

        for det in crop.keys():
            try:
                im_, var_, norm_, pix_bins = crop[det].resample_on_wavelength(
                    ra, dec, **args)
            except NoExtraction:
                continue

            got_extraction = True
            im = im + im_
            var = var + var_
            norm = norm + norm_

        if not got_extraction:
            raise NoExtraction

        return im, var, norm, pix_bins

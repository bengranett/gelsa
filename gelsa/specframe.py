import os
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import utils
from .sgs import dmutils, frame_coordinates, relative_flux, psf_model

from . import spec_imager
from . import photframe
from .spec_crop import SpecCrop, NoExtraction
from . import continuum_subtraction_base as csb

from scipy import ndimage


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
                print(f"found {frame_path}")

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
                    print("could not find location table automatically")
            try:
                self.framecoord.update_optical_model_from_location_table(loctable_path)
            except:
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
            # self.hdu_loaded = False
            del self.zero_order_mask
            self.zero_order_mask = None
            self._detector_cache = {}

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
        # if x is not None:
        #     xfov, yfov = self.detector_model.getFOVPosition(x, y, detector)
        #     pos_fov = (np.mean(xfov), np.mean(yfov))
        # else:
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
                print(f"Masking persistence {self.params['persistence_mask_threshold']} " \
                      f"fraction of pixels: {np.sum(invalid)/invalid.size}")

        # store in cache
        if self.params['use_cache']:
            self._detector_cache[cache_key] = (fullimage, fullmask, fullvar)

        return fullimage, fullmask, fullvar

    def median_filter(self, image, filter_size_pix=0, tilt=0):
        """ """
        if filter_size_pix <= 0:
            return

        filtered_thresh = self.params['median_filter_mask_threshold']

        print(f"Applying median filter with size {filter_size_pix} pixels and {tilt=}")
        invalid = np.logical_not(np.isfinite(image))
        image[invalid] = 0
        rotated_image = ndimage.rotate(image, tilt, reshape=False)
        filtered_rotated_image = ndimage.median_filter(rotated_image, filter_size_pix, axes=1)
        rotated_back = ndimage.rotate(filtered_rotated_image, -tilt, reshape=False, cval=np.nan)
        resid = image - rotated_back
        invalid = np.logical_not(np.isfinite(resid))
        resid[invalid] = 0

        n = 5
        kernel = np.array([np.ones(n), -1*np.ones(n), np.ones(n)])
        kernel = kernel / np.sum(kernel)

        filtered = ndimage.median_filter(resid, 15, axes=1)
        filtered = ndimage.convolve(filtered, kernel)

        invalid2 = np.abs(filtered) > filtered_thresh
        resid[invalid2] = 0
        print(f"theshold {filtered_thresh}")
        print(f"Percentiles (95, 99, 99.5) {np.percentile(np.abs(filtered), (95, 99, 99.5))}")
        print(f"median artifact masked pixel fraction: {np.sum(invalid2)/invalid2.size:0.5f}")
        invalid = invalid  | invalid2
        return resid, invalid

    def robust_local_linear_background(self, image, mask, window):
        ny, nx = image.shape
        half = window // 2
        cont = np.full_like(image, np.nan, dtype=float)

        x = np.arange(nx, dtype=float)

        slope_max = 5e-17  # conservative, from observed continuum slopes in plots

        for y in range(ny):
            row = image[y]
            row_mask = mask[y]

            for i in range(nx):
                i0 = max(0, i - half)
                i1 = min(nx, i + half + 1)

                xs = x[i0:i1]
                ys = row[i0:i1]
                ms = row_mask[i0:i1]

                good = np.isfinite(ys) & (~ms)
                if np.sum(good) < 7:
                    continue

                xs_g = xs[good]
                ys_g = ys[good]

                # iterative sigma clipping (2 iterations)
                for _ in range(2):
                    x0 = xs_g.mean()
                    y0 = ys_g.mean()
                    xs_c = xs_g - x0
                    ys_c = ys_g - y0

                    denom = np.sum(xs_c**2)
                    if denom <= 0:
                        break

                    slope = np.sum(xs_c * ys_c) / denom
                    model = slope * xs_g + (y0 - slope * x0)
                    resid = ys_g - model

                    sig = np.nanstd(resid)
                    if not np.isfinite(sig) or sig == 0:
                        break

                    keep = np.abs(resid) < 3.0 * sig
                    if np.sum(keep) < 7:
                        break

                    xs_g = xs_g[keep]
                    ys_g = ys_g[keep]

                if len(xs_g) < 7:
                    continue

                q = np.nanpercentile(ys_g, 35)
                low = ys_g <= q
                if np.sum(low) >= 7:
                    xs_g = xs_g[low]
                    ys_g = ys_g[low]
                # final slope fit
                x0 = xs_g.mean()
                y0 = ys_g.mean()
                xs_c = xs_g - x0
                ys_c = ys_g - y0

                denom = np.sum(xs_c**2)
                if denom <= 0:
                    cont[y, i] = np.nanmedian(ys_g)
                    continue

                slope = np.sum(xs_c * ys_c) / denom

                # slope cap
                if abs(slope) > slope_max:
                    slope = 0.0

                # anchor intercept to median (prevents upward bias)
                median = np.nanmedian(ys_g)
                intercept = median - slope * x0

                cont[y, i] = slope * i + intercept

        return cont

    def _subtract_median_continuum(self, crop, detx_, dety_, wavelength_range):
        """Mask emission lines and subtract a running-median continuum from a crop.

        Mutates ``crop.image`` (residual after continuum subtraction) and
        ``crop.mask`` (True where the residual is invalid) in place.
        """
        print(f"apply_median_filter = True")
        ny, nx = crop.image.shape
        line_mask = np.zeros((ny, nx), dtype=bool)
        wave_min, wave_max = wavelength_range
        hole_width = int(self.params.get('median_filter_line_hole_pix', 6))
        tol_wave = 1.5 * float(self.params.get('wavelength_step', 13.4))
        merge_pad = 3
        lam_list = (
            6564.61,
            4862.69,
            4341.69,
            4102.92,
            3971.19,
            3890.15,
            3836.48,
            3798.98,
            3771.70,
            3751.22,
            12821.59,
            10941.09,
            10052.13,
            9548.59,
            9231.55,
            9017.39,
            3728.49,
            6732.71,
            6718.32,
            9533.20,
            9071.10
        )
        if crop.redshift > 0:
            for lam_rest in lam_list:
                lam_obs = lam_rest * (1.0 + crop.redshift)
                if not (wave_min <= lam_obs <= wave_max):
                    continue
                if len(crop.wavelength_trace) == 0:
                    continue
                idx = np.argmin(np.abs(crop.wavelength_trace - lam_obs))
                if np.abs(crop.wavelength_trace[idx] - lam_obs) > tol_wave:
                    continue
                x0 = int(round(detx_[idx] - crop.bbox[0]))
                y0 = int(round(dety_[idx] - crop.bbox[2]))
                if 0 <= y0 < ny:
                    half = hole_width // 2
                    x_start = max(0, x0 - half)
                    x_end = min(nx, x0 + half + (hole_width % 2))
                    line_mask[y0, x_start:x_end] = True
        """ this block makes a masking bridge bw the OIII doublet
        oiii1 = 4960.30 * (1.0 + crop.redshift)
        oiii2 = 5008.24 * (1.0 + crop.redshift)
        if (wave_min <= oiii1 <= wave_max) or (wave_min <= oiii2 <= wave_max):
            if len(crop.wavelength_trace) > 0:
                idx0 = np.argmin(np.abs(crop.wavelength_trace - oiii1))
                idx1 = np.argmin(np.abs(crop.wavelength_trace - oiii2))
                if (np.abs(crop.wavelength_trace[idx0] - oiii1) <= tol_wave) and (np.abs(crop.wavelength_trace[idx1] - oiii2) <= tol_wave):
                    x0 = int(round(detx_[idx0] - crop.bbox[0]))
                    x1 = int(round(detx_[idx1] - crop.bbox[0]))
                    y0 = int(round(dety_[idx0] - crop.bbox[2]))
                    if x1 < x0:
                        x0, x1 = x1, x0
                    if 0 <= y0 < ny:
                        x_start = max(0, x0 - merge_pad)
                        x_end = min(nx, x1 + merge_pad + 1)
                        line_mask[y0, x_start:x_end] = True
        """
        filter_size = int(self.params.get('median_filter_size_pix', 31))
        if filter_size <= 0:
            return

        if filter_size < 3:
            filter_size = 3
        if filter_size % 2 == 0:
            filter_size += 1
        filt_n = max(3, filter_size)
        tilt = float(crop.frame.params.get('tilt', 0.0))
        filt_n = int(min(filt_n, nx))

        if tilt != 0:
            image_rot = ndimage.rotate(
                crop.image.astype(float), tilt,
                reshape=False, order=1, mode='reflect', cval=np.nan
            )
            mask_rot = ndimage.rotate(
                line_mask.astype(float), tilt,
                reshape=False, order=0, mode='constant', cval=0.0
            ) > 0.5

            image_rot_masked = image_rot.copy()
            image_rot_masked[mask_rot] = np.nan

            ### >>> VARIANCE MASK ADDED <<<
            var = crop.var
            var_rot = ndimage.rotate(
                var.astype(float), tilt,
                reshape=False, order=1, mode='reflect', cval=np.nan
            )
            var_mask_rot = var_rot > (10 * np.nanmedian(var_rot))
            image_rot_masked[var_mask_rot] = np.nan
            ### <<< VARIANCE MASK ADDED <<<


            # >>> BEGIN MAD-BASED CONTINUUM QUALITY CHECK <<<
            """
            mad_rot = ndimage.generic_filter(
                image_rot_masked,
                lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))),
                size=(1, filt_n),
                mode='reflect'
            )

            n_valid_rot = ndimage.generic_filter(
                np.isfinite(image_rot_masked).astype(float),
                np.nansum,
                size=(1, filt_n),
                mode='reflect'
            )

            mad_ref = 45.0
            mad_max = 200.0
            bad_nvalid = (n_valid_rot < 5)
            #bad_mad = (mad_rot > mad_ref)
            bad_mad = (mad_rot > mad_ref)
            mad_eff = np.clip(mad_rot, mad_ref, mad_max)
            scale = (mad_eff / mad_ref) ** 2

            var_rot[bad_mad] *= scale[bad_mad]
            var_rot[bad_nvalid] *= 300.0
            var_back = ndimage.rotate(
                        var_rot, -tilt,
                        reshape=False, order=1, mode='reflect', cval=np.nan
                    )
            crop.var = var_back
            """
            # >>> END MAD-BASED CONTINUUM QUALITY CHECK <<<


            cont_rot = _rolling_nanmedian(image_rot_masked, filt_n)

            #cont_rot = self.robust_local_linear_background(
            #    image_rot_masked,
            #    mask=mask_rot,
            #    window=filt_n
            #)

            cont_back = ndimage.rotate(
                cont_rot, -tilt,
                reshape=False, order=1, mode='reflect', cval=np.nan
            )
            line_mask_back = ndimage.rotate(
                mask_rot.astype(float), -tilt,
                reshape=False, order=0, mode='constant', cval=0.0
            ) > 0.5

        else:
            image_masked = crop.image.astype(float).copy()
            image_masked[line_mask] = np.nan

            ### >>> VARIANCE MASK ADDED <<<
            var = crop.var
            var_mask = var > (10 * np.nanmedian(var))
            image_masked[var_mask] = np.nan
            ### <<< VARIANCE MASK ADDED <<<

            cont_back = _rolling_nanmedian(image_masked, filt_n)

            line_mask_back = line_mask.copy()

        resid = crop.image.astype(float) - cont_back
        bad = ~np.isfinite(resid)
        resid[bad] = 0

        crop.image = resid
        #crop.mask |= line_mask_back
        crop.mask[bad] = True

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
            print(f"selecting detector {select_detector}")
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
            crop = SpecCrop.crop_trace(self, det, detx_, dety_,
                                       continuum_subtraction=self._cs,
                                       **crop_args)
            crop.center = (ra, dec)
            crop.wavelength_trace = wave_trace[sel]
            crop.redshift = redshift

            # if crop.frame.params.get('apply_median_filter', False):
                # self._subtract_median_continuum(crop, detx_, dety_, wavelength_range)

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

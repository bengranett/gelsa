import os
import json
import numpy as np
from astropy.io import fits
from astropy import wcs

from .sgs import dmutils


class PhotFrame:
    """
    Represents a photometric frame (observation) in the Gelsa simulation.

    This class handles the parameters, loading, and coordinate transformations
    for a specific photometric observation frame (e.g., a NISP imaging exposure).
    It manages WCS information, allowing conversion between sky (RA, Dec) and
    detector pixel coordinates. It can load existing FITS frames or be used
    to define new ones based on pointing and WCS parameters.
    """

    _default_params = {
        'ra': 0,
        'dec': 0,
        'pa': 0,
        'filter': None,
        'pixscale': 0.3,
        'exptime_sec': 87.,
        'sigma2_det': 2.33,
        'sir_layout': True,
    }

    detector_list = [11, 21, 31, 41,
                     12, 22, 32, 42,
                     13, 23, 33, 43,
                     14, 24, 34, 44]

    def __init__(self, **kwargs):
        """
        Initialize a PhotFrame object.

        Sets up default parameters and updates them with any provided keyword
        arguments. Initializes internal state.

        Args:
            **kwargs: Keyword arguments to override default parameters like
                      'ra', 'dec', 'pa', 'filter', 'pixscale', 'sir_layout', etc.
        """
        self.params = self._default_params.copy()
        self.params.update(kwargs)
        self.hdu_loaded = False
        self._setup()

    def __str__(self):
        """
        Return a string representation of the PhotFrame object.

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

    def _setup(self):
        """
        Internal setup method.

        Currently empty, but can be used for future initialization logic
        common to both new and loaded frames before WCS/data loading.
        """
        pass

    def _load_wcs_from_json(self, path):
        """
        Load WCS parameters for all detectors from a JSON file.

        Parses a JSON file containing WCS information (CRPIX, CD matrix, etc.)
        for each of the 16 detectors and creates astropy.wcs.WCS objects.
        Sets the center coordinates based on the frame's RA, Dec, and PA.

        Args:
            path (str): Path to the JSON file containing WCS parameters.
        """
        with open(path, "rt") as inp:
            pack_list = json.load(inp)
        self._wcs_list = []
        for det_i in range(16):
            pack = pack_list[det_i]
            header = {
                'CTYPE1': pack['CTYPE'][0],
                'CTYPE2': pack['CTYPE'][1],
                'CRPIX1': pack['CRPIX'][0],
                'CRPIX2': pack['CRPIX'][1],
                'CD1_1': pack['CD'][0][0],
                'CD1_2': pack['CD'][0][1],
                'CD2_1': pack['CD'][1][0],
                'CD2_2': pack['CD'][1][1],
                'NAXIS1': pack['NAXIS'][0],
                'NAXIS2': pack['NAXIS'][1],
            }
            self._wcs_list.append(wcs.WCS(header))
        self._set_center()

    def _set_center(self):
        """
        Set the sky center (CRVAL) and rotation (CD matrix) for all WCS objects.

        Uses the 'ra', 'dec', and 'pa' parameters of the PhotFrame instance
        to update the CRVAL and apply the rotation defined by the position angle
        to the CD matrix of each detector's WCS.
        """
        theta = np.radians(self.params['pa'])
        for wcs_ in self._wcs_list:
            wcs_.wcs.crval = [self.params['ra'], self.params['dec']]
            # Rotate by PA...
            cd = wcs_.wcs.cd
            r = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            wcs_.wcs.cd = np.dot(cd, r)

    def _load_wcs(self, ):
        """
        Load WCS information from the headers of the loaded FITS file.

        Iterates through the science extensions (DET*.SCI) of the loaded HDUList
        and creates an astropy.wcs.WCS object for each detector from its header.
        Assumes the FITS file follows the expected NISP multi-extension format.
        """
        self._wcs_list = []
        for det_i in range(16):
            detector_name = self.detector_list[det_i]
            try:
                header = self._hdul[f"DET{detector_name}"].header
            except KeyError:
                return
            self._wcs_list.append(wcs.WCS(header))

    def load_frame(self, frame_path=None):
        """
        Load frame data and metadata from a NISP photometric FITS file.

        Reads the primary header, extracts basic metadata (RA, Dec, PA, Obs ID, etc.),
        and loads the WCS information for all detectors from their respective
        extension headers.

        Args:
            frame_path (str, optional): Path to the FITS science frame file.
                                        Can be .gz compressed. Defaults to None.
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
        self.params['DATE-OBS'] = self._header['DATE-OBS']

        self.params['ra'] = self._header['RA']
        self.params['dec'] = self._header['DEC']
        self.params['pa'] = self._header['PA']

        self.params['PTGID'] = self._header['PTGID']
        self.params['OBS_ID'] = self._header['OBS_ID']
        self.params['DITHOBS'] = self._header['DITHOBS']
        self._setup()
        self._load_wcs()
        self.hdu_loaded = True

    @staticmethod
    def rotate_coord_nir_to_sir(x, y, det, inverse=False):
        """
        Rotate pixel coordinates between NIR default layout and SIR layout.

        The SIR pipeline often uses a different detector orientation convention
        than the raw NIR data. This function applies the necessary rotation
        and flipping based on the detector index.

        Args:
            x (np.ndarray): Input x pixel coordinates.
            y (np.ndarray): Input y pixel coordinates.
            det (np.ndarray): Detector indices (0-15) corresponding to coordinates.
            inverse (bool, optional): If True, rotate from SIR back to NIR.
                                      Defaults to False (NIR to SIR).

        Returns:
            tuple: (x_out, y_out) rotated pixel coordinates.
        """
        det = det * np.ones(len(x))

        flip = (det % 4) > 1
        x_out = np.zeros(len(x), dtype='d')
        y_out = np.zeros(len(x), dtype='d')
        if inverse:
            x_out[flip] = y[flip]
            y_out[flip] = 2040 - x[flip]
            x_out[~flip] = 2040 - y[~flip]
            y_out[~flip] = x[~flip]
        else:
            x_out[flip] = 2040 - y[flip]
            y_out[flip] = x[flip]
            x_out[~flip] = y[~flip]
            y_out[~flip] = 2040 - x[~flip]
        return x_out, y_out

    @staticmethod
    def rotate_image_nir_to_sir(det, im):
        """
        Rotate a detector image from NIR default layout to SIR layout.

        Applies the appropriate 90-degree rotation based on the detector index
        to match the SIR pipeline's orientation convention.

        Args:
            det_index (int): Detector index (0-15).
            im (np.ndarray): The 2D image array for the detector.

        Returns:
            np.ndarray: The rotated image array.
        """
        flip = (det % 4) > 1
        if flip:
            # Rotate data of the upper rows of detectors
            return np.rot90(im, -1)
        return np.rot90(im, 1)

    @property
    def hdul(self):
        """
        Access the FITS HDUList object for the frame.

        Loads the FITS file on first access if not already loaded.

        Returns:
            astropy.io.fits.HDUList: The HDUList object.

        Raises:
            AttributeError: If the frame was not initialized with a file path.
            FileNotFoundError: If the FITS file specified by 'frame_path' does not exist.
        """
        if self._loaded_hdu:
            return self._hdul
        self._hdul = fits.open(self.fits_path, decompress_in_memory=True)
        self.hdu_loaded = True
        return self._hdul

    def close(self):
        """
        Close the associated FITS file handle, if it was loaded.

        Deletes internal references to the HDUList and header.
        """
        if self.hdu_loaded:
            self._hdul.close()
            self.hdu_loaded = False

    def get_detector(self, detector_index):
        """
        Get the pixel data arrays (flux, RMS, DQ) for a specific detector.

        Retrieves the data from the corresponding FITS extensions. Applies
        rotation to match the SIR layout if `params['sir_layout']` is True.

        Args:
            detector_index (int): NISP detector index (0-15).

        Returns:
            tuple: (flux, rms, dq)
                   flux (ndarray): Science image data (e.g., counts or flux).
                   rms (ndarray): RMS noise map.
                   dq (ndarray): Data quality mask (0=good).

        Raises:
            IndexError: If detector_index is out of range (0-15).
            KeyError: If the required FITS extensions (e.g., DET*.SCI) are not found.
            AttributeError: If the FITS file has not been loaded.
        """
        det_name = self.detector_list[detector_index]

        try:
            header = self._hdul[f"DET{det_name}.SCI"].header
            flux = self._hdul[f"DET{det_name}.SCI"].data
            rms = self._hdul[f"DET{det_name}.RMS"].data
            dq = self._hdul[f"DET{det_name}.DQ"].data
        except KeyError:
            header = self._hdul[f"PERSIS.IMG.DET{det_name}"].header
            flux = self._hdul[f"PERSIS.IMG.DET{det_name}"].data
            rms = None
            dq = None

        self._wcs = wcs.WCS(header)

        if self.params['sir_layout']:
            flux = self.rotate_image_nir_to_sir(detector_index, flux)
            if rms is not None:
                rms = self.rotate_image_nir_to_sir(detector_index, rms)
            if dq is not None:
                dq = self.rotate_image_nir_to_sir(detector_index, dq)

        if rms is None:
            return flux

        return flux, rms, dq

    def radec_to_pixel(self, ra, dec):
        """
        Compute detector pixel coordinates (x, y, det_index) from RA, Dec.

        Iterates through the WCS of each detector to find which detector(s)
        contain the given sky coordinates. Applies rotation to SIR layout
        if `params['sir_layout']` is True.

        Args:
            ra (float or np.ndarray): Sky position RA in degrees.
            dec (float or np.ndarray): Sky position Dec in degrees.

        Returns:
            tuple: (x, y, detector_index)
                   x (float or ndarray): Pixel x-coordinate(s). -1 if off all detectors.
                   y (float or ndarray): Pixel y-coordinate(s). -1 if off all detectors.
                   detector_index (int or ndarray): Detector index (0-15). -1 if off all detectors.

        Raises:
            AttributeError: If WCS list (`_wcs_list`) has not been loaded.
        """

        try:
            len(ra)
            scalar = False
        except TypeError:
            scalar = True
            ra = np.array([ra])
            dec = np.array([dec])

        x_out = np.zeros(len(ra), dtype='d') - 1
        y_out = np.zeros(len(ra), dtype='d') - 1
        det_out = np.zeros(len(ra), dtype=int) - 1
        for det_i in range(16):
            wcs_ = self._wcs_list[det_i]
            x, y = wcs_.wcs_world2pix(ra, dec, 0)
            on_detector = (x >= 0) & (y >= 0) & (x < 2040) & (y < 2040)
            x_out[on_detector] = x[on_detector]
            y_out[on_detector] = y[on_detector]
            det_out[on_detector] = det_i
            if self.params['sir_layout']:
                x_out[on_detector], y_out[on_detector] = self.rotate_coord_nir_to_sir(
                    x_out[on_detector], y_out[on_detector], det_i)
        if scalar:
            return x_out[0], y_out[0], det_out[0]
        return x_out, y_out, det_out

    def pixel_to_radec(self, x, y, det):
        """
        Compute sky position (RA, Dec) from detector pixel coordinates.

        Uses the WCS object corresponding to the given detector index (`det`).
        Applies inverse rotation from SIR layout if `params['sir_layout']` is True.

        Args:
            x (float or np.ndarray): Detector pixel x-coordinate(s).
            y (float or np.ndarray): Detector pixel y-coordinate(s).
            det (int or np.ndarray): NISP detector index/indices (0-15).

        Returns:
            tuple: (RA, Dec)
                   RA (float or ndarray): Right Ascension coordinate(s) in degrees. NaN if input pixel is off-detector.
                   Dec (float or ndarray): Declination coordinate(s) in degrees. NaN if input pixel is off-detector.

        Raises:
            AttributeError: If WCS list (`_wcs_list`) has not been loaded.
            IndexError: If any value in `det` is out of range (0-15).
        """
        try:
            len(x)
            scalar = False
        except TypeError:
            scalar = True
            x = np.array([x])
            y = np.array([y])
        det = (np.ones(len(x))*det).astype(int)

        if self.params['sir_layout']:
            x, y = self.rotate_coord_nir_to_sir(x, y, det, inverse=True)

        ra_out = np.zeros(len(x), dtype='d')
        dec_out = np.zeros(len(x), dtype='d')

        for det_i in np.unique(det):
            sel = det == det_i
            wcs_ = self._wcs_list[det_i]
            ra, dec = wcs_.wcs_pix2world(x[sel], y[sel], 0)
            ra_out[sel] = ra
            dec_out[sel] = dec
        if scalar:
            return ra_out[0], dec_out[0]
        return ra_out, dec_out

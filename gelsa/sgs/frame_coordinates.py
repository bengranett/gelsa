import h5py
import numpy as np
from .poly import fast_cheb_eval, fast_cheb_coeffs_kernel, fast_polyval2d
from .tan_projection import _tan_projection_jit, _tan_projection_inverse_jit


class FrameCoordinates:
    mm_pix_scale = 16.606974929470375  # arcsec/mm

    def __init__(self, params, optical_params=None, detector_model=None):
        """Initialise coordinate transforms for one spectral frame.

        Parameters
        ----------
        params : dict
            Frame metadata; must contain 'RA', 'DEC', 'PA', and 'grism_name'.
        optical_params : dict, optional
            SGS optical model parameters (distortion coefficients, ExtraTilt).
        detector_model : DetectorModel, optional
            Detector geometry used for FOV-to-pixel transforms and FOV masking.
        """
        self.params = params
        if optical_params is not None:
            self.optical_params = optical_params
        if detector_model is not None:
            self.detector_model = detector_model
        self._set_wcs()

    def _set_wcs(self):
        """Build the WCS dict from the current pointing (RA, DEC, PA)."""
        ra = self.params['RA']
        dec = self.params['DEC']
        pa = self.params['PA']
        orientation = np.deg2rad(-pa + 90)
        cd_matrix = np.array([[np.cos(orientation), -np.sin(orientation)],
                              [np.sin(orientation), np.cos(orientation)]])
        rot_matrix = np.array([[0, -1], [1, 0]])
        cd_matrix = rot_matrix.dot(cd_matrix)
        self._wcs = {
            'crval': np.array([ra, dec], dtype=np.float64),
            'crpix': np.array([1., 1.], dtype=np.float64),
            'cd': self.mm_pix_scale * cd_matrix / 3600.,
            'ctype': ('RA---TAN', 'DEC--TAN'),
        }

    @property
    def detector_model(self):
        """DetectorModel providing FOV geometry and pixel transforms."""
        return self._detector_model

    @detector_model.setter
    def detector_model(self, detector_model):
        """Set the detector model.

        Parameters
        ----------
        detector_model : DetectorModel
            Detector geometry used for FOV-to-pixel transforms and FOV masking.
        """
        self._detector_model = detector_model

    @property
    def optical_params(self):
        """SGS optical model dict (distortion coefficients, ExtraTilt)."""
        return self._optical_params

    @optical_params.setter
    def optical_params(self, optical_params):
        """Set optical params and rebuild the extra-tilt rotation matrix.

        Parameters
        ----------
        optical_params : dict
            SGS optical model parameters (distortion coefficients, ExtraTilt).
        """
        self._optical_params = optical_params
        self._set_extra_tilt_matrix()

    def _set_extra_tilt_matrix(self):
        """Build the 2x2 rotation matrix for the optical ExtraTilt angle."""
        tilt = self.optical_params['ExtraTilt']
        cos_ = np.cos(np.deg2rad(tilt))
        sin_ = np.sin(np.deg2rad(tilt))
        self._extra_tilt_matrix = np.array([[cos_, -sin_], [sin_, cos_]])

    def set_pointing_center(self, ra, dec, pa):
        """Update the pointing and rebuild the WCS parameter dict.

        Parameters
        ----------
        ra, dec : float
            New pointing centre in degrees.
        pa : float
            New position angle in degrees.
        """
        self.params['RA'] = ra
        self.params['DEC'] = dec
        self.params['PA'] = pa
        self._set_wcs()

    def update_optical_model_from_location_table(self, location_table_path):
        """Replace WCS parameters with values from an SGS location table.

        Parameters
        ----------
        location_table_path : str
            Path to the HDF5 location table produced by the SGS pipeline.
        """
        # print("updating WCS from location table")
        with h5py.File(location_table_path) as lt:
            params = lt['OPT']
            cd_matrix = params['CD'][:]
            crpix = params['CRPIX'][:]
            crval = params['CRVAL'][:]
            ctype = params['CTYPE'][:]
        self._wcs['cd'] = np.asarray(cd_matrix, dtype=np.float64)
        self._wcs['crval'] = np.asarray(crval, dtype=np.float64)
        self._wcs['crpix'] = np.asarray(crpix, dtype=np.float64)
        self._wcs['ctype'] = ctype

    def arcsec_to_pixel(self, x):
        """Convert angular size from arcseconds to pixels.

        Parameters
        ----------
        x : float or ndarray
            Value(s) in arcseconds.

        Returns
        -------
        float or ndarray
            Value(s) in pixels.
        """
        return x / self.params['pixel_scale']

    def pixel_to_arcsec(self, x):
        """Convert angular size from pixels to arcseconds.

        Parameters
        ----------
        x : float or ndarray
            Value(s) in pixels.

        Returns
        -------
        float or ndarray
            Value(s) in arcseconds.
        """
        return x * self.params['pixel_scale']

    def is_within_fov(self, xpos, ypos):
        """Check whether a point falls within the overall detector envelope.

        Parameters
        ----------
        xpos, ypos : float
            FOV position in mm.

        Returns
        -------
        bool
            True if the point lies inside the rectangular detector envelope.
        """
        limits = self.detector_model.envelope
        return bool(
            (limits[0][0] < xpos < limits[1][0])
            and (limits[0][1] < ypos < limits[1][1])
        )

    def getUndistortedObjectPosition(self, ra, dec):
        """TAN projection: sky coordinates to undistorted FOV mm positions.

        Parameters
        ----------
        ra, dec : float or ndarray
            Sky coordinates in degrees.

        Returns
        -------
        ndarray, shape (2,) or (2, N)
            Undistorted FOV positions [x, y] in mm.
        """
        ra_arr = np.asarray(ra, dtype=np.float64)
        dec_arr = np.asarray(dec, dtype=np.float64)
        scalar = ra_arr.ndim == 0
        ra_1d = ra_arr.ravel()
        dec_1d = dec_arr.ravel()

        wcs = self._wcs
        ra0, dec0 = wcs['crval']
        crpix = wcs['crpix']
        cdinv = np.linalg.inv(wcs['cd'])

        x, y = _tan_projection_jit(
            ra_1d, dec_1d, ra0, dec0, cdinv, crpix[0], crpix[1]
        )

        if scalar:
            return np.array([x[0], y[0]])
        return np.array([x, y])

    def getUndistortedObjectPosition_inverse(self, x, y):
        """Inverse TAN projection: FOV mm coordinates to sky positions.

        Parameters
        ----------
        x, y : float or ndarray
            Undistorted FOV positions in mm.

        Returns
        -------
        ndarray, shape (2,) or (2, N)
            Sky coordinates [ra, dec] in degrees.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        scalar = x_arr.ndim == 0
        x_1d = x_arr.ravel()
        y_1d = y_arr.ravel()

        wcs = self._wcs
        ra0, dec0 = wcs['crval']
        crpix = wcs['crpix']
        cd = wcs['cd']

        ra, dec = _tan_projection_inverse_jit(
            x_1d, y_1d, ra0, dec0, cd, crpix[0], crpix[1]
        )

        if scalar:
            return np.array([ra[0], dec[0]])
        return np.array([ra, dec])

    def apply_distortion_jit(self, xmm_undist, ymm_undist, coef_x, coef_y):
        """Apply a 2D polynomial distortion map to FOV coordinates.

        Parameters
        ----------
        xmm_undist, ymm_undist : ndarray
            Undistorted FOV positions in mm.
        coef_x, coef_y : ndarray
            2D polynomial coefficient matrices for x and y distortion.

        Returns
        -------
        xmm, ymm : ndarray
            Distorted FOV positions in mm.
        """
        xmm = fast_polyval2d(xmm_undist, ymm_undist, coef_x)
        ymm = fast_polyval2d(xmm_undist, ymm_undist, coef_y)
        return xmm, ymm

    def getReferencePosition_jit(self, ra=None, dec=None,
                                 xmm_undist=None, ymm_undist=None,
                                 order=1):
        """Map sky coordinates to distorted FOV mm coords (no spectral shift).

        Applies the SGS distortion polynomial on top of the TAN projection.
        Either (ra, dec) or pre-computed (xmm_undist, ymm_undist) is required.

        Parameters
        ----------
        ra, dec : ndarray, optional
            Sky coordinates in degrees.
        xmm_undist, ymm_undist : ndarray, optional
            Undistorted FOV positions in mm from getUndistortedObjectPosition.
        order : int
            Dispersion order (0 or 1); selects the distortion coefficient set.

        Returns
        -------
        xmm, ymm : ndarray
            Distorted reference FOV positions in mm.
        """
        if xmm_undist is None:
            xmm_undist, ymm_undist = self.getUndistortedObjectPosition(
                ra, dec
            )

        if order == 1:
            coef = self.optical_params['Reference']
        elif order == 0:
            coef = self.optical_params['Displacements']
        else:
            raise ValueError("Invalid order")

        valid = self.detector_model.is_within_fov(
            xmm_undist, ymm_undist, eps_mm=5
        )
        num_valid = np.sum(valid)

        xmm = xmm_undist.copy()  # avoid modifying input
        ymm = ymm_undist.copy()
        if num_valid > 0:
            x_dist, y_dist = self.apply_distortion_jit(
                xmm_undist[valid], ymm_undist[valid],
                coef['x'], coef['y']
            )
            xmm[valid] = x_dist
            ymm[valid] = y_dist

        if self.optical_params['ExtraTilt'] != 0:
            raise NotImplementedError

        return xmm, ymm

    def apply_extra_tilt_rotation(self, pos_mm, pivot_mm):
        """Rotate a FOV position around a pivot by the ExtraTilt angle.

        Parameters
        ----------
        pos_mm : array-like, shape (2,)
            Input FOV position [x, y] in mm.
        pivot_mm : array-like, shape (2,)
            Pivot point [x, y] in mm to rotate around.

        Returns
        -------
        ndarray, shape (2,)
            Rotated FOV position [x, y] in mm.
        """
        _pos_mm = np.array(pos_mm) - np.array(pivot_mm)
        return self._extra_tilt_matrix.dot(_pos_mm) + pivot_mm

    def normalize(self, x, lim):
        """Map x from [lim[0], lim[1]] to [-1, 1] for Chebyshev evaluation.

        Parameters
        ----------
        x : float or ndarray
            Input value(s) to normalize.
        lim : array-like, shape (2,)
            [lower, upper] bounds of the input domain.

        Returns
        -------
        float or ndarray
            Normalized value(s) in [-1, 1].
        """
        return (2 * x - lim[0] - lim[1]) / (lim[1] - lim[0])

    def get_coefficients_jit(self, x, y, params):
        """Evaluate 2D Chebyshev model to get per-point polynomial coeffs.

        Parameters
        ----------
        x, y : ndarray
            FOV positions in mm.
        params : dict
            Dispersion model with keys 'global_ranges' and 'model'
            (list of 2D Chebyshev coefficient matrices, one per degree).

        Returns
        -------
        result : ndarray, shape (n_coeffs, N)
            Evaluated Chebyshev coefficients for each of the N input positions.
        """
        lim_x = params['global_ranges'][0]
        lim_y = params['global_ranges'][1]
        x_norm = self.normalize(x, lim_x)
        y_norm = self.normalize(y, lim_y)

        model_matrices = params.get('_model_matrices_cache')
        if model_matrices is None:
            n = len(params['model'])
            shape = params['model'][0].shape
            model_matrices = np.zeros((n, shape[0], shape[1]))
            for i in range(n):
                model_matrices[i] = params['model'][i]
            params['_model_matrices_cache'] = model_matrices

        result = fast_cheb_coeffs_kernel(x_norm, y_norm, model_matrices)

        return result

    def displacement_jit(self, x, y, wavelength, order=1):
        """Compute spectral displacement from FOV position and wavelength.

        Uses the IDS (inverse dispersion) and CRV (curvature) models to
        convert wavelength to an along-dispersion shift dx, then a
        cross-dispersion shift dy, and adds them to the input positions.

        Parameters
        ----------
        x, y : ndarray or float
            Reference FOV positions in mm.
        wavelength : float or ndarray
            Wavelength in Angstroms.
        order : int
            Dispersion order (0 or 1).

        Returns
        -------
        res_x, res_y : ndarray or float
            Dispersed FOV positions in mm.
        """
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        wavelength_arr = np.ones(len(x)) * wavelength

        ids_params = self.ids_params['Orders'][order]
        crv_params = self.crv_params['Orders'][order]
        c_ids = self.get_coefficients_jit(x, y, ids_params)
        wavelength_norm = self.normalize(
            wavelength_arr, ids_params['local_ranges']
        )
        dx = fast_cheb_eval(wavelength_norm, c_ids)
        c_crv = self.get_coefficients_jit(x, y, crv_params)
        dx_norm = self.normalize(dx, crv_params['local_ranges'])
        dy = fast_cheb_eval(dx_norm, c_crv)
        grism = self.params['grism_name']
        if grism in ['RGS000', 'BGS000']:
            res_x, res_y = x + dx, y + dy
        elif grism == 'RGS180':
            res_x, res_y = x - dx, y - dy
        else:
            raise ValueError("Unknown grism!")

        if scalar:
            return res_x[0], res_y[0]
        return res_x, res_y

    def radec_to_fov(self, ra, dec, wavelength, dispersion_order=1):
        """Map sky coordinates and wavelength to FOV mm coordinates.

        Applies TAN projection, optical distortion, and spectral displacement
        in sequence.

        Parameters
        ----------
        ra, dec : float or ndarray
            Sky coordinates in degrees.
        wavelength : float or ndarray
            Wavelength in Angstroms.
        dispersion_order : int
            Dispersion order (0 or 1).

        Returns
        -------
        xfov, yfov : float or ndarray
            FOV positions in mm.
        """
        scalar = np.ndim(ra) == 0
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)

        wavelength_arr = np.ones(len(ra)) * wavelength

        if dispersion_order not in [0, 1]:
            raise ValueError("Order error")
        xmm_undist, ymm_undist = self.getUndistortedObjectPosition(ra, dec)
        ref_x, ref_y = self.getReferencePosition_jit(
            xmm_undist=xmm_undist, ymm_undist=ymm_undist,
            order=dispersion_order
        )
        valid = self.detector_model.is_within_fov(ref_x, ref_y, eps_mm=5)
        xfov = ref_x.copy()  # avoid modifying ref_x
        yfov = ref_y.copy()
        xfov[valid], yfov[valid] = self.displacement_jit(
            ref_x[valid], ref_y[valid], wavelength_arr[valid],
            order=dispersion_order
        )
        if self.optical_params['ExtraTilt'] != 0:
            raise NotImplementedError

        if scalar:
            return xfov[0], yfov[0]
        return xfov, yfov

    def _newton_fov_to_radec(self, xfov, yfov, wavelength, ra, dec,
                             dispersion_order=1, n_iter=2, h=1e-6):
        """Vectorized Newton solver for the inverse of radec_to_fov.

        Each iteration evaluates radec_to_fov three times on all N points
        simultaneously (centre + two finite-difference steps) to form and
        apply the 2x2 Jacobian update.

        Parameters
        ----------
        xfov, yfov : ndarray
            Target FOV positions in mm.
        wavelength : ndarray
            Wavelength in Angstroms, one value per point.
        ra, dec : ndarray
            Initial sky-coordinate guess in degrees (copied before update).
        dispersion_order : int
            Dispersion order passed through to radec_to_fov.
        n_iter : int
            Number of Newton iterations.
        h : float
            Finite-difference step size in degrees for Jacobian estimation.

        Returns
        -------
        ra, dec : ndarray
            Refined sky coordinates in degrees.
        """
        ra = ra.copy()
        dec = dec.copy()
        for _ in range(n_iter):
            fx, fy = self.radec_to_fov(ra, dec, wavelength, dispersion_order)
            fx_h, fy_h = self.radec_to_fov(
                ra + h, dec, wavelength, dispersion_order
            )
            fx_k, fy_k = self.radec_to_fov(
                ra, dec + h, wavelength, dispersion_order
            )

            # 2x2 Jacobian columns
            J00 = (fx_h - fx) / h  # d(xfov)/d(ra)
            J10 = (fy_h - fy) / h  # d(yfov)/d(ra)
            J01 = (fx_k - fx) / h  # d(xfov)/d(dec)
            J11 = (fy_k - fy) / h  # d(yfov)/d(dec)

            det = J00 * J11 - J01 * J10
            rx = xfov - fx
            ry = yfov - fy
            ra += (J11 * rx - J01 * ry) / det
            dec += (J00 * ry - J10 * rx) / det

        return ra, dec

    def fov_to_radec(self, xfov, yfov, wavelength, dispersion_order=1):
        """Invert radec_to_fov: map FOV mm coordinates and wavelength to sky.

        Uses the inverse TAN projection as an initial guess, then refines
        with a vectorized Newton solver.

        Parameters
        ----------
        xfov, yfov : float or ndarray
            FOV positions in mm.
        wavelength : float or ndarray
            Wavelength in Angstroms.
        dispersion_order : int
            Dispersion order (0 or 1).

        Returns
        -------
        ra, dec : float or ndarray
            Sky coordinates in degrees; NaN for points outside the FOV.
        """
        scalar = np.ndim(xfov) == 0
        xfov = np.atleast_1d(xfov)
        yfov = np.atleast_1d(yfov)

        wavelength = np.ones(len(xfov)) * wavelength

        valid = self.detector_model.is_within_fov(xfov, yfov, eps_mm=10)

        guess = self.getUndistortedObjectPosition_inverse(
            xfov[valid], yfov[valid]
        )
        ra_, dec_ = self._newton_fov_to_radec(
            xfov[valid], yfov[valid], wavelength[valid],
            guess[0], guess[1], dispersion_order
        )

        ra = np.zeros(len(xfov), dtype='d') + np.nan
        dec = np.zeros(len(xfov), dtype='d') + np.nan
        ra[valid] = ra_
        dec[valid] = dec_
        if scalar:
            return ra[0], dec[0]
        return ra, dec

    def radec_to_pixel(self, ra, dec, wavelength, dispersion_order=1):
        """Map sky coordinates and wavelength to detector pixel position.

        Parameters
        ----------
        ra, dec : float or ndarray
            Sky coordinates in degrees.
        wavelength : float or ndarray
            Wavelength in Angstroms.
        dispersion_order : int
            Dispersion order (0 or 1).

        Returns
        -------
        x, y : float or ndarray
            Pixel coordinates (1-indexed centre convention).
        det_id : int or ndarray
            Detector ID; -1 for points that fall off all detectors.
        """
        if dispersion_order not in [0, 1]:
            raise ValueError("get_pixel not initialized for dispersion order "
                             f"{dispersion_order}")

        xfov, yfov = self.radec_to_fov(
            ra, dec, wavelength, dispersion_order=dispersion_order
        )

        x, y, det_id = self.detector_model.getPixel(xfov, yfov)

        x += 0.5
        y += 0.5

        return x, y, det_id

    def pixel_to_radec(self, x, y, detid, wavelength, dispersion_order=1):
        """Invert radec_to_pixel: pixel coordinates and wavelength to sky.

        Parameters
        ----------
        x, y : ndarray
            Pixel coordinates (1-indexed centre convention).
        detid : ndarray of int
            Detector ID for each point.
        wavelength : float or ndarray
            Wavelength in Angstroms.
        dispersion_order : int
            Dispersion order (0 or 1).

        Returns
        -------
        ra, dec : ndarray
            Sky coordinates in degrees.
        """
        if dispersion_order not in [0, 1]:
            raise ValueError("get_pixel not initialized for dispersion order "
                             f"{dispersion_order}")
        x_ = x - 0.5
        y_ = y - 0.5
        xfov, yfov = self.detector_model.getFOVPosition(x_, y_, detid)
        out = self.fov_to_radec(
            xfov, yfov, wavelength, dispersion_order=dispersion_order
        )
        return out

import os
import json

from . import specframe, photframe
from . import zero_order_mask
from .sgs import (
    detector_model,
    optical_model,
    displacement_model,
    sensitivity_model,
    relative_flux,
    psf_model
)
from .esa import datalabs_paths


class Gelsa:
    """
    Main class for handling Gelsa simulation setup and frame loading.

    This class manages configuration settings, loads calibration models
    (detector, optical, dispersion, sensitivity, relative flux, PSF),
    and provides methods to create or load spectral (`SpecFrame`) and
    photometric (`PhotFrame`) frames based on the loaded configuration
    and calibration data.
    """

    _default_config = {
        'workdir': '.',
        'datadir': '.',
        'calibdir': None,
        'config_file': None,
        'abs_file': "EUC_SIR_W-AbsoluteFlux-SCALE_ALL_20250218T221007.269385Z.fits",
        'opt_file': 'SIR_Calib_Opt_EUCLID_2.0.1-CALIBRATION-pcasenov-PLAN-000000-MT0S7VWE-20250123-180643-0-new_opt_cal-0.xml',
        'ids_file': 'SIR_Calib_Ids_EUCLID_2.0.1-CALIBRATION-pcasenov-PLAN-000001-GQLJE83P-20250211-181132-0-new_ids_calib-0.xml',
        'crv_file': 'SIR_Calib_Crv_EUCLID_2.0.1-CALIBRATION-pcasenov-PLAN-000001-6NJP8VE4-20250211-163247-0-new_crv_cal-0.xml',
        'location_table': None,
        'detector_slots_path': 'EUC_SIR_DETMODEL_REAL_DATA_OP_02.csv',
        'rel_flux_file': 'EUC_SIR_W-RelativeFlux-SCALE_20250716T035001.134254Z.fits',
        'psf_file': 'psf_wavelength.json',
        # options
        'use_psf': True,
        'use_chromatic_psf': False,
        'use_relative_flux_loss': True,
        'zero_order_catalog': None,
        'nir_wcs': 'nir_wcs.json',
        'use_persistence_mask': False,
    }

    def __init__(self, config_file=None, config={}, **kwargs):
        """
        Initialize the Gelsa object.

        Loads configuration from defaults, a specified file, a dictionary,
        and keyword arguments.

        Args:
            config_file (str, optional): Path to a JSON configuration file. Defaults to None.
            config (dict, optional): Dictionary containing configuration settings. Defaults to {}.
            **kwargs: Additional configuration settings passed as keyword arguments.
        """
        self.config = self._default_config.copy()
        self.load_config(config_file, config, **kwargs)
        self.find_calibdir(self)
        self.check_calibdir()
        
    def find_calibdir(self, config_file):
        """ """
        if self.config['calibdir'] is not None:
            return
        for path in datalabs_paths.data_paths:
            if os.path.exists(path):
                print(f"Found calibration directory {path}")
                self.config['calibdir'] = path
                return
        print("warning, calibdir is not set")
        
    def check_calibdir(self):
        """ """
        file_list = ('abs_file', 'opt_file', 'crv_file', 'ids_file', 'detector_slots_path', 'rel_flux_file')
        calibdir = self.config['calibdir']
        print(f"looking in {calibdir}")
        for key in file_list:
            filename = self.config[key]
            path = os.path.join(calibdir, filename)
            if os.path.exists(path):
                message = 'found'
            else:
                message = 'not found'
            print(f"{key}: {filename} - {message}")

    def update_config(self, config):
        """
        Update the current configuration with new settings.

        Only keys already present in the default configuration are updated.
        A warning is printed for unknown keys.

        Args:
            config (dict): Dictionary containing configuration settings to update.
        """
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f"Warning! unknown option: {key}={value}")

    def load_config(self, config_path=None, config=None, **kwargs):
        """
        Load configuration settings from multiple sources.

        Configuration is loaded in the following order, with later sources
        overwriting earlier ones:
        1. Default configuration (`_default_config`).
        2. Configuration from a JSON file (`config_path`).
        3. Configuration from a dictionary (`config`).
        4. Configuration from keyword arguments (`**kwargs`).

        Args:
            config_path (str, optional): Path to a JSON configuration file. Defaults to None.
            config (dict, optional): Dictionary containing configuration settings. Defaults to None.
            **kwargs: Additional configuration settings passed as keyword arguments.
        """
        if config_path:
            with open(config_path, "rt") as inp:
                config_from_file = json.load(inp)
            self.update_config(config_from_file)
        if config:
            self.update_config(config)
        self.update_config(kwargs)

    def write_config(self, path):
        """
        Write the current configuration to a JSON file.

        Args:
            path (str): The path to the output JSON file.
        """
        with open(path, "wt") as out:
            json.dump(self.config, out, indent=" ")

    @property
    def detector_model(self):
        """
        Load and return the DetectorModel instance.

        Loads the detector model based on the 'detector_slots_path' specified
        in the configuration. The model is cached after the first load.

        Returns:
            detector_model.DetectorModel: The loaded detector model instance.
        """
        try:
            return self._detector_model
        except AttributeError:
            pass
        path = os.path.join(self.config['workdir'], self.config['calibdir'], self.config['detector_slots_path'])
        self._detector_model = detector_model.DetectorModel(
            detector_slots_path=path
        )
        return self._detector_model

    @property
    def optical_model(self):
        """
        Load and return the OpticalModel instance.

        Loads the optical model based on the 'opt_file' specified in the
        configuration. The model is cached after the first load.

        Returns:
            optical_model.OpticalModel: The loaded optical model instance.
        """
        try:
            return self._optical_model
        except AttributeError:
            pass
        path = os.path.join(self.config['workdir'], self.config['calibdir'], self.config['opt_file'])
        self._optical_model = optical_model.OpticalModel(
            path=path
        )
        return self._optical_model

    @property
    def dispersion_model(self):
        """
        Load and return the dispersion models (IDS and CRV).

        Loads the Inverse Dispersion Solution (IDS) and Curvature (CRV) models
        based on the 'ids_file' and 'crv_file' specified in the configuration.
        The models are cached after the first load.

        Returns:
            dict: A dictionary containing the loaded 'ids' and 'crv'
                  DisplacementModel instances.
        """
        try:
            return self._dispersion_model
        except AttributeError:
            pass
        path = os.path.join(self.config['workdir'], self.config['calibdir'], self.config['ids_file'])
        ids_model = displacement_model.DisplacementModel(
            path=path
        )
        path = os.path.join(self.config['workdir'], self.config['calibdir'], self.config['crv_file'])
        crv_model = displacement_model.DisplacementModel(
            path=path
        )
        self._dispersion_model = {
            'ids': ids_model,
            'crv': crv_model
        }
        return self._dispersion_model

    @property
    def sensitivity_model(self):
        """
        Load and return the SensitivityModel instance.

        Loads the sensitivity (absolute calibration) model based on the
        'abs_file' specified in the configuration. The model is cached
        after the first load.

        Returns:
            sensitivity_model.SensitivityModel: The loaded sensitivity model instance.
        """
        try:
            return self._sensitivity_model
        except AttributeError:
            pass
        path = os.path.join(self.config['workdir'], self.config['calibdir'], self.config['abs_file'])
        self._sensitivity_model = sensitivity_model.SensitivityModel(
            path=path,
            datadir=self.config['datadir'],
        )
        return self._sensitivity_model

    @property
    def relative_flux_model(self):
        """
        Load and return the RelativeFluxCalibration instance, if enabled.

        Loads the relative flux loss model based on the 'rel_flux_file'
        specified in the configuration, only if 'use_relative_flux_loss'
        is True. Otherwise, returns None. The model is cached after the
        first load.

        Returns:
            relative_flux.RelativeFluxCalibration or None: The loaded relative
            flux model instance or None if disabled.
        """
        try:
            return self._relative_flux_model
        except AttributeError:
            pass
        if self.config['use_relative_flux_loss']:
            path = os.path.join(self.config['workdir'], self.config['calibdir'], self.config['rel_flux_file'])
            self._relative_flux_model = relative_flux.RelativeFluxCalibration(
                path=path,
                datadir=self.config['datadir']
            )
        else:
            self._relative_flux_model = None
        return self._relative_flux_model

    @property
    def psf_model(self):
        """
        Load and return the appropriate PSF model instance.

        Loads either the default PSF model, the chromatic PSF model, or None,
        based on the 'use_psf' and 'use_chromatic_psf' settings in the
        configuration. The model is cached after the first load.

        Returns:
            psf_model.PSFModel, psf_model.DefaultPSFModel, or None: The loaded
            PSF model instance.
        """
        try:
            return self._psf_model
        except AttributeError:
            pass
        if not self.config['use_psf']:
            self._psf_model = None
        elif self.config['use_chromatic_psf']:
            path = os.path.join(self.config['workdir'], self.config['calibdir'], self.config['psf_file'])
            self._psf_model = psf_model.PSFModel(
                path=path,
            )
        else:
            self._psf_model = psf_model.DefaultPSFModel()
        return self._psf_model

    @property
    def zero_order_mask(self):
        """
        Load and return the ZeroOrderMask instance, if enabled.

        Loads the zero-order mask based on the 'zero_order_catalog' specified
        in the configuration. If 'zero_order_catalog' is not set, returns None.
        The mask object is cached after the first load.

        Returns:
            zero_order_mask.ZeroOrderMask or None: The loaded zero-order mask
            instance or None if disabled.
        """
        if not self.config['zero_order_catalog']:
            return None
        try:
            return self._zero_order_mask
        except AttributeError:
            path = os.path.join(self.config['workdir'], self.config['calibdir'], self.config['zero_order_catalog'])
            self._zero_order_mask = zero_order_mask.ZeroOrderMask(path)
        return self._zero_order_mask

    def load_spec_frame(self, frame_path, loctable_path=None,
                        persistence_path=None, **frame_args):
        """
        Load an existing spectral frame (SpecFrame) from a file.

        Initializes a SpecFrame, loads data from the specified frame path
        (adjusting for workdir), optionally loads a location table, and sets
        up all relevant calibration models (optical, detector, dispersion, etc.).

        Args:
            frame_path (str): Path to the spectral frame file (relative to workdir).
            loctable_path (str, optional): Path to the location table file
                                           (relative to workdir). Defaults to None.

        Returns:
            specframe.SpecFrame: The loaded and configured spectral frame instance.
        """
        frame = specframe.SpecFrame(**frame_args)

        full_frame_path = os.path.join(self.config['workdir'], frame_path)

        full_loctable_path = None
        full_persistence_path = None

        if loctable_path is not None:
            full_loctable_path = os.path.join(self.config['workdir'], loctable_path)

        if self.config['use_persistence_mask'] and persistence_path is not None:
            full_persistence_path = os.path.join(self.config['workdir'], persistence_path)

        frame.load_frame(
            full_frame_path,
            loctable_path=full_loctable_path,
        )

        frame.set_optical_model(self.optical_model)
        frame.set_detector_model(self.detector_model)
        frame.set_dispersion_model(self.dispersion_model)
        frame.set_sensitivity(self.sensitivity_model)
        frame.set_relative_flux_model(self.relative_flux_model)
        frame.set_psf_model(self.psf_model)
        frame.set_zero_order_mask(self.zero_order_mask)
        if self.config['use_persistence_mask']:
            frame.set_persistence_mask(full_persistence_path)

        return frame

    def new_spec_frame(self, ra, dec, pa, **params):
        """
        Create a new, empty spectral frame (SpecFrame) at a given pointing.

        Initializes a SpecFrame with the specified pointing (RA, Dec, PA) and
        any additional parameters. Sets up all relevant calibration models
        (optical, detector, dispersion, etc.). This frame does not contain
        pixel data initially.

        Args:
            ra (float): Right Ascension of the frame center (degrees).
            dec (float): Declination of the frame center (degrees).
            pa (float): Position Angle of the frame (degrees).
            **params: Additional parameters to pass to the SpecFrame constructor.

        Returns:
            specframe.SpecFrame: The newly created and configured spectral frame instance.
        """
        frame = specframe.SpecFrame(RA=ra, DEC=dec, PA=pa, **params)
        frame.set_optical_model(self.optical_model)
        frame.set_detector_model(self.detector_model)
        frame.set_dispersion_model(self.dispersion_model)
        frame.set_sensitivity(self.sensitivity_model)
        frame.set_relative_flux_model(self.relative_flux_model)
        frame.set_psf_model(self.psf_model)
        frame.set_zero_order_mask(None)
        # Note: Zero order mask is not set here as it depends on the frame data
        return frame

    # aliases
    load_frame = load_spec_frame
    new_frame = new_spec_frame

    def load_phot_frame(self, frame_path, **params):
        """
        Load an existing photometric frame (PhotFrame) from a file.

        Initializes a PhotFrame, loads data from the specified frame path
        (adjusting for workdir), and sets up the WCS.

        Args:
            frame_path (str): Path to the photometric frame file (relative to workdir).
            **params: Additional parameters to pass to the PhotFrame constructor.

        Returns:
            photframe.PhotFrame: The loaded photometric frame instance.
        """
        frame = photframe.PhotFrame(**params)
        full_frame_path = os.path.join(self.config['workdir'], frame_path)
        frame.load_frame(full_frame_path)
        return frame

    def new_phot_frame(self, ra=None, dec=None, pa=None, **params):
        """
        Create a new photometric frame (PhotFrame) definition.

        Initializes a PhotFrame with the specified pointing (RA, Dec, PA) and
        any additional parameters. Loads the WCS information from the JSON file
        specified in the configuration ('nir_wcs').

        Args:
            ra (float, optional): Right Ascension of the frame center (degrees). Defaults to None.
            dec (float, optional): Declination of the frame center (degrees). Defaults to None.
            pa (float, optional): Position Angle of the frame (degrees). Defaults to None.
            **params: Additional parameters to pass to the PhotFrame constructor.

        Returns:
            photframe.PhotFrame: The newly created photometric frame instance.
        """
        frame = photframe.PhotFrame(ra=ra, dec=dec, pa=pa, **params)
        wcs_path = os.path.join(self.config['workdir'], self.config['calibdir'], self.config['nir_wcs'])
        frame._load_wcs_from_json(wcs_path)
        return frame

.. _concepts:

*************
Core concepts
*************

This page introduces the objects you use in every `gelsa` workflow.


Frames, grisms and detectors
============================

A NISP spectroscopic exposure is represented by a :class:`~gelsa.specframe.SpecFrame`.
It holds the pixel data of the 16 detectors (numbered 0–15) together with the
instrument models that describe where light lands on them.

Each exposure is taken with one of three grisms, set by the ``grism_name``
parameter:

* ``RGS000``: red grism, 0° dispersion direction
* ``RGS180``: red grism, 180° dispersion direction
* ``BGS000``: blue grism

A frame can be loaded from a SIR science frame FITS file, or created empty at a
chosen pointing to simulate data (see :doc:`simulation`).


Coordinates
===========

At its core, `gelsa` maps between sky position, wavelength and detector pixel,
using the SIR optical, dispersion and detector models:

.. code:: python

    # sky position + wavelength -> pixel position and detector index
    x, y, det = frame.radec_to_pixel(ra, dec, wavelength)

    # pixel position on a detector + wavelength -> sky position
    ra, dec = frame.pixel_to_radec(x, y, det, wavelength)

Conventions used throughout the library:

* RA and Dec are in degrees.
* Wavelengths are in Ångström, in vacuum.
* ``det`` is the detector index 0–15, and −1 when the position falls off the
  focal plane.
* The functions take and return numpy arrays, so you can transform many
  points in one call.


The ``Gelsa`` object and calibration files
==========================================

:class:`gelsa.Gelsa` is the entry point. It reads a configuration, loads the
SIR calibration products when they are first used, and creates frames with all
models attached:

.. code:: python

    import gelsa

    G = gelsa.Gelsa(calibdir="path/to/SIR/calibration")

    frame = G.load_spec_frame("EUC_SIR_W-SCIFRM_....fits")   # real data
    sim = G.new_spec_frame(ra=150.1, dec=2.2, pa=0)          # empty frame to simulate

The calibration products are looked up in ``calibdir``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Configuration key
     - SIR calibration product
   * - ``opt_file``
     - Optical model (``SIR_Calib_Opt_*.xml``)
   * - ``ids_file``
     - Inverse dispersion solution (``SIR_Calib_Ids_*.xml``)
   * - ``crv_file``
     - Trace curvature (``SIR_Calib_Crv_*.xml``)
   * - ``abs_file``
     - Absolute flux calibration (``EUC_SIR_W-AbsoluteFlux-*.fits``)
   * - ``rel_flux_file``
     - Relative flux loss (``EUC_SIR_W-RelativeFlux-*.fits``)
   * - ``detector_slots_path``
     - Detector layout (``EUC_SIR_DETMODEL_*.csv``)

When ``calibdir`` is not given, `gelsa` searches the standard DR1 calibration
directories on `ESA Datalabs <https://datalabs.esa.int>`_. On Datalabs,
``gelsa.Gelsa()`` works without any arguments. When `gelsa` starts, it prints
whether each calibration file was found.

Other options switch parts of the model on or off, for example ``use_psf``,
``use_relative_flux_loss`` or ``zero_order_catalog``. See
:class:`gelsa.Gelsa` for the full list.

Configuration file
------------------

You can save the settings to a JSON file and load them back later:

.. code:: python

    G.write_config("gelsa_config.json")
    G = gelsa.Gelsa(config_file="gelsa_config.json")

Keyword arguments override values from the file, which override the defaults:

.. code:: python

    G = gelsa.Gelsa(config_file="gelsa_config.json", use_psf=False)


Cutouts and stacking
====================

:meth:`SpecFrame.cutout() <gelsa.specframe.SpecFrame.cutout>` follows the
spectral trace of a source across the detectors and returns one
:class:`~gelsa.spec_crop.SpecCrop` per detector that the trace falls on. Each
crop holds the image, mask and variance of that detector region.

A source is usually observed in many exposures. :class:`~gelsa.analysis.Analysis`
collects a set of frames, cuts out the source in each one, and combines the
cutouts into:

* a stacked 2D spectrum resampled on a common wavelength grid, and the 1D
  spectrum extracted from it
  (:meth:`~gelsa.analysis.Analysis.build_stacked_2D_spectrum`),
* an emission line map on a sky grid
  (:meth:`~gelsa.analysis.Analysis.build_stacked_line_map`).

The next two pages show both steps, first on simulated data and then on
Euclid data.

Spectra of neighbouring sources overlap on the detector.
:class:`~gelsa.decontam.Decontamination` models the target and its neighbours
together and fits all their spectra jointly (see :doc:`decontamination`).

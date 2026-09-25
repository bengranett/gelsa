.. _decontamination:

*****************************
Decontaminated 1D spectra
*****************************

In slitless spectroscopy, the spectra of neighbouring sources overlap on the
detector. The flux measured along a target's trace therefore includes light
from other objects. :class:`~gelsa.decontam.Decontamination` removes it by
modelling the target and its neighbours together, and fitting the spectra of
all of them at once.

This walkthrough runs on `ESA Datalabs <https://datalabs.esa.int>`_ and needs
the ``esa`` extra (see :ref:`how_to_install`). A complete version is in the
notebook ``GELSA_1d_spectra_decontamination.ipynb``.


How it works
============

1. **Contaminant selection.** For each SIR exposure, sources in the MER
   catalogue are selected as possible contaminants if their spectra can
   overlap the target's. That means they lie within the length of a spectrum
   along the dispersion direction, and within
   ``contaminant_separation_arcsec`` (5″ by default) across it. A flux cut on
   the catalogue keeps only the neighbours bright enough to matter.
2. **Morphology from MER images.** Each source, target included, is modelled
   with its own light distribution: the MER image cut out with the
   segmentation map.
3. **Forward model.** The SIR instrument model disperses every source through
   every exposure. The spectrum of each source is unknown and is sampled on a
   wavelength grid with ``wavelength_step`` Å spacing.
4. **Joint fit.** The spectra of all sources are fitted together, by weighted
   linear least squares over the pixels of all the exposures. The fit is
   regularized by a penalty on the curvature of each spectrum, with strength
   λ, plus a small ridge term (``reg_floor``). Outlier pixels are rejected by
   sigma clipping, and the fit is then repeated.

The fit is run for every value of λ in ``reg_lambda_list``, and one result is
returned for each value, so that you can choose between them (see
`Choosing the regularization`_).


Set up
======

.. code:: python

    import numpy as np
    from matplotlib import pyplot as plt
    from astroquery.esa.euclid.core import EuclidClass

    import gelsa
    from gelsa import decontam, decontam_utils
    from gelsa.esa import euclid_archive

    Euclid = EuclidClass()
    Euclid.login(credentials_file="password")
    EA = euclid_archive.EuclidArchive(Euclid)

    G = gelsa.Gelsa()

Choose a target. The survey and patch select the MER products and SIR frames
to use:

.. code:: python

    ra, dec = 268.5026717144212, 63.7361903645274
    survey = "deep_visits"
    patch_id = 57

    # model neighbours brighter than H = 20.5 (AB), converted to µJy
    flux_cut_microjy = 1e6 * 10**(-0.4 * (20.5 - 8.9))


Add the target, catalogue and images
====================================

.. code:: python

    decon = decontam.Decontamination()
    decon.add_target(ra, dec)

    cat = EA.load_mer_catalog(ra, dec, survey=survey, patch_id=patch_id,
                              radius_deg=0.2)
    cat = cat[cat["FLUX_H_TEMPLFIT"] > flux_cut_microjy]
    decon.add_catalog(cat)

The MER tile images and segmentation maps provide the morphology of every
source. Load all the tiles the catalogue covers:

.. code:: python

    for tile in np.unique(list(cat["TILE_INDEX"])):
        image, segmap, wcs = EA.load_tile_image(tile, patch_id)
        decon.add_image(image, segmap, wcs, tile=tile)

By default, :meth:`~gelsa.esa.euclid_archive.EuclidArchive.load_tile_image`
loads the NIR J image.


Add the SIR exposures
=====================

.. code:: python

    paths = EA.query_sir_frames(ra, dec, patch_id=patch_id, radius_deg=0.5)
    decon.add_frame(*[G.load_frame(p) for p in paths])

Frames with a pointing ID that has already been added are skipped.


Fit
===

.. code:: python

    results = decon.fit(reg_lambda_list=[100], reg_floor=0.001,
                        sigma_clip=5, masking_iterations=1, nthreads=4)

Any parameter in ``decon.params`` can be passed to
:meth:`~gelsa.decontam.Decontamination.fit`. The main ones are:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Meaning
   * - ``reg_lambda_list``
     - [0.1 … 1000]
     - Regularization strengths λ to fit; one result each.
   * - ``reg_floor``
     - 1e-4
     - Ridge term added to the curvature penalty.
   * - ``wavelength_step``
     - 13
     - Wavelength sampling of the fitted spectra (Å).
   * - ``contaminant_separation_arcsec``
     - 5
     - Distance across the dispersion direction within which neighbours are
       modelled.
   * - ``sigma_clip``
     - 10
     - Threshold for rejecting outlier pixels (in σ).
   * - ``masking_iterations``
     - 1
     - Number of clip-and-refit rounds.
   * - ``nsamples``
     - 10000
     - Number of samples used to build the forward model of each source.
   * - ``compute_var``
     - False
     - Also compute the variance of the fitted spectra.
   * - ``nthreads``
     - 4
     - Number of parallel workers.


Results
=======

``results`` has one entry for each λ. Each entry is a pair
``(galaxies, fit_info)``:

* ``galaxies`` lists the fitted sources, targets first. For each source,
  ``gal.wavelength`` is the wavelength grid (Å) and ``gal.sed_params`` is the
  decontaminated spectrum (erg s⁻¹ cm⁻² Å⁻¹).
* ``fit_info`` is a dictionary with the data, model and residual vectors, χ²
  statistics, and the pixel masks used in the fit.

.. code:: python

    galaxies, fit_info = results[0]
    target = galaxies[0]

    plt.plot(target.wavelength / 10, target.sed_params * 1e16)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (10$^{-16}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")

To compare with the SIR pipeline spectrum of the same source:

.. code:: python

    sir = EA.load_sir_combined_spectrum(ra, dec, patch_id=patch_id, survey=survey)[0]
    data = sir["data"]
    plt.plot(data["WAVELENGTH"] / 10, data["SIGNAL"], c="grey", label="SIR")

Choosing the regularization
---------------------------

Weak regularization (small λ) gives noisy spectra; strong regularization
smooths out real features. When you fit several values, choose the one that
minimizes the generalized cross-validation score over the target's pixels,
``fit_info["gcv_targ"]``:

.. code:: python

    results = decon.fit(reg_lambda_list=[1, 10, 100, 1000])
    best = min(results, key=lambda r: r[1]["gcv_targ"])
    target = best[0][0]

``fit_info["gcv"]`` is the same score computed over all the fitted pixels. It
is mostly set by the contaminant and background pixels, so use it only as a
diagnostic.

Check the fit
-------------

:func:`~gelsa.decontam_utils.plot2d_target` shows the data, model and
residuals around the target in each exposure:

.. code:: python

    for galaxies, fit_info in results:
        decontam_utils.plot2d_target(fit_info, decon.specimager_list, vmax=200)

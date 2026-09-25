.. _simulation:

*********************
Simulate spectrograms
*********************

This example simulates a single emission line galaxy in one grism exposure,
then extracts its spectrum and builds an emission line map. You only need the
SIR calibration files (see :doc:`concepts`); no Euclid observations are used.


Set up a frame
==============

Create a :class:`gelsa.Gelsa` instance and an empty spectroscopic frame at a
chosen pointing:

.. code:: python

    import numpy as np
    from matplotlib import pyplot as plt

    import gelsa
    from gelsa import analysis, galaxy, visu, utils

    G = gelsa.Gelsa(calibdir="path/to/SIR/calibration")

    frame = G.new_spec_frame(ra=0, dec=0, pa=0, grism_name="RGS000")

Choose the source position by asking where pixel (1000, 1000) on detector 5
sees light at 1.5 µm:

.. code:: python

    ra, dec = frame.pixel_to_radec(1000, 1000, 5, 15000)


Define a galaxy
===============

A :class:`~gelsa.galaxy.Galaxy` has a position, a redshift, a light profile, a
continuum and a set of emission lines. Here it has a Gaussian profile and an Hα
line (with [NII]) at redshift 1.2:

.. code:: python

    redshift = 1.2

    gal = galaxy.Galaxy(
        ra=ra, dec=dec,
        redshift=redshift,
        fwhm_arcsec=0.5,
        axis_ratio=1,
    )
    gal.set_flux_Ha(2e-15, NII_Ha_ratio=0.2)   # erg/s/cm²

Render the galaxy into the frame. With ``noise=True``, detector noise is added:

.. code:: python

    frame.add_sources([gal], noise=True)


Look at the spectrogram
=======================

:meth:`~gelsa.specframe.SpecFrame.cutout` returns a dictionary of
:class:`~gelsa.spec_crop.SpecCrop` objects, one for each detector the trace
falls on. :func:`gelsa.visu.show` displays a crop:

.. code:: python

    crops = frame.cutout(ra, dec, redshift=redshift, padx=5, pady=25)
    crop = crops[5]
    visu.show(crop)


Extract the spectrum
====================

:class:`~gelsa.analysis.Analysis` works the same way on one frame as on many.
Add the frame, cut out the source, and resample it onto a wavelength grid:

.. code:: python

    A = analysis.Analysis(G)
    A.add_frame(frame)
    A.crop(ra, dec, redshift=redshift, padx=5, pady=25)

    spec = A.build_stacked_2D_spectrum(wave_range=(12000, 19000))

``spec`` is a dictionary. Its main entries are:

* ``wavelength``: the wavelength grid (Å),
* ``spec2d``, ``spec2d_var``: the 2D spectrum and its variance,
* ``spec1d``, ``spec1d_var``: the extracted 1D spectrum and its variance,
* ``pixel_perp``: the offset from the trace, in pixels.

.. code:: python

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    extent = (spec["wavelength"][0], spec["wavelength"][-1],
              spec["pixel_perp"][0], spec["pixel_perp"][-1])
    spec2d = np.ma.array(spec["spec2d"], mask=spec["spec2d_norm"] == 0)
    ax1.imshow(visu.normalize_image(spec2d), extent=extent, aspect="auto")

    ax2.plot(spec["wavelength"], spec["spec1d"])
    ax2.set_xlabel("Wavelength (Å)")

The Hα line appears at 6564.6 × (1 + z) ≈ 14442 Å.


Build an emission line map
==========================

A line map resamples the dispersed light at one wavelength back onto the sky.
Define the output grid with an :class:`astropy.wcs.WCS`, here 41 × 41 pixels
of 0.3″ centred on the source:

.. code:: python

    wcs = utils.make_wcs(ra, dec, pixel_size_arcsec=0.3, shape=(41, 41))

    line_wavelength = 6564.61 * (1 + redshift)
    line_map, line_var, norm, count = A.build_stacked_line_map(
        line_wavelength, wcs, width=50)

    plt.subplot(projection=wcs)
    plt.imshow(line_map, origin="lower")


Many exposures
==============

To simulate a dithered observation, create several frames with slightly
different pointings or grisms, add the same galaxy to each, and pass all of
them to the same ``Analysis`` before calling ``crop``:

.. code:: python

    A = analysis.Analysis(G)
    for grism in ("RGS000", "RGS180"):
        f = G.new_spec_frame(ra=0, dec=0, pa=0, grism_name=grism)
        f.add_sources([gal], noise=True)
        A.add_frame(f)

    A.crop(ra, dec, redshift=redshift, padx=5, pady=25)
    spec = A.build_stacked_2D_spectrum(wave_range=(12000, 19000))

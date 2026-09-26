.. _real_data:

*******************************
Euclid data on ESA Datalabs
*******************************

This walkthrough finds every SIR exposure of a galaxy, stacks its spectrum,
and builds an Hα line map. It runs on
`ESA Datalabs <https://datalabs.esa.int>`_, where the Euclid data products and
the SIR calibration files are mounted on the file system.

The example target is a galaxy at z = 0.751 in the COSMOS field. This
walkthrough needs the ``esa`` extra (see :ref:`how_to_install`).

.. code:: python

    import numpy as np
    from matplotlib import pyplot as plt

    import gelsa
    from gelsa import analysis, visu
    from gelsa.esa.euclid_archive import EuclidArchive

    ra, dec, redshift = 150.09382, 2.503955, 0.751


Connect to the archive
======================

The archive is queried with ``astroquery``.
:class:`~gelsa.esa.euclid_archive.EuclidArchive` wraps a logged-in connection
with queries for the products `gelsa` needs:

.. code:: python

    from astroquery.esa.euclid.core import EuclidClass

    Euclid = EuclidClass()
    Euclid.login(credentials_file="password")   # a file with your username and password
    archive = EuclidArchive(Euclid)

Query results are cached on disk in ``~/.cache/gelsa``, so running a query
again costs nothing. Set ``GELSA_CACHE_DIR`` to use a different location.

On Datalabs, the calibration files are found automatically:

.. code:: python

    G = gelsa.Gelsa()


Direct image and line map grid
==============================

Load a MER image stamp around the target. Its WCS also sets the grid of the
line map:

.. code:: python

    stamp, wcs = archive.load_stamp(ra, dec, filter="NIR_H", survey="cosmos",
                                    width=101, height=101)
    wcs.array_shape = stamp.shape

    plt.subplot(projection=wcs)
    plt.imshow(visu.normalize_image(stamp, levels=(1, 99.9)), cmap="binary_r")


Find the SIR exposures
======================

:meth:`~gelsa.esa.euclid_archive.EuclidArchive.query_sir_frames` returns the
paths of the SIR science frames near a position, one for each pointing. To keep
one grism only, filter the paths by name:

.. code:: python

    paths = archive.query_sir_frames(ra, dec, radius_deg=0.5)
    paths = [p for p in paths if "BGS" in p]    # blue grism only


Stack the spectrum
==================

Give the frame list to :class:`~gelsa.analysis.Analysis`, cut out the target
in each exposure, and stack:

.. code:: python

    A = analysis.Analysis(G)
    A.read_file_list(paths)
    A.crop(ra, dec, redshift=redshift, padx=50, pady=50)

    spec = A.build_stacked_2D_spectrum(wave_range=(9200, 13700))

    plt.plot(spec["wavelength"], spec["spec1d"])
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")

Exposures in which the target does not fall on a detector are skipped.

Continuum subtraction
---------------------

To isolate emission lines, subtract a median filtered continuum from each
cutout before stacking:

.. code:: python

    from gelsa import continuum_subtraction_base as csb

    cs = csb.MedianFilterCutout(median_filter_size_pix=50)
    A.crop(ra, dec, redshift=redshift, padx=50, pady=50,
           continuum_subtraction=cs)


Emission line map
=================

Resample the light at the observed Hα wavelength onto the stamp's WCS:

.. code:: python

    line_map, line_var, norm, count = A.build_stacked_line_map(
        6564.61 * (1 + redshift), wcs)

    plt.subplot(projection=wcs)
    plt.imshow(visu.normalize_image(line_map, levels=(1, 99)))


Interactive line map studio
===========================

In a Jupyter notebook on Datalabs,
:class:`~gelsa.esa.line_map_studio.LineMapStudio` runs this whole pipeline
from widgets: log in, choose a target, stack, adjust the colours, and export
publication-quality figures.

.. code:: python

    from gelsa.esa import line_map_studio

    studio = line_map_studio.make_line_map_studio()
    studio

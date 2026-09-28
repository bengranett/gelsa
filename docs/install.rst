.. _how_to_install:

************
Installation
************

Install with pip
================

`gelsa` needs Python 3.10 or later. Install the latest release from
`PyPI <https://pypi.org/project/gelsa/>`_:

.. code:: bash

    python -m pip install gelsa

The required dependencies (numpy, scipy, numba, astropy, …) are installed
automatically.

To get the development version, with changes not yet released, install from
GitHub instead:

.. code:: bash

    python -m pip install git+https://github.com/elsa-euclid/gelsa.git

The packages used by the :mod:`gelsa.esa` modules, which query the Euclid
archive and provide the interactive line map studio, are installed too:
astroquery, `azulero <https://github.com/kabasset/azulero>`_, ipywidgets and
IPython.

Install from source
===================

To modify the code, clone the repository and make an editable install. The
``dev`` extra adds the test and lint tools:

.. code:: bash

    git clone https://github.com/elsa-euclid/gelsa.git
    cd gelsa
    python -m pip install -e ".[dev]"
    pytest

Some tests need the SIR calibration files. They are skipped when those files
are not available.

To build this documentation locally:

.. code:: bash

    python -m pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html

Verify the installation
=======================

.. code:: python

    import gelsa
    print(gelsa.__version__)

Next, read :doc:`guide/concepts` to learn which calibration files `gelsa`
needs and where to find them.

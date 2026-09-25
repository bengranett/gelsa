.. _how_to_install:

************
Installation
************

Install with pip
================

Install the latest version directly from GitHub:

.. code:: bash

    python -m pip install git+https://github.com/elsa-euclid/gelsa.git

The required dependencies (numpy, scipy, numba, astropy, …) are installed
automatically.

Euclid archive and notebook tools
---------------------------------

The :mod:`gelsa.esa` modules query the Euclid archive and provide the
interactive line map studio. They need extra packages, installed with the
``esa`` extra. Version 2.1 of `azulero <https://github.com/kabasset/azulero>`_
is not yet on PyPI, so install it from GitHub first:

.. code:: bash

    python -m pip install git+https://github.com/kabasset/azulero.git@master
    python -m pip install "gelsa[esa] @ git+https://github.com/elsa-euclid/gelsa.git"

Install from source
===================

To modify the code, clone the repository and make an editable install. The
``dev`` extra adds the test and lint tools:

.. code:: bash

    git clone https://github.com/elsa-euclid/gelsa.git
    cd gelsa
    python -m pip install -e ".[dev]"      # or ".[dev,esa]"
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

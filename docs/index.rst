Welcome to Gelsa!
=================
*Advanced tools for Euclid grism spectroscopy*

*Gelsa* is a Python library to facilitate the analysis of 2D dispersed images from *Euclid*'s slitless spectroscopic observations. Developed as part of the `ELSA project <https://elsa-euclid.github.io>`_ (Euclid Legacy Science Advanced analysis tools), `gelsa` builds on the products of the Euclid Science Ground Segment and SIR processing function.

With `gelsa` you can:

* map between sky position, wavelength and detector pixel using the SIR instrument models,
* simulate NISP spectrograms of galaxies with emission lines,
* cut out, resample and stack 2D and 1D spectra from many exposures,
* build spatially resolved emission line maps,
* extract 1D spectra decontaminated from overlapping neighbours,
* query and read Euclid data products on `ESA Datalabs <https://datalabs.esa.int>`_.

The source code is on `GitHub <https://github.com/elsa-euclid/gelsa>`_.


.. toctree::
   :maxdepth: 1
   :caption: Get started

   install
   guide/concepts
   guide/simulation
   guide/real_data
   guide/decontamination

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api


************
Contributors
************

* **Lead developer**: Ben Granett

* **Contributors**: Louis Gabarra, Fabio Rigamonti, Francesca Passalacqua, Federico Lepri, Nadir El Ariny

***********
How to cite
***********

* Emission line maps: **Gabarra et al, in prep, 2025.**

Please include the acknowledgement for the ELSA project below in all publications.

****************
Acknowledgements
****************

.. grid:: 2

  .. grid-item::
    :columns: 2

    .. image:: _static/EN_fundedbyEU_VERTICAL_RGB_NEG.png
      :class: only-dark
      :width: 150
      :alt: Funded by the European Union

    .. image:: _static/EN_fundedbyEU_VERTICAL_RGB_POS.png
      :class: only-light
      :width: 150
      :alt: Funded by the European Union

  .. grid-item::
    :columns: 10

    "ELSA: Euclid Legacy Science Advanced analysis tools" (Grant Agreement no. 101135203) is funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or Innovate UK. Neither the European Union nor the granting authority can be held responsible for them. UK participation is funded through the UK Horizon guarantee scheme under Innovate UK grant 10093177.

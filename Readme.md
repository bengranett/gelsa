GELSA: Advanced tools for Euclid grism spectroscopy
===================================================

Overview
--------

*Gelsa* is a Python library to facilitate the analysis of 2D dispersed images from *Euclid*'s slitless spectroscopic observations. Developed as part of the [ELSA project](https://elsa-euclid.github.io) (Euclid Legacy Science Advanced analysis tools),
`gelsa` builds on the products of the Euclid Science Ground Segment and SIR processing function.


**Links**
* [Documentation](https://elsa-euclid.github.io/gelsa/)
* [Source code](https://github.com/elsa-euclid/gelsa)

**Key Features:**

*   **Data Handling:** Read *Euclid* SGS data products.
*   **Spectra location:** Map between sky (RA, Dec), wavelength, and detector pixel coordinates using SIR instrument models.
*   **Simulation:** Generate NISP spectrograms based on source properties and instrument models.
*   **Spectral Extraction:** Extract 1D and 2D spectra.
*   **Emission line maps:** Remap spectra to make spatial emission line maps.
*   **Visualization:** Routines for displaying spectrograms and associated data.


Contributors
------------

Ben Granett (benjamin.granett@inaf.it), Louis Gabarra, Fabio Rigamonti, Francesca Passalacqua, Federico Lepri, Nadir El Ariny



Install
-------

```
python -m pip install git+https://github.com/elsa-euclid/gelsa.git
```

The Euclid archive tools and notebook GUIs in `gelsa.esa` need the `esa` extra. It depends on [azulero](https://github.com/kabasset/azulero) 2.1, which is not yet on PyPI:
```
python -m pip install git+https://github.com/kabasset/azulero.git@master
python -m pip install "gelsa[esa] @ git+https://github.com/elsa-euclid/gelsa.git"
```

For development, clone the repository and install in editable mode:
```
python -m pip install -e ".[dev]"
```

Usage
-----

On [ESA Datalabs](https://datalabs.esa.int) the SIR calibration files are found automatically; elsewhere, pass `calibdir`.
The following code loads a SIR science frame, cuts out the spectrum of a source, and displays it.

```python
import gelsa
from gelsa import visu

G = gelsa.Gelsa()

frame = G.load_spec_frame("EUC_SIR_W-SCIFRM_....fits")

ra, dec, z = 150.09382, 2.503955, 0.751
crops = frame.cutout(ra, dec, z)     # one SpecCrop per detector the trace falls on

for det, crop in crops.items():
    visu.show(crop, levels=(1, 99))
```

See the [documentation](https://elsa-euclid.github.io/gelsa/) for simulations, stacking many exposures and emission line maps.

License
-------

`gelsa` is released under the MIT License (see `LICENSE.txt`).

Third-party code
----------------

`gelsa/_vendor/fast_interp.py` is a lightly modified copy of [fast_interp](https://github.com/dbstein/fast_interp) by David Stein, distributed under the Apache License 2.0 (see `gelsa/_vendor/LICENSE.fast_interp`).

Acknowledgments
---------------

![EU funding](docs/_static/EN_FundedbytheEU_RGB_POS.png "EU funding"){width=150}

“ELSA: Euclid Legacy Science Advanced analysis tools” (Grant Agreement no. 101135203) is funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or Innovate UK. Neither the European Union nor the granting authority can be held responsible for them. UK participation is funded through the UK Horizon guarantee scheme under Innovate UK grant 10093177.

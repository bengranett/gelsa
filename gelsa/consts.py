import numpy as np

c = 2.99792458e18       #: speed of light (angstrom/sec)
planck = 6.6260693E-27  #: Planck constant (erg sec)
fwhm_to_sigma = 1/(2*np.sqrt(2*np.log(2)))

line_list = {
    'Hb': 4862.68,
    'O3_4960': 4960.295,
    'O3_5008': 5008.240,
    'N2_6549': 6549.86,
    'Ha': 6564.61,
    'N2_6585': 6585.27,
    'S2_6718': 6718.29,
    'S2_6732': 6732.67,
}

# this list sets the order
line_names = ('Ha', 'N2_6549', 'N2_6585', 'S2_6718', 'S2_6732',
              'Hb', 'O3_4960', 'O3_5008')

# set up array of line wavelengths
lines = []
for name in line_names:
    lines.append(line_list[name])

line_indices = {}
for i, name in enumerate(line_names):
    line_indices[name] = i

# rest-frame wavelengths (angstrom) masked before continuum subtraction:
# H-alpha, Balmer series, Paschen series, [OII], [SII], [SIII]
extended_wavelength_list = (
    6564.61,
    4862.69,
    4341.69,
    4102.92,
    3971.19,
    3890.15,
    3836.48,
    3798.98,
    3771.70,
    3751.22,
    12821.59,
    10941.09,
    10052.13,
    9548.59,
    9231.55,
    9017.39,
    3728.49,
    6732.71,
    6718.32,
    9533.20,
    9071.10,
)

import logging
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import healpy
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class SkyBin:

    def __init__(self, nside=4096):
        """ """
        self.nside = nside
        self._counts = defaultdict(float)
        self._norm = defaultdict(float)
        self._center = np.zeros(3, dtype='d')
        self._center_count = 0

    def bin(self, ra, dec, weight):
        """ """
        vec = healpy.ang2vec(ra, dec, lonlat=True)
        self._center += np.sum(vec, axis=0)
        self._center_count += len(ra)

        pix = healpy.ang2pix(self.nside, ra, dec, lonlat=True, nest=False)


        count = np.bincount(pix, weights=weight)
        norm = np.bincount(pix)

        nonzero = np.where((norm > 0) & (count > 0))[0]

        for ind in nonzero:
            self._counts[ind] += count[ind]
            self._norm[ind] += norm[ind]

    def plot(self, log=False, reso=2, xsize=1000, ysize=1000, **plot_params):
        """ """
        if self._center_count == 0:
            raise ValueError("nothing to plot")

        center = self._center / self._center_count
        center /= np.sum(center**2)

        center_ra_dec = healpy.vec2ang(center, lonlat=True)
        center_ra_dec = np.round(center_ra_dec, 2)

        skymap = np.zeros(12*self.nside**2, dtype='d')
        skymap += healpy.UNSEEN

        for ind in self._norm.keys():
            if self._counts[ind] > 0:
                skymap[ind] = self._counts[ind] / self._norm[ind]

        logger.debug(f"skymap total {np.sum(skymap[skymap>0])}")

        if log:
            valid = skymap > 0
            skymap[valid] = np.log10(skymap[valid])

        logger.debug(f"skymap percentiles (1, 10, 50, 90, 99): {np.percentile(skymap, (1, 10, 50, 90, 99))}")
        healpy.gnomview(skymap, rot=center_ra_dec, coord='C', nest=False, reso=reso,
                        xsize=xsize, ysize=ysize, badcolor='None', bgcolor='None', **plot_params)
        healpy.graticule(dmer=1, dpar=1, local=False)


def plot_footprint(path, **plot_params):
    tree = ET.parse(path)
    root = tree.getroot()
    logger.debug(f"footprint root {root}")
    footprintlist = root.find('Data/VmspDetectorFootprintList')
    logger.debug(f"footprint list {footprintlist}")
    for footprint in footprintlist.findall('VmspDetectorFootprint'):
        poly_list = footprint.find('VmspPolygonList')
        for poly in poly_list.findall('Polygon'):
            xx = []
            yy = []
            for vert in poly.findall('Vertex'):
                xx.append(float(vert.find('C1').text))
                yy.append(float(vert.find('C2').text))
            xx.append(xx[0])
            yy.append(yy[0])
            healpy.projplot(xx, yy, lonlat=True, **plot_params)


def load_catalog(xml_path, workdir='.'):
    """ """
    logger.info(f"loading {xml_path}")
    tree = ET.parse(os.path.join(xml_path))
    root = tree.getroot()
    try:
        datafile = root.find('Data/Catalog/DataContainer/FileName').text
    except AttributeError:
        datafile = root.find('Data/DataStorage/DataContainer/FileName').text
    cat = Table.read(os.path.join(workdir, 'data', datafile))
    return cat

def load_catalog_list(path_list, workdir='.'):
    cats = []
    for path in path_list:
        cats.append(load_catalog(path, workdir))
    return vstack(cats)

def plot_catalogs(path_list, workdir='.'):
    """ """
    SB = SkyBin()

    for path in path_list:
        cat = load_catalog(path, workdir)
        SB.bin(cat['RIGHT_ASCENSION'], cat['DECLINATION'], cat['N_DITH'])

    SB.plot(reso=0.1)

def load_scienceframe(xml_path, workdir='.'):
    """ """
    logger.info(f"loading {xml_path}")
    tree = ET.parse(os.path.join(xml_path))
    root = tree.getroot()
    datafile = root.find('Data/DataStorage/DataContainer/FileName').text

    hdul = fits.open(os.path.join(workdir, 'data', datafile))
    return hdul


def show_image(im, mask=None, levels=(1,90), **plot_params):
    """ """
    valid = np.isfinite(im)
    if mask is not None:
        valid &= mask == 0
    low, high = np.percentile(im[valid], levels)
    logger.debug(f"min {low} max {high}")
    im_ = im.copy()
    im_[~valid] = 0
    plt.imshow(im_, vmin=low, vmax=high, origin='lower', **plot_params)
    return low, high

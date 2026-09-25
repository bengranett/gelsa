"""Queries against the Euclid science archive (TAP).

Wrap a logged-in connection in :class:`EuclidArchive` and ask it for products:

    from astroquery.esa.euclid.core import EuclidClass
    from euclid_archive import EuclidArchive

    Euclid = EuclidClass(environment='IDR')
    Euclid.login(credentials_file='password')
    archive = EuclidArchive(Euclid)

    frames = archive.query_sir_frames(ra, dec)
    rows = archive.query_mosaic(tile_index)

Every query goes through :func:`_cached_launch_job`, so repeating one costs
nothing and results survive a kernel restart. The query text is the cache key:
keep it byte for byte if an existing cache entry is to be reused.

The ``dr1.*`` tables are readable anonymously; the ``catalogue.*`` ones - the
combined spectra and the MER catalogues - need a login.
"""
import hashlib
import os
import pickle
import glob
import xml.etree.ElementTree as ET
import numpy as np
import astropy
from astropy import units
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.table import Table, vstack

import warnings


# Query results are kept per user rather than beside the module, which may sit
# in a read-only site-packages and must not collect pickles in a git checkout.
# GELSA_CACHE_DIR overrides the location.
CACHE_DIR = os.environ.get('GELSA_CACHE_DIR') or os.path.join(
    os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'),
    'gelsa', 'query_cache')

# In-memory layer in front of the on-disk cache, keyed by query text.
_query_cache = {}


def _cache_path(query):
    digest = hashlib.sha256(query.encode('utf-8')).hexdigest()[:32]
    return os.path.join(CACHE_DIR, f'{digest}.pkl')


def _read_cache_file(path):
    """Return a cached result, or None if the file is missing or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)['result']
    except Exception as e:
        print(f"Warning: ignoring unreadable cache file {path}: {e}")
        return None


def _write_cache_file(path, query, result):
    """Write a result to the disk cache; a failure here must not break the query."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        # Write to a temp file first so an interrupted write can't leave a half file
        # that later reads would choke on.
        tmp = f'{path}.tmp{os.getpid()}'
        with open(tmp, 'wb') as f:
            pickle.dump({'query': query, 'result': result}, f)
        os.replace(tmp, path)
    except Exception as e:
        print(f"Warning: could not write cache file {path}: {e}")


def _cached_launch_job(euclid_class, query):
    """Run a TAP query, reusing a cached result if this exact query already ran.

    Results are cached in memory and under ``CACHE_DIR``, so they survive a module
    reload or a restarted kernel. Use :func:`clear_query_cache` to force fresh queries.
    """
    if query in _query_cache:
        return _query_cache[query]

    result = _read_cache_file(_cache_path(query))
    if result is None:
        result = euclid_class.launch_job(query).get_results()
        _write_cache_file(_cache_path(query), query, result)

    _query_cache[query] = result
    return result


def clear_query_cache(disk=True):
    """Drop cached query results, forcing the next queries to hit the archive again.

    Also removes the on-disk cache unless ``disk=False``.
    """
    _query_cache.clear()
    if not disk:
        return
    try:
        names = os.listdir(CACHE_DIR)
    except FileNotFoundError:
        return
    for name in names:
        if name.endswith('.pkl'):
            os.remove(os.path.join(CACHE_DIR, name))


def _survey_tables(survey):
    """The spectra and MER catalogue tables of the deep or the wide survey."""
    tables_mer = {
        'deep_visits': 'catalogue.mer_catalogue_deep_visits',
        'deep': 'catalogue.mer_catalogue_deep_survey',
        'wide': 'catalogue.mer_catalogue_wide_survey',
        'cosmos': 'catalogue.mer_catalogue_deep_mode',
    }
    tables_sir = {
        'deep_visits': 'catalogue.spectra_source_deep_visits',
        'deep': 'catalogue.spectra_source_deep',
        'wide': 'catalogue.spectra_source_wide',
        'cosmos': 'catalogue.spectra_source_deep',
    }
    survey = survey.lower()
    if 'visits' in survey:
        survey = 'deep_visits'
    elif 'wide' in survey:
        survey = 'wide'
    elif 'cosmos' in survey:
        survey = 'cosmos'
    else:
        survey = 'deep'
    
    return tables_sir[survey], tables_mer[survey]


class EuclidArchive:
    """Product queries against one archive connection."""

    def __init__(self, connection):
        """ """
        self._connection = connection

    @property
    def connection(self):
        """The astroquery ``EuclidClass`` the queries run on."""
        return self._connection

    def query(self, adql):
        """Run an ADQL query through the cache."""
        return _cached_launch_job(self._connection, adql)

    # --- MER products ----------------------------------------------------

    def query_mer_mosaic(self, tile_index, filter=None, patch_id=None):
        """ """
        query = f"""SELECT CONCAT(mer.datalabs_path,'/',mer.file_name) AS full_path, mer.filter_name
                FROM dr1.mosaic_product AS mer
                WHERE mer.tile_index={tile_index}"""
        if patch_id is not None:
            query += f" and mer.patch_id_list='{patch_id}'"
        if filter is not None:
            query += f" and mer.filter_name='{filter}'"
        return self.query(query)

    def query_mer_segmap(self, tile_index, patch_id=None):
        """ """
        query = f"""SELECT CONCAT(mer.datalabs_path,'/',mer.file_name) AS full_path
                FROM dr1.mer_segmentation_map AS mer
                WHERE mer.tile_index={tile_index}"""
        if patch_id is not None:
            query += f" and mer.patch_id_list='{patch_id}'"
        return self.query(query)

    def query_mer_tile(self, ra, dec, radius_deg=0.3, survey='Deep'):
        """ """
        _, mer_table = _survey_tables(survey)

        query = f"""SELECT DISTINCT tile_index
                FROM {mer_table} 
                WHERE
                DISTANCE({ra}, {dec}, right_ascension, declination)<{radius_deg}
                """
        results = self.query(query)
        tile_list = list(results['tile_index'])
        return tile_list

    
    def query_mer_catalog_by_tile(self, tile_index, patch_id=None):
        """Query path to MER final catalogue tile.
        """
        query = f"""SELECT datalabs_path, CONCAT(file_name_list) as file_name_list
                FROM dr1.mer_final_catalogue
                WHERE tile_index_list='{{{tile_index}}}'"""
        if patch_id is not None:
            query += f" and patch_id_list='{{{patch_id}}}'"
        results = self.query(query)
        row = results[0]
        path = row['datalabs_path']
        file_list = row['file_name_list'][1:-1].split(",")
        for file_name in file_list:
            if file_name.startswith("EUC_MER_FINAL-CAT"):
                return os.path.join(path, file_name)
        return None
    
    def query_mer_catalog_by_radec(self, ra, dec, patch_id=None, survey='deep', radius_deg=0.3):
        """Query path to MER final catalogue tile.
        """
        # first determine the tile indices
        tile_list = np.unique(list(self.query_mer_tile(ra, dec, survey=survey, radius_deg=radius_deg)))
        # query catalog paths
        path_list = []
        for tile in tile_list:
            path = self.query_mer_catalog_by_tile(tile, patch_id)
            path_list.append(path)

        return path_list

    def load_mer_catalog(self, ra, dec, patch_id=None, survey='deep', radius_deg=0.3):
        """ """
        path_list = self.query_mer_catalog_by_radec(ra, dec, patch_id=patch_id, survey=survey, radius_deg=radius_deg)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', units.UnitsWarning)
            warnings.simplefilter('ignore', astropy.utils.metadata.MergeConflictWarning)
            cat_list = []
            for path in path_list:
                cat_list.append(Table.read(path))
            cat = vstack(cat_list)
        return cat
    
    def load_fits_image(self, path):
        """ """
        with fits.open(path) as hdul:
            image = hdul[0].data[:]
            wcs = WCS(hdul[0].header)
        return image, wcs

    def load_tile_image(self, tile, patch_id=None, filter='NIR_J'):
        """Load an image by tile index, filter and optional patch ID.
        
        Returns image, segmap, wcs
        """
        results = self.query_mer_mosaic(tile, patch_id=patch_id, filter=filter)
        results_seg = self.query_mer_segmap(tile, patch_id=patch_id)

        image = None
        wcs = None
        for row in results:
            if row['filter_name'] == filter:
                image, wcs = self.load_fits_image(row['full_path'])

        if image is None:
            raise Exception("image not found")

        segmap, _ = self.load_fits_image(results_seg[0]['full_path'])
        return image, segmap, wcs
    
    def load_stamp(self, ra, dec, width=51, height=51, filter='NIR_H', radius_deg=10./3600, survey='DEEP', patch_id=None, hdu=0):
        """ """
        tile_index = self.query_mer_tile(ra, dec, radius_deg=radius_deg, survey=survey)
        tile_index = tile_index[0]
        results = self.query_mer_mosaic(tile_index, filter=filter, patch_id=patch_id)
        filter_select = results['filter_name'] == filter
        if np.sum(filter_select) == 0:
            print(f"filter {filter} not found. options: {np.unique(results['filter_name'])}")
        results = results[filter_select]
        path = results['full_path'][0]
        
        with fits.open(path) as hdul:
            image = hdul[hdu].data[:]
            header = hdul[hdu].header
            wcs = WCS(header)

        if width <= 0 or height <= 0:
            return image, wcs
            
        sky = SkyCoord(ra, dec, unit=units.deg)
        x, y = wcs.world_to_pixel(sky)

        x = int(np.round(x))
        y = int(np.round(y))
        x0 = x - width//2
        y0 = y - height//2
        x1 = x0 + width
        y1 = y0 + height
        x0_ = 0
        y0_ = 0
        x1_ = width
        y1_ = height
        if x0 < 0:
            x0_ = -x0
            x0 = 0
        if y0 < 0:
            y0_ = -y0
            y0 = 0
        if x1 > image.shape[1]-1:
            x1_ = image.shape[1] - 1 - x0
            x1 = image.shape[1] - 1
        if y1 > image.shape[0]-1:
            y1_ = image.shape[0] - 1 - y0
            y1 = image.shape[0] - 1
        image_stamp = np.zeros((height, width), dtype=image.dtype)

        image_stamp[y0_:y1_, x0_:x1_] = image[y0:y1, x0:x1]

        wcs = wcs[y0:y1, x0:x1]
        return image_stamp, wcs
    
    # --- SIR science frames ----------------------------------------------

    def _lookup_patch_id(self, directory, patch_id, product_id):
        """ """
        filename = f"{product_id}.xml"
        path = os.path.join(directory, filename)
        root = ET.parse(path).getroot()
        this_patch_id = int(root.find('Data/ObservationSequence/PatchId').text)
        if this_patch_id == patch_id:
            data_filename = root.find('Data/DataStorage/DataContainer/FileName').text
            return data_filename
        return None
                
    def _read_pointing_id(self, directory, product_id):
        """ """
        filename = f"{product_id}.xml"
        path = os.path.join(directory, filename)
        root = ET.parse(path).getroot()
        pointing_id = int(root.find('Data/ObservationSequence/PointingId').text)
        return pointing_id
    
    def query_sir_frames(self, ra, dec, radius_deg=20, patch_id=None, unique_pointing_id=True, table="dr1.sir_science_frame"):
        """ """
        query = f"""SELECT datalabs_path, file_name, product_id
                    FROM {table}
                    WHERE DISTANCE({ra}, {dec}, ra, dec)<{radius_deg}"""
        rows = self.query(query)
        ptgid_cache = {}
        frame_list = []
        for row in rows:
            directory = row['datalabs_path']
            product_id = row['product_id']
            if patch_id is not None:
                filename = self._lookup_patch_id(directory, patch_id, product_id)
                if filename is None:
                    continue
            else:
                filename = row["file_name"]
            if unique_pointing_id:
                ptgid = self._read_pointing_id(directory, product_id)
                if ptgid in ptgid_cache:
                    continue
                ptgid_cache[ptgid] = 1
            path = os.path.join(row['datalabs_path'], filename)
            if path.endswith(".gz"):
                path = path[:-3]
            frame_list.append(path)
        return frame_list
    

    # --- SIR combined spectra --------------------------------------------

    def query_sir_combined_spectra_by_object_id(self, object_id, patch_id=None, survey="deep"):
        """ """
        spe_table, mer_table = _survey_tables(survey)
        query = f"""SELECT CONCAT(spe.datalabs_path,'/',spe.file_name) AS full_path, spe.hdu_index, spe.source_id, spe.ra_obj, spe.dec_obj, mer.tile_index, mer.patch_id_list
                FROM {spe_table} AS spe
                JOIN {mer_table} AS mer
                ON mer.object_id = spe.source_id
                WHERE spe.source_id={object_id}"""
        if patch_id is not None:
            query += f" and mer.patch_id_list='{{{patch_id}}}'"
        return self.query(query)

    def query_sir_combined_spectra_by_radec(self, ra, dec, patch_id=None, survey='deep', radius_arcsec=0.3):
        """The radius is in arcseconds here, not degrees."""
        spe_table, mer_table = _survey_tables(survey)

        radius_deg = radius_arcsec / 3600

        query = f"""SELECT CONCAT(spe.datalabs_path,'/',spe.file_name) AS full_path, spe.hdu_index, spe.source_id, spe.ra_obj, spe.dec_obj, mer.tile_index, mer.patch_id_list
                FROM {spe_table} AS spe
                JOIN {mer_table}  AS mer
                ON mer.object_id = spe.source_id
                WHERE
                DISTANCE({ra}, {dec}, spe.ra_obj, spe.dec_obj)<{radius_deg}
                """
        if patch_id is not None:
            query += f" and mer.patch_id_list='{{{patch_id}}}'"
        return self.query(query)

    def query_sir_combined_spectra(self, ra=None, dec=None, object_id=None,
                                   survey='deep', radius_arcsec=0.3):
        """ """
        if object_id is not None:
            return self.query_sir_combined_spectra_by_object_id(object_id, survey=survey)
        elif ra is not None:
            return self.query_sir_combined_spectra_by_radec(ra, dec, survey=survey,
                                                            radius_arcsec=radius_arcsec)
        raise ValueError




    def load_sir_combined_spectrum(self, ra=None, dec=None, object_id=None, radius_arcsec=0.3, survey='deep', patch_id=None):
        """ """
        if object_id is not None:
            results = self.query_sir_combined_spectra_by_object_id(object_id, patch_id=patch_id, survey=survey)
        elif ra is not None:
            results = self.query_sir_combined_spectra_by_radec(ra, dec, patch_id=patch_id, survey=survey,
                                                               radius_arcsec=radius_arcsec)
        pack_list = []
        for row in results:
            spec_pack = {}
            path = row['full_path']
            hdu_index = row['hdu_index']
            with fits.open(path) as hdul:
                hdu = hdul[hdu_index]
                header = hdul[hdu_index-1].header
                data = hdu.data[:]
                spec_pack['data'] = data
                info = {
                    'object_id': header['OBJ_ID'],
                    'ndith': header['N_DITH'],
                    'ra': header['RA_OBJ'],
                    'dec': header['DEC_OBJ']
                }
                spec_pack['header'] = info
            pack_list.append(spec_pack)
        return pack_list
    
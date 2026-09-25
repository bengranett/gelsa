import os
import glob
import xml.etree.ElementTree as ET
from astropy.io import fits


def load_scienceframe(xml_path, workdir='.'):
    """ """
    if xml_path.endswith(".fits") or xml_path.endswith(".fits.gz"):
        hdul = fits.open(xml_path, decompress_in_memory=True)
        return hdul, None

    print(f"loading {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    datafile = root.find('Data/DataStorage/DataContainer/FileName').text

    workdir = os.path.dirname(xml_path)
    print(f"{workdir=}")
    data_path = os.path.join(workdir, 'data', datafile)
    print(f"loading {data_path}")
    hdul = fits.open(data_path, decompress_in_memory=True)
    return hdul, root


def find_location_table(science_frame_path):
    """Find the path to the location table from the science frame path"""
    with fits.open(science_frame_path) as hdul:
        pointing_id = hdul[0].header['PTGID']
    basedir = os.path.split(science_frame_path)
    pathglob = os.path.join(basedir[0], "../../DpdSirLocationTable/*.xml")
    datapath = os.path.join(basedir[0], "../../DpdSirLocationTable/data")
    path_list = glob.glob(pathglob)
    filename = None
    for path in path_list:
        root = ET.parse(path).getroot()
        this_ptgid = root.find("Data/Observation/PointingId").text
        if this_ptgid == str(pointing_id):
            filename = root.find("Data/CollectionFile/DataContainer/FileName").text
            break
    if filename is None:
        raise ValueError(f"Location table not found {pointing_id=}")
    return os.path.join(datapath, filename)

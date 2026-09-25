import sys
import numpy as np
import xml.etree.ElementTree as ET


def string_to_array(s, type=float):
    """ """
    return np.array([type(v) for v in s.split()])


class DisplacementModel:
    grisms = ('RGS000', 'RGS180', 'BGS000')
    def __init__(self, path):
        """ """
        self._load(path)

    def _load(self, path):
        """ """
        print(f"loading {path}")
        root = ET.parse(path).getroot()
        self.models = []
        for grism_name in self.grisms:
            for grism_elem in root.findall(f'Data/{grism_name}'):
                tilt = int(grism_elem.find('GWATilt').text)
                extra_tilt = float(grism_elem.find('ExtraTilt').text)
                # print(f"loading {grism_name=} {tilt=} {extra_tilt=}")

                pack_orders = {}
                for specorder_elem in grism_elem.findall('SpectralOrder'):
                    pack = self._read_specorder(specorder_elem)
                    pack_orders[pack['order']] = pack


                grism_pack = {
                    'Grism': grism_name,
                    'GWATilt': tilt,
                    'ExtraTilt': extra_tilt,
                    'Orders': pack_orders
                }
                self.models.append(grism_pack)

    def _read_specorder(self, element):
        """ """
        order = int(element.find("Order").text)
        (f"reading order {order}")
        local_model_deg = int(element.find("LocalModelDeg").text)
        local_ranges = string_to_array(element.find("LocalRanges").text)
        global_ranges1 = string_to_array(element.find("GlobalRanges/DoubleRow1").text)
        global_ranges2 = string_to_array(element.find("GlobalRanges/DoubleRow2").text)
        model = self._read_model(element)
        return {
            'order': order,
            'local_model_deg': local_model_deg,
            'local_ranges': local_ranges,
            'global_ranges': np.array([global_ranges1, global_ranges2]),
            'model': model
        }

    def _read_model(self, element):
        """ """
        pack = {}
        for model_elem in element.findall("Model"):
            degree = int(model_elem.find("Degree").text)
            matrix_shape = string_to_array(model_elem.find("Matrix/Shape").text, int)
            matrix = string_to_array(model_elem.find("Matrix/Data").text)
            matrix = np.reshape(matrix, matrix_shape)
            pack[degree] = matrix
        return pack

    def get_model(self, grism_name='BGS000', tilt=0):
        """ """
        for pack in self.models:
            if (pack['Grism'] == grism_name) and (pack['GWATilt'] == tilt):
                # print(f"found displacement model for {pack['Grism']} {pack['GWATilt']}", file=sys.stderr)
                return pack
        raise ValueError(f"No displacement model found for {grism_name=} {tilt=}")



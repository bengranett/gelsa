import sys
import numpy as np
import xml.etree.ElementTree as ET


def string_to_array(s, type=float):
    """ """
    return np.array([type(v) for v in s.split()])


class OpticalModel:
    grisms = ('RGS000', 'RGS180', 'BGS000')
    def __init__(self, path):
        """ """
        self._load(path)

    def _load(self, path):
        """ """
        # print(f"loading {path}")
        root = ET.parse(path).getroot()
        self.models = []
        for grism_name in self.grisms:
            for grism_elem in root.findall(f'Data/{grism_name}'):
                tilt = float(grism_elem.find('GWATilt').text)
                extra_tilt = float(grism_elem.find('ExtraTilt').text)
                # print(f"loading {grism_name=} {tilt=} {extra_tilt=}")

                reference = self._read_model(grism_elem.find('Reference'))
                pivot = self._read_model(grism_elem.find('Pivot'))
                displacements = self._read_model(grism_elem.find('Displacements'))

                pack = {
                    'Grism': grism_name,
                    'GWATilt': tilt,
                    'ExtraTilt': extra_tilt,
                    'Reference': reference,
                    'Pivot': pivot,
                    'Displacements': displacements
                }
                self.models.append(pack)

    def _read_model(self, element):
        """ """
        try:
            order = int(element.find('Order').text)
        except AttributeError:
            order = None
        try:
            ref_lambda = float(element.find('ReferenceLambda').text)
        except AttributeError:
            ref_lambda = None
        x = string_to_array(element.find('OptXModel/Data').text)
        y = string_to_array(element.find('OptYModel/Data').text)
        x_shape = string_to_array(element.find('OptXModel/Shape').text, type=int)
        y_shape = string_to_array(element.find('OptXModel/Shape').text, type=int)
        x = x.reshape(x_shape)
        y = y.reshape(y_shape)
        pack = dict(
            order=order,
            reference_lambda=ref_lambda,
            x=x,
            y=y,
        )
        return pack

    def get_model(self, grism_name='BGS000', tilt=0):
        """ """
        for pack in self.models:
            if (pack['Grism'] == grism_name) and (pack['GWATilt'] == tilt):
                # print(f"found optical model for {pack['Grism']} {pack['GWATilt']}", file=sys.stderr)
                return pack
        raise ValueError(f"No optical model found for {grism_name=} {tilt=}")


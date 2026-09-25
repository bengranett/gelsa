import numpy as np
from astropy.table import Table


class ZeroOrderMask:
    width_x = 31
    width_y = 13
    offx=0
    offy=0

    def __init__(self, path):
        """Zero order mask"""
        self.load_catalog(path)

    def load_catalog(self, path):
        """Load catalog"""
        self._cat = Table.read(path)
        # sel = self._cat['H'] < 19
        # self._cat = self._cat[sel]

    def create_zero_order_mask(self, frame):
        """ """
        x, y, det = frame.radec_to_pixel(
            self._cat['RIGHT_ASCENSION'],
            self._cat['DECLINATION'],
            15000*np.ones(len(self._cat)),
            dispersion_order=0
        )
        masks = {}
        shape = frame.detector_model.params['ny_pixels'], frame.detector_model.params['nx_pixels']
        for d in np.unique(det):
            if d < 0:
                continue
            if d not in masks:
                masks[d] = np.ones(shape, dtype=bool)
            sel = det == d
            xlow, xhigh = x[sel]-self.width_x//2+self.offx, x[sel]+self.width_x//2+self.offx
            ylow, yhigh = y[sel]-self.width_y//2+self.offy, y[sel]+self.width_y//2+self.offy

            xlow = xlow.astype(int)
            xhigh = xhigh.astype(int)
            ylow = ylow.astype(int)
            yhigh = yhigh.astype(int)

            for i in range(len(xlow)):
                masks[d][ylow[i]:yhigh[i], xlow[i]:xhigh[i]] = 0

        return masks

    def plot_zero_order_mask(self, frame, detector_index):
        """ """
        x, y, det = frame.radec_to_pixel(
            self._cat['RIGHT_ASCENSION'],
            self._cat['DECLINATION'],
            15000*np.ones(len(self._cat)),
            dispersion_order=0
        )
        sel = det == detector_index
        xlow, xhigh = x[sel]-self.width_x//2+self.offx, x[sel]+self.width_x//2+self.offx
        ylow, yhigh = y[sel]-self.width_y//2+self.offy, y[sel]+self.width_y//2+self.offy

        xlow = xlow.astype(int)
        xhigh = xhigh.astype(int)
        ylow = ylow.astype(int)
        yhigh = yhigh.astype(int)

        from matplotlib import pyplot as plt
        for i in range(len(xlow)):
            plt.plot([xlow[i], xhigh[i], xhigh[i], xlow[i], xlow[i]],
                     [ylow[i], ylow[i], yhigh[i], yhigh[i], ylow[i]], c='r')



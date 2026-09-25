"""Compact live editor for the azulero IYJH colour image of the MER stamps.

The boxes are those of the *Continuum colour* tab of
:mod:`gelsa.esa.line_map_studio`, with the same defaults, laid out beside
the image so the whole thing fits in a notebook cell::

    from gelsa.esa import azulero_gui
    colour = azulero_gui.make_azulero_gui(stamps, wcs)
    colour

Every change redraws the image. ``colour.rgb()`` is the current RGB image and
``colour.transform()`` the :class:`azulero.image.color.Transform` behind it, so
a setting found here can be handed on, e.g. to ``LineMapStudio.apply_transform``.
"""
import io
import time

import numpy as np

from matplotlib.figure import Figure

import ipywidgets as widgets
from IPython.display import display

from azulero.image import color

from .line_map_studio import (
    DEFAULT_TRANSFORM, compute_continuum_rgb, number)

# Narrower than the studio's boxes, so three columns sit beside the image.
BOX = dict(width='190px', label_width='100px')


class AzuleroGUI:
    """Number boxes for the azulero transform beside a live preview.

    Parameters
    ----------
    stamps : numpy.ndarray
        ``(4, ny, nx)`` stack of VIS, NIR_Y, NIR_J, NIR_H stamps.
    wcs : astropy.wcs.WCS
        WCS of the stamps, for the sky axes and for azulero.
    transform : azulero.image.color.Transform, optional
        Where to start; the studio's defaults if not given.
    image_px : int
        Width of the preview in screen pixels.
    """

    def __init__(self, stamps, wcs, transform=None, image_px=340):
        self.stamps = np.asarray(stamps)
        if self.stamps.ndim != 3 or self.stamps.shape[0] != 4:
            raise ValueError(
                f"stamps must have shape (4, ny, nx), got {self.stamps.shape}")
        self.wcs = wcs
        self.image_px = image_px
        self._rgb = None
        self._suspend = False
        self._build_widgets(transform or DEFAULT_TRANSFORM)
        self.refresh()

    def _build_widgets(self, start):
        """ """
        self.w_stretch = number(
            'Stretch', start.stretch, 20.0, 35.0, 0.01, **BOX,
            tooltip='asinh softening point in AB mag: higher shows fainter light')
        self.w_black = number('Black point', start.bw[0], 20.0, 35.0, 0.01,
                              **BOX, tooltip='AB mag mapped to output zero')
        self.w_white = number('White point', start.bw[1], 16.0, 30.0, 0.01,
                              **BOX, tooltip='AB mag mapped to output one')
        self.w_overshoot = number('Neg. overshoot', start.neg_overshoot,
                                  0.0, 1.0, 0.002, **BOX)
        self.w_sharpen = number('Sharpen', start.sharpen_strength,
                                0.0, 2.0, 0.005, **BOX)
        self.w_saturation = number('Saturation', start.saturation,
                                   0.0, 3.0, 0.005, **BOX)
        self.w_hue = number('Hue shift', start.hue, -180.0, 180.0, 0.1, **BOX)
        self.w_nir_to_l = number('NIR to L', start.nir_to_l, 0.0, 1.0, 0.005,
                                 **BOX)
        self.w_i_to_b = number('I to B', start.i_to_b, 0.0, 1.0, 0.005, **BOX)
        self.w_y_to_g = number('Y to G', start.y_to_g, 0.0, 1.0, 0.005, **BOX)
        self.w_j_to_r = number('J to R', start.j_to_r, 0.0, 1.0, 0.005, **BOX)
        self.w_scaling = [
            number(name, v, 0.0, 8.0, 0.01, **BOX)
            for name, v in zip(('I gain', 'Y gain', 'J gain', 'H gain'),
                               start.iyjh_scaling)]

        self.w_reset = widgets.Button(
            description='Reset colour', icon='undo',
            layout=widgets.Layout(width='150px'),
            tooltip='Back to the studio defaults this GUI opens with')
        self.w_reset.on_click(lambda _: self.apply_transform(DEFAULT_TRANSFORM))
        self.w_azulero = widgets.Button(
            description='Azulero defaults', icon='history',
            layout=widgets.Layout(width='170px'),
            tooltip="Back to azulero's own Transform defaults, which are "
                    "tuned for wide-field mosaics rather than these stamps")
        self.w_azulero.on_click(lambda _: self.apply_transform(color.Transform()))

        self.image = widgets.Image(
            format='png', layout=widgets.Layout(width=f'{self.image_px}px'))
        self.status = widgets.HTML(value='')

        for w in self._boxes():
            w.observe(self._on_change, names='value')

        def column(title, boxes):
            return widgets.VBox(
                [widgets.HTML(f"<b>{title}</b>")] + boxes,
                layout=widgets.Layout(margin='0 6px 0 0'))

        controls = widgets.VBox([
            widgets.HBox([
                column('Range', [self.w_stretch, self.w_black, self.w_white,
                                 self.w_overshoot, self.w_sharpen]),
                column('Colour', [self.w_saturation, self.w_hue,
                                  self.w_nir_to_l, self.w_i_to_b,
                                  self.w_y_to_g, self.w_j_to_r]),
                column('Band gain', self.w_scaling),
            ]),
            widgets.HBox([self.w_reset, self.w_azulero, self.status],
                         layout=widgets.Layout(align_items='center',
                                               margin='6px 0 0 0')),
        ])
        self.widget = widgets.HBox(
            [self.image, controls],
            layout=widgets.Layout(align_items='flex-start', grid_gap='12px'))

    def _boxes(self):
        """Every box whose change redraws the image."""
        return [self.w_stretch, self.w_black, self.w_white, self.w_overshoot,
                self.w_sharpen, self.w_saturation, self.w_hue, self.w_nir_to_l,
                self.w_i_to_b, self.w_y_to_g, self.w_j_to_r] + self.w_scaling

    def transform(self):
        """Return the :class:`azulero.image.color.Transform` the boxes describe."""
        return color.Transform(
            iyjh_scaling=tuple(w.value for w in self.w_scaling),
            sharpen_strength=self.w_sharpen.value,
            nir_to_l=self.w_nir_to_l.value,
            i_to_b=self.w_i_to_b.value,
            y_to_g=self.w_y_to_g.value,
            j_to_r=self.w_j_to_r.value,
            hue=self.w_hue.value,
            saturation=self.w_saturation.value,
            stretch=self.w_stretch.value,
            bw=(self.w_black.value, self.w_white.value),
            neg_overshoot=self.w_overshoot.value,
        )

    def apply_transform(self, transform):
        """Load a transform into the boxes and redraw once."""
        self._suspend = True
        try:
            self.w_stretch.value = transform.stretch
            self.w_black.value = transform.bw[0]
            self.w_white.value = transform.bw[1]
            self.w_overshoot.value = transform.neg_overshoot
            self.w_sharpen.value = transform.sharpen_strength
            self.w_saturation.value = transform.saturation
            self.w_hue.value = transform.hue
            self.w_nir_to_l.value = transform.nir_to_l
            self.w_i_to_b.value = transform.i_to_b
            self.w_y_to_g.value = transform.y_to_g
            self.w_j_to_r.value = transform.j_to_r
            for w, v in zip(self.w_scaling, transform.iyjh_scaling):
                w.value = v
        finally:
            self._suspend = False
        self.refresh()

    def rgb(self):
        """The RGB image, ``(ny, nx, 3)`` floats in [0, 1], for the current boxes."""
        return self._rgb

    def refresh(self, _=None):
        """Recompute the colour image and redraw the preview."""
        t0 = time.time()
        try:
            self._rgb = compute_continuum_rgb(self.stamps, self.wcs,
                                              self.transform())
            fig = Figure(figsize=(3.4, 3.4))
            ax = fig.add_subplot(111, projection=self.wcs)
            ax.imshow(self._rgb, origin='lower', interpolation='nearest')
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
        except Exception as e:
            self.status.value = (
                f"<span style='color:#c0392b'>Render failed: {e}</span>")
            return
        self.image.value = buf.getvalue()
        self.status.value = (
            f"<span style='color:#888'>{time.time() - t0:.2f} s</span>")

    def _on_change(self, change):
        if self._suspend:
            return
        self.refresh()

    def _ipython_display_(self):
        display(self.widget)


def make_azulero_gui(stamps, wcs, transform=None, image_px=340):
    """Build the colour image GUI. See :class:`AzuleroGUI` for the arguments."""
    return AzuleroGUI(stamps, wcs, transform=transform, image_px=image_px)

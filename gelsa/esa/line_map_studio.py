"""Stand-alone GUI for publication-quality Euclid emission line map figures.

Everything the figure needs is driven from the widgets: the archive login, the
MER stamps, the SIR frames, the stacked line map, the colour transform and the
export.  A notebook only has to open it::

    from gelsa.esa import line_map_studio
    studio = line_map_studio.make_line_map_studio()
    studio

A target is either typed in as RA, Dec and redshift, or picked from a
``targets.txt`` file with columns ``name,ra_deg,dec_deg,redshift``.  Where the
line map is centred can be given either as a redshift plus a rest-frame line,
or directly as the observed wavelength, which is what the stacker actually
uses.

Wavelengths are in nanometres throughout the interface, the labels and the
FITS header, and converted to the Angstrom that gelsa works in at the call.

The line map is coloured with :mod:`gelsa.colorize`: ``colorize.colorize``
traces each bright continuum pixel along its dispersion direction and paints
the trace with a wavelength colour map, so the hue at a point says which
wavelength the light landing there came from, and ``colorize.spectro_composite``
uses the stacked line flux as the brightness under that hue.  This is the
recipe of ``colorize/dinosaur_zoom.ipynb``.
"""
import inspect
import io
import json
import logging
import os
import time
import traceback

import numpy as np

from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.figure import Figure
import matplotlib as mpl

import ipywidgets as widgets
from IPython.display import display, clear_output

import gelsa
from gelsa import analysis, colorize, visu

from gelsa.elsa import branding
import gelsa.continuum_subtraction_base as csb

import azulero.process as proc
from azulero.image import color


# Everything the interface shows or takes is in nanometres; gelsa and azulero
# work in Angstrom, so the conversion happens at the call boundary only.
ANGSTROM_PER_NM = 10.0

# ``colorize.colorize`` gained a ``wavelength_range`` argument that sets the
# colour map normalization without touching which wavelengths are traced.
# Older installs lack it, and then the range cannot be pinned at all.
COLORIZE_TAKES_RANGE = 'wavelength_range' in inspect.signature(
    colorize.colorize).parameters

# Rest-frame vacuum wavelengths in nm of the lines Euclid grism data usually
# shows.
LINE_LIST = {
    'Ha 656.5': 656.461,
    'Hb 486.3': 486.268,
    '[OIII] 496.0': 496.0295,
    '[OIII] 500.8': 500.8240,
    '[NII] 655.0': 654.986,
    '[NII] 658.5': 658.527,
    '[SII] 671.8': 671.829,
    '[SII] 673.3': 673.267,
    '[SIII] 953.0': 953.0,
}

FILTERS = ('VIS', 'NIR_Y', 'NIR_J', 'NIR_H')

PANEL_CONTINUUM = 'Continuum'
PANEL_LINE = 'Line map'
PANEL_OVERLAY = 'Overlay'
PANEL_HUE = 'Wavelength hue'
PANEL_COUNT = 'Exposures'
PANEL_ORDER = [PANEL_CONTINUUM, PANEL_LINE, PANEL_OVERLAY, PANEL_HUE,
               PANEL_COUNT]

# Panels the figure opens with.
DEFAULT_PANELS = (PANEL_CONTINUUM, PANEL_LINE, PANEL_OVERLAY)

# Suffix each panel gets when the panels are exported to files of their own.
PANEL_SLUG = {
    PANEL_CONTINUUM: 'direct',
    PANEL_LINE: 'linemap',
    PANEL_OVERLAY: 'overlay',
    PANEL_HUE: 'huemap',
    PANEL_COUNT: 'exposures',
}

# Where the colour transform starts, and what ``Reset colour`` returns to:
# the azulero defaults with the adjustments that suit these stamps.
DEFAULT_TRANSFORM = color.Transform(
    stretch=29.0,
    bw=(29.0, 25.0),
    sharpen_strength=2.0,
    iyjh_scaling=(2.5, 1.3, 1.2, 1.0),
)

# Target the GUI opens on.
DEFAULT_RA = 150.09382
DEFAULT_DEC = 2.503955
DEFAULT_REDSHIFT = 0.751
DEFAULT_OBSERVED_NM = 1149.463

# Hue scale to use when it is pinned rather than read off the trace frame.
# Spans both grisms, from the blue end of BGS to the red end of RGS, so every
# target reads against one scale whichever grism its line falls in.
DEFAULT_HUE_RANGE_NM = (920.0, 1900.0)

DEFAULT_CREDENTIALS = '/media/home/my_workspace/password'
DEFAULT_CONFIG = '/media/home/my_workspace/gelsa-spectra/calib/gelsa_config.json'
DEFAULT_CALIBDIR = '/media/home/my_workspace/gelsa-spectra/calib/'


class Quiet:
    """Hold a noisy logger down while a routine that spams it runs."""

    def __init__(self, name='azulero', level=logging.ERROR):
        self.name = name
        self.level = level

    def __enter__(self):
        self.logger = logging.getLogger(self.name)
        self.previous = self.logger.level
        self.logger.setLevel(self.level)

    def __exit__(self, *exc):
        self.logger.setLevel(self.previous)
        return False


def number(description, value, vmin, vmax, step, tooltip='',
           width='215px', label_width='118px'):
    """A bounded number box for a figure parameter.

    Typed in directly, and stepped by ``step`` with the arrow keys or the
    spinner, which is how a nearly-right figure gets nudged the last bit.
    """
    return widgets.BoundedFloatText(
        description=description, value=value, min=vmin, max=vmax, step=step,
        layout=widgets.Layout(width=width),
        style={'description_width': label_width},
        description_tooltip=tooltip)


def trace_sweep(frame, wcs, fixed_angstrom, nsamples=200):
    """How far a frame's dispersion sweeps across the field.

    ``colorize`` paints a pixel by walking its trace across the frame's whole
    sensitivity range and mapping each wavelength back to the sky at the fixed
    wavelength.  What makes a frame useful is how far that walk travels: a
    trace crossing the cut-out paints a rainbow over it, while one that stays
    put paints a single spot and leaves the line map grey.

    Counting how many samples land inside the field does not measure that, and
    in fact prefers the wrong frames -- some map every wavelength back to the
    same pixel, so all their samples are "inside" while the trace goes nowhere.
    The score here is instead the extent the in-field part of the trace covers
    as a fraction of the field, which is 0 both for a frame that misses the
    target and for one that collapses to a point.

    This is the walk ``colorize`` does, run once for the field centre, so a
    usable frame can be picked before spending a minute on the full trace.
    """
    lo, hi = frame.params['wavelength_range']
    waves = np.linspace(lo, hi, nsamples)
    ny, nx = wcs.array_shape
    ra0, dec0 = wcs.celestial.all_pix2world(nx // 2, ny // 2, 0)
    try:
        x, y = frame.framecoord.radec_to_fov(
            ra0 * np.ones_like(waves), dec0 * np.ones_like(waves), waves)
        ra_t, dec_t = frame.framecoord.fov_to_radec(x, y, fixed_angstrom)
        px, py = wcs.celestial.wcs_world2pix(ra_t, dec_t, 0)
    except Exception:
        return 0.0
    inside = np.isfinite(px) & np.isfinite(py)
    inside &= (px > 0) & (py > 0) & (px < nx) & (py < ny)
    if np.sum(inside) < 2:
        return 0.0
    extent = np.hypot(np.ptp(px[inside]), np.ptp(py[inside]))
    return float(min(extent / max(nx, ny), 1.0))


def read_targets(path):
    """Read a ``name,ra_deg,dec_deg,redshift`` target list.

    Blank lines and lines starting with ``#`` are skipped, as is a header row.
    Whitespace-separated files are accepted as well as comma-separated ones,
    and the redshift column may be missing.

    Returns
    -------
    list of dict
        One ``{'name', 'ra', 'dec', 'redshift'}`` per target.
    """
    targets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = ([p.strip() for p in line.split(',')] if ',' in line
                     else line.split())
            if len(parts) < 3:
                continue
            try:
                ra, dec = float(parts[1]), float(parts[2])
            except ValueError:
                continue  # header row
            redshift = None
            if len(parts) > 3:
                try:
                    redshift = float(parts[3])
                except ValueError:
                    redshift = None
            targets.append({'name': parts[0], 'ra': ra, 'dec': dec,
                            'redshift': redshift})
    return targets


def compute_continuum_rgb(stamps, wcs, transform):
    """Run the azulero IYJH -> RGB transform and return a float RGB image.

    ``process_iyjh`` works in place and returns BGR, so the input is copied and
    the channels flipped here.
    """
    iyjh = np.array(stamps, dtype=np.float32, copy=True)
    iyjh[~np.isfinite(iyjh)] = 0
    with Quiet():
        bgr = proc.process_iyjh(iyjh, wcs, transform=transform)
    return np.clip(np.asarray(bgr)[:, :, ::-1], 0, 1)


class LineMapStudio:
    """Widget front end for the whole line map figure pipeline.

    Parameters
    ----------
    targets_file : str
        Target list read by the ``Targets`` dropdown.
    credentials_file, config_file, calibdir : str
        Defaults for the archive login and the gelsa calibration, all editable
        in the GUI.
    outdir : str
        Where ``Save figure`` and ``Save settings`` write.
    Euclid : astroquery.esa.euclid.core.EuclidClass, optional
        An already logged-in archive connection, if the notebook made one.
    G : gelsa.Gelsa, optional
        An already configured Gelsa instance.
    """

    def __init__(self, targets_file='targets.txt',
                 credentials_file=DEFAULT_CREDENTIALS,
                 config_file=DEFAULT_CONFIG, calibdir=DEFAULT_CALIBDIR,
                 outdir='.', Euclid=None, G=None):
        self.outdir = outdir
        self.Euclid = Euclid
        self.G = G
        self.EA = None
        self.A = None
        self.targets = []

        # Data of the target currently loaded.
        self.stamps = None
        self.wcs = None
        self.frame = None          # one SpecFrame, kept open for colorize
        self.frame_paths = []      # every SIR frame used for the stack
        self.frame_index = None    # which of them self.frame came from
        self.line_map = None       # (elmap, var, norm, count)
        self.hue_rgb = None        # wavelength-coloured trace image
        self.hue_wavelength = None  # wavelength the hue map was built at
        # Sensitivity range of the trace frame, in Angstrom.  Kept separately
        # so a figure rebuilt from cached FITS can still label the hue
        # colourbar without a SpecFrame to ask.
        self.wavelength_range = None
        self.hue_coverage = None   # fraction of the field the hue map reaches

        self._rgb_cache_key = None
        self._rgb_cache = None
        self._suspend = False

        self._build_widgets(targets_file, credentials_file, config_file,
                            calibdir)
        self._load_targets()
        self.refresh()

    # ------------------------------------------------------------------
    # widget construction
    # ------------------------------------------------------------------
    def _build_widgets(self, targets_file, credentials_file, config_file,
                       calibdir):
        """ """
        style = {'description_width': '130px'}
        wide = widgets.Layout(width='430px')
        mid = widgets.Layout(width='230px')
        small = widgets.Layout(width='170px')

        # --- connection ------------------------------------------------
        self.w_environment = widgets.Text(
            description='Environment', value='IDR', layout=small,
            style={'description_width': '90px'})
        self.w_credentials = widgets.Text(
            description='Credentials file', value=credentials_file,
            layout=wide, style=style)
        self.w_config = widgets.Text(
            description='Gelsa config', value=config_file, layout=wide,
            style=style)
        self.w_calibdir = widgets.Text(
            description='Calib dir', value=calibdir, layout=wide, style=style)
        self.w_connect = widgets.Button(
            description='Connect', icon='plug', button_style='primary',
            layout=widgets.Layout(width='150px'))
        self.w_connect_status = widgets.HTML(value=self._connection_summary())
        self.w_connect.on_click(self._on_connect)

        # --- target ----------------------------------------------------
        self.w_targets_file = widgets.Text(
            description='Targets file', value=targets_file, layout=wide,
            style=style)
        self.w_targets_reload = widgets.Button(
            description='Reload', icon='refresh',
            layout=widgets.Layout(width='110px'))
        self.w_target_select = widgets.Dropdown(
            description='Target', options=[('(none)', None)],
            layout=widgets.Layout(width='330px'), style=style)
        self.w_targets_reload.on_click(lambda _: self._load_targets())
        self.w_target_select.observe(self._on_target_select, names='value')

        self.w_ra = widgets.FloatText(
            description='RA (deg)', value=DEFAULT_RA, step=1e-5,
            layout=mid, style={'description_width': '80px'})
        self.w_dec = widgets.FloatText(
            description='Dec (deg)', value=DEFAULT_DEC, step=1e-5,
            layout=mid, style={'description_width': '80px'})

        self.w_center_mode = widgets.RadioButtons(
            description='Centre on', options=['Redshift', 'Observed wavelength'],
            value='Redshift', layout=widgets.Layout(width='320px'),
            style={'description_width': '90px'})
        self.w_redshift = widgets.FloatText(
            description='Redshift', value=DEFAULT_REDSHIFT, step=1e-4,
            layout=mid,
            style={'description_width': '80px'})
        self.w_rest_line = widgets.Dropdown(
            description='Rest line', options=list(LINE_LIST), value='Ha 656.5',
            layout=mid, style={'description_width': '80px'})
        self.w_observed = widgets.FloatText(
            description='Observed (nm)', value=DEFAULT_OBSERVED_NM, step=0.1,
            layout=mid,
            style={'description_width': '100px'})
        self.w_center_summary = widgets.HTML(value='')
        for w in (self.w_center_mode, self.w_redshift, self.w_rest_line,
                  self.w_observed):
            w.observe(self._on_center_change, names='value')

        # --- retrieval parameters --------------------------------------
        self.w_survey = widgets.Text(
            description='MER survey', value='cosmos', layout=small,
            style={'description_width': '90px'})
        self.w_stamp_size = widgets.BoundedIntText(
            description='Field (pix)', value=101, min=11, max=1201, step=10,
            layout=small, style={'description_width': '90px'},
            description_tooltip='Size of the cut-out in MER mosaic pixels, '
                                'which are 0.1 arcsec')
        self.w_radius = widgets.BoundedFloatText(
            description='Search (deg)', value=0.5, min=0.01, max=2.0,
            step=0.05, layout=small, style={'description_width': '100px'})
        self.w_grism_filter = widgets.Text(
            description='Frame filter', value='BGS', layout=small,
            style={'description_width': '90px'},
            description_tooltip='Substrings a SIR frame path may contain, '
                                'comma separated: BGS for the blue grism '
                                'alone, BGS,RGS to stack both, empty for '
                                'every frame the query returned')
        self.w_one_per_pointing = widgets.Checkbox(
            description='One frame per pointing', value=True, indent=False,
            layout=widgets.Layout(width='210px'))
        self.w_max_frames = widgets.BoundedIntText(
            description='Max frames', value=0, min=0, max=2000, step=10,
            layout=small, style={'description_width': '90px'},
            description_tooltip='0 uses every frame found')
        self.w_cs_method = widgets.Dropdown(
            description='Continuum', value='cutout',
            options=[('Median filter on cutout', 'cutout'),
                     ('Median filter on frame', 'frame'),
                     ('Leave it in', None)],
            layout=widgets.Layout(width='290px'),
            style={'description_width': '80px'},
            description_tooltip='How the continuum is subtracted before the '
                                'line flux is stacked')
        self.w_cs_size = widgets.BoundedIntText(
            description='Filter (pix)', value=50, min=1, max=400, step=5,
            layout=small, style={'description_width': '90px'})
        self.w_cs_method.observe(self._on_cs_method, names='value')
        self.w_padx = widgets.BoundedIntText(
            description='padx', value=50, min=0, max=500, step=5,
            layout=small, style={'description_width': '60px'},
            description_tooltip='Detector pixels kept either side of the '
                                'trace along x, the dispersion direction')
        self.w_pady = widgets.BoundedIntText(
            description='pady', value=50, min=0, max=500, step=5,
            layout=small, style={'description_width': '60px'},
            description_tooltip='Detector pixels kept above and below the '
                                'trace; the median filter needs blank sky '
                                'here or it subtracts the source itself')
        self.w_ndrops = widgets.BoundedIntText(
            description='ndrops', value=100, min=1, max=2000, step=10,
            layout=small, style={'description_width': '70px'})
        self.w_stack_width = widgets.BoundedIntText(
            description='Stack width', value=50, min=5, max=400, step=5,
            layout=small, style={'description_width': '90px'})

        self.w_load = widgets.Button(
            description='Load target', icon='download', button_style='primary',
            layout=widgets.Layout(width='170px'))
        self.w_restack = widgets.Button(
            description='Restack line map', icon='repeat',
            layout=widgets.Layout(width='190px'),
            tooltip='Re-run only the line map stack, reusing the crops')
        self.w_load.on_click(self._on_load)
        self.w_restack.on_click(self._on_restack)
        self.w_load_status = widgets.HTML(value='')
        self.w_progress = widgets.IntProgress(
            value=0, min=0, max=1, bar_style='info',
            layout=widgets.Layout(width='420px', visibility='hidden'))
        self.w_progress_label = widgets.HTML(value='')
        self.w_log = widgets.Output(layout=widgets.Layout(
            height='170px', overflow='auto', border='1px solid #ccc'))

        connection_tab = widgets.VBox([
            widgets.HTML(
                "<b>Archive and calibration</b> - connect once per kernel."),
            widgets.HBox([self.w_environment, self.w_connect,
                          self.w_connect_status]),
            self.w_credentials, self.w_config, self.w_calibdir,
        ])

        target_tab = widgets.VBox([
            widgets.HTML("<b>Target</b> - pick one from the file, or type "
                         "the coordinates in."),
            widgets.HBox([self.w_targets_file, self.w_targets_reload]),
            widgets.HBox([self.w_target_select, self.w_ra, self.w_dec]),
            widgets.HTML(
                "Where the line map is centred. With <b>Redshift</b> the "
                "observed wavelength is <code>rest * (1 + z)</code>; with "
                "<b>Observed wavelength</b> it is used as typed and the "
                "redshift is ignored."),
            widgets.HBox([self.w_center_mode,
                          widgets.VBox([self.w_redshift, self.w_rest_line]),
                          self.w_observed]),
            self.w_center_summary,
            widgets.HTML(
                "<hr><b>Retrieval</b> - the field has to be wide enough for "
                "the rainbow to show. Dispersion runs at about 4.5 nm per "
                "arcsec, so 101 MER pixels (10 arcsec) spans roughly 45 nm of "
                "colour; a few arcsec would come out one flat hue."),
            widgets.HBox([self.w_survey, self.w_stamp_size, self.w_radius,
                          self.w_grism_filter]),
            widgets.HBox([self.w_one_per_pointing, self.w_max_frames]),
            widgets.HBox([self.w_cs_method, self.w_cs_size, self.w_padx,
                          self.w_pady]),
            widgets.HTML(
                "<span style='color:#888'><b>padx</b> and <b>pady</b> set how "
                "much detector is kept around the trace. Widening <b>pady</b> "
                "gives the median filter blank sky to measure the continuum "
                "against; too tight and it subtracts the source along with "
                "it.</span>"),
            widgets.HBox([self.w_ndrops, self.w_stack_width, self.w_load,
                          self.w_restack]),
            self.w_load_status,
            widgets.HBox([self.w_progress, self.w_progress_label]),
            self.w_log,
        ])

        # --- continuum colour ------------------------------------------
        default = DEFAULT_TRANSFORM
        self.w_stretch = number(
            'Stretch', default.stretch, 20.0, 35.0, 0.01,
            tooltip='asinh softening point in AB mag: higher shows fainter light')
        self.w_black = number('Black point', default.bw[0], 20.0, 35.0, 0.01,
                              tooltip='AB mag mapped to output zero')
        self.w_white = number('White point', default.bw[1], 16.0, 30.0, 0.01,
                              tooltip='AB mag mapped to output one')
        self.w_overshoot = number('Neg. overshoot', default.neg_overshoot,
                                  0.0, 1.0, 0.002)
        self.w_saturation = number('Saturation', default.saturation,
                                   0.0, 3.0, 0.005)
        self.w_hue = number('Hue shift', default.hue, -180.0, 180.0, 0.1)
        self.w_sharpen = number('Sharpen', default.sharpen_strength,
                                0.0, 2.0, 0.005)
        self.w_nir_to_l = number('NIR to L', default.nir_to_l, 0.0, 1.0, 0.005)
        self.w_i_to_b = number('I to B', default.i_to_b, 0.0, 1.0, 0.005)
        self.w_y_to_g = number('Y to G', default.y_to_g, 0.0, 1.0, 0.005)
        self.w_j_to_r = number('J to R', default.j_to_r, 0.0, 1.0, 0.005)
        self.w_scaling = [
            number(name, v, 0.0, 8.0, 0.01, width='175px', label_width='75px')
            for name, v in zip(('I gain', 'Y gain', 'J gain', 'H gain'),
                               default.iyjh_scaling)]

        self.w_reset_color = widgets.Button(
            description='Reset colour', icon='undo',
            layout=widgets.Layout(width='160px'),
            tooltip='Back to the values this GUI opens with')
        self.w_reset_color.on_click(self._on_reset_color)
        self.w_azulero_defaults = widgets.Button(
            description='Azulero defaults', icon='history',
            layout=widgets.Layout(width='180px'),
            tooltip="Back to azulero's own Transform defaults, which are "
                    "tuned for wide-field mosaics rather than these stamps")
        self.w_azulero_defaults.on_click(self._on_azulero_defaults)

        color_tab = widgets.VBox([
            widgets.HTML(
                "The azulero IYJH transform. <b>Stretch</b>, <b>black</b> and "
                "<b>white point</b> set the asinh dynamic range in AB "
                "magnitudes and do most of the work; the blend sliders decide "
                "how much of each band reaches each output channel. Every "
                "box takes an exact number, and its arrows step finely. "
                "<b>Reset colour</b> goes back to the values this GUI opens "
                "with; <b>Azulero defaults</b> goes back to the library's own, "
                "which are set for wide mosaics and start fainter here."),
            widgets.HBox([
                widgets.VBox([self.w_stretch, self.w_black, self.w_white,
                              self.w_overshoot]),
                widgets.VBox([self.w_saturation, self.w_hue, self.w_sharpen,
                              self.w_nir_to_l]),
            ]),
            widgets.HBox([
                widgets.VBox([self.w_i_to_b, self.w_y_to_g, self.w_j_to_r]),
                widgets.VBox(self.w_scaling),
            ]),
            widgets.HBox([self.w_reset_color, self.w_azulero_defaults]),
        ])

        # --- line map colouring ----------------------------------------
        self.w_color_mode = widgets.Dropdown(
            description='Colouring', value='Wavelength rainbow',
            options=['Wavelength rainbow', 'Single hue'],
            layout=widgets.Layout(width='300px'), style=style)
        self.w_cmap = widgets.Dropdown(
            description='Wavelength cmap', value='rainbow',
            options=['rainbow', 'turbo', 'jet', 'hsv', 'gist_rainbow',
                     'nipy_spectral', 'viridis', 'plasma', 'cool'],
            layout=widgets.Layout(width='290px'), style=style)
        self.w_trace_level = number(
            'Trace percentile', 80.0, 50.0, 99.9, 0.01,
            tooltip='Only continuum pixels above this percentile are traced')
        self.w_trace_alpha = number(
            'Trace weight', 2.0, 0.5, 6.0, 0.01,
            tooltip='Exponent applied to the direct image before thresholding')
        self.w_wave_step = widgets.BoundedFloatText(
            description='Wave step (nm)', value=1.0, min=0.1, max=20.0,
            step=0.1, layout=widgets.Layout(width='210px'),
            style={'description_width': '115px'})
        self.w_source_lo = number('Source low %', 10.0, 0.0, 99.0, 0.05)
        self.w_source_hi = number('Source high %', 99.9, 50.0, 100.0, 0.01)
        self.w_fix_hue_range = widgets.Checkbox(
            description='Fix hue range', value=True, indent=False,
            layout=widgets.Layout(width='160px'),
            description_tooltip='Pin the wavelengths the colour map spans '
                                'instead of taking the trace frame\'s own '
                                'range, so targets traced from different '
                                'frames share one scale')
        self.w_hue_lo = number('Hue low (nm)', DEFAULT_HUE_RANGE_NM[0],
                               400.0, 2500.0, 1.0, width='195px',
                               label_width='95px')
        self.w_hue_hi = number('Hue high (nm)', DEFAULT_HUE_RANGE_NM[1],
                               400.0, 2500.0, 1.0, width='195px',
                               label_width='95px')
        self.w_trace_grism = widgets.Text(
            description='Trace grism', value='BGS',
            layout=widgets.Layout(width='230px'),
            style={'description_width': '100px'},
            description_tooltip='Prefer a frame of this grism for the trace. '
                                'The hue scale spans that grism\'s range, so '
                                'keeping every target on one grism is what '
                                'makes their colours comparable. Empty takes '
                                'whichever frame comes first.')
        self.w_hue_frame = widgets.Dropdown(
            description='Trace frame', options=[('(load a target)', None)],
            layout=widgets.Layout(width='430px'), style=style,
            description_tooltip='The dispersion of this one exposure sets the '
                                'hue; the line map itself stacks them all')
        self.w_build_hue = widgets.Button(
            description='Build hue map', icon='paint-brush',
            button_style='primary', layout=widgets.Layout(width='180px'))
        self.w_build_hue.on_click(self._on_build_hue)
        self.w_hue_status = widgets.HTML(value='')

        self.w_line_lo = number('Line low %', 70.0, 0.0, 99.0, 0.01)
        self.w_line_hi = number('Line high %', 99.9, 50.0, 100.0, 0.005)
        self.w_line_power = number('Line gamma', 0.8, 0.1, 3.0, 0.005)
        self.w_line_smooth = number('Smooth (pix)', 0.5, 0.0, 5.0, 0.01)
        self.w_single_hue = number('Hue', 0.15, 0.0, 1.0, 0.001,
                                   tooltip='Used by the Single hue mode')
        self.w_single_sat = number('Tint saturation', 1.0, 0.0, 1.0, 0.002)
        self.w_alpha = number('Overlay opacity', 0.8, 0.0, 1.0, 0.002)
        self.w_alpha_power = number('Opacity gamma', 1.0, 0.1, 4.0, 0.005)

        line_tab = widgets.VBox([
            widgets.HTML(
                "<b>Wavelength rainbow</b> uses <code>gelsa.colorize</code>: "
                "each bright continuum pixel is traced along its dispersion "
                "direction and painted with the wavelength colour map, so hue "
                "says which wavelength the light at a point came from. The "
                "stacked line flux then sets the brightness under that hue. "
                "The trace is slow, so it is built on demand - press "
                "<b>Build hue map</b> after changing anything in this block. "
                "By default the colour map spans the trace frame's own "
                "sensitivity range; <b>fix hue range</b> pins it instead, "
                "which is what lets blue-grism and red-grism targets share "
                "one scale. "
                "The hue comes from a single exposure's dispersion, so which "
                "one is chosen decides which way the rainbow runs, while the "
                "line map underneath stacks every exposure."),
            widgets.HBox([self.w_color_mode, self.w_cmap, self.w_wave_step]),
            widgets.HBox([self.w_trace_grism, self.w_hue_frame]),
            widgets.HBox([self.w_fix_hue_range, self.w_hue_lo, self.w_hue_hi]),
            widgets.HBox([
                widgets.VBox([self.w_trace_level, self.w_trace_alpha]),
                widgets.VBox([self.w_source_lo, self.w_source_hi]),
                widgets.VBox([self.w_build_hue, self.w_hue_status]),
            ]),
            widgets.HTML("<hr>How the stacked line flux is stretched into "
                         "brightness, and how hard it is pressed onto the "
                         "continuum in the <b>Overlay</b> panel."),
            widgets.HBox([
                widgets.VBox([self.w_line_lo, self.w_line_hi,
                              self.w_line_power, self.w_line_smooth]),
                widgets.VBox([self.w_single_hue, self.w_single_sat,
                              self.w_alpha, self.w_alpha_power]),
            ]),
        ])

        # --- layout -----------------------------------------------------
        self.w_clean = widgets.Checkbox(
            description='Clean image (no frame, no text)', value=True,
            indent=False, layout=widgets.Layout(width='290px'),
            description_tooltip='Hide the axes frame, ticks, labels, title, '
                                'colourbars and every annotation text')
        self.w_panels = {
            name: widgets.Checkbox(
                description=name, value=(name in DEFAULT_PANELS),
                indent=False,
                layout=widgets.Layout(width='150px'))
            for name in PANEL_ORDER}
        self.w_panel_width = widgets.BoundedFloatText(
            description='Panel width (in)', value=4.0, min=1.0, max=16.0,
            step=0.25, layout=widgets.Layout(width='210px'), style=style)
        self.w_panel_height = widgets.BoundedFloatText(
            description='Panel height (in)', value=4.0, min=1.0, max=16.0,
            step=0.25, layout=widgets.Layout(width='210px'), style=style)
        self.w_dpi = widgets.BoundedIntText(
            description='Preview DPI', value=150, min=50, max=600, step=10,
            layout=widgets.Layout(width='190px'),
            style={'description_width': '95px'})
        self.w_title = widgets.Text(
            description='Title', value='', layout=wide, style=style)
        self.w_show_wcs = widgets.Checkbox(
            description='Sky axes', value=True, indent=False,
            layout=widgets.Layout(width='120px'))
        self.w_show_grid = widgets.Checkbox(
            description='Grid', value=False, indent=False,
            layout=widgets.Layout(width='90px'))
        self.w_show_cbar = widgets.Checkbox(
            description='Colourbars', value=True, indent=False,
            layout=widgets.Layout(width='130px'))
        self.w_show_labels = widgets.Checkbox(
            description='Panel labels', value=True, indent=False,
            layout=widgets.Layout(width='140px'))
        self.w_show_scalebar = widgets.Checkbox(
            description='Scale bar', value=False, indent=False,
            layout=widgets.Layout(width='120px'))
        self.w_scalebar_arcsec = widgets.BoundedFloatText(
            description='arcsec', value=1.0, min=0.05, max=300.0, step=0.1,
            layout=widgets.Layout(width='160px'),
            style={'description_width': '55px'})
        self.w_show_compass = widgets.Checkbox(
            description='N/E compass', value=False, indent=False,
            layout=widgets.Layout(width='140px'))
        self.w_annot_color = widgets.ColorPicker(
            description='Annotation colour', value='#ffffff', concise=False,
            layout=widgets.Layout(width='260px'), style=style)
        self.w_font_size = widgets.BoundedIntText(
            description='Font size', value=11, min=4, max=40,
            layout=widgets.Layout(width='170px'),
            style={'description_width': '80px'})
        self.w_facecolor = widgets.ColorPicker(
            description='Background', value='#ffffff', concise=False,
            layout=widgets.Layout(width='240px'),
            style={'description_width': '90px'})
        self.w_count_cmap = widgets.Dropdown(
            description='Exposure cmap', value='viridis',
            options=sorted(['viridis', 'magma', 'plasma', 'inferno', 'cividis',
                            'Greys', 'bone', 'gray']),
            layout=widgets.Layout(width='240px'), style=style)

        layout_tab = widgets.VBox([
            widgets.HTML(
                "<b>Clean image</b> strips the frame, the ticks and every "
                "piece of text, and lets the pixels fill the figure edge to "
                "edge - what a cover image or a figure with its own caption "
                "wants. The graphical annotations still honour their own "
                "checkboxes, they just lose their labels."),
            widgets.HBox([self.w_clean] + [self.w_panels[n] for n in PANEL_ORDER[:2]]),
            widgets.HBox([self.w_panels[n] for n in PANEL_ORDER[2:]]),
            widgets.HBox([self.w_panel_width, self.w_panel_height, self.w_dpi,
                          self.w_facecolor]),
            self.w_title,
            widgets.HBox([self.w_show_wcs, self.w_show_grid, self.w_show_cbar,
                          self.w_show_labels]),
            widgets.HBox([self.w_show_scalebar, self.w_scalebar_arcsec,
                          self.w_show_compass]),
            widgets.HBox([self.w_annot_color, self.w_font_size,
                          self.w_count_cmap]),
        ])

        # --- export -----------------------------------------------------
        self.w_filename = widgets.Text(
            description='File name', value='line_map', layout=wide, style=style)
        self.w_format = widgets.Dropdown(
            description='Format', options=['pdf', 'png', 'svg', 'eps'],
            value='pdf', layout=widgets.Layout(width='170px'),
            style={'description_width': '60px'})
        self.w_save_dpi = widgets.BoundedIntText(
            description='Save DPI', value=300, min=72, max=1200, step=50,
            layout=widgets.Layout(width='190px'),
            style={'description_width': '80px'})
        self.w_tight = widgets.Checkbox(
            description='Tight bounding box', value=True, indent=False,
            layout=widgets.Layout(width='190px'))
        self.w_transparent = widgets.Checkbox(
            description='Transparent', value=False, indent=False,
            layout=widgets.Layout(width='140px'))
        self.w_separate = widgets.Checkbox(
            description='One file per panel', value=True, indent=False,
            layout=widgets.Layout(width='200px'),
            description_tooltip='Write the direct image, the line map and any '
                                'other panel to files of their own instead of '
                                'one combined figure')
        self.w_save_button = widgets.Button(
            description='Save figure', icon='download', button_style='success',
            layout=widgets.Layout(width='160px'))
        self.w_save_button.on_click(self._on_save)
        self.w_save_status = widgets.HTML(value='')
        self.w_params_file = widgets.Text(
            description='Settings file', value='line_map_params.json',
            layout=wide, style=style)
        self.w_params_save = widgets.Button(
            description='Save settings', icon='save',
            layout=widgets.Layout(width='150px'))
        self.w_params_load = widgets.Button(
            description='Load settings', icon='folder-open',
            layout=widgets.Layout(width='150px'))
        self.w_params_save.on_click(self._on_params_save)
        self.w_params_load.on_click(self._on_params_load)
        self.w_fits_button = widgets.Button(
            description='Save line map FITS', icon='table',
            layout=widgets.Layout(width='210px'))
        self.w_fits_button.on_click(self._on_save_fits)

        export_tab = widgets.VBox([
            widgets.HTML(
                "Write the figure exactly as previewed. Vector formats keep "
                "the annotations sharp; the pixels stay a raster either way. "
                "With <b>one file per panel</b> each panel is written on its "
                "own - the direct image as <code>&lt;name&gt;_direct</code>, "
                "the line map as <code>&lt;name&gt;_linemap</code> and so on - "
                "so they can be placed side by side, or stepped through, "
                "outside the figure."),
            widgets.HBox([self.w_filename, self.w_format, self.w_save_dpi]),
            widgets.HBox([self.w_separate, self.w_tight, self.w_transparent]),
            widgets.HBox([self.w_save_button, self.w_fits_button]),
            self.w_save_status,
            widgets.HTML("<hr>Settings can be carried between sessions."),
            widgets.HBox([self.w_params_file, self.w_params_save,
                          self.w_params_load]),
        ])

        self.tabs = widgets.Tab(children=[
            connection_tab, target_tab, color_tab, line_tab, layout_tab,
            export_tab, branding.acknowledgements()])
        for i, name in enumerate(['Archive & calibration', 'Target & data',
                                  'Continuum colour', 'Line colouring',
                                  'Layout', 'Export', 'Acknowledgements']):
            self.tabs.set_title(i, name)
        # Open on the target: the connection is set up once, or handed in.
        self.tabs.selected_index = 1

        self.image = widgets.Image(format='png',
                                   layout=widgets.Layout(max_width='100%'))
        self.status = widgets.HTML(value='')

        for w in self._live_widgets():
            w.observe(self._on_change, names='value')

        self.banner = branding.header(
            'Line Map Studio',
            f'Euclid emission line maps with gelsa {gelsa.version.version}')
        self.widget = widgets.VBox(
            [self.banner, self.tabs, self.status, self.image])
        self._on_center_change()

    def continuum_subtraction(self):
        """The continuum subtraction to hand to ``Analysis.crop``.

        ``MedianFilterCutout`` filters each extracted cutout, which is what the
        demo notebook uses; ``MedianFilterFrame`` filters the whole frame
        instead, which is slower but sees more of the trace. ``None`` leaves
        the continuum in, so the map is of the total flux at that wavelength.
        """
        method = self.w_cs_method.value
        if method is None:
            return None
        cls = (csb.MedianFilterCutout if method == 'cutout'
               else csb.MedianFilterFrame)
        return cls(median_filter_size_pix=self.w_cs_size.value)

    def _on_cs_method(self, change):
        self.w_cs_size.disabled = change['new'] is None

    def _live_widgets(self):
        """Widgets whose value change triggers a redraw."""
        return [
            self.w_stretch, self.w_black, self.w_white, self.w_overshoot,
            self.w_saturation, self.w_hue, self.w_sharpen, self.w_nir_to_l,
            self.w_i_to_b, self.w_y_to_g, self.w_j_to_r,
            self.w_color_mode, self.w_line_lo, self.w_line_hi,
            self.w_line_power, self.w_line_smooth, self.w_single_hue,
            self.w_single_sat, self.w_alpha, self.w_alpha_power,
            self.w_fix_hue_range, self.w_hue_lo, self.w_hue_hi,
            self.w_clean, self.w_panel_width, self.w_panel_height, self.w_dpi,
            self.w_title, self.w_show_wcs, self.w_show_grid, self.w_show_cbar,
            self.w_show_labels, self.w_show_scalebar, self.w_scalebar_arcsec,
            self.w_show_compass, self.w_annot_color, self.w_font_size,
            self.w_facecolor, self.w_count_cmap,
        ] + self.w_scaling + list(self.w_panels.values())

    # ------------------------------------------------------------------
    # target handling
    # ------------------------------------------------------------------
    def _connection_summary(self):
        bits = []
        bits.append('archive ' + ('ready' if self.EA is not None else 'not connected'))
        bits.append('gelsa ' + ('ready' if self.G is not None else 'not loaded'))
        colour = '#27ae60' if (self.EA is not None and self.G is not None) else '#888'
        return f"<span style='color:{colour}'>{', '.join(bits)}</span>"

    def _load_targets(self):
        """Fill the target dropdown from the targets file."""
        # Resolved against the working directory, not outdir: the target list
        # lives beside the notebook, while outdir is where figures go.
        path = self.w_targets_file.value
        try:
            self.targets = read_targets(path)
        except OSError as e:
            self.w_load_status.value = (
                f"<span style='color:#c0392b'>Could not read targets: {e}</span>")
            self.w_target_select.options = [('(none)', None)]
            return
        options = [('(none)', None)] + [
            (f"{t['name']}  ({t['ra']:.5f}, {t['dec']:.5f}"
             + (f", z={t['redshift']:g})" if t['redshift'] is not None else ')'),
             i) for i, t in enumerate(self.targets)]
        self._suspend = True
        try:
            self.w_target_select.options = options
        finally:
            self._suspend = False
        self.w_load_status.value = (
            f"<span style='color:#888'>{len(self.targets)} target(s) in "
            f"{os.path.abspath(path)}</span>")

    def _on_target_select(self, change):
        """Copy the chosen target's coordinates into the input boxes."""
        index = change['new']
        if index is None:
            return
        target = self.targets[index]
        self._suspend = True
        try:
            self.w_ra.value = target['ra']
            self.w_dec.value = target['dec']
            if target['redshift'] is not None:
                self.w_redshift.value = target['redshift']
                self.w_center_mode.value = 'Redshift'
            if not self.w_filename.value or self.w_filename.value == 'line_map':
                self.w_filename.value = target['name']
        finally:
            self._suspend = False
        self._on_center_change()

    def observed_wavelength(self):
        """The wavelength the line map is centred on, in nm."""
        if self.w_center_mode.value == 'Observed wavelength':
            return float(self.w_observed.value)
        rest = LINE_LIST[self.w_rest_line.value]
        return rest * (1 + float(self.w_redshift.value))

    def hue_range_angstrom(self):
        """The wavelengths the colour map spans, in Angstrom, or None.

        None means the trace frame's own sensitivity range is used, which is
        what ``colorize`` does by default. Pinning it lets targets traced from
        different frames -- blue grism at one redshift, red grism at another --
        be read against one scale. It changes only how the colour map is
        normalized: the trace still walks the wavelengths the frame really
        disperses.
        """
        if not self.w_fix_hue_range.value:
            return None
        lo, hi = self.w_hue_lo.value, self.w_hue_hi.value
        if hi <= lo:
            # Editing the boxes passes through states where the high value has
            # not caught up yet; falling back beats refusing to draw.
            return None
        return lo * ANGSTROM_PER_NM, hi * ANGSTROM_PER_NM

    def observed_angstrom(self):
        """The same wavelength in Angstrom, which is what gelsa takes."""
        return self.observed_wavelength() * ANGSTROM_PER_NM

    def _on_center_change(self, change=None):
        """Keep the two ways of naming the centre wavelength in step."""
        by_redshift = self.w_center_mode.value == 'Redshift'
        self.w_redshift.disabled = not by_redshift
        self.w_rest_line.disabled = not by_redshift
        self.w_observed.disabled = by_redshift
        observed = self.observed_wavelength()
        if by_redshift:
            self._suspend = True
            try:
                self.w_observed.value = round(observed, 3)
            finally:
                self._suspend = False
            rest = LINE_LIST[self.w_rest_line.value]
            text = (f"{self.w_rest_line.value} at z={self.w_redshift.value:g} "
                    f"&rarr; <b>{observed:.2f} nm</b> observed "
                    f"(rest {rest:.3f} nm)")
        else:
            text = (f"Line map centred on <b>{observed:.2f} nm</b> as "
                    f"typed; the redshift box is ignored.")
        if self.hue_wavelength is not None and \
                abs(self.hue_wavelength - observed) > 0.1:
            text += (" <span style='color:#c0392b'>The hue map was built at "
                     f"{self.hue_wavelength:.2f} nm - rebuild it.</span>")
        self.w_center_summary.value = text

    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------
    def connect(self):
        """Log in to the archive and configure gelsa. Raises on failure.

        An ``Euclid`` or ``G`` handed to the constructor is reused, so a batch
        run pays for the login and the calibration load only once.
        """
        from astroquery.esa.euclid.core import EuclidClass
        from . import euclid_archive

        with self.w_log:
            if self.Euclid is None:
                self.Euclid = EuclidClass(environment=self.w_environment.value)
                self.Euclid.login(credentials_file=self.w_credentials.value)
            self.EA = euclid_archive.EuclidArchive(self.Euclid)
            if self.G is None:
                self.G = gelsa.Gelsa(
                    config_file=self.w_config.value,
                    calibdir=self.w_calibdir.value,
                    zero_order_catalog=None)
        self.w_connect_status.value = self._connection_summary()

    def _on_connect(self, _=None):
        """ """
        self.w_connect.disabled = True
        self.w_connect_status.value = "<span style='color:#888'>connecting...</span>"
        try:
            self.connect()
        except Exception as e:
            self._report_error(self.w_connect_status, 'Connection failed', e)
        finally:
            self.w_connect.disabled = False

    def _progress_start(self, label, total):
        """Show the bar for a stage of ``total`` steps.

        Widget writes reach the browser as soon as they are made, so the bar
        keeps moving even though the kernel is busy inside the stage.
        """
        self.w_progress.max = max(int(total), 1)
        self.w_progress.value = 0
        self.w_progress.bar_style = 'info'
        self.w_progress.layout.visibility = 'visible'
        self.w_progress_label.value = f"<span style='color:#888'>{label}</span>"
        self._progress_text = label

    def _progress_step(self, value):
        """Move the bar to ``value`` steps done."""
        self.w_progress.value = min(int(value), self.w_progress.max)
        self.w_progress_label.value = (
            f"<span style='color:#888'>{self._progress_text} "
            f"{self.w_progress.value}/{self.w_progress.max}</span>")

    def _progress_done(self, failed=False):
        """Hide the bar once a stage is over."""
        self.w_progress.bar_style = 'danger' if failed else 'success'
        self.w_progress.layout.visibility = 'hidden'
        self.w_progress_label.value = ''

    def _report_error(self, target_widget, prefix, error):
        """Show a short message on the widget and the traceback in the log."""
        target_widget.value = (
            f"<span style='color:#c0392b'>{prefix}: {error}</span>")
        with self.w_log:
            print(f"{prefix}: {error}")
            traceback.print_exc()

    def load_target(self):
        """Retrieve stamps, SIR frames and crops, then stack. Raises on failure."""
        if self.EA is None or self.G is None:
            raise RuntimeError('not connected: call connect() first')
        ra, dec = self.w_ra.value, self.w_dec.value
        try:
            with self.w_log:
                clear_output()
                self.w_load_status.value = 'Loading MER stamps...'
                self._progress_start('MER stamps', len(FILTERS))
                stamps = []
                for i, filt in enumerate(FILTERS):
                    stamp, wcs = self.EA.load_stamp(
                        ra, dec, filter=filt, survey=self.w_survey.value,
                        width=self.w_stamp_size.value,
                        height=self.w_stamp_size.value)
                    stamps.append(stamp)
                    self._progress_step(i + 1)
                self.stamps = np.array(stamps)
                self.wcs = wcs
                self.wcs.array_shape = self.stamps.shape[1:]

                self.w_load_status.value = 'Querying SIR frames...'
                paths = self.EA.query_sir_frames(
                    ra, dec, radius=self.w_radius.value)
                needle = self.w_grism_filter.value.strip()
                wanted = [n.strip() for n in needle.split(',') if n.strip()]
                if wanted:
                    paths = [p for p in paths
                             if any(n in p for n in wanted)]
                if not paths:
                    raise RuntimeError(
                        f"no SIR frames within {self.w_radius.value} deg "
                        f"matching '{needle}'")

                if self.w_one_per_pointing.value:
                    self.w_load_status.value = (
                        f'Reading headers of {len(paths)} frames...')
                    self._progress_start('Reading frame headers', len(paths))
                    paths = self._one_per_pointing(paths)
                if self.w_max_frames.value > 0:
                    paths = paths[:self.w_max_frames.value]

                # One frame stays open: colorize needs its focal plane
                # model, and it has to be a frame that actually sees the
                # target -- a 0.5 degree search returns plenty that do not.
                self.frame_paths = list(paths)
                score = self.choose_trace_frame()
                self.w_load_status.value = (
                    f'Tracing from frame {self.frame_index} '
                    f'(sweep {score:.0%} of the field)...')

                self.w_load_status.value = (
                    f'Cropping {len(paths)} frames around the target...')
                self.A = analysis.Analysis(self.G)
                self.A.read_file_list(paths)
                self._progress_start('Cropping frames', len(paths))
                self.A.crop(
                    ra, dec,
                    redshift=self.w_redshift.value,
                    padx=self.w_padx.value,
                    pady=self.w_pady.value,
                    continuum_subtraction=self.continuum_subtraction(),
                    progress_hook=lambda n: self._progress_step(self.w_progress.value + n))
                self._progress_step(len(paths))
        except Exception:
            self._progress_done(failed=True)
            raise
        self._progress_done()

        self._rgb_cache_key = None
        self.hue_rgb = None
        self.hue_wavelength = None
        self._sync_hue_frame_options(self.frame_index or 0)
        self.stack_line_map()

    def _on_load(self, _=None):
        """ """
        self.w_load.disabled = True
        try:
            self.load_target()
        except Exception as e:
            self._report_error(self.w_load_status, 'Load failed', e)
            return
        finally:
            self.w_load.disabled = False
        self._build_hue_if_stale()

    def _build_hue_if_stale(self):
        """Trace the hue map if the rainbow needs one and it is out of date.

        Without it the rainbow composite has no hue to paint with and the line
        map does not show, so the GUI builds it as soon as a stack exists
        rather than waiting for *Build hue map*. Scripts that drive
        ``load_target`` and ``build_hue_map`` themselves are left alone.
        """
        if self.w_color_mode.value != 'Wavelength rainbow':
            return
        if self.line_map is None or self.frame is None:
            return
        if self.hue_rgb is not None and self.hue_wavelength is not None and \
                abs(self.hue_wavelength - self.observed_wavelength()) <= 0.1:
            return
        ready = self.w_load_status.value
        self.w_load_status.value = 'Tracing the hue map...'
        self.w_build_hue.disabled = True
        try:
            self.build_hue_map()
        except Exception as e:
            self._report_error(self.w_load_status, 'Hue map failed', e)
            return
        finally:
            self.w_build_hue.disabled = False
        # build_hue_map reports on the Line colouring tab; repeat an empty
        # trace here, where the user is looking.
        if not self.hue_coverage:
            ready += (" <span style='color:#c0392b'>The hue map is empty, so "
                      "the line map is grey: pick another trace frame on the "
                      "Line colouring tab.</span>")
        self.w_load_status.value = ready

    def seed(self, stamps, wcs, analysis, ra, dec, redshift=None,
             rest_line='Ha 656.5', observed_nm=None, line_map=None,
             build_hue=True):
        """Take over data a notebook has already built, instead of loading it.

        This leaves the studio where ``load_target`` would: the stamps, the
        crops and a trace frame are in place, so the line map can be restacked
        and the hue map built from the GUI straight away. ``Load target``
        still works for a new target once the archive is connected.

        Parameters
        ----------
        stamps : numpy.ndarray
            ``(4, ny, nx)`` stack of VIS, NIR_Y, NIR_J, NIR_H stamps.
        wcs : astropy.wcs.WCS
            WCS of the stamps; the line map is stacked on it.
        analysis : gelsa.analysis.Analysis
            With its crops already made by ``Analysis.crop``. Its Gelsa
            instance is reused, and its SIR frames are the trace candidates.
        ra, dec : float
            Target position in degrees.
        redshift : float, optional
            Centres the line map on ``rest_line`` at this redshift.
        rest_line : str
            Key of :data:`LINE_LIST`.
        observed_nm : float, optional
            Centre on this observed wavelength instead of a redshifted line.
        line_map : tuple, optional
            ``(elmap, var, norm, count)`` from ``build_stacked_line_map`` at
            that wavelength. Stacked here if not given.
        build_hue : bool
            Trace the hue map straight away, as *Load target* does, so the
            rainbow line map shows without pressing *Build hue map*.
        """
        stamps = np.asarray(stamps)
        if stamps.ndim != 3 or stamps.shape[0] != len(FILTERS):
            raise ValueError(
                f"stamps must have shape (4, ny, nx), got {stamps.shape}")
        if line_map is not None and \
                np.shape(line_map[0]) != stamps.shape[1:]:
            raise ValueError(
                f"line map shape {np.shape(line_map[0])} does not match the "
                f"stamps {stamps.shape[1:]}")
        if rest_line not in LINE_LIST:
            raise ValueError(f"rest_line must be one of {list(LINE_LIST)}")

        self.A = analysis
        self.G = analysis.G
        if self.EA is None and self.Euclid is not None:
            from . import euclid_archive
            self.EA = euclid_archive.EuclidArchive(self.Euclid)
        self.w_connect_status.value = self._connection_summary()

        self.stamps = stamps
        self.wcs = wcs
        self.wcs.array_shape = stamps.shape[1:]

        self._suspend = True
        try:
            self.w_target_select.value = None
            self.w_ra.value = ra
            self.w_dec.value = dec
            self.w_stamp_size.value = stamps.shape[1]
            self.w_rest_line.value = rest_line
            if redshift is not None:
                self.w_redshift.value = redshift
            if observed_nm is not None:
                self.w_observed.value = observed_nm
                self.w_center_mode.value = 'Observed wavelength'
            else:
                self.w_center_mode.value = 'Redshift'
        finally:
            self._suspend = False
        self._on_center_change()

        self.frame_paths = [pack['frame_path'] for pack in analysis.sir_pack
                            if 'frame_path' in pack]
        self._rgb_cache_key = None
        self.hue_rgb = None
        self.hue_wavelength = None
        self.w_load_status.value = 'Choosing a trace frame...'
        score = self.choose_trace_frame()
        self._sync_hue_frame_options(self.frame_index)

        if line_map is None:
            self.stack_line_map()
        else:
            self.line_map = line_map
            field = stamps.shape[1] * self._pixel_scale_arcsec()
            self.w_load_status.value = (
                f"<span style='color:#27ae60'>Seeded from the notebook: "
                f"{len(analysis.crop_list)} crops, {field:.1f} arcsec field, "
                f"line map at {self.observed_wavelength():.2f} nm, peak "
                f"{np.nanmax(line_map[0]):.3g} erg/cm2/s; tracing from frame "
                f"{self.frame_index} (sweep {score:.0%} of the field).</span>")
            self.refresh()
        if build_hue:
            self._build_hue_if_stale()

    def choose_trace_frame(self, max_tries=12, good=0.25):
        """Pick a frame whose dispersion sweeps the field, and open it.

        Frames of the preferred grism are tried first, then the rest, and the
        first one that sweeps at least ``good`` of the field is kept. The grism matters beyond
        geometry: ``colorize`` spreads the colour map over the trace frame's
        whole sensitivity range, so a red-grism frame would give this target a
        1198-1891 nm hue scale while its neighbours run 924-1368 nm, and the
        two could not be compared. Only if nothing sweeps the field does the
        best-scoring candidate win, and the caller is told the score is poor.

        Returns the sweep of the frame it settled on, as a fraction of the
        field.
        """
        best_index, best_score = None, -1.0
        fixed = self.observed_angstrom()
        prefer = self.w_trace_grism.value.strip()
        candidates = list(enumerate(self.frame_paths))
        if prefer:
            candidates.sort(key=lambda c: prefer not in c[1])
        for index, path in candidates[:max_tries]:
            frame = self.G.load_frame(path)
            score = trace_sweep(frame, self.wcs, fixed)
            if score > best_score:
                best_index, best_score, best_frame = index, score, frame
            if score >= good:
                break
        if best_index is None:
            raise RuntimeError('no SIR frames to trace the hue map from')
        self.frame = best_frame
        self.frame_index = best_index
        self.wavelength_range = best_frame.params['wavelength_range']
        return best_score

    def _sync_hue_frame_options(self, index=None):
        """Keep the trace-frame dropdown listing the frames actually loaded.

        Its value has to be one of its options, so the two are always set
        together; a bare index would otherwise be rejected.
        """
        self._suspend = True
        try:
            self.w_hue_frame.options = [
                (f'{i}: {os.path.basename(path)[:70]}', i)
                for i, path in enumerate(self.frame_paths)
            ] or [('(load a target)', None)]
            if index is not None and 0 <= index < len(self.frame_paths):
                self.w_hue_frame.value = index
        finally:
            self._suspend = False

    def set_trace_frame(self, index):
        """Open a particular frame from the list as the trace frame.

        An index only means anything against the frame list it was read from,
        so pinning one is for re-tracing an existing target, not for a fresh
        retrieval where the query may come back in a different order.

        Returns the sweep of that frame, for reporting.
        """
        path = self.frame_paths[index]
        frame = self.G.load_frame(path)
        self.frame = frame
        self.frame_index = index
        self.wavelength_range = frame.params['wavelength_range']
        self._sync_hue_frame_options(index)
        return trace_sweep(frame, self.wcs, self.observed_angstrom())

    def _one_per_pointing(self, paths):
        """Keep one frame per pointing, as the demo notebook does.

        Every exposure carries its own pointing id, including the blue and red
        exposures of the same field, so this stays correct with both grisms in
        the list.
        """
        seen = {}
        keep = []
        for i, path in enumerate(paths):
            frame = self.G.load_frame(path)
            pointing = frame.params['PTGID']
            self._progress_step(i + 1)
            if pointing in seen:
                continue
            seen[pointing] = True
            keep.append(path)
        return keep

    def stack_line_map(self):
        """Stack the line map at the current wavelength. Raises on failure.

        Reuses the crops, so this is the cheap way to move the centre
        wavelength once a target is loaded.
        """
        if self.A is None or self.wcs is None:
            raise RuntimeError('no crops: call load_target() first')
        observed = self.observed_wavelength()
        self.w_load_status.value = (
            f'Stacking the line map at {observed:.2f} nm over '
            f'{len(self.A.crop_list)} crops...')
        try:
            with self.w_log:
                crops = self.A.crop_list
                self._progress_start('Stacking line map', len(crops))
                try:
                    self.line_map = self.A.build_stacked_line_map(
                        self.observed_angstrom(), self.wcs,
                        width=self.w_stack_width.value,
                        ndrops=self.w_ndrops.value,
                        progress_hook=lambda n: self._progress_step(self.w_progress.value + n))
                except AttributeError as e:
                    # The stacker leaves a scalar zero to take the shape of
                    # when not one crop yielded an extraction.
                    raise RuntimeError(
                        f"nothing could be extracted from any of the "
                        f"{len(crops)} crops at {observed:.2f} nm") from e
                self._progress_step(len(crops))
        except Exception:
            self._progress_done(failed=True)
            raise
        self._progress_done()
        field = self.stamps.shape[1] * self._pixel_scale_arcsec()
        self.w_load_status.value = (
            f"<span style='color:#27ae60'>Ready: {len(self.A.crop_list)} "
            f"crops, {field:.1f} arcsec field, line map at {observed:.2f} "
            f"nm, peak {np.nanmax(self.line_map[0]):.3g} "
            "erg/cm2/s.</span>")
        self.refresh()

    def _on_restack(self, _=None):
        """ """
        self.w_restack.disabled = True
        try:
            self.stack_line_map()
        except Exception as e:
            self._report_error(self.w_load_status, 'Stacking failed', e)
            return
        finally:
            self.w_restack.disabled = False
        self._build_hue_if_stale()

    def build_hue_map(self):
        """Trace the dispersion of every bright continuum pixel into a hue map.

        This is ``colorize.colorize``: hue encodes the wavelength whose light
        would land at that point, which is what turns a flat line map into the
        rainbow image. Raises on failure.
        """
        if self.frame is None or self.stamps is None:
            raise RuntimeError('no trace frame: call load_target() first')
        self.w_hue_status.value = 'Tracing...'
        self._progress_start('Tracing dispersion', 1)
        t0 = time.time()
        try:
            with self.w_log:
                wanted = self.w_hue_frame.value
                if wanted is not None and wanted != self.frame_index:
                    self.frame = self.G.load_frame(self.frame_paths[wanted])
                    self.frame_index = wanted
                source = visu.normalize_image(
                    np.asarray(self.stamps[0], dtype='d'),
                    levels=(self.w_source_lo.value, self.w_source_hi.value),
                    power=1)
                fixed = self.hue_range_angstrom()
                own = self.frame.params['wavelength_range']
                extra = {}
                if COLORIZE_TAKES_RANGE:
                    # None leaves colorize on the frame's own range, which is
                    # what it did before the argument existed.
                    extra['wavelength_range'] = fixed
                elif fixed is not None:
                    raise RuntimeError(
                        'the installed gelsa.colorize.colorize has no '
                        'wavelength_range argument, so the hue range cannot '
                        'be pinned; reinstall gelsa from its source checkout, '
                        'or clear "fix hue range"')
                self.hue_rgb = colorize.colorize(
                    self.frame, np.ma.filled(source, 0), self.wcs,
                    self.observed_angstrom(),
                    level=self.w_trace_level.value,
                    alpha=self.w_trace_alpha.value,
                    wave_step=self.w_wave_step.value * ANGSTROM_PER_NM,
                    cmap=self.w_cmap.value, **extra)
            self.hue_wavelength = self.observed_wavelength()
            self.wavelength_range = fixed if fixed is not None else own
            self.hue_coverage = float(
                np.mean(self.hue_rgb.sum(axis=2) > 0))
        except Exception:
            self._progress_done(failed=True)
            raise
        self._progress_step(1)
        self._progress_done()
        if self.hue_coverage:
            self.w_hue_status.value = (
                f"<span style='color:#27ae60'>Hue map built in "
                f"{time.time() - t0:.1f} s, covering "
                f"{self.hue_coverage:.0%} of the field.</span>")
        else:
            # Without hue the composite has nothing to colour with, and the
            # line map comes out grey.
            self.w_hue_status.value = (
                "<span style='color:#c0392b'>The hue map is empty: frame "
                f"{self.frame_index} does not disperse this field at "
                f"{self.observed_wavelength():.1f} nm, so the line map would "
                "be grey. Pick another trace frame.</span>")
        self._on_center_change()
        self.refresh()

    def _on_build_hue(self, _=None):
        """ """
        self.w_build_hue.disabled = True
        try:
            self.build_hue_map()
        except Exception as e:
            self._report_error(self.w_hue_status, 'Hue map failed', e)
        finally:
            self.w_build_hue.disabled = False

    # ------------------------------------------------------------------
    # image construction
    # ------------------------------------------------------------------
    def transform(self):
        """Return the :class:`azulero.image.color.Transform` the sliders describe."""
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

    def continuum_rgb(self):
        """The continuum RGB image for the current transform, cached."""
        if self.stamps is None:
            return None
        transform = self.transform()
        key = json.dumps([transform.iyjh_scaling, transform.sharpen_strength,
                          transform.nir_to_l, transform.i_to_b,
                          transform.y_to_g, transform.j_to_r, transform.hue,
                          transform.saturation, transform.stretch,
                          transform.bw, transform.neg_overshoot], default=list)
        if key != self._rgb_cache_key:
            self._rgb_cache = compute_continuum_rgb(self.stamps, self.wcs,
                                                    self.transform())
            self._rgb_cache_key = key
        return self._rgb_cache

    def _line_flux(self):
        """The stacked line map, optionally smoothed."""
        if self.line_map is None:
            return None
        elmap = np.asarray(self.line_map[0], dtype='d')
        if self.w_line_smooth.value > 0:
            from scipy import ndimage
            elmap = ndimage.gaussian_filter(elmap, self.w_line_smooth.value)
        return elmap

    def line_rgb(self):
        """The coloured line map: rainbow hue from colorize, or a single tint."""
        elmap = self._line_flux()
        if elmap is None:
            return None
        levels = (self.w_line_lo.value, self.w_line_hi.value)
        if self.w_color_mode.value == 'Wavelength rainbow':
            if self.hue_rgb is None:
                return None
            return np.clip(colorize.spectro_composite(
                elmap, self.hue_rgb, self.wcs, self.wcs,
                levels=levels, power=self.w_line_power.value), 0, 1)
        norm = visu.normalize_image(elmap, levels=levels,
                                    power=self.w_line_power.value)
        return np.clip(visu.colorize_image(
            np.ma.filled(norm, 0), hue=self.w_single_hue.value,
            saturation=self.w_single_sat.value), 0, 1)

    def available_panels(self):
        """The ticked panels that actually have their data, in figure order."""
        rgb = self.continuum_rgb()
        line_rgb = self.line_rgb()
        wanted = [n for n in PANEL_ORDER if self.w_panels[n].value]
        return self._filter_panels(wanted, rgb, line_rgb)

    def _filter_panels(self, wanted, rgb, line_rgb):
        """Drop the panels whose data has not been built yet."""
        panels = []
        for name in wanted:
            if name == PANEL_CONTINUUM and rgb is None:
                continue
            if name in (PANEL_LINE, PANEL_OVERLAY) and line_rgb is None:
                continue
            if name == PANEL_OVERLAY and rgb is None:
                continue
            if name == PANEL_HUE and self.hue_rgb is None:
                continue
            if name == PANEL_COUNT and self.line_map is None:
                continue
            panels.append(name)
        return panels

    def figure(self, panels=None):
        """Build the matplotlib figure for the current settings.

        Parameters
        ----------
        panels : list of str, optional
            Which panels to draw. Defaults to the ones ticked in the Layout
            tab; passing a single name renders that panel on its own, which is
            how the per-panel export works.
        """
        clean = self.w_clean.value
        rgb = self.continuum_rgb()
        line_rgb = self.line_rgb()
        panels = self._filter_panels(
            [n for n in PANEL_ORDER if self.w_panels[n].value]
            if panels is None else panels, rgb, line_rgb)

        n = len(panels)
        fig = Figure(figsize=(self.w_panel_width.value * max(n, 1),
                              self.w_panel_height.value),
                     dpi=self.w_dpi.value)
        fig.set_facecolor(self.w_facecolor.value)

        if n == 0:
            if not clean:
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.text(0.5, 0.5, 'Load a target to build the figure',
                        ha='center', va='center', fontsize=13, color='#888')
            return fig

        fs = self.w_font_size.value
        annot = self.w_annot_color.value

        for i, name in enumerate(panels):
            projection = self.wcs if (self.w_show_wcs.value and not clean) else None
            ax = fig.add_subplot(1, n, i + 1, projection=projection)
            ax.set_facecolor('k')
            mappable = None
            label = name

            if name == PANEL_CONTINUUM:
                ax.imshow(rgb, origin='lower', interpolation='nearest')
                label = 'VIS + Y J H'
            elif name == PANEL_LINE:
                ax.imshow(line_rgb, origin='lower', interpolation='nearest')
                mappable = self._wavelength_mappable()
                label = f'{self.observed_wavelength():.1f} nm'
            elif name == PANEL_OVERLAY:
                composite = visu.make_image_composite(
                    rgb, line_rgb, alpha_power=self.w_alpha_power.value,
                    alpha=self.w_alpha.value)
                ax.imshow(np.clip(composite, 0, 1), origin='lower',
                          interpolation='nearest')
                label = f'Continuum + {self.observed_wavelength():.1f} nm'
            elif name == PANEL_HUE:
                ax.imshow(np.clip(self.hue_rgb, 0, 1), origin='lower',
                          interpolation='nearest')
                mappable = self._wavelength_mappable()
                label = 'Wavelength of origin'
            elif name == PANEL_COUNT:
                mappable = ax.imshow(self.line_map[3], origin='lower',
                                     cmap=self.w_count_cmap.value,
                                     interpolation='nearest')
                label = 'Exposure count'

            self._decorate(ax, fig, mappable, label, first=(i == 0), fs=fs,
                           annot=annot, panel=name, clean=clean)

        if clean:
            # Edge to edge: no frame, no margins, nothing but pixels.
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                                wspace=0, hspace=0)
        else:
            if self.w_title.value:
                fig.suptitle(self.w_title.value, fontsize=fs + 3)
            fig.tight_layout()
        return fig

    def _wavelength_mappable(self):
        """Colourbar for the rainbow: hue against wavelength over the frame range.

        ``colorize`` spreads the colour map across the frame's whole
        sensitivity range, which the frame reports in Angstrom, unless the
        range has been pinned.
        """
        if self.w_color_mode.value != 'Wavelength rainbow':
            return None
        fixed = self.hue_range_angstrom()
        if fixed is not None:
            self.wavelength_range = fixed
        elif self.frame is not None:
            self.wavelength_range = self.frame.params['wavelength_range']
        if self.wavelength_range is None:
            return None
        lo, hi = self.wavelength_range
        return mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(lo / ANGSTROM_PER_NM,
                                      hi / ANGSTROM_PER_NM),
            cmap=self.w_cmap.value)

    def _decorate(self, ax, fig, mappable, label, first, fs, annot, panel,
                  clean):
        """Axes labels, colourbar, scale bar and compass."""
        if clean:
            ax.set_axis_off()
        elif self.w_show_wcs.value:
            ax.coords[0].set_axislabel('RA (J2000)', fontsize=fs)
            ax.coords[1].set_axislabel('Dec (J2000)' if first else '',
                                       fontsize=fs)
            ax.coords[0].set_ticklabel(size=fs - 2)
            ax.coords[1].set_ticklabel(size=fs - 2, visible=first)
            if self.w_show_grid.value:
                ax.grid(color=annot, ls='dotted', alpha=0.4)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if self.w_show_labels.value and not clean:
            ax.text(0.03, 0.97, label, transform=ax.transAxes, va='top',
                    ha='left', color=annot, fontsize=fs, path_effects=_halo())

        if mappable is not None and self.w_show_cbar.value and not clean:
            if panel == PANEL_COUNT:
                unit = 'Exposure count'
            else:
                unit = 'Wavelength of origin (nm)'
            cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(unit, fontsize=fs - 1)
            cb.ax.tick_params(labelsize=fs - 3)

        if self.w_show_scalebar.value:
            self._draw_scalebar(ax, annot, fs, clean)
        if self.w_show_compass.value:
            self._draw_compass(ax, annot, fs, clean)

    def _pixel_scale_arcsec(self):
        """Mean pixel scale of the stamp WCS, in arcsec."""
        return float(np.mean(proj_plane_pixel_scales(self.wcs.celestial))) * 3600.0

    def _draw_scalebar(self, ax, annot, fs, clean):
        """Horizontal bar of a round angular size, bottom right."""
        arcsec = self.w_scalebar_arcsec.value
        length_pix = arcsec / self._pixel_scale_arcsec()
        (xlo, xhi), (ylo, yhi) = _axes_extent(ax)
        nx, ny = xhi - xlo, yhi - ylo
        if length_pix > 0.8 * nx:
            return
        x1 = xhi - 0.06 * nx
        x0 = x1 - length_pix
        y = ylo + 0.08 * ny
        ax.plot([x0, x1], [y, y], color=annot, lw=2.5, solid_capstyle='butt')
        if not clean:
            ax.text((x0 + x1) / 2, y + 0.025 * ny, f'{arcsec:g}"', color=annot,
                    ha='center', va='bottom', fontsize=fs - 1,
                    path_effects=_halo())

    def _draw_compass(self, ax, annot, fs, clean):
        """North and East arrows read off the WCS, top right."""
        (xlo, xhi), (ylo, yhi) = _axes_extent(ax)
        nx, ny = xhi - xlo, yhi - ylo
        x0, y0 = xlo + 0.86 * nx, ylo + 0.84 * ny
        length = 0.10 * min(nx, ny)
        ra0, dec0 = self.wcs.celestial.all_pix2world(x0, y0, 0)
        step = self._pixel_scale_arcsec() / 3600.0 * length
        for dra, ddec, text in ((0.0, step, 'N'),
                                (step / np.cos(np.radians(dec0)), 0.0, 'E')):
            x1, y1 = self.wcs.celestial.all_world2pix(ra0 + dra, dec0 + ddec, 0)
            dx, dy = float(x1) - x0, float(y1) - y0
            norm = np.hypot(dx, dy)
            if norm == 0:
                continue
            dx, dy = dx / norm * length, dy / norm * length
            ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                        arrowprops=dict(color=annot, width=0.8, headwidth=5,
                                        headlength=5))
            if not clean:
                ax.text(x0 + 1.35 * dx, y0 + 1.35 * dy, text, color=annot,
                        ha='center', va='center', fontsize=fs - 1,
                        path_effects=_halo())

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def refresh(self, _=None):
        """Re-render the figure into the image widget."""
        rainbow = self.w_color_mode.value == 'Wavelength rainbow'
        self.w_single_hue.disabled = rainbow
        self.w_single_sat.disabled = rainbow
        self.w_build_hue.disabled = not rainbow
        self.w_cmap.disabled = not rainbow
        self.w_fix_hue_range.disabled = not rainbow
        for w in (self.w_hue_lo, self.w_hue_hi):
            w.disabled = not (rainbow and self.w_fix_hue_range.value)
        self.w_scalebar_arcsec.disabled = not self.w_show_scalebar.value
        for w in (self.w_show_wcs, self.w_show_grid, self.w_show_cbar,
                  self.w_show_labels, self.w_title):
            w.disabled = self.w_clean.value

        note = ''
        if self.w_fix_hue_range.value and \
                self.w_hue_hi.value <= self.w_hue_lo.value:
            note = (" &middot; <span style='color:#c0392b'>the hue range is "
                    "empty, so the trace frame's own range is in use</span>")

        t0 = time.time()
        try:
            fig = self.figure()
        except Exception as e:
            self._report_error(self.status, 'Render failed', e)
            return
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.w_dpi.value,
                    bbox_inches='tight',
                    pad_inches=0 if self.w_clean.value else 0.1,
                    facecolor=fig.get_facecolor())
        self.image.value = buf.getvalue()
        self.status.value = (
            f"<span style='color:#888'>rendered in {time.time() - t0:.2f} s"
            f"</span>{note}")

    def _on_change(self, change):
        if self._suspend:
            return
        self.refresh()

    def apply_transform(self, transform):
        """Load a :class:`azulero.image.color.Transform` into the colour boxes.

        The boxes are filled with the redraw suspended, so the figure is
        rebuilt once at the end rather than on every box.
        """
        default = transform
        self._suspend = True
        try:
            self.w_stretch.value = default.stretch
            self.w_black.value = default.bw[0]
            self.w_white.value = default.bw[1]
            self.w_overshoot.value = default.neg_overshoot
            self.w_saturation.value = default.saturation
            self.w_hue.value = default.hue
            self.w_sharpen.value = default.sharpen_strength
            self.w_nir_to_l.value = default.nir_to_l
            self.w_i_to_b.value = default.i_to_b
            self.w_y_to_g.value = default.y_to_g
            self.w_j_to_r.value = default.j_to_r
            for w, v in zip(self.w_scaling, default.iyjh_scaling):
                w.value = v
        finally:
            self._suspend = False
        self.refresh()

    def _on_reset_color(self, _=None):
        """Back to the values the GUI opens with."""
        self.apply_transform(DEFAULT_TRANSFORM)

    def _on_azulero_defaults(self, _=None):
        """Back to azulero's own untouched ``Transform`` defaults."""
        self.apply_transform(color.Transform())

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def params(self):
        """Every GUI setting as a plain dict, for saving or reporting."""
        out = {w.description: w.value for w in self._live_widgets()
               if not isinstance(w, widgets.Dropdown)}
        out['iyjh_scaling'] = [w.value for w in self.w_scaling]
        out['panels'] = [n for n in PANEL_ORDER if self.w_panels[n].value]
        out['color_mode'] = self.w_color_mode.value
        out['count_cmap'] = self.w_count_cmap.value
        out['cmap'] = self.w_cmap.value
        out['trace'] = {'level': self.w_trace_level.value,
                        'alpha': self.w_trace_alpha.value,
                        'wave_step': self.w_wave_step.value,
                        'source_levels': [self.w_source_lo.value,
                                          self.w_source_hi.value],
                        'frame_index': self.frame_index,
                        'grism': self.w_trace_grism.value,
                        'hue_range_nm': ([self.w_hue_lo.value,
                                          self.w_hue_hi.value]
                                         if self.w_fix_hue_range.value
                                         else None)}
        out['target'] = {'ra': self.w_ra.value, 'dec': self.w_dec.value,
                         'redshift': self.w_redshift.value,
                         'rest_line': self.w_rest_line.value,
                         'center_mode': self.w_center_mode.value,
                         'observed': self.observed_wavelength()}
        # Everything that decides what is retrieved and stacked. Changing any
        # of these invalidates a cached line map; the render settings above do
        # not.
        out['data'] = {'field_pix': self.w_stamp_size.value,
                       'survey': self.w_survey.value,
                       'search_radius_deg': self.w_radius.value,
                       'frame_filter': self.w_grism_filter.value,
                       'one_per_pointing': self.w_one_per_pointing.value,
                       'max_frames': self.w_max_frames.value,
                       'continuum': self.w_cs_method.value,
                       'continuum_size_pix': self.w_cs_size.value,
                       'padx': self.w_padx.value,
                       'pady': self.w_pady.value,
                       'ndrops': self.w_ndrops.value,
                       'stack_width': self.w_stack_width.value}
        out['export'] = {'name': self.w_filename.value,
                         'format': self.w_format.value,
                         'dpi': self.w_save_dpi.value,
                         'tight': self.w_tight.value,
                         'transparent': self.w_transparent.value,
                         'separate_panels': self.w_separate.value}
        return out

    def write_figure(self, path, panels=None):
        """Render ``panels`` and write them to ``path`` at the export settings."""
        fig = self.figure(panels=panels)
        fig.savefig(
            path, dpi=self.w_save_dpi.value,
            bbox_inches='tight' if self.w_tight.value else None,
            pad_inches=0 if self.w_clean.value else 0.1,
            transparent=self.w_transparent.value,
            facecolor=None if self.w_transparent.value else fig.get_facecolor())
        return os.path.abspath(path)

    def save_figures(self):
        """Write the figure, combined or one file per panel, and return the paths.

        Raises on failure, so a script can tell that nothing was written.
        """
        name = self.w_filename.value.strip() or 'figure'
        fmt = self.w_format.value
        if not self.w_separate.value:
            return [self.write_figure(
                os.path.join(self.outdir, f'{name}.{fmt}'))]
        panels = self.available_panels()
        if not panels:
            raise RuntimeError('no panel has its data yet')
        return [self.write_figure(
                    os.path.join(self.outdir,
                                 f'{name}_{PANEL_SLUG[panel]}.{fmt}'),
                    panels=[panel])
                for panel in panels]

    def _on_save(self, _=None):
        """ """
        try:
            written = self.save_figures()
        except Exception as e:
            self._report_error(self.w_save_status, 'Save failed', e)
            return
        listing = '<br>'.join(written)
        self.w_save_status.value = (
            f"<span style='color:#27ae60'>Wrote {len(written)} file"
            f"{'s' if len(written) != 1 else ''}:<br>{listing}</span>")

    def _on_save_fits(self, _=None):
        """Write the stacked line map, its variance and exposure count to FITS."""
        if self.line_map is None:
            self.w_save_status.value = (
                "<span style='color:#c0392b'>No line map to write.</span>")
            return
        from astropy.io import fits
        name = self.w_filename.value.strip() or 'figure'
        path = os.path.join(self.outdir, f'{name}_linemap.fits')
        header = self.wcs.to_header()
        header['WAVELEN'] = (self.observed_wavelength(),
                             'Observed wavelength of the line map [nm]')
        header['RA_OBJ'] = (self.w_ra.value, 'Target RA [deg]')
        header['DEC_OBJ'] = (self.w_dec.value, 'Target Dec [deg]')
        if self.w_center_mode.value == 'Redshift':
            header['REDSHIFT'] = (self.w_redshift.value, 'Target redshift')
        elmap, elmap_var, elmap_norm, elmap_count = self.line_map
        try:
            fits.HDUList([
                fits.PrimaryHDU(np.asarray(elmap, dtype='f4'), header=header),
                fits.ImageHDU(np.asarray(elmap_var, dtype='f4'), header=header,
                              name='VARIANCE'),
                fits.ImageHDU(np.asarray(elmap_count, dtype='i4'),
                              header=header, name='COUNT'),
            ]).writeto(path, overwrite=True)
        except Exception as e:
            self._report_error(self.w_save_status, 'FITS write failed', e)
            return
        self.w_save_status.value = (
            f"<span style='color:#27ae60'>Wrote {os.path.abspath(path)}</span>")

    def _on_params_save(self, _=None):
        path = os.path.join(self.outdir, self.w_params_file.value)
        try:
            with open(path, 'w') as f:
                json.dump(self.params(), f, indent=2)
        except Exception as e:
            self._report_error(self.w_save_status, 'Could not write settings', e)
            return
        self.w_save_status.value = (
            f"<span style='color:#27ae60'>Settings written to "
            f"{os.path.abspath(path)}</span>")

    def apply_params(self, saved):
        """Load a settings dict as written by :meth:`params`.

        Unknown keys are ignored, so a file from an older version still loads,
        and a hand-edited file missing whole blocks leaves those settings at
        whatever they are now.
        """
        self._suspend = True
        try:
            by_description = {w.description: w for w in self._live_widgets()}
            for key, value in saved.items():
                w = by_description.get(key)
                if w is not None and not isinstance(w, widgets.Dropdown):
                    try:
                        w.value = value
                    except Exception:
                        pass
            for w, v in zip(self.w_scaling, saved.get('iyjh_scaling', [])):
                w.value = v
            for name in PANEL_ORDER:
                self.w_panels[name].value = name in saved.get('panels', [])
            for key, w in (('color_mode', self.w_color_mode),
                           ('count_cmap', self.w_count_cmap),
                           ('cmap', self.w_cmap)):
                if saved.get(key) in w.options:
                    w.value = saved[key]
            trace = saved.get('trace', {})
            for key, w in (('level', self.w_trace_level),
                           ('alpha', self.w_trace_alpha),
                           ('wave_step', self.w_wave_step)):
                if key in trace:
                    w.value = trace[key]
            if 'source_levels' in trace:
                self.w_source_lo.value, self.w_source_hi.value = \
                    trace['source_levels']
            if 'grism' in trace:
                self.w_trace_grism.value = trace['grism']
            if 'hue_range_nm' in trace:
                span = trace['hue_range_nm']
                self.w_fix_hue_range.value = span is not None
                if span:
                    self.w_hue_lo.value, self.w_hue_hi.value = span
            if trace.get('frame_index') is not None and self.frame_paths:
                self._sync_hue_frame_options(int(trace['frame_index']))
            target = saved.get('target', {})
            for key, w in (('ra', self.w_ra), ('dec', self.w_dec),
                           ('redshift', self.w_redshift),
                           ('observed', self.w_observed)):
                if key in target:
                    w.value = target[key]
            if target.get('rest_line') in LINE_LIST:
                self.w_rest_line.value = target['rest_line']
            if target.get('center_mode') in self.w_center_mode.options:
                self.w_center_mode.value = target['center_mode']

            data = saved.get('data', {})
            for key, w in (('field_pix', self.w_stamp_size),
                           ('survey', self.w_survey),
                           ('search_radius_deg', self.w_radius),
                           ('frame_filter', self.w_grism_filter),
                           ('one_per_pointing', self.w_one_per_pointing),
                           ('max_frames', self.w_max_frames),
                           ('continuum_size_pix', self.w_cs_size),
                           ('padx', self.w_padx), ('pady', self.w_pady),
                           ('ndrops', self.w_ndrops),
                           ('stack_width', self.w_stack_width)):
                if key in data:
                    w.value = data[key]
            if 'continuum' in data and \
                    data['continuum'] in [v for _, v in self.w_cs_method.options]:
                self.w_cs_method.value = data['continuum']

            export = saved.get('export', {})
            for key, w in (('name', self.w_filename), ('dpi', self.w_save_dpi),
                           ('tight', self.w_tight),
                           ('transparent', self.w_transparent),
                           ('separate_panels', self.w_separate)):
                if key in export:
                    w.value = export[key]
            if export.get('format') in self.w_format.options:
                self.w_format.value = export['format']
        finally:
            self._suspend = False
        self._on_cs_method({'new': self.w_cs_method.value})
        self._on_center_change()
        self.refresh()

    def _on_params_load(self, _=None):
        """ """
        path = os.path.join(self.outdir, self.w_params_file.value)
        try:
            with open(path) as f:
                saved = json.load(f)
        except Exception as e:
            self._report_error(self.w_save_status, 'Could not read settings', e)
            return
        self.apply_params(saved)
        self.w_save_status.value = (
            f"<span style='color:#27ae60'>Loaded {os.path.abspath(path)}</span>")

    def _ipython_display_(self):
        display(self.widget)


def _axes_extent(ax):
    """Current ``((xlo, xhi), (ylo, yhi))`` of an axes, ordered low to high.

    Annotations are placed from this rather than from an array shape, so they
    land correctly whatever the panel is showing.
    """
    xlo, xhi = sorted(ax.get_xlim())
    ylo, yhi = sorted(ax.get_ylim())
    return (xlo, xhi), (ylo, yhi)


def _halo(lw=2, color='k'):
    """Dark outline behind annotation text, so it reads over bright pixels."""
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw, foreground=color)]


def make_line_map_studio(targets_file='targets.txt',
                         credentials_file=DEFAULT_CREDENTIALS,
                         config_file=DEFAULT_CONFIG, calibdir=DEFAULT_CALIBDIR,
                         outdir='.', Euclid=None, G=None):
    """Build the studio GUI. See :class:`LineMapStudio` for the arguments."""
    return LineMapStudio(targets_file=targets_file,
                         credentials_file=credentials_file,
                         config_file=config_file, calibdir=calibdir,
                         outdir=outdir, Euclid=Euclid, G=G)

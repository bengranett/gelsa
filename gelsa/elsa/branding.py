"""ELSA branding for the notebook GUIs.

Styled after the ELSA website (https://elsa-euclid.github.io/): the reverse
(white on transparent) logo on the site's dark ``#1b1f22`` background, with
Source Sans Pro in uppercase, widely letter-spaced headings.

The images live in ``gelsa/logos/`` and are installed with the package as
package data. Two versions of the logo are there, both made from the ELSA artwork
``Elsa-BW-Transparent-reverse.png`` used on the website:

* ``elsa_logo_reverse_256.png`` -- the artwork as is, for dark backgrounds
  such as the GUI banner.
* ``elsa_badge_256.png`` -- the same laid on a black disc, which reads on light
  and dark notebook themes alike; the notebook headers use it.

``eu_funded_vertical_320.png`` is the official "Funded by the European Union"
emblem (``docs/_static/EN_fundedbyEU_VERTICAL_RGB_POS.png``) on a white
panel, so it too reads on either theme.
"""
import base64
import functools
from importlib import resources

import ipywidgets as widgets

NAME = 'ELSA'
FULL_NAME = 'Euclid Legacy Science Advanced analysis tools'
URL = 'https://elsa-euclid.github.io/'

# Wording of the gelsa docs and the ELSA website footer.
ACKNOWLEDGEMENT = (
    '"ELSA: Euclid Legacy Science Advanced analysis tools" (Grant Agreement '
    'no. 101135203) is funded by the European Union. Views and opinions '
    'expressed are however those of the author(s) only and do not necessarily '
    'reflect those of the European Union or Innovate UK. Neither the European '
    'Union nor the granting authority can be held responsible for them. UK '
    'participation is funded through the UK Horizon guarantee scheme under '
    'Innovate UK grant 10093177.')

BACKGROUND = '#1b1f22'
FONT = "'Source Sans Pro', 'Source Sans 3', 'Helvetica Neue', Arial, sans-serif"


@functools.lru_cache(maxsize=None)
def asset_png(name):
    """PNG bytes of one of the files in ``gelsa/logos/``, or None if missing.

    Missing only in an install made before the logos were packaged; the
    banner and the acknowledgement are then drawn without the images.
    """
    try:
        return resources.files('gelsa').joinpath(f'logos/{name}').read_bytes()
    except OSError:
        return None


def _img(name, width, height=None, style='', alt=''):
    """An ``<img>`` with the file inlined as a data URI, or '' if it is missing."""
    png = asset_png(name)
    if png is None:
        return ''
    size = f'width="{width}"' + (f' height="{height}"' if height else '')
    return (f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}" '
            f'{size} style="{style}" alt="{alt}">')


def header(title, subtitle='', logo_px=72):
    """A banner in the ELSA website style, to sit above a GUI.

    The logo is inlined as a data URI so the banner needs no file server and
    survives being saved with the notebook's widget state.
    """
    logo = _img('elsa_logo_reverse_256.png', logo_px, logo_px,
                style='display:block', alt=f'{NAME} logo')
    sub = (f"<div style='margin-top:6px; font-weight:300; "
           f"color:rgba(255,255,255,0.7)'>{subtitle}</div>" if subtitle else '')
    html = f"""
<div style="display:flex; align-items:center; gap:18px; padding:12px 18px;
            background:{BACKGROUND}; color:#ffffff; font-family:{FONT};
            border-radius:4px;">
  <a href="{URL}" target="_blank" title="{NAME}: {FULL_NAME}">{logo}</a>
  <div style="border-left:1px solid #ffffff; padding-left:18px;">
    <div style="font-size:1.6em; font-weight:600; text-transform:uppercase;
                letter-spacing:0.3em; line-height:1.2">{title}</div>
    <div style="margin-top:4px; font-size:0.8em; font-weight:300;
                text-transform:uppercase; letter-spacing:0.2em;
                color:rgba(255,255,255,0.7)">{NAME} &middot; {FULL_NAME}</div>
    {sub}
  </div>
</div>"""
    return widgets.HTML(html, layout=widgets.Layout(margin='0 0 8px 0'))


def acknowledgements(logo_px=150):
    """The EU emblem beside the ELSA funding acknowledgement."""
    logo = _img('eu_funded_vertical_320.png', logo_px, style='flex:0 0 auto',
                alt='Funded by the European Union')
    html = f"""
<div style="display:flex; align-items:flex-start; gap:20px; max-width:900px;">
  {logo}
  <div style="line-height:1.5">
    <p><b>Please include the acknowledgement for the ELSA project below in all
    publications.</b></p>
    <p>{ACKNOWLEDGEMENT}</p>
    <p><a href="{URL}" target="_blank">{URL}</a></p>
  </div>
</div>"""
    return widgets.HTML(html)

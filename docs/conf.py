# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from datetime import timezone, datetime
from importlib.metadata import version as _version, PackageNotFoundError
from pathlib import Path

sys.path.insert(0, str(Path('..',).resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Gelsa'
author = 'Ben Granett and the GELSA contributors'
copyright = f"2024–{datetime.now(tz=timezone.utc).year}, " + author

# The version is read from the installed package so it follows pyproject.toml.
try:
    release = _version('gelsa')
except PackageNotFoundError:
    release = 'unknown'
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_design'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The API reference under api/ is generated from the docstrings at build time.
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
# Dependencies of the optional ``esa`` extra, mocked so the API pages build
# without them.
autodoc_mock_imports = ['azulero', 'ipywidgets', 'IPython', 'astroquery']
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_css_files = ["mycss.css"]

html_logo = "_static/elsa-color-300.png"

html_theme_options = {
    "icon_links": [

        {
            "name": "GitHub",
            "url": "https://github.com/elsa-euclid/gelsa",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "ELSA",
            "url": "https://elsa-euclid.github.io/",
            "icon": "_static/elsa-color-icon.png",
            "type": "local",
        },
    ],
}

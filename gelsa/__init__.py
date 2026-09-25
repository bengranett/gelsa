import logging

from . import version
from .gelsa import Gelsa

__all__ = ['Gelsa', 'enable_logging']
__version__ = version.__version__


def enable_logging(level=logging.INFO):
    """Show gelsa log messages at level and above on stderr, e.g. in a notebook.

    By default Python only shows warnings and errors. Calling this again
    changes the level without adding another handler.
    """
    logger = logging.getLogger(__name__)
    if not any(getattr(h, '_gelsa_handler', False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s %(name)s: %(message)s'))
        handler._gelsa_handler = True
        logger.addHandler(handler)
    logger.setLevel(level)

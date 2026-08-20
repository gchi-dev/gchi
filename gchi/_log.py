"""
_log.py shared logger for the package   

default level is warning, so progress messages are silent unless turned on,
while anything that affects results (a fallback, a skipped metric, an
error) always prints. 

turn progress messages on with: gchi.set_verbose(True)
or pass verbose=True directly to calculate_all() / prepare_inputs().
"""

import logging

logger = logging.getLogger("gchi")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)

_stream_handler_attached = False


def set_verbose(verbose=True):
    """
    turn progress messages on or off. warnings and errors always show
    regardless of this setting.
    """
    global _stream_handler_attached
    if verbose and not _stream_handler_attached:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        _stream_handler_attached = True
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

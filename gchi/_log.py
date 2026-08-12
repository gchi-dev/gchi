"""
_log.py -- shared logger for the whole package.

standard library logging, standard library conventions:
  - progress messages ("calculating FWI...", "processing year 2020...")  -> logger.info()
  - fallback/assumption notices ("no mask file provided", "guessed units") -> logger.warning()
  - real errors (caught exceptions in calculate_all)                     -> logger.error()

default level is WARNING, so progress messages are silent unless turned on,
while anything that affects your results (a fallback, a skipped metric, an
error) always prints. turn progress messages on with:

    gchi.set_verbose(True)

or pass verbose=True directly to calculate_all() / prepare_inputs().

a NullHandler is attached by default (library convention -- doesn't print
anything on its own). the first time set_verbose() is called, a simple
StreamHandler is attached so INFO messages actually show up somewhere
without requiring the user to configure logging themselves.
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

"""
gchi — Global Climate Hazard Index
"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module=r"dask\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"xesmf\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")

from ._log import logger, set_verbose

from .inputs import (
    prepare_inputs,
    calculate_base_period_percentiles,
    show_expected_ds_format,
    help,
)

from .thresholds import severity_thresholds, fwi_thresholds

# heat stress
from .heat import (
    AT, at_values,
    HI, hi_values,
    Hu, hu_values,
    WBT, wbt_values,
    WBGT, wbgt_values,
    UTCIhot, utci_hot_values,
    HWF,
    TXC, txc_values,
    TR, tr_values,
)

# cold
from .cold import (
    UTCIcold, utci_cold_values,
    TNXp,
)

# fire
from .fire import (
    FI, fi_values,
    HDW, hdw_values,
    FWI, fwi_values,
)

# air quality
from .aq import (
    O3, o3_values,
    PM2pt5, pm25_values,
)

# drought
from .drought import (
    DSD, dsd_values,
    SPI,
    SMSXp,
)

# disease
from .disease import (
    VSmalaria,
    VSzika,
    VSdengueAeg,
    VSdengueAlb,
    VbrS,
)

# weather
from .weather import (
    PRXmm,
    PR1day,
    PR5day,
    pr_values,
)

from .calculate_all import calculate_all, GCHIResults

from .composites import category_averages, composite_average, DEFAULT_CATEGORIES

from ._core import _is_prepared

# ---------------------------------------------------------------------------
# auto-prep on direct metric calls
#
# if a user calls e.g. gchi.FI(ds_dict) straight on raw data, run
# prepare_inputs() first with default settings, so they don't have to
# remember to call it themselves. if the data already went through
# prepare_inputs (checked via the gchi_prepared attr set there), this is
# skipped -- so it never preps twice.
#
# note: this only wraps the top-level gchi.<metric>() calls exported below.
# calculate_all() preps ds_dict once itself and then calls the underlying
# metric functions directly (imported from their own modules), so it never
# goes through this path and never re-preps per metric.
# ---------------------------------------------------------------------------

def _auto_prep_wrapper(fn):
    def wrapped(ds_dict, *args, **kwargs):
        if not _is_prepared(ds_dict):
            logger.info(f"{fn.__name__}: input not prepped -- running prepare_inputs() with default settings...")
            ds_dict = prepare_inputs(ds_dict)
        return fn(ds_dict, *args, **kwargs)
    return wrapped

_auto_prep_metrics = [
    "AT", "HI", "Hu", "WBT", "WBGT", "UTCIhot", "HWF", "TXC", "TR",
    "UTCIcold", "TNXp",
    "FI", "HDW", "FWI",
    "O3", "PM2pt5",
    "DSD", "SPI", "SMSXp",
    "VSmalaria", "VSzika", "VSdengueAeg", "VSdengueAlb", "VbrS",
    "PRXmm", "PR1day", "PR5day",
]

for _name in _auto_prep_metrics:
    globals()[_name] = _auto_prep_wrapper(globals()[_name])
del _name

# ---------------------------------------------------------------------------
# auto-prep for the base period too
#
# calculate_base_period_percentiles() is the equivalent entry point for your
# historical/base dataset -- it takes raw arrays (tas=, tasmin=, pr=, mrsos=)
# and turns them into the percentile dict that HWF/TNXp/SPI/SMSXp/PR1day/PR5day
# need. those functions only ever see base_dict *after* percentiles are already
# computed, so prepping has to happen here, before that, not at the metric call.
#
# same rule as above: only preps arrays that aren't already prepped, so calling
# this twice, or calling it after you've prepped things yourself, does nothing
# extra.
# ---------------------------------------------------------------------------

def _auto_prep_wrapper_base(fn):
    def wrapped(*args, **kwargs):
        raw_vars = {k: v for k, v in kwargs.items() if hasattr(v, "attrs")}
        if raw_vars and not _is_prepared(raw_vars):
            logger.info(f"{fn.__name__}: base period inputs not prepped -- running prepare_inputs() with default settings...")
            prepped = prepare_inputs(raw_vars)
            kwargs.update(prepped)
        return fn(*args, **kwargs)
    return wrapped

calculate_base_period_percentiles = _auto_prep_wrapper_base(calculate_base_period_percentiles)

__version__ = "0.0.0"

__all__ = [
    # inputs
    "prepare_inputs",
    "calculate_base_period_percentiles",
    "show_expected_ds_format",
    "help",
    "severity_thresholds",
    "fwi_thresholds",
    # heat
    "AT", "at_values",
    "HI", "hi_values",
    "Hu", "hu_values",
    "WBT", "wbt_values",
    "WBGT", "wbgt_values",
    "UTCIhot", "utci_hot_values",
    "HWF",
    "TXC", "txc_values",
    "TR", "tr_values",
    # cold
    "UTCIcold", "utci_cold_values",
    "TNXp",
    # fire
    "FI", "fi_values",
    "HDW", "hdw_values",
    "FWI", "fwi_values",
    # aq
    "O3", "o3_values",
    "PM2pt5", "pm25_values",
    # drought
    "DSD", "dsd_values",
    "SPI",
    "SMSXp",
    # disease
    "VSmalaria",
    "VSzika",
    "VSdengueAeg",
    "VSdengueAlb",
    "VbrS",
    # weather
    "PRXmm",
    "PR1day",
    "PR5day",
    "pr_values",
    # all
    "calculate_all",
    "GCHIResults",
    "set_verbose",
    "category_averages",
    "composite_average",
    "DEFAULT_CATEGORIES",
]

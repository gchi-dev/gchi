"""
gchi — Global Climate Hazard Index
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    CDD, cdd_values,
    SPI,
    SPEI,
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

from .calculate_all import calculate_all

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
    "CDD", "cdd_values",
    "SPI",
    "SPEI",
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
]

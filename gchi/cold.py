"""
cold extremes: TNXp, UTCIcold
"""

import numpy as np
import xarray as xr

from ._core import (
    _check_and_convert_units, _annual_exceedance_frac, _assign_severity_level, _get_tsteps, _ann_frac, _nan_mask, _add_metric_metadata
)
from .heat import _utci_values
from .thresholds import severity_thresholds as _default_thresholds
from ._log import logger


def utci_cold_values(ds_dict, hum_var='both'):
    """Raw UTCIcold values (°C)."""
    return _utci_values(ds_dict, hum_var=hum_var, hotorcold='cold')


def UTCIcold(ds_dict, hum_var='both', severity_thresholds=None):
    """
    UTCI cold stress exceedance levels (days below threshold).
    Call utci_cold_values() to get raw UTCI without level assignment.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["UTCIcold"]
    UTCI = _utci_values(ds_dict, hum_var=hum_var, hotorcold='cold')
    UTCI_levels = _annual_exceedance_frac(UTCI, severity_thresholds=severity_thresholds, var_name="UTCIcold", exceedance_dir="below")
    result = _assign_severity_level(UTCI_levels)
    return _add_metric_metadata(result, "UTCIcold", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year")


def TNXp(ds_dict, base_dict, severity_thresholds=None, temp_max=15):
    """
    Days where daily min temperature < Xth percentile of base period tasmin. Tmax must be < X °C (15°C Default)
    Thresholds are spatially varying (per grid cell) so uses bespoke exceedance method.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["TNXp"]

    TN = _check_and_convert_units(da=ds_dict["tasmin"], input_var="tasmin", conv_type="C")
    nan_mask = _nan_mask(TN)
    # compute steps_per_year from full calendar days before temp_max masking
    # -- if done after, tropical cells with almost no days < 15C get a tiny
    # denominator and their fraction inflates to ~1.0
    steps_per_year = _get_tsteps(TN)
    # mask used to identify cells that are always too warm (never cold enough)
    # these will be set to 0 fraction rather than NaN
    always_warm_mask = (TN < temp_max).resample(time="1YE").sum() == 0
    TN = TN.where(TN < temp_max)

    tasmin_base_percentile_vals = [
        float(k.split("_")[1].replace("pt", ".").replace("p", ""))
        for k in base_dict.keys() if k.startswith('tasmin_') and k.endswith('p')
    ]
    tasmin_base_percentile_keys = sorted(
        [k for k in base_dict.keys() if k.startswith('tasmin_') and k.endswith('p')],
        key=lambda k: float(k.replace('tasmin_', '').replace('pt', '.').replace('p', '')),
        reverse=True,
    )

    if sorted(tasmin_base_percentile_vals) != sorted(severity_thresholds):
        logger.warning(
            f"cannot calculate TNXp — base period tasmin percentiles don't match severity_thresholds.\n"
            f"base period: {tasmin_base_percentile_vals}. thresholds: {severity_thresholds}. skipping..."
        )
        return None

    n = len(tasmin_base_percentile_keys)

    da_list = []
    for key in tasmin_base_percentile_keys:
        th = base_dict[key]
        da_count = (TN < th).resample(time='1YE').sum(dim='time', skipna=True)
        da_list.append(da_count)

    # keys sorted descending (10th first), coords ascending: 10th->1, 0.5th->4
    # level 1 = least extreme cold, level 4 = most extreme cold
    TNXp_val = xr.concat(da_list, dim='level')
    TNXp_val = TNXp_val.assign_coords(level=np.arange(1, n + 1))
    TNXp_val = _ann_frac(TNXp_val, steps_per_year).rename("TNXp")

    # level N assigned if exceedance fraction > p/100
    # i.e. more cold days than the p% historically expected
    # iterate lowest->highest so most extreme (0.5th pct) overwrites
    severity_level = xr.zeros_like(TNXp_val.isel(level=0).drop_vars("level"), dtype=int)
    for i, p in enumerate(tasmin_base_percentile_vals):
        frac_thresh = p / 100  # 10th pct -> 0.10, 0.5th pct -> 0.005
        level_coord = i + 1    # 10th->1, 5th->2, 2nd->3, 0.5th->4
        exceeds = TNXp_val.sel(level=level_coord) > frac_thresh
        severity_level = severity_level.where(~exceeds, other=level_coord)

    severity_level = severity_level.where(TNXp_val.notnull().any("level"))
    severity_level.name = "TNXp_severity_level"
    TNXp_val = xr.merge([TNXp_val, severity_level], compat="override")

    # cells where it never gets cold enough (always_warm_mask) -> 0, not NaN
    TNXp_val = TNXp_val.where(~always_warm_mask, other=0)
    # cells that are genuinely missing data -> NaN
    TNXp_val = TNXp_val.where(~nan_mask)
    TNXp_val.attrs['level_thresholds'] = [
        {"level": int(i + 1), "threshold_value": tasmin_base_percentile_vals[i],
         "unit": "percentile", "source": "base period distribution"}
        for i in range(len(tasmin_base_percentile_keys))
    ]
    return _add_metric_metadata(TNXp_val, "TNXp", ds_dict, units="fraction of year", notes=f"days below base period tasmin percentile thresholds with tasmax < {temp_max} degC. spatially varying thresholds.")

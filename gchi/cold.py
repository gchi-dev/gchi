"""
cold extremes: TNXp, UTCIcold
"""

import numpy as np
import xarray as xr

from ._core import (
    _check_and_convert_units, _annual_exceedance_frac, _assign_hazard_level, _get_tsteps, _ann_frac, _nan_mask
)
from .heat import _utci_values
from .thresholds import hazard_thresholds as _default_thresholds


def utci_cold_values(ds_dict, hum_var='both'):
    """Raw UTCIcold values (°C)."""
    return _utci_values(ds_dict, hum_var=hum_var, hotorcold='cold')


def UTCIcold(ds_dict, hum_var='both', hazard_thresholds=None):
    """
    UTCI cold stress exceedance levels (days below threshold).
    Call utci_cold_values() to get raw UTCI without level assignment.
    """
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["UTCIcold"]
    UTCI = _utci_values(ds_dict, hum_var=hum_var, hotorcold='cold')
    UTCI_levels = _annual_exceedance_frac(UTCI, hazard_thresholds=hazard_thresholds, var_name="UTCIcold", exceedance_dir="below")
    return _assign_hazard_level(UTCI_levels)


def TNXp(ds_dict, base_dict, hazard_thresholds=None, temp_max=15):
    """
    Days where daily min temperature < Xth percentile of base period tasmin. Tmax must be < X °C (15°C Default)
    Thresholds are spatially varying (per grid cell) so uses bespoke exceedance method.
    """
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["TNXp"]

    TN = _check_and_convert_units(da=ds_dict["tasmin"], input_var="tasmin", conv_type="C")
    nan_mask = _nan_mask(TN)
    TN = TN.where(TN < temp_max)
    steps_per_year = _get_tsteps(TN)

    tasmin_base_percentile_vals = [
        float(k.split("_")[1].replace("pt", ".").replace("p", ""))
        for k in base_dict.keys() if k.startswith('tasmin_') and k.endswith('p')
    ]
    tasmin_base_percentile_keys = sorted(
        [k for k in base_dict.keys() if k.startswith('tasmin_') and k.endswith('p')],
        key=lambda k: float(k.replace('tasmin_', '').replace('pt', '.').replace('p', '')),
        reverse=True,
    )

    if sorted(tasmin_base_percentile_vals) != sorted(hazard_thresholds):
        print(
            f"cannot calculate TNXp — base period tasmin percentiles don't match hazard_thresholds.\n"
            f"base period: {tasmin_base_percentile_vals}. thresholds: {hazard_thresholds}. skipping..."
        )
        return None

    da_list = []
    for key in tasmin_base_percentile_keys:
        th = base_dict[key]
        da_count = (TN < th).resample(time='1YE').sum(dim='time', skipna=True)
        da_list.append(da_count)

    TNXp_val = xr.concat(da_list, dim='level')
    TNXp_val = TNXp_val.assign_coords(level=np.arange(1, len(tasmin_base_percentile_keys) + 1))
    TNXp_val.attrs['level_values'] = tasmin_base_percentile_keys
    TNXp_val = _ann_frac(TNXp_val, steps_per_year).rename("TNXp")
    TNXp_val = _assign_hazard_level(TNXp_val)
    TNXp_val = TNXp_val.where(~nan_mask)
    TNXp_val.attrs['level_thresholds'] = [
        {"level": int(i + 1), "threshold_value": tasmin_base_percentile_vals[i],
         "unit": "percentile", "source": "base period distribution"}
        for i in range(len(tasmin_base_percentile_keys))
    ]
    return TNXp_val

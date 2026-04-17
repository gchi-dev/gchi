"""
extreme weather metrics: PRXmm, PR1day, PR5day

all work well chunked spatially.
PR1day and PR5day use spatially varying thresholds from base_dict.
"""

import numpy as np
import xarray as xr

from ._core import (
    _check_and_convert_units, _annual_exceedance_frac, _assign_hazard_level,
    _get_tsteps, _ann_frac, _nan_mask
)
from .thresholds import hazard_thresholds as _default_thresholds


def pr_values(ds_dict):
    """daily precipitation (mm day-1) — shared starting point for all precip metrics"""
    return _check_and_convert_units(da=ds_dict['pr'], input_var="pr", conv_type="mm day-1")


def PRXmm(ds_dict, hazard_thresholds=None):
    """
    Fraction of year where daily precipitation > X mm.
    Default thresholds: 20, 30, 40, 50 mm.
    """
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["PRXmm"]
    PR = pr_values(ds_dict)
    PRXmm_levels = _annual_exceedance_frac(PR, hazard_thresholds=hazard_thresholds, var_name="PRXmm")
    return _assign_hazard_level(PRXmm_levels)


def PR1day(ds_dict, base_dict, hazard_thresholds=None):
    """
    Fraction of year where daily rainfall > Xth percentile of the base period distribution.
    Minimum 1 mm/day to count (wet day condition).
    Thresholds are spatially varying (per grid cell from base_dict).

    base_dict needs 'pr_{p}p' keys matching hazard_thresholds percentile values.
    """
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["PR1day"]

    PR = pr_values(ds_dict).chunk({"lat": -1, "lon": -1})
    steps_per_year = _get_tsteps(PR)
    nanmask = _nan_mask(PR)
    PR = PR.where(PR > 1)  # wet day condition

    pr_base_percentile_vals = [
        float(k.split("_")[1].replace("pt", ".").replace("p", ""))
        for k in base_dict.keys() if k.startswith('pr_') and k.endswith('p')
    ]
    pr_base_percentile_keys = sorted(
        [k for k in base_dict.keys() if k.startswith('pr_') and k.endswith('p')],
        key=lambda k: float(k.replace('pr_', '').replace('pt', '.').replace('p', '')),
        reverse=True,
    )

    if sorted(pr_base_percentile_vals) != sorted(hazard_thresholds):
        print(
            f"cannot calculate PR1day — base period pr percentiles don't match hazard_thresholds.\n"
            f"base period: {pr_base_percentile_vals}. thresholds: {hazard_thresholds}. skipping..."
        )
        return None

    da_list = []
    for key in pr_base_percentile_keys:
        th = base_dict[key].chunk({"lat": -1, "lon": -1})
        da_count = (PR > th).resample(time='1YE').sum(dim='time', skipna=True)
        da_list.append(da_count)

    PR1day_val = xr.concat(da_list, dim='level')
    PR1day_val = PR1day_val.assign_coords(level=np.arange(1, len(pr_base_percentile_keys) + 1))
    PR1day_val.attrs['level_values'] = pr_base_percentile_keys
    PR1day_val = _ann_frac(PR1day_val, steps_per_year).rename("PR1day")
    PR1day_val = _assign_hazard_level(PR1day_val)
    PR1day_val = PR1day_val.where(~nanmask)
    PR1day_val.attrs['level_thresholds'] = [
        {"level": int(i + 1), "threshold_value": pr_base_percentile_vals[i],
         "unit": "percentile", "source": "base period distribution"}
        for i in range(len(pr_base_percentile_keys))
    ]
    return PR1day_val


def PR5day(ds_dict, base_dict, hazard_thresholds=None):
    """
    Fraction of year where days are part of a 5-day precipitation event exceeding
    the Xth percentile of the base period annual maximum 5-day precipitation.

    All days within the 5-day window count, not just the center day.
    Thresholds are spatially varying (per grid cell from base_dict).

    base_dict needs 'rx5day_{p}p' keys matching hazard_thresholds percentile values.
    """
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["PR5day"]

    PR = pr_values(ds_dict).chunk({'lat': -1, 'lon': -1})
    steps_per_year = _get_tsteps(PR)
    nanmask = _nan_mask(PR)

    rx5day_base_percentile_vals = [
        float(k.split("_")[1].replace("pt", ".").replace("p", ""))
        for k in base_dict.keys() if k.startswith('rx5day_') and k.endswith('p')
    ]
    rx5day_base_percentile_keys = sorted(
        [k for k in base_dict.keys() if k.startswith('rx5day_') and k.endswith('p')],
        key=lambda k: float(k.replace('rx5day_', '').replace('pt', '.').replace('p', '')),
        reverse=True,
    )

    if sorted(rx5day_base_percentile_vals) != sorted(hazard_thresholds):
        print(
            f"cannot calculate PR5day — base period rx5day percentiles don't match hazard_thresholds.\n"
            f"base period: {rx5day_base_percentile_vals}. thresholds: {hazard_thresholds}. skipping..."
        )
        return None

    window = 5
    half_window = window // 2
    rolling_sum = PR.rolling(time=window, center=True).sum()

    da_list = []
    for key in rx5day_base_percentile_keys:
        th = base_dict[key].chunk({'lat': -1, 'lon': -1})

        # identify 5-day windows that exceed the threshold
        mask_center = rolling_sum > th

        # expand so all 5 days in the event count, not just the center
        mask_expanded = xr.zeros_like(PR, dtype=bool)
        for shift in range(-half_window, half_window + 1):
            mask_expanded |= mask_center.shift(time=shift, fill_value=False)

        da_count = PR.where(mask_expanded).resample(time="1YE").count()
        da_list.append(da_count)

    PR5day_val = xr.concat(da_list, dim='level')
    PR5day_val = PR5day_val.assign_coords(level=np.arange(1, len(rx5day_base_percentile_keys) + 1))
    PR5day_val.attrs['level_values'] = rx5day_base_percentile_keys
    PR5day_val = _ann_frac(PR5day_val, steps_per_year).rename("PR5day")
    PR5day_val = _assign_hazard_level(PR5day_val)
    PR5day_val = PR5day_val.where(~nanmask)
    PR5day_val.attrs['level_thresholds'] = [
        {"level": int(i + 1), "threshold_value": rx5day_base_percentile_vals[i],
         "unit": "percentile", "source": "base period distribution"}
        for i in range(len(rx5day_base_percentile_keys))
    ]
    return PR5day_val

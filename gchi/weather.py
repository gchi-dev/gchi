"""
extreme weather metrics: PRXmm, PR1day, PR5day

all work well chunked spatially.
PR1day and PR5day use spatially varying thresholds from base_dict.
"""

import numpy as np
import xarray as xr

from ._core import (
    _check_and_convert_units, _annual_exceedance_frac, _assign_severity_level,
    _get_tsteps, _ann_frac, _nan_mask, _add_metric_metadata
)
from .thresholds import severity_thresholds as _default_thresholds


def pr_values(ds_dict):
    """daily precipitation (mm day-1) — shared starting point for all precip metrics"""
    return _check_and_convert_units(da=ds_dict['pr'], input_var="pr", conv_type="mm day-1")


def PRXmm(ds_dict, severity_thresholds=None):
    """
    Fraction of year where daily precipitation > X mm.
    Default thresholds: 20, 30, 40, 50 mm.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["PRXmm"]
    PR = pr_values(ds_dict)
    PRXmm_levels = _annual_exceedance_frac(PR, severity_thresholds=severity_thresholds, var_name="PRXmm")
    result = _assign_severity_level(PRXmm_levels)
    return _add_metric_metadata(result, "PRXmm", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year")


def PR1day(ds_dict, base_dict, percentile_thresholds=None):
    """
    Fraction of days per year where daily precipitation exceeds local percentile
    thresholds derived from the base period distribution of all days -- at the
    90th, 95th, 98th, and 99.5th percentiles by default.
    Level N assigned if exceedance fraction > (100-p)/100 of all days:
      level 1: > 10% of year. level 2: > 5%. level 3: > 2%. level 4: > 0.5%.

    Parameters
    ----------
    percentile_thresholds : list of float, optional
        Percentile thresholds to use. Must match the percentiles used when
        calculate_base_period_percentiles() was called. Default: [90, 95, 98, 99.5].

    base_dict needs 'pr_{p}p' keys from calculate_base_period_percentiles().
    """
    if percentile_thresholds is None:
        percentile_thresholds = _default_thresholds["PR1day"]
    PR = pr_values(ds_dict).chunk({"lat": -1, "lon": -1})
    nanmask = _nan_mask(PR)
    # get calendar steps before wet day masking -- needed for correct denominator
    steps_per_year = _get_tsteps(PR)
    PR = PR.where(PR > 1)  # wet day condition

    # sort descending (99.5th first) -- level 1 = most extreme (99.5th pct)
    # level 4 = least extreme but still above normal (90th pct)
    pr_base_percentile_keys = sorted(
        [k for k in base_dict.keys() if k.startswith('pr_') and k.endswith('p')],
        key=lambda k: float(k.replace('pr_', '').replace('pt', '.').replace('p', '')),
        reverse=True,
    )

    if not pr_base_percentile_keys:
        print("cannot calculate PR1day -- no pr percentile keys found in base_dict. skipping...")
        return None

    pr_base_percentile_vals = [
        float(k.replace('pr_', '').replace('pt', '.').replace('p', ''))
        for k in pr_base_percentile_keys
    ]

    # validate that user thresholds match what was computed in base_dict
    if sorted(percentile_thresholds) != sorted(pr_base_percentile_vals):
        print(
            f"cannot calculate PR1day -- percentile_thresholds {sorted(percentile_thresholds)} "
            f"don't match base_dict percentiles {sorted(pr_base_percentile_vals)}. "
            f"rerun calculate_base_period_percentiles() with matching percentiles, or "
            f"don't pass percentile_thresholds to use the base_dict defaults. skipping..."
        )
        return None

    # filter keys to only those matching requested thresholds (in case base_dict has extras)
    pr_base_percentile_keys = [
        k for k in pr_base_percentile_keys
        if float(k.replace('pr_', '').replace('pt', '.').replace('p', '')) in percentile_thresholds
    ]
    pr_base_percentile_vals = [
        float(k.replace('pr_', '').replace('pt', '.').replace('p', ''))
        for k in pr_base_percentile_keys
    ]

    # frac_thresholds match level order:
    # level 1 (99.5th pct): > 0.5% of year  -- rare but very extreme
    # level 4 (90th pct):   > 10% of year   -- more common, less extreme
    frac_thresholds = [(100 - p) / 100 for p in pr_base_percentile_vals]

    da_list = []
    for key in pr_base_percentile_keys:
        th = base_dict[key].chunk({"lat": -1, "lon": -1})
        da_count = (PR > th).resample(time='1YE').sum(dim='time', skipna=True)
        da_list.append(da_count)

    n = len(pr_base_percentile_keys)
    # keys sorted descending (99.5th first), so reassign coords:
    # 99.5th pct → level 4 (most extreme), 90th pct → level 1 (least extreme)
    PR1day_val = xr.concat(da_list, dim='level')
    PR1day_val = PR1day_val.assign_coords(level=np.arange(n, 0, -1))  # [4,3,2,1]
    PR1day_val = (PR1day_val / steps_per_year).rename("PR1day")

    # level assignment: year gets level N if it exceeds the Nth percentile threshold
    # more than (100-p)/100 of the year. highest level exceeded wins.
    severity_level = xr.zeros_like(PR1day_val.isel(level=0).drop_vars("level"), dtype=int)
    for i, p in enumerate(pr_base_percentile_vals):
        frac_thresh = (100 - p) / 100
        level_coord = n - i  # 99.5th→4, 98th→3, 95th→2, 90th→1
        exceeds = PR1day_val.sel(level=level_coord) > frac_thresh
        severity_level = severity_level.where(~exceeds, other=level_coord)

    severity_level = severity_level.where(PR1day_val.notnull().any("level"))
    severity_level.name = "PR1day_severity_level"
    PR1day_val = xr.merge([PR1day_val, severity_level], compat="override")
    PR1day_val = PR1day_val.sortby("level")  # sort ascending so level 1=90th, level 4=99.5th
    PR1day_val = PR1day_val.where(~nanmask)
    PR1day_val = _add_metric_metadata(PR1day_val, "PR1day", ds_dict, severity_thresholds=percentile_thresholds, units="fraction of year", notes="percentile thresholds from base period all-day distribution. level N: exceedance > (100-p)/100 of year.")
    PR1day_val.attrs['level_thresholds'] = [
        {"level": n - i, "percentile": pr_base_percentile_vals[i],
         "exceedance_frac_threshold": (100 - pr_base_percentile_vals[i]) / 100,
         "unit": "percentile", "source": "base period distribution"}
        for i in range(n)
    ]
    return PR1day_val


def PR5day(ds_dict, base_dict, percentile_thresholds=None):
    """
    Fraction of year where days are part of a 5-day precipitation event exceeding
    the Xth percentile of base period 5-day rolling sums (wet windows only, > 5mm).

    All days within the 5-day window count, not just the center day.
    Level assignment identical to PR1day: level N requires > (100-p)/100 fraction of year.
    e.g. level 1 (90th pct): > 10% of year. level 4 (99.5th pct): > 0.5% of year.

    Parameters
    ----------
    percentile_thresholds : list of float, optional
        Percentile thresholds to use. Must match the percentiles used when
        calculate_base_period_percentiles() was called. Default: [90, 95, 98, 99.5].

    base_dict needs 'pr5day_{p}p' keys from calculate_base_period_percentiles().
    """
    if percentile_thresholds is None:
        percentile_thresholds = _default_thresholds["PR5day"]
    PR = pr_values(ds_dict).chunk({'lat': -1, 'lon': -1})
    steps_per_year = _get_tsteps(PR)
    nanmask = _nan_mask(PR)

    # sort descending (99.5th first) -- level 1 = most extreme
    pr5day_base_percentile_keys = sorted(
        [k for k in base_dict.keys() if k.startswith('pr5day_') and k.endswith('p')],
        key=lambda k: float(k.replace('pr5day_', '').replace('pt', '.').replace('p', '')),
        reverse=True,
    )

    if not pr5day_base_percentile_keys:
        print("cannot calculate PR5day -- no pr5day percentile keys found in base_dict. skipping...")
        return None

    pr5day_base_percentile_vals = [
        float(k.replace('pr5day_', '').replace('pt', '.').replace('p', ''))
        for k in pr5day_base_percentile_keys
    ]

    if sorted(percentile_thresholds) != sorted(pr5day_base_percentile_vals):
        print(
            f"cannot calculate PR5day -- percentile_thresholds {sorted(percentile_thresholds)} "
            f"don't match base_dict percentiles {sorted(pr5day_base_percentile_vals)}. "
            f"rerun calculate_base_period_percentiles() with matching percentiles, or "
            f"don't pass percentile_thresholds to use the base_dict defaults. skipping..."
        )
        return None

    pr5day_base_percentile_keys = [
        k for k in pr5day_base_percentile_keys
        if float(k.replace('pr5day_', '').replace('pt', '.').replace('p', '')) in percentile_thresholds
    ]
    pr5day_base_percentile_vals = [
        float(k.replace('pr5day_', '').replace('pt', '.').replace('p', ''))
        for k in pr5day_base_percentile_keys
    ]

    window = 5
    half_window = window // 2
    # only count wet windows -- consistent with how pr5day percentiles were computed
    rolling_sum = PR.rolling(time=window, center=True).sum().where(lambda x: x > 5)  # approx 1mm/day avg

    da_list = []
    for key in pr5day_base_percentile_keys:
        th = base_dict[key].chunk({'lat': -1, 'lon': -1})

        # identify 5-day windows that exceed the threshold
        mask_center = rolling_sum > th

        # expand so all 5 days in the event count, not just the center
        mask_expanded = xr.zeros_like(PR, dtype=bool)
        for shift in range(-half_window, half_window + 1):
            shifted = mask_center.shift(time=shift, fill_value=False)
            mask_expanded = xr.where(shifted, True, mask_expanded).astype(bool)

        da_count = PR.where(mask_expanded).resample(time="1YE").count()
        da_list.append(da_count)

    n = len(pr5day_base_percentile_keys)
    PR5day_val = xr.concat(da_list, dim='level')
    PR5day_val = PR5day_val.assign_coords(level=np.arange(n, 0, -1))  # [4,3,2,1]
    PR5day_val = (PR5day_val / steps_per_year).rename("PR5day")

    severity_level = xr.zeros_like(PR5day_val.isel(level=0).drop_vars("level"), dtype=int)
    for i, p in enumerate(pr5day_base_percentile_vals):
        frac_thresh = (100 - p) / 100
        level_coord = n - i  # 99.5th→4, 98th→3, 95th→2, 90th→1
        exceeds = PR5day_val.sel(level=level_coord) > frac_thresh
        severity_level = severity_level.where(~exceeds, other=level_coord)

    severity_level = severity_level.where(PR5day_val.notnull().any("level"))
    severity_level.name = "PR5day_severity_level"
    PR5day_val = xr.merge([PR5day_val, severity_level], compat="override")
    PR5day_val = PR5day_val.sortby("level")  # sort ascending so level 1=90th, level 4=99.5th
    PR5day_val = PR5day_val.where(~nanmask)
    PR5day_val = _add_metric_metadata(PR5day_val, "PR5day", ds_dict, severity_thresholds=percentile_thresholds, units="fraction of year", notes="percentile thresholds from base period 5-day rolling sums (wet windows > 5mm). level N: exceedance > (100-p)/100 of year.")
    PR5day_val.attrs['level_thresholds'] = [
        {"level": n - i, "percentile": pr5day_base_percentile_vals[i],
         "exceedance_frac_threshold": (100 - pr5day_base_percentile_vals[i]) / 100,
         "unit": "percentile", "source": "base period distribution"}
        for i in range(n)
    ]
    return PR5day_val

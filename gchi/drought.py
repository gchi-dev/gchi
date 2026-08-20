"""
drought metrics: DSD, SPI, SMSXp

note. all work well chunked spatiallyßßß
SPI uses apply_ufunc with dask="parallelized"
"""

import numpy as np
import xarray as xr
from scipy import stats

from ._core import (
    _check_and_convert_units, _annual_exceedance_frac, _assign_severity_level,
    _get_tsteps, _ann_frac, _nan_mask, _add_metric_metadata,
)
from .thresholds import severity_thresholds as _default_thresholds
from ._log import logger

def _extract_da(val, var_name):
    if isinstance(val, xr.DataArray):
        return val
    elif isinstance(val, xr.Dataset):
        if var_name in val:
            return val[var_name]
        return val[list(val.data_vars)[0]]
    raise TypeError(f"base_dict['{var_name}'] must be a DataArray or Dataset, got {type(val)}")

def dsd_values(ds_dict):
    """
    Daily precipitation (mm day-1) — for DSD calculation
    Exposed so users can inspect the raw precip field if needed
    """
    return _check_and_convert_units(da=ds_dict['pr'], input_var="pr", conv_type="mm day-1")


def DSD(ds_dict, severity_thresholds=None, min_threshold=10):
    """
    Dry Spell Days — fraction of year that falls within dry spells of
    at least min_threshold days (default 10)

    Uses a rolling window approach: counts all days that are part of a qualifying dry spell
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["DSD"]

    PR = dsd_values(ds_dict)
    steps_per_year = _get_tsteps(PR)

    dry_mask = PR < 1

    rolling_sum = PR.where(dry_mask).rolling(time=min_threshold, center=False).count()
    window_all_dry = rolling_sum == min_threshold

    mask_expanded = xr.zeros_like(dry_mask, dtype=bool)
    for shift in range(min_threshold):
        mask_expanded |= window_all_dry.shift(time=shift, fill_value=False)

    DSD_val = PR.where(mask_expanded).resample(time="1YE").count()
    DSD_val = DSD_val.where(~_nan_mask(PR))
    DSD_val = _ann_frac(DSD_val, steps_per_year).rename("DSD")
    result = _assign_severity_level(DSD_val, frac_thresholds=severity_thresholds)
    return _add_metric_metadata(result, "DSD", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"dry spell days: dry spell >= {min_threshold} days with pr < 1 mm/day.")


def SPI(ds_dict, base_dict, timescale=6, severity_thresholds=None):
    """
    Standardized Precipitation Index (SPI)
    Fits a gamma distribution per calendar month to the base period and applies it to the study period
    Follows McKee et al. (1993) / climate_indices methodology
        McKee, T., Doesken, N., & Kleist, J. (1993). The Relationship of Drought Frequency and Duration of Time Scales. Eighth Conference on Applied Climatology, 17–22.
    Default thresholds from USDM drought classification (https://droughtmonitor.unl.edu/About/AbouttheData/DroughtClassification.aspx)
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["SPI"]
 
    base_pr = _check_and_convert_units(da=_extract_da(base_dict['pr'], 'pr'), input_var="pr", conv_type="mm day-1")
    base_acc = base_pr.resample(time="1ME").sum().chunk({"time": -1}).rolling(time=timescale, min_periods=timescale).sum().dropna('time')
 
    study_pr = _check_and_convert_units(da=ds_dict['pr'], input_var="pr", conv_type="mm day-1")
    study_acc = study_pr.resample(time="1ME").sum().chunk({"time": -1}).rolling(time=timescale, min_periods=timescale).sum().dropna('time')
 
    base_months  = xr.ones_like(base_acc)  * xr.DataArray(base_acc.time.dt.month.values,  dims="time")
    study_months = xr.ones_like(study_acc) * xr.DataArray(study_acc.time.dt.month.values, dims="time")
 
    def _fit_and_apply_gamma(hist_data, hist_months, study_data, study_months):
        out = np.full(study_data.shape, np.nan)
        for month in range(1, 13):
            hist_m = hist_data[hist_months == month]
            hist_m = hist_m[~np.isnan(hist_m)]
            if len(hist_m) < 4:
                continue
            study_sel  = (study_months == month) & ~np.isnan(study_data)
            study_vals = study_data[study_sel]
            # probability-of-zero mass (McKee et al. 1993 / Thom 1966)
            q        = np.mean(hist_m == 0)
            hist_pos = hist_m[hist_m > 0]
            if len(hist_pos) < 4:
                continue
            params = stats.gamma.fit(hist_pos, floc=0)
            cdf    = stats.gamma.cdf(study_vals, *params)
            cdf    = q + (1.0 - q) * cdf                              # blend zero mass
            spi    = stats.norm.ppf(np.clip(cdf, 1e-6, 1.0 - 1e-6))
            spi    = np.where(np.isinf(spi), np.nan, spi)
            out[study_sel] = np.clip(spi, -3.09, 3.09)
        return out
 
    SPI_val = xr.apply_ufunc(
        _fit_and_apply_gamma,
        base_acc, base_months, study_acc, study_months,
        input_core_dims=[['time'], ['time'], ['time'], ['time']],
        output_core_dims=[['time']],
        exclude_dims=set(("time",)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"time": len(study_acc.time)}},
    )
    SPI_val = SPI_val.assign_coords(time=study_acc.time)
    SPI_levels = _annual_exceedance_frac(SPI_val, severity_thresholds=severity_thresholds, var_name="SPI", exceedance_dir="below")
    SPI_levels = SPI_levels.where(~_nan_mask(study_pr))
    result = _assign_severity_level(SPI_levels)
    return _add_metric_metadata(result, "SPI", ds_dict, severity_thresholds=severity_thresholds, units="standardised index (dimensionless)", notes=f"gamma distribution fitted per calendar month to base period pr. timescale={timescale} months.")
 
def SMSXp(ds_dict, base_dict, severity_thresholds=None):
    """
    Days where daily surface soil moisture (mrsos) < Xth percentile of base
    period mrsos. Same method as TNXp — spatially varying thresholds, bespoke
    exceedance method but no unit conversion applied because thresholds relative to base (assuming same units)
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["SMSXp"]

    SM = ds_dict["mrsos"]
    nan_mask = _nan_mask(SM)
    steps_per_year = _get_tsteps(SM)

    mrsos_base_percentile_vals = [
        float(k.split("_")[1].replace("pt", ".").replace("p", ""))
        for k in base_dict.keys() if k.startswith('mrsos_') and k.endswith('p')
    ]
    mrsos_base_percentile_keys = sorted(
        [k for k in base_dict.keys() if k.startswith('mrsos_') and k.endswith('p')],
        key=lambda k: float(k.replace('mrsos_', '').replace('pt', '.').replace('p', '')),
        reverse=True,
    )

    if sorted(mrsos_base_percentile_vals) != sorted(severity_thresholds):
        logger.warning(
            f"cannot calculate SMSXp — base period mrsos percentiles don't match severity_thresholds.\n"
            f"base period: {mrsos_base_percentile_vals}. thresholds: {severity_thresholds}. skipping..."
        )
        return None

    n = len(mrsos_base_percentile_keys)
    da_list = []
    for key in mrsos_base_percentile_keys:
        th = base_dict[key]
        da_count = (SM < th).resample(time='1YE').sum(dim='time', skipna=True)
        da_list.append(da_count)

    SMSXp_val = xr.concat(da_list, dim='level')
    SMSXp_val = SMSXp_val.assign_coords(level=np.arange(1, n + 1))
    SMSXp_val = _ann_frac(SMSXp_val, steps_per_year).rename("SMSXp")

    severity_level = xr.zeros_like(SMSXp_val.isel(level=0).drop_vars("level"), dtype=int)
    for i, p in enumerate(mrsos_base_percentile_vals):
        frac_thresh = p / 100
        level_coord = i + 1
        exceeds = SMSXp_val.sel(level=level_coord) > frac_thresh
        severity_level = severity_level.where(~exceeds, other=level_coord)

    severity_level = severity_level.where(SMSXp_val.notnull().any("level"))
    severity_level.name = "SMSXp_severity_level"
    SMSXp_val = xr.merge([SMSXp_val, severity_level], compat="override")
    SMSXp_val = SMSXp_val.where(~nan_mask)
    SMSXp_val.attrs['level_thresholds'] = [
        {"level": int(i + 1), "threshold_value": mrsos_base_percentile_vals[i],
         "unit": "percentile", "source": "base period distribution"}
        for i in range(len(mrsos_base_percentile_keys))
    ]
    return _add_metric_metadata(SMSXp_val, "SMSXp", ds_dict, units="fraction of year",
                                 notes="days below base period mrsos percentile thresholds. spatially varying thresholds. no unit conversion applied.")
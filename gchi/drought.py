"""
drought metrics: CDD, SPI, SPEI

all work well chunked spatially.
SPI and SPEI use apply_ufunc with dask="parallelized".
"""

import numpy as np
import xarray as xr
from scipy import stats

from ._core import (
    _check_and_convert_units, _annual_exceedance_frac, _assign_severity_level,
    _get_tsteps, _ann_frac, _nan_mask, _add_metric_metadata,
)
from .thresholds import severity_thresholds as _default_thresholds

def _extract_da(val, var_name):
    """
    Extract a DataArray from either a DataArray or Dataset.
    Supports passing a raw dict of Datasets as base_dict (e.g. esm2_base_dict)
    or the output of calculate_base_period_percentiles() which has DataArrays.
    """
    if isinstance(val, xr.DataArray):
        return val
    elif isinstance(val, xr.Dataset):
        if var_name in val:
            return val[var_name]
        return val[list(val.data_vars)[0]]
    raise TypeError(f"base_dict['{var_name}'] must be a DataArray or Dataset, got {type(val)}")

def cdd_values(ds_dict):
    """
    Daily precipitation (mm day-1) — for CDD calculation.
    Exposed so users can inspect the raw precip field if needed.
    """
    return _check_and_convert_units(da=ds_dict['pr'], input_var="pr", conv_type="mm day-1")


def CDD(ds_dict, severity_thresholds=None, min_threshold=5):
    """
    Consecutive Dry Days — fraction of year that falls within dry spells of
    at least min_threshold days (default 5).

    Uses a rolling window approach: counts all days that are part of a qualifying dry spell.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["CDD"]

    PR = cdd_values(ds_dict)
    steps_per_year = _get_tsteps(PR)

    dry_mask = PR < 1

    rolling_sum = PR.where(dry_mask).rolling(time=min_threshold, center=False).count()
    window_all_dry = rolling_sum == min_threshold

    mask_expanded = xr.zeros_like(dry_mask, dtype=bool)
    for shift in range(min_threshold):
        mask_expanded |= window_all_dry.shift(time=shift, fill_value=False)

    CDD_val = PR.where(mask_expanded).resample(time="1YE").count()
    CDD_val = CDD_val.where(~_nan_mask(PR))
    CDD_val = _ann_frac(CDD_val, steps_per_year).rename("CDD")
    result = _assign_severity_level(CDD_val, frac_thresholds=severity_thresholds)
    return _add_metric_metadata(result, "CDD", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"consecutive dry days: dry spell >= {min_threshold} days with pr < 1 mm/day.")

def CDD(ds_dict, severity_thresholds=None, min_threshold=10):
    """
    Consecutive Dry Days — fraction of year that falls within dry spells of
    at least min_threshold days (default 10).

    Uses a rolling window approach: counts all days that are part of a qualifying dry spell.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["CDD"]

    PR = cdd_values(ds_dict)
    steps_per_year = _get_tsteps(PR)

    dry_mask = PR < 1

    rolling_sum = PR.where(dry_mask).rolling(time=min_threshold, center=False).count()
    window_all_dry = rolling_sum == min_threshold

    mask_expanded = xr.zeros_like(dry_mask, dtype=bool)
    for shift in range(min_threshold):
        mask_expanded |= window_all_dry.shift(time=shift, fill_value=False)

    CDD_val = PR.where(mask_expanded).resample(time="1YE").count()
    CDD_val = CDD_val.where(~_nan_mask(PR))
    CDD_val = _ann_frac(CDD_val, steps_per_year).rename("CDD")
    result = _assign_severity_level(CDD_val, frac_thresholds=severity_thresholds)
    return _add_metric_metadata(result, "CDD", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"consecutive dry days: dry spell >= {min_threshold} days with pr < 1 mm/day.")

def SPI(ds_dict, base_dict, timescale=6, severity_thresholds=None):
    """
    Standardized Precipitation Index.
    Fits a gamma distribution per calendar month to the base period and applies it to the study period.
    Follows McKee et al. (1993) / climate_indices methodology.
    Default thresholds from USDM drought classification.
 
    Parameters
    ----------
    base_dict : dict
        output from calculate_base_period_percentiles() — needs 'pr'
    ds_dict : dict
        study period data — needs 'pr'
    timescale : int
        accumulation timescale in months (default 6)
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
    SPI_val = SPI_val.where(~_nan_mask(study_pr))
    SPI_levels = _annual_exceedance_frac(SPI_val, severity_thresholds=severity_thresholds, var_name="SPI", exceedance_dir="below")
    result = _assign_severity_level(SPI_levels)
    return _add_metric_metadata(result, "SPI", ds_dict, severity_thresholds=severity_thresholds, units="standardised index (dimensionless)", notes=f"gamma distribution fitted per calendar month to base period pr. timescale={timescale} months.")
 
 
def SPEI(ds_dict, base_dict, timescale=6, severity_thresholds=None):
    """
    Standardized Precipitation-Evapotranspiration Index.
    Fits a generalized logistic distribution per calendar month to the water balance (P - PET)
    over the base period and applies it to the study period.
    Follows McKee et al. (1993) / climate_indices methodology.
 
    Parameters
    ----------
    base_dict : dict
        needs 'pr' and 'evspsblpot'
    ds_dict : dict
        needs 'pr' and 'evspsblpot'
    timescale : int
        accumulation timescale in months (default 6)
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["SPEI"]
 
    PR_base         = _check_and_convert_units(da=base_dict['pr'],         input_var="pr",         conv_type="mm day-1")
    EVSPSBLPOT_base = _check_and_convert_units(da=base_dict['evspsblpot'], input_var="evspsblpot", conv_type="mm day-1")
    base_acc = (PR_base - EVSPSBLPOT_base).resample(time="1ME").sum().chunk({"time": -1}).rolling(time=timescale, min_periods=timescale).sum().dropna('time')
 
    PR         = _check_and_convert_units(da=_extract_da(ds_dict['pr'],         'pr'),         input_var="pr",         conv_type="mm day-1")
    EVSPSBLPOT = _check_and_convert_units(da=_extract_da(ds_dict['evspsblpot'], 'evspsblpot'), input_var="evspsblpot", conv_type="mm day-1")
    study_acc = (PR - EVSPSBLPOT).resample(time="1ME").sum().chunk({"time": -1}).rolling(time=timescale, min_periods=timescale).sum().dropna('time')
 
    base_months  = xr.ones_like(base_acc)  * xr.DataArray(base_acc.time.dt.month.values,  dims="time")
    study_months = xr.ones_like(study_acc) * xr.DataArray(study_acc.time.dt.month.values, dims="time")
 
    def _fit_and_apply_genlog(hist_data, hist_months, study_data, study_months):
        out = np.full(study_data.shape, np.nan)
        for month in range(1, 13):
            hist_m = hist_data[hist_months == month]
            hist_m = hist_m[~np.isnan(hist_m)]
            if len(hist_m) < 4:
                continue
            study_sel  = (study_months == month) & ~np.isnan(study_data)
            study_vals = study_data[study_sel]
            # genlogistic is defined over all reals — no zero-mass adjustment needed
            params = stats.genlogistic.fit(hist_m)
            cdf    = stats.genlogistic.cdf(study_vals, *params)
            spei   = stats.norm.ppf(np.clip(cdf, 1e-6, 1.0 - 1e-6))
            spei   = np.where(np.isinf(spei), np.nan, spei)
            out[study_sel] = np.clip(spei, -3.09, 3.09)
        return out
 
    SPEI_val = xr.apply_ufunc(
        _fit_and_apply_genlog,
        base_acc, base_months, study_acc, study_months,
        input_core_dims=[['time'], ['time'], ['time'], ['time']],
        output_core_dims=[['time']],
        exclude_dims=set(("time",)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"time": len(study_acc.time)}},
    )
    SPEI_val = SPEI_val.assign_coords(time=study_acc.time)
    SPEI_val = SPEI_val.where(~_nan_mask(PR))
    SPEI_levels = _annual_exceedance_frac(SPEI_val, severity_thresholds=severity_thresholds, var_name="SPEI", exceedance_dir="below")
    result = _assign_severity_level(SPEI_levels)
    return _add_metric_metadata(result, "SPEI", ds_dict, severity_thresholds=severity_thresholds, units="standardised index (dimensionless)", notes=f"generalised logistic fitted per calendar month to P-PET. timescale={timescale} months. PET var: evspsblpot.")
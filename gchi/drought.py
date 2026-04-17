"""
drought metrics: CDD, SPI, SPEI

all work well chunked spatially.
SPI and SPEI use apply_ufunc with dask="parallelized".
"""

import numpy as np
import xarray as xr
from scipy import stats

from ._core import (
    _check_and_convert_units, _annual_exceedance_frac, _assign_hazard_level,
    _get_tsteps, _ann_frac,
)
from .thresholds import hazard_thresholds as _default_thresholds


def cdd_values(ds_dict):
    """
    Daily precipitation (mm day-1) — for CDD calculation.
    Exposed so users can inspect the raw precip field if needed.
    """
    return _check_and_convert_units(da=ds_dict['pr'], input_var="pr", conv_type="mm day-1")


def CDD(ds_dict, hazard_thresholds=None, min_threshold=10):
    """
    Consecutive Dry Days — fraction of year that falls within dry spells of
    at least min_threshold days (default 10).

    Uses a rolling window approach: counts all days that are part of a qualifying dry spell.
    """
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["CDD"]

    PR = cdd_values(ds_dict)
    steps_per_year = _get_tsteps(PR)

    dry_mask = PR < 1

    rolling_sum = PR.where(dry_mask).rolling(time=min_threshold, center=False).count()
    window_all_dry = rolling_sum == min_threshold

    mask_expanded = xr.zeros_like(dry_mask, dtype=bool)
    for shift in range(min_threshold):
        mask_expanded |= window_all_dry.shift(time=shift, fill_value=False)

    CDD_val = PR.where(mask_expanded).resample(time="1YE").count()
    CDD_val = _ann_frac(CDD_val, steps_per_year).rename("CDD")
    return _assign_hazard_level(CDD_val, frac_thresholds=hazard_thresholds)


def SPI(base_dict, ds_dict, timescale=6, hazard_thresholds=None):
    """
    Standardized Precipitation Index.
    Fits a gamma distribution to the base period and applies it to the study period.
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
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["SPI"]

    base_pr = _check_and_convert_units(da=base_dict['pr'], input_var="pr", conv_type="mm day-1")
    base_acc = base_pr.resample(time="1ME").sum().rolling(time=timescale).sum().dropna('time')

    study_pr = _check_and_convert_units(da=ds_dict['pr'], input_var="pr", conv_type="mm day-1")
    study_acc = study_pr.resample(time="1ME").sum().rolling(time=timescale).sum().dropna('time')

    def _fit_and_apply_gamma(hist_data, study_data):
        hist_data = hist_data[~np.isnan(hist_data)]
        out = np.full(study_data.shape, np.nan)
        study_mask = ~np.isnan(study_data)
        if len(hist_data) < 30:
            return out
        params = stats.gamma.fit(hist_data, floc=0)
        cdf = stats.gamma.cdf(study_data[study_mask], *params)
        out[study_mask] = stats.norm.ppf(np.clip(cdf, 1e-6, 0.999999))
        return out

    SPI_val = xr.apply_ufunc(
        _fit_and_apply_gamma, base_acc, study_acc,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time']],
        exclude_dims=set(("time",)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    SPI_val = SPI_val.assign_coords(time=study_acc.time)

    SPI_levels = _annual_exceedance_frac(SPI_val, hazard_thresholds=hazard_thresholds, var_name="SPI")
    return _assign_hazard_level(SPI_levels)


def SPEI(base_dict, ds_dict, timescale=6, hazard_thresholds=None):
    """
    Standardized Precipitation-Evapotranspiration Index.
    Fits a generalized logistic distribution to the water balance (P - PET)
    over the base period and applies it to the study period.

    Parameters
    ----------
    base_dict : dict
        needs 'pr' and 'evspsblpot'
    ds_dict : dict
        needs 'pr' and 'evspsblpot'
    timescale : int
        accumulation timescale in months (default 6)
    """
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["SPEI"]

    PR_base = _check_and_convert_units(da=base_dict['pr'], input_var="pr", conv_type="mm day-1")
    EVSPSBLPOT_base = _check_and_convert_units(da=base_dict['evspsblpot'], input_var="evspsblpot", conv_type="mm day-1")
    base_acc = (PR_base - EVSPSBLPOT_base).resample(time="1ME").sum().rolling(time=timescale).sum().dropna('time')

    PR = _check_and_convert_units(da=ds_dict['pr'], input_var="pr", conv_type="mm day-1")
    EVSPSBLPOT = _check_and_convert_units(da=ds_dict['evspsblpot'], input_var="evspsblpot", conv_type="mm day-1")
    study_acc = (PR - EVSPSBLPOT).resample(time="1ME").sum().rolling(time=timescale).sum().dropna('time')

    def _fit_and_apply_genlog(hist_data, study_data):
        hist_clean = hist_data[~np.isnan(hist_data)]
        out = np.full(study_data.shape, np.nan)
        study_mask = ~np.isnan(study_data)
        if len(hist_clean) < 30:
            return out
        params = stats.genlogistic.fit(hist_clean)
        cdf = stats.genlogistic.cdf(study_data[study_mask], *params)
        out[study_mask] = stats.norm.ppf(np.clip(cdf, 1e-6, 0.999999))
        return out

    SPEI_val = xr.apply_ufunc(
        _fit_and_apply_genlog, base_acc, study_acc,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time']],
        exclude_dims=set(("time",)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    SPEI_val = SPEI_val.assign_coords(time=study_acc.time)

    SPEI_levels = _annual_exceedance_frac(SPEI_val, hazard_thresholds=hazard_thresholds, var_name="SPEI")
    return _assign_hazard_level(SPEI_levels)

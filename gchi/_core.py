"""
internal helpers — unit conversion, exceedance counting, level assignment, etc.
not intended to be called directly by users
"""

import numpy as np
import xarray as xr
import xesmf as xe
from ._log import logger


def _drop_all_bounds(da):
    """drop X_bounds coordinates (can cause merging issues)"""
    drop_bnds = [v for v in da.coords if ('_bounds' in v) or ('_bnds' in v)]
    return da.drop_vars(drop_bnds)


def _is_prepared(ds_dict):
    """true if every array in ds_dict already went through prepare_inputs()"""
    return all(
        getattr(da, "attrs", {}).get("gchi_prepared", False)
        for da in ds_dict.values() if hasattr(da, "attrs")
    )


def _sanity_check_units(da: xr.DataArray, units_attr: str):
    """check data value range based on units specified"""
    da = _drop_all_bounds(da)
    try:
        sample = da.isel(time=0) if "time" in da.dims else da
        minv = float(sample.min(skipna=True))
        maxv = float(sample.max(skipna=True))
    except Exception as e:
        logger.warning(f"data value spot check failed: {e}\nskipping units spot check. recommended: add units attrs and re-run")
        return

    if units_attr == "C":
        if not (-125 < minv < 65 and -125 < maxv < 65):
            logger.warning(f"values {minv:.1f}–{maxv:.1f} outside reasonable range for temperature in °C.")
    elif units_attr == "K":
        if not (150 < minv < 340 and 150 < maxv < 340):
            logger.warning(f"values {minv:.1f}–{maxv:.1f} outside reasonable range for temperature in K.")
    elif units_attr == "F":
        if not (-195 < minv < 150 and -195 < maxv < 150):
            logger.warning(f"values {minv:.1f}–{maxv:.1f} outside reasonable range for temperature in °F.")
    elif units_attr == "fraction":
        if not (0 <= minv and maxv <= 1.10):
            logger.warning(f"values {minv:.1f}–{maxv:.1f} outside reasonable range for relative humidity (fraction). clipped to 0.001-0.999999.")
    elif units_attr == "%":
        if not (0 <= minv and maxv <= 110):
            logger.warning(f"values {minv:.1f}–{maxv:.1f} outside reasonable range for relative humidity (%). clipped to 0.1-99.9999.")
    elif units_attr == "hPa":
        if not (100 <= minv <= 1200 and 100 <= maxv <= 1200):
            logger.warning(f"values {minv:.1f}–{maxv:.1f} outside reasonable range for pressure (hPa).")
    elif units_attr == "mm day-1":
        if maxv > 300:
            logger.warning(f"max value {maxv:.3f} unusually large for precipitation (mm/day): check units.")
        elif minv < 0:
            logger.warning(f"min value {minv:.3f} negative for precipitation (mm/day): check units.")
    elif units_attr == "m s-1":
        if maxv > 100:
            logger.warning(f"max value {maxv:.3f} unusually large for wind speed (m s-1): check units.")
        if minv < 0:
            logger.warning(f"min value {minv:.3f} negative for wind speed (m s-1): check it's speed not u/v component.")
    elif units_attr == "km h-1":
        if maxv > 360:
            logger.warning(f"max value {maxv:.3f} unusually large for wind speed (km h-1): check units.")
        if minv < 0:
            logger.warning(f"min value {minv:.3f} negative for wind speed (km h-1): check it's speed not u/v component.")
    elif units_attr == "psu":
        if maxv > 50:
            logger.warning(f"max value {maxv:.3f} unusually large for sea surface salinity (psu): check units.")
        if minv < 5:
            logger.warning(f"min value {minv:.3f} unusually small for sea surface salinity (psu).")
    elif units_attr == "kg kg-1":
        if maxv > 1e-4:
            logger.warning(f"max value {maxv:.3f} unusually large for aerosol inputs (kg kg-1): check units.")
        if minv < 0:
            logger.warning(f"min value {maxv:.3f} negative for aerosol inputs (kg kg-1): check units.")


def _check_and_convert_units(da: xr.DataArray, input_var: str, conv_type: str):
    """
    Check and convert the units of a DataArray.

    Parameters
    ----------
    da : xr.DataArray
    input_var : str
        variable name (for print messages)
    conv_type : str
        target unit — one of 'C','K','F','fraction','%','hPa','Pa','mm day-1','m s-1','km h-1','psu','kg kg-1'

    Returns
    -------
    xr.DataArray with updated units attribute
    """
    da_out = da.copy()

    # extract and normalize units string
    raw_units = None
    for k in da.attrs:
        if k.lower() in ["unit", "units"]:
            raw_units = da.attrs[k]
            break

    if raw_units:
        u = str(raw_units).lower().replace(" ", "").replace("degrees", "").replace("deg", "").replace("°", "")
    else:
        u = None

    # normalized unit categories — non-exhaustive but covers most cmip6 + common variants
    temp_c = {"c", "celsius", "centigrade"}
    temp_k = {"k", "kelvin"}
    temp_f = {"f", "fahrenheit"}
    percent = {"%", "percent", "pct"}
    frac = {"fraction", "frac"}
    pa = {"pa", "pascal", "pascals"}
    hpa = {"hpa", "mb", "millibar", "millibars"}
    precip_kg_ms = {"kgm-2s-1", "kg/m2s", "kgm2s-1", "kgm^-2s^-1"}
    precip_mm_day = {"mmday-1", "mm/day"}
    wind_m_s = {"ms-1", "m/s", "ms^-1"}
    wind_km_h = {"kmh-1", "km/h", "kmh^-1"}
    salinity_psu = {"psu", "practicalsalinityunits"}
    pm_kg_kg = {"kgkg-1", "kg/kg", "kgkg^-1"}

    if u in temp_c:
        units_attr = "C"
    elif u in temp_k:
        units_attr = "K"
    elif u in temp_f:
        units_attr = "F"
    elif u in percent:
        units_attr = "%"
    elif u in frac:
        units_attr = "fraction"
    elif u in pa:
        units_attr = "Pa"
    elif u in hpa:
        units_attr = "hPa"
    elif u in precip_kg_ms:
        units_attr = "kg m-2 s-1"
    elif u in precip_mm_day:
        units_attr = "mm day-1"
    elif u in wind_m_s:
        units_attr = "m s-1"
    elif u in wind_km_h:
        units_attr = "km h-1"
    elif u in salinity_psu:
        units_attr = "psu"
    elif u in pm_kg_kg:
        units_attr = "kg kg-1"
    else:
        units_attr = None

    # if units missing, try to guess from data range
    guessed = False
    if units_attr is None:
        guessed = True
        try:
            sample = da.isel(time=0) if "time" in da.dims else da
            minv = float(sample.min(skipna=True))
            maxv = float(sample.max(skipna=True))
        except Exception as e:
            logger.warning(f"data value spot check failed: {e}\nskipping units spot check. add units attrs and re-run")
            minv, maxv = np.nan, np.nan

        if np.isnan(minv) or np.isnan(maxv):
            units_attr = "unknown"
        elif conv_type in ["C", "F", "K"]:
            if -60 < minv < 60 and -50 < maxv < 60:
                units_attr = "C"
            elif 120 < minv < 370 and 150 < maxv < 370:
                units_attr = "K"
            elif -60 < minv < 140 and 32 < maxv < 140:
                units_attr = "F"
        elif conv_type in ["%", "fraction"]:
            units_attr = "fraction" if maxv <= 10 else "%"
        elif conv_type in ["Pa", "hPa"]:
            if 100 < minv < 1200 and 100 < maxv < 1200:
                units_attr = "hPa"
            elif 10000 < minv < 120000 and 10000 < maxv < 120000:
                units_attr = "Pa"
        elif conv_type in ["mm day-1"]:
            if 0 <= minv <= 0.005 and 0 <= maxv <= 0.02:
                units_attr = "kg m-2 s-1"
            elif 0 <= minv <= 300 and 0 <= maxv <= 300:
                units_attr = "mm day-1"
        elif conv_type in ["m s-1", "km h-1"]:
            if maxv < 100:
                logger.warning(f"no units attribute found for wind speed. assuming m s-1. please check.")
                units_attr = "m s-1"
            else:
                logger.warning(f"no units attribute found for wind speed. assuming km h-1. please check.")
                units_attr = "km h-1"
        elif conv_type == "psu":
            logger.warning(f"no units attribute found for sea surface salinity. assuming psu. please check.")
            units_attr = "psu"
        elif conv_type == "kg kg-1":
            logger.warning(f"no units attribute found for PM inputs. assuming kg kg-1. please check.")
            units_attr = "kg kg-1"
        else:
            units_attr = "unknown"

        if units_attr not in [None, "unknown"]:
            logger.warning(f"guessed {input_var} units as '{units_attr}' based on data values. min: {round(minv, 3)} max: {round(maxv, 3)}.")
        else:
            logger.warning(f"could not guess {input_var} units. min: {round(minv, 3)} max: {round(maxv, 3)}.\nplease add a units attribute and re-run.")

    # already correct units — just sanity check and return
    if units_attr == conv_type:
        da_out.attrs["units"] = conv_type
        _sanity_check_units(da=da_out, units_attr=conv_type)
        return da_out

    # do conversions
    if units_attr == "C":
        if conv_type == "K":
            da_out = da + 273.15
        elif conv_type == "F":
            da_out = da * 9 / 5 + 32
    elif units_attr == "K":
        if conv_type == "C":
            da_out = da - 273.15
        elif conv_type == "F":
            da_out = (da - 273.15) * 9 / 5 + 32
    elif units_attr == "F":
        if conv_type == "C":
            da_out = (da - 32) * 5 / 9
        elif conv_type == "K":
            da_out = (da - 32) * 5 / 9 + 273.15

    if units_attr == "%" and conv_type == "fraction":
        da_out = da / 100
        da_out = da_out.clip(0, 1)
    elif units_attr == "fraction" and conv_type == "%":
        da_out = da * 100
        da_out = da_out.clip(0, 100)

    if units_attr == "Pa" and conv_type == "hPa":
        da_out = da / 100
    elif units_attr == "hPa" and conv_type == "Pa":
        da_out = da * 100

    if units_attr == "kg m-2 s-1" and conv_type == "mm day-1":
        da_out = da * 86400

    if units_attr == "kts":
        if conv_type == "m s-1":
            da_out = da * 0.5144444
        elif conv_type == "km h-1":
            da_out = da * 1.852
    elif units_attr == "m s-1" and conv_type == "km h-1":
        da_out = da * 3.6
    elif units_attr == "km h-1" and conv_type == "m s-1":
        da_out = da * 0.2777778

    if units_attr == "0.001" and conv_type == "psu":
        da_out = da  # 1 part per thousand ~ 1 psu

    da_out.attrs["units"] = conv_type
    da_out.attrs["units_guessed"] = str(guessed)
    _sanity_check_units(da=da_out, units_attr=conv_type)

    return da_out


def _get_tsteps(da):
    """count time steps per year — returns resampled annual series to preserve time dim"""
    try:
        steps_per_year = da.resample(time="1YE").count(dim="time")
        steps_per_year.attrs["units"] = "time steps yr-1"
    except Exception:
        steps_per_year = 365
        logger.warning("could not calculate time steps per year. assuming daily data.")
    return steps_per_year


def _ann_frac(da, steps_per_year):
    """convert annual counts to fraction of year -- preserves time dimension"""
    return da / steps_per_year


def _nan_mask(da):
    """True where ALL timesteps in a year are NaN — used to restore NaNs after resample"""
    return da.isnull().resample(time='1YE').all(dim='time')

def _apply_nan_mask(da_resampled, nan_mask):
    """restore NaNs after resampling — nan_mask must be computed from pre-resample data"""
    return da_resampled.where(~nan_mask)


def _annual_exceedance_frac(da, severity_thresholds, var_name, exceedance_dir="above"):
    """
    Count annual days exceeding each threshold, return as fraction of year.

    da : xr.DataArray with 'time' dimension
    severity_thresholds : list of threshold values (will be sorted)
    var_name : output DataArray name
    exceedance_dir : 'above' (da > th) or 'below' (da < th)

    Returns DataArray with dims ('time', 'level', ...) and 'level_values' attr
    """
    if exceedance_dir.lower() == "below":
        thresholds = np.sort(severity_thresholds)[::-1]
    else:
        thresholds = np.sort(severity_thresholds)

    steps_per_year = _get_tsteps(da)
    # compute before any resampling so land/ocean NaNs are captured
    nan_mask = _nan_mask(da)

    da_list = []
    for th in thresholds:
        if exceedance_dir.lower() == "above":
            da_count = (da > th).resample(time='1YE').sum(dim='time', skipna=True)
        elif exceedance_dir.lower() == "below":
            da_count = (da < th).resample(time='1YE').sum(dim='time', skipna=True)
        else:
            logger.warning(f"exceedance direction '{exceedance_dir}' not recognized. must be 'above' or 'below'")
            return None
        da_list.append(da_count)

    da_exceed = xr.concat(da_list, dim='level')
    da_exceed = _apply_nan_mask(da_exceed, nan_mask)
    da_exceed = da_exceed.assign_coords(level=np.arange(1, len(thresholds) + 1))
    da_exceed.attrs['level_values'] = thresholds.tolist()
    da_exceed = _ann_frac(da_exceed, steps_per_year).rename(var_name)

    return da_exceed


def _annual_exceedance_frac_aq(da, severity_thresholds, var_name, exceedance_dir="above"):
    """
    Conditional annual exceedance for air quality: only counts exceedances in years where
    the annual average itself crosses the threshold. Otherwise count = 0.

    da : xr.DataArray with 'time' dimension
    severity_thresholds : list of threshold values
    var_name : output DataArray name
    """
    if exceedance_dir.lower() == "below":
        thresholds = np.sort(severity_thresholds)[::-1]
    else:
        thresholds = np.sort(severity_thresholds)

    steps_per_year = _get_tsteps(da)
    # compute before any resampling so land/ocean NaNs are captured
    all_nan_mask = _nan_mask(da)
    da_annual_mean = da.resample(time='1YE').mean(dim='time')

    da_list = []
    for th in thresholds:
        if exceedance_dir.lower() == "above":
            da_count = (da > th).resample(time='1YE').sum(dim='time', skipna=True)
            annual_mean_crosses = da_annual_mean > th
        elif exceedance_dir.lower() == "below":
            da_count = (da < th).resample(time='1YE').sum(dim='time', skipna=True)
            annual_mean_crosses = da_annual_mean < th
        else:
            raise ValueError(f"exceedance direction '{exceedance_dir}' not recognized.")

        da_count_conditional = xr.where(annual_mean_crosses, da_count, 0)
        da_count_conditional = xr.where(all_nan_mask, np.nan, da_count_conditional)
        da_list.append(da_count_conditional)

    da_exceed = xr.concat(da_list, dim='level')
    da_exceed = da_exceed.assign_coords(level=np.arange(1, len(thresholds) + 1))
    da_exceed.attrs['level_values'] = thresholds.tolist()
    da_exceed = _ann_frac(da_exceed, steps_per_year).rename(var_name)

    return da_exceed


def _annual_exceedance_frac_fwi(da_fwi, da_zones, fwi_thresholds, var_name='FWI'):
    """
    FWI-specific exceedance — uses spatially varying thresholds based on environmental zone.

    da_fwi : xr.DataArray (lat, lon, time)
    da_zones : xr.DataArray (lat, lon) integer zone codes 1–18
    fwi_thresholds : dict mapping letter -> [t1, t2, t3, t4]
    """
    zone_letters = np.vectorize(
        lambda x: chr(ord('@') + int(x)) if not np.isnan(x) else None
    )(da_zones.values)

    thresh_array = np.full((*zone_letters.shape, 4), np.nan)
    for i in range(zone_letters.shape[0]):
        for j in range(zone_letters.shape[1]):
            letter = zone_letters[i, j]
            if letter in fwi_thresholds:
                thresh_array[i, j, :] = fwi_thresholds[letter]

    thresh_da = xr.DataArray(
        thresh_array,
        dims=['lat', 'lon', 'level'],
        coords={'lat': da_zones.lat, 'lon': da_zones.lon, 'level': [1, 2, 3, 4]},
    )

    da_list = []
    for lvl in [1, 2, 3, 4]:
        th = thresh_da.sel(level=lvl)
        da_count = (da_fwi > th).resample(time='1YE').sum('time')
        da_list.append(da_count)

    # computed before resampling above
    nan_mask = _nan_mask(da_fwi)
    da_exceed = xr.concat(da_list, dim='level').assign_coords(level=[1, 2, 3, 4])
    da_exceed = _apply_nan_mask(da_exceed, nan_mask)

    steps_per_year = _get_tsteps(da_fwi)
    da_exceed = _ann_frac(da_exceed, steps_per_year).rename(var_name)
    da_exceed.attrs['level_values'] = 'spatially varying — see fwi_thresholds'

    return da_exceed


def _assign_severity_level(da, frac_thresholds=None):
    """
    Assign severity level (1–4) per year per grid cell based on highest threshold crossed.

    For vars with a 'level' dimension (threshold-based exceedance):
        level is the highest level where exceedance fraction > min_days_frac (0.01)

    For vars without a 'level' dimension (single value per year):
        compares annual value against frac_thresholds directly

    Returns a Dataset with the original da + a {name}_severity_level variable.
    """
    min_days_frac = 0.01  # must exceed > 1% of year to be valid

    if "level" in da.dims:
        severity_level = xr.zeros_like(da.isel(level=0), dtype=int)
        for i in range(da.level.size):
            severity_level = severity_level.where(da.isel(level=i) <= min_days_frac, other=i + 1)
        # restore NaN where da was NaN
        severity_level = severity_level.where(da.notnull().any('level'))
    else:
        thresholds = np.sort(frac_thresholds)
        severity_level = xr.zeros_like(da, dtype=int)
        for i, th in enumerate(thresholds):
            severity_level = severity_level.where(da <= th, other=i + 1)
        severity_level = severity_level.where(da.notnull())

    severity_level.name = f"{da.name}_severity_level"
    severity_level.attrs["calculation_notes"] = (
        "severity level 1–4: highest threshold crossed per year per grid cell. 0 = no threshold crossed."
    )

    return xr.merge([da, severity_level], compat="override")


def _tetens_sat_vapor_pressure(T_celsius):
    """saturation vapor pressure (kPa) via Tetens equation — used in AT, HI, Hu, HDW"""
    es_positive = 0.611 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))
    es_negative = 0.611 * np.exp(21.87 * T_celsius / (T_celsius + 265.5))
    return xr.where(T_celsius > 0, es_positive, es_negative)


def _regrid_xr(ds_in, regrid_to, method='bilinear', name=None):
    """regrid ds_in to the target grid"""
    regridder = xe.Regridder(ds_in, regrid_to, method=method, periodic=True, ignore_degenerate=True)
    ds_out = regridder(ds_in)
    if isinstance(ds_out, xr.DataArray):
        ds_out.name = name
        ds_in.name = name
        temp_ds = ds_out.to_dataset()
        for var in temp_ds.data_vars:
            temp_ds[var].attrs = ds_in.attrs
        temp_ds.attrs = ds_in.to_dataset().attrs
        temp_ds.attrs["regridded"] = "True"
        ds_out = temp_ds[ds_out.name]
    elif isinstance(ds_out, xr.Dataset):
        for var in ds_out.data_vars:
            ds_out[var].attrs = ds_in[var].attrs
        ds_out.attrs = ds_in.attrs
        ds_out.attrs["regridded"] = "True"
    return ds_out


def _add_metric_metadata(result, metric_name, ds_dict, severity_thresholds=None,
                          units="fraction of year", notes=None, extra=None):
    """
    Add standardised gchi metadata to a metric output Dataset.
    Preserves any input model/simulation attrs found in ds_dict.
    Safe to call -- if anything fails, prints a warning and returns result unchanged.

    Parameters
    ----------
    result : xr.Dataset
        output from a metric function
    metric_name : str
        e.g. "AT", "WBGT", "PR1day"
    ds_dict : dict
        input data dict -- used to harvest model/simulation attrs
    severity_thresholds : list, optional
        threshold values used for level assignment
    units : str
        units of the metric output values (default "fraction of year")
    notes : str, optional
        methodological decisions or approximations made
    extra : dict, optional
        any additional metric-specific metadata key/value pairs
    """
    from .inputs import SOFTWARE_VERSION
    from .thresholds import severity_thresholds as _default_thresholds
    import datetime

    try:
        # collect input model attrs from ds_dict -- cmip6 and similar models
        # carry these on their DataArrays. we harvest whatever is there so
        # non-cmip inputs work fine too.
        # attrs to drop from source data — noisy/irrelevant for gchi outputs
        _drop_attrs = {
            "standard_name", "comment", "cell_methods", "cell_measures",
            "history", "long_name", "original_name", "description",
        }

        # attrs to harvest from source DataArrays if present
        _model_keys = [
            "source_id", "model_id", "institution_id", "experiment_id",
            "variant_label", "realization", "grid_label", "mip_era",
            "activity_id", "parent_experiment_id", "table_id",
        ]
        input_attrs = {}
        for da in ds_dict.values():
            if hasattr(da, "attrs"):
                for k in _model_keys:
                    if k in da.attrs and k not in input_attrs:
                        input_attrs[k] = da.attrs[k]

        _metric_category = {
            "AT": "heat", "HI": "heat", "Hu": "heat", "WBT": "heat",
            "WBGT": "heat", "UTCIhot": "heat", "TXC": "heat",
            "TR": "heat", "HWF": "heat",
            "UTCIcold": "cold", "TNXp": "cold",
            "FI": "fire", "HDW": "fire", "FWI": "fire",
            "O3": "aq", "PM2pt5": "aq",
            "CDD": "drought", "SPI": "drought", "SPEI": "drought",
            "VSmalaria": "disease", "VSzika": "disease",
            "VSdengueAeg": "disease", "VSdengueAlb": "disease", "VbrS": "disease",
            "PRXmm": "weather", "PR1day": "weather", "PR5day": "weather",
        }

        # determine if default or custom thresholds
        default_thresholds = _default_thresholds.get(metric_name)
        if severity_thresholds is not None and default_thresholds is not None:
            threshold_source = (
                "gchi_default" if sorted(severity_thresholds) == sorted(default_thresholds)
                else "user_defined"
            )
        elif severity_thresholds is not None:
            threshold_source = "user_defined"
        else:
            threshold_source = "gchi_default"
            severity_thresholds = default_thresholds

        # build gchi metadata
        gchi_attrs = {
            "software": "gchi",
            "gchi_category": _metric_category.get(metric_name, "other"),
            "software_version": SOFTWARE_VERSION,
            "metric": metric_name,
            "units": units,
            "severity_level_assignment": (
                "highest level where annual exceedance fraction exceeds threshold. "
                "0 = no threshold exceeded."
            ),
            "severity_thresholds": str(severity_thresholds),
            "severity_threshold_source": threshold_source,
            "date_created": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        if notes:
            gchi_attrs["methodological_notes"] = notes
        if extra:
            gchi_attrs.update(extra)
        if input_attrs:
            gchi_attrs.update({f"input_{k}": v for k, v in input_attrs.items()})

        # per-level threshold attrs
        if severity_thresholds is not None:
            for i, th in enumerate(severity_thresholds):
                gchi_attrs[f"level_{i+1}_threshold"] = str(th)

        # apply to dataset and all variables — drop noisy source attrs
        result = result.copy()
        result.attrs = {**input_attrs, **gchi_attrs}

        for var in result.data_vars:
            # start from existing var attrs, drop unwanted ones, then overlay gchi attrs
            var_attrs = {k: v for k, v in result[var].attrs.items()
                         if k not in _drop_attrs}
            if "_severity_level" in var:
                var_attrs["units"] = "severity level (0-4)"
                var_attrs["long_name"] = f"{metric_name} severity level"
            else:
                var_attrs["units"] = units
                var_attrs["long_name"] = f"{metric_name} annual exceedance fraction"
            result[var].attrs = {**var_attrs, **gchi_attrs}

    except Exception as e:
        logger.warning(f"could not add metadata to {metric_name} output: {e}")

    return result

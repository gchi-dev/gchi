"""
air quality metrics: O3, PM2pt5

both work well chunked spatially.

input data can be daily or monthly -- resolution is detected automatically.
daily data is averaged to monthly before exceedance counting. threshold set
(O3day vs O3mon, PM2pt5day vs PM2pt5mon) is chosen based on the raw input
resolution, since daily data implies a different exposure context than monthly.
users can always override with a custom hazard_thresholds list.
"""

import xarray as xr

from ._core import (
    _check_and_convert_units, _annual_exceedance_frac, _annual_exceedance_frac_aq,
    _assign_hazard_level, _get_surface, _get_tsteps,
)
from .thresholds import hazard_thresholds as _default_thresholds


def _detect_daily(da):
    """
    Returns True if the DataArray looks like daily data (> 15 time steps per year).
    Called before resampling so we can pick the right threshold set.
    """
    steps_per_year = _get_tsteps(da)
    try:
        return bool(steps_per_year.mean() > 15)
    except Exception:
        return False


def o3_values(ds_dict):
    """
    Surface ozone concentration (ug m-3).
    Converts from CMIP6 mol mol-1 using air density from tas and ps.
    Surface level is extracted here as a fallback if prepare_inputs was not run
    (prepare_inputs handles this more efficiently before regridding).
    """
    O3 = ds_dict["o3"]
    # fallback surface extraction -- no-op if already 2D (prepare_inputs ran)
    if ("plev" in O3.dims) or ("lev" in O3.dims):
        O3 = _get_surface(O3, "o3")
    units_attr = None
    for k in O3.attrs:
        if k.lower() in ["unit", "units"]:
            units_attr = O3.attrs[k]
            break

    convert_o3 = True
    if units_attr is None:
        print("no units attribute for ozone. assuming mol mol-1.")
    elif units_attr in ["mol mol-1", "mol/mol", "mol mol^-1"]:
        convert_o3 = True
    elif units_attr in ["ug/m^3", "ug m^-3", "ug m-3"]:
        convert_o3 = False
    elif units_attr not in ["mol mol-1", "mol/mol", "mol mol^-1", "ug m-3", "ug/m3", "ug m^-3", "ug/m^3"]:
        print(f"ozone units '{units_attr}' not recognized. must be mol mol-1 or ug m-3. skipping O3...")
        return None

    if convert_o3:
        T = _check_and_convert_units(da=ds_dict["tas"], input_var="tas", conv_type="K")
        ps = _check_and_convert_units(da=ds_dict["ps"], input_var="ps", conv_type="Pa")
        M_O3 = 48    # molecular weight of ozone g/mol
        R = 8.314    # J/(mol·K)
        O3 = 1e6 * (O3 * (M_O3 * ps) / (R * T))

    return O3


def O3(ds_dict, hazard_thresholds=None, mda8_scale_file=None, mda8_scale_varname="o3"):
    """
    Surface ozone exceedance levels.
    Converts from CMIP6 mol mol-1 to ug m-3, scales to MDA8, then counts
    annual exceedances. Always resamples to monthly internally.

    Input resolution (daily or monthly) is detected automatically and used to
    pick the appropriate default threshold set (O3day vs O3mon). Override with
    a custom hazard_thresholds list if needed.

    Note: input should be surface level only. use prepare_inputs or pass
    surface level directly -- gchi does not slice levels inside this function.

    Parameters
    ----------
    ds_dict : dict
    hazard_thresholds : list, optional
        override default thresholds. if None, picked automatically based on
        detected input resolution (O3day for daily input, O3mon for monthly).
    mda8_scale_file : str, optional
        path to MDA8 scale factor file. if None, raw monthly O3 is returned.
    mda8_scale_varname : str
        variable name in the scale factor file (default 'o3')
    """
    O3_val = o3_values(ds_dict)
    if O3_val is None:
        return None

    # detect resolution before collapsing -- determines which threshold set to use
    is_daily = _detect_daily(O3_val)
    if is_daily:
        print("  O3: detected daily input -- using O3day thresholds")
    else:
        print("  O3: detected monthly input -- using O3mon thresholds")

    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["O3day"] if is_daily else _default_thresholds["O3mon"]

    # always collapse to monthly
    O3_val = O3_val.resample(time="1ME").mean().load()

    if mda8_scale_file is None:
        print("  mda8_scale_file not provided -- skipping MDA8 scaling. using raw monthly O3.")
    else:
        mda8_scale_factor = xr.open_dataset(mda8_scale_file)[mda8_scale_varname]
        O3_val = O3_val.groupby("time.month") / mda8_scale_factor

    # conditional exceedance -- only counts months in years where annual mean
    # also exceeds the threshold, avoids noise from borderline years
    O3_levels = _annual_exceedance_frac_aq(O3_val, hazard_thresholds=hazard_thresholds, var_name="O3")
    return _assign_hazard_level(O3_levels)


def pm25_values(ds_dict):
    """
    PM2.5 concentration (ug m-3) from CMIP6 aerosol mass fraction variables.
    Surface level extracted here as fallback if prepare_inputs was not run.
    """
    BC = ds_dict["mmrbc"]
    OA = ds_dict["mmroa"]
    SO4 = ds_dict["mmrso4"]
    SS = ds_dict["mmrss"]
    DU = ds_dict["mmrdust"]

    # fallback surface extraction -- no-op if already 2D (prepare_inputs ran)
    if ("plev" in BC.dims) or ("lev" in BC.dims):
        BC = _get_surface(BC, "mmrbc")
        OA = _get_surface(OA, "mmroa")
        SO4 = _get_surface(SO4, "mmrso4")
        SS = _get_surface(SS, "mmrss")
        DU = _get_surface(DU, "mmrdust")

    # check if already in ug m-3
    attrs_lower = {k.lower(): v for k, v in BC.attrs.items()}
    found_key = next((k for k in {"unit", "units"} if k in attrs_lower), None)
    if found_key and attrs_lower[found_key] in ["ug/m^3", "ug m^-3", "ug m-3"]:
        print("  detected PM2.5 input units as ug m-3. assuming all aerosol variables have equivalent units.")
        return BC + OA + SO4 + (0.25 * SS) + (0.1 * DU)

    # convert from kg kg-1 and compute air density
    BC = _check_and_convert_units(da=BC, input_var="mmrbc", conv_type="kg kg-1")
    OA = _check_and_convert_units(da=OA, input_var="mmroa", conv_type="kg kg-1")
    SO4 = _check_and_convert_units(da=SO4, input_var="mmrso4", conv_type="kg kg-1")
    SS = _check_and_convert_units(da=SS, input_var="mmrss", conv_type="kg kg-1")
    DU = _check_and_convert_units(da=DU, input_var="mmrdust", conv_type="kg kg-1")

    T = _check_and_convert_units(da=ds_dict["tas"], input_var="tas", conv_type="K")
    ps = _check_and_convert_units(da=ds_dict["ps"], input_var="ps", conv_type="Pa")
    rho = ps / (287 * T)

    pm2pt5 = BC + OA + SO4 + (0.25 * SS) + (0.1 * DU)
    return (pm2pt5 * rho) * 1e9  # kg kg-1 * kg m-3 -> ug m-3


def PM2pt5(ds_dict, hazard_thresholds=None):
    """
    PM2.5 exceedance levels. Always resamples to monthly internally.
    Call pm25_values() to get raw PM2.5 concentration without level assignment.

    Input resolution (daily or monthly) is detected automatically and used to
    pick the appropriate default threshold set (PM2pt5day vs PM2pt5mon).
    Override with a custom hazard_thresholds list if needed.

    Parameters
    ----------
    ds_dict : dict
    hazard_thresholds : list, optional
        override default thresholds. if None, picked automatically based on
        detected input resolution.
    """
    pm2pt5 = pm25_values(ds_dict)

    # detect resolution before collapsing
    is_daily = _detect_daily(pm2pt5)
    if is_daily:
        print("  PM2pt5: detected daily input -- using PM2pt5day thresholds")
    else:
        print("  PM2pt5: detected monthly input -- using PM2pt5mon thresholds")

    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["PM2pt5day"] if is_daily else _default_thresholds["PM2pt5mon"]

    # always collapse to monthly
    pm2pt5 = pm2pt5.resample(time="1ME").mean()

    pm2pt5_levels = _annual_exceedance_frac_aq(pm2pt5, hazard_thresholds=hazard_thresholds, var_name="PM2pt5")
    return _assign_hazard_level(pm2pt5_levels)

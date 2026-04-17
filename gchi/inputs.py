"""
input preparation and base period percentile calculation
"""

import xarray as xr
from ._core import _check_and_convert_units, _drop_all_bounds, _regrid_xr, _get_surface

SOFTWARE_VERSION = "0.0.0"

_DEFAULT_BASE_YEARS = (1980, 2014)

_DEFAULT_PERCENTILES = {
    "tas_calday":  [90],               # calendar-day percentile
    "pr":          [90, 95, 98, 99.5], # all-year wet-day percentiles
    "tasmin":      [10, 5, 2, 0.5],    # all-year cold-tail percentiles
    "rx5day":      [90, 95, 98, 99.5], # annual 5-day max precip percentiles
}


def show_expected_ds_format():
    """
    Prints info about the expected input dictionary format.

    ds_dict keys should be standard CMIP6 shortnames, values should be xarray DataArrays.
    """
    variables = {
        "daily_max_surface_temperature": "tasmax",
        "daily_min_surface_temperature": "tasmin",
        "temperature_surface": "tas",
        "precipitation": "pr",
        "relative_humidity_surface": "hurs",
        "specific_humidity_surface": "huss",
        "surface_pressure": "ps",
        "wind_speed_surface": "sfcWind",
        "daily_max_wind_speed_surface": "sfcWindmax",
        "mass_fraction_of_elemental_carbon_dry_aerosol_particles_in_air": "mmrbc",
        "mass_fraction_of_dust_dry_aerosol_particles_in_air": "mmrdust",
        "mass_fraction_of_particulate_organic_matter_dry_aerosol_particles_in_air": "mmroa",
        "mass_fraction_of_sulfate_dry_aerosol_particles_in_air": "mmrso4",
        "mass_fraction_of_sea_salt_dry_aerosol_particles_in_air": "mmrss",
        "mole_fraction_of_ozone_in_air": "o3",
        "sea_surface_salinity": "sos",
        "sea_surface_temperature": "tos",
    }

    expected_units = {
        "tasmax": "K (or C/F)",
        "tasmin": "K (or C/F)",
        "tas": "K (or C/F)",
        "pr": "kg m-2 s-1 (or mm day-1)",
        "hurs": "% (or fraction)",
        "huss": "None",
        "ps": "Pa (or hPa/mb)",
        "sfcWind": "m s-1",
        "sfcWindmax": "m s-1",
        "mmrbc": "kg kg-1",
        "mmrdust": "kg kg-1",
        "mmroa": "kg kg-1",
        "mmrso4": "kg kg-1",
        "mmrss": "kg kg-1",
        "o3": "mol mol-1",
        "sos": "0.001",
        "tos": "degC",
    }

    print("\nExpected input dictionary `ds_dict` format:\n")
    print("`ds_dict` should have variable keys linked to an xarray DataArray. For example:")
    print("ds_dict = {'tasmax': tasmax_da, 'pr': pr_da, ...}\n")
    print("Key in ds_dict (shortname) : Description / expected units\n")
    for desc, shortname in variables.items():
        units = expected_units.get(shortname, "unknown")
        print(f"{shortname:<12} : {desc} ; expected units: {units}")
    print("\nNotes:")
    print("- Each value should be an xarray.DataArray with a 'time' dimension.")
    print("- Spatial dimensions (lat/lon) are optional depending on the variable.")
    print("- It is strongly recommended to include a 'units' attribute.")


def help():
    show_expected_ds_format()


def _grids_match(da, target_grid):
    """check if a DataArray already sits on the target grid -- if so, skip regridding"""
    import numpy as np
    for dim in ['lat', 'lon']:
        if dim not in da.coords or dim not in target_grid.coords:
            return False
        if len(da[dim]) != len(target_grid[dim]):
            return False
        if not np.allclose(da[dim].values, target_grid[dim].values, atol=1e-4):
            return False
    return True


# ocean vars -- skip land masking for these
_OCEAN_VARS = {"tos", "sos"}

# vars that may have a vertical dimension -- extract surface before regridding
# so we don't regrid the full column unnecessarily
_SURFACE_VARS = {"o3", "mmrbc", "mmrdust", "mmroa", "mmrso4", "mmrss"}

# applied after regridding and land masking
_ANTARCTIC_LAT = -60


def prepare_inputs(ds_dict, spatial_chunk="auto",
                   model_grid_file=None, regrid=True, regrid_method="bilinear",
                   coastal_mask_file=None, coastal_mask_var="coastal_mask",
                   mask_land=True, land_mask_file=None, land_mask_var="land_mask"):
    """
    Chunk all DataArrays in ds_dict for efficient computation.
    Time is kept as one contiguous chunk (required for quantile/groupby ops).
    Spatial dimensions are chunked to spatial_chunk.
    Always rechunks -- safe to call on numpy or already-dask inputs.

    Steps applied in order:
      - surface extraction for column vars (o3, mmr*) -- done before regrid
         so only the surface level is regridded, not the full column
      - regrid to target grid (skipped if grids match or regrid=False)
      - land mask (skipped for ocean vars tos/sos, or if mask_land=False)
      - drop antarctica (lat < -60)
      - coastal mask (tos/sos only)
      - chunk + drop bounds

    if a user skips prepare_inputs, surface extraction still happens as a
    fallback inside the metric functions (o3, pm25) themselves.

    Parameters
    ----------
    ds_dict : dict
        dict of xr.DataArrays keyed by CMIP6 shortname
    spatial_chunk : int or 'auto'
        chunk size for spatial dimensions
    model_grid_file : str, optional
        path to target grid file for regridding. if None, regridding is skipped.
    regrid : bool
        set to False to skip regridding entirely. default True.
    regrid_method : str
        regridding method passed to xesmf (default 'bilinear')
    coastal_mask_file : str, optional
        path to coastal mask file (used for tos/sos only)
    coastal_mask_var : str
        variable name in the coastal mask dataset (default 'coastal_mask')
    mask_land : bool
        apply a land mask to non-ocean variables. default True.
        set to False to skip land masking entirely.
    land_mask_file : str, optional
        path to land mask file. the mask should be True over land (will be set to NaN).
        if None and mask_land=True, masking is skipped with a warning.
    land_mask_var : str
        variable name in the land mask dataset (default 'land_mask')

    Returns
    -------
    dict of chunked xr.DataArrays
    """
    import numpy as np
    xr.set_options(keep_attrs=True)
    print("preparing inputs for efficient computation...")

    # load target grid once up front if needed
    target_grid = None
    if model_grid_file is not None and regrid:
        target_grid = xr.open_dataset(model_grid_file)

    ds_dict_prepared = {}
    for key, da in ds_dict.items():

        # surface extraction for column vars -- before regrid so we don't
        #    regrid the full column. _get_surface is a no-op if no vertical dim.
        if key in _SURFACE_VARS:
            has_vert = any(d in da.dims for d in ["lev", "plev"])
            if has_vert:
                print(f"  {key}: extracting surface level before regrid...")
                da = _get_surface(da, key)

        # regridding
        if target_grid is not None:
            if _grids_match(da, target_grid):
                print(f"  {key}: already on target grid -- skipping regrid")
            else:
                print(f"  {key}: regridding...")
                da = _regrid_xr(da, target_grid, method=regrid_method, name=key)
        elif not regrid:
            pass  # user explicitly disabled regrid

        # land mask (skip for ocean vars)
        # load land mask once up front and validate its grid
        land_mask = None
        if mask_land:
            if land_mask_file is not None:
                land_mask_ds = xr.open_dataset(land_mask_file)
                land_mask = land_mask_ds[land_mask_var]
                # check that the land mask grid matches the data (or target grid if regridding)
                reference = target_grid if target_grid is not None else next(iter(ds_dict.values()))
                if not _grids_match(land_mask, reference):
                    raise ValueError(
                        "land mask grid does not match the model output / target grid. "
                        "options: (1) pass a land_mask_file that matches your model output or target grid, "
                        "(2) run prepare_inputs with regrid=True and a matching model_grid_file, "
                        "or (3) set mask_land=False to skip land masking."
                    )
            else:
                print("  WARNING: mask_land=True but no land_mask_file provided -- skipping land masking.")


        if land_mask is not None and key not in _OCEAN_VARS:
            da = da.where(land_mask)  # land_mask True = land, set to NaN

        # drop antarctica
        if "lat" in da.coords:
            da = da.where(da.lat > _ANTARCTIC_LAT)

        # coastal mask (ocean vars only)
        if key in _OCEAN_VARS:
            if coastal_mask_file is not None:
                coastal_mask = xr.open_dataset(coastal_mask_file)
                da = da.where(coastal_mask[coastal_mask_var])
            else:
                print(f"  coastal mask file not provided -- tos/sos will not be masked to coastal cells")

        # chunk + drop bounds
        chunk_dict = {dim: -1 if dim == "time" else spatial_chunk for dim in da.dims}
        ds_dict_prepared[key] = _drop_all_bounds(da.chunk(chunk_dict))
        #print(f"  {key}: chunks {chunk_dict}")

        has_unit = any(k.lower() in ["unit", "units"] for k in da.attrs)
        if not has_unit:
            print(f"  WARNING: '{key}' has no 'units' attribute -- units will be guessed. add units attr to avoid errors.")

    print("input preparation complete.")
    return ds_dict_prepared


def calculate_base_period_percentiles(
    tas=None,
    tasmax=None,
    tasmin=None,
    pr=None,
    base_years=_DEFAULT_BASE_YEARS,
    tas_calday_percentiles=_DEFAULT_PERCENTILES["tas_calday"],
    pr_percentiles=_DEFAULT_PERCENTILES["pr"],
    tasmin_percentiles=_DEFAULT_PERCENTILES["tasmin"],
    rx5day_percentiles=_DEFAULT_PERCENTILES["rx5day"],
    wet_day_threshold=1.0,
):
    """
    Calculate climatological percentile thresholds over a base period.
    Used by HWF, TNXp, PR1day, PR5day, SPI, SPEI.

    Run prepare_inputs(ds_dict) before calling this for best performance.
    All operations are dask-native -- no data loaded into memory until .compute().

    Parameters
    ----------
    tas : xr.DataArray, optional
        Daily mean surface temperature. Any units (K or degC) -- converted to degC.
    tasmax : xr.DataArray, optional
        Daily maximum surface temperature. Any units -- converted to degC.
    tasmin : xr.DataArray, optional
        Daily minimum surface temperature. Any units -- converted to degC.
    pr : xr.DataArray, optional
        Daily precipitation. Any units -- converted to mm day-1.
    base_years : tuple of (int, int)
        Start and end years inclusive. Default: (1980, 2014).
    tas_calday_percentiles : list of float
        Calendar-day percentile(s) for tas. Default: [90].
    pr_percentiles : list of float
        All-year wet-day percentile(s) for pr. Default: [90, 95, 98, 99.5].
    tasmin_percentiles : list of float
        All-year cold-tail percentile(s) for tasmin. Default: [10, 5, 2, 0.5].
    rx5day_percentiles : list of float
        Percentile(s) of annual max 5-day pr. Default: [90, 95, 98, 99.5].
    wet_day_threshold : float
        Minimum pr (mm day-1) to count as a wet day. Default: 1.0.

    Returns
    -------
    base_dict : dict
        Keys include 'tas', 'tasmax', 't{p}p_calday', 'tasmin_{p}p', 'pr_{p}p', 'rx5day_{p}p'.
        All DataArrays carry attrs: software_version, base_period_start/end, percentile, units.
    """
    xr.set_options(keep_attrs=True)
    base_start, base_end = int(base_years[0]), int(base_years[1])

    if not any(da is not None for da in [tas, tasmax, tasmin, pr]):
        raise ValueError("at least one of tas, tasmax, tasmin, or pr must be provided.")

    def _slice_to_base(da, var_name):
        years_in_data = da.time.dt.year
        data_start, data_end = int(years_in_data.min()), int(years_in_data.max())
        if base_start == _DEFAULT_BASE_YEARS[0] and base_end == _DEFAULT_BASE_YEARS[1]:
            if data_start > base_start or data_end < base_end:
                print(
                    f"WARNING ({var_name}): data covers {data_start}–{data_end}, "
                    f"which doesn't fully span the default base period "
                    f"{base_start}–{base_end}. proceeding with available years."
                )
        da_base = da.sel(time=da.time.dt.year.isin(range(base_start, base_end + 1)))
        if da_base.time.size == 0:
            raise ValueError(
                f"{var_name}: no data found within base period {base_start}–{base_end}. "
                f"data covers {data_start}–{data_end}."
            )
        return da_base, int(da_base.time.dt.year.min()), int(da_base.time.dt.year.max())

    def _base_attrs(percentile, units, var_name, actual_start, actual_end, notes):
        return {
            "software_version": SOFTWARE_VERSION,
            "base_period_start": actual_start,
            "base_period_end": actual_end,
            "base_period_source": "default" if (
                base_start == _DEFAULT_BASE_YEARS[0] and base_end == _DEFAULT_BASE_YEARS[1]
            ) else "custom",
            "percentile": percentile,
            "units": units,
            "variable": var_name,
            "calculation_notes": notes,
        }

    def _unpack_quantiles(da_q, percentiles, key_fmt, units, var_name,
                          actual_start, actual_end, notes_fmt):
        """split a batched multi-percentile quantile result into individual DataArrays"""
        results = {}
        single = "quantile" not in da_q.dims
        for p in percentiles:
            da_p = da_q.drop_vars("quantile", errors="ignore") if single else \
                   da_q.sel(quantile=p / 100.0).drop_vars("quantile")
            key = key_fmt(p)
            da_p.name = key
            da_p.attrs = _base_attrs(
                percentile=p, units=units, var_name=var_name,
                actual_start=actual_start, actual_end=actual_end,
                notes=notes_fmt(p),
            )
            results[key] = da_p
        return results

    base_dict = {}

    # tas -- calendar-day percentiles
    if tas is not None:
        print("calculating tas base period percentiles...")
        tas_base, actual_start, actual_end = _slice_to_base(
            _check_and_convert_units(da=tas, input_var="tas", conv_type="C"), "tas"
        )
        tas_base = tas_base.chunk({"time": -1})
        base_dict["tas"] = tas_base  # stored for HWF annual mean comparison

        q_vals = [p / 100.0 for p in tas_calday_percentiles]
        tXp_all = tas_base.groupby("time.dayofyear").quantile(q_vals, dim="time", skipna=True)
        base_dict.update(_unpack_quantiles(
            tXp_all, tas_calday_percentiles,
            key_fmt=lambda p: f"t{int(p)}p_calday",
            units="degC", var_name="tas",
            actual_start=actual_start, actual_end=actual_end,
            notes_fmt=lambda p: f"calendar-day {p}th percentile of tas (degC). one value per dayofyear per grid cell.",
        ))

    # tasmax -- stored for HWF (no percentile calc needed, just a slice)
    if tasmax is not None:
        print("converting tasmax to degC for base period storage...")
        tasmax_base, *_ = _slice_to_base(
            _check_and_convert_units(da=tasmax, input_var="tasmax", conv_type="C"), "tasmax"
        )
        base_dict["tasmax"] = tasmax_base

    # tasmin -- all-year cold-tail percentiles
    if tasmin is not None:
        print("calculating tasmin base period percentiles...")
        tasmin_base, actual_start, actual_end = _slice_to_base(
            _check_and_convert_units(da=tasmin, input_var="tasmin", conv_type="C"), "tasmin"
        )
        tasmin_base = tasmin_base.chunk({"time": -1})

        q_vals = [p / 100.0 for p in tasmin_percentiles]
        tnp_all = tasmin_base.quantile(q_vals, dim="time", skipna=True)
        base_dict.update(_unpack_quantiles(
            tnp_all, tasmin_percentiles,
            key_fmt=lambda p: f"tasmin_{str(p).replace('.', 'pt')}p",
            units="degC", var_name="tasmin",
            actual_start=actual_start, actual_end=actual_end,
            notes_fmt=lambda p: f"all-year {p}th percentile of tasmin (degC). one value per grid cell.",
        ))

    # pr -- wet-day percentiles + rx5day percentiles
    if pr is not None:
        print("calculating pr base period percentiles...")
        pr_base, actual_start, actual_end = _slice_to_base(
            _check_and_convert_units(da=pr, input_var="pr", conv_type="mm day-1"), "pr"
        )
        pr_base = pr_base.chunk({"time": -1})
        base_dict["pr"] = pr_base  # stored for SPI/SPEI

        pr_wet = pr_base.where(pr_base >= wet_day_threshold)
        q_vals = [p / 100.0 for p in pr_percentiles]
        prp_all = pr_wet.quantile(q_vals, dim="time", skipna=True)
        base_dict.update(_unpack_quantiles(
            prp_all, pr_percentiles,
            key_fmt=lambda p: f"pr_{str(p).replace('.', 'pt')}p",
            units="mm day-1", var_name="pr",
            actual_start=actual_start, actual_end=actual_end,
            notes_fmt=lambda p: (
                f"all-year {p}th percentile of pr on wet days "
                f"(>= {wet_day_threshold} mm day-1). one value per grid cell."
            ),
        ))

        # rx5day -- uses the already-rechunked pr_base
        print("calculating rx5day base period percentiles...")
        annual_max_rx5 = (
            pr_base
            .rolling(time=5, min_periods=5)
            .sum()
            .groupby("time.year")
            .max(dim="time", skipna=True)
        )
        annual_max_rx5 = annual_max_rx5.chunk({"year": -1})

        q_vals = [p / 100.0 for p in rx5day_percentiles]
        rx5p_all = annual_max_rx5.quantile(q_vals, dim="year", skipna=True)
        base_dict.update(_unpack_quantiles(
            rx5p_all, rx5day_percentiles,
            key_fmt=lambda p: f"rx5day_{str(p).replace('.', 'pt')}p",
            units="mm", var_name="rx5day",
            actual_start=actual_start, actual_end=actual_end,
            notes_fmt=lambda p: (
                f"all-year {p}th percentile of annual maximum 5-day accumulated "
                f"precipitation (mm). rolling sum uses min_periods=5. one value per grid cell."
            ),
        ))

    print(f"base period percentiles complete. keys: {list(base_dict.keys())}")
    return base_dict

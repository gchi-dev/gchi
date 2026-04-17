import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
from scipy import stats

import newt
SOFTWARE_VERSION = "0.0.0"

# code reminders
# might have to back filled nan-ed coastal grid cells in vibrio. see calc_thresholds project dir for a method
# have a summary output that summarizes the software version, where default thresholds applied, and any approximations (aka simplified equations) used
# make a func to convert the dims to time/lat/lon 
# test utci with tmrt calculation
# WBT is NEWT
# WBGT is first try - modified Brimicombe et al. (2023) - using NEWT instead of Stull. Only able to do this if we have the variables to calculate mean radiative temperature. If not, used the weighted approach in Shwingshackl. 
# think about whether to use tasmax or tas for heat stress variables.maybe give user option to swithch. Because for UTCI it is mean rad temp vs tas but we use TX in the main variable. Cannot get mrt max of the day unfortunately. maybe just use average temp for utci 
# ^^ for now I think it is best to use TA if calculating MRT so you don't underestimate UTCI on sunny days due to not accounting for peak sun. Use TX otherwise. For WBGT still use TX . make an argument so users can pick which var to use for all functs between tx and ta
# default should be schingshackl, but option to do more explicit calc
# pull software version from somewhere else
# warn users against default HWF detrend if only a few years of data because we remove the trend of the study period itself if only annual data then trend prob 0 and results will appear seasonal
# check HWF calculation to ensure it makes sense the detrending part 
# prob make calculate base percentiles faster
# add level attributes to the exceedance days func 
# add UTCI cold
# TNXp might not be the best metric because if days > 3 thats normal for the 90th percentile value. no level relevance. it should exceed the relative number of days of each percentile right? maybe do it that way. or drop the metric
# add metadata to levels and what not 
# in preprocess, if tsteps > 365 then resample and mean (most vars) sum for accum (precip)
# TNXp and R1day indices - need to be > # days percentile 
# R5day check this, but I think don't need minimum days because already 5-day annual max, so if 5-day exceeds its already exceeding 3 days. 
# masks needed
# maybe add an fwi option to just have 4 levels as fallback 
# in aq exceedance func and elsewhere allow the opportunity to do daily thresholds the normal way if the data are daily 
# for fwi, see if want to keep as is with envrionemtnal zones read-in then thresholds calc, or just pull thresholds file (preferred)
# for fwi, give option to overwrite with plain threshols like in ozone
# the prep messes up units attr. 
# check whether base dict should be da input or ds bc spi expects da 
# ozone level check (optional) orders by plev if lev coord then the top is the one with less nans or if no nans at surf, top is one with more ozone. opposite for ozone. show a figure in one of the papers. prob the joss one 

# =================
# !! THRESHOLDS DICTIONARY !!
# =================
hazard_thresholds = {
    "AT": [28, 32, 35, 40],  # °C
    "HI": [27, 32, 41, 54],  # °C
    "Hu": [30, 40, 45, 54],  # °C
    "WBT": [27.4, 28.9, 30.3, 35],  # °C
    "WBGT": [29, 30.5, 32, 37],  # °C
    "UTCIhot": [26, 32, 38, 46],  # °C
    "HWF": [0.052, 0.077, 0.110, 0.173],  # fraction of year
    "TR": [0.312, 0.532, 0.918, 0.997],  # fraction of year
    "TXC": [30, 35, 40, 45],  # °C
    "UTCIcold": [0, -13, -27, -40],  # °C
    "TNXp": [10, 5, 2, 0.5],  # unit percentile
    "FI": [0.852, 1.125, 1.579, 2.418], # FI index, unitless 
    "FWI": [12.23, 22.95, 36.83, 50],  # index
    "HDW": [0.011, 0.017, 0.027, 0.041],  # HDW index, unitless 
    "O3mon": [60, 65, 70, 100],  # ug/m^3
    "O3day": [100, 110, 120, 160],  # ug/m^3
    "PM2pt5mon": [5, 15, 25, 35],  # ug/m^3
    "PM2pt5day": [15, 37.5, 50, 75],  # ug/m^3
    "CDD": [0.86, 0.951, 0.992, 0.999], # fraction of year 
    "VSmalaria": [0, 0.333, 0.583, 0.833],  # fraction of year: marginal - endemic
    "VSzika": [0, 0.333, 0.583, 0.833],  # fraction of year: marginal - endemic
    "VSdengueAeg": [0, 0.333, 0.583, 0.833],  # fraction of year: marginal - endemic
    "VSdengueAlb": [0, 0.333, 0.583, 0.833],  # fraction of year: marginal - endemic
    "PR1day": [90, 95, 98, 99.5], # percentile
    "PR5day": [90, 95, 98, 99.5], # percentile
    "PRXmm": [20, 30, 40, 50], # mm/day
    "SPI": [-0.8, -1.3, -1.6, -2],  # index unitless
    "SPEI": [-0.8, -1.3, -1.6, -2],  # index unitless
    "VbrS": [0, 0.083, 0.167, 0.417],  # actual percentiles where 0, 0, 0.167, 0.417, but to differentiate L1 and L2 1 month is chosen for L2 (approx 97.2th percentile)
}

# thresholds from Table 1 Kudlackova et al 2025 (https://iopscience.iop.org/article/10.1088/1748-9326/ad97cf#erlad97cffA1)
fwi_thesholds = {
    'D': [5.06, 10.9, 16.83, 22.45],
    'E': [7.59, 15.75, 24.01, 30.98],
    'F': [6.65, 14.1, 24.08, 36.7],
    'G': [8.97, 18.38, 31.41, 46.85],
    'H': [9.95, 20.73, 35.1, 51.26],
    'I': [20.73, 33.82, 48.65, 61.98],
    'J': [13.73, 23.33, 36.7, 52.44],
    'K': [18.1, 28.04, 40.82, 55.14],
    'L': [20.86, 32.19, 46.13, 61.99],
    'M': [5.43, 11.41, 21.83, 37.91],
    'N': [10.28, 20.44, 35.18, 54.96],
    'O': [10.28, 20.44, 35.18, 54.96], # copied from N (no thresholds in Kudlackova, most will be masked by fuel mask)
    'P': [5.95, 12.61, 30.85, 62.0], # copied from Q (no thresholds in Kudlackova, most will be masked by fuel mask)
    'Q': [5.95, 12.61, 30.85, 62.0],
    'R': [6.57, 12.86, 23.64, 39.84]
    }

def show_expected_ds_format():
    """
    Prints information about the expected input dataset dictionary for this package.
    
    - Users should provide a dictionary `ds_dict` where keys are the standard shortnames
      listed below, and values are xarray DataArrays.
    - Each DataArray should have a 'time' dimension and optionally spatial dimensions (lat/lon).
    - Each variable should ideally have a 'units' attribute. Known / expected units are listed.
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
        "sea_surface_temperature": "tos"
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
        "tos": "degC"
    }
    
    print("\nExpected input dictionary `ds_dict` format:\n")
    print("\n`ds_dict` should have variable keys linked to an Xarray data array. For example:")
    print("ds_dict = {'tasmax': tasmax_xr_dataset.tasmax,' 'pr': pr_xr_dataset.pr,...}\n")
    print("These data arrays serve as the input for the index calculations and should follow the format below\n")
    print("Key in ds_dict (shortname) : Description / expected units\n")
    
    for desc, shortname in variables.items():
        units = expected_units.get(shortname, "unknown")
        print(f"{shortname:<12} : {desc} ; expected units: {units}")
    
    print("\nNotes:")
    print("- Each value in ds_dict should be an xarray.DataArray with a 'time' dimension.")
    print("- Spatial dimensions (lat/lon) are optional depending on the variable.")
    print("- It is recommended to include a 'units' attribute in each DataArray.")
    print("- The program will automatically attempt unit conversion for known units.")

def help():
    """
    Potentially useful information for running program
    """
    show_expected_ds_format()


# =================
# !! BASE FUNCTIONS !!
# =================    
def _sanity_check_units(da: xr.DataArray, units_attr: str):
    """
    Check data value range based on units specified 
    """
    # drop bnd coordinates
    da = _drop_all_bounds(da)
    # ------- 4) sanity checks -------
    try:
        if "time" in da.dims:
            sample = da.isel(time=0)
        else:
            sample = da
        minv = float(sample.min(skipna=True))
        maxv = float(sample.max(skipna=True))
    except Exception as e:
        print(f"Data value spot check failed: {e}\nSkipping units spot check. Recommended: Add units attributes and re-run")
        minv, maxv = np.nan, np.nan

    # temperature 
    if units_attr == "C":
        if not (-100 < minv < 60 and -100 < maxv < 60):
            print(f"WARNING: values {minv:.1f}–{maxv:.1f} outside reasonable range for temperature in °C.")
    elif units_attr == "K":
        if not (150 < minv < 400 and 150 < maxv < 400):
            print(f"WARNING: values {minv:.1f}–{maxv:.1f} outside reasonable range for temperature in K.")
    elif units_attr == "F":
        if not (-150 < minv < 140 and -150 < maxv < 140):
            print(f"WARNING: values {minv:.1f}–{maxv:.1f} outside reasonable range for temperature in °F.")
    # humidity / fraction
    elif units_attr in ["fraction"]:
        if not (0 <= minv and maxv <= 1.10):
            print(f"WARNING: values {minv:.1f}–{maxv:.1f} outside reasonable range for relative humidity units '{units_attr}'. Models can simulate non-physical RH values < 0 or > 1. Values clipped to 0.01-0.999999.")
    elif units_attr in ["%"]:
        if not (0 <= minv and maxv <= 110):
            print(f"WARNING: values {minv:.1f}–{maxv:.1f} outside reasonable range for relative humidity units '{units_attr}'. Models can simulate non-physical RH values < 0 or > 100 %. Values clipped to 0.1-99.9999.")
    # pressure 
    elif units_attr in ["hPa"]:
        if not (100 <= minv <= 1200 and 100 <= maxv <= 1200):
            print(f"WARNING: values {minv:.1f}–{maxv:.1f} outside reasonable range for pressure units '{units_attr}'.")
    # precip 
    elif units_attr == "mm day-1":
        if maxv > 300:
            print(f"WARNING: max value {maxv:.3f} unusually large for precipitation (mm/day): check units.")
        elif minv < -0:
            print(f"WARNING: min value {minv:.3f} negative for precipitation (mm/day): check units.")
    
    # wind
    elif units_attr == "m s-1":
        if maxv > 100:
            print(f"WARNING: max value {maxv:.3f} unusually large for wind speed (m s-1): check units.")
        if minv < 0:
            print(f"WARNING: min value {minv:.3f} negative for wind speed (m s-1): ensure input is wind speed (sfcWind or sqrt(u**2 + v**2)) not u or v vector. check units.")
    
    elif units_attr == "km h-1":
        if maxv > 360:
            print(f"WARNING: max value {maxv:.3f} unusually large for wind speed (km h-1): check units.")
        if minv < 0:
            print(f"WARNING: min value {minv:.3f} negative for wind speed (km h-1): ensure input is wind speed (sfcWind or sqrt(u**2 + v**2)) not u or v vector. check units.")

    # salinity
    elif units_attr == "psu":
        if maxv > 43:
            print(f"WARNING: max value {maxv:.3f} unusually large for sea surface salinity (psu): check units.")
        if minv < 5:
            print(f"WARNING: min value {minv:.3f} unusually small for sea surface salinity (psu): ensure input is sea surface salinity (0.001 or psu).")

    # pm2.5 aerosols
    elif units_attr == "kg kg-1":
        if maxv > 1e-4:
            print(f"WARNING: max value {maxv:.3f} unusually large for aerosol inputs (kg kg-1): check units.")
        if minv < 0:
            print(f"WARNING: min value {maxv:.3f} negative for aerosol inputs in (kg kg-1): check units.")

def _check_and_convert_units(da: xr.DataArray, input_var: str, conv_type: str):
    """
    Check and convert the units of a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        the input data array
    conv_type : str
        target unit type (one of 'C','K','F','fraction','%','hPa','Pa','mm day-1')

    Returns
    -------
    xr.DataArray
        converted data array with updated units attribute
    """
    da_out = da.copy()

    # ------- 1) extract and normalize attribute strings -------
    raw_units = None
    for k in da.attrs:
        if k.lower() in ["unit", "units"]:
            raw_units = da.attrs[k]
            break
    
    # normalize unit string for comparison
    if raw_units:
        u = str(raw_units).lower().replace(" ", "").replace("degrees", "").replace("deg","").replace("°","")
    else:
        u = None
    
    # known normalized categories
    # we expect most users will input CMIP6 output so units should be standard, but users may use alternative input datasets
    # here is a non-exhaustive list of unit attribute names for relevant variables
    temp_c = {"c","celsius","celsius","centigrade"}
    temp_k = {"k","kelvin"}
    temp_f = {"f","fahrenheit"}
    percent = {"%","percent","pct"}
    frac = {"fraction","frac"}
    pa = {"pa","pascal","pascals"}
    hpa = {"hpa","mb","millibar","millibars"}
    precip_kg_ms = {"kgm-2s-1","kg/m2s","kgm2s-1","kgm^-2s^-1"}  
    precip_mm_day = {"mmday-1","mm/day"}
    wind_m_s = {"ms-1", "m/s", "ms^-1"}
    wind_km_h = {"kmh-1", "km/h", "kmh^-1"}
    salinity_psu = {"psu", "practicalsalinityunits"}
    pm_kg_kg = {"kgkg-1", "kg/kg", "kgkg^-1"}

    # unify to internal label
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

    # ------- 2) if no unit attribute, try guessing from data ------- 
    guessed = False
    if units_attr is None:
        guessed = True
        # use 1% and 99% quantiles for a quick, efficient range check
        try:
            if "time" in da.dims:
                sample = da.isel(time=0)
            else:
                sample = da
            minv = float(sample.min(skipna=True))
            maxv = float(sample.max(skipna=True))
        except Exception as e:
            print(f"Data value spot check failed: {e}\nSkipping units spot check. Recommended: Add units attributes and re-run")
            minv, maxv = np.nan, np.nan
        # guess temperature
        if (minv == np.nan) | (maxv == np.nan):
            units_attr = "unknown"
        elif conv_type in ["C", "F", "K"]:
            if -60 < minv < 60 and -50 < maxv < 60:
                units_attr = "C"
            elif 120 < minv < 370 and 150 < maxv < 370:
                units_attr = "K"
            elif -60 < minv < 140 and 32 < maxv < 140:
                units_attr = "F"
        # guess relative humidity
        elif conv_type in ["%", "fraction"]:
            if maxv <= 10:
                units_attr = "fraction"
            elif maxv > 10:
                units_attr = "%"
        # guess pressure 
        elif conv_type in ["Pa", "hPa"]:
            if 100 < minv < 1200 and 100 < maxv < 1200:
                units_attr = "hPa"
            elif 10000 < minv < 120000 and 10000 < maxv < 120000:
                units_attr = "Pa"
        # precipitation (typical range for kg m-2 s-1)
        elif conv_type in ["mm day-1"]:
            if 0 <= minv <= 0.005 and 0 <= maxv <= 0.02:
                units_attr = "kg m-2 s-1"
            elif 0 <= minv <= 300 and 0 <= maxv <= 300:
                units_attr = "mm day-1"
        # wind
        elif conv_type in ["m s-1", "km h-1"]:
            if maxv < 100:
                print("No units attribute found for wind speed. Assuming units are m s-1. Please check units.")
                units_attr = "m s-1"
            else:
                print("No units attribute found for wind speed. Assuming units are km h-1. Please check units.")
                units_attr = "km h-1"
        # salinity
        elif conv_type in ["psu"]:
            print("No units attribute found for sea surface salinity. Assuming units are in psu. Please check units.")
        # pm2.5 
        elif conv_type in ["kg kg-1"]:
            print("No units attribute found for PM inputs. Assuming units are in kg kg-1. Please check units.")
        # else 
        else:
            units_attr = "unknown"
        
        if units_attr not in [None, "unknown"]:
            print(f"Guessed {input_var} units as '{units_attr}' based on sampled data values. min: {round(minv, 3)} max: {round(maxv, 3)}.")
        else:
            print(f"Could not guess {input_var} units based on values. min: {round(minv, 3)} max: {round(maxv, 3)}.\nPlease check units, add units attribute, and re-run.")

    # if units attr is correct, check if reasonable range
    if units_attr == conv_type:
        # Update units attr
        da_out.attrs["units"] = conv_type
        _sanity_check_units(da = da_out, units_attr = conv_type)
    else:
        # ------- Do conversions -------
        # Temperature conversions
        if units_attr == "C":
            if conv_type == "K":
                da_out = da + 273.15
            elif conv_type == "F":
                da_out = da * 9/5 + 32
        elif units_attr == "K":
            if conv_type == "C":
                da_out = da - 273.15
            elif conv_type == "F":
                da_out = (da - 273.15) * 9/5 + 32
        elif units_attr == "F":
            if conv_type == "C":
                da_out = (da - 32) * 5/9
            elif conv_type == "K":
                da_out = (da - 32) * 5/9 + 273.15

        # % <-> fraction
        if units_attr == "%" and conv_type == "fraction":
            da_out = da / 100
            da_out = da_out.clip(0,1) # clip range to be physically meaningful
        elif units_attr == "fraction" and conv_type == "%":
            da_out = da * 100
            da_out = da_out.clip(0,100) # clip range to be physically meaningful

        # Pressure conversions
        if units_attr == "Pa" and conv_type in ["hPa"]:
            da_out = da / 100
        elif units_attr in ["hPa"] and conv_type == "Pa":
            da_out = da * 100

        # Precipitation conversions
        if units_attr == "kg m-2 s-1" and conv_type == "mm day-1":
            da_out = da * 86400  # kg m^-2 s^-1 -> mm/day
        
        # Wind conversions
        if units_attr == "kts":
            if conv_type == "m s-1":
                da_out = da * 0.5144444 # kts to m s-1
            elif conv_type == "km h-1":
                da_out = da * 1.852 # kts to km h-1 
        elif units_attr == "m s-1":
            if conv_type == "km h-1":
                da_out = da * 3.6 # m s-1 to km h-1 
        elif units_attr == "km h-1":
            if conv_type == "m s-1":
                da_out = da * 0.2777778 # km h-1 to m s-1

        # salinity conversions
        if units_attr in ["0.001"]:
            if conv_type == "psu":
                da_out = da # 1 part per thousand ~ 1 psu
    

        # Update units attr
        da_out.attrs["units"] = conv_type
        da_out.attrs["units_guessed"] = str(guessed)
        _sanity_check_units(da=da_out, units_attr=conv_type)

    return da_out

def _get_tsteps(da):
    """
    count tsteps in a year for annual fraction output
    """
    # get number of time steps in a year
    try:
        steps_per_year = da.groupby('time.year').count('time')
        # Number of time steps per year
        steps_per_year = da.groupby('time.year').count('time')
        steps_per_year.attrs["units"] = "time steps yr-1"
    except:
        steps_per_year = 365
        print("Number of time steps could not be calculated. Assuming input data are daily data.")

    return steps_per_year

def _ann_frac(da, steps_per_year):
    """
    Check the time resolution of a data array
    Useful for "fraction of year" calculation
    """
    ann_frac = da.groupby('time.year') / steps_per_year
    
    return ann_frac


def _annual_exceedance_frac(da, hazard_thresholds, var_name, exceedance_dir="above"):
    """
    Count annual days exceeding thresholds, keeping all spatial dimensions.
    
    da: xarray DataArray with 'time' dimension
    hazard_thresholds: list/array of thresholds for levels (will be sorted)
    var_name: name of output DataArray
    
    Returns: DataArray with dims ('time', 'level', ...) where ... are original spatial dims
             and attribute 'level_values' storing thresholds
    """
    if exceedance_dir.lower() == "below": # e.g. days < 0°C (cold days)
        thresholds = np.sort(hazard_thresholds)[::-1] # reverse dir because < thresh
    else:
        thresholds = np.sort(hazard_thresholds)

    steps_per_year = _get_tsteps(da)
    
    da_list = []
    for _, th in enumerate(thresholds):
        # Boolean mask -> count per year while keeping spatial dims
        #da_count = da.where(da > th).resample(time='1YE').count(dim='time').where(nan_mask)
        if exceedance_dir.lower() == "above": # e.g. days > 40°C
            da_count = (da > th).resample(time='1YE').sum(dim='time', skipna=True)
        elif exceedance_dir.lower() == "below": # e.g. days < 0°C (cold days)
            da_count = (da < th).resample(time='1YE').sum(dim='time', skipna=True)
        else:
            print(f"Exceedance direction {exceedance_dir} not recognized. Must be 'above' or 'below'")

        da_list.append(da_count)
    
    # Concatenate along new 'level' dimension
    da_exceed = xr.concat(da_list, dim='level')
    da_exceed = da_exceed.assign_coords(level=np.arange(1, len(thresholds)+1))
    da_exceed.attrs['level_values'] = thresholds.tolist()
    da_exceed = _ann_frac(da_exceed, steps_per_year).rename(var_name)
    
    return da_exceed

def _annual_exceedance_frac_aq(da, hazard_thresholds, var_name, exceedance_dir="above"):
    """
    Conditional annual exceedance: only count timestep exceedances in years where
    the annual average itself crosses the threshold. Otherwise, count = 0 (or NaN if all NaN).
    
    da: xarray DataArray with 'time' dimension
    hazard_thresholds: list/array of thresholds for levels (will be sorted)
    var_name: name of output DataArray
    Returns: DataArray with dims ('time', 'level', ...) where ... are original spatial dims
             and attribute 'level_values' storing thresholds
    """
    if exceedance_dir.lower() == "below":
        thresholds = np.sort(hazard_thresholds)[::-1]
    else:
        thresholds = np.sort(hazard_thresholds)

    steps_per_year = _get_tsteps(da)
    # implement daily logic here 
    # if steps_per_year.mean(["lat", "lon"]) > 15:
    # print("assuming this is daily data. doing daily aq threshold")
    # da = da.resample(time="1D").mean()
    #  return _annual_exceedance_frac(da, hazard_thresholds, var_name, exceedance_dir="above")

    # Get annual mean, if concentration > annual threshold, proceed to count months that exceed. else 0 exceedances. 
    da_annual_mean = da.resample(time='1YE').mean(dim='time')
    all_nan_mask = da.isnull().resample(time='1YE').all(dim='time')

    da_list = []
    for th in thresholds:
        # Raw timestep exceedance count per year
        if exceedance_dir.lower() == "above":
            da_count = (da > th).resample(time='1YE').sum(dim='time', skipna=True)
            # did the annual mean cross the threshold this year?
            annual_mean_crosses = da_annual_mean > th
        elif exceedance_dir.lower() == "below":
            da_count = (da < th).resample(time='1YE').sum(dim='time', skipna=True)
            annual_mean_crosses = da_annual_mean < th
        else:
            raise ValueError(f"Exceedance direction '{exceedance_dir}' not recognized. "
                             "Must be 'above' or 'below'.")

        #    - annual mean crosses threshold -> keep da_count
        #    - annual mean does NOT cross    -> set to 0
        #    - entire year is NaN            -> set to NaN
        da_count_conditional = xr.where(annual_mean_crosses, da_count, 0)
        da_count_conditional = xr.where(all_nan_mask, np.nan, da_count_conditional)

        da_list.append(da_count_conditional)

    # concatenate along new 'level' dimension
    da_exceed = xr.concat(da_list, dim='level')
    da_exceed = da_exceed.assign_coords(level=np.arange(1, len(thresholds) + 1))
    da_exceed.attrs['level_values'] = thresholds.tolist()
    da_exceed = _ann_frac(da_exceed, steps_per_year).rename(var_name)
    return da_exceed

def _annual_exceedance_frac_fwi(da_fwi, da_zones, fwi_thesholds, var_name='FWI'):
    """
    FWI-specific annual exceedance fraction.
    Spatially varying thresholds based on environmental zone.
    
    da_fwi          : xr.DataArray (lat, lon, time)
    da_zones        : xr.DataArray (lat, lon) integer zone codes 1–18
    fwi_thesholds : dict mapping letter -> [t1, t2, t3, t4]
    """
    # Build (lat, lon, level) threshold DataArray
    zone_letters = np.vectorize(
        lambda x: chr(ord('@') + int(x)) if not np.isnan(x) else None
    )(da_zones.values)

    thresh_array = np.full((*zone_letters.shape, 4), np.nan)
    for i in range(zone_letters.shape[0]):
        for j in range(zone_letters.shape[1]):
            letter = zone_letters[i, j]
            if letter in fwi_thesholds:
                thresh_array[i, j, :] = fwi_thesholds[letter]

    thresh_da = xr.DataArray(
        thresh_array,
        dims=['lat', 'lon', 'level'],
        coords={'lat': da_zones.lat, 'lon': da_zones.lon, 'level': [1, 2, 3, 4]}
    )

    # Exceedance per level
    da_list = []
    for lvl in [1, 2, 3, 4]:
        th = thresh_da.sel(level=lvl)
        da_count = (da_fwi > th).resample(time='1YE').sum('time')
        da_list.append(da_count)

    da_exceed = xr.concat(da_list, dim='level').assign_coords(level=[1, 2, 3, 4])

    # Annual fraction
    steps_per_year = _get_tsteps(da_fwi)
    da_exceed = _ann_frac(da_exceed, steps_per_year).rename(var_name)
    da_exceed.attrs['level_values'] = 'spatially varying — see fwi_thesholds'

    return da_exceed

def _assign_hazard_level(da, frac_thresholds=None):
    """
    Assign a hazard level (1–4) per year per grid cell based on the highest
    threshold crossed

    For threshold-based vars (da has a 'level' dimension):
        Uses the precomputed exceedance counts. Level is the highest level
        where days exceeded > 3/365 (frac_thresholds is ignored)

    For single-value vars (da has no 'level' dimension):
        Compares the annual value directly against frac_thresholds
    Returns a Dataset with the original da and a new {name}_hazard_level variable.
    """

    min_days_frac = 0.01 # must exceed 99th percentile annual event (> 0.01 frac of year) to be valid

    if "level" in da.dims:
        hazard_level = xr.zeros_like(da.isel(level=0), dtype=int)
        for i in range(da.level.size):
            hazard_level = hazard_level.where(da.isel(level=i) <= min_days_frac, other=i + 1)
        # restore NaN where da was NaN, otherwise they are written as level 4 be da > th if th is nan 
        hazard_level = hazard_level.where(da.notnull().any('level'))
    else:
        thresholds = np.sort(frac_thresholds)
        hazard_level = xr.zeros_like(da, dtype=int)
        for i, th in enumerate(thresholds):
            hazard_level = hazard_level.where(da <= th, other=i + 1)
        
        # restore NaN where da was NaN, otherwise they are written as level 4 be da > th if th is nan 
        hazard_level = hazard_level.where(da.notnull())

    hazard_level.name = f"{da.name}_hazard_level"
    hazard_level.attrs["calculation_notes"] = (
        "Hazard level 1–4: highest threshold crossed per year per grid cell. "
        "0 = no threshold crossed."
    )

    return xr.merge([da, hazard_level])

def _get_surface(da, var):
    """
    Get lowest non-NaN value along vertical coordinate,
    automatically handling direction, and return only the surface value
    while keeping the vertical dimension (lev or plev) with the corresponding value.
    Fallback for directionality of model level dim: If variable is ozone, surface < top, if aerosol, surface > top
    There should be attributes ('up'/'down') to infer directionality but in practice these metadata are not always correct
    Rather than reconstructing pressure levels with model levels, which may require additional data, just infer the direction
    """

    # cannot index below on a chunked array so compute first if chunked
    try:
        da = da.compute()
    except:
        None
    
    # detect vertical dimension
    if ("lev" not in da.dims) & ("plev" not in da.dims):
        return da 
    vdim = next(d for d in da.dims if d in ["lev", "plev"])
    
    if vdim == "plev":
        # sort descending (high pressure to low pressure, surface first)
        da = da.sortby(vdim, ascending=False)
    else:
        # lev: detect surface direction using NaN pattern / max heuristic
        sample = da.isel({vdim: [0, -1]})
        firstlev_nan = sample.isel({vdim: 0}).isnull().sum()
        lastlev_nan = sample.isel({vdim: -1}).isnull().sum()
        
        if firstlev_nan != lastlev_nan:
            if firstlev_nan > lastlev_nan:
                da = da.isel({vdim: slice(None, None, -1)})
        else:
            # fallback: lowest ozone = surface
            firstlev_max = sample.isel({vdim: 0}).max()
            lastlev_max = sample.isel({vdim: -1}).max()
            if var == "o3":
                if firstlev_max > lastlev_max:
                    da = da.isel({vdim: slice(None, None, -1)})
            else:
                # aerosols greater concentration at surface
                if firstlev_max < lastlev_max:
                    da = da.isel({vdim: slice(None, None, -1)})
    
    # find first non-NaN value along vertical (surface)
    mask = da.notnull()
    #idx = mask[var].argmax(dim=vdim)
    idx = mask.argmax(dim=vdim)
    valid = mask.any(dim=vdim)
    
    # select surface value, keeping vertical dimension
    surface = da.isel({vdim: idx}).where(valid)
    
    return surface

 
# =================
# INPUT PREPARATION
# =================
def _regrid_xr(ds_in, regrid_to, method='bilinear'):
    regridder = xe.Regridder(ds_in, regrid_to, method=method, periodic=True, ignore_degenerate=True)
    ds_out = regridder(ds_in)
    # restore attributes
    for var in ds_out.data_vars:
        ds_out[var].attrs = ds_in[var].attrs
    ds_out.attrs = ds_in.attrs # restore global attrs
    ds_out.attrs["regridded"] = "True"

    return ds_out

# ds_clim_rg = _regrid_xr(ds_clim, target_grid, method='bilinear') # regrid to 1x1

def _drop_all_bounds(da):
    """
    Drop X_bounds dimensions from datasets (can lead to merging issues)
    """
    drop_bnds = [varname for varname in da.coords if (('_bounds' in varname ) | ('_bnds' in varname))]

    return da.drop(drop_bnds)

def prepare_inputs(ds_dict, spatial_chunk="auto", 
                   model_grid_file=None, regrid_method='bilinear', 
                   coastal_mask_file=None, coastal_mask_var="coastal_mask"):
    xr.set_options(keep_attrs=True)
    """
    Chunk all DataArrays in ds_dict for efficient computation.
    time is kept as one contiguous chunk (required for quantile/groupby).
    Spatial dimensions are chunked to spatial_chunk.
    Always rechunks — safe to call on numpy or already-Dask inputs.
    """
    print("Preparing inputs for efficient computation...")
    ds_dict_prepared = {}
    for key, da in ds_dict.items():
        # regrid model if need be 
        if model_grid_file is not None:
            target_grid = xr.open_dataset(model_grid_file)
            da = _regrid_xr(da, target_grid, method=regrid_method)
        else:
            print(f"Model target grid file !!!!! Note to self, implement cloud logic")

        # for vibrio, get coast data 
        if key in ["tos", "sos"]:
            if coastal_mask_file is not None:
                coastal_mask = xr.open_dataset(coastal_mask_file)
                da = da.where(coastal_mask[coastal_mask_var])
            else:
                print(f"coastal mask input file !!!!! Note to self, implement cloud logic")

        chunk_dict = {dim: -1 if dim == "time" else spatial_chunk for dim in da.dims}
        ds_dict_prepared[key] = da.chunk(chunk_dict)
        ds_dict_prepared[key] = _drop_all_bounds(ds_dict_prepared[key]) # drop bnd coordinates
        print(f"  {key}: chunks {chunk_dict}")
        
        # check for units attr
        has_unit = any(k.lower() in ['unit', 'units'] for k in da.attrs)
        if has_unit == False:
            print(f" WARNING: variable {key} 'units' attribute not found. Units will be guessed based on values, but there may be errors in these conversions. It is strongly advised to set a units attribute before running.")

    print("Input preparation complete.")
 
    return ds_dict_prepared
 
# =================
# BASE PERIOD PERCENTILES
# =================
 

_DEFAULT_BASE_YEARS = (1980, 2014)
 
_DEFAULT_PERCENTILES = {
    "tas_calday":  [90],               # calendar-day percentile, one value per day-of-year per grid cell
    "pr":          [90, 95, 98, 99.5], # all-year (wet-day) percentiles
    "tasmin":      [10, 5, 2, 0.5],   # all-year cold-tail percentiles
    "rx5day":      [90, 95, 98, 99.5] # annual 5-day max precip percentiles
}
 
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
    Calculate climatological percentile thresholds over a base period for use in
    hazard exceedance functions (e.g. heatwave_days).
 
    Run prepare_inputs(ds_dict) before calling this for best performance.
    All operations are Dask-native — no data is loaded into memory.
 
    Parameters
    ----------
    tas : xr.DataArray, optional
        Daily mean surface temperature. Any units (K or °C) — converted to °C.
    tasmax : xr.DataArray, optional
        Daily maximum surface temperature. Any units (K or °C) — converted to °C.
    tasmin : xr.DataArray, optional
        Daily minimum surface temperature. Any units (K or °C) — converted to °C.
    pr : xr.DataArray, optional
        Daily precipitation. Any units (kg m-2 s-1 or mm day-1) — converted to mm day-1.
    rx5day: xr.DataArray, optional
        Maximum annual 5-day cumulative precip. Any units (kg m-2 s-1 or mm day-1) — converted to mm day-1.
    base_years : tuple of (int, int), optional
        Start and end years (inclusive). Default: (1980, 2014).
    tas_calday_percentiles : list of float, optional
        Calendar-day percentile(s) for tas. Default: [90].
    pr_percentiles : list of float, optional
        All-year wet-day percentile(s) for pr. Default: [90, 95, 98, 99.5].
    tasmin_percentiles : list of float, optional
        All-year cold-tail percentile(s) for tasmin. Default: [10, 5, 2, 0.05].
    wet_day_threshold : float, optional
        Minimum pr (mm day-1) to count as a wet day. Default: 1.0.
 
    Returns
    -------
    base_dict : dict
        "tas"            — base-period tas (°C), full time series, for computing
                           annual mean in heatwave_days
        "t{p}p_calday"   — calendar-day {p}th percentile of tas (°C);
                           dims: (dayofyear, [lat, lon])
        "tasmax"         — base-period tasmax (°C) for use in heatwave_days
        "tasmin_{p}p"    — all-year {p}th percentile of tasmin (°C)
        "pr_{p}p"        — all-year {p}th percentile of wet-day pr (mm day-1)
        "rx5dayr_{p}p"        — {p}th percentile of max annual 5-day pr (mm)
 
        All DataArrays carry attrs: software_version, base_period_start,
        base_period_end, base_period_source, percentile, units, calculation_notes.
    """
    xr.set_options(keep_attrs=True)
    base_start, base_end = int(base_years[0]), int(base_years[1])
 
    if not any(da is not None for da in [tas, tasmax, tasmin, pr]):
        raise ValueError("At least one of tas, tasmax, tasmin, or pr must be provided.")
 
    def _slice_to_base(da, var_name):
        years_in_data = da.time.dt.year
        data_start, data_end = int(years_in_data.min()), int(years_in_data.max())
        if base_start == _DEFAULT_BASE_YEARS[0] and base_end == _DEFAULT_BASE_YEARS[1]:
            if data_start > base_start or data_end < base_end:
                print(
                    f"WARNING ({var_name}): data cover {data_start}–{data_end}, "
                    f"which does not fully span the default base period "
                    f"{base_start}–{base_end}. Proceeding with available years."
                )
        da_base = da.sel(time=da.time.dt.year.isin(range(base_start, base_end + 1)))
        if da_base.time.size == 0:
            raise ValueError(
                f"{var_name}: no data found within base period {base_start}–{base_end}. "
                f"Data cover {data_start}–{data_end}."
            )
        return da_base, int(da_base.time.dt.year.min()), int(da_base.time.dt.year.max())
 
    def _base_attrs(percentile, units, var_name, actual_start, actual_end, notes):
        return {
            "software_version":   SOFTWARE_VERSION,
            "base_period_start":  actual_start,
            "base_period_end":    actual_end,
            "base_period_source": "default" if (base_start == _DEFAULT_BASE_YEARS[0] and base_end == _DEFAULT_BASE_YEARS[1]) else "custom",
            "percentile":         percentile,
            "units":              units,
            "variable":           var_name,
            "calculation_notes":  notes,
        }
 
    base_dict = {}
 
    # -----------------------------------------------------------------------
    # tas — calendar-day percentile (for heatwave detection)
    # -----------------------------------------------------------------------
    if tas is not None:
        print("Calculating tas base period percentiles...")
        tas_base, actual_start, actual_end = _slice_to_base(
            _check_and_convert_units(da=tas, input_var="tas", conv_type="C"), "tas"
        )
        # store full base time series for annual mean calculation in heatwave_days
        base_dict["tas"] = tas_base
        for p in tas_calday_percentiles:
            tXp = tas_base.groupby("time.dayofyear").quantile(p / 100.0, dim="time", skipna=True)
            key = f"t{int(p)}p_calday" if len(tas_calday_percentiles) == 1 else f"t{int(p)}p_calday"
            tXp.name = key
            tXp.attrs = _base_attrs(
                percentile=p, units="°C", var_name="tas",
                actual_start=actual_start, actual_end=actual_end,
                notes=f"Calendar-day {p}th percentile of tas (°C). One value per dayofyear per grid cell.",
            )
            base_dict[key] = tXp
 
    # -----------------------------------------------------------------------
    # tasmax — stored for heatwave_days
    # -----------------------------------------------------------------------
    if tasmax is not None:
        print("Converting tasmax to °C for base period storage...")
        tasmax_base, actual_start, actual_end = _slice_to_base(
            _check_and_convert_units(da=tasmax, input_var="tasmax", conv_type="C"), "tasmax"
        )
        base_dict["tasmax"] = tasmax_base
 
    # -----------------------------------------------------------------------
    # tasmin — all-year cold-tail percentiles
    # -----------------------------------------------------------------------
    if tasmin is not None:
        print("Calculating tasmin base period percentiles...")
        tasmin_base, actual_start, actual_end = _slice_to_base(
            _check_and_convert_units(da=tasmin, input_var="tasmin", conv_type="C"), "tasmin"
        )
        for p in tasmin_percentiles:
            tnp = tasmin_base.quantile(p / 100.0, dim="time", skipna=True)
            # cold-tail threshold: values > 0°C are not meaningful
            key = f"tasmin_{str(p).replace('.', 'pt')}p"
            tnp.name = key
            tnp.attrs = _base_attrs(
                percentile=p, units="°C", var_name="tasmin",
                actual_start=actual_start, actual_end=actual_end,
                notes=f"All-year {p}th percentile of tasmin (°C). One value per grid cell.",
            )
            base_dict[key] = tnp
 
    # -----------------------------------------------------------------------
    # pr — all-year wet-day percentiles
    # -----------------------------------------------------------------------
    if pr is not None:
        print("Calculating pr base period percentiles...")
        pr_base, actual_start, actual_end = _slice_to_base(
            _check_and_convert_units(da=pr, input_var="pr", conv_type="mm day-1"), "pr"
        )
        pr_wet = pr_base.where(pr_base >= wet_day_threshold)
        for p in pr_percentiles:
            prp = pr_wet.quantile(p / 100.0, dim="time", skipna=True)
            key = f"pr_{str(p).replace('.', 'pt')}p"
            prp.name = key
            prp.attrs = _base_attrs(
                percentile=p, units="mm day-1", var_name="pr",
                actual_start=actual_start, actual_end=actual_end,
                notes=f"All-year {p}th percentile of pr on wet days (>= {wet_day_threshold} mm day-1). One value per grid cell.",
            )
            base_dict[key] = prp

        # -------------------------------------------------------------------
        # rx5day — percentiles of annual maximum 5-day precipitation
        # -------------------------------------------------------------------
        print("Calculating rx5day base period percentiles...")
        annual_max_rx5 = (
            pr_base
            .rolling(time=5, min_periods=5)
            .sum()
            .groupby("time.year")
            .max(dim="time", skipna=True)
        )
        for p in rx5day_percentiles:
            rx5p = annual_max_rx5.chunk(dict(year=-1)).quantile(p / 100.0, dim="year", skipna=True)
            key = f"rx5day_{str(p).replace('.', 'pt')}p"
            rx5p.name = key
            rx5p.attrs = _base_attrs(
                percentile=p, units="mm", var_name="rx5day",
                actual_start=actual_start, actual_end=actual_end,
                notes=(
                    f"All-year {p}th percentile of annual maximum 5-day accumulated "
                    f"precipitation (mm). Rolling sum uses min_periods=5 (no partial windows). "
                    f"One value per grid cell."
                ),
            )
            base_dict[key] = rx5p
 
    print(f"Base period percentiles complete. Keys in base_dict: {list(base_dict.keys())}")
 
    return base_dict

def calculate_base_period_percentiles2(
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
    base_start, base_end = int(base_years[0]), int(base_years[1])

    if not any(da is not None for da in [tas, tasmax, tasmin, pr]):
        raise ValueError("At least one of tas, tasmax, tasmin, or pr must be provided.")

    def _slice_to_base(da, var_name):
        years_in_data = da.time.dt.year
        data_start, data_end = int(years_in_data.min()), int(years_in_data.max())
        if base_start == _DEFAULT_BASE_YEARS[0] and base_end == _DEFAULT_BASE_YEARS[1]:
            if data_start > base_start or data_end < base_end:
                print(
                    f"WARNING ({var_name}): data cover {data_start}–{data_end}, "
                    f"which does not fully span the default base period "
                    f"{base_start}–{base_end}. Proceeding with available years."
                )
        da_base = da.sel(time=da.time.dt.year.isin(range(base_start, base_end + 1)))
        if da_base.time.size == 0:
            raise ValueError(
                f"{var_name}: no data found within base period {base_start}–{base_end}. "
                f"Data cover {data_start}–{data_end}."
            )
        return da_base, int(da_base.time.dt.year.min()), int(da_base.time.dt.year.max())

    def _base_attrs(percentile, units, var_name, actual_start, actual_end, notes):
        return {
            "software_version":   SOFTWARE_VERSION,
            "base_period_start":  actual_start,
            "base_period_end":    actual_end,
            "base_period_source": "default" if (
                base_start == _DEFAULT_BASE_YEARS[0]
                and base_end == _DEFAULT_BASE_YEARS[1]
            ) else "custom",
            "percentile":         percentile,
            "units":              units,
            "variable":           var_name,
            "calculation_notes":  notes,
        }

    def _unpack_quantiles(da_q, percentiles, key_fmt, units, var_name,
                          actual_start, actual_end, notes_fmt):
        """
        Split a multi-percentile quantile result into individual DataArrays.
        da_q has a 'quantile' dimension when multiple percentiles are requested.
        """
        results = {}
        single = da_q.ndim == 0 or "quantile" not in da_q.dims
        for p in percentiles:
            if len(percentiles) == 1 or single:
                da_p = da_q.drop_vars("quantile", errors="ignore")
            else:
                da_p = da_q.sel(quantile=p / 100.0).drop_vars("quantile")
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

    # -----------------------------------------------------------------------
    # tas — calendar-day percentiles (batched)
    # -----------------------------------------------------------------------
    if tas is not None:
        print("Calculating tas base period percentiles...")
        tas_base, actual_start, actual_end = _slice_to_base(
            _check_and_convert_units(da=tas, input_var="tas", conv_type="C"), "tas"
        )
        # Rechunk so the full time axis is contiguous — quantile needs it
        tas_base = tas_base.chunk({"time": -1})
        base_dict["tas"] = tas_base

        # Single grouped quantile call for all percentiles at once
        q_vals = [p / 100.0 for p in tas_calday_percentiles]
        tXp_all = (
            tas_base
            .groupby("time.dayofyear")
            .quantile(q_vals, dim="time", skipna=True)
        )
        base_dict.update(_unpack_quantiles(
            tXp_all, tas_calday_percentiles,
            key_fmt=lambda p: f"t{int(p)}p_calday",
            units="°C", var_name="tas",
            actual_start=actual_start, actual_end=actual_end,
            notes_fmt=lambda p: (
                f"Calendar-day {p}th percentile of tas (°C). "
                "One value per dayofyear per grid cell."
            ),
        ))

    # -----------------------------------------------------------------------
    # tasmax — stored for heatwave_days (no change needed, just a slice)
    # -----------------------------------------------------------------------
    if tasmax is not None:
        print("Converting tasmax to °C for base period storage...")
        tasmax_base, *_ = _slice_to_base(
            _check_and_convert_units(da=tasmax, input_var="tasmax", conv_type="C"), "tasmax"
        )
        base_dict["tasmax"] = tasmax_base

    # -----------------------------------------------------------------------
    # tasmin — all-year cold-tail percentiles (batched)
    # -----------------------------------------------------------------------
    if tasmin is not None:
        print("Calculating tasmin base period percentiles...")
        tasmin_base, actual_start, actual_end = _slice_to_base(
            _check_and_convert_units(da=tasmin, input_var="tasmin", conv_type="C"), "tasmin"
        )
        tasmin_base = tasmin_base.chunk({"time": -1})

        q_vals = [p / 100.0 for p in tasmin_percentiles]
        tnp_all = tasmin_base.quantile(q_vals, dim="time", skipna=True)
        base_dict.update(_unpack_quantiles(
            tnp_all, tasmin_percentiles,
            key_fmt=lambda p: f"tasmin_{str(p).replace('.', 'pt')}p",
            units="°C", var_name="tasmin",
            actual_start=actual_start, actual_end=actual_end,
            notes_fmt=lambda p: (
                f"All-year {p}th percentile of tasmin (°C). One value per grid cell."
            ),
        ))

    # -----------------------------------------------------------------------
    # pr — wet-day percentiles + rx5day percentiles (batched)
    # -----------------------------------------------------------------------
    if pr is not None:
        print("Calculating pr base period percentiles...")
        pr_base, actual_start, actual_end = _slice_to_base(
            _check_and_convert_units(da=pr, input_var="pr", conv_type="mm day-1"), "pr"
        )
        # Rechunk once for all downstream operations
        pr_base = pr_base.chunk({"time": -1})

        # Wet-day mask computed once
        pr_wet = pr_base.where(pr_base >= wet_day_threshold)

        q_vals = [p / 100.0 for p in pr_percentiles]
        prp_all = pr_wet.quantile(q_vals, dim="time", skipna=True)
        base_dict.update(_unpack_quantiles(
            prp_all, pr_percentiles,
            key_fmt=lambda p: f"pr_{str(p).replace('.', 'pt')}p",
            units="mm day-1", var_name="pr",
            actual_start=actual_start, actual_end=actual_end,
            notes_fmt=lambda p: (
                f"All-year {p}th percentile of pr on wet days "
                f"(>= {wet_day_threshold} mm day-1). One value per grid cell."
            ),
        ))

        # rx5day — already rechunked pr_base, no mid-graph rechunk needed
        print("Calculating rx5day base period percentiles...")
        annual_max_rx5 = (
            pr_base
            .rolling(time=5, min_periods=5)
            .sum()
            .groupby("time.year")
            .max(dim="time", skipna=True)
        )
        # year dimension is small (~35 values); chunk fully for quantile
        annual_max_rx5 = annual_max_rx5.chunk({"year": -1})

        q_vals = [p / 100.0 for p in rx5day_percentiles]
        rx5p_all = annual_max_rx5.quantile(q_vals, dim="year", skipna=True)
        base_dict.update(_unpack_quantiles(
            rx5p_all, rx5day_percentiles,
            key_fmt=lambda p: f"rx5day_{str(p).replace('.', 'pt')}p",
            units="mm", var_name="rx5day",
            actual_start=actual_start, actual_end=actual_end,
            notes_fmt=lambda p: (
                f"All-year {p}th percentile of annual maximum 5-day accumulated "
                f"precipitation (mm). Rolling sum uses min_periods=5 "
                f"(no partial windows). One value per grid cell."
            ),
        ))

    print(f"Base period percentiles complete. Keys in base_dict: {list(base_dict.keys())}")
    return base_dict
 

# =================
# HEAT STRESS INDICATORS
# =================

def AT(ds_dict, hazard_thresholds = hazard_thresholds["AT"]):
    """
    apparent temperature ('feels like' temperature)
    combines air, humidity, and wind speed
    """
        
    TX = ds_dict['tasmax'] # needs to be in celcius
    TX = _check_and_convert_units(da=TX, input_var="tasmax", conv_type="C")
    RH = ds_dict['hurs'] # relative humidity surface needs to be fraction 0-1
    RH = _check_and_convert_units(da=RH, input_var="hurs", conv_type="fraction")

    # ensure values are between 0 and 1
    RH = (RH.clip(0.001, 0.999999)).values

    # saturation vapor pressure via Tetens' equation
    es_positive = 0.611 * np.exp(17.27 * TX / (TX + 237.3))
    es_negative = 0.611 * np.exp(21.87 * TX / (TX + 265.5))
    es = xr.where(TX > 0, es_positive, es_negative) # apply Tetens' equation 
    VP = es*(RH/100) # make sure this / 100 should actually be there  
    
    # Simplified apparent temperature formula
    AT = (0.92*TX) + (0.22*VP) - 1.3 # from Zhao et al., 2015

    # Make a check that the hazard thresholds are a dictionary, if not or if fails, print error and say that the hazard_thresholds dictionary needs to be a specific form (print form)
    AT_levels = _annual_exceedance_frac(AT, hazard_thresholds, var_name="AT")
    AT_levels = _assign_hazard_level(AT_levels)
    return AT_levels


def HI(ds_dict, hazard_thresholds = hazard_thresholds["HI"]):
    """
    NOAA heat index https://www.wpc.ncep.noaa.gov/html/heatindex.shtml
    see text S5 for equation https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1029%2F2020EF001885&file=2020EF001885-sup-0001-Supporting+Information+SI-S01.pdf
    """
    TX = ds_dict['tasmax'] # needs to be in F
    TX = _check_and_convert_units(da=TX, input_var="tasmax", conv_type="F")
    RH = ds_dict['hurs'] # relative humidity surface needs to %
    RH = _check_and_convert_units(da=RH, input_var="hurs", conv_type="%")

    # ensure in reasonable range 0-100
    RH = (RH.clip(0.1, 99.9999)).values # clip values between 0 and 100%
    
    # constants
    c0 = -42.379 # F
    c1 = 2.04901523
    c2 = 10.14333127 # F
    c3 = -0.22475541
    c4 = -0.00683783 # F^-1
    c5 = -0.05481717 # F
    c6 = 0.00122874 # F^-1
    c7 = 0.00085282
    c8 = -0.00000199 # F^-1

    # set up equations
    HI1 = c0 + c1*TX + c2*RH + c3*TX*RH + c4*TX**2 + c5*RH**2 + c6*TX**2*RH + c7*TX*RH**2 + c8*TX**2*RH**2
    HIA1 = ((13 - RH)/4) * np.sqrt(17 - np.abs(TX-95)/17)
    HIA2 = ((RH - 85)/10) * (87 - TX)/5

    # calculate HI based on proper equation for RH and T thresholds
    #T_ORIG = TX.copy() # keep original values for conditional masking below

    HI_A = HI1 - HIA1 # if RH < 13% and 80 F < T < 112 F
    mask_T_A = (TX > 80) & (TX < 112) & (RH < 13)

    HI_B = HI1 + HIA2 # if RH > 85% and 80 F < T < 87 F
    mask_T_B =  ~mask_T_A & (TX > 80) & (TX < 87) & (RH > 85)

    HI_C = HI1 # all else where T > 80 F
    mask_T_C =  ~mask_T_A & ~mask_T_B & (TX > 80)

    HI_D = 0.5 * (TX + 61 + 1.2*(TX - 68) + 0.094*RH) # if T < 80 F
    mask_T_D = (TX < 80)

    # iteratively mask and apply HI equation based on threshold specifications
    HI = xr.where(mask_T_A, HI_A,
      xr.where(mask_T_B, HI_B,
      xr.where(mask_T_C, HI_C,
      xr.where(mask_T_D, HI_D, TX))))  # Fallback to TX 
    
    HI = (HI-32)*(5/9) # convert F to C

    # Make a check that the hazard thresholds are a dictionary, if not or if fails, print error and say that the hazard_thresholds dictionary needs to be a specific form (print form)
    HI_levels = _annual_exceedance_frac(HI, hazard_thresholds, var_name="HI")
    HI_levels = _assign_hazard_level(HI_levels)
    return HI_levels


def Hu(ds_dict, hazard_thresholds = hazard_thresholds["Hu"]):
    """
    Calculate Humidex (Canadian humidity index https://publications.gc.ca/site/eng/9.865813/publication.html)
    """
    
    TX = ds_dict['tasmax'] # needs to be in C
    TX = _check_and_convert_units(da=TX, input_var="tasmax", conv_type="C")
    RH = ds_dict['hurs'] # relative humidity surface needs to be fraction 0-1
    RH = _check_and_convert_units(da=RH, input_var="hurs", conv_type="fraction")
    RH = (RH.clip(0.001, 0.999999)).values # clip values between 0 and 1 

    # saturation vapor pressure via Tetens' equation
    es_positive = 0.611 * np.exp(17.27 * TX / (TX + 237.3))
    es_negative = 0.611 * np.exp(21.87 * TX / (TX + 265.5))
    es = xr.where(TX > 0, es_positive, es_negative) # apply Tetens' equation 
    # vapor pressure = RH * saturation vp (es)
    e = RH*es

    # humidex = T + h
    h = (5/9)*(e-10)
    HU = TX + h 
    
    HU_levels = _annual_exceedance_frac(HU, hazard_thresholds, var_name="Hu")
    HU_levels = _assign_hazard_level(HU_levels)
    return HU_levels

# Find out a way to just have the NEWT functions needed in here instead of pulling from the git clone
def _wbt_values(ds_dict):
    '''
    WBT calculation via Noniterative Evaluation of Wet bulb Temperature (NEWT); Rogers & Warren 2024
    Faster and more accurate than Davies-Jones 2008 & Stull 2011. Also former adaptations of Davies-Jones had bugs. 
    https://essopenarchive.org/users/714325/articles/698601-fast-and-accurate-calculation-of-wet-bulb-temperature-for-humid-heat-extremes?commit=a888ca0a0d28f09b4b49826f987739fa5180ebec
    https://github.com/AusClimateService/atmos
    https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4866
    '''

    # read in variables
    p = ds_dict["ps"] 
    p = _check_and_convert_units(da=p, input_var="ps", conv_type="Pa") # NEWT wants surface pressure in Pa
    TX = ds_dict["tasmax"]
    TX = _check_and_convert_units(da=TX, input_var="tasmax", conv_type="K") # NEWT wants T in Kelvin

    # Use specific humidity directly unless not avail then esimate via RH
    if "huss" in ds_dict.keys():
        q = ds_dict["huss"]
    else:
        RH = ds_dict['hurs'] # relative humidity surface needs to %
        RH = _check_and_convert_units(da=RH, input_var="hurs", conv_type="%")
        RH = RH.clip(0.1, 99.9999)
        q = newt.specific_humidity_from_relative_humidity(p, TX, RH/100.0)

    # Calculate WBT
    Twp = newt.pseudo_wet_bulb_temperature(p, TX, q)

    # convert from Kelvin to Celcius 
    Twp = Twp - 273.15

    return Twp

def WBT(ds_dict, hazard_thresholds=hazard_thresholds["WBT"]):
    '''
    get exceedances
    '''

    Twp = _wbt_values(ds_dict=ds_dict)

    WBT_levels = _annual_exceedance_frac(Twp, hazard_thresholds, var_name="WBT")
    WBT_levels = _assign_hazard_level(WBT_levels)

    return WBT_levels


## !!!! try the Liljegren WBGT first, but if not all variables avail, use Schwingshackl approximation !!!!
## !!!! use kong and huber python code https://doi.org/10.1029/2021EF002334 and https://zenodo.org/records/5980536
def _scale_windspeed(va, h):
    """
    Scaling wind speed from 10 metres (most CMIP6 models) to height h
        :param va: (float array) 10m wind speed [m/s]
        :param h: (float array) height at which wind speed needs to be scaled [m]
        returns wind speed at height h
    Reference: Bröde et al. (2012)
    https://doi.org/10.1007/s00484-011-0454-1
    from thermofeel (https://github.com/ecmwf/thermofeel)
    """
    c = 1 / np.log10(10 / 0.01)  #
    c = 0.333333333333
    vh = va * np.log10(h / 0.01) * c

    return vh

def _calculate_bgt(ds_dict, mrt):
    """
    using calculate_bgt() from thermofeel (https://github.com/ecmwf/thermofeel)
    Globe temperature
        :param t2_k: (float array) 2m temperature [K]
        :param mrt: (float array) mean radiant temperature [K]
        :param va: (float array) wind speed at 10 meters [m/s]
        returns globe temperature [K]
    Reference: Guo et al. 2018
    https://doi.org/10.1016/j.enbuild.2018.08.029
    """

    t2_k = _check_and_convert_units(da=ds_dict['tasmax'], input_var="tasmax", conv_type="K")
    va = _check_and_convert_units(da=ds_dict['sfcWind'], input_var="sfcWind", conv_type="m s-1")

    v = _scale_windspeed(
        va, 1.1
    )  # formula requires wind speed at 1.1m (i.e., at the level of the globe)

    # a = 1
    d = (1.1e8 * v**0.6) / (0.95 * 0.15**0.4)
    e = -(mrt**4) - d * t2_k

    q = 12 * e
    s = 27 * (d**2)
    delta = ((s + np.sqrt(s**2 - 4 * (q**3))) / 2) ** (1 / 3)
    Q = 0.5 * np.sqrt((1 / 3) * (delta + q / delta))

    bgt = -Q + 0.5 * np.sqrt(-4 * (Q**2) + d / Q)

    return bgt

def WBGT(ds_dict, hazard_thresholds = hazard_thresholds["WBGT"], hum_var="both"):
    """
    Default: Brimicombe WBGT approximation
        -Wet Bulb Globe Temperature using Brimicombe et al. 2023 (https://doi.org/10.1029/2022GH000701).
        -Modified by using NEWT instead of Stull 2011 WBT calculation
    Fallback: Schwingshackl et al., 2021 approximation
        -WBGT in the shade is defined as weighted mean of Twb and tas in Schwingshackl et al., 2021
        -https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020EF001885
    """

    # Wet bulb temp via NEWT
    Twb = _wbt_values(ds_dict) # in °C
    TX =_check_and_convert_units(da=ds_dict['tasmax'], input_var="tasmax", conv_type="C")
    TA =_check_and_convert_units(da=ds_dict['tas'], input_var="tas", conv_type="C")

    if all(k in ds_dict for k in ['rsdsdiff', 'rsus', 'rlus', 'rsdscs', 'rsdscsdiff']): 
        tmrt = _calculate_mean_radiant_temperature(ds_dict)
        bgt = _calculate_bgt(ds_dict = ds_dict, mrt = tmrt) - 273.15 # K to °C
        WBGT = 0.7*Twb + 0.2*bgt + 0.1*TA # if calculating tmrt, use daily average temp
    else:
        WBGT = 0.7 * Twb + 0.3 * TX # basically assumes that MRT == air temp (reference condition)

    WBGT_levels = _annual_exceedance_frac(WBGT, hazard_thresholds, var_name="WBGT")
    WBGT_levels = _assign_hazard_level(WBGT_levels)
        
    return WBGT_levels

def _cos_solar_zenith_angle_daily(time, lat):
    """
    Daytime-mean cosine of solar zenith angle, integrated over daylight hours.
    For use with daily-mean radiation data where no time of day is known.
    
    Based on Di Napoli et al. 2020 eq. 12 (https://doi.org/10.1007/s00484-020-01900-5)
    and earthkit-meteo (https://github.com/ecmwf/earthkit-meteo)
    Longitude is not needed for the daily mean (no hour angle offset matters
    when integrating over the full daylight window).
    """
     # --- Julian day and fractional year ---
    times = pd.DatetimeIndex(time.values)
    JD = xr.DataArray(times.dayofyear.values, dims=['time'], coords={'time': time})

    g_rad = np.radians((360.0 / 365.25) * JD)

    # --- Solar declination (Spencer 1971) in radians ---
    delta = (
        0.006918
        - 0.399912 * np.cos(g_rad)
        + 0.070257 * np.sin(g_rad)
        - 0.006758 * np.cos(2 * g_rad)
        + 0.000907 * np.sin(2 * g_rad)
        - 0.002697 * np.cos(3 * g_rad)
        + 0.001480 * np.sin(3 * g_rad)
    )  # radians, shape (time,)

    # --- Latitude in radians, shape (lat,) ---
    phi = np.radians(lat)  # xr.DataArray with dim 'lat'

    # --- Sunrise/sunset hour angle h0 (Di Napoli eq. 11) ---
    # cos(h0) = -tan(delta) * tan(phi)
    # Broadcasting: delta is (time,), phi is (lat,) -> result is (time, lat)
    cos_h0 = (-np.tan(delta) * np.tan(phi)).clip(-1.0, 1.0)
    h0 = np.arccos(cos_h0)  # radians, (time, lat)

    # --- Daytime-mean cossza (Di Napoli eq. 12, symmetric simplification) ---
    # cos θ₀ = sin δ sin φ + sin(h0)/h0 * cos δ cos φ
    safe_h0 = h0.where(h0 > 1e-6, other=1e-6)

    cossza = (
        np.sin(delta) * np.sin(phi)
        + np.cos(delta) * np.cos(phi) * np.sin(h0) / safe_h0
    )

    # Zero out polar night
    cossza = cossza.where(h0 > 1e-6, other=0.0)

    return cossza.clip(0.0, 1.0)



def _calculate_mean_radiant_temperature(ds_dict):
    """
    Code adapted from ECMWF thermofeel (https://github.com/ecmwf/thermofeel)

    MRT - Mean Radiant Temperature
    cossza computed via func from earhkit-meteo.solar.calculate_cos_solar_zenith_angle
        :param ssrd: (float array) surface solar radiation downwards [W m-2]
        :param ssr: (float array) surface net solar radiation [W m-2]
        :param dsrp: (float array) direct solar radiation [W m-2]
        :param strd: (float array) surface thermal radiation downwards [W m-2]
        :param fdir: (float array) total sky direct solar radiation at surface [W m-2]
        :param strr: (float array) surface net thermal radiation [W m-2]
        :param cossza: (float array) cosine of solar zenith angle [dimentionless]
        returns mean radiant temperature [K]
    Reference: Di Napoli et al. (2020)
    https://link.springer.com/article/10.1007/s00484-020-01900-5
    """
    # variables needed: ['rsdsdiff', 'rsus', 'rlus', 'rsdscs', 'rsdscsdiff', 'time', 'lat', 'lon']

    # ssrd  = ds_dict['rsds']
    # ssr   = ds_dict['rsds'] - ds_dict['rsus']     # net SW = down - up 
    dsrp  = ds_dict['rsdscs'] - ds_dict['rsdscsdiff']  # surface downwelling SW - surf diffuse downwelling SW (clear sky)  
    # strd  = ds_dict['rlds']
    # fdir  = ds_dict['rsds'] - ds_dict['rsdsdiff'] # total sky conditions surface downwelling SW - surf diffuse downwelling SW
    # strr  = ds_dict['rlds'] - ds_dict['rlus']     # net LW thermal = down - up
    cossza = _cos_solar_zenith_angle_daily(ds_dict['rsdscs'].time, ds_dict['rsdscs'].lat)  # compute from geometry

    # original code from thermofeel
    # dsw = ssrd - fdir
    # rsw = ssrd - ssr
    # lur = strd - strr
    # CMIP6 equivalent 

    # Istar = dsrp
    dsw = ds_dict['rsdsdiff'] # total sky surface diffuse downwelling SR
    rsw = ds_dict['rsus'] # surface upwelling SR
    lur = ds_dict['rlus'] # surface upwelling LR

    # calculate fp projected factor area
    gamma = np.arcsin(cossza) * 180 / np.pi
    fp = 0.308 * np.cos(to_radians * gamma * (0.998 - gamma * gamma / 50000))

    # calculate mean radiant temperature
    mrt = np.power(
        (
            (1 / 0.0000000567)
            * (
                0.5 * strd
                + 0.5 * lur
                + (0.7 / 0.97) * (0.5 * dsw + 0.5 * rsw + fp * dsrp)
            )
        ),
        0.25,
    )

    return mrt

# UTCI
def _UTCI(ds_dict, hum_var='both', tmrt=None, hotorcold=None, hazard_thresholds=hazard_thresholds):
    """
    Calculate Universal Thermal Climate Index (UTCI) using Bröde et al. 2012 method.
    Code adapted from https://gist.github.com/leozqi/ae8943c93b899cbbdc059ce9bc11390f
    
    ds_dict : xr.Dataset
        Dataset with CMIP6 variables:
        - 'tasmax': max daily air temperature (K)
        - 'sfcWind' or 'uas'/'vas': wind speed (m/s) at 10m
        - 'huss' (kg/kg) and/or 'hurs' (%) for humidity
        - 'rsds' (optional): downward shortwave radiation (W/m²) for Tmrt estimation
    hum_var : str
        'both' (use both huss/hurs), 'huss', or 'hurs'
    tmrt : xr.DataArray, optional
        Mean radiant temperature (°C). If required variables not present, set to mean surface air temperature (Shwingshackl et al. 2021; https://doi.org/10.1029/2020EF001885). This is the reference environment definition in Broede

    Reference:
    Bröde et al. 2012
    UTCI valid for:
    - Air temperature: -50 to +50 °C
    - Wind speed: 0.5 to 17 m/s (at 10m)
    - Mean radiant temp: Ta-30 to Ta+70 °C
    """
    # set UTCIhot or UTCIcold
    if hotorcold.lower() == "hot":
        varname = "UTCIhot"
        hazard_thresholds = hazard_thresholds[varname]
        exceedance_dir = "above"
        tas_var = "tasmax"
    elif hotorcold.lower() == "cold":
        varname = "UTCIcold"
        hazard_thresholds = hazard_thresholds[varname]
        exceedance_dir = "below"
        tas_var = "tasmin"
    else:
        print("Argument 'hotorcold' not passed. pass hotorcold='hot' for heat stress thresholds or hotorcold='cold' for cold thresholds")
    
    # air temp in C
    TXN = ds_dict[tas_var] 
    TXN = _check_and_convert_units(da=TXN, input_var=tas_var, conv_type="C")
    TA = ds_dict['tas'] 
    TA = _check_and_convert_units(da=TA, input_var="tas", conv_type="C")

    RH = ds_dict['hurs'] # relative humidity surface needs to %
    RH = _check_and_convert_units(da=RH, input_var="hurs", conv_type="fraction")
    RH = RH.clip(0.001, 0.999999)

    ps = ds_dict['ps']
    ps = _check_and_convert_units(da=ps, input_var="ps", conv_type="Pa")
    
    # Get wind speed (ensure 10m height, clip to valid range)
    if 'sfcWind' in ds_dict:
        va = _check_and_convert_units(da=ds_dict['sfcWind'], input_var="sfcWind", conv_type="m s-1")
    else:
        va = np.sqrt(ds_dict['uas']**2 + ds_dict['vas']**2)
    va = va.clip(0.5, 17.0)
    
    # Get vapor pressure in kPa
    if hum_var in ('both', 'hurs'):
        if all(k in ds_dict for k in ['rsdsdiff', 'rsus', 'rlus', 'rsdscs', 'rsdscsdiff']):
            # Use hurs directly if available, more stable
            es_hpa = _sat_vapor_pressure(TA) * RH 
        else:
            # Use hurs directly if available, more stable
            es_hpa = _sat_vapor_pressure(TXN) * RH 
    else:  # huss
        # Convert specific humidity to relative humidity
        p_hpa = ps * 0.01  # Pa to hPa
        # Mixing ratio from specific humidity
        r = ds_dict['huss'] / (1 - ds_dict['huss'])  # kg/kg
        es_hpa = r * p_hpa / (0.622 + r)  # actual vapor pressure (hPa)
    
    es_hpa = es_hpa.clip(0.0, 20.0) # Broede hard cap
    Pa = es_hpa / 10.0 # hPa to kPa
    
    # Mean radiant temperature - need to estimate since it is not a cmip6 input
    if all(k in ds_dict for k in ['rsdsdiff', 'rsus', 'rlus', 'rsdscs', 'rsdscsdiff']): 
        # must use daily mean temperature to have an accurate D_Tmrt because MRT is calculated via daily mean radiation values
        # if TX is used, tmrt - TX will likely be negative, meaning UTCI could be underestimated on sunny days 
        tmrt = _calculate_mean_radiant_temperature(ds_dict) - 273.15 # K to C 
        D_Tmrt = (tmrt - TA).clip(-30, 70) 
        # 6th order polynomial approximation
        UTCI = _utci_polynomial(TA, va, D_Tmrt, Pa) 
        # Mask invalid ranges
        valid = (TA >= -50) & (TA <= 50)   
    else:
        # If solar radiation available, use it; otherwise assume Tmrt = TXN
        # use daily max (or min for UTCIcold) temperature (TXN) and set MRT to TXN
        tmrt = TXN
        D_Tmrt = 0 # should be 0 because tmrt is set to TXN
        # 6th order polynomial approximation
        UTCI = _utci_polynomial(TXN, va, D_Tmrt, Pa) 
        # Mask invalid ranges
        valid = (TXN >= -50) & (TXN <= 50) 
 
    UTCI_levels = _annual_exceedance_frac(UTCI.where(valid), hazard_thresholds=hazard_thresholds, var_name=varname, exceedance_dir=exceedance_dir)
    UTCI_levels = _assign_hazard_level(UTCI_levels)

    return UTCI_levels


def _sat_vapor_pressure(ta_celsius):
    """
    Calculate saturation vapor pressure over water (hPa) using ITS-90 formulation.
    Hardy 1998 formula. for UTCI. translated from Broede fortran code 2009 (https://gist.github.com/leozqi/ae8943c93b899cbbdc059ce9bc11390f)
    """
    tk = ta_celsius + 273.15
    g = np.array([-2.8365744e3, -6.028076559e3, 1.954263612e1, -2.737830188e-2,
                  1.6261698e-5, 7.0229056e-10, -1.8680009e-13, 2.7150305])
    
    es = g[7] * np.log(tk)
    for i in range(7):
        es = es + g[i] * tk**(i-2)
    
    return np.exp(es) * 0.01  # Pa to hPa


def _utci_polynomial(Ta, va, D_Tmrt, Pa):
    """
    6th order polynomial approximation for UTCI. translated from Broede fortran code 2009 (https://gist.github.com/leozqi/ae8943c93b899cbbdc059ce9bc11390f)
    """
    
    return (Ta +
        6.07562052e-01 +
        -2.27712343e-02 * Ta +
        8.06470249e-04 * Ta**2 +
        -1.54271372e-04 * Ta**3 +
        -3.24651735e-06 * Ta**4 +
        7.32602852e-08 * Ta**5 +
        1.35959073e-09 * Ta**6 +
        -2.25836520 * va +
        8.80326035e-02 * Ta*va +
        2.16844454e-03 * Ta**2*va +
        -1.53347087e-05 * Ta**3*va +
        -5.72983704e-07 * Ta**4*va +
        -2.55090145e-09 * Ta**5*va +
        -7.51269505e-01 * va**2 +
        -4.08350271e-03 * Ta*va**2 +
        -5.21670675e-05 * Ta**2*va**2 +
        1.94544667e-06 * Ta**3*va**2 +
        1.14099531e-08 * Ta**4*va**2 +
        1.58137256e-01 * va**3 +
        -6.57263143e-05 * Ta*va**3 +
        2.22697524e-07 * Ta**2*va**3 +
        -4.16117031e-08 * Ta**3*va**3 +
        -1.27762753e-02 * va**4 +
        9.66891875e-06 * Ta*va**4 +
        2.52785852e-09 * Ta**2*va**4 +
        4.56306672e-04 * va**5 +
        -1.74202546e-07 * Ta*va**5 +
        -5.91491269e-06 * va**6 +
        3.98374029e-01 * D_Tmrt +
        1.83945314e-04 * Ta*D_Tmrt +
        -1.73754510e-04 * Ta**2*D_Tmrt +
        -7.60781159e-07 * Ta**3*D_Tmrt +
        3.77830287e-08 * Ta**4*D_Tmrt +
        5.43079673e-10 * Ta**5*D_Tmrt +
        -2.00518269e-02 * va*D_Tmrt +
        8.92859837e-04 * Ta*va*D_Tmrt +
        3.45433048e-06 * Ta**2*va*D_Tmrt +
        -3.77925774e-07 * Ta**3*va*D_Tmrt +
        -1.69699377e-09 * Ta**4*va*D_Tmrt +
        1.69992415e-04 * va**2*D_Tmrt +
        -4.99204314e-05 * Ta*va**2*D_Tmrt +
        2.47417178e-07 * Ta**2*va**2*D_Tmrt +
        1.07596466e-08 * Ta**3*va**2*D_Tmrt +
        8.49242932e-05 * va**3*D_Tmrt +
        1.35191328e-06 * Ta*va**3*D_Tmrt +
        -6.21531254e-09 * Ta**2*va**3*D_Tmrt +
        -4.99410301e-06 * va**4*D_Tmrt +
        -1.89489258e-08 * Ta*va**4*D_Tmrt +
        8.15300114e-08 * va**5*D_Tmrt +
        7.55043090e-04 * D_Tmrt**2 +
        -5.65095215e-05 * Ta*D_Tmrt**2 +
        -4.52166564e-07 * Ta**2*D_Tmrt**2 +
        2.46688878e-08 * Ta**3*D_Tmrt**2 +
        2.42674348e-10 * Ta**4*D_Tmrt**2 +
        1.54547250e-04 * va*D_Tmrt**2 +
        5.24110970e-06 * Ta*va*D_Tmrt**2 +
        -8.75874982e-08 * Ta**2*va*D_Tmrt**2 +
        -1.50743064e-09 * Ta**3*va*D_Tmrt**2 +
        -1.56236307e-05 * va**2*D_Tmrt**2 +
        -1.33895614e-07 * Ta*va**2*D_Tmrt**2 +
        2.49709824e-09 * Ta**2*va**2*D_Tmrt**2 +
        6.51711721e-07 * va**3*D_Tmrt**2 +
        1.94960053e-09 * Ta*va**3*D_Tmrt**2 +
        -1.00361113e-08 * va**4*D_Tmrt**2 +
        -1.21206673e-05 * D_Tmrt**3 +
        -2.18203660e-07 * Ta*D_Tmrt**3 +
        7.51269482e-09 * Ta**2*D_Tmrt**3 +
        9.79063848e-11 * Ta**3*D_Tmrt**3 +
        1.25006734e-06 * va*D_Tmrt**3 +
        -1.81584736e-09 * Ta*va*D_Tmrt**3 +
        -3.52197671e-10 * Ta**2*va*D_Tmrt**3 +
        -3.36514630e-08 * va**2*D_Tmrt**3 +
        1.35908359e-10 * Ta*va**2*D_Tmrt**3 +
        4.17032620e-10 * va**3*D_Tmrt**3 +
        -1.30369025e-09 * D_Tmrt**4 +
        4.13908461e-10 * Ta*D_Tmrt**4 +
        9.22652254e-12 * Ta**2*D_Tmrt**4 +
        -5.08220384e-09 * va*D_Tmrt**4 +
        -2.24730961e-11 * Ta*va*D_Tmrt**4 +
        1.17139133e-10 * va**2*D_Tmrt**4 +
        6.62154879e-10 * D_Tmrt**5 +
        4.03863260e-13 * Ta*D_Tmrt**5 +
        1.95087203e-12 * va*D_Tmrt**5 +
        -4.73602469e-12 * D_Tmrt**6 +
        5.12733497 * Pa +
        -3.12788561e-01 * Ta*Pa +
        -1.96701861e-02 * Ta**2*Pa +
        9.99690870e-04 * Ta**3*Pa +
        9.51738512e-06 * Ta**4*Pa +
        -4.66426341e-07 * Ta**5*Pa +
        5.48050612e-01 * va*Pa +
        -3.30552823e-03 * Ta*va*Pa +
        -1.64119440e-03 * Ta**2*va*Pa +
        -5.16670694e-06 * Ta**3*va*Pa +
        9.52692432e-07 * Ta**4*va*Pa +
        -4.29223622e-02 * va**2*Pa +
        5.00845667e-03 * Ta*va**2*Pa +
        1.00601257e-06 * Ta**2*va**2*Pa +
        -1.81748644e-06 * Ta**3*va**2*Pa +
        -1.25813502e-03 * va**3*Pa +
        -1.79330391e-04 * Ta*va**3*Pa +
        2.34994441e-06 * Ta**2*va**3*Pa +
        1.29735808e-04 * va**4*Pa +
        1.29064870e-06 * Ta*va**4*Pa +
        -2.28558686e-06 * va**5*Pa +
        -3.69476348e-02 * D_Tmrt*Pa +
        1.62325322e-03 * Ta*D_Tmrt*Pa +
        -3.14279680e-05 * Ta**2*D_Tmrt*Pa +
        2.59835559e-06 * Ta**3*D_Tmrt*Pa +
        -4.77136523e-08 * Ta**4*D_Tmrt*Pa +
        8.64203390e-03 * va*D_Tmrt*Pa +
        -6.87405181e-04 * Ta*va*D_Tmrt*Pa +
        -9.13863872e-06 * Ta**2*va*D_Tmrt*Pa +
        5.15916806e-07 * Ta**3*va*D_Tmrt*Pa +
        -3.59217476e-05 * va**2*D_Tmrt*Pa +
        3.28696511e-05 * Ta*va**2*D_Tmrt*Pa +
        -7.10542454e-07 * Ta**2*va**2*D_Tmrt*Pa +
        -1.24382300e-05 * va**3*D_Tmrt*Pa +
        -7.38584400e-09 * Ta*va**3*D_Tmrt*Pa +
        2.20609296e-07 * va**4*D_Tmrt*Pa +
        -7.32469180e-04 * D_Tmrt**2*Pa +
        -1.87381964e-05 * Ta*D_Tmrt**2*Pa +
        4.80925239e-06 * Ta**2*D_Tmrt**2*Pa +
        -8.75492040e-08 * Ta**3*D_Tmrt**2*Pa +
        2.77862930e-05 * va*D_Tmrt**2*Pa +
        -5.06004592e-06 * Ta*va*D_Tmrt**2*Pa +
        1.14325367e-07 * Ta**2*va*D_Tmrt**2*Pa +
        2.53016723e-06 * va**2*D_Tmrt**2*Pa +
        -1.72857035e-08 * Ta*va**2*D_Tmrt**2*Pa +
        -3.95079398e-08 * va**3*D_Tmrt**2*Pa +
        -3.59413173e-07 * D_Tmrt**3*Pa +
        7.04388046e-07 * Ta*D_Tmrt**3*Pa +
        -1.89309167e-08 * Ta**2*D_Tmrt**3*Pa +
        -4.79768731e-07 * va*D_Tmrt**3*Pa +
        7.96079978e-09 * Ta*va*D_Tmrt**3*Pa +
        1.62897058e-09 * va**2*D_Tmrt**3*Pa +
        3.94367674e-08 * D_Tmrt**4*Pa +
        -1.18566247e-09 * Ta*D_Tmrt**4*Pa +
        3.34678041e-10 * va*D_Tmrt**4*Pa +
        -1.15606447e-10 * D_Tmrt**5*Pa +
        -2.80626406 * Pa**2 +
        5.48712484e-01 * Ta*Pa**2 +
        -3.99428410e-03 * Ta**2*Pa**2 +
        -9.54009191e-04 * Ta**3*Pa**2 +
        1.93090978e-05 * Ta**4*Pa**2 +
        -3.08806365e-01 * va*Pa**2 +
        1.16952364e-02 * Ta*va*Pa**2 +
        4.95271903e-04 * Ta**2*va*Pa**2 +
        -1.90710882e-05 * Ta**3*va*Pa**2 +
        2.10787756e-03 * va**2*Pa**2 +
        -6.98445738e-04 * Ta*va**2*Pa**2 +
        2.30109073e-05 * Ta**2*va**2*Pa**2 +
        4.17856590e-04 * va**3*Pa**2 +
        -1.27043871e-05 * Ta*va**3*Pa**2 +
        -3.04620472e-06 * va**4*Pa**2 +
        5.14507424e-02 * D_Tmrt*Pa**2 +
        -4.32510997e-03 * Ta*D_Tmrt*Pa**2 +
        8.99281156e-05 * Ta**2*D_Tmrt*Pa**2 +
        -7.14663943e-07 * Ta**3*D_Tmrt*Pa**2 +
        -2.66016305e-04 * va*D_Tmrt*Pa**2 +
        2.63789586e-04 * Ta*va*D_Tmrt*Pa**2 +
        -7.01199003e-06 * Ta**2*va*D_Tmrt*Pa**2 +
        -1.06823306e-04 * va**2*D_Tmrt*Pa**2 +
        3.61341136e-06 * Ta*va**2*D_Tmrt*Pa**2 +
        2.29748967e-07 * va**3*D_Tmrt*Pa**2 +
        3.04788893e-04 * D_Tmrt**2*Pa**2 +
        -6.42070836e-05 * Ta*D_Tmrt**2*Pa**2 +
        1.16257971e-06 * Ta**2*D_Tmrt**2*Pa**2 +
        7.68023384e-06 * va*D_Tmrt**2*Pa**2 +
        -5.47446896e-07 * Ta*va*D_Tmrt**2*Pa**2 +
        -3.59937910e-08 * va**2*D_Tmrt**2*Pa**2 +
        -4.36497725e-06 * D_Tmrt**3*Pa**2 +
        1.68737969e-07 * Ta*D_Tmrt**3*Pa**2 +
        2.67489271e-08 * va*D_Tmrt**3*Pa**2 +
        3.23926897e-09 * D_Tmrt**4*Pa**2 +
        -3.53874123e-02 * Pa**3 +
        -2.21201190e-01 * Ta*Pa**3 +
        1.55126038e-02 * Ta**2*Pa**3 +
        -2.63917279e-04 * Ta**3*Pa**3 +
        4.53433455e-02 * va*Pa**3 +
        -4.32943862e-03 * Ta*va*Pa**3 +
        1.45389826e-04 * Ta**2*va*Pa**3 +
        2.17508610e-04 * va**2*Pa**3 +
        -6.66724702e-05 * Ta*va**2*Pa**3 +
        3.33217140e-05 * va**3*Pa**3 +
        -2.26921615e-03 * D_Tmrt*Pa**3 +
        3.80261982e-04 * Ta*D_Tmrt*Pa**3 +
        -5.45314314e-09 * Ta**2*D_Tmrt*Pa**3 +
        -7.96355448e-04 * va*D_Tmrt*Pa**3 +
        2.53458034e-05 * Ta*va*D_Tmrt*Pa**3 +
        -6.31223658e-06 * va**2*D_Tmrt*Pa**3 +
        3.02122035e-04 * D_Tmrt**2*Pa**3 +
        -4.77403547e-06 * Ta*D_Tmrt**2*Pa**3 +
        1.73825715e-06 * va*D_Tmrt**2*Pa**3 +
        -4.09087898e-07 * D_Tmrt**3*Pa**3 +
        6.14155345e-01 * Pa**4 +
        -6.16755931e-02 * Ta*Pa**4 +
        1.33374846e-03 * Ta**2*Pa**4 +
        3.55375387e-03 * va*Pa**4 +
        -5.13027851e-04 * Ta*va*Pa**4 +
        1.02449757e-04 * va**2*Pa**4 +
        -1.48526421e-03 * D_Tmrt*Pa**4 +
        -4.11469183e-05 * Ta*D_Tmrt*Pa**4 +
        -6.80434415e-06 * va*D_Tmrt*Pa**4 +
        -9.77675906e-06 * D_Tmrt**2*Pa**4 +
        8.82773108e-02 * Pa**5 +
        -3.01859306e-03 * Ta*Pa**5 +
        1.04452989e-03 * va*Pa**5 +
        2.47090539e-04 * D_Tmrt*Pa**5 +
        1.48348065e-03 * Pa**6)

def UTCIhot(ds_dict, hum_var='both', tmrt=None, hazard_thresholds=hazard_thresholds):
    return _UTCI(ds_dict, hum_var=hum_var, tmrt=tmrt, hotorcold="hot", hazard_thresholds=hazard_thresholds)


def HWF(ds_dict, base_dict, percentile_base=90, hazard_thresholds = hazard_thresholds["HWF"], hwd_threshold=3, detrend=True):
    """
    Heatwave Frequency (HWF)
    Count all days that are part of a heatwave:
        - daily mean temp > calendar-day Xth percentile (90th default)
        - daily mean temp > annual mean for the grid point
        - percentile_base is the percentile threshold in the base period (e.g. 90 = 90th percentile threshold). If threshold = 90, heatwave is >= 3 days positive anom vs. 90th percentile value
        - heatwave lasts at least X consecutive days (3 default)

       detrend : bool
        If True (default), remove the linear warming trend from T before comparison
        so that heatwave detection reflects variability rather than the forced trend.
        In future projections, temperatures may shift higher causing more heatwave days,
        even if not necessarily part of a heatwave.
        The trend is estimated over the full ds_dict time series and removed.
        If False, raw T values are used.
    """

    # get calendar-day percentile threshold
    varname = f"t{str(percentile_base)}p_calday"
    TXp_calday = base_dict[varname]
    TXp_calday = _check_and_convert_units(da=TXp_calday, input_var=varname, conv_type="C") 

    # daily mean temperature
    T = ds_dict['tas'] 
    T = _check_and_convert_units(da=T, input_var="tas", conv_type="C") # could do average of tasmin and tasmax if wanted 

    steps_per_year = _get_tsteps(T)

    # remove linear warming trend per grid cell via polyfit, preserving the mean
    # does not detrend if data are less than 10 years
    if detrend:
        def _detrend(arr):
            t = np.arange(arr.shape[0])
            slope, intercept = np.polyfit(t, arr, 1)
            return arr - (slope * t + intercept) + arr.mean()

        T = xr.apply_ufunc(
            _detrend,
            T.chunk({"lat": -1, "lon": -1, "time": -1}).compute(),
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            output_dtypes=[T.dtype],
        )

    # mask warmer-than-average annual temperature only
    T_base_avg = base_dict["tas"].mean("time")
    T_base_avg = _check_and_convert_units(da=T_base_avg, input_var="tas", conv_type="C") 
    T_warm = T.where(T > T_base_avg)

    # difference from calendar-day percentile threshold
    T_anom = T_warm.groupby("time.dayofyear") - TXp_calday

    # boolean mask of heatwave candidate days
    heat_mask = T_anom > 0

    # rolling sum over hwd_threshold days (non-centered)
    rolling_sum = T_anom.where(heat_mask).rolling(time=hwd_threshold, center=False).count()
    window_all_hot = rolling_sum == hwd_threshold  # True where all days in window exceed threshold

    # expand True to all days in each consecutive heatwave
    mask_expanded = xr.zeros_like(heat_mask, dtype=bool)
    for shift in range(hwd_threshold):
        mask_expanded |= window_all_hot.shift(time=shift, fill_value=False)

    # count number of days per year that are part of a heatwave
    HWF = _ann_frac(T.where(mask_expanded).resample(time="1YE").count(), steps_per_year).rename("HWF")
    HWF = _assign_hazard_level(HWF, frac_thresholds=hazard_thresholds)

    return HWF

def TXC(ds_dict, hazard_thresholds=hazard_thresholds["TXC"]):
    """
    Days > X°C as general "hot day" temperature thresholds  
    Included as representative of other impact metrics within the same category
    """
    TX = ds_dict['tasmax'] 
    TX = _check_and_convert_units(da=TX, input_var="tasmax", conv_type="C") 

    # get days > 35C
    TXC_levels = _annual_exceedance_frac(TX, hazard_thresholds, var_name="TXC")
    TXC_levels = _assign_hazard_level(TXC_levels)

    return TXC_levels

def TR(ds_dict, TR_thresh = 20, hazard_thresholds=hazard_thresholds["TR"]):
    """
    Tropical nigths = days min T > 20°C 
    Proxy for hot nights which have been associated with increased mortality in regions across the globe
    Also a metric from ETCCDI
    """

    # get calendar-day percentile threshold
    TN = ds_dict["tasmin"]
    TN = _check_and_convert_units(da=TN, input_var="tasmin", conv_type="C") 

    # get number of time steps in a year
    steps_per_year = _get_tsteps(TN)

    # get tropical nights (<20°C, default)
    TR = TN.where(TN > TR_thresh).resample(time="1YE").count()
    TR = _ann_frac(TR, steps_per_year).rename("TR")
    TR = _assign_hazard_level(TR, frac_thresholds=hazard_thresholds)

    return TR

# =================
# COLD EXTREMES
# =================
def TNXp(ds_dict, base_dict, hazard_thresholds = hazard_thresholds["TNXp"], temp_max=15):
    '''
    Days < Xth percentile of temperature (must be < X °C (15°C Default)) 
    '''

    TN = ds_dict["tasmin"]
    TN = _check_and_convert_units(da=TN, input_var="tasmin", conv_type="C") 
    TN = TN.where(TN < temp_max)

    # get number of time steps in a year
    steps_per_year = _get_tsteps(TN)

    tasmin_base_percentile_vals = [float(k.split("_")[1].replace("pt", ".").replace("p", "")) for k in base_dict.keys() if k.startswith('tasmin_') and k.endswith('p')]
    tasmin_base_percentile_keys = sorted(
        [k for k in base_dict.keys() if k.startswith('tasmin_') and k.endswith('p')],
        key=lambda k: float(k.replace('tasmin_', '').replace('pt', '.').replace('p', '')),
        reverse=True
        )
    if sorted(tasmin_base_percentile_vals) != sorted(hazard_thresholds):
        print(f"Cannot calculate TNXp because based period tasmin percentiles do not match hazard_thresholds\n base period percentiles: {tasmin_base_percentile_vals}. hazard_thresholds: {hazard_thresholds}. \nSkipping...")
        return None
    
    # thresholds are lat/lon dependent so use a bespoke exceedance counting method
    da_list = []
    for _, key in enumerate(tasmin_base_percentile_keys):
        th = base_dict[key]
        da_count = (TN < th).resample(time='1YE').sum(dim='time', skipna=True)
        da_list.append(da_count)
    
    # Concatenate along new 'level' dimension
    TNXp = xr.concat(da_list, dim='level')
    TNXp = TNXp.assign_coords(level=np.arange(1, len(tasmin_base_percentile_keys) + 1))
    TNXp.attrs['level_values'] = tasmin_base_percentile_keys
    TNXp = _ann_frac(TNXp, steps_per_year).rename("TNXp")
    TNXp = _assign_hazard_level(TNXp)
    TNXp.attrs['level_thresholds'] = [
        {
            "level": int(i + 1),
            "threshold_value": tasmin_base_percentile_vals[i],
            "unit": "percentile",
            "source": "base period distribution"
        }
        for i in range(len(tasmin_base_percentile_keys))
    ]

    return TNXp

def UTCIcold(ds_dict, hum_var='both', tmrt=None, hazard_thresholds=hazard_thresholds):
    return _UTCI(ds_dict, hum_var=hum_var, tmrt=tmrt, hotorcold="cold", hazard_thresholds=hazard_thresholds)


# =================
# EXTREME WEATHER
# =================
# Quilcaille et al., 2023
# large ensemble recommendation from quilcaille et al., 2023 is average RH

# maybe include the comparison indices in Sharples
def FI(ds_dict,  hazard_thresholds = hazard_thresholds["FI"]):
    """
    Fire danger index (Sharples et al., 2009) https://doi.org/10.1016/j.envsoft.2008.10.012
    """
    
    U = ds_dict["sfcWind"]
    U = _check_and_convert_units(da=U, input_var="sfcWind", conv_type="km h-1") 
    
    FMI = _FMI(ds_dict)

    # wind speed threshold = 1 to ensure fire danger always > 0
    FI = (U.where(U > 1, 1))/FMI

    # count number of days per year FI > thresholds
    FI_levels = _annual_exceedance_frac(FI, hazard_thresholds=hazard_thresholds, var_name="FI")
    FI_levels = _assign_hazard_level(FI_levels)

    return FI_levels
    
def _FMI(ds_dict):
    """
    Fuel moisture index (Sharples et al., 2009) https://doi.org/10.1016/j.envsoft.2008.10.012
    """

    T = ds_dict["tas"]
    T = _check_and_convert_units(da=T, input_var="tas", conv_type="C") 
    RH = ds_dict["hurs"]
    RH = _check_and_convert_units(da=RH, input_var="hurs", conv_type="%") 
    RH = RH.clip(0.1, 99.9999)

    FMI = 10 - 0.25*(T - RH)

    return FMI

def HDW(ds_dict, hazard_thresholds=hazard_thresholds["HDW"]):
    """
    Hot-Dry-Windy index from Srock et al., 2018 https://doi.org/10.3390/atmos9070279
    Dangerous days are days where HDW > Xth percentile for that grid cell (90th is default)
    """

    T = ds_dict["tas"]
    T = _check_and_convert_units(da=T, input_var="tas", conv_type="C") 
    RH = ds_dict["hurs"]
    RH = _check_and_convert_units(da=RH, input_var="hurs", conv_type="%") 
    RH = RH.clip(0.1, 99.9999)
    U = ds_dict["sfcWind"]
    U = _check_and_convert_units(da=U, input_var="sfcWind", conv_type="m s-1") 

    # saturation vapor pressure via Tetens' equation
    es_positive = 0.611 * np.exp(17.27 * T / (T + 237.3))
    es_negative = 0.611 * np.exp(21.87 * T / (T + 265.5))
    es = xr.where(T > 0, es_positive, es_negative) # apply Tetens' equation 

    # vapor pressure deficit
    VPD = (1 - RH/100)*es
    HDW = U*VPD

    # count number of days per year HWD > thresholds
    HDW_levels = _annual_exceedance_frac(HDW, hazard_thresholds=hazard_thresholds, var_name="HDW")
    HDW_levels = _assign_hazard_level(HDW_levels)

    return HDW_levels

# !!!!
# Canadian FWI 
# !!!!

"""
Canadian Fire Weather Index (FWI) System - Pythonic xarray implementation
Based on Quilcaille et al., 2023 (https://doi.org/10.5194/essd-15-2153-2023)

This version uses xarray's native operations for cleaner, more efficient code:
- Automatic handling of all dimensions (time, lat, lon, member, etc.)
- Vectorized operations where possible
- No manual loops over ensemble members or spatial dimensions
- Uses .shift() for temporal dependencies instead of manual indexing
"""

def _get_day_length_factor(lat, month):
    """
    Calculate effective day length factor using xarray-native operations
    Must match original logic exactly for identical results
    
    Parameters:
    -----------
    lat : xr.DataArray
        Latitude coordinate
    month : xr.DataArray
        Month values (1-12) with time dimension
    
    Returns:
    --------
    day_length_factor : xr.DataArray
        Latitude and month-dependent day length adjustment
    """
    # Day length table - matches original exactly
    # Order matters! Apply in same sequence as original
    day_length_table = np.array([
        [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0],   # 20°N
        [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8],      # 40°N
        [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],        # 0° (equator)
        [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2],      # 40°S
        [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8]    # 60°S
    ])
    
    # Initialize with zeros
    day_length = xr.zeros_like(lat, dtype=float)
    
    # Get month index (handle both scalar and array)
    if isinstance(month, (int, np.integer)):
        month_idx = month - 1
    else:
        month_idx = month.values - 1
    
    # Apply in same order as original for exact match
    day_length = xr.where(lat >= 20, day_length_table[0, month_idx], day_length)
    day_length = xr.where(lat >= 40, day_length_table[1, month_idx], day_length)
    day_length = xr.where((lat > -20) & (lat < 20), day_length_table[2, month_idx], day_length)
    day_length = xr.where(lat <= -20, day_length_table[3, month_idx], day_length)
    day_length = xr.where(lat <= -40, day_length_table[4, month_idx], day_length)
    
    return day_length


def _apply_overwintering(dc, month, lat):
    """
    Apply overwintering to DC using vectorized operations
    
    Parameters:
    -----------
    dc : xr.DataArray
        Drought Code values
    month : xr.DataArray
        Month (1-12)
    lat : xr.DataArray
        Latitude
        
    Returns:
    --------
    dc_adjusted : xr.DataArray
        DC with overwintering applied
    """
    # Winter months by hemisphere (vectorized)
    winter_north = ((month >= 11) | (month <= 3)) & (lat >= 30)
    winter_south = ((month >= 5) & (month <= 9)) & (lat <= -30)
    
    # Apply wetting factor (0.75) in winter
    return xr.where(winter_north | winter_south, dc * 0.75, dc)



def _ffmc_step(temp, rh, wind, rain, ffmc_prev):
    """Fine Fuel Moisture Code - single time step (all dimensions)"""
    # Moisture content from FFMC
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
    
    # Rain effect (vectorized conditionals)
    rain_effect = (
        xr.where(rain > 1.5,
            42.5 * rain * np.exp(-100.0/(251.0-mo)) * (1.0 - np.exp(-6.93/rain)) + 
            0.0015 * (mo - 150.0)**2 * np.sqrt(rain),
        xr.where(rain > 0.5,
            42.5 * rain * np.exp(-100.0/(251.0-mo)) * (1.0 - np.exp(-6.93/rain)),
            0))
    )
    rf = (mo + rain_effect).clip(0, 250)
    
    # Equilibrium moisture
    ed = 0.942 * rh**0.679 + 11.0 * np.exp((rh-100.0)/10.0) + \
         0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    ew = 0.618 * rh**0.753 + 10.0 * np.exp((rh-100.0)/10.0) + \
         0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    
    # Drying/wetting rates
    ko = 0.424 * (1.0 - (rh/100.0)**1.7) + 0.0694 * np.sqrt(wind) * (1.0 - (rh/100.0)**8)
    kd = ko * 0.581 * np.exp(0.0365 * temp)
    kw = ko * 0.581 * np.exp(0.0365 * temp)
    
    # Moisture content (vectorized logic)
    m = xr.where(rf > ed,
                 ew + (rf - ew) * 10.0**(-kd),
                 xr.where(rf < ew,
                         ed - (ed - rf) * 10.0**(-kw),
                         rf))
    
    # Convert back to FFMC
    return (59.5 * (250.0 - m) / (147.2 + m)).clip(0, 101)


def _dmc_step(temp, rh, rain, dmc_prev, day_length):
    """Duff Moisture Code - single time step"""
    # Rain effect
    re = xr.where(rain > 1.5, 0.92 * rain - 1.27, 0.0)
    
    mo = 20.0 + np.exp(5.6348 - dmc_prev/43.43)
    
    b = xr.where(dmc_prev <= 33,
                100.0 / (0.5 + 0.3 * dmc_prev),
                xr.where(dmc_prev <= 65,
                        14.0 - 1.3 * np.log(dmc_prev),
                        6.2 * np.log(dmc_prev) - 17.2))
    
    mr = mo + 1000.0 * re / (48.77 + b * re)
    pr = (244.72 - 43.43 * np.log(mr - 20.0)).clip(min=0)
    
    # Drying
    k = 1.894 * (temp.clip(min=-1.1) + 1.1) * (100.0 - rh) * day_length * 1e-4
    
    return xr.where(rain > 1.5, pr + k, dmc_prev + k).clip(min=0)


def _dc_step(temp, rain, dc_prev, day_length, month, lat):
    """Drought Code - single time step"""
    # Apply overwintering
    dc_prev = _apply_overwintering(dc_prev, month, lat)
    
    # Rain effect
    rd = xr.where(rain > 2.8, 0.83 * rain - 1.27, 0.0)
    
    qo = 800.0 * np.exp(-dc_prev/400.0)
    qr = qo + 3.937 * rd
    dr = (400.0 * np.log(800.0/qr)).clip(min=0)
    
    # Drying
    v = 0.36 * (temp.clip(min=-2.8) + 2.8) + day_length
    
    return xr.where(rain > 2.8, dr + v, dc_prev + v).clip(min=0)


def _isi_from_ffmc(wind, ffmc):
    """Initial Spread Index from FFMC and wind"""
    mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * np.exp(-0.1386 * mo) * (1.0 + mo**5.31 / 4.93e7)
    fw = np.exp(0.05039 * wind)
    return 0.208 * fw * ff


def _bui_from_codes(dmc, dc):
    """Buildup Index from DMC and DC"""
    return xr.where(
        dmc <= 0.4 * dc,
        0.8 * dmc * dc / (dmc + 0.4 * dc),
        dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc)**1.7)
    ).clip(min=0)


def _fwi_from_isi_bui(isi, bui):
    """Fire Weather Index from ISI and BUI"""
    bb = xr.where(
        bui > 80,
        0.1 * isi * (1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui))),
        0.1 * isi * (0.626 * bui**0.809 + 2.0)
    )
    
    return xr.where(bb <= 1, bb, np.exp(2.72 * (0.434 * np.log(bb))**0.647))


def FWI(ds_dict, use_hursmin=True, init_values=None, fwi_mask_file=None, environmental_zone_file=None):
    """
    Calculate Canadian Fire Weather Index using xarray-native operations
    
    Automatically handles all dimensions (time, lat, lon, member, etc.)
    No manual looping required - xarray broadcasts operations correctly
    
    Parameters:
    -----------
    ds_dict : dict
        Dictionary of xr.DataArrays with any combination of dimensions:
        - 'tasmax': max temperature (K)
        - 'pr': precipitation (kg/m2/s)
        - 'sfcWind': wind speed (m/s)
        - 'hursmin' or 'hurs': relative humidity (%)
    use_hursmin : bool
        Prefer hursmin over hurs if both available
    init_values : dict, optional
        Initial values {'ffmc': 85, 'dmc': 6, 'dc': 15}
        
    Returns:
    --------
    fwi : xr.DataArray
        Daily FWI with all original dimensions preserved
    components : dict
        All FWI components (ffmc, dmc, dc, isi, bui, fwi)
        
    Examples:
    ---------
    # Works with any dimensions:
    # - Single gridpoint: (time,)
    # - Spatial: (time, lat, lon)
    # - Ensemble: (time, lat, lon, member)
    # - Model comparison: (time, lat, lon, member, model)
    """
    print("Calculating Canadian Fire Weather Index...")

    # FWI does not perform well if chunked so load if chunked
    for key, da in ds_dict.items():
        if key in ["tasmax", "pr", "sfcWind", "hursmin", "hurs"]:
            if da.chunks is not None:  # means it's dask-backed/chunked
                ds_dict[key] = da.load()

    # import FWI mask 
    # when more than 80 % of the surface of the grid cell is flagged as bare areas, water, snow, and ice or sparsely vegetated, 
    # it is considered to be infrequent burning. Adaptation of Quilcaille et al (2023), similar to Abatzoglou et al. (2019)
    # !!!!! Note to self: implement logic to pull from cloud  !!!
    if fwi_mask_file is not None:
        fwi_mask = xr.open_dataset(fwi_mask_file).mask_infreq_burning # True if 
    else: 
        print("NEED TO IMPLEMENET logic for cloud mask pull")

    # Convert units
    TX = ds_dict["tasmax"]
    TX = _check_and_convert_units(da=TX, input_var="tasmax", conv_type="C").where(~fwi_mask)
    precip = ds_dict["pr"]
    precip = _check_and_convert_units(da=precip, input_var="pr", conv_type="mm day-1").where(~fwi_mask)
    wind = ds_dict["sfcWind"]
    wind = _check_and_convert_units(da=wind, input_var="sfcWind", conv_type="km h-1").where(~fwi_mask)
    
    # Get relative humidity
    if use_hursmin and 'hursmin' in ds_dict:
        print("Using hursmin")
        rh = ds_dict['hursmin']
        rh = _check_and_convert_units(da=rh, input_var="hursmin", conv_type="%").where(~fwi_mask) 
    else:
        print("Using hurs")
        rh = ds_dict['hurs']
        rh = _check_and_convert_units(da=rh, input_var="hurs", conv_type="%").where(~fwi_mask) 
    rh = rh.clip(0, 100)
    
    # Get temporal and spatial coordinates
    lat = TX.lat
    time = TX.time
    if hasattr(time, 'dt'):
        month = time.dt.month
    else:
        month = xr.DataArray(
            [(i // 30) % 12 + 1 for i in range(len(time))],
            dims=['time'],
            coords={'time': time}
        )
    
    # Pre-calculate day length for all time steps
    # This ensures proper broadcasting
    day_length = xr.concat(
        [_get_day_length_factor(lat, m) for m in month.values],
        dim='time'
    )
    day_length['time'] = time
    
    # Initialize arrays
    if init_values is None:
        init_values = {'ffmc': 85.0, 'dmc': 6.0, 'dc': 15.0}
    
    # Create empty arrays with same structure as inputs
    template = TX * 0 # Preserves all dimensions and coordinates
    ffmc = template.copy()
    dmc = template.copy()
    dc = template.copy()
    isi = template.copy()
    bui = template.copy()
    fwi = template.copy()

    n_times = len(TX.time)
    print(f"Processing {n_times} time steps...")
    
    for i in range(n_times):
        if i % 365 == 0:
            print(f"  Day {i}/{n_times}")
        
        # Get current values (preserves all non-time dimensions)
        t = TX.isel(time=i)
        r = rh.isel(time=i)
        w = wind.isel(time=i)
        p = precip.isel(time=i)
        dl = day_length.isel(time=i)
        m = month.isel(time=i)
        
        # Previous values (or init)
        if i == 0:
            ffmc_prev = init_values['ffmc']
            dmc_prev = init_values['dmc']
            dc_prev = init_values['dc']
        else:
            ffmc_prev = ffmc.isel(time=i-1)
            dmc_prev = dmc.isel(time=i-1)
            dc_prev = dc.isel(time=i-1)
        
        # Calculate (broadcasts over all remaining dimensions)
        ffmc[dict(time=i)] = _ffmc_step(t, r, w, p, ffmc_prev)
        dmc[dict(time=i)] = _dmc_step(t, r, p, dmc_prev, dl)
        dc[dict(time=i)] = _dc_step(t, p, dc_prev, dl, m, lat)
        isi[dict(time=i)] = _isi_from_ffmc(w, ffmc.isel(time=i))
        bui[dict(time=i)] = _bui_from_codes(dmc.isel(time=i), dc.isel(time=i))
        fwi[dict(time=i)] = _fwi_from_isi_bui(isi.isel(time=i), bui.isel(time=i))
    
    # components = {
    #     'ffmc': ffmc,
    #     'dmc': dmc,
    #     'dc': dc,
    #     'isi': isi,
    #     'bui': bui,
    #     'fwi': fwi
    # } # may not need these outputs
        
    # count number of days per year FWI > thresholds
    # FWI_levels = _annual_exceedance_frac(fwi, hazard_thresholds=hazard_thresholds, var_name="FWI")
    # FWI_levels = _assign_hazard_level(FWI_levels)

    # !!!!! Note to self, make cloud pull logic
    if environmental_zone_file is not None:
        environmental_zones = xr.open_dataset(environmental_zone_file).environmental_zone
    else:
        # make cloud logic
        return None
    
    FWI_levels = _annual_exceedance_frac_fwi(fwi, environmental_zones, fwi_thesholds)
    FWI_levels = _assign_hazard_level(FWI_levels)

    return FWI_levels

# =================
# AIR QUALITY
# =================
# need to incorporate thesholds, annual exceedance, etc here 
def O3(ds_dict, hazard_thresholds=None, mda8_scale_file=None, timeres="mon", mda8_scale_varname="o3", hazard_thresholds_dict=hazard_thresholds):
    """
    Surface ozone concentration.
    Convert from CMIP units (mol mol-1) to ppb
    Note: CMIP6 o3 variable is typically in model levels and 
    metadata will have the equation for converting from model level to pressure level.
    Input the o3 variable at the surface level (i.e. do not include other model/pressure levels)
    For now, this step is the preprocessing responsibility of the user because different models have
    different model level -> pressure level equations.
    There is an attribute "positive" with values "up" or "down" which could indicate whether model levels
    are ascending or descending from surface to TOA, but we found these attributes to be unreliable.
    
    To avoid incorrectly selecting a non-surface level, we recommend using the model level to pressure level 
    conversion, then extracting the highest pressure level (i.e. the surface)  

    mda8 = maximum daily average 8-hour concentration 
    """

    # need additional variables for conversion, so do it in function
    O3 = ds_dict["o3"]

    if ("plev" in O3.dims) or ("lev" in O3.dims):
        O3 = _get_surface(O3, "o3")
    
    units_attr = None
    for k in O3.attrs:
        if k.lower() in ["unit", "units"]:
            units_attr = O3.attrs[k]
            break

    # ozone conversions
    if units_attr is None:
        print("No units attribute passed for ozone. Assuming units are mol/mol")
        convert_o3 = True
    elif units_attr in ["mol mol-1", "mol/mol", "mol mol^-1"]:
        convert_o3 = True
    elif units_attr in ["µg/m^3", "µg m^-3", "µg m-3"]:
        # already in correct units, just keep as is
        convert_o3 = False  
    # if already ug/m^3, continue, else flag
    elif units_attr not in ["mol mol-1", "mol/mol", "mol mol^-1", "µg m-3", "µg/m3", "µg m^-3", "µg/m^3"]:
        print(f"Units attribute: {units_attr} not recognized. Must be mol mol-1 or µg m-3. Skipping O3 calculation...")
        return None
    
    # convert from mol mol-1 to µg/m^3
    if convert_o3 == True:
        # get approx air density
        T = ds_dict["tas"]
        T = _check_and_convert_units(da=T, input_var="tas", conv_type="K") 
        ps = ds_dict["ps"]
        ps = _check_and_convert_units(da=ps, input_var="ps", conv_type="Pa") 

        # convert mol mol-1 to µg/m^3
        M_O3 = 48  # molecular weight of ozone g/mol
        R = 8.314  # J/(mol·K)
        O3 = 1e6*(O3 * (M_O3 * ps) / (R * T))
        

    # Ensure one level for O3 (must be surface level) 
    if "plev" in O3.dims:
        num_plev = len(O3.plev)
        if num_plev > 1:
            print("Please only input the surface pressure level 'plev' for 'o3' variable")
    elif "lev" in O3.dims:
        num_plev = len(O3.lev)
        if num_plev > 1:
            print("Please only input the surface pressure level 'lev' for 'o3' variable")
    
    # scale factor is fraction that average is of mda8 (monthly average)
    # So to scale from monthly average to mda8, divide by scale factor
    mda8_scale_factor = xr.open_dataset(mda8_scale_file)[mda8_scale_varname] # scale from avg to mda8 
    # check if monthly or daily 
    if timeres.lower() in ["day", "daily"]:
        # if user passes custom thresholds, use them. otherwise pull from GCHI defaults
        hazard_thresholds = hazard_thresholds if hazard_thresholds is not None else hazard_thresholds_dict["O3day"]
        # daily data 
        O3 = O3.resample(time="1D").mean()
        O3.values = (O3.groupby("time.month")/mda8_scale_factor).values
        # fraction of year o3 concentration > daily mda8 thresholds
        O3_levels = _annual_exceedance_frac(O3, hazard_thresholds=hazard_thresholds, var_name="O3")
        O3_levels = _assign_hazard_level(O3_levels)

    elif timeres.lower() in ["mon", "monthly"]:
        # if user passes custom thresholds, use them. otherwise pull from GCHI defaults
        hazard_thresholds = hazard_thresholds if hazard_thresholds is not None else hazard_thresholds_dict["O3mon"]
        # monthly data 
        O3 = O3.resample(time="1ME").mean()
        O3.values = (O3.groupby("time.month")/mda8_scale_factor).values
        # If the annual average exceeds the threshold, count # of months. else, no exceedance 
        O3_levels = _annual_exceedance_frac_aq(O3, hazard_thresholds=hazard_thresholds, var_name="O3")
        O3_levels = _assign_hazard_level(O3_levels)

    else:
        steps_per_year = _get_tsteps(O3)
        if steps_per_year.max(["time", "lat", "lon"]) > 12:
            # daily data 
            O3 = O3.resample(time="1D").mean()
            O3.values = (O3.groupby("time.month")/mda8_scale_factor).values
            # if user passes custom thresholds, use them. otherwise pull from GCHI defaults
            hazard_thresholds = hazard_thresholds if hazard_thresholds is not None else hazard_thresholds_dict["O3day"]
            # fraction of year o3 concentration > daily mda8 thresholds
            O3_levels = _annual_exceedance_frac(O3, hazard_thresholds=hazard_thresholds, var_name="O3")
            O3_levels = _assign_hazard_level(O3_levels)
            res_detected = "daily"
        elif (steps_per_year.max(["time", "lat", "lon"]) > 1) & (steps_per_year.mean(["time", "lat", "lon"]) <= 12):
             # monthly data 
            O3 = O3.resample(time="1ME").mean()
             # if user passes custom thresholds, use them. otherwise pull from GCHI defaults
            hazard_thresholds = hazard_thresholds if hazard_thresholds is not None else hazard_thresholds_dict["O3mon"]
            O3.values = (O3.groupby("time.month")/mda8_scale_factor).values
            # If the annual average exceeds the threshold, count # of months. else, no exceedance 
            O3_levels = _annual_exceedance_frac_aq(O3, hazard_thresholds=hazard_thresholds, var_name="O3")
            O3_levels = _assign_hazard_level(O3_levels)
            res_detected = "monthly"
        else:
            print("Could not detect time resolution on ozone input. Please check. It should be monthly or daily. Then pass argument 'timeres' = 'day' or 'mon'")

        print(f"Please check time resolution on ozone input. Detected {res_detected} resolution.")
    
    return O3_levels

def PM2pt5(ds_dict,  hazard_thresholds=None,timeres="mon",hazard_thresholds_dict=hazard_thresholds):
    """
    Estimate PM2.5 concentration from aerosol components
    PM2.5 = BC + OA + SO4 + (0.25*SS) + (0.1*DU)
    Turnock et al. 2020 eq. 1 https://doi.org/10.5194/acp-20-14547-2020 

    Note: CMIP6 aer vars typically in model levels and 
    metadata will have the equation for converting from model level to pressure level.
    Input the variables at the surface level (i.e. do not include other model/pressure levels)
    For now, this step is the preprocessing responsibility of the user because different models have
    different model level -> pressure level equations.
    There is an attribute "positive" with values "up" or "down" which could indicate whether model levels
    are ascending or descending from surface to TOA, but we found these attributes to be unreliable.
    
    To avoid incorrectly selecting a non-surface level, we recommend using the model level to pressure level 
    conversion, then extracting the highest pressure level (i.e. the surface)  
    """
    
    # get aerosol species
    BC = ds_dict["mmrbc"]
    OA = ds_dict["mmroa"]
    SO4 = ds_dict["mmrso4"]
    SS = ds_dict["mmrss"]
    DU = ds_dict["mmrdust"]

    if ("plev" in BC.dims) or ("lev" in BC.dims):
        BC = _get_surface(BC, "mmrbc")
        OA = _get_surface(OA, "mmroa")
        SO4 = _get_surface(SO4, "mmrso4")
        SS = _get_surface(SS, "mmrss")
        DU = _get_surface(DU, "mmrdust")

    # thresholds are in µg m-3, so if already correct units, avoid conversion
    for da in [BC, OA, SO4, SS, DU]:
        attrs_lower = {k.lower(): v for k, v in da.attrs.items()}
        target_keys = {"unit", "units"}
        found_key = next((k for k in target_keys if k in attrs_lower), None)
        if found_key: 
            if da.attrs[found_key] in ["µg/m^3", "µg m^-3", "µg m-3"]:
                # already in correct units, just sum 
                print(f"Detected pm2.5 input units as µg m-3. Assuming all variables have equivalent units")
                pm2pt5 = BC + OA + SO4 + (0.25*SS) + (0.1*DU) # in mixing ratio
                break

        # if in kg kg-1, sum and convert         
        BC = _check_and_convert_units(da=BC, input_var="mmrbc", conv_type="kg kg-1") 
        OA = _check_and_convert_units(da=OA, input_var="mmroa", conv_type="kg kg-1") 
        SO4 = _check_and_convert_units(da=SO4, input_var="mmrso4", conv_type="kg kg-1") 
        SS = _check_and_convert_units(da=SS, input_var="mmrss", conv_type="kg kg-1") 
        DU = _check_and_convert_units(da=DU, input_var="mmrdust", conv_type="kg kg-1") 

        # get approx air density
        T = ds_dict["tas"]
        T = _check_and_convert_units(da=T, input_var="tas", conv_type="K") 
        ps = ds_dict["ps"]
        ps = _check_and_convert_units(da=ps, input_var="ps", conv_type="Pa") 
        rho = ps / (287*T)

        # calculate offline due to most models not having PM2.5 cmip6 diagnostic available
        pm2pt5 = BC + OA + SO4 + (0.25*SS) + (0.1*DU) # in mixing ratio
        # to get ug/m^3 do 1e9 mixing ratio * density 
        pm2pt5 = (pm2pt5 * rho)*1e9

    # check if monthly or daily 
    if timeres.lower() in ["day", "daily"]:
        # if user passes custom thresholds, use them. otherwise pull from GCHI defaults
        hazard_thresholds = hazard_thresholds if hazard_thresholds is not None else hazard_thresholds_dict["PM2pt5day"]
        # daily data 
        pm2pt5 = pm2pt5.resample(time="1D").mean()
        # fraction of year pm2.5 concentration > daily thresholds
        pm2pt5_levels = _annual_exceedance_frac(pm2pt5, hazard_thresholds=hazard_thresholds, var_name="PM2pt5")
        pm2pt5_levels = _assign_hazard_level(pm2pt5_levels)
    elif timeres.lower() in ["mon", "monthly"]:
        # if user passes custom thresholds, use them. otherwise pull from GCHI defaults
        hazard_thresholds = hazard_thresholds if hazard_thresholds is not None else hazard_thresholds_dict["PM2pt5mon"]
        # monthly data 
        pm2pt5 = pm2pt5.resample(time="1ME").mean()
        # If the annual average exceeds the threshold, count # of months. else, no exceedance 
        pm2pt5_levels = _annual_exceedance_frac_aq(pm2pt5, hazard_thresholds=hazard_thresholds, var_name="PM2pt5")
        pm2pt5_levels = _assign_hazard_level(pm2pt5_levels)
    else:
        steps_per_year = _get_tsteps(pm2pt5)
        if steps_per_year.max(["time", "lat", "lon"]) > 12:
            # daily data 
            pm2pt5 = pm2pt5.resample(time="1D").mean()
            # if user passes custom thresholds, use them. otherwise pull from GCHI defaults
            hazard_thresholds = hazard_thresholds if hazard_thresholds is not None else hazard_thresholds_dict["PM2pt5day"]
            # fraction of year o3 concentration > daily mda8 thresholds
            pm2pt5_levels = _annual_exceedance_frac(pm2pt5, hazard_thresholds=hazard_thresholds, var_name="PM2pt5")
            pm2pt5_levels = _assign_hazard_level(pm2pt5_levels)
            res_detected = "daily"
        elif (steps_per_year.max(["time", "lat", "lon"]) > 1) & (steps_per_year.mean(["time", "lat", "lon"]) <= 12):
            # monthly data 
            pm2pt5 = pm2pt5.resample(time="1ME").mean()
             # if user passes custom thresholds, use them. otherwise pull from GCHI defaults
            hazard_thresholds = hazard_thresholds if hazard_thresholds is not None else hazard_thresholds_dict["PM2pt5mon"]
            # If the annual average exceeds the threshold, count # of months. else, no exceedance 
            pm2pt5_levels = _annual_exceedance_frac_aq(pm2pt5, hazard_thresholds=hazard_thresholds, var_name="PM2pt5")
            pm2pt5_levels = _assign_hazard_level(pm2pt5_levels)
            res_detected = "monthly"
        else:
            print("Could not detect time resolution on PM2.5 inputs. Please check. It should be monthly or daily. Then pass argument 'timeres' = 'day' or 'mon'")

        print(f"Please check time resolution on PM2.5 inputs. Detected {res_detected} resolution.")

    return pm2pt5_levels


# =================
# DROUGHT
# =================

def CDD(ds_dict, hazard_thresholds=hazard_thresholds["CDD"], min_threshold=10):
    """
    Consecutive Dry Days (CDD).
    Maximum number of consecutive days with precipitation < 1mm.
    Only count dry spells of at least X days (10 is default) to account for health-relevant dry spells
    """
    PR = ds_dict['pr']
    PR = _check_and_convert_units(da=PR, input_var="pr", conv_type="mm day-1") 
    
    # Count annual days in dry spells greater than X days     
    # boolean mask of dry days
    dry_mask = (PR < 1)  # true where precipitation < 1mm

    # rolling sum over X days, not centered (each day is start of 10-day window)
    # then count the windows where all X are dry 
    rolling_sum = PR.where(dry_mask).rolling(time=min_threshold, center=False).count()
    window_all_dry = rolling_sum == min_threshold  # boolean

    # expand the True values to cover all X days in the window where all dry
    # you end up with all days True that are part of a dry spell of at least X days
    mask_expanded = xr.zeros_like(dry_mask, dtype=bool)
    for shift in range(min_threshold):
        mask_expanded |= window_all_dry.shift(time=shift, fill_value=False)

    CDD = PR.where(mask_expanded).resample(time="1YE").count()

    # count number of days per year that are part of a CDD
    # get number of time steps in a year
    steps_per_year = _get_tsteps(PR)
    CDD = _ann_frac(CDD, steps_per_year).rename("CDD")
    CDD = _assign_hazard_level(CDD, frac_thresholds=hazard_thresholds)
    
    return CDD


def SPI(base_dict, ds_dict, timescale=6, hazard_thresholds = hazard_thresholds["SPI"]):
    """
    SPI using a fixed historical baseline (base_dict) 
    to evaluate a study period (ds_dict).
    Default thresholds from USDM drought classification https://droughtmonitor.unl.edu/About/AbouttheData/DroughtClassification.aspx
    """

    # Process Baseline Accumulation
    base_pr = _check_and_convert_units(da=base_dict['pr'], input_var="pr", conv_type="mm day-1") 
    base_pr = base_pr.resample(time="1ME").sum()
    base_acc = base_pr.rolling(time=timescale).sum().dropna('time')

    # Process Study Period Accumulation
    study_pr = _check_and_convert_units(da=ds_dict['pr'], input_var="pr", conv_type="mm day-1") 
    study_pr = study_pr.resample(time="1ME").sum()
    study_acc = study_pr.rolling(time=timescale).sum().dropna('time')

    def _fit_and_apply_gamma(hist_data, study_data):
        hist_data = hist_data[~np.isnan(hist_data)]
        # Output array initialized with NaNs matching the length of study_data
        out = np.full(study_data.shape, np.nan)
        study_mask = ~np.isnan(study_data)
        
        if len(hist_data) < 30: 
            return out
        
        # Fit parameters based ONLY on historical baseline
        params = stats.gamma.fit(hist_data, floc=0)
        
        # Apply those parameters to the valid indices of study data
        cdf = stats.gamma.cdf(study_data[study_mask], *params)
        out[study_mask] = stats.norm.ppf(np.clip(cdf, 1e-6, 0.999999))
        return out

    # Apply UFUNC with alignment override for different time periods
    SPI = xr.apply_ufunc(
        _fit_and_apply_gamma, base_acc, study_acc,
        input_core_dims=[['time'], ['time']], 
        output_core_dims=[['time']],
        exclude_dims=set(("time",)), 
        vectorize=True, 
        dask="parallelized",
        output_dtypes=[float]
    )
    
    # Put the study period time coordinates back on
    SPI = SPI.assign_coords(time=study_acc.time)

    # count number of days per year > thresholds
    SPI_levels = _annual_exceedance_frac(SPI, hazard_thresholds=hazard_thresholds, var_name="SPI")
    SPI_levels = _assign_hazard_level(SPI_levels)

    return SPI_levels


def SPEI(base_dict, ds_dict, timescale=6, hazard_thresholds = hazard_thresholds["SPEI"]):
    """
    Standardized Precipitation-Evapotranspiration Index (SPEI).
    Calculates annual month counts for specific USDM drought categories 
    using a fixed historical baseline.
    Default thresholds from USDM drought classification https://droughtmonitor.unl.edu/About/AbouttheData/DroughtClassification.aspx
    """
    
    # Baseline Water Balance (P - PET)
    # Using 1ME for Month End frequency
    PR_base = _check_and_convert_units(da=base_dict['pr'], input_var="pr", conv_type="mm day-1") 
    EVSPSBL_base = _check_and_convert_units(da=base_dict['evspsbl'], input_var="evspsbl", conv_type="mm day-1") 
    base_diff = PR_base - EVSPSBL_base
    base_acc = base_diff.resample(time="1ME").sum().rolling(time=timescale).sum().dropna('time')

    # Study Period Water Balance
    PR = _check_and_convert_units(da=ds_dict['pr'], input_var="pr", conv_type="mm day-1") 
    EVSPSBL = _check_and_convert_units(da=ds_dict['evspsbl'], input_var="evspsbl", conv_type="mm day-1")
    study_diff = PR - EVSPSBL
    study_acc = study_diff.resample(time="1ME").sum().rolling(time=timescale).sum().dropna('time')

    def _fit_and_apply_genlog(hist_data, study_data):
        # Remove NaNs from baseline for fitting
        hist_clean = hist_data[~np.isnan(hist_data)]
        
        # Initialize output array matching study_data length (the future period)
        out = np.full(study_data.shape, np.nan)
        study_mask = ~np.isnan(study_data)
        
        # Need enough data points to fit the distribution
        if len(hist_clean) < 30: 
            return out
        
        # Fit parameters based ONLY on historical baseline
        params = stats.genlogistic.fit(hist_clean)
        
        # Apply those parameters to the valid indices of the study data
        cdf = stats.genlogistic.cdf(study_data[study_mask], *params)
        out[study_mask] = stats.norm.ppf(np.clip(cdf, 1e-6, 0.999999))
        return out

    # Apply UFUNC with alignment override for non-overlapping periods
    SPEI = xr.apply_ufunc(
        _fit_and_apply_genlog, base_acc, study_acc,
        input_core_dims=[['time'], ['time']], 
        output_core_dims=[['time']],
        exclude_dims=set(("time",)), # Essential to prevent ValueError with mismatched dates
        vectorize=True, 
        dask="parallelized",
        output_dtypes=[float]
    )
    
    # Re-attach the future time coordinates
    SPEI = SPEI.assign_coords(time=study_acc.time)

    # count number of days per year > thresholds
    SPEI_levels = _annual_exceedance_frac(SPEI, hazard_thresholds=hazard_thresholds, var_name="SPEI")
    SPEI_levels = _assign_hazard_level(SPEI_levels)

    return SPEI_levels

# !! START HERE !! 
# =================
# VBD metrics
# =================


# !!!!!!!!!!!!!!!!!!!!!!!
# Disease

def VSmalaria(ds_dict, T_range=[22.9, 27.8], VBD_mask_file = None, hazard_thresholds=hazard_thresholds["VSmalaria"]):
    """
    malaria transmission suitability 
    """
    print("Calculating malaria_suitability...")

    # import VBD mask 
    # at least two consecutive months of NDVI above 0.125 (Ryan et al 2015; https://doi.org/10.1089/vbz.2015.1822)
    # !!!!! Note to self: implement logic to pull from cloud  !!!
    if VBD_mask_file is not None:
        VBD_mask = xr.open_dataset(VBD_mask_file).aridity_mask # True if 
    else: 
        print("NEED TO IMPLEMENET logic for cloud mask pull")

    T = ds_dict["tas"]
    T = _check_and_convert_units(da=T, input_var="tas", conv_type="C") 
    T = T.resample(time="1ME").mean()

    VSmalaria = T.where((T >= T_range[0]) & (T <= T_range[1]))
    VSmalaria = VSmalaria.resample(time="1YE").count()
    VSmalaria = VSmalaria.where(VBD_mask) # mask arid regions 

    # get number of time steps in a year
    steps_per_year = _get_tsteps(T)
    
    VSmalaria_levels = _ann_frac(VSmalaria, steps_per_year).rename("VSmalaria")
    VSmalaria_levels = _assign_hazard_level(VSmalaria_levels, frac_thresholds=hazard_thresholds)
    
    return VSmalaria_levels

def VSzika(ds_dict, T_range=[23.9, 34], VBD_mask_file=None, hazard_thresholds=hazard_thresholds["VSzika"]):
    """
    Zika transmission suitability
    """
    print("Calculating zika_suitability...")
    # import VBD mask 
    # at least two consecutive months of NDVI above 0.125 (Ryan et al 2015; https://doi.org/10.1089/vbz.2015.1822)
    # !!!!! Note to self: implement logic to pull from cloud  !!!
    if VBD_mask_file is not None:
        VBD_mask = xr.open_dataset(VBD_mask_file).aridity_mask # True if 
    else: 
        print("NEED TO IMPLEMENET logic for cloud mask pull")

    T = ds_dict["tas"]
    T = _check_and_convert_units(da=T, input_var="tas", conv_type="C") 
    T = T.resample(time="1ME").mean()

    VSzika = T.where((T >= T_range[0]) & (T <= T_range[1]))
    VSzika = VSzika.resample(time="1YE").count()
    VSzika = VSzika.where(VBD_mask) # mask arid regions 

    # get number of time steps in a year
    steps_per_year = _get_tsteps(T)
    
    VSzika_levels = _ann_frac(VSzika, steps_per_year).rename("VSzika")
    VSzika_levels = _assign_hazard_level(VSzika_levels, frac_thresholds=hazard_thresholds)
    
    return VSzika_levels


def VSdengueAeg(ds_dict, T_range=[19.9, 29.4], VBD_mask_file=None, hazard_thresholds=hazard_thresholds["VSdengueAeg"]):
    """
    Dengue (Aedes aegypti) transmission suitability
    """
    print("Calculating dengue_aegypti_suitability...")
    # import VBD mask 
    # at least two consecutive months of NDVI above 0.125 (Ryan et al 2015; https://doi.org/10.1089/vbz.2015.1822)
    # !!!!! Note to self: implement logic to pull from cloud  !!!
    if VBD_mask_file is not None:
        VBD_mask = xr.open_dataset(VBD_mask_file).aridity_mask # True if 
    else: 
        print("NEED TO IMPLEMENET logic for cloud mask pull")

    T = ds_dict["tas"]
    T = _check_and_convert_units(da=T, input_var="tas", conv_type="C") 
    T = T.resample(time="1ME").mean()
    
    VSdengueAeg = T.where((T >= T_range[0]) & (T <= T_range[1]))
    VSdengueAeg = VSdengueAeg.resample(time="1YE").count()
    VSdengueAeg = VSdengueAeg.where(VBD_mask) # mask arid regions 

    # get number of time steps in a year
    steps_per_year = _get_tsteps(T)
    
    VSdengueAeg_levels = _ann_frac(VSdengueAeg, steps_per_year).rename("VSdengueAeg")
    VSdengueAeg_levels = _assign_hazard_level(VSdengueAeg_levels, frac_thresholds=hazard_thresholds)
    
    return VSdengueAeg_levels


def VSdengueAlb(ds_dict, T_range=[21.3, 34], VBD_mask_file=None, hazard_thresholds=hazard_thresholds["VSdengueAeg"]):
    """
    Dengue (Aedes albopictus) transmission suitability
    """
    print("Calculating dengue_albopictus_suitability...")
    
    # import VBD mask 
    # at least two consecutive months of NDVI above 0.125 (Ryan et al 2015; https://doi.org/10.1089/vbz.2015.1822)
    # !!!!! Note to self: implement logic to pull from cloud  !!!
    if VBD_mask_file is not None:
        VBD_mask = xr.open_dataset(VBD_mask_file).aridity_mask # True if 
    else: 
        print("NEED TO IMPLEMENET logic for cloud mask pull")

    T = ds_dict["tas"]
    T = _check_and_convert_units(da=T, input_var="tas", conv_type="C") 
    T = T.resample(time="1ME").mean()
    
    VSdengueAlb = T.where((T >= T_range[0]) & (T <= T_range[1]))
    VSdengueAlb = VSdengueAlb.resample(time="1YE").count()
    VSdengueAlb = VSdengueAlb.where(VBD_mask) # mask arid regions 

    # get number of time steps in a year
    steps_per_year = _get_tsteps(T)
    
    VSdengueAlb_levels = _ann_frac(VSdengueAlb, steps_per_year).rename("VSdengueAlb")
    VSdengueAlb_levels = _assign_hazard_level(VSdengueAlb_levels, frac_thresholds=hazard_thresholds)
    
    return VSdengueAlb_levels

def VbrS(ds_dict, salinity_max=28, SST_min=18, hazard_thresholds=hazard_thresholds["VbrS"]):
    """
    Vibrio bacteria suitability (coastal areas).
    Edit this calculation.
    from Trinanes et al. 2021  
    Trinanes uses a threshold of < 30km for coast, but model grid sizes exceed than and 
    salinity is unlikely to be lower than 28 psu in non-coastal areas  
    However, if a user is interested in a coast mask, it can be provided by G-CHI creators at request 
    """
    print("Calculating vibrio_suitability...")
    
    # extract sst and salinity data
    SST = ds_dict["tos"]
    SST = _check_and_convert_units(da=SST, input_var="tos", conv_type="C") 
    SSS = ds_dict["sos"] 
    SSS = _check_and_convert_units(da=SSS, input_var="sos", conv_type="psu") # convert to PSU (actually 1 ppt might == 1 psu)

    SST = SST.where(coast_mask)
    SSS = SSS.where(coast_mask)

    # get monthly means  
    SST = SST.resample(time = "1ME").mean()
    SSS = SSS.resample(time = "1ME").mean()

    # get number of time steps in a year
    steps_per_year = _get_tsteps(SST)

    VbrS = SST.where((SST >= SST_min) & (SSS < salinity_max)).resample(time="1YE").count()
    VbrS = _ann_frac(VbrS, steps_per_year).rename("VbrS")
    VbrS = _assign_hazard_level(VbrS, frac_thresholds=hazard_thresholds)

    return VbrS

# !!! start here !!!!
# !!!!!!!!!!!!!!!!!!!!!!!
# Weather

def PRXmm(ds_dict, hazard_thresholds=hazard_thresholds["PRXmm"]):
    """
    Days with precipitation > Xmm 
    """
    
    PR = ds_dict['pr']
    PR = _check_and_convert_units(da=PR, input_var="pr", conv_type="mm day-1") 
    
    # Count days > X mm
    PRXmm_levels = _annual_exceedance_frac(PR, hazard_thresholds=hazard_thresholds, var_name="PRXmm")
    PRXmm_levels = _assign_hazard_level(PRXmm_levels)
    
    return PRXmm_levels

def PR1day(ds_dict, base_dict, hazard_thresholds=hazard_thresholds["PR1day"]):
    """
    Days rainfall exceeds Xth percentile of rainfall
    A minimum of 1 mm to count 
    """
    
    PR = ds_dict['pr']
    PR = _check_and_convert_units(da=PR, input_var="pr", conv_type="mm day-1")
    
    # get number of time steps in a year
    steps_per_year = _get_tsteps(PR)
    PR = PR.where(PR > 1) # minimum of 1 mm to be a rainy day

    pr_base_percentile_vals = [float(k.split("_")[1].replace("pt", ".").replace("p", "")) for k in base_dict.keys() if k.startswith('pr_') and k.endswith('p')]
    pr_base_percentile_keys = sorted(
        [k for k in base_dict.keys() if k.startswith('pr_') and k.endswith('p')],
        key=lambda k: float(k.replace('pr_', '').replace('pt', '.').replace('p', '')),
        reverse=True
        )
    if sorted(pr_base_percentile_vals) != sorted(hazard_thresholds):
        print(f"Cannot calculate PR1day because based period pr percentiles do not match hazard_thresholds\n base period percentiles: {pr_base_percentile_vals}. hazard_thresholds: {hazard_thresholds}. \nSkipping...")
        return None
    
    # thresholds are lat/lon dependent so use a bespoke exceedance counting method
    da_list = []
    for _, key in enumerate(pr_base_percentile_keys):
        th = base_dict[key]
        da_count = (PR > th).resample(time='1YE').sum(dim='time', skipna=True)
        da_list.append(da_count)
    
    # Concatenate along new 'level' dimension
    PR1day = xr.concat(da_list, dim='level')
    PR1day = PR1day.assign_coords(level=np.arange(1, len(pr_base_percentile_keys) + 1))
    PR1day.attrs['level_values'] = pr_base_percentile_keys
    PR1day = _ann_frac(PR1day, steps_per_year).rename("PR1day")
    PR1day = _assign_hazard_level(PR1day)
    PR1day.attrs['level_thresholds'] = [
        {
            "level": int(i + 1),
            "threshold_value": pr_base_percentile_vals[i],
            "unit": "percentile",
            "source": "base period distribution"
        }
        for i in range(len(pr_base_percentile_keys))
    ]

    return PR1day


def PR5day(ds_dict, base_dict, hazard_thresholds=hazard_thresholds["PR5day"]):
    """
    Number of occurances of 5-day precipitation > Xth percentile annual max 5-day pr (default).
    """
    # maybe percentiles grid cell based and then the number of days in a period where 5-day rainfall > historical percentile

    PR = ds_dict['pr']
    PR = _check_and_convert_units(da=PR, input_var="pr", conv_type="mm day-1")
    # get number of time steps in a year
    steps_per_year = _get_tsteps(PR)

    rx5day_base_percentile_vals = [float(k.split("_")[1].replace("pt", ".").replace("p", "")) for k in base_dict.keys() if k.startswith('rx5day_') and k.endswith('p')]
    rx5day_base_percentile_keys = sorted(
        [k for k in base_dict.keys() if k.startswith('rx5day_') and k.endswith('p')],
        key=lambda k: float(k.replace('rx5day_', '').replace('pt', '.').replace('p', '')),
        reverse=True
        )
    if sorted(rx5day_base_percentile_vals) != sorted(hazard_thresholds):
        print(f"Cannot calculate PR5day because based period rx5day percentiles do not match hazard_thresholds\n base period percentiles: {rx5day_base_percentile_vals}. hazard_thresholds: {hazard_thresholds}. \nSkipping...")
        return None
    
    # thresholds are lat/lon dependent so use a bespoke exceedance counting method
    # apply rolling sum and get all days that are a part of 5-day precip > X mm 
    window = 5
    half_window = window // 2
    rolling_sum = PR.rolling(time=window, center=True).sum()

    da_list = []
    for _, key in enumerate(rx5day_base_percentile_keys):
        th = base_dict[key]

        # identify 5-day centers exceeding threshold
        mask_center = rolling_sum > th  
        # make ±2 days True without overwriting existing Trues (all days in 5 period count in wet spell)
        mask_expanded = xr.zeros_like(PR, dtype=bool)  # all False initially
        for shift in range(-half_window, half_window + 1):
            mask_expanded |= mask_center.shift(time=shift, fill_value=False)
        # count annual total days involved in 5-day precip event > X mm
        da_count = PR.where(mask_expanded).resample(time="1YE").count()

        da_list.append(da_count)
    
    # Concatenate along new 'level' dimension
    PR5day = xr.concat(da_list, dim='level')
    PR5day = PR5day.assign_coords(level=np.arange(1, len(rx5day_base_percentile_keys) + 1))
    PR5day.attrs['level_values'] = rx5day_base_percentile_keys
    PR5day = _ann_frac(PR5day, steps_per_year).rename("PR5day")
    PR5day = _assign_hazard_level(PR5day)
    PR5day.attrs['level_thresholds'] = [
        {
            "level": int(i + 1),
            "threshold_value": rx5day_base_percentile_vals[i],
            "unit": "percentile",
            "source": "base period distribution"
        }
        for i in range(len(rx5day_base_percentile_keys))
    ]

    return PR5day


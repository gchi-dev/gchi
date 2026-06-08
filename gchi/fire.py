"""
fire danger metrics: FI, HDW, FWI

note on chunking:
- FI and HDW: fast chunked
- FWI: does NOT work chunked — uses a time-step loop and will load data automatically
"""

import numpy as np
import xarray as xr

from ._core import (
    _check_and_convert_units, _annual_exceedance_frac, _annual_exceedance_frac_fwi,
    _assign_severity_level, _tetens_sat_vapor_pressure, _add_metric_metadata,
)
from .thresholds import severity_thresholds as _default_thresholds, fwi_thresholds


def _fmi_values(ds_dict):
    """fuel moisture index (Sharples et al. 2009)"""
    T = _check_and_convert_units(da=ds_dict["tas"], input_var="tas", conv_type="C")
    RH = _check_and_convert_units(da=ds_dict["hurs"], input_var="hurs", conv_type="%")
    RH = RH.clip(0.1, 99.9999)
    return 10 - 0.25 * (T - RH)


def fi_values(ds_dict, fire_mask_file=None):
    """
    Fire danger index values (Sharples et al. 2009).
    https://doi.org/10.1016/j.envsoft.2008.10.012
    """

    if fire_mask_file is not None:
        fwi_mask = xr.open_dataset(fire_mask_file).mask_infreq_burning
    else:
        print("no FWI mask file provided — proceeding without infrequent burning mask")
        fwi_mask = False  # no masking

    U = _check_and_convert_units(da=ds_dict["sfcWind"], input_var="sfcWind", conv_type="km h-1")
    FMI = _fmi_values(ds_dict)
    return ((U.where(U > 1, 1)) / FMI).where(~fwi_mask)


def _load_fire_mask(fire_mask_file):
    """load infrequent burning mask — True where burning is infrequent (mask out)"""
    if fire_mask_file is not None:
        return xr.open_dataset(fire_mask_file).mask_infreq_burning
    return None


def FI(ds_dict, severity_thresholds=None, fire_mask_file=None):
    """
    Fire danger index exceedance levels.
    Call fi_values() to get raw FI without level assignment.

    Parameters
    ----------
    fire_mask_file : str, optional
        path to infrequent burning mask file. cells where burning is infrequent
        are masked out before exceedance counting.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["FI"]
    FI_val = fi_values(ds_dict, fire_mask_file=fire_mask_file)
    fire_mask = _load_fire_mask(fire_mask_file)
    if fire_mask is not None:
        FI_val = FI_val.where(~fire_mask)
    FI_levels = _annual_exceedance_frac(FI_val, severity_thresholds=severity_thresholds, var_name="FI")
    result = _assign_severity_level(FI_levels)
    return _add_metric_metadata(result, "FI", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"Sharples et al. 2009 fire danger index. fire_mask_file={fire_mask_file}")


def hdw_values(ds_dict, fire_mask_file=None):
    """
    Hot-Dry-Windy index values (Srock et al. 2018).
    https://doi.org/10.3390/atmos9070279
    """
    if fire_mask_file is not None:
        fwi_mask = xr.open_dataset(fire_mask_file).mask_infreq_burning
    else:
        print("no FWI mask file provided — proceeding without infrequent burning mask")
        fwi_mask = False  # no masking

    T = _check_and_convert_units(da=ds_dict["tas"], input_var="tas", conv_type="C")
    RH = _check_and_convert_units(da=ds_dict["hurs"], input_var="hurs", conv_type="fraction")
    RH = RH.clip(0.001, 0.999999)
    U = _check_and_convert_units(da=ds_dict["sfcWind"], input_var="sfcWind", conv_type="m s-1")

    es = _tetens_sat_vapor_pressure(T)
    VPD = (1 - RH) * es
    return (U * VPD).where(~fwi_mask)


def HDW(ds_dict, severity_thresholds=None, fire_mask_file=None):
    """
    HDW exceedance levels.
    Call hdw_values() to get raw HDW without level assignment.

    Parameters
    ----------
    fire_mask_file : str, optional
        path to infrequent burning mask file. cells where burning is infrequent
        are masked out before exceedance counting.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["HDW"]
    HDW_val = hdw_values(ds_dict, fire_mask_file=fire_mask_file)
    fire_mask = _load_fire_mask(fire_mask_file)
    if fire_mask is not None:
        HDW_val = HDW_val.where(~fire_mask)
    HDW_levels = _annual_exceedance_frac(HDW_val, severity_thresholds=severity_thresholds, var_name="HDW")
    result = _assign_severity_level(HDW_levels)
    return _add_metric_metadata(result, "HDW", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"Srock et al. 2018 hot-dry-windy index. fire_mask_file={fire_mask_file}")


# =================
# Canadian FWI
# =================
# Based on Quilcaille et al., 2023 (https://doi.org/10.5194/essd-15-2153-2023)
# FWI does NOT chunk well — uses a time-step loop, data is loaded automatically.
# This is a known limitation; no chunking approach avoids this for sequential indices.

def _get_day_length_factor(lat, month):
    """day length factor — latitude and month dependent"""
    day_length_table = np.array([
        [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0],   # >= 20°N
        [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8],      # >= 40°N
        [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],        # equator
        [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2],      # <= -20°S
        [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8],   # <= -40°S
    ])

    day_length = xr.zeros_like(lat, dtype=float)
    month_idx = month - 1 if isinstance(month, (int, np.integer)) else month.values - 1

    day_length = xr.where(lat >= 20, day_length_table[0, month_idx], day_length)
    day_length = xr.where(lat >= 40, day_length_table[1, month_idx], day_length)
    day_length = xr.where((lat > -20) & (lat < 20), day_length_table[2, month_idx], day_length)
    day_length = xr.where(lat <= -20, day_length_table[3, month_idx], day_length)
    day_length = xr.where(lat <= -40, day_length_table[4, month_idx], day_length)

    return day_length


def _apply_overwintering(dc, month, lat):
    """apply overwintering wetting factor to DC in winter months"""
    winter_north = ((month >= 11) | (month <= 3)) & (lat >= 30)
    winter_south = ((month >= 5) & (month <= 9)) & (lat <= -30)
    return xr.where(winter_north | winter_south, dc * 0.75, dc)


def _ffmc_step(temp, rh, wind, rain, ffmc_prev):
    """fine fuel moisture code — single time step"""
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

    rain_effect = (
        xr.where(rain > 1.5,
            42.5 * rain * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rain)) +
            0.0015 * (mo - 150.0)**2 * np.sqrt(rain),
        xr.where(rain > 0.5,
            42.5 * rain * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rain)),
            0))
    )
    rf = (mo + rain_effect).clip(0, 250)

    ed = 0.942 * rh**0.679 + 11.0 * np.exp((rh - 100.0) / 10.0) + \
         0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    ew = 0.618 * rh**0.753 + 10.0 * np.exp((rh - 100.0) / 10.0) + \
         0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))

    ko = 0.424 * (1.0 - (rh / 100.0)**1.7) + 0.0694 * np.sqrt(wind) * (1.0 - (rh / 100.0)**8)
    kd = ko * 0.581 * np.exp(0.0365 * temp)
    kw = ko * 0.581 * np.exp(0.0365 * temp)

    m = xr.where(rf > ed,
                 ew + (rf - ew) * 10.0**(-kd),
                 xr.where(rf < ew,
                          ed - (ed - rf) * 10.0**(-kw),
                          rf))

    return (59.5 * (250.0 - m) / (147.2 + m)).clip(0, 101)


def _dmc_step(temp, rh, rain, dmc_prev, day_length):
    """duff moisture code — single time step"""
    re = xr.where(rain > 1.5, 0.92 * rain - 1.27, 0.0)
    mo = 20.0 + np.exp(5.6348 - dmc_prev / 43.43)

    b = xr.where(dmc_prev <= 33,
                 100.0 / (0.5 + 0.3 * dmc_prev),
                 xr.where(dmc_prev <= 65,
                           14.0 - 1.3 * np.log(dmc_prev),
                           6.2 * np.log(dmc_prev) - 17.2))

    mr = mo + 1000.0 * re / (48.77 + b * re)
    pr = (244.72 - 43.43 * np.log(mr - 20.0)).clip(min=0)  
    k = 1.894 * (temp.clip(min=-1.1) + 1.1) * (100.0 - rh) * day_length * 1e-4

    return xr.where(rain > 1.5, pr + k, dmc_prev + k).clip(min=0)


def _dc_step(temp, rain, dc_prev, day_length, month, lat):
    """drought code — single time step"""
    dc_prev = _apply_overwintering(dc_prev, month, lat)

    rd = xr.where(rain > 2.8, 0.83 * rain - 1.27, 0.0)
    qo = 800.0 * np.exp(-dc_prev / 400.0)
    qr = (qo + 3.937 * rd)
    dr = (400.0 * np.log(800.0 / qr)).clip(min=0)
    v = 0.36 * (temp.clip(min=-2.8) + 2.8) + day_length

    return xr.where(rain > 2.8, dr + v, dc_prev + v).clip(min=0)


def _isi_from_ffmc(wind, ffmc):
    """initial spread index from FFMC and wind"""
    mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * np.exp(-0.1386 * mo) * (1.0 + mo**5.31 / 4.93e7)
    fw = np.exp(0.05039 * wind)
    return 0.208 * fw * ff


def _bui_from_codes(dmc, dc):
    """buildup index from DMC and DC"""
    return xr.where(
        dmc <= 0.4 * dc,
        0.8 * dmc * dc / (dmc + 0.4 * dc),
        dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc)**1.7),
    ).clip(min=0)


def _fwi_from_isi_bui(isi, bui):
    """fire weather index from ISI and BUI"""
    bb = xr.where(
        bui > 80,
        0.1 * isi * (1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui))),
        0.1 * isi * (0.626 * bui**0.809 + 2.0),
    )
    return xr.where(bb <= 1, bb, np.exp(2.72 * (0.434 * np.log(bb))**0.647))


def fwi_values(ds_dict, use_hursmin=True, init_values=None, fwi_mask_file=None, spatial_chunk=20):
    """
    Daily FWI index values.
    Does NOT work chunked — data will be chunked only along spatial dims. This is a known limitation
    of the sequential FWI algorithm.

    Parameters
    ----------
    ds_dict : dict
    use_hursmin : bool
        use hursmin over hurs if both available
    init_values : dict, optional
        initial values {'ffmc': 85, 'dmc': 6, 'dc': 15}
    fwi_mask_file : str, optional
        path to infrequent burning mask file

    Returns
    -------
    xr.DataArray of daily FWI values
    """
    print("calculating FWI...")

    # FWI does not work chunked — load if chunked
    for key in ["tasmax", "pr", "sfcWind", "hursmin", "hurs"]:
        ds_dict[key] = ds_dict[key].chunk({"time": -1, "lat": spatial_chunk, "lon": spatial_chunk})  # chunk entire time dim because sequential build up of drying 
        #if key in ds_dict and ds_dict[key].chunks is not None:
            #ds_dict[key] = ds_dict[key].load() # this blows up ram, only use if manageable amount of data 

    if fwi_mask_file is not None:
        fwi_mask = xr.open_dataset(fwi_mask_file).mask_infreq_burning
    else:
        print("no FWI mask file provided — proceeding without infrequent burning mask")
        fwi_mask = False  # no masking

    TX = _check_and_convert_units(da=ds_dict["tasmax"], input_var="tasmax", conv_type="C").where(~fwi_mask)
    precip = _check_and_convert_units(da=ds_dict["pr"], input_var="pr", conv_type="mm day-1").where(~fwi_mask)
    wind = _check_and_convert_units(da=ds_dict["sfcWind"], input_var="sfcWind", conv_type="km h-1").where(~fwi_mask)

    if use_hursmin and 'hursmin' in ds_dict:
        print("using hursmin")
        rh = _check_and_convert_units(da=ds_dict['hursmin'], input_var="hursmin", conv_type="%").where(~fwi_mask)
    else:
        print("using hurs")
        rh = _check_and_convert_units(da=ds_dict['hurs'], input_var="hurs", conv_type="%").where(~fwi_mask)
    rh = rh.clip(0, 100)

    lat = TX.lat
    time = TX.time
    if hasattr(time, 'dt'):
        month = time.dt.month
    else:
        month = xr.DataArray(
            [(i // 30) % 12 + 1 for i in range(len(time))],
            dims=['time'], coords={'time': time},
        )

    day_length = xr.concat(
        [_get_day_length_factor(lat, m) for m in month.values],
        dim='time',
    )
    day_length['time'] = time

    if init_values is None:
        init_values = {'ffmc': 85.0, 'dmc': 6.0, 'dc': 15.0}

    template = TX * 0
    ffmc = template.copy()
    dmc = template.copy()
    dc = template.copy()
    isi = template.copy()
    bui = template.copy()
    fwi = template.copy()

    n_times = len(TX.time)
    print(f"processing {n_times} time steps...")

    for i in range(n_times):
        if i % 365 == 0:
            print(f"  day {i}/{n_times}")

        t = TX.isel(time=i)
        r = rh.isel(time=i)
        w = wind.isel(time=i)
        p = precip.isel(time=i)
        dl = day_length.isel(time=i)
        m = month.isel(time=i)

        ffmc_prev = init_values['ffmc'] if i == 0 else ffmc.isel(time=i - 1)
        dmc_prev = init_values['dmc'] if i == 0 else dmc.isel(time=i - 1)
        dc_prev = init_values['dc'] if i == 0 else dc.isel(time=i - 1)

        ffmc[dict(time=i)] = _ffmc_step(t, r, w, p, ffmc_prev)
        dmc[dict(time=i)] = _dmc_step(t, r, p, dmc_prev, dl)
        dc[dict(time=i)] = _dc_step(t, p, dc_prev, dl, m, lat)
        isi[dict(time=i)] = _isi_from_ffmc(w, ffmc.isel(time=i))
        bui[dict(time=i)] = _bui_from_codes(dmc.isel(time=i), dc.isel(time=i))
        fwi[dict(time=i)] = _fwi_from_isi_bui(isi.isel(time=i), bui.isel(time=i))

    return fwi


def FWI(ds_dict, use_hursmin=True, init_values=None,
        fwi_mask_file=None, environmental_zone_file=None):
    """
    Canadian Fire Weather Index exceedance levels with spatially-varying thresholds.

    FWI does NOT work chunked — data is loaded automatically.
    Call fwi_values() to get raw daily FWI without level assignment.

    Parameters
    ----------
    ds_dict : dict
    use_hursmin : bool
    init_values : dict, optional
    fwi_mask_file : str, optional
        path to infrequent burning mask
    environmental_zone_file : str, optional
        path to environmental zone file (required for spatially-varying thresholds)
    """
    fwi = fwi_values(ds_dict, use_hursmin=use_hursmin,
                     init_values=init_values, fwi_mask_file=fwi_mask_file)

    if environmental_zone_file is None:
        print("no environmental zone file provided — cannot assign spatially-varying FWI levels. returning raw FWI.")
        return fwi

    environmental_zones = xr.open_dataset(environmental_zone_file).environmental_zone

    FWI_levels = _annual_exceedance_frac_fwi(fwi, environmental_zones, fwi_thresholds)
    result = _assign_severity_level(FWI_levels)
    return _add_metric_metadata(result, "FWI", ds_dict, units="fraction of year", notes="Canadian FWI. spatially varying thresholds from Kudlackova et al. 2025 environmental zones.")

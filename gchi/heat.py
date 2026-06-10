"""
heat stress metrics: AT, HI, Hu, WBT, WBGT, UTCIhot, HWF, TXC, TR
"""

import numpy as np
import xarray as xr
import pandas as pd
from . import newt

from ._core import (
    _check_and_convert_units, _annual_exceedance_frac, _assign_severity_level,
    _get_tsteps, _ann_frac, _tetens_sat_vapor_pressure, _nan_mask, _add_metric_metadata,
)
from .thresholds import severity_thresholds as _default_thresholds


# =================
# intermediate calculations (private)
# these are separated so they can be computed once and reused across metrics
# =================

def _wbt_values(ds_dict):
    """
    WBT calculation via NEWT (Rogers & Warren 2024).
    Faster and more accurate than Davies-Jones 2008 & Stull 2011.
    https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4866

    Returns Twp in °C
    """
    p = _check_and_convert_units(da=ds_dict["ps"], input_var="ps", conv_type="Pa")
    TX = _check_and_convert_units(da=ds_dict["tasmax"], input_var="tasmax", conv_type="K")

    if "huss" in ds_dict:
        q = _check_and_convert_units(da=ds_dict['huss'], input_var="huss", conv_type="fraction")
    else:
        # derive q from RH using newt's saturation_specific_humidity
        RH = _check_and_convert_units(da=ds_dict['hurs'], input_var="hurs", conv_type="fraction")
        RH = RH.clip(0.001, 0.999999)
        qs = newt.saturation_specific_humidity(p.values, TX.values)
        q = RH * qs

    q_vals = q.values if hasattr(q, 'values') else q
    Twp_vals = newt.pseudo_wet_bulb_temperature(p.values, TX.values, q_vals)
    Twp = TX.copy(data=Twp_vals)  # wrap back into DataArray (K)
    return Twp - 273.15  # K to °C


def _scale_windspeed(va, h):
    """
    Scale wind speed from 10m to height h.
    From Bröde et al. (2012) / thermofeel.
    """
    c = 1 / np.log10(10 / 0.01)
    return va * np.log10(h / 0.01) * c


def _cos_solar_zenith_angle_daily(time, lat):
    """
    Daytime-mean cosine of solar zenith angle, integrated over daylight hours.
    Based on Di Napoli et al. 2020 eq. 12
    """
    times = pd.DatetimeIndex(time.values)
    JD = xr.DataArray(times.dayofyear.values, dims=['time'], coords={'time': time})
    g_rad = np.radians((360.0 / 365.25) * JD)

    delta = (
        0.006918
        - 0.399912 * np.cos(g_rad)
        + 0.070257 * np.sin(g_rad)
        - 0.006758 * np.cos(2 * g_rad)
        + 0.000907 * np.sin(2 * g_rad)
        - 0.002697 * np.cos(3 * g_rad)
        + 0.001480 * np.sin(3 * g_rad)
    )

    phi = np.radians(lat)
    cos_h0 = (-np.tan(delta) * np.tan(phi)).clip(-1.0, 1.0)
    h0 = np.arccos(cos_h0)
    safe_h0 = h0.where(h0 > 1e-6, other=1e-6)

    cossza = (
        np.sin(delta) * np.sin(phi)
        + np.cos(delta) * np.cos(phi) * np.sin(h0) / safe_h0
    )
    return cossza.where(h0 > 1e-6, other=0.0).clip(0.0, 1.0)


def _calculate_mrt(ds_dict):
    """
    Mean Radiant Temperature (MRT) in K.
    Adapted from ECMWF thermofeel / Di Napoli et al. 2020.
    Required variables: rsdsdiff, rsus, rlus, rsdscs, rsdscsdiff
    """
    to_radians = np.pi / 180

    dsrp = ds_dict['rsdscs'] - ds_dict['rsdscsdiff']
    cossza = _cos_solar_zenith_angle_daily(ds_dict['rsdscs'].time, ds_dict['rsdscs'].lat)

    dsw = ds_dict['rsdsdiff']
    rsw = ds_dict['rsus']
    lur = ds_dict['rlus']

    gamma = np.arcsin(cossza) * 180 / np.pi
    fp = 0.308 * np.cos(to_radians * gamma * (0.998 - gamma * gamma / 50000))

    mrt = np.power(
        (1 / 0.0000000567) * (
            0.5 * lur  # strd approximation via rlus
            + 0.5 * lur
            + (0.7 / 0.97) * (0.5 * dsw + 0.5 * rsw + fp * dsrp)
        ),
        0.25,
    )
    return mrt  # K


def _calculate_bgt(ds_dict, mrt):
    """
    Globe temperature (K).
    From Guo et al. 2018 / thermofeel.
    """
    t2_k = _check_and_convert_units(da=ds_dict['tasmax'], input_var="tasmax", conv_type="K")
    va = _check_and_convert_units(da=ds_dict['sfcWind'], input_var="sfcWind", conv_type="m s-1")
    v = _scale_windspeed(va, 1.1)

    d = (1.1e8 * v ** 0.6) / (0.95 * 0.15 ** 0.4)
    e = -(mrt ** 4) - d * t2_k

    q = 12 * e
    s = 27 * (d ** 2)
    delta = ((s + np.sqrt(s ** 2 - 4 * (q ** 3))) / 2) ** (1 / 3)
    Q = 0.5 * np.sqrt((1 / 3) * (delta + q / delta))

    return -Q + 0.5 * np.sqrt(-4 * (Q ** 2) + d / Q)  # K


_MRT_VARS = {'rsdsdiff', 'rsus', 'rlus', 'rsdscs', 'rsdscsdiff'}


def _has_mrt_vars(ds_dict):
    return all(k in ds_dict for k in _MRT_VARS)


def _sat_vapor_pressure_its90(ta_celsius):
    """
    Saturation vapor pressure (hPa) via Hardy 1998 / ITS-90.
    Used in UTCI. Translated from Bröde Fortran 2009.
    """
    tk = ta_celsius + 273.15
    g = np.array([-2.8365744e3, -6.028076559e3, 1.954263612e1, -2.737830188e-2,
                  1.6261698e-5, 7.0229056e-10, -1.8680009e-13, 2.7150305])
    es = g[7] * np.log(tk)
    for i in range(7):
        es = es + g[i] * tk ** (i - 2)
    return np.exp(es) * 0.01  # Pa to hPa


def _utci_polynomial(Ta, va, D_Tmrt, Pa):
    """
    6th order polynomial approximation for UTCI.
    Translated from Bröde Fortran 2009.
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


def _utci_values(ds_dict, hum_var='both', hotorcold='hot'):
    """
    Calculate UTCI index values (°C) — no exceedance or level assignment.

    Parameters
    ----------
    ds_dict : dict
    hum_var : str
        'both', 'huss', or 'hurs'
    hotorcold : str
        'hot' uses tasmax, 'cold' uses tasmin

    Returns
    -------
    xr.DataArray of UTCI values, masked to valid input ranges
    """
    tas_var = "tasmax" if hotorcold.lower() == "hot" else "tasmin"

    TXN = _check_and_convert_units(da=ds_dict[tas_var], input_var=tas_var, conv_type="C")
    TA = _check_and_convert_units(da=ds_dict['tas'], input_var="tas", conv_type="C")
    RH = _check_and_convert_units(da=ds_dict['hurs'], input_var="hurs", conv_type="fraction")
    RH = RH.clip(0.001, 0.999999)
    ps = _check_and_convert_units(da=ds_dict['ps'], input_var="ps", conv_type="Pa")

    if 'sfcWind' in ds_dict:
        va = _check_and_convert_units(da=ds_dict['sfcWind'], input_var="sfcWind", conv_type="m s-1")
    else:
        va = np.sqrt(ds_dict['uas'] ** 2 + ds_dict['vas'] ** 2)
    va = va.clip(0.5, 17.0)

    if hum_var in ('both', 'hurs'):
        if _has_mrt_vars(ds_dict):
            es_hpa = _sat_vapor_pressure_its90(TA) * RH
        else:
            es_hpa = _sat_vapor_pressure_its90(TXN) * RH
    else:
        p_hpa = ps * 0.01
        r = ds_dict['huss'] / (1 - ds_dict['huss'])
        es_hpa = r * p_hpa / (0.622 + r)

    es_hpa = es_hpa.clip(0.0, 20.0)
    Pa = es_hpa / 10.0

    if _has_mrt_vars(ds_dict):
        tmrt = _calculate_mrt(ds_dict) - 273.15
        D_Tmrt = (tmrt - TA).clip(-30, 70)
        UTCI = _utci_polynomial(TA, va, D_Tmrt, Pa)
        valid = (TA >= -50) & (TA <= 50)
    else:
        D_Tmrt = 0
        UTCI = _utci_polynomial(TXN, va, D_Tmrt, Pa)
        valid = (TXN >= -50) & (TXN <= 50)

    return UTCI.where(valid)


# =================
# public metric functions
# each has a _values() variant that returns the raw metric without level assignment
# =================

def at_values(ds_dict):
    """
    Apparent temperature (°C) — 'feels like' temperature.
    Combines air temp, vapor pressure. From Zhao et al. 2015.
    """
    TX = _check_and_convert_units(da=ds_dict['tasmax'], input_var="tasmax", conv_type="C")
    RH = _check_and_convert_units(da=ds_dict['hurs'], input_var="hurs", conv_type="fraction")
    RH = RH.clip(0.001, 0.999999)

    es = _tetens_sat_vapor_pressure(TX)
    VP = es * RH
    return (0.92 * TX) + (0.22 * VP) - 1.3


def AT(ds_dict, severity_thresholds=None):
    """
    Apparent temperature exceedance levels.
    Calls at_values() then computes annual exceedance fraction and severity levels.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["AT"]
    AT_val = at_values(ds_dict)
    AT_levels = _annual_exceedance_frac(AT_val, severity_thresholds, var_name="AT")
    result = _assign_severity_level(AT_levels)
    return _add_metric_metadata(result, "AT", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year")


def hi_values(ds_dict):
    """
    NOAA heat index (°C).
    https://www.wpc.ncep.noaa.gov/html/heatindex.shtml
    """
    TX = _check_and_convert_units(da=ds_dict['tasmax'], input_var="tasmax", conv_type="F")
    RH = _check_and_convert_units(da=ds_dict['hurs'], input_var="hurs", conv_type="%")
    RH = RH.clip(0.1, 99.9999)

    c0 = -42.379
    c1 = 2.04901523
    c2 = 10.14333127
    c3 = -0.22475541
    c4 = -0.00683783
    c5 = -0.05481717
    c6 = 0.00122874
    c7 = 0.00085282
    c8 = -0.00000199

    HI1 = (c0 + c1*TX + c2*RH + c3*TX*RH + c4*TX**2 + c5*RH**2
           + c6*TX**2*RH + c7*TX*RH**2 + c8*TX**2*RH**2)

    HI_A = HI1 - ((13 - RH)/4) * np.sqrt(17 - np.abs(TX - 95)/17)
    mask_A = (TX > 80) & (TX < 112) & (RH < 13)

    HI_B = HI1 + ((RH - 85)/10) * (87 - TX)/5
    mask_B = ~mask_A & (TX > 80) & (TX < 87) & (RH > 85)

    mask_C = ~mask_A & ~mask_B & (TX > 80)
    mask_D = TX < 80

    HI_D_val = 0.5 * (TX + 61 + 1.2*(TX - 68) + 0.094*RH)

    HI = xr.where(mask_A, HI_A,
         xr.where(mask_B, HI_B,
         xr.where(mask_C, HI1,
         xr.where(mask_D, HI_D_val, TX))))

    return (HI - 32) * (5/9)  # F to C


def HI(ds_dict, severity_thresholds=None):
    """NOAA heat index exceedance levels."""
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["HI"]
    HI_val = hi_values(ds_dict)
    HI_levels = _annual_exceedance_frac(HI_val, severity_thresholds, var_name="HI")
    result = _assign_severity_level(HI_levels)
    return _add_metric_metadata(result, "HI", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year")


def hu_values(ds_dict):
    """
    Humidex (°C) — Canadian humidity index.
    https://publications.gc.ca/site/eng/9.865813/publication.html
    """
    TX = _check_and_convert_units(da=ds_dict['tasmax'], input_var="tasmax", conv_type="C")
    RH = _check_and_convert_units(da=ds_dict['hurs'], input_var="hurs", conv_type="fraction")
    RH = RH.clip(0.001, 0.999999)

    es = _tetens_sat_vapor_pressure(TX)
    e = RH * es
    h = (5/9) * (e*10 - 10) # convert kPa to hPa before applying formula
    return TX + h


def Hu(ds_dict, severity_thresholds=None):
    """Humidex exceedance levels."""
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["Hu"]
    Hu_val = hu_values(ds_dict)
    Hu_levels = _annual_exceedance_frac(Hu_val, severity_thresholds, var_name="Hu")
    result = _assign_severity_level(Hu_levels)
    return _add_metric_metadata(result, "Hu", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year")


def WBT(ds_dict, severity_thresholds=None, Twb=None):
    """
    Wet Bulb Temperature exceedance levels via NEWT (Rogers & Warren 2024).
    Call wbt_values() to get raw WBT without level assignment.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["WBT"]
    if Twb is None: # if you do not want to re-calculate WBT for WBGT, you should calculate WBT from wbt_values beforehand, then pass as arg 
        Twb = _wbt_values(ds_dict)
    WBT_levels = _annual_exceedance_frac(Twb, severity_thresholds, var_name="WBT")
    result = _assign_severity_level(WBT_levels)
    return _add_metric_metadata(result, "WBT", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes="WBT via NEWT pseudo wet-bulb temperature (Rogers & Warren 2024)")


def wbt_values(ds_dict):
    """Raw WBT values (°C). Wrapper around internal _wbt_values."""
    return _wbt_values(ds_dict)


def wbgt_values(ds_dict, Twb=None):
    """
    WBGT (°C).
    Default: Brimicombe et al. 2023 approach using NEWT for WBT.
    Fallback: Schwingshackl et al. 2021 approximation if MRT vars not available.
    """
    if Twb is None:
        Twb = _wbt_values(ds_dict)
    TX = _check_and_convert_units(da=ds_dict['tasmax'], input_var="tasmax", conv_type="C")

    if _has_mrt_vars(ds_dict):
        TA = _check_and_convert_units(da=ds_dict['tas'], input_var="tas", conv_type="C")
        tmrt = _calculate_mrt(ds_dict)
        bgt = _calculate_bgt(ds_dict=ds_dict, mrt=tmrt) - 273.15
        return 0.7 * Twb + 0.2 * bgt + 0.1 * TA
    else:
        # assumes MRT == air temp (reference condition)
        return 0.7 * Twb + 0.3 * TX


def WBGT(ds_dict, severity_thresholds=None, Twb=None):
    """WBGT exceedance levels. Call wbgt_values() to get raw WBGT."""
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["WBGT"]
    WBGT_val = wbgt_values(ds_dict, Twb=Twb)
    WBGT_levels = _annual_exceedance_frac(WBGT_val, severity_thresholds, var_name="WBGT")
    result = _assign_severity_level(WBGT_levels)
    return _add_metric_metadata(result, "WBGT", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes="if MRT vars not in ds_dict, falls back to Schwingshackl 2021 approximation using tasmax as dry bulb")


def utci_hot_values(ds_dict, hum_var='both'):
    """Raw UTCIhot values (°C)."""
    return _utci_values(ds_dict, hum_var=hum_var, hotorcold='hot')


def UTCIhot(ds_dict, hum_var='both', severity_thresholds=None):
    """
    UTCI heat stress exceedance levels.
    Call utci_hot_values() to get raw UTCI without level assignment.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["UTCIhot"]
    UTCI = _utci_values(ds_dict, hum_var=hum_var, hotorcold='hot')
    UTCI_levels = _annual_exceedance_frac(UTCI, severity_thresholds=severity_thresholds, var_name="UTCIhot")
    result = _assign_severity_level(UTCI_levels)
    return _add_metric_metadata(result, "UTCIhot", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year")


def HWF(ds_dict, base_dict, percentile_base=90,
        severity_thresholds=None, hwd_threshold=3, detrend=True):
    """
    Heatwave Frequency — fraction of year where days are part of a heatwave.
    A heatwave is >= hwd_threshold consecutive days where:
        - daily mean T > calendar-day Xth percentile (90th default)
        - daily mean T > base period annual mean

    Parameters
    ----------
    ds_dict : dict
    base_dict : dict
        Output from calculate_base_period_percentiles() — needs 'tas' and 't{percentile_base}p_calday'.
    percentile_base : int
        Percentile threshold (default 90). Must match a key in base_dict.
    severity_thresholds : list, optional
    hwd_threshold : int
        Minimum consecutive days for a heatwave (default 3).
    detrend : bool
        If True, remove linear warming trend before heatwave detection.
        Useful for future projections — prevents long-term warming from inflating counts.
        Warn users against detrending if data span is very short.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["HWF"]

    varname = f"t{str(percentile_base)}p_calday"
    TXp_calday = _check_and_convert_units(da=base_dict[varname], input_var=varname, conv_type="C").chunk({"lat": -1, "lon": -1})

    T = _check_and_convert_units(da=ds_dict['tas'], input_var="tas", conv_type="C")
    steps_per_year = _get_tsteps(T)

    if detrend:
        def _detrend(arr):
            t = np.arange(arr.shape[0])
            slope, intercept = np.polyfit(t, arr, 1)
            # subtract trend relative to t=0, so early years unchanged
            # late years get progressively cooler (if trend positive warming)
            return arr - slope * t

        T = xr.apply_ufunc(
            _detrend,
            T.chunk({"lat": -1, "lon": -1, "time": -1}).compute(),
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            output_dtypes=[T.dtype],
        )

    T_base_avg = _check_and_convert_units(da=base_dict["tas"].mean("time"), input_var="tas", conv_type="C").chunk({"lat": -1, "lon": -1})
    T_warm = T.where(T > T_base_avg).chunk({"time": -1})

    T_anom = T_warm.groupby("time.dayofyear") - TXp_calday
    heat_mask = (T_anom > 0).chunk({"time": -1})

    rolling_sum = T_anom.where(heat_mask).compute().rolling(time=hwd_threshold, center=False).count()
    window_all_hot = rolling_sum == hwd_threshold

    mask_expanded = xr.zeros_like(heat_mask, dtype=bool)
    for shift in range(hwd_threshold):
        mask_expanded |= window_all_hot.shift(time=shift, fill_value=False)

    hwf_counts = T.where(mask_expanded).resample(time="1YE").count()
    hwf_counts = hwf_counts.where(~_nan_mask(T))
    HWF_val = _ann_frac(hwf_counts, steps_per_year).rename("HWF")
    result = _assign_severity_level(HWF_val, frac_thresholds=severity_thresholds)
    return _add_metric_metadata(result, "HWF", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"calendar-day {percentile_base}th percentile threshold. detrend={detrend}. hwd_threshold={hwd_threshold} consecutive days")


def txc_values(ds_dict):
    """Raw daily max temperature values (°C) — for TXC exceedance."""
    return _check_and_convert_units(da=ds_dict['tasmax'], input_var="tasmax", conv_type="C")


def TXC(ds_dict, severity_thresholds=None):
    """
    Days exceeding absolute temperature thresholds (30, 35, 40, 45°C default).
    General 'hot day' metric.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["TXC"]
    TX = txc_values(ds_dict)
    TXC_levels = _annual_exceedance_frac(TX, severity_thresholds, var_name="TXC")
    result = _assign_severity_level(TXC_levels)
    return _add_metric_metadata(result, "TXC", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year")


def tr_values(ds_dict):
    """Raw daily min temperature values (°C) — for tropical nights."""
    return _check_and_convert_units(da=ds_dict['tasmin'], input_var="tasmin", conv_type="C")


def TR(ds_dict, TR_thresh=20, severity_thresholds=None):
    """
    Tropical nights — fraction of year where daily min T > TR_thresh (20°C default).
    Associated with increased heat mortality.
    """
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["TR"]
    TN = tr_values(ds_dict)
    steps_per_year = _get_tsteps(TN)
    TR_val = TN.where(TN > TR_thresh).resample(time="1YE").count()
    TR_val = TR_val.where(~_nan_mask(TN))
    TR_val = _ann_frac(TR_val, steps_per_year).rename("TR")
    result = _assign_severity_level(TR_val, frac_thresholds=severity_thresholds)
    return _add_metric_metadata(result, "TR", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"tropical nights: tasmin > {TR_thresh} degC")

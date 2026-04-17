"""
disease suitability metrics: VSmalaria, VSzika, VSdengueAeg, VSdengueAlb, VbrS

all fast when chunked spatially.
all VBD metrics share the same structure — temperature range suitability + mask.
"""

import xarray as xr

from ._core import _check_and_convert_units, _ann_frac, _assign_hazard_level, _get_tsteps, _nan_mask
from .thresholds import hazard_thresholds as _default_thresholds


def _vbd_suitability(ds_dict, T_range, VBD_mask_file, var_name, hazard_thresholds):
    """
    Shared logic for vector-borne disease suitability.
    Counts fraction of months per year where T is within the suitable range.

    ds_dict needs 'tas'.
    mask file should contain an aridity_mask variable (True = suitable vegetation).
    """
    if VBD_mask_file is not None:
        VBD_mask = xr.open_dataset(VBD_mask_file).aridity_mask
    else:
        print(f"no VBD mask file provided for {var_name} — proceeding without aridity mask")
        VBD_mask = True

    T = _check_and_convert_units(da=ds_dict["tas"], input_var="tas", conv_type="C")
    T = T.resample(time="1ME").mean()
    steps_per_year = _get_tsteps(T)

    VS = T.where((T >= T_range[0]) & (T <= T_range[1]))
    VS = VS.resample(time="1YE").count()
    VS = VS.where(~_nan_mask(T)).where(VBD_mask)

    VS_levels = _ann_frac(VS, steps_per_year).rename(var_name)
    return _assign_hazard_level(VS_levels, frac_thresholds=hazard_thresholds)


def VSmalaria(ds_dict, T_range=[22.9, 27.8], VBD_mask_file=None, hazard_thresholds=None):
    """
    Malaria transmission suitability — fraction of year with suitable temperature.
    Temperature range from literature.
    """
    print("calculating malaria suitability...")
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["VSmalaria"]
    return _vbd_suitability(ds_dict, T_range, VBD_mask_file, "VSmalaria", hazard_thresholds)


def VSzika(ds_dict, T_range=[23.9, 34], VBD_mask_file=None, hazard_thresholds=None):
    """
    Zika transmission suitability — fraction of year with suitable temperature.
    """
    print("calculating zika suitability...")
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["VSzika"]
    return _vbd_suitability(ds_dict, T_range, VBD_mask_file, "VSzika", hazard_thresholds)


def VSdengueAeg(ds_dict, T_range=[19.9, 29.4], VBD_mask_file=None, hazard_thresholds=None):
    """
    Dengue (Aedes aegypti) transmission suitability.
    """
    print("calculating dengue aegypti suitability...")
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["VSdengueAeg"]
    return _vbd_suitability(ds_dict, T_range, VBD_mask_file, "VSdengueAeg", hazard_thresholds)


def VSdengueAlb(ds_dict, T_range=[21.3, 34], VBD_mask_file=None, hazard_thresholds=None):
    """
    Dengue (Aedes albopictus) transmission suitability.
    """
    print("calculating dengue albopictus suitability...")
    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["VSdengueAlb"]
    return _vbd_suitability(ds_dict, T_range, VBD_mask_file, "VSdengueAlb", hazard_thresholds)


def VbrS(ds_dict, salinity_max=28, SST_min=18, coast_mask_file=None, hazard_thresholds=None):
    """
    Vibrio bacteria suitability (coastal areas).
    Suitability = monthly SST >= SST_min and SSS < salinity_max.
    From Trinanes et al. 2021.

    Salinity < 28 psu is used as a coastal proxy since model grid cells are
    too coarse for a distance-to-coast mask. A proper coastal mask can be
    requested from the G-CHI team.

    Parameters
    ----------
    ds_dict : dict
        needs 'tos' and 'sos'
    salinity_max : float
        upper salinity threshold for vibrio suitability (default 28 psu)
    SST_min : float
        lower SST threshold for vibrio suitability (default 18°C)
    coast_mask_file : str, optional
        path to coastal mask file — if None, no masking is applied
    """
    print("calculating vibrio suitability...")

    SST = _check_and_convert_units(da=ds_dict["tos"], input_var="tos", conv_type="C")
    SSS = _check_and_convert_units(da=ds_dict["sos"], input_var="sos", conv_type="psu")

    if coast_mask_file is not None:
        coast_mask = xr.open_dataset(coast_mask_file).coastal_mask
        SST = SST.where(coast_mask)
        SSS = SSS.where(coast_mask)
    else:
        print("no coastal mask provided — vibrio suitability will not be masked to coastal cells")

    SST = SST.resample(time="1ME").mean()
    SSS = SSS.resample(time="1ME").mean()
    steps_per_year = _get_tsteps(SST)

    if hazard_thresholds is None:
        hazard_thresholds = _default_thresholds["VbrS"]

    VbrS_val = SST.where((SST >= SST_min) & (SSS < salinity_max)).resample(time="1YE").count()
    VbrS_val = VbrS_val.where(~_nan_mask(SST))
    VbrS_val = _ann_frac(VbrS_val, steps_per_year).rename("VbrS")
    return _assign_hazard_level(VbrS_val, frac_thresholds=hazard_thresholds)

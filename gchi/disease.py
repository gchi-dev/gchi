"""
disease suitability metrics: VSmalaria, VSzika, VSdengueAeg, VSdengueAlb, VbrS

all fast when chunked spatially.
all VBD metrics share the same structure — temperature range suitability + mask.
"""

import xarray as xr

from ._core import _check_and_convert_units, _ann_frac, _assign_severity_level, _get_tsteps, _nan_mask, _add_metric_metadata
from .thresholds import severity_thresholds as _default_thresholds
from ._log import logger


def _vbd_suitability(ds_dict, T_range, VBD_mask_file, var_name, severity_thresholds):
    """
    Shared logic for vector-borne disease suitability.
    Counts fraction of months per year where T is within the suitable range.

    ds_dict needs 'tas'.
    mask file should contain an aridity_mask variable (True = suitable vegetation).
    """
    if VBD_mask_file == "default":
        from ._remote_data import get_default_data_file
        VBD_mask_file = get_default_data_file("vbd_mask")

    if VBD_mask_file is not None:
        VBD_mask = xr.open_dataset(VBD_mask_file).aridity_mask
    else:
        logger.warning(f"no VBD mask file provided for {var_name} — proceeding without aridity mask")
        VBD_mask = True

    T = _check_and_convert_units(da=ds_dict["tas"], input_var="tas", conv_type="C")
    T = T.resample(time="1ME").mean()
    steps_per_year = _get_tsteps(T)

    VS = T.where((T >= T_range[0]) & (T <= T_range[1]))
    VS = VS.resample(time="1YE").count()
    VS = VS.where(~_nan_mask(T)).where(VBD_mask)

    VS_levels = _ann_frac(VS, steps_per_year).rename(var_name)
    return _assign_severity_level(VS_levels, frac_thresholds=severity_thresholds)


def VSmalaria(ds_dict, T_range=[22.9, 27.8], VBD_mask_file="default", severity_thresholds=None):
    """
    Malaria transmission suitability — fraction of year with suitable temperature.
    Temperature range from literature.
    """
    logger.info("calculating malaria suitability...")
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["VSmalaria"]
    result = _vbd_suitability(ds_dict, T_range, VBD_mask_file, "VSmalaria", severity_thresholds)
    return _add_metric_metadata(result, "VSmalaria", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"Anopheles suitability. T_range={T_range}. VBD_mask_file={VBD_mask_file}.")


def VSzika(ds_dict, T_range=[23.9, 34], VBD_mask_file="default", severity_thresholds=None):
    """
    Zika transmission suitability — fraction of year with suitable temperature.
    """
    logger.info("calculating zika suitability...")
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["VSzika"]
    result = _vbd_suitability(ds_dict, T_range, VBD_mask_file, "VSzika", severity_thresholds)
    return _add_metric_metadata(result, "VSzika", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"Aedes aegypti zika suitability. T_range={T_range}. VBD_mask_file={VBD_mask_file}.")


def VSdengueAeg(ds_dict, T_range=[21.3, 34], VBD_mask_file="default", severity_thresholds=None):
    """
    Dengue (Aedes aegypti) transmission suitability.
    """
    logger.info("calculating dengue aegypti suitability...")
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["VSdengueAeg"]
    result = _vbd_suitability(ds_dict, T_range, VBD_mask_file, "VSdengueAeg", severity_thresholds)
    return _add_metric_metadata(result, "VSdengueAeg", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"Aedes aegypti dengue suitability. T_range={T_range}. VBD_mask_file={VBD_mask_file}.")


def VSdengueAlb(ds_dict, T_range=[19.9, 29.4], VBD_mask_file="default", severity_thresholds=None):
    """
    Dengue (Aedes albopictus) transmission suitability.
    """
    logger.info("calculating dengue albopictus suitability...")
    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["VSdengueAlb"]
    result = _vbd_suitability(ds_dict, T_range, VBD_mask_file, "VSdengueAlb", severity_thresholds)
    return _add_metric_metadata(result, "VSdengueAlb", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"Aedes albopictus dengue suitability. T_range={T_range}. VBD_mask_file={VBD_mask_file}.")


def VbrS(ds_dict, salinity_max=28, SST_min=18, severity_thresholds=None):
    """
    Vibrio bacteria suitability 
    Suitability = monthly SST >= SST_min and SSS < salinity_max.

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
    """
    logger.info("calculating vibrio suitability...")

    SST = _check_and_convert_units(da=ds_dict["tos"], input_var="tos", conv_type="C")
    SSS = _check_and_convert_units(da=ds_dict["sos"], input_var="sos", conv_type="psu")

    SST = SST.resample(time="1ME").mean()
    SSS = SSS.resample(time="1ME").mean()
    steps_per_year = _get_tsteps(SST)

    if severity_thresholds is None:
        severity_thresholds = _default_thresholds["VbrS"]

    VbrS_val = SST.where((SST >= SST_min) & (SSS < salinity_max)).resample(time="1YE").count()
    VbrS_val = VbrS_val.where(~_nan_mask(SST))
    VbrS_val = _ann_frac(VbrS_val, steps_per_year).rename("VbrS")
    result = _assign_severity_level(VbrS_val, frac_thresholds=severity_thresholds)
    return _add_metric_metadata(result, "VbrS", ds_dict, severity_thresholds=severity_thresholds, units="fraction of year", notes=f"Vibrio suitability: SST >= {SST_min} degC and SSS < {salinity_max} psu. Trinanes et al. 2021.")

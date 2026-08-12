"""
calculate_all — runs every metric, skipping those with missing inputs or errors

usage:
    results = gchi.calculate_all(ds_dict, base_dict)

    # optional file paths for metrics that need external files
    results = gchi.calculate_all(
        ds_dict, base_dict,
        fwi_mask_file="path/to/fwi_mask.nc",
        environmental_zone_file="path/to/env_zones.nc",
        VBD_mask_file="path/to/vbd_mask.nc",
        mda8_scale_file="path/to/mda8_scale.nc",
    )

returns a GCHIResults object -- behaves like a normal dict of {metric_name: xr.Dataset}
for all metrics that ran successfully, but also remembers which metrics were skipped
(missing inputs) or failed (error). call results.summary() any time afterwards to
print/reprint that -- you don't have to catch it off the initial printout.
"""

import traceback

from .heat import AT, HI, Hu, WBT, WBGT, UTCIhot, HWF, TXC, TR, wbt_values
from .cold import UTCIcold, TNXp
from .fire import FI, HDW, FWI
from .aq import O3, PM2pt5
from .drought import CDD, SPI, SMSXp
from .disease import VSmalaria, VSzika, VSdengueAeg, VSdengueAlb, VbrS
from .weather import PRXmm, PR1day, PR5day
from ._core import _is_prepared
from ._log import logger, set_verbose


class GCHIResults(dict):
    """
    dict of {metric_name: xr.Dataset} for metrics that ran successfully, plus
    a record of what was skipped (missing inputs) or failed (error) so the
    summary can be reprinted any time, not just right after calculate_all runs.
    """
    def __init__(self, results, skipped, failed):
        super().__init__(results)
        self.skipped = skipped
        self.failed = failed

    def summary(self):
        print(f"calculated : {len(self)} metrics -- {list(self.keys())}")
        if self.skipped:
            print(f"skipped    : {len(self.skipped)} metrics")
            for name, reason in self.skipped.items():
                print(f"  {name}: {reason}")
        if self.failed:
            print(f"failed     : {len(self.failed)} metrics")
            for name, reason in self.failed.items():
                print(f"  {name}: {reason}")


# required ds_dict keys per metric
# if any key is missing, the metric is skipped without even trying
_REQUIRED_VARS = {
    "AT":          {"tasmax", "hurs"},
    "HI":          {"tasmax", "hurs"},
    "Hu":          {"tasmax", "hurs"},
    "WBT":         {"tasmax", "ps"},        # also huss or hurs
    "WBGT":        {"tasmax", "tas", "ps"}, # also huss or hurs
    "UTCIhot":     {"tasmax", "tas", "hurs", "ps"},
    "UTCIcold":    {"tasmin", "tas", "hurs", "ps"},
    "HWF":         {"tas"},
    "TXC":         {"tasmax"},
    "TR":          {"tasmin"},
    "TNXp":        {"tasmin"},
    "FI":          {"sfcWind", "tas", "hurs"},
    "HDW":         {"sfcWind", "tas", "hurs"},
    "FWI":         {"tasmax", "pr", "sfcWind"},  # also hurs or hursmin
    "O3":          {"o3", "tas", "ps"},
    "PM2pt5":      {"mmrbc", "mmrdust", "mmroa", "mmrso4", "mmrss", "tas", "ps"},
    "CDD":         {"pr"},
    "SPI":         {"pr"},
    "SMSXp":       {"mrsos"},
    "VSmalaria":   {"tas"},
    "VSzika":      {"tas"},
    "VSdengueAeg": {"tas"},
    "VSdengueAlb": {"tas"},
    "VbrS":        {"tos", "sos"},
    "PRXmm":       {"pr"},
    "PR1day":      {"pr"},
    "PR5day":      {"pr"},
}

# required base_dict keys per metric (if base_dict is provided)
# if any key is missing, the metric is skipped
_REQUIRED_BASE = {
    "HWF":    {"tas"},       # also needs t{p}p_calday, checked at runtime
    "TNXp":   set(),         # checked at runtime (needs tasmin_{p}p keys)
    "SPI":    {"pr"},
    "SMSXp":   set(),       # checked at runtime (needs mrsos_{p}p keys)
    "PR1day": set(),         # checked at runtime (needs pr_{p}p keys)
    "PR5day": set(),         # checked at runtime (needs rx5day_{p}p keys)
}


def _check_vars(ds_dict, base_dict, metric):
    """
    Returns (can_run, reason) — reason is a short string explaining why it can't run.
    Checks ds_dict keys and (where relevant) base_dict keys.
    """
    required = _REQUIRED_VARS.get(metric, set())
    missing_ds = required - set(ds_dict.keys())

    # WBT/WBGT/UTCIhot/UTCIcold need huss or hurs, not necessarily both
    if metric in {"WBT", "WBGT", "UTCIhot", "UTCIcold"}:
        if "huss" not in ds_dict and "hurs" not in ds_dict:
            missing_ds.add("huss or hurs")

    # FWI needs hurs or hursmin
    if metric == "FWI":
        if "hurs" not in ds_dict and "hursmin" not in ds_dict:
            missing_ds.add("hurs or hursmin")

    if missing_ds:
        return False, f"missing ds_dict vars: {', '.join(sorted(missing_ds))}"

    # base_dict checks
    if metric in _REQUIRED_BASE:
        if base_dict is None:
            return False, "base_dict not provided"
        required_base = _REQUIRED_BASE[metric]
        missing_base = required_base - set(base_dict.keys())
        if missing_base:
            return False, f"missing base_dict vars: {', '.join(sorted(missing_base))}"

    return True, None


def calculate_all(
    ds_dict,
    base_dict=None,
    # file paths for metrics that need external files -- default 'default' downloads
    # and caches gchi's reference files (https://zenodo.org/records/19239161) the
    # first time each is needed. pass None to skip masking/scaling for that metric,
    # or your own path to use a custom file.
    fwi_mask_file="default",
    environmental_zone_file="default",
    VBD_mask_file="default",
    mda8_scale_file="default",
    mda8_scale_varname="o3",
    # optional: run prepare_inputs automatically before calculating
    # leave model_grid_file=None if you don't want any regridding to happen
    model_grid_file="default",
    regrid=True,
    regrid_method="bilinear",
    spatial_chunk="auto",
    mask_land=True,
    land_mask_file="default",
    land_mask_var="land_mask",
    # optional overrides
    TR_thresh=20,
    percentile_base=90,
    verbose=False,
):
    """
    Run all gchi metrics on ds_dict, skipping those with missing inputs.

    Parameters
    ----------
    ds_dict : dict
        dict of xr.DataArrays keyed by CMIP6 shortname
    base_dict : dict, optional
        output from calculate_base_period_percentiles(), built from your own
        historical/base-period data. needed for HWF, TNXp, SPI, SMSXp, PR1day, PR5day.
        NOT derived from ds_dict automatically -- if omitted, those metrics are
        skipped rather than guessing a base period from the study period data.
    fwi_mask_file : str, optional
        path to infrequent burning mask file (for FWI)
    environmental_zone_file : str, optional
        path to environmental zone file (for FWI spatially-varying thresholds)
    VBD_mask_file : str, optional
        path to aridity mask file (for VSmalaria, VSzika, VSdengue*)
    mda8_scale_file : str, optional
        path to MDA8 scale factor file (for O3)
    mda8_scale_varname : str
        variable name in mda8_scale_file (default 'o3')
    TR_thresh : float
        tropical nights threshold (default 20°C)
    percentile_base : int
        percentile base for HWF (default 90)

    Returns
    -------
    GCHIResults
        dict-like {metric_name: xr.Dataset} for each metric that ran successfully.
        skipped and failed metrics are printed to console, and are also stored on
        the returned object as .skipped / .failed -- call .summary() to reprint.
    """

    set_verbose(verbose)

    # run prepare_inputs automatically, unless ds_dict has already been through it
    # (e.g. user called prepare_inputs themselves beforehand) -- never prep twice
    from .inputs import prepare_inputs
    if not _is_prepared(ds_dict):
        logger.info("running prepare_inputs before calculate_all...")
        ds_dict = prepare_inputs(ds_dict, spatial_chunk=spatial_chunk,
                                 model_grid_file=model_grid_file,
                                 regrid=regrid,
                                 regrid_method=regrid_method,
                                 mask_land=mask_land,
                                 land_mask_file=land_mask_file,
                                 land_mask_var=land_mask_var)
    else:
        logger.info("ds_dict already prepped -- skipping prepare_inputs.")

    # if base_dict is provided with raw data (tas/tasmin/pr as DataArrays or Datasets),
    # run calculate_base_period_percentiles on it to get the percentile thresholds needed
    # by HWF, TNXp, PR1day, PR5day. SPI uses the raw pr directly from base_dict.
    if base_dict is not None:
        if not _is_prepared(base_dict):
            logger.info("running prepare_inputs on base_dict...")
            base_dict = prepare_inputs(base_dict,
                        model_grid_file=model_grid_file,
                        regrid=regrid,
                        regrid_method=regrid_method,
                        mask_land=mask_land,
                        land_mask_file=land_mask_file,
                        land_mask_var=land_mask_var
                    )
        else:
            logger.info("base_dict already prepped -- skipping prepare_inputs.")
        from .inputs import calculate_base_period_percentiles
        needs_pct = not any(k.endswith("p_calday") or k.endswith("p") and "_" in k
                            for k in base_dict.keys())
        if needs_pct:
            logger.info("base_dict looks like raw data -- computing percentile thresholds automatically...")
            base_dict_pct = calculate_base_period_percentiles(
                tas=base_dict.get("tas"),
                tasmin=base_dict.get("tasmin"),
                pr=base_dict.get("pr"),
                mrsos=base_dict.get("mrsos"),
            )
            # merge: keep raw data (for SPI) + add computed percentiles
            base_dict = {**base_dict, **base_dict_pct}
    else:
        # do NOT derive a base period from ds_dict -- ds_dict is the study period,
        # and the base period must be a deliberately-chosen, separate historical
        # dataset. if you don't pass base_dict, HWF/TNXp/SPI/SMSXp/PR1day/PR5day
        # are simply skipped (see _check_vars / _REQUIRED_BASE above).
        logger.warning("base_dict not provided -- HWF/TNXp/SPI/SMSXp/PR1day/PR5day will be skipped. "
                       "pass a base_dict (built from your own historical data via "
                       "calculate_base_period_percentiles) to include them.")

    results = {}
    skipped = {}   # metric -> reason (missing inputs)
    failed  = {}   # metric -> error message

    def _run(name, fn):
        """attempt to run fn(), store result or failure"""
        can_run, reason = _check_vars(ds_dict, base_dict, name)
        if not can_run:
            skipped[name] = reason
            logger.info(f"skipping {name} — {reason}")
            return
        logger.info(f"running {name}...")
        try:
            result = fn()
            if result is not None:
                results[name] = result
            else:
                skipped[name] = "returned None (likely a threshold mismatch — check warnings above)"
                logger.warning(f"skipping {name} — returned None")
        except Exception as e:
            failed[name] = str(e)
            logger.error(f"ERROR in {name}: {e}")
            logger.error(f"    {traceback.format_exc().splitlines()[-2]}")  # one-liner from traceback

    logger.info("=== calculate_all ===")

    # --- heat stress ---
    logger.info("-- heat stress --")
    _run("AT",      lambda: AT(ds_dict))
    _run("HI",      lambda: HI(ds_dict))
    _run("Hu",      lambda: Hu(ds_dict))

    # WBT and WBGT both need the wet-bulb temp (Twb) -- compute it once and
    # share it between them instead of calculating it twice
    _twb_cache = {}
    def _get_twb():
        if "twb" not in _twb_cache:
            _twb_cache["twb"] = wbt_values(ds_dict)
        return _twb_cache["twb"]

    _run("WBT",     lambda: WBT(ds_dict, Twb=_get_twb()))
    _run("WBGT",    lambda: WBGT(ds_dict, Twb=_get_twb()))
    _run("UTCIhot", lambda: UTCIhot(ds_dict))
    _run("HWF",     lambda: HWF(ds_dict, base_dict, percentile_base=percentile_base))
    _run("TXC",     lambda: TXC(ds_dict))
    _run("TR",      lambda: TR(ds_dict, TR_thresh=TR_thresh))

    # --- cold ---
    logger.info("-- cold extremes --")
    _run("UTCIcold", lambda: UTCIcold(ds_dict))
    _run("TNXp",     lambda: TNXp(ds_dict, base_dict))

    # --- fire ---
    logger.info("-- fire --")
    _run("FI",  lambda: FI(ds_dict, fire_mask_file=fwi_mask_file))
    _run("HDW", lambda: HDW(ds_dict, fire_mask_file=fwi_mask_file))
    _run("FWI", lambda: FWI(ds_dict,
                            fwi_mask_file=fwi_mask_file,
                            environmental_zone_file=environmental_zone_file))

    # --- air quality ---
    logger.info("-- air quality --")
    _run("O3",     lambda: O3(ds_dict, mda8_scale_file=mda8_scale_file,
                               mda8_scale_varname=mda8_scale_varname))
    _run("PM2pt5", lambda: PM2pt5(ds_dict))

    # --- drought ---
    logger.info("-- drought --")
    _run("CDD",  lambda: CDD(ds_dict))
    _run("SPI",  lambda: SPI(ds_dict, base_dict))
    _run("SMSXp", lambda: SMSXp(ds_dict, base_dict))

    # --- disease ---
    logger.info("-- disease --")
    _run("VSmalaria",   lambda: VSmalaria(ds_dict, VBD_mask_file=VBD_mask_file))
    _run("VSzika",      lambda: VSzika(ds_dict, VBD_mask_file=VBD_mask_file))
    _run("VSdengueAeg", lambda: VSdengueAeg(ds_dict, VBD_mask_file=VBD_mask_file))
    _run("VSdengueAlb", lambda: VSdengueAlb(ds_dict, VBD_mask_file=VBD_mask_file))
    _run("VbrS",        lambda: VbrS(ds_dict))

    # --- weather ---
    logger.info("-- weather --")
    _run("PRXmm", lambda: PRXmm(ds_dict))
    _run("PR1day", lambda: PR1day(ds_dict, base_dict))
    _run("PR5day", lambda: PR5day(ds_dict, base_dict))

    # --- summary ---
    results = GCHIResults(results, skipped, failed)
    logger.info("=== calculate_all complete ===")
    results.summary()

    return results
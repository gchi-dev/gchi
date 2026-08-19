import gchi
from gchi import GCHIResults


def test_calculate_all_skips_missing_vars(ds_dict):
    """
    test if only one diag passed (pr-only here) 
    metrics needing other vars should be skipped not crash
    """
    minimal = {"pr": ds_dict["pr"]}
    results = gchi.calculate_all(
        minimal, model_grid_file=None, land_mask_file=None, regrid=False,
    )
    assert "DSD" in results
    assert "AT" in results.skipped
    assert "missing ds_dict vars" in results.skipped["AT"]


def test_calculate_all_returns_gchiresults(ds_dict):
    results = gchi.calculate_all(
        ds_dict, model_grid_file=None, land_mask_file=None, regrid=False,
    )
    assert isinstance(results, GCHIResults)
    assert isinstance(results, dict)
    # dict-like access still works
    assert "AT" in results
    assert hasattr(results, "skipped")
    assert hasattr(results, "failed")


def test_calculate_all_never_derives_base_dict_from_ds_dict(ds_dict):
    """
    if no base_dict passed, base-period metrics should be skipped
    """
    results = gchi.calculate_all(
        ds_dict, model_grid_file=None, land_mask_file=None, regrid=False,
    )
    for metric in ["HWF", "TNXp", "SPI", "SMSXp", "PR1day", "PR5day"]:
        assert metric in results.skipped
        assert results.skipped[metric] == "base_dict not provided"


def test_calculate_all_with_base_dict_runs_base_period_metrics(ds_dict, hist_dict):
    prepped_hist = gchi.prepare_inputs(hist_dict, regrid=False, mask_land=False)
    base_dict = gchi.calculate_base_period_percentiles(
        tas=prepped_hist["tas"], tasmin=prepped_hist["tasmin"],
        pr=prepped_hist["pr"], mrsos=prepped_hist["mrsos"],
        base_years=(2010, 2011),
    )
    results = gchi.calculate_all(
        ds_dict, base_dict, model_grid_file=None, land_mask_file=None, regrid=False,
    )
    assert "TNXp" in results
    assert "SPI" in results

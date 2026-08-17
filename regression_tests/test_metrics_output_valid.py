import gchi


def _assert_valid_severity(ds, metric):
    """
    check that every {metric}_severity_level output stay within the 0-4 scale
    """
    key = f"{metric}_severity_level"
    assert key in ds, f"{metric} result missing {key}"
    sev = ds[key].compute()
    valid = float(sev.min(skipna=True))
    assert valid >= 0, f"{metric} severity level below 0"
    assert float(sev.max(skipna=True)) <= 4, f"{metric} severity level above 4"


def test_at_runs_and_severity_bounded(ds_dict):
    r = gchi.AT(ds_dict)
    _assert_valid_severity(r, "AT")


def test_hi_runs_and_severity_bounded(ds_dict):
    r = gchi.HI(ds_dict)
    _assert_valid_severity(r, "HI")


def test_dsd_runs_and_severity_bounded(ds_dict):
    r = gchi.DSD(ds_dict)
    _assert_valid_severity(r, "DSD")


def test_prxmm_runs_and_severity_bounded(ds_dict):
    r = gchi.PRXmm(ds_dict)
    _assert_valid_severity(r, "PRXmm")


def test_vsmalaria_runs_and_severity_bounded(ds_dict):
    r = gchi.VSmalaria(ds_dict, VBD_mask_file=None)
    _assert_valid_severity(r, "VSmalaria")


def test_vbrs_runs_and_severity_bounded(ds_dict):
    r = gchi.VbrS(ds_dict)
    _assert_valid_severity(r, "VbrS")


def test_fi_runs_and_severity_bounded(ds_dict):
    r = gchi.FI(ds_dict, fire_mask_file=None)
    _assert_valid_severity(r, "FI")

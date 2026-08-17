import numpy as np
import xarray as xr

import gchi


def _sev(values, name):
    da = xr.DataArray(values, dims=["cell"])
    return xr.Dataset({f"{name}_severity_level": da})


def test_category_average_equal_weight_default():
    results = {
        "AT": _sev([1.0, 2.0], "AT"),
        "HI": _sev([3.0, 4.0], "HI"),
    }
    cats = gchi.category_averages(results, categories={"heat stress": ["AT", "HI"]})
    expected = [(1 + 3) / 2, (2 + 4) / 2]
    np.testing.assert_allclose(cats["heat stress"].values, expected)


def test_category_average_custom_weights():
    results = {
        "AT": _sev([1.0, 2.0], "AT"),
        "HI": _sev([3.0, 4.0], "HI"),
    }
    cats = gchi.category_averages(
        results, categories={"heat stress": ["AT", "HI"]},
        metric_weights={"AT": 3.0},  # HI defaults to weight 1
    )
    expected = [(1 * 3 + 3 * 1) / 4, (2 * 3 + 4 * 1) / 4]
    np.testing.assert_allclose(cats["heat stress"].values, expected)


def test_category_average_missing_metric_skipped_gracefully():
    results = {"AT": _sev([1.0, 2.0], "AT")}  # HI not present
    cats = gchi.category_averages(results, categories={"heat stress": ["AT", "HI"]})
    np.testing.assert_allclose(cats["heat stress"].values, [1.0, 2.0])


def test_category_average_empty_category_omitted():
    results = {}
    cats = gchi.category_averages(results, categories={"heat stress": ["AT", "HI"]})
    assert "heat stress" not in cats.data_vars


def test_zero_as_nan_excludes_zero_not_counts_as_zero():
    results = {"VbrS": _sev([0.0, 0.0, 2.0, 4.0], "VbrS")}
    cats = gchi.category_averages(results, categories={"disease": ["VbrS"]})
    vals = cats["disease"].values
    assert np.isnan(vals[0]) and np.isnan(vals[1])
    np.testing.assert_allclose(vals[2:], [2.0, 4.0])


def test_zero_as_nan_can_be_disabled():
    results = {"VbrS": _sev([0.0, 2.0], "VbrS")}
    cats = gchi.category_averages(
        results, categories={"disease": ["VbrS"]}, zero_as_nan_metrics=set(),
    )
    np.testing.assert_allclose(cats["disease"].values, [0.0, 2.0])


def test_composite_average_equal_weight_default():
    results = {
        "AT": _sev([2.0, 2.5], "AT"),
        "FI": _sev([0.0, 1.0], "FI"),
    }
    cats = gchi.category_averages(
        results, categories={"heat stress": ["AT"], "fire weather": ["FI"]},
    )
    comp = gchi.composite_average(cats)
    expected = [(2.0 + 0.0) / 2, (2.5 + 1.0) / 2]
    np.testing.assert_allclose(comp.values, expected)


def test_composite_average_custom_category_weights():
    results = {
        "AT": _sev([2.0], "AT"),
        "FI": _sev([0.0], "FI"),
    }
    cats = gchi.category_averages(
        results, categories={"heat stress": ["AT"], "fire weather": ["FI"]},
    )
    comp = gchi.composite_average(cats, category_weights={"heat stress": 2.0})
    expected = [(2.0 * 2 + 0.0 * 1) / 3]
    np.testing.assert_allclose(comp.values, expected)

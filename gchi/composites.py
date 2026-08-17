"""
composites.py -- categorical and composite severity averages across gchi metrics.

category_averages() collapses each category's member metrics (their
{metric}_severity_level fields) into one severity field per category,
via a weighted mean across metrics (equal weight by default).

composite_average() then collapses those category fields into a single
overall severity field, via a weighted mean across categories (equal
weight by default).

both operate purely over the metric/category dimension -- whatever other
dims your results carry (time, lat, lon, model, ...) are left untouched.
reduce over those yourself (e.g. .mean(["model", "time"])), before or
after, depending on your workflow.
"""

import xarray as xr

from ._log import logger

# default grouping of gchi metrics into hazard categories.
# override via the `categories` argument to category_averages() -- metrics
# not present in your results are skipped automatically, no need to prune
# this list yourself.
DEFAULT_CATEGORIES = {
    "heat stress":        ["AT", "HI", "Hu", "WBT", "WBGT", "UTCIhot", "TXC", "TR", "HWF"],
    "cold stress":        ["UTCIcold", "TNXp"],
    "fire weather":       ["FI", "HDW", "FWI"],
    "air quality":        ["O3", "PM2pt5"],
    "drought":            ["DSD", "SPI", "SMSXp"],
    "infectious disease": ["VSmalaria", "VSzika", "VSdengueAeg", "VSdengueAlb", "VbrS"],
    "weather extremes":   ["PRXmm", "PR1day", "PR5day"],
}

# metrics where a severity level of exactly 0 means "not applicable here"
# rather than "no hazard" -- e.g. VbrS is only meaningful in coastal cells
# and reports a technically-valid 0 everywhere else. treated as NaN
# (excluded from the average, not counted as a real 0) so it doesn't drag
# down cells it was never meant to describe. pass zero_as_nan_metrics=set()
# to disable this.
DEFAULT_ZERO_AS_NAN = {"VbrS"}


def _get_severity(results, metric):
    """pull {metric}_severity_level out of results[metric], or None if unavailable"""
    ds = results.get(metric)
    if ds is None:
        return None
    key = f"{metric}_severity_level"
    if key not in ds:
        return None
    return ds[key]


def category_averages(results, categories=None, metric_weights=None, zero_as_nan_metrics=None):
    """
    Collapse each category's metrics into one severity field per category.

    Parameters
    ----------
    results : dict-like {metric_name: xr.Dataset}
        e.g. the output of calculate_all(), or any dict of metric result
        Datasets each containing a '{metric}_severity_level' variable.
    categories : dict {category_name: [metric_names]}, optional
        defaults to DEFAULT_CATEGORIES.
    metric_weights : dict {metric_name: weight}, optional
        weight for each metric within its category average. metrics not
        listed default to 1 (equal weighting -- the default behavior).
    zero_as_nan_metrics : set of str, optional
        metrics where severity level 0 means "not applicable" and should
        be excluded (as NaN) rather than counted as a real 0. defaults to
        DEFAULT_ZERO_AS_NAN ({"VbrS"}). pass an empty set to disable.

    Returns
    -------
    xr.Dataset
        one variable per category (named as given in `categories`), each
        the weighted mean of its member metrics' severity_level fields.
        missing metrics are skipped; categories with zero available
        metrics are omitted (with a warning).
    """
    categories = categories or DEFAULT_CATEGORIES
    metric_weights = metric_weights or {}
    zero_as_nan_metrics = DEFAULT_ZERO_AS_NAN if zero_as_nan_metrics is None else set(zero_as_nan_metrics)

    cat_das = {}
    for cat_name, metrics in categories.items():
        das, weights = [], []
        for m in metrics:
            da = _get_severity(results, m)
            if da is None:
                logger.info(f"category_averages: '{m}' not available -- skipping in '{cat_name}'")
                continue
            da = da.astype(float)
            if m in zero_as_nan_metrics:
                da = da.where(da > 0)
            das.append(da.assign_coords(metric=m))
            weights.append(metric_weights.get(m, 1.0))

        if not das:
            logger.warning(f"category_averages: '{cat_name}' had no available metrics -- skipping")
            continue

        stacked = xr.concat(das, dim="metric")
        weights_da = xr.DataArray(weights, dims="metric", coords={"metric": stacked.metric})
        cat_das[cat_name] = stacked.weighted(weights_da).mean("metric", skipna=True)

    return xr.Dataset(cat_das)


def composite_average(category_ds, category_weights=None):
    """
    Collapse a category_averages() Dataset into a single composite severity field.

    Parameters
    ----------
    category_ds : xr.Dataset
        output of category_averages() -- one variable per category.
    category_weights : dict {category_name: weight}, optional
        weight for each category. categories not listed default to 1
        (equal weighting -- the default behavior).

    Returns
    -------
    xr.DataArray
        weighted mean across categories.
    """
    category_weights = category_weights or {}
    stacked = category_ds.to_array(dim="category")
    weights = [category_weights.get(c, 1.0) for c in stacked.category.values]
    weights_da = xr.DataArray(weights, dims="category", coords={"category": stacked.category})
    return stacked.weighted(weights_da).mean("category", skipna=True)

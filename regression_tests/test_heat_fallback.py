import numpy as np
import xarray as xr

import gchi
from gchi.heat import wbt_values


def _mkda(time, lat, lon, base, spread, units, seed):
    rng = np.random.RandomState(seed)
    data = base + spread * rng.randn(len(time), len(lat), len(lon))
    da = xr.DataArray(
        data, dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )
    da.attrs["units"] = units
    return da


def test_wbt_falls_back_to_hurs_when_huss_resolution_mismatched(grid):
    """
    possible that users will pass hurs and huss at different temporal resos for diff gchi metrics
    if pass huss at monthly resolution alongside daily tasmax/ps/hurs, wbt 
    should fall back to deriving specific humidity from hurs instead
    """
    lat, lon = grid
    daily = xr.date_range("2010-01-01", periods=365, freq="D")
    monthly = xr.date_range("2010-01-01", periods=12, freq="MS")

    ds_mismatch = {
        "tasmax": _mkda(daily, lat, lon, 300, 5, "K", 1),
        "ps":     _mkda(daily, lat, lon, 101325, 100, "Pa", 2),
        "hurs":   _mkda(daily, lat, lon, 60, 10, "%", 3).clip(1, 99),
        "huss":   _mkda(monthly, lat, lon, 0.01, 0.002, "fraction", 4).clip(0.0001, 0.02),
    }

    result = wbt_values(ds_mismatch) 
    assert result is not None

    ds_hurs_only = {k: v for k, v in ds_mismatch.items() if k != "huss"}
    expected = wbt_values(ds_hurs_only)
    xr.testing.assert_allclose(result, expected)


def test_wbt_uses_huss_when_resolution_matches(grid):
    """
    when huss genuinely matches tasmax's resolution, it should still be used for wbt 
    """
    lat, lon = grid
    daily = xr.date_range("2010-01-01", periods=365, freq="D")

    ds_matching = {
        "tasmax": _mkda(daily, lat, lon, 300, 5, "K", 1),
        "ps":     _mkda(daily, lat, lon, 101325, 100, "Pa", 2),
        "hurs":   _mkda(daily, lat, lon, 60, 10, "%", 3).clip(1, 99),
        "huss":   _mkda(daily, lat, lon, 0.01, 0.002, "fraction", 4).clip(0.0001, 0.02),
    }

    result = wbt_values(ds_matching)
    ds_hurs_only = {k: v for k, v in ds_matching.items() if k != "huss"}
    hurs_only_result = wbt_values(ds_hurs_only)

    assert not result.equals(hurs_only_result)

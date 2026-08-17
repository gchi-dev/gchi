import xarray as xr

from gchi._core import _check_and_convert_units


def _da(value, units):
    return xr.DataArray([value], dims=["x"], attrs={"units": units})


def test_kelvin_to_celsius():
    out = _check_and_convert_units(da=_da(273.15, "K"), input_var="tas", conv_type="C")
    assert abs(float(out[0]) - 0.0) < 1e-6


def test_celsius_to_kelvin():
    out = _check_and_convert_units(da=_da(0.0, "degC"), input_var="tas", conv_type="K")
    assert abs(float(out[0]) - 273.15) < 1e-6


def test_percent_to_fraction():
    out = _check_and_convert_units(da=_da(50.0, "%"), input_var="hurs", conv_type="fraction")
    assert abs(float(out[0]) - 0.5) < 1e-6


def test_fraction_passthrough():
    out = _check_and_convert_units(da=_da(0.5, "fraction"), input_var="hurs", conv_type="fraction")
    assert abs(float(out[0]) - 0.5) < 1e-6


def test_pa_to_hpa():
    out = _check_and_convert_units(da=_da(101325.0, "Pa"), input_var="ps", conv_type="hPa")
    assert abs(float(out[0]) - 1013.25) < 1e-6


def test_ms_to_kmh():
    out = _check_and_convert_units(da=_da(10.0, "m s-1"), input_var="sfcWind", conv_type="km h-1")
    assert abs(float(out[0]) - 36.0) < 1e-6

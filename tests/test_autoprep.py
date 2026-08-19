import gchi
from gchi._core import _is_prepared


def test_prepare_inputs_marks_data_as_prepared(raw_ds_dict):
    assert not _is_prepared(raw_ds_dict)
    prepped = gchi.prepare_inputs(raw_ds_dict, regrid=False, mask_land=False)
    assert _is_prepared(prepped)


def test_calling_metric_on_already_prepped_data_does_not_reprep(ds_dict):
    """
    try calling metric on already prepperd ds_dict fixture
    """
    result = gchi.AT(ds_dict)  # would raise via the no_network fixture if it re-prepped
    assert result is not None


def test_direct_metric_call_on_raw_data_auto_preps(raw_ds_dict, monkeypatch):
    """
    calling a metric on raw (unprepped) data should trigger prepare_inputs
    automatically 
    """
    import gchi.inputs as inputs_mod

    calls = []
    original = inputs_mod.prepare_inputs

    def _spy(*args, **kwargs):
        calls.append(1)
        kwargs.setdefault("regrid", False)
        kwargs.setdefault("mask_land", False)
        return original(*args, **kwargs)

    monkeypatch.setattr(gchi, "prepare_inputs", _spy)
    gchi.TXC(raw_ds_dict)
    assert len(calls) == 1, "expected prepare_inputs to be called exactly once via auto-prep"

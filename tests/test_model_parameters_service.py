"""Tests for shared model-parameter helper service."""

from ModelExplorer.services.model_parameters import ensure_pd_parameter_defaults, merge_parameter_values


def test_merge_parameter_values_prioritizes_hidden_defaults() -> None:
    visible = {"scale": 1.0, "radius": 50.0}
    hidden = {"background": 0.0}
    merged = merge_parameter_values(visible, hidden)
    assert merged == {"scale": 1.0, "radius": 50.0, "background": 0.0}


def test_ensure_pd_parameter_defaults_adds_pd_n_only_for_present_params() -> None:
    params = {"radius": 10.0, "background": 0.0}
    ensure_pd_parameter_defaults(params, ["radius", "length"], pd_n=42)
    assert params["radius_pd_n"] == 42
    assert "length_pd_n" not in params

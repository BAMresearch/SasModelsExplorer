"""Tests for McSAS3 run-configuration export helpers."""

from __future__ import annotations

import pytest

from ModelExplorer.services.run_config_export import build_mcsas3_run_configuration


def test_build_run_configuration_uses_selected_fit_parameter() -> None:
    config = build_mcsas3_run_configuration(
        model_expression="sphere@hardsphere",
        parameters={"radius": 50.0, "background": 0.0, "volfraction": 0.3},
        fit_parameter="radius",
        fit_limits=(3.0, 300.0),
        log_random=True,
    )
    assert config["modelName"] == "sphere@hardsphere"
    assert config["fitParameterLimits"] == {"radius": [3.0, 300.0]}
    assert config["staticParameters"]["background"] == 0.0
    assert config["staticParameters"]["volfraction"] == 0.3
    assert "radius" not in config["staticParameters"]
    assert config["logRandom"] is True
    assert config["nRep"] == 10
    assert config["maxIter"] == 100000
    assert config["maxAccept"] == 10000
    assert config["nContrib"] == 300
    assert config["convCrit"] == 1.0
    assert config["nCores"] == 5


def test_build_run_configuration_rejects_non_numeric_fit_parameter() -> None:
    with pytest.raises(ValueError):
        build_mcsas3_run_configuration(
            model_expression="sphere",
            parameters={"radius_pd_type": "gaussian", "radius": 50.0},
            fit_parameter="radius_pd_type",
            fit_limits=(3.0, 300.0),
            log_random=True,
        )

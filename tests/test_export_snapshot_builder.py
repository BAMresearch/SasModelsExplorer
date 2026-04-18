"""Tests for export snapshot and payload builders."""

from __future__ import annotations

import numpy as np

from ModelExplorer.services.export_snapshot_builder import (
    SnapshotInputs,
    build_csv_model_parameter_table,
    build_export_snapshot,
    build_hdf5_state_payload,
    build_run_config_template_payload,
)
from ModelExplorer.types import OverlayData


def _build_snapshot():
    overlay = OverlayData(
        Q=np.array([0.1, 0.2]),
        I=np.array([1.0, 2.0]),
        ISigma=np.array([0.1, 0.1]),
        label="example",
    )
    return build_export_snapshot(
        SnapshotInputs(
            model_expression="sphere",
            q_unit="1/nm",
            intensity_unit="1/(m sr)",
            parameters={"scale": 1.0, "radius": 50.0},
            fit_parameter_names=["scale"],
            q_values=np.array([0.1, 0.2], dtype=float),
            model_intensity=np.array([1.1, 2.1], dtype=float),
            overlay_data=overlay,
            metadata={"instrument": "demo"},
        )
    )


def test_build_run_config_template_payload() -> None:
    snapshot = _build_snapshot()
    payload = build_run_config_template_payload(snapshot)
    assert payload.model_expression == "sphere"
    assert payload.template["fit_parameters"] == ["scale"]
    assert payload.template["units"] == {"Q": "1/nm", "I": "1/(m sr)"}


def test_build_csv_model_parameter_table() -> None:
    snapshot = _build_snapshot()
    table = build_csv_model_parameter_table(snapshot)
    assert table.headers == ("parameter", "value")
    assert ("radius", "50.0") in table.rows
    assert ("scale", "1.0") in table.rows


def test_build_hdf5_state_payload() -> None:
    snapshot = _build_snapshot()
    payload = build_hdf5_state_payload(snapshot)
    assert "model/Q" in payload.datasets
    assert "model/I" in payload.datasets
    assert "data/Q" in payload.datasets
    assert payload.attributes["model_expression"] == "sphere"
    assert payload.attributes["instrument"] == "demo"

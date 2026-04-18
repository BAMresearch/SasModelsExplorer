"""Unit tests for service-level HDF5 state IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ModelExplorer.services.export_contracts import ExportSnapshot
from ModelExplorer.services.hdf5_state_io import (
    StateWriteContext,
    dataset_float,
    dataset_scalar_to_python,
    dataset_text,
    parse_parameter_json,
    parse_string_list_json,
    read_state_hdf5,
    write_state_hdf5_tree,
)


class _DummyBaseData:
    def __init__(self, signal: object, units: str, uncertainties: dict[str, np.ndarray] | None = None) -> None:
        self.signal = np.asarray(signal, dtype=float)
        self.units = units
        self.uncertainties = uncertainties or {}


class _DummyDataBundle(dict[str, _DummyBaseData]):
    description: str = ""


def _build_snapshot() -> ExportSnapshot:
    return ExportSnapshot(
        model_expression="sphere",
        q_unit="1/nm",
        intensity_unit="1/(m sr)",
        parameters={"radius": 42.0, "scale": 1.0, "use_structure": True},
        fit_parameter_names=("radius",),
        q_values=np.array([0.05, 0.1, 0.2], dtype=float),
        model_intensity=np.array([15.0, 12.0, 8.0], dtype=float),
        overlay_data=None,
        metadata={"source": "unit-test"},
    )


def _stage_context(snapshot: ExportSnapshot) -> StateWriteContext:
    q_vals = np.array([0.1, 0.2], dtype=float)
    i_vals = np.array([10.0, 9.0], dtype=float)
    i_sigma = np.array([0.3, 0.3], dtype=float)
    q_sigma = np.array([0.001, 0.001], dtype=float)
    stage_frame = pd.DataFrame(
        {
            "Q": q_vals,
            "I": i_vals,
            "ISigma": i_sigma,
            "QSigma": q_sigma,
            "comment": ["a", "b"],
        }
    )

    stage_bundle = _DummyDataBundle(
        {
            "Q": _DummyBaseData(signal=q_vals, units="1/Angstrom"),
            "signal": _DummyBaseData(signal=i_vals, units="1/(cm sr)", uncertainties={"ISigma": i_sigma}),
        }
    )
    stage_bundle.description = "binned stage"

    return StateWriteContext(
        snapshot=snapshot,
        q_min=0.01,
        q_max=0.5,
        selected_data_stage="sample_binned",
        fit_parameter_names=["radius"],
        hidden_parameter_defaults={"radius_pd_n": 35, "magnetic": False},
        data_loading_yaml='sourceQUnits: "1/nm"\nnbins: 100\n',
        data_loading_preset="quickstart.yaml",
        data_source_file="/tmp/example.dat",
        data_mode_label="Binned data",
        stage_bundles={"sample_binned": stage_bundle},
        stage_frames={"sample_binned": stage_frame},
    )


def _parameter_metadata(name: str) -> tuple[str, float | None, float | None]:
    if name == "radius":
        return "nm", 1.0, 500.0
    if name == "scale":
        return "", 0.1, 10.0
    return "", None, None


def test_scalar_and_json_helpers() -> None:
    assert dataset_scalar_to_python(np.float64(3.5)) == 3.5
    assert dataset_scalar_to_python(np.int64(7)) == 7
    assert dataset_scalar_to_python(b"abc") == "abc"
    assert dataset_scalar_to_python(True) is True

    assert parse_parameter_json('{"radius": 10.0, "enabled": true}') == {"radius": 10.0, "enabled": True}
    assert parse_parameter_json("[1, 2, 3]") == {}
    assert parse_parameter_json("{broken") == {}
    assert parse_string_list_json('["a", 2, true]') == ["a", "2", "True"]
    assert parse_string_list_json('{"not": "a-list"}') == []
    assert parse_string_list_json("invalid") == []


def test_dataset_helpers_with_hdf5_group(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    output = tmp_path / "helpers.h5"
    with h5py.File(output, "w") as h5f:
        grp = h5f.create_group("x")
        grp.create_dataset("as_text", data="hello", dtype=h5py.string_dtype(encoding="utf-8"))
        grp.create_dataset("as_float", data=1.25)
        assert dataset_text(grp, "as_text", "fallback") == "hello"
        assert dataset_text(grp, "missing", "fallback") == "fallback"
        assert dataset_float(grp, "as_float", -1.0) == pytest.approx(1.25)
        assert dataset_float(grp, "missing", -1.0) == pytest.approx(-1.0)


def test_write_and_read_state_hdf5_tree_roundtrip(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    snapshot = _build_snapshot()
    context = _stage_context(snapshot)
    output = tmp_path / "service_roundtrip.h5"

    with h5py.File(output, "w") as h5f:
        write_state_hdf5_tree(h5f, context, parameter_metadata_getter=_parameter_metadata)

    with h5py.File(output, "r") as h5f:
        root = h5f["/analyses/SasModelsExplorer"]
        assert root["model/Q"].attrs["units"] == "1/nm"
        assert root["model/I"].attrs["units"] == "1/(m sr)"
        assert float(root["parameters/radius"].attrs["min"]) == 1.0
        assert float(root["parameters/radius"].attrs["max"]) == 500.0
        assert root["data/sample_binned/Q"].attrs["units"] == "1/Angstrom"
        assert root["data/sample_binned/Q"].attrs["uncertainties"] == "QSigma"
        assert root["data/sample_binned/I"].attrs["units"] == "1/(cm sr)"
        assert root["data/sample_binned/I"].attrs["uncertainties"] == "ISigma"
        assert root["SME_settings/data_loading_preset"][()].decode("utf-8") == "quickstart.yaml"
        assert "schema_version" in root["SME_settings"]

    with h5py.File(output, "r") as h5f:
        state = read_state_hdf5(
            h5f,
            str(output),
            _DummyBaseData,
            _DummyDataBundle,
            default_intensity_unit="1/(m sr)",
        )

    assert state.model_expression == "sphere"
    assert state.q_unit == "1/nm"
    assert state.q_min == pytest.approx(0.01)
    assert state.q_max == pytest.approx(0.5)
    assert state.parameters["radius"] == 42.0
    assert state.parameters["use_structure"] is True
    assert state.hidden_defaults["radius_pd_n"] == 35
    assert state.hidden_defaults["magnetic"] is False
    assert state.fit_names == ["radius"]
    assert state.yaml_preset == "quickstart.yaml"
    assert state.source_file == "/tmp/example.dat"
    assert "sample_binned" in state.stage_bundles
    bundle = state.stage_bundles["sample_binned"]
    assert bundle["Q"].units == "1/angstrom"
    assert bundle["signal"].units == "1/(cm sr)"
    assert "ISigma" in bundle["signal"].uncertainties


def test_read_state_hdf5_rejects_missing_root(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    output = tmp_path / "bad_schema.h5"
    with h5py.File(output, "w") as h5f:
        h5f.create_dataset("anything", data=np.array([1.0], dtype=float))

    with h5py.File(output, "r") as h5f:
        with pytest.raises(ValueError, match="Unsupported state schema"):
            read_state_hdf5(
                h5f,
                str(output),
                _DummyBaseData,
                _DummyDataBundle,
                default_intensity_unit="1/(m sr)",
            )

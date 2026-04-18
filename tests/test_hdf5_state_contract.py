"""Contract tests for SasModelsExplorer HDF5 state schema and compatibility."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ModelExplorer.modelexplorer import SasModelApp
from ModelExplorer.services.export_snapshot_builder import SnapshotInputs, build_export_snapshot


class _LineEditStub:
    def __init__(self, value: str) -> None:
        self._value = value

    def text(self) -> str:
        return self._value


class _ComboStub:
    def __init__(self, *, text: str, data: str) -> None:
        self._text = text
        self._data = data

    def currentText(self) -> str:
        return self._text

    def currentData(self) -> str:
        return self._data


class _StageBundle(dict):
    def __init__(self, *, q_units: str, i_units: str, description: str) -> None:
        super().__init__({"Q": SimpleNamespace(units=q_units), "signal": SimpleNamespace(units=i_units)})
        self.description = description


class _DataPanelStub:
    def __init__(
        self,
        *,
        stage_frames: dict[str, pd.DataFrame],
        stage_bundles: dict[str, object],
        yaml_text: str,
        preset_name: str | None,
        source_file: str,
        selected_stage: str,
    ) -> None:
        self._stage_frames = dict(stage_frames)
        self._stage_bundles = dict(stage_bundles)
        self._yaml_text = yaml_text
        self._preset_name = preset_name
        self.file_path_line = _LineEditStub(source_file)
        self.data_mode_combo = _ComboStub(text="Binned data", data=selected_stage)

    def get_stage_frames(self) -> dict[str, pd.DataFrame]:
        return dict(self._stage_frames)

    def get_stage_bundles(self) -> dict[str, object]:
        return dict(self._stage_bundles)

    def get_yaml_config_text(self) -> str:
        return self._yaml_text

    def get_selected_preset_name(self) -> str | None:
        return self._preset_name


class _FitPanelStub:
    def __init__(self, selected: list[str]) -> None:
        self._selected = list(selected)

    def get_selected_parameters(self) -> list[str]:
        return list(self._selected)


class _DummyBaseData:
    def __init__(self, signal: object, units: str, uncertainties: dict[str, np.ndarray] | None = None) -> None:
        self.signal = np.asarray(signal, dtype=float)
        self.units = units
        self.uncertainties = uncertainties or {}


class _DummyDataBundle(dict):
    description: str = ""


class _FakeApp:
    _dataset_scalar_to_python = staticmethod(SasModelApp._dataset_scalar_to_python)
    _dataset_text = staticmethod(SasModelApp._dataset_text)
    _dataset_float = staticmethod(SasModelApp._dataset_float)

    _build_state_write_context = SasModelApp._build_state_write_context
    _write_state_hdf5_tree = SasModelApp._write_state_hdf5_tree

    _read_state_hdf5 = SasModelApp._read_state_hdf5
    _read_new_state_hdf5 = SasModelApp._read_new_state_hdf5

    def __init__(self) -> None:
        self.i_units = ["1/(m sr)"]
        self.hidden_parameter_defaults = {"radius_pd_n": 35}
        self.q_min_input = _LineEditStub("0.01")
        self.q_max_input = _LineEditStub("0.5")
        self.fit_panel = _FitPanelStub(["radius"])
        self.data_panel = _DataPanelStub(
            stage_frames={
                "sample_binned": pd.DataFrame(
                    {
                        "Q": np.array([0.1, 0.2], dtype=float),
                        "I": np.array([10.0, 9.0], dtype=float),
                        "ISigma": np.array([0.3, 0.3], dtype=float),
                        "QSigma": np.array([0.001, 0.001], dtype=float),
                    }
                )
            },
            stage_bundles={
                "sample_binned": _StageBundle(
                    q_units="1/Angstrom",
                    i_units="1/(cm sr)",
                    description="loaded binned stage",
                )
            },
            yaml_text='sourceQUnits: "1/nm"\nnbins: 100\n',
            preset_name="quickstart.yaml",
            source_file="/tmp/example.dat",
            selected_stage="sample_binned",
        )
        self._metadata = {
            "radius": ("nm", 1.0, 500.0),
            "scale": ("", 0.1, 10.0),
            "radius_pd_type": ("", None, None),
        }

    def _parameter_export_metadata(self, param_name: str) -> tuple[str, float | None, float | None]:
        return self._metadata.get(param_name, ("", None, None))


def _build_snapshot() -> object:
    return build_export_snapshot(
        SnapshotInputs(
            model_expression="sphere",
            q_unit="1/nm",
            intensity_unit="1/(m sr)",
            parameters={"radius": 42.0, "scale": 1.0, "radius_pd_type": "gaussian"},
            fit_parameter_names=["radius"],
            q_values=np.array([0.05, 0.1, 0.2], dtype=float),
            model_intensity=np.array([15.0, 12.0, 8.0], dtype=float),
            overlay_data=None,
            metadata={"demo": True},
        )
    )


def test_hdf5_state_schema_writes_dataset_units_and_uncertainty_links(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    app = _FakeApp()
    snapshot = _build_snapshot()
    output = tmp_path / "state_contract.h5"

    with h5py.File(output, "w") as h5f:
        app._write_state_hdf5_tree(h5f, snapshot)

    with h5py.File(output, "r") as h5f:
        model_group = h5f["/analyses/SasModelsExplorer/model"]
        assert model_group["Q"].attrs["units"] == "1/nm"
        assert model_group["I"].attrs["units"] == "1/(m sr)"
        assert "Q_unit" not in model_group.attrs
        assert "I_unit" not in model_group.attrs

        parameters_group = h5f["/analyses/SasModelsExplorer/parameters"]
        assert "parameter_name" not in parameters_group["radius"].attrs
        assert parameters_group["radius"].attrs["units"] == "nm"
        assert float(parameters_group["radius"].attrs["min"]) == 1.0
        assert float(parameters_group["radius"].attrs["max"]) == 500.0

        stage_group = h5f["/analyses/SasModelsExplorer/data/sample_binned"]
        assert stage_group["Q"].attrs["units"] == "1/Angstrom"
        assert stage_group["QSigma"].attrs["units"] == "1/Angstrom"
        assert stage_group["I"].attrs["units"] == "1/(cm sr)"
        assert stage_group["ISigma"].attrs["units"] == "1/(cm sr)"
        assert stage_group["I"].attrs["uncertainties"] == "ISigma"
        assert stage_group["Q"].attrs["uncertainties"] == "QSigma"

        settings_group = h5f["/analyses/SasModelsExplorer/SME_settings"]
        assert settings_group["data_loading_yaml"][()].decode("utf-8").startswith('sourceQUnits: "1/nm"')
        assert settings_group["data_loading_preset"][()].decode("utf-8") == "quickstart.yaml"


def test_hdf5_state_read_new_schema_restores_yaml_preset_and_units(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    app = _FakeApp()
    snapshot = _build_snapshot()
    output = tmp_path / "state_new_read.h5"

    with h5py.File(output, "w") as h5f:
        app._write_state_hdf5_tree(h5f, snapshot)

    with h5py.File(output, "r") as h5f:
        state = app._read_new_state_hdf5(h5f, str(output), _DummyBaseData, _DummyDataBundle)

    assert state["yaml_text"] == 'sourceQUnits: "1/nm"\nnbins: 100\n'
    assert state["yaml_preset"] == "quickstart.yaml"
    assert state["stage_name"] == "sample_binned"
    stage_bundles = state["stage_bundles"]
    assert "sample_binned" in stage_bundles
    bundle = stage_bundles["sample_binned"]
    assert bundle["Q"].units == "1/angstrom"
    assert bundle["signal"].units == "1/(cm sr)"
    assert "ISigma" in bundle["signal"].uncertainties


def test_hdf5_state_rejects_legacy_schema(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    app = _FakeApp()
    output = tmp_path / "state_legacy_read.h5"

    with h5py.File(output, "w") as h5f:
        h5f.attrs["model_expression"] = "sphere"
        h5f.create_dataset("data/Q", data=np.array([0.1, 0.2], dtype=float))
        h5f.create_dataset("data/I", data=np.array([10.0, 9.0], dtype=float))

    with h5py.File(output, "r") as h5f:
        with pytest.raises(ValueError, match="Unsupported state schema"):
            app._read_state_hdf5(h5f, str(output), _DummyBaseData, _DummyDataBundle)

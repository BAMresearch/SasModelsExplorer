"""Integration-style tests from data loading into fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ModelExplorer.services import data_loader
from ModelExplorer.services.fitting_engine import fit_model
from ModelExplorer.types import OverlayData


class DummyBaseData:
    def __init__(self, signal, units, uncertainties=None):
        self.signal = np.asarray(signal, dtype=float)
        self.units = units
        self.uncertainties = uncertainties or {}


class DummyDataBundle(dict):
    description = None
    default_plot = "signal"


class DummyProcessingBackend:
    def __init__(self, processing):
        self.processing = processing

    def prepare_processing_from_file(self, data_path, *, result_index, workflow_config):
        _ = data_path, result_index, workflow_config
        return self.processing

    def selected_bundle_from_processing(self, processing, *, stage_name):
        return processing[stage_name]

    def frame_from_bundle(self, bundle):
        return pd.DataFrame(
            {
                "Q": np.asarray(bundle["Q"].signal, dtype=float),
                "I": np.asarray(bundle["signal"].signal, dtype=float),
                "ISigma": np.asarray(next(iter(bundle["signal"].uncertainties.values())), dtype=float),
            }
        )


class DummyModel:
    def make_kernel(self, data):
        return data[0]


class DummyModelInfo:
    def __init__(self):
        class Params:
            pd_1d = []

        self.parameters = Params()


class DummyParameter:
    def __init__(self, default, limits):
        self.default = default
        self.limits = limits


def _dummy_bundle(q, i, sigma):
    bundle = DummyDataBundle()
    bundle["signal"] = DummyBaseData(i, "1/(m sr)", uncertainties={"propagate_to_all": np.asarray(sigma)})
    bundle["Q"] = DummyBaseData(q, "1/nm", uncertainties={})
    return bundle


def test_data_to_fit_integration(tmp_path, monkeypatch):
    pytest.importorskip("scipy", reason="scipy not installed")
    data_path = tmp_path / "data.dat"
    data_path.write_text("dummy")

    q_vals = np.array([0.1, 0.2, 0.3], dtype=float)
    i_vals = 2.0 * q_vals
    sigma = np.full_like(q_vals, 0.01)
    backend = DummyProcessingBackend({"sample_binned": _dummy_bundle(q_vals, i_vals, sigma)})

    bundle, used_kind, count = data_loader.load_data_bundle(
        data_path,
        "sample_binned",
        "sourceQUnits: 1/Angstrom\nsourceIntensityUnits: 1/(m sr)",
        backend,
    )

    assert used_kind == "sample_binned"
    assert count == 3

    overlay = OverlayData(
        Q=bundle["Q"].signal,
        I=bundle["signal"].signal,
        ISigma=next(iter(bundle["signal"].uncertainties.values())),
        label="data",
    )

    def fake_compute_intensity(kernel, parameters):
        return parameters["scale"] * kernel

    import ModelExplorer.sasmodels_adapter as sasmodels_adapter

    monkeypatch.setattr(sasmodels_adapter, "compute_intensity", fake_compute_intensity)

    result = fit_model(
        model=DummyModel(),
        model_info=DummyModelInfo(),
        parameters={"scale": 1.0},
        fit_names=["scale"],
        parameter_defs={"scale": DummyParameter(1.0, (0.0, 10.0))},
        data=overlay,
        q_unit="1/Angstrom",
        max_nfev=50,
        intensity_scale=1.0,
    )

    assert result.success
    assert abs(result.parameters["scale"] - 2.0) < 1e-2

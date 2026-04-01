"""Unit tests for canonical data-loader helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ModelExplorer.services import data_loader


class DummyBaseData:
    def __init__(self, signal, units, uncertainties=None):
        self.signal = np.asarray(signal, dtype=float)
        self.units = units
        self.uncertainties = uncertainties or {}


class DummyDataBundle(dict):
    description = None
    default_plot = "signal"


def _dummy_bundle(q, i, sigma=None):
    signal_unc = {}
    if sigma is not None:
        signal_unc["propagate_to_all"] = np.asarray(sigma, dtype=float)
    bundle = DummyDataBundle()
    bundle["signal"] = DummyBaseData(i, "1/(m sr)", uncertainties=signal_unc)
    bundle["Q"] = DummyBaseData(q, "1/nm", uncertainties={})
    return bundle


class DummyProcessingBackend:
    def __init__(self, processing):
        self.processing = processing
        self.last_result_index = None
        self.last_workflow_config = None

    def prepare_processing_from_file(self, data_path, *, result_index, workflow_config):
        self.last_result_index = result_index
        self.last_workflow_config = dict(workflow_config)
        return self.processing

    def selected_bundle_from_processing(self, processing, *, stage_name):
        return processing[stage_name]

    def frame_from_bundle(self, bundle):
        frame = {
            "Q": np.asarray(bundle["Q"].signal, dtype=float),
            "I": np.asarray(bundle["signal"].signal, dtype=float),
        }
        uncertainties = bundle["signal"].uncertainties
        if uncertainties:
            frame["ISigma"] = np.asarray(next(iter(uncertainties.values())), dtype=float)
        return pd.DataFrame(frame)


def test_parse_yaml_config_defaults():
    config = data_loader.parse_yaml_config("")
    assert config.nbins == 100
    assert config.Q_unit == "1/nm"
    assert config.I_unit == "1/(m sr)"


def test_parse_yaml_config_values():
    yaml_text = """
nbins: 10
Q_unit: 1/Angstrom
I_unit: 1/(cm sr)
IEmin: 0.02
omitQRanges: [[0.1, 0.2]]
"""
    config = data_loader.parse_yaml_config(yaml_text)
    assert config.nbins == 10
    assert config.Q_unit == "1/angstrom"
    assert config.I_unit == "1/(cm sr)"
    assert config.IEmin == 0.02
    assert config.omitQRanges == [[0.1, 0.2]]


def test_load_data_bundle_builds_bundle(tmp_path):
    data_path = tmp_path / "data.dat"
    data_path.write_text("dummy")
    stage_bundle = _dummy_bundle(
        q=[0.1, 0.2, 0.3],
        i=[1.0, 2.0, 3.0],
        sigma=[0.1, 0.2, 0.3],
    )
    backend = DummyProcessingBackend({"sample_binned": stage_bundle})

    bundle, used_kind, count = data_loader.load_data_bundle(
        data_path,
        "sample_binned",
        "",
        backend,
    )

    assert used_kind == "sample_binned"
    assert count == 3
    assert bundle is stage_bundle
    assert backend.last_result_index == 1
    np.testing.assert_allclose(bundle["signal"].signal, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(bundle["Q"].signal, [0.1, 0.2, 0.3])


def test_load_data_bundle_uses_stage_fallback(tmp_path):
    data_path = tmp_path / "data.dat"
    data_path.write_text("dummy")
    backend = DummyProcessingBackend({"sample_binned": _dummy_bundle(q=[0.1, 0.2], i=[1.0, 2.0], sigma=[0.1, 0.1])})

    _, used_kind, count = data_loader.load_data_bundle(
        data_path,
        "sample_raw",
        "",
        backend,
    )

    assert used_kind == "sample_binned"
    assert count == 2


def test_load_data_bundle_accepts_legacy_stage_alias(tmp_path):
    data_path = tmp_path / "data.dat"
    data_path.write_text("dummy")
    backend = DummyProcessingBackend({"sample_binned": _dummy_bundle(q=[0.1], i=[1.0], sigma=[0.1])})

    _, used_kind, count = data_loader.load_data_bundle(
        data_path,
        "binnedData",
        "",
        backend,
    )

    assert used_kind == "sample_binned"
    assert count == 1


def test_load_data_bundle_errors_on_missing_columns(tmp_path):
    data_path = tmp_path / "data.dat"
    data_path.write_text("dummy")
    stage_bundle = _dummy_bundle(q=[0.1, 0.2], i=[1.0, 2.0], sigma=[0.1, 0.2])
    backend = DummyProcessingBackend({"sample_binned": stage_bundle})

    def bad_frame(bundle):
        return pd.DataFrame({"Q": bundle["Q"].signal})

    backend.frame_from_bundle = bad_frame

    with pytest.raises(ValueError):
        data_loader.load_data_bundle(
            data_path,
            "sample_binned",
            "",
            backend,
        )

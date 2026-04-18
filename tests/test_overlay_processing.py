"""Tests for overlay conversion and chi-square helpers."""

from __future__ import annotations

import numpy as np

from ModelExplorer.services.overlay_processing import overlay_from_bundle, reduced_chi_square


class DummyBaseData:
    def __init__(self, signal, units, uncertainties=None):
        self.signal = np.asarray(signal, dtype=float)
        self.units = units
        self.uncertainties = uncertainties or {}

    def to_units(self, units: str) -> None:
        _ = units


class UnitTrackingData(DummyBaseData):
    calls: list[str] = []

    def to_units(self, units: str) -> None:
        UnitTrackingData.calls.append(units)


def test_overlay_from_bundle_combines_uncertainty_keys() -> None:
    bundle = {
        "signal": DummyBaseData([2.0, 1.0], "1/(m sr)", uncertainties={"a": [0.3, 0.4], "b": [0.4, 0.3]}),
        "Q": DummyBaseData([0.2, 0.1], "1/nm", uncertainties={}),
    }
    overlay = overlay_from_bundle(bundle, "1/nm", "1/(m sr)")
    assert overlay is not None
    np.testing.assert_allclose(overlay.Q, [0.1, 0.2])
    np.testing.assert_allclose(overlay.I, [1.0, 2.0])
    np.testing.assert_allclose(overlay.ISigma, [0.5, 0.5])


def test_overlay_from_bundle_normalizes_q_unit_for_conversion() -> None:
    UnitTrackingData.calls.clear()
    bundle = {
        "signal": UnitTrackingData([2.0, 1.0], "1/(m sr)", uncertainties={"ISigma": [0.4, 0.3]}),
        "Q": UnitTrackingData([0.2, 0.1], "1/nm", uncertainties={}),
    }
    overlay = overlay_from_bundle(bundle, "1/\u00c5ngstr\u00f6m", "1/(m sr)")
    assert overlay is not None
    assert "1/angstrom" in UnitTrackingData.calls


def test_reduced_chi_square_returns_expected_values() -> None:
    bundle = {
        "signal": DummyBaseData([2.0, 4.0, 8.0], "1/(m sr)", uncertainties={"ISigma": [0.1, 0.1, 0.1]}),
        "Q": DummyBaseData([0.1, 0.2, 0.4], "1/nm", uncertainties={}),
    }
    overlay = overlay_from_bundle(bundle, "1/nm", "1/(m sr)")
    assert overlay is not None

    model_q = np.array([0.1, 0.2, 0.4], dtype=float)
    model_i = np.array([2.0, 4.0, 8.0], dtype=float)
    chi2, dof, points = reduced_chi_square(model_q, model_i, overlay, n_parameters=1, intensity_scale=1.0)
    assert chi2 is not None and chi2 < 1e-12
    assert dof == 2
    assert points == 3

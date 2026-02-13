import numpy as np
import pytest

from ModelExplorer.services import data_loader
from ModelExplorer.services.fitting_engine import fit_model
from ModelExplorer.types import OverlayData


class DummyMcData1D:
    def __init__(self, filename, nbins, csvargs, pathDict, IEmin, dataRange, omitQRanges, resultIndex):
        self.filename = filename
        self.nbins = nbins
        self.csvargs = csvargs
        self.pathDict = pathDict
        self.IEmin = IEmin
        self.dataRange = dataRange
        self.omitQRanges = omitQRanges
        self.resultIndex = resultIndex
        self.rawData = None
        self.clippedData = None
        self.binnedData = None


class DummyBaseData:
    def __init__(self, signal, units, uncertainties, rank_of_data):
        self.signal = signal
        self.units = units
        self.uncertainties = uncertainties
        self.rank_of_data = rank_of_data


class DummyDataBundle(dict):
    description = None
    default_plot = None


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


def test_data_to_fit_integration(tmp_path, monkeypatch):
    pytest.importorskip("scipy", reason="scipy not installed")
    data_path = tmp_path / "data.dat"
    data_path.write_text("dummy")

    mds = DummyMcData1D(
        filename=data_path,
        nbins=100,
        csvargs={},
        pathDict=None,
        IEmin=0.01,
        dataRange=[-np.inf, np.inf],
        omitQRanges=[],
        resultIndex=1,
    )

    Q_vals = np.array([0.1, 0.2, 0.3], dtype=float)
    I_vals = 2.0 * Q_vals
    sigma = np.full_like(Q_vals, 0.01)
    mds.binnedData = {"Q": Q_vals, "I": I_vals, "ISigma": sigma}

    def mc_factory(**kwargs):
        return mds

    bundle, used_kind, count = data_loader.load_data_bundle(
        data_path,
        "binnedData",
        "Q_unit: 1/Angstrom\nI_unit: 1/(m sr)",
        mc_factory,
        DummyBaseData,
        DummyDataBundle,
    )

    assert used_kind == "binnedData"
    assert count == 3

    overlay = OverlayData(
        Q=bundle["Q"].signal,
        I=bundle["I"].signal,
        ISigma=bundle["I"].uncertainties["ISigma"],
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

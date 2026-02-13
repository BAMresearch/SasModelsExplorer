import numpy as np
import pytest

from ModelExplorer.services import data_loader


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
    assert config.Q_unit == "1/Angstrom"
    assert config.I_unit == "1/(cm sr)"
    assert config.IEmin == 0.02
    assert config.omitQRanges == [[0.1, 0.2]]


def test_load_data_bundle_builds_bundle(tmp_path):
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
    mds.binnedData = {
        "Q": np.array([0.1, 0.2, 0.3]),
        "I": np.array([1.0, 2.0, 3.0]),
        "ISigma": np.array([0.1, 0.2, 0.3]),
    }

    def mc_factory(**kwargs):
        return mds

    bundle, used_kind, count = data_loader.load_data_bundle(
        data_path,
        "binnedData",
        "",
        mc_factory,
        DummyBaseData,
        DummyDataBundle,
    )

    assert used_kind == "binnedData"
    assert count == 3
    assert bundle.default_plot == "I"
    assert "I" in bundle and "Q" in bundle
    np.testing.assert_allclose(bundle["I"].signal, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(bundle["Q"].signal, [0.1, 0.2, 0.3])


def test_load_data_bundle_errors_on_missing_columns(tmp_path):
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
    mds.binnedData = {"Q": np.array([0.1, 0.2])}

    def mc_factory(**kwargs):
        return mds

    with pytest.raises(ValueError):
        data_loader.load_data_bundle(
            data_path,
            "binnedData",
            "",
            mc_factory,
            DummyBaseData,
            DummyDataBundle,
        )

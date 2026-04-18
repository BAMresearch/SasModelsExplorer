import numpy as np

from ModelExplorer.services.fitting_engine import fit_model
from ModelExplorer.types import OverlayData


class DummyParameter:
    def __init__(self, default, limits):
        self.default = default
        self.limits = limits


class DummyModel:
    def make_kernel(self, data):
        return data


class DummyModelInfo:
    def __init__(self):
        class Params:
            pd_1d = []

        self.parameters = Params()


def test_fit_model_requires_uncertainties():
    overlay = OverlayData(Q=np.array([0.1, 0.2]), I=np.array([1.0, 2.0]), ISigma=None, label="data")
    result = fit_model(
        model=DummyModel(),
        model_info=DummyModelInfo(),
        parameters={"scale": 1.0},
        fit_names=["scale"],
        parameter_defs={"scale": DummyParameter(1.0, (0.0, 10.0))},
        data=overlay,
        q_unit="1/nm",
        max_nfev=10,
    )
    assert not result.success
    assert "uncertainties" in result.message


def test_fit_model_no_params():
    overlay = OverlayData(Q=np.array([0.1, 0.2]), I=np.array([1.0, 2.0]), ISigma=np.array([0.1, 0.2]), label="data")
    result = fit_model(
        model=DummyModel(),
        model_info=DummyModelInfo(),
        parameters={},
        fit_names=[],
        parameter_defs={},
        data=overlay,
        q_unit="1/nm",
        max_nfev=10,
    )
    assert not result.success
    assert "Select parameters" in result.message

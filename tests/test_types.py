import numpy as np

from ModelExplorer.types import DataConfig, OverlayData


def test_data_config_defaults():
    cfg = DataConfig()
    assert cfg.nbins == 100
    assert cfg.IEmin == 0.01
    assert cfg.Q_unit == "1/nm"
    assert cfg.I_unit == "1/(m sr)"
    assert cfg.dataRange[0] == -np.inf
    assert cfg.dataRange[1] == np.inf


def test_overlay_data_fields():
    data = OverlayData(Q=np.array([1.0]), I=np.array([2.0]), ISigma=None, label="demo")
    assert data.label == "demo"
    assert data.I.shape == (1,)

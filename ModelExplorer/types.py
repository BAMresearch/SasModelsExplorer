# ModelExplorer/types.py

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DataConfig:
    nbins: int = 100
    csvargs: dict = field(default_factory=dict)
    pathDict: Optional[dict] = None
    IEmin: float = 0.01
    dataRange: list = field(default_factory=lambda: [-np.inf, np.inf])
    omitQRanges: list = field(default_factory=list)
    resultIndex: int = 1
    Q_unit: str = "1/nm"
    I_unit: str = "1/(m sr)"


@dataclass
class OverlayData:
    Q: np.ndarray
    I: np.ndarray  # noqa: E741
    ISigma: Optional[np.ndarray]
    label: str

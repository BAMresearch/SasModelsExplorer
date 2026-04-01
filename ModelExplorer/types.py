# ModelExplorer/types.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ParameterValue: TypeAlias = float | int | str
ParameterMapping: TypeAlias = dict[str, ParameterValue]
MetadataValue: TypeAlias = str | int | float | bool
MetadataMapping: TypeAlias = dict[str, MetadataValue]


@dataclass(slots=True)
class DataConfig:
    """Configuration used for canonical McSAS3 data preparation."""

    nbins: int = 100
    csvargs: dict[str, object] = field(default_factory=dict)
    pathDict: dict[str, str] | None = None
    IEmin: float = 0.01
    dataRange: list[float] = field(default_factory=lambda: [-np.inf, np.inf])
    omitQRanges: list[list[float]] = field(default_factory=list)
    resultIndex: int = 1
    Q_unit: str = "1/nm"
    I_unit: str = "1/(m sr)"


@dataclass(slots=True)
class OverlayData:
    """Overlay data used for plotting and fitting against the model curve."""

    Q: FloatArray
    I: FloatArray  # noqa: E741
    ISigma: FloatArray | None
    label: str


@dataclass(slots=True)
class ModelSessionState:
    """Serializable model session state for export and persistence workflows."""

    model_expression: str
    q_unit: str
    q_min: float
    q_max: float
    parameters: ParameterMapping
    hidden_defaults: ParameterMapping = field(default_factory=dict)


@dataclass(slots=True)
class DataSelectionState:
    """Serializable data-selection state for export and persistence workflows."""

    source_file: Path | None
    stage_name: str
    points_loaded: int
    description: str = ""

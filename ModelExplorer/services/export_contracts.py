"""Typed export payload contracts for upcoming UI export actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ModelExplorer.types import MetadataMapping, OverlayData, ParameterMapping


@dataclass(slots=True, frozen=True)
class ExportSnapshot:
    """Single source of truth for all non-UI export pathways."""

    model_expression: str
    q_unit: str
    intensity_unit: str
    parameters: ParameterMapping
    fit_parameter_names: tuple[str, ...]
    q_values: NDArray[np.float64]
    model_intensity: NDArray[np.float64]
    overlay_data: OverlayData | None = None
    metadata: MetadataMapping = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RunConfigTemplatePayload:
    """Normalized payload for future McSAS3 run-config template export."""

    model_expression: str
    fit_parameter_names: tuple[str, ...]
    parameters: ParameterMapping
    q_unit: str
    intensity_unit: str
    template: dict[str, object]


@dataclass(slots=True, frozen=True)
class CsvModelParameterTable:
    """Tabular parameter payload for CSV export."""

    headers: tuple[str, str]
    rows: tuple[tuple[str, str], ...]


@dataclass(slots=True, frozen=True)
class Hdf5StatePayload:
    """Dataset/attribute payload for future HDF5 persistence export."""

    datasets: dict[str, NDArray[np.float64]]
    attributes: MetadataMapping


class RunConfigTemplateExporter(Protocol):
    """Exporter protocol for run-config template payloads."""

    def build_payload(self, snapshot: ExportSnapshot) -> RunConfigTemplatePayload:
        """Build a run-template payload from a generic export snapshot."""


class CsvModelExporter(Protocol):
    """Exporter protocol for CSV parameter payloads."""

    def build_payload(self, snapshot: ExportSnapshot) -> CsvModelParameterTable:
        """Build a CSV table payload from a generic export snapshot."""


class Hdf5StateExporter(Protocol):
    """Exporter protocol for HDF5 state payloads."""

    def build_payload(self, snapshot: ExportSnapshot) -> Hdf5StatePayload:
        """Build an HDF5 payload from a generic export snapshot."""

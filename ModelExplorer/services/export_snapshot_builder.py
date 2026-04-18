"""Pure helpers that build export snapshot and payload objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ModelExplorer.types import MetadataValue, OverlayData, ParameterMapping, ParameterValue

from .export_contracts import (
    CsvModelParameterTable,
    ExportSnapshot,
    Hdf5StatePayload,
    RunConfigTemplatePayload,
)


@dataclass(slots=True, frozen=True)
class SnapshotInputs:
    """Input values required to build an immutable ``ExportSnapshot``."""

    model_expression: str
    q_unit: str
    intensity_unit: str
    parameters: Mapping[str, ParameterValue]
    fit_parameter_names: Sequence[str]
    q_values: NDArray[np.float64]
    model_intensity: NDArray[np.float64]
    overlay_data: OverlayData | None = None
    metadata: Mapping[str, MetadataValue] | None = None


def build_export_snapshot(inputs: SnapshotInputs) -> ExportSnapshot:
    """Create a normalized ``ExportSnapshot`` with copied mutable data."""

    metadata = dict(inputs.metadata or {})
    return ExportSnapshot(
        model_expression=inputs.model_expression,
        q_unit=inputs.q_unit,
        intensity_unit=inputs.intensity_unit,
        parameters=dict(inputs.parameters),
        fit_parameter_names=tuple(inputs.fit_parameter_names),
        q_values=np.array(inputs.q_values, copy=True, dtype=float),
        model_intensity=np.array(inputs.model_intensity, copy=True, dtype=float),
        overlay_data=inputs.overlay_data,
        metadata=metadata,
    )


def build_run_config_template_payload(snapshot: ExportSnapshot) -> RunConfigTemplatePayload:
    """Build a normalized McSAS3-template payload from an export snapshot."""

    template: dict[str, object] = {
        "model_name": snapshot.model_expression,
        "fit_parameters": list(snapshot.fit_parameter_names),
        "parameters": dict(snapshot.parameters),
        "units": {
            "Q": snapshot.q_unit,
            "I": snapshot.intensity_unit,
        },
    }
    return RunConfigTemplatePayload(
        model_expression=snapshot.model_expression,
        fit_parameter_names=snapshot.fit_parameter_names,
        parameters=dict(snapshot.parameters),
        q_unit=snapshot.q_unit,
        intensity_unit=snapshot.intensity_unit,
        template=template,
    )


def build_csv_model_parameter_table(snapshot: ExportSnapshot) -> CsvModelParameterTable:
    """Build a two-column CSV payload with parameter name/value pairs."""

    rows = tuple((name, str(value)) for name, value in sorted(snapshot.parameters.items()))
    return CsvModelParameterTable(headers=("parameter", "value"), rows=rows)


def build_hdf5_state_payload(snapshot: ExportSnapshot) -> Hdf5StatePayload:
    """Build an HDF5-friendly payload from a normalized export snapshot."""

    datasets: dict[str, NDArray[np.float64]] = {
        "model/Q": np.array(snapshot.q_values, copy=True, dtype=float),
        "model/I": np.array(snapshot.model_intensity, copy=True, dtype=float),
    }
    if snapshot.overlay_data is not None:
        datasets["data/Q"] = np.array(snapshot.overlay_data.Q, copy=True, dtype=float)
        datasets["data/I"] = np.array(snapshot.overlay_data.I, copy=True, dtype=float)
        if snapshot.overlay_data.ISigma is not None:
            datasets["data/ISigma"] = np.array(snapshot.overlay_data.ISigma, copy=True, dtype=float)

    attributes = dict(snapshot.metadata)
    attributes["model_expression"] = snapshot.model_expression
    attributes["q_unit"] = snapshot.q_unit
    attributes["intensity_unit"] = snapshot.intensity_unit
    attributes["fit_parameter_names"] = ",".join(snapshot.fit_parameter_names)
    return Hdf5StatePayload(datasets=datasets, attributes=attributes)

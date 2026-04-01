"""Typed HDF5 state read/write helpers for SasModelsExplorer."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np

from ModelExplorer.services.export_contracts import ExportSnapshot
from ModelExplorer.services.overlay_processing import BaseDataLike
from ModelExplorer.types import ParameterMapping, ParameterValue
from ModelExplorer.utils.units import normalize_unit_label


class H5AttrsLike(Protocol):
    """Minimal mapping-like surface of HDF5 attributes."""

    def get(self, key: str, default: object | None = None) -> object:
        """Return an attribute value with fallback default."""

    def __contains__(self, key: object) -> bool:
        """Return whether an attribute key exists."""

    def __setitem__(self, key: str, value: object) -> None:
        """Set an attribute key/value pair."""


class H5DatasetLike(Protocol):
    """Minimal HDF5 dataset surface used by state IO."""

    attrs: H5AttrsLike

    def __getitem__(self, key: object) -> object:
        """Return dataset contents for key/index payloads."""


class H5GroupLike(Protocol):
    """Minimal HDF5 group/file surface used by state IO."""

    attrs: H5AttrsLike

    def create_group(self, name: str) -> "H5GroupLike":
        """Create and return a nested group."""

    def create_dataset(
        self,
        name: str,
        data: object,
        dtype: object | None = None,
    ) -> H5DatasetLike:
        """Create and return a dataset."""

    def get(self, name: str) -> object | None:
        """Return nested node by key, or ``None`` when missing."""

    def __getitem__(self, key: str) -> object:
        """Return a nested node by key."""

    def __contains__(self, key: object) -> bool:
        """Return whether a key exists."""

    def keys(self) -> Iterable[str]:
        """Return group keys."""


class BaseDataFactory(Protocol):
    """Callable factory protocol for constructing BaseData-like payloads."""

    def __call__(
        self,
        *,
        signal: object,
        units: str,
        uncertainties: Mapping[str, object] | None = None,
    ) -> BaseDataLike:
        """Build a BaseData-like object."""


class DataBundleLike(Protocol):
    """Mapping-like DataBundle with writable description."""

    description: str

    def __getitem__(self, key: str) -> BaseDataLike:
        """Return entry by key."""

    def __iter__(self) -> Iterable[str]:
        """Iterate over available keys."""

    def __len__(self) -> int:
        """Return number of keys."""

    def get(self, key: str, default: BaseDataLike | None = None) -> BaseDataLike | None:
        """Return entry by key with fallback default."""


class DataBundleFactory(Protocol):
    """Callable factory protocol for constructing DataBundle-like payloads."""

    def __call__(self, values: Mapping[str, BaseDataLike]) -> DataBundleLike:
        """Build a DataBundle-like mapping."""


class FrameLike(Protocol):
    """Minimal dataframe-like protocol used for HDF5 stage export."""

    columns: Iterable[str]

    def __getitem__(self, key: str) -> object:
        """Return the selected column payload."""


StageBundleMap: TypeAlias = dict[str, DataBundleLike]
ParameterMetadataGetter: TypeAlias = Callable[[str], tuple[str, float | None, float | None]]


@dataclass(slots=True, frozen=True)
class StateWriteContext:
    """Normalized non-UI context payload used for writing state trees."""

    snapshot: ExportSnapshot
    q_min: float
    q_max: float
    selected_data_stage: str
    fit_parameter_names: Sequence[str]
    hidden_parameter_defaults: Mapping[str, ParameterValue]
    data_loading_yaml: str
    data_loading_preset: str | None
    data_source_file: str | None
    data_mode_label: str
    stage_bundles: Mapping[str, object]
    stage_frames: Mapping[str, object]
    exporter: str = "SasModelsExplorer"
    schema_version: int = 2


@dataclass(slots=True)
class LoadedExplorerState:
    """Typed state payload loaded from an explorer HDF5 file."""

    model_expression: str
    q_unit: str
    intensity_unit: str
    q_min: float
    q_max: float
    parameters: ParameterMapping
    hidden_defaults: ParameterMapping
    fit_names: list[str]
    source_file: str
    stage_name: str
    yaml_text: str
    yaml_preset: str
    stage_bundles: StageBundleMap

    def __getitem__(self, key: str) -> object:
        """Provide backward-compatible dict-like state access by key."""

        return getattr(self, key)

    def get(self, key: str, default: object | None = None) -> object | None:
        """Provide backward-compatible ``dict.get`` style state access."""

        return getattr(self, key, default)


def dataset_scalar_to_python(value: object) -> ParameterValue:
    """Normalize HDF5 scalar payload values to Python scalars."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def dataset_from_group(group: H5GroupLike | None, name: str) -> H5DatasetLike | None:
    """Return a named dataset from a group, or ``None`` when missing."""

    if group is None:
        return None
    node = group.get(name)
    if node is None:
        return None
    return cast(H5DatasetLike, node)


def dataset_text(group: H5GroupLike | None, name: str, default: str = "") -> str:
    """Read a string-like dataset value from a group with fallback default."""

    dataset = dataset_from_group(group, name)
    if dataset is None:
        return default
    try:
        raw = dataset[()]
    except Exception:
        return default
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def dataset_float(group: H5GroupLike | None, name: str, default: float) -> float:
    """Read a float dataset value with fallback default."""

    dataset = dataset_from_group(group, name)
    if dataset is None:
        return default
    try:
        return float(dataset_scalar_to_python(dataset[()]))
    except Exception:
        return default


def parse_parameter_json(payload: str) -> ParameterMapping:
    """Parse JSON mapping payload and coerce values to export-safe scalars."""

    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}

    parsed: ParameterMapping = {}
    for key, value in raw.items():
        parsed[str(key)] = dataset_scalar_to_python(value)
    return parsed


def parse_string_list_json(payload: str) -> list[str]:
    """Parse a JSON list payload into a list of strings."""

    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _write_model_group(sme_group: H5GroupLike, snapshot: ExportSnapshot) -> None:
    model_group = sme_group.create_group("model")
    model_q_ds = model_group.create_dataset("Q", data=np.array(snapshot.q_values, copy=True, dtype=float))
    model_i_ds = model_group.create_dataset("I", data=np.array(snapshot.model_intensity, copy=True, dtype=float))
    model_q_ds.attrs["units"] = snapshot.q_unit
    model_i_ds.attrs["units"] = snapshot.intensity_unit


def _write_parameters_group(
    sme_group: H5GroupLike,
    snapshot: ExportSnapshot,
    text_dtype: object,
    parameter_metadata_getter: ParameterMetadataGetter,
) -> None:
    parameters_group = sme_group.create_group("parameters")
    for name, value in sorted(snapshot.parameters.items()):
        if isinstance(value, str):
            ds = parameters_group.create_dataset(name, data=value, dtype=text_dtype)
        else:
            ds = parameters_group.create_dataset(name, data=dataset_scalar_to_python(value))

        units, min_val, max_val = parameter_metadata_getter(name)
        ds.attrs["units"] = units
        if min_val is not None:
            ds.attrs["min"] = float(min_val)
        if max_val is not None:
            ds.attrs["max"] = float(max_val)


def _write_data_group(
    sme_group: H5GroupLike,
    snapshot: ExportSnapshot,
    text_dtype: object,
    stage_bundles: Mapping[str, object],
    stage_frames: Mapping[str, object],
) -> None:
    data_group = sme_group.create_group("data")

    for stage_name, frame_obj in sorted(stage_frames.items()):
        frame = cast(FrameLike, frame_obj)
        stage_group = data_group.create_group(stage_name)
        stage_q_unit = snapshot.q_unit
        stage_i_unit = snapshot.intensity_unit
        stage_bundle_obj = stage_bundles.get(stage_name)
        if stage_bundle_obj is not None:
            stage_bundle = cast(Mapping[str, BaseDataLike], stage_bundle_obj)
            try:
                q_node = stage_bundle.get("Q")
                if q_node is not None:
                    stage_q_unit = str(getattr(q_node, "units", stage_q_unit))
            except Exception:
                pass
            try:
                i_node = stage_bundle.get("signal")
                if i_node is None:
                    i_node = stage_bundle.get("I")
                if i_node is not None:
                    stage_i_unit = str(getattr(i_node, "units", stage_i_unit))
            except Exception:
                pass

        for column_name in frame.columns:
            values = np.asarray(frame[column_name])
            if np.issubdtype(values.dtype, np.number):
                ds = stage_group.create_dataset(column_name, data=np.asarray(values, dtype=float))
            else:
                text_values = np.asarray([str(item) for item in values], dtype=object)
                ds = stage_group.create_dataset(column_name, data=text_values, dtype=text_dtype)

            if column_name in {"Q", "QSigma"}:
                ds.attrs["units"] = stage_q_unit
            elif column_name in {"I", "ISigma"}:
                ds.attrs["units"] = stage_i_unit

        if "I" in stage_group and "ISigma" in stage_group:
            cast(H5DatasetLike, stage_group["I"]).attrs["uncertainties"] = "ISigma"
        if "Q" in stage_group and "QSigma" in stage_group:
            cast(H5DatasetLike, stage_group["Q"]).attrs["uncertainties"] = "QSigma"

        if stage_bundle_obj is not None:
            description = str(getattr(stage_bundle_obj, "description", "") or "")
            if description:
                stage_group.attrs["description"] = description


def _write_settings_group(
    sme_group: H5GroupLike,
    context: StateWriteContext,
    text_dtype: object,
) -> None:
    settings_group = sme_group.create_group("SME_settings")
    settings_group.create_dataset("schema_version", data=int(context.schema_version))
    settings_group.create_dataset("q_unit", data=context.snapshot.q_unit, dtype=text_dtype)
    settings_group.create_dataset("intensity_unit", data=context.snapshot.intensity_unit, dtype=text_dtype)
    settings_group.create_dataset("q_min", data=float(context.q_min))
    settings_group.create_dataset("q_max", data=float(context.q_max))
    settings_group.create_dataset("selected_data_stage", data=context.selected_data_stage, dtype=text_dtype)
    settings_group.create_dataset(
        "fit_parameter_names_json",
        data=json.dumps(list(context.fit_parameter_names)),
        dtype=text_dtype,
    )
    settings_group.create_dataset(
        "hidden_parameter_defaults_json",
        data=json.dumps(dict(context.hidden_parameter_defaults)),
        dtype=text_dtype,
    )
    settings_group.create_dataset("data_loading_yaml", data=context.data_loading_yaml, dtype=text_dtype)
    if context.data_loading_preset:
        settings_group.create_dataset("data_loading_preset", data=context.data_loading_preset, dtype=text_dtype)
    if context.data_source_file:
        settings_group.create_dataset("data_source_file", data=context.data_source_file, dtype=text_dtype)
    settings_group.create_dataset("data_mode_label", data=context.data_mode_label, dtype=text_dtype)
    settings_group.create_dataset("exporter", data=context.exporter, dtype=text_dtype)


def write_state_hdf5_tree(
    h5f: H5GroupLike,
    context: StateWriteContext,
    *,
    parameter_metadata_getter: ParameterMetadataGetter,
) -> None:
    """Write the complete `/analyses/SasModelsExplorer` state tree."""

    import h5py

    text_dtype = h5py.string_dtype(encoding="utf-8")
    analyses_group = h5f.create_group("analyses")
    sme_group = analyses_group.create_group("SasModelsExplorer")
    sme_group.create_dataset("model_name", data=context.snapshot.model_expression, dtype=text_dtype)

    _write_model_group(sme_group, context.snapshot)
    _write_parameters_group(sme_group, context.snapshot, text_dtype, parameter_metadata_getter)
    _write_data_group(
        sme_group,
        context.snapshot,
        text_dtype,
        stage_bundles=context.stage_bundles,
        stage_frames=context.stage_frames,
    )
    _write_settings_group(sme_group, context, text_dtype)


def read_new_state_hdf5(
    h5f: H5GroupLike,
    file_path: str,
    base_data_cls: BaseDataFactory,
    data_bundle_cls: DataBundleFactory,
    *,
    default_intensity_unit: str,
) -> LoadedExplorerState:
    """Read state payload from the `/analyses/SasModelsExplorer` tree."""

    sme_group = cast(H5GroupLike, h5f["/analyses/SasModelsExplorer"])
    settings_group = cast(H5GroupLike | None, sme_group.get("SME_settings"))
    model_group = cast(H5GroupLike | None, sme_group.get("model"))

    model_expression = dataset_text(sme_group, "model_name", "").strip()
    if not model_expression:
        raise ValueError("State file does not contain /analyses/SasModelsExplorer/model_name.")

    model_q_unit = ""
    model_i_unit = ""
    if model_group is not None:
        if "Q" in model_group:
            model_q_unit = str(cast(H5DatasetLike, model_group["Q"]).attrs.get("units", ""))
        if "I" in model_group:
            model_i_unit = str(cast(H5DatasetLike, model_group["I"]).attrs.get("units", ""))

    q_unit = dataset_text(settings_group, "q_unit", model_q_unit or "1/nm")
    intensity_unit = dataset_text(settings_group, "intensity_unit", model_i_unit or default_intensity_unit)
    q_min = dataset_float(settings_group, "q_min", 0.01)
    q_max = dataset_float(settings_group, "q_max", 10.0)

    parameters: ParameterMapping = {}
    parameters_group = cast(H5GroupLike | None, sme_group.get("parameters"))
    if parameters_group is not None:
        for item_name in parameters_group.keys():
            dataset = cast(H5DatasetLike, parameters_group[item_name])
            parameters[str(item_name)] = dataset_scalar_to_python(dataset[()])

    hidden_defaults = parse_parameter_json(dataset_text(settings_group, "hidden_parameter_defaults_json", "{}"))
    fit_names = parse_string_list_json(dataset_text(settings_group, "fit_parameter_names_json", "[]"))
    source_file = dataset_text(settings_group, "data_source_file", "").strip()
    stage_name = dataset_text(settings_group, "selected_data_stage", "").strip()
    yaml_text = dataset_text(settings_group, "data_loading_yaml", "")
    yaml_preset = dataset_text(settings_group, "data_loading_preset", "")

    stage_bundles: StageBundleMap = {}
    data_group = cast(H5GroupLike | None, sme_group.get("data"))
    if data_group is not None:
        for candidate_stage in data_group.keys():
            stage_group = cast(H5GroupLike, data_group[candidate_stage])
            if "Q" not in stage_group or "I" not in stage_group:
                continue
            q_dataset = cast(H5DatasetLike, stage_group["Q"])
            i_dataset = cast(H5DatasetLike, stage_group["I"])
            q_data = np.asarray(q_dataset[()], dtype=float)
            i_data = np.asarray(i_dataset[()], dtype=float)
            if q_data.size == 0 or i_data.size == 0:
                continue
            i_sigma = (
                np.asarray(cast(H5DatasetLike, stage_group["ISigma"])[()], dtype=float)
                if "ISigma" in stage_group
                else None
            )
            stage_q_unit = str(q_dataset.attrs.get("units", q_unit))
            stage_i_unit = str(i_dataset.attrs.get("units", intensity_unit))
            q_obj = base_data_cls(signal=q_data, units=normalize_unit_label(stage_q_unit))
            i_unc = {} if i_sigma is None else {"ISigma": i_sigma}
            i_obj = base_data_cls(signal=i_data, units=stage_i_unit, uncertainties=i_unc)
            bundle = data_bundle_cls({"Q": q_obj, "signal": i_obj})
            description = str(stage_group.attrs.get("description", "")).strip()
            if description:
                bundle.description = description
            else:
                bundle.description = f"Imported from {Path(file_path).name} ({candidate_stage})"
            stage_bundles[str(candidate_stage)] = bundle

    return LoadedExplorerState(
        model_expression=model_expression,
        q_unit=q_unit,
        intensity_unit=intensity_unit,
        q_min=q_min,
        q_max=q_max,
        parameters=parameters,
        hidden_defaults=hidden_defaults,
        fit_names=fit_names,
        source_file=source_file,
        stage_name=stage_name,
        yaml_text=yaml_text,
        yaml_preset=yaml_preset,
        stage_bundles=stage_bundles,
    )


def read_state_hdf5(
    h5f: H5GroupLike,
    file_path: str,
    base_data_cls: BaseDataFactory,
    data_bundle_cls: DataBundleFactory,
    *,
    default_intensity_unit: str,
) -> LoadedExplorerState:
    """Read state payload from the currently supported HDF5 schema."""

    if "/analyses/SasModelsExplorer" not in h5f:
        raise ValueError(
            "Unsupported state schema. Expected group '/analyses/SasModelsExplorer'. "
            "Please re-export state files with the current SasModelsExplorer version."
        )
    return read_new_state_hdf5(
        h5f,
        file_path,
        base_data_cls,
        data_bundle_cls,
        default_intensity_unit=default_intensity_unit,
    )

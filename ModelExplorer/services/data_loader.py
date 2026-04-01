"""Helpers for parsing read-config YAML and selecting canonical data bundles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray
from pandas import DataFrame

from ModelExplorer.types import DataConfig
from ModelExplorer.utils.units import DEFAULT_I_UNIT, DEFAULT_Q_UNIT, normalize_unit_label

from .mcsas3_backend import ProcessingBackend

CANONICAL_STAGE_ALIASES = {
    "rawData": "sample_raw",
    "clippedData": "sample_clipped",
    "binnedData": "sample_binned",
}
CANONICAL_STAGE_FALLBACKS = ("sample_binned", "sample_clipped", "sample_raw")
DEFAULT_NBINS = 100
DEFAULT_IEMIN = 0.01
DEFAULT_RESULT_INDEX = 1
DEFAULT_DATA_RANGE = [-np.inf, np.inf]


def parse_yaml_config(yaml_text: str) -> DataConfig:
    """Parse read-config YAML into a normalized ``DataConfig``."""

    raw = _parse_yaml_mapping(yaml_text)
    q_unit = normalize_unit_label(
        _as_optional_str(raw.get("sourceQUnits"))
        or _as_optional_str(raw.get("QUnits"))
        or _as_optional_str(raw.get("Q_unit"))
        or _as_optional_str(raw.get("q_unit"))
        or DEFAULT_Q_UNIT
    )
    i_unit = normalize_unit_label(
        _as_optional_str(raw.get("sourceIntensityUnits"))
        or _as_optional_str(raw.get("IUnits"))
        or _as_optional_str(raw.get("I_unit"))
        or _as_optional_str(raw.get("i_unit"))
        or DEFAULT_I_UNIT
    )

    return DataConfig(
        nbins=_as_int(raw.get("nbins"), DEFAULT_NBINS),
        csvargs=_as_mapping(raw.get("csvargs")),
        pathDict=_as_path_dict(raw.get("pathDict")),
        IEmin=_as_float(raw.get("IEmin"), DEFAULT_IEMIN),
        dataRange=_as_float_list(raw.get("dataRange"), default=DEFAULT_DATA_RANGE),
        omitQRanges=_as_nested_float_list(raw.get("omitQRanges")),
        resultIndex=_as_int(raw.get("resultIndex"), DEFAULT_RESULT_INDEX),
        Q_unit=q_unit,
        I_unit=i_unit,
    )


def load_data_bundle(
    data_path: Path,
    data_kind: str,
    yaml_text: str,
    backend: ProcessingBackend,
) -> tuple[object, str, int]:
    """Load, select, and validate a canonical bundle for overlay plotting."""

    raw = _parse_yaml_mapping(yaml_text)
    config = parse_yaml_config(yaml_text)
    processing = _load_processing_data(data_path, raw, config, backend)
    bundle, used_kind = _select_bundle(processing, data_kind, backend)
    if bundle is None or used_kind is None:
        raise ValueError("No data available after loading.")

    frame = backend.frame_from_bundle(bundle)
    q_vals, _i_vals, _sigma, _q_sigma = _extract_data_arrays(frame)
    if q_vals.size == 0:
        raise ValueError("No finite data points found.")

    bundle.description = f"{data_path.name} ({used_kind})"  # type: ignore[attr-defined]
    return bundle, used_kind, int(q_vals.size)


def _parse_yaml_mapping(yaml_text: str) -> dict[str, object]:
    """Parse YAML text and enforce mapping-style root."""

    if yaml_text.strip():
        try:
            raw = yaml.safe_load(yaml_text)
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML error: {exc}") from exc
    else:
        raw = {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("YAML configuration must be a mapping.")
    return raw


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except Exception:
            return default
    return default


def _as_float(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return default
    return default


def _as_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


def _as_path_dict(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items()}
    return None


def _as_float_list(value: object, *, default: list[float] | None = None) -> list[float]:
    fallback = list(default or [])
    if value is None:
        return fallback
    if isinstance(value, list):
        converted: list[float] = []
        for item in value:
            try:
                converted.append(float(item))
            except Exception:
                continue
        return converted if len(converted) > 0 else fallback
    return fallback


def _as_nested_float_list(value: object) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    nested: list[list[float]] = []
    for item in value:
        if not isinstance(item, list):
            continue
        nested.append(_as_float_list(item))
    return nested


def _load_processing_data(
    data_path: Path,
    raw: dict[str, object],
    config: DataConfig,
    backend: ProcessingBackend,
) -> object:
    """Build canonical processing data from file and normalized read-config."""

    workflow_config: dict[str, object] = {
        "loader": raw.get("loader", None),
        "csvargs": config.csvargs,
        "pathDict": config.pathDict,
        "IEmin": float(config.IEmin),
        "dataRange": config.dataRange,
        "omitQRanges": config.omitQRanges,
        "nbins": int(config.nbins),
        "sourceQUnits": config.Q_unit,
        "sourceIntensityUnits": config.I_unit,
    }

    qemin = raw.get("qemin", raw.get("QEMin", None))
    if qemin is not None:
        workflow_config["qemin"] = _as_float(qemin, default=0.01)

    analysis_stage = raw.get("analysisStage", None)
    if analysis_stage is not None:
        workflow_config["analysisStage"] = _canonical_stage_name(str(analysis_stage))

    return backend.prepare_processing_from_file(
        data_path,
        result_index=int(config.resultIndex),
        workflow_config=workflow_config,
    )


def _canonical_stage_name(stage_name: str) -> str:
    """Return canonical stage names while preserving legacy aliases."""

    return CANONICAL_STAGE_ALIASES.get(stage_name, stage_name)


def _select_bundle(
    processing: object,
    data_kind: str,
    backend: ProcessingBackend,
) -> tuple[object | None, str | None]:
    """Select requested stage bundle with canonical fallback ordering."""

    requested_stage = _canonical_stage_name(data_kind)
    lookup_stages = [requested_stage]
    lookup_stages.extend(stage for stage in CANONICAL_STAGE_FALLBACKS if stage not in lookup_stages)

    for stage_name in lookup_stages:
        try:
            bundle = backend.selected_bundle_from_processing(processing, stage_name=stage_name)
            frame = backend.frame_from_bundle(bundle)
        except Exception:
            continue
        if len(frame) > 0:
            return bundle, stage_name
    return None, None


def _extract_data_arrays(
    data_df: DataFrame,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    """Extract sorted finite Q/I/sigma arrays from a stage dataframe."""

    if "Q" not in data_df or "I" not in data_df:
        raise ValueError("Data frame must include 'Q' and 'I' columns.")

    q_vals = np.asarray(data_df["Q"], dtype=float)
    i_vals = np.asarray(data_df["I"], dtype=float)

    sigma: NDArray[np.float64] | None = None
    for key in ("ISigma", "IError", "IStd", "ISEM"):
        if key in data_df:
            sigma = np.asarray(data_df[key], dtype=float)
            break

    q_sigma: NDArray[np.float64] | None = None
    for key in ("QSigma", "QError", "QStd", "QSEM"):
        if key in data_df:
            q_sigma = np.asarray(data_df[key], dtype=float)
            break

    mask = np.isfinite(q_vals) & np.isfinite(i_vals)
    if sigma is not None:
        mask &= np.isfinite(sigma)
    if q_sigma is not None:
        mask &= np.isfinite(q_sigma)

    q_vals = q_vals[mask]
    i_vals = i_vals[mask]
    if sigma is not None:
        sigma = sigma[mask]
    if q_sigma is not None:
        q_sigma = q_sigma[mask]

    order = np.argsort(q_vals)
    q_vals = q_vals[order]
    i_vals = i_vals[order]
    if sigma is not None:
        sigma = sigma[order]
    if q_sigma is not None:
        q_sigma = q_sigma[order]

    return q_vals, i_vals, sigma, q_sigma

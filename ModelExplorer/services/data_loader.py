# ModelExplorer/services/data_loader.py

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import yaml

from ModelExplorer.types import DataConfig
from ModelExplorer.utils.units import DEFAULT_I_UNIT, DEFAULT_Q_UNIT, normalize_unit_label

CANONICAL_STAGE_ALIASES = {
    "rawData": "sample_raw",
    "clippedData": "sample_clipped",
    "binnedData": "sample_binned",
}
CANONICAL_STAGE_FALLBACKS = ("sample_binned", "sample_clipped", "sample_raw")


def parse_yaml_config(yaml_text: str) -> DataConfig:
    raw = _parse_yaml_mapping(yaml_text)
    q_unit = normalize_unit_label(
        raw.get("sourceQUnits") or raw.get("QUnits") or raw.get("Q_unit") or raw.get("q_unit") or DEFAULT_Q_UNIT
    )
    i_unit = normalize_unit_label(
        raw.get("sourceIntensityUnits") or raw.get("IUnits") or raw.get("I_unit") or raw.get("i_unit") or DEFAULT_I_UNIT
    )

    cfg = DataConfig(
        nbins=int(raw.get("nbins", DataConfig.nbins)),
        csvargs=raw.get("csvargs", {}) or {},
        pathDict=raw.get("pathDict", None),
        IEmin=float(raw.get("IEmin", DataConfig.IEmin)),
        dataRange=raw.get("dataRange", [-np.inf, np.inf]) or [-np.inf, np.inf],
        omitQRanges=raw.get("omitQRanges", []) or [],
        resultIndex=int(raw.get("resultIndex", DataConfig.resultIndex)),
        Q_unit=q_unit,
        I_unit=i_unit,
    )
    return cfg


def load_data_bundle(
    data_path: Path,
    data_kind: str,
    yaml_text: str,
    prepare_processing_from_file: Any,
    selected_bundle_from_processing: Any,
    frame_from_bundle: Any,
) -> Tuple[Any, str, int]:
    raw = _parse_yaml_mapping(yaml_text)
    config = parse_yaml_config(yaml_text)
    processing = _load_processing_data(data_path, raw, config, prepare_processing_from_file)
    bundle, used_kind = _select_bundle(processing, data_kind, selected_bundle_from_processing, frame_from_bundle)
    if bundle is None or used_kind is None:
        raise ValueError("No data available after loading.")

    Q_vals, _I_vals, _sigma, _q_sigma = _extract_data_arrays(frame_from_bundle(bundle))
    if Q_vals.size == 0:
        raise ValueError("No finite data points found.")

    bundle.description = f"{data_path.name} ({used_kind})"
    return bundle, used_kind, Q_vals.size


def _parse_yaml_mapping(yaml_text: str) -> dict[str, Any]:
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


def _load_processing_data(
    data_path: Path,
    raw: dict[str, Any],
    config: DataConfig,
    prepare_processing_from_file: Any,
) -> Any:
    workflow_config = {
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
        workflow_config["qemin"] = float(qemin)
    analysis_stage = raw.get("analysisStage", None)
    if analysis_stage is not None:
        workflow_config["analysisStage"] = _canonical_stage_name(str(analysis_stage))

    return prepare_processing_from_file(
        data_path,
        result_index=int(config.resultIndex),
        **workflow_config,
    )


def _canonical_stage_name(stage_name: str) -> str:
    return CANONICAL_STAGE_ALIASES.get(stage_name, stage_name)


def _select_bundle(
    processing: Any,
    data_kind: str,
    selected_bundle_from_processing: Any,
    frame_from_bundle: Any,
) -> Tuple[Optional[Any], Optional[str]]:
    requested_stage = _canonical_stage_name(data_kind)
    lookup_stages = [requested_stage]
    lookup_stages.extend(stage for stage in CANONICAL_STAGE_FALLBACKS if stage not in lookup_stages)

    for stage_name in lookup_stages:
        try:
            bundle = selected_bundle_from_processing(processing, stage_name=stage_name)
        except Exception:
            continue
        try:
            frame = frame_from_bundle(bundle)
        except Exception:
            continue
        if frame is not None and len(frame) > 0:
            return bundle, stage_name
    return None, None


def _extract_data_arrays(
    data_df: Any,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if "Q" not in data_df or "I" not in data_df:
        raise ValueError("Data frame must include 'Q' and 'I' columns.")

    Q_vals = np.asarray(data_df["Q"], dtype=float)
    I_vals = np.asarray(data_df["I"], dtype=float)

    sigma = None
    for key in ("ISigma", "IError", "IStd", "ISEM"):
        if key in data_df:
            sigma = np.asarray(data_df[key], dtype=float)
            break

    q_sigma = None
    for key in ("QSigma", "QError", "QStd", "QSEM"):
        if key in data_df:
            q_sigma = np.asarray(data_df[key], dtype=float)
            break

    mask = np.isfinite(Q_vals) & np.isfinite(I_vals)
    if sigma is not None:
        mask &= np.isfinite(sigma)
    if q_sigma is not None:
        mask &= np.isfinite(q_sigma)

    Q_vals = Q_vals[mask]
    I_vals = I_vals[mask]
    if sigma is not None:
        sigma = sigma[mask]
    if q_sigma is not None:
        q_sigma = q_sigma[mask]

    order = np.argsort(Q_vals)
    Q_vals = Q_vals[order]
    I_vals = I_vals[order]
    if sigma is not None:
        sigma = sigma[order]
    if q_sigma is not None:
        q_sigma = q_sigma[order]

    return Q_vals, I_vals, sigma, q_sigma

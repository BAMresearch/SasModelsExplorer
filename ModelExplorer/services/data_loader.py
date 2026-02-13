# ModelExplorer/services/data_loader.py

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import yaml

from ModelExplorer.types import DataConfig
from ModelExplorer.utils.units import DEFAULT_I_UNIT, DEFAULT_Q_UNIT, normalize_unit_label


def parse_yaml_config(yaml_text: str) -> DataConfig:
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

    q_unit = normalize_unit_label(raw.get("Q_unit") or raw.get("q_unit") or DEFAULT_Q_UNIT)
    i_unit = normalize_unit_label(raw.get("I_unit") or raw.get("i_unit") or DEFAULT_I_UNIT)

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
    McData1D: Any,
    BaseData: Any,
    DataBundle: Any,
) -> Tuple[Any, str, int]:
    config = parse_yaml_config(yaml_text)
    mds = _load_mcsas3_data(data_path, config, McData1D)
    data_df, used_kind = _select_data_frame(mds, data_kind)
    if data_df is None or used_kind is None:
        raise ValueError("No data available after loading.")

    Q_vals, I_vals, sigma, q_sigma = _extract_data_arrays(data_df)
    if Q_vals.size == 0:
        raise ValueError("No finite data points found.")

    bundle = _build_data_bundle(
        Q_vals,
        I_vals,
        sigma,
        q_sigma,
        config,
        data_path,
        used_kind,
        BaseData,
        DataBundle,
    )
    return bundle, used_kind, Q_vals.size


def _load_mcsas3_data(data_path: Path, config: DataConfig, McData1D: Any) -> Any:
    return McData1D(
        filename=data_path,
        nbins=int(config.nbins),
        csvargs=config.csvargs,
        pathDict=config.pathDict,
        IEmin=float(config.IEmin),
        dataRange=config.dataRange,
        omitQRanges=config.omitQRanges,
        resultIndex=int(config.resultIndex),
    )


def _select_data_frame(mds: Any, data_kind: str) -> Tuple[Optional[Any], Optional[str]]:
    data_df = getattr(mds, data_kind, None)
    if data_df is None or len(data_df) == 0:
        for fallback in ("binnedData", "clippedData", "rawData"):
            data_df = getattr(mds, fallback, None)
            if data_df is not None and len(data_df) > 0:
                data_kind = fallback
                break
    if data_df is None or len(data_df) == 0:
        return None, None
    return data_df, data_kind


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


def _build_data_bundle(
    Q_vals: np.ndarray,
    I_vals: np.ndarray,
    sigma: Optional[np.ndarray],
    q_sigma: Optional[np.ndarray],
    config: DataConfig,
    data_path: Path,
    data_kind: str,
    BaseData: Any,
    DataBundle: Any,
) -> Any:
    bundle = DataBundle()
    signal_unc = {"ISigma": sigma} if sigma is not None else {}
    q_unc = {"QSigma": q_sigma} if q_sigma is not None else {}

    bundle["I"] = BaseData(
        signal=I_vals,
        units=config.I_unit,
        uncertainties=signal_unc,
        rank_of_data=1,
    )
    bundle["Q"] = BaseData(
        signal=Q_vals,
        units=config.Q_unit,
        uncertainties=q_unc,
        rank_of_data=1,
    )
    bundle.default_plot = "I"
    bundle.description = f"{data_path.name} ({data_kind})"
    return bundle

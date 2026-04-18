"""Helpers for converting canonical bundles into overlay and fit statistics."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ModelExplorer.types import OverlayData
from ModelExplorer.utils.units import normalize_unit_label


@runtime_checkable
class BaseDataLike(Protocol):
    """Protocol describing the BaseData surface used by this repository."""

    signal: object
    units: object
    uncertainties: Mapping[str, object]

    def to_units(self, units: str) -> None:
        """Convert the signal and uncertainties to the given units."""


def combined_uncertainty(data: BaseDataLike) -> NDArray[np.float64] | None:
    """Return a combined uncertainty array from a BaseData-like object."""

    uncertainties = data.uncertainties or {}
    if len(uncertainties) == 0:
        return None
    if "ISigma" in uncertainties:
        return np.asarray(uncertainties["ISigma"], dtype=float)

    variance = np.zeros_like(np.asarray(data.signal, dtype=float), dtype=float)
    for value in uncertainties.values():
        variance += np.asarray(value, dtype=float) ** 2
    return np.asarray(np.sqrt(variance), dtype=float)


def overlay_from_bundle(
    data_bundle: Mapping[str, BaseDataLike] | None,
    target_q_unit: str,
    target_i_unit: str,
) -> OverlayData | None:
    """Convert a canonical bundle into an ``OverlayData`` object in target units."""

    if data_bundle is None:
        return None

    data_i = data_bundle.get("signal")
    if data_i is None:
        data_i = data_bundle.get("I")
    data_q = data_bundle.get("Q")
    if data_i is None or data_q is None:
        return None

    i_copy = deepcopy(data_i)
    q_copy = deepcopy(data_q)

    try:
        q_copy.to_units(normalize_unit_label(target_q_unit))
        i_copy.to_units(target_i_unit)
    except Exception:
        pass

    q_vals = np.asarray(q_copy.signal, dtype=float)
    i_vals = np.asarray(i_copy.signal, dtype=float)
    sigma = combined_uncertainty(i_copy)

    finite_mask = np.isfinite(q_vals) & np.isfinite(i_vals)
    if sigma is not None:
        finite_mask &= np.isfinite(sigma)

    q_vals = q_vals[finite_mask]
    i_vals = i_vals[finite_mask]
    if sigma is not None:
        sigma = sigma[finite_mask]

    if q_vals.size == 0:
        return None

    order = np.argsort(q_vals)
    q_vals = q_vals[order]
    i_vals = i_vals[order]
    if sigma is not None:
        sigma = sigma[order]

    label = str(getattr(data_bundle, "description", "Data") or "Data")
    return OverlayData(Q=q_vals, I=i_vals, ISigma=sigma, label=label)


def interpolate_model_intensity(
    q_model: NDArray[np.float64],
    i_model: NDArray[np.float64],
    q_data: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate model intensity from model grid onto the overlay data grid."""

    if np.any(q_model <= 0) or np.any(i_model <= 0) or np.any(q_data <= 0):
        return np.interp(q_data, q_model, i_model, left=np.nan, right=np.nan)

    log_q = np.log10(q_model)
    log_i = np.log10(i_model)
    log_qd = np.log10(q_data)
    log_i_interp = np.interp(log_qd, log_q, log_i, left=np.nan, right=np.nan)
    return np.asarray(10**log_i_interp, dtype=float)


def reduced_chi_square(
    q_model: NDArray[np.float64],
    i_model: NDArray[np.float64],
    overlay: OverlayData | None,
    *,
    n_parameters: int,
    intensity_scale: float,
) -> tuple[float | None, int | None, int | None]:
    """Compute reduced chi-square for model-vs-overlay comparison."""

    if overlay is None or overlay.ISigma is None:
        return None, None, None

    try:
        interpolated = interpolate_model_intensity(
            np.asarray(q_model, dtype=float),
            np.asarray(i_model, dtype=float),
            np.asarray(overlay.Q, dtype=float),
        )
    except Exception:
        return None, None, None

    scaled_model = interpolated * intensity_scale
    valid = np.isfinite(scaled_model) & np.isfinite(overlay.I) & np.isfinite(overlay.ISigma) & (overlay.ISigma > 0)
    if int(np.sum(valid)) < 2:
        return None, None, None

    points = int(np.sum(valid))
    dof = max(points - n_parameters, 1)
    chi2 = np.sum(((overlay.I[valid] - scaled_model[valid]) / overlay.ISigma[valid]) ** 2) / dof
    return float(chi2), int(dof), points

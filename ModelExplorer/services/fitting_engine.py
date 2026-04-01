"""Model fitting service around scipy least-squares."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ModelExplorer.types import OverlayData, ParameterMapping
from ModelExplorer.utils.units import MODEL_INTENSITY_SCALE, create_unit_registry, normalize_unit_label

from .model_parameters import ensure_pd_parameter_defaults


class ModelKernelProvider(Protocol):
    """Protocol for model objects that can create sasmodels kernels."""

    def make_kernel(self, data: list[NDArray[np.float64]]) -> object:
        """Create and return a kernel object for the given Q arrays."""


class ModelInfoLike(Protocol):
    """Protocol for model info objects exposing one-dimensional PD parameters."""

    parameters: object


@dataclass(slots=True, frozen=True)
class FitResult:
    """Result payload returned by ``fit_model``."""

    success: bool
    message: str
    parameters: ParameterMapping


def fit_model(
    model: ModelKernelProvider,
    model_info: ModelInfoLike,
    parameters: ParameterMapping,
    fit_names: list[str],
    parameter_defs: Mapping[str, object],
    data: OverlayData,
    q_unit: str,
    max_nfev: int,
    intensity_scale: float = MODEL_INTENSITY_SCALE,
) -> FitResult:
    """Fit selected model parameters against overlay data."""

    try:
        from scipy.optimize import least_squares
    except Exception:
        return FitResult(False, "scipy is required for fitting.", parameters)

    if not fit_names:
        return FitResult(False, "Select parameters to fit.", parameters)
    if data.ISigma is None:
        return FitResult(False, "Data uncertainties are required for fitting.", parameters)
    sigma = data.ISigma

    ureg = create_unit_registry()
    q_unit = normalize_unit_label(q_unit)
    try:
        kernel_q = np.asarray(data.Q * ureg.Quantity(1, q_unit).to("1/Ang").magnitude, dtype=float)
        kernel = model.make_kernel([kernel_q])
    except Exception as exc:
        return FitResult(False, f"Kernel error: {exc}", parameters)

    pd_names = list(getattr(getattr(model_info, "parameters", object()), "pd_1d", []))
    ensure_pd_parameter_defaults(parameters, pd_names)

    x0, bounds, used_names = _build_bounds(parameters, fit_names, parameter_defs)
    if not used_names:
        return FitResult(False, "No numeric parameters selected.", parameters)

    def residuals(x: NDArray[np.float64]) -> NDArray[np.float64]:
        for name, value in zip(used_names, x, strict=False):
            parameters[name] = float(value)
        model_i = _compute_model_intensity(kernel, parameters, intensity_scale)
        return (np.asarray(model_i, dtype=float) - data.I) / sigma

    result = least_squares(
        residuals,
        x0,
        bounds=bounds,
        max_nfev=max_nfev,
    )

    for name, value in zip(used_names, result.x, strict=False):
        parameters[name] = float(value)

    message = "Fit complete." if result.success else f"Fit stopped: {result.message}"
    return FitResult(bool(result.success), message, parameters)


def _compute_model_intensity(kernel: object, parameters: ParameterMapping, scale: float) -> NDArray[np.float64]:
    """Compute scaled model intensity for a kernel and parameter mapping."""

    from ModelExplorer.sasmodels_adapter import compute_intensity

    return np.asarray(compute_intensity(kernel, dict(parameters)), dtype=float) * scale


def _build_bounds(
    parameters: ParameterMapping,
    fit_names: list[str],
    parameter_defs: Mapping[str, object],
) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]], list[str]]:
    """Build initial values and scipy-compatible bounds for selected fit parameters."""

    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    used_names: list[str] = []

    for name in fit_names:
        param_obj = parameter_defs.get(name)
        if param_obj is None:
            continue
        limits = getattr(param_obj, "limits", None) or (-np.inf, np.inf)
        low, high = float(limits[0]), float(limits[1])
        value = float(parameters.get(name, float(getattr(param_obj, "default", 0.0))))
        if np.isfinite(low) and value <= low:
            value = low + 1e-12
        if np.isfinite(high) and value >= high:
            value = high - 1e-12
        used_names.append(name)
        x0.append(value)
        lower.append(low)
        upper.append(high)

    x0_arr = np.array(x0, dtype=float)
    bounds = (np.array(lower, dtype=float), np.array(upper, dtype=float))
    return x0_arr, bounds, used_names

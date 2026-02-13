# ModelExplorer/services/fitting_engine.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ModelExplorer.types import OverlayData
from ModelExplorer.utils.units import MODEL_INTENSITY_SCALE, create_unit_registry, normalize_unit_label


@dataclass
class FitResult:
    success: bool
    message: str
    parameters: Dict[str, float]


def fit_model(
    model: Any,
    model_info: Any,
    parameters: Dict[str, float],
    fit_names: List[str],
    parameter_defs: Dict[str, object],
    data: OverlayData,
    q_unit: str,
    max_nfev: int,
    intensity_scale: float = MODEL_INTENSITY_SCALE,
) -> FitResult:
    try:
        from scipy.optimize import least_squares
    except Exception:
        return FitResult(False, "scipy is required for fitting.", parameters)

    if not fit_names:
        return FitResult(False, "Select parameters to fit.", parameters)

    if data.ISigma is None:
        return FitResult(False, "Data uncertainties are required for fitting.", parameters)

    ureg = create_unit_registry()
    q_unit = normalize_unit_label(q_unit)
    try:
        kernel = model.make_kernel([data.Q * ureg.Quantity(1, q_unit).to("1/Ang").magnitude])
    except Exception as exc:
        return FitResult(False, f"Kernel error: {exc}", parameters)

    _prepare_pd_parameters(parameters, model_info)

    x0, bounds, used_names = _build_bounds(parameters, fit_names, parameter_defs)
    if not used_names:
        return FitResult(False, "No numeric parameters selected.", parameters)

    def residuals(x: np.ndarray) -> np.ndarray:
        for name, value in zip(used_names, x):
            parameters[name] = float(value)
        model_I = _compute_model_intensity(kernel, parameters, intensity_scale)
        return (model_I - data.I) / data.ISigma

    result = least_squares(
        residuals,
        x0,
        bounds=bounds,
        max_nfev=max_nfev,
    )

    for name, value in zip(used_names, result.x):
        parameters[name] = float(value)

    if result.success:
        message = "Fit complete."
    else:
        message = f"Fit stopped: {result.message}"
    return FitResult(result.success, message, parameters)


def _compute_model_intensity(kernel: Any, parameters: Dict[str, float], scale: float) -> np.ndarray:
    from ModelExplorer.sasmodels_adapter import compute_intensity

    return compute_intensity(kernel, parameters) * scale


def _prepare_pd_parameters(parameters: Dict[str, float], model_info: Any) -> None:
    pd_params = [param for param in parameters.keys() if param in model_info.parameters.pd_1d]
    for param in pd_params:
        parameters[param + "_pd_n"] = 35


def _build_bounds(
    parameters: Dict[str, float],
    fit_names: List[str],
    parameter_defs: Dict[str, object],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], List[str]]:
    x0 = []
    lower = []
    upper = []
    used_names = []

    for name in fit_names:
        param_obj = parameter_defs.get(name)
        if param_obj is None:
            continue
        limits = getattr(param_obj, "limits", None) or (-np.inf, np.inf)
        low, high = float(limits[0]), float(limits[1])
        value = float(parameters.get(name, param_obj.default))
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

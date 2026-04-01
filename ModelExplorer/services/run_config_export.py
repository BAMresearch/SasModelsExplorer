"""Helpers to build McSAS3 run-configuration payloads from current UI state."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

from ModelExplorer.types import ParameterValue


def build_mcsas3_run_configuration(
    *,
    model_expression: str,
    parameters: Mapping[str, ParameterValue],
    fit_parameter: str,
    fit_limits: tuple[float, float],
    log_random: bool = True,
    n_contrib: int = 300,
    max_iter: int = 100000,
    max_accept: int = 10000,
    conv_crit: float = 1.0,
    n_rep: int = 10,
    n_cores: int = 5,
    model_dtype: str = "default",
) -> dict[str, Any]:
    """Build a McSAS3 run-configuration mapping from a model expression and parameter mapping."""

    if fit_parameter not in parameters:
        raise ValueError(f"Fit parameter '{fit_parameter}' is not available in current parameter mapping.")

    fit_value = parameters[fit_parameter]
    if not isinstance(fit_value, Real) or isinstance(fit_value, bool):
        raise ValueError(f"Fit parameter '{fit_parameter}' must be numeric.")

    low, high = float(fit_limits[0]), float(fit_limits[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Fit parameter limits must be finite.")
    if high <= low:
        raise ValueError("Fit parameter upper limit must be greater than lower limit.")
    if log_random and low <= 0:
        raise ValueError("For logRandom optimization the lower fit limit must be > 0.")

    static_parameters: dict[str, ParameterValue] = {}
    for key, value in parameters.items():
        if key == fit_parameter:
            continue
        static_parameters[key] = _to_builtin_scalar(value)

    return {
        "modelName": model_expression,
        "nContrib": int(n_contrib),
        "modelDType": model_dtype,
        "fitParameterLimits": {fit_parameter: [low, high]},
        "staticParameters": static_parameters,
        "maxIter": int(max_iter),
        "maxAccept": int(max_accept),
        "convCrit": float(conv_crit),
        "nRep": int(n_rep),
        "nCores": int(n_cores),
        "logRandom": bool(log_random),
    }


def _to_builtin_scalar(value: ParameterValue) -> ParameterValue:
    """Convert NumPy scalar values to built-in Python scalars."""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value

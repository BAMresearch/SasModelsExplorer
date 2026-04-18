"""Shared helpers for preparing model parameter dictionaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping

from ModelExplorer.types import ParameterMapping, ParameterValue


def merge_parameter_values(
    visible_values: Mapping[str, ParameterValue],
    hidden_defaults: Mapping[str, ParameterValue],
) -> ParameterMapping:
    """Merge visible UI values with hidden defaults into a single mutable mapping."""

    merged = dict(visible_values)
    merged.update(hidden_defaults)
    return merged


def ensure_pd_parameter_defaults(
    parameters: MutableMapping[str, ParameterValue],
    pd_parameter_names: Iterable[str],
    *,
    pd_n: int = 35,
) -> None:
    """Ensure required ``*_pd_n`` keys exist for all selected polydisperse parameters."""

    for parameter_name in pd_parameter_names:
        if parameter_name in parameters:
            parameters[f"{parameter_name}_pd_n"] = pd_n

"""Adapters around sasmodels APIs used by the GUI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import sasmodels.core
import sasmodels.direct_model
import sasmodels.modelinfo
from numpy.typing import NDArray

from .utils.list_to_columnar_string import list_to_columnar_string


class SupportsModelInfo(Protocol):
    """Protocol for sasmodels model objects that expose ``info.parameters``."""

    info: sasmodels.modelinfo.ModelInfo

    def make_kernel(self, data: list[NDArray[np.float64]]) -> object:
        """Create a sasmodels kernel for the provided Q-array payload."""


def load_model_and_info(model_name: str) -> tuple[SupportsModelInfo, sasmodels.modelinfo.ModelInfo]:
    """Load a sasmodels model and its corresponding ``ModelInfo``."""

    model = sasmodels.core.load_model(model_name)
    model_info = sasmodels.core.load_model_info(model_name)
    return model, model_info


def build_parameter_list(
    model: SupportsModelInfo,
    model_info: sasmodels.modelinfo.ModelInfo,
    pd_types: Sequence[str],
) -> list[sasmodels.modelinfo.Parameter]:
    """Build a deduplicated parameter list with polydispersity extras."""

    parameters: list[sasmodels.modelinfo.Parameter] = []
    seen_parameters: set[str] = set()
    base_parameters = model_info.parameters.common_parameters + model_info.parameters.call_parameters

    for parameter in base_parameters:
        if parameter.name in seen_parameters:
            continue
        seen_parameters.add(parameter.name)
        parameters.append(parameter)

        if parameter.name in model.info.parameters.pd_1d:
            pd_param = sasmodels.modelinfo.Parameter(
                f"{parameter.name}_pd",
                units="",
                default=0,
                limits=(0, 1),
                description=f"relative polydispersity of parameter {parameter.name}",
            )
            parameters.append(pd_param)

            pd_type_param = sasmodels.modelinfo.Parameter(
                f"{parameter.name}_pd_type",
                limits=[[list(pd_types)]],
                units="",
                default="gaussian",
                description=f"polydispersity distribution shape for parameter {parameter.name}",
            )
            pd_type_param.choices = list(pd_types)
            parameters.append(pd_type_param)

    return parameters


def is_magnetic_parameter(name: str) -> bool:
    """Return ``True`` for magnetic/polarization sasmodels parameters."""

    magnetic_suffixes = ("_M0", "_mtheta", "_mphi")
    return name.startswith("up_") or name.endswith(magnetic_suffixes)


def split_magnetic_parameters(
    parameters: list[sasmodels.modelinfo.Parameter],
    include_magnetic: bool,
) -> tuple[list[sasmodels.modelinfo.Parameter], dict[str, Any]]:
    """Split full parameter list into visible params and hidden magnetic defaults."""

    if include_magnetic:
        return parameters, {}

    visible: list[sasmodels.modelinfo.Parameter] = []
    hidden_defaults: dict[str, Any] = {}
    for parameter in parameters:
        if is_magnetic_parameter(parameter.name):
            hidden_defaults[parameter.name] = parameter.default
        else:
            visible.append(parameter)
    return visible, hidden_defaults


def compute_intensity(kernel: object, parameters: dict[str, Any]) -> NDArray[np.float64]:
    """Run a sasmodels kernel and return intensity as float array."""

    return np.asarray(sasmodels.direct_model.call_kernel(kernel, parameters), dtype=float)


def generate_model_info_text(ncols: int = 3, padding: str = "   ") -> str:
    """Generate help text listing available models grouped by category."""

    categories = [sasmodels.core.load_model_info(model).category for model in sasmodels.core.list_models()]
    groupings = [category.split(":")[0] for category in categories]

    info_text = (
        "Sasmodels can be specified as one of the following. They can also be composed by "
        "multiplication/division or addition/subtraction.\n"
    )
    info_text += (
        'For example: "cylinder+sphere" will add these two models. A structure factor can be applied '
        'with the @-operator, e.g. "sphere@hardsphere".\n\n'
    )

    for group in list(dict.fromkeys(groupings)):
        info_text += f"Available {group} Models:\n"
        info_text += " -- \n"
        model_list = [
            sasmodels.core.load_model_info(model).id
            for model in sasmodels.core.list_models()
            if sasmodels.core.load_model_info(model).category.startswith(group)
        ]
        info_text += list_to_columnar_string(model_list, ncols=ncols, padding=padding, ordering="cols")

    return info_text

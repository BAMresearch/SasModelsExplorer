# ModelExplorer/utils/units.py

from typing import Optional

import pint

MODEL_INTENSITY_SCALE = 100.0
DEFAULT_Q_UNIT = "1/nm"
DEFAULT_I_UNIT = "1/(m sr)"


def normalize_unit_label(value: Optional[str]) -> str:
    """Normalize angstrom spellings to a pint-compatible unit token."""

    if not value:
        return ""
    return (
        str(value)
        .replace("\u00c5ngstr\u00f6m", "angstrom")
        .replace("Angstrom", "angstrom")
        .replace("\u00c5", "angstrom")
    )


def create_unit_registry() -> pint.UnitRegistry:
    """Create a pint registry with project-specific unit aliases."""

    ureg: pint.UnitRegistry = pint.UnitRegistry(auto_reduce_dimensions=True)
    ureg.define("percent = 0.01 = %")
    ureg.define("Angstrom = 1e-10*m = Ang = angstrom")
    ureg.define("item = 1")
    return ureg

# ModelExplorer/utils/units.py

from typing import Optional

import pint

MODEL_INTENSITY_SCALE = 100.0
DEFAULT_Q_UNIT = "1/nm"
DEFAULT_I_UNIT = "1/(m sr)"


def normalize_unit_label(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).replace("\u00c5ngstr\u00f6m", "Angstrom").replace("\u00c5", "Angstrom")


def create_unit_registry() -> pint.UnitRegistry:
    ureg = pint.UnitRegistry(auto_reduce_dimensions=True)
    ureg.define("percent = 0.01 = %")
    ureg.define("Angstrom = 1e-10*m = Ang = angstrom")
    ureg.define("item = 1")
    return ureg

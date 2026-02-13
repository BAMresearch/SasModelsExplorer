import numpy as np

from ModelExplorer.utils import units


def test_normalize_unit_label_angstrom():
    assert units.normalize_unit_label("1/\u00c5ngstr\u00f6m") == "1/Angstrom"
    assert units.normalize_unit_label("\u00c5") == "Angstrom"


def test_unit_registry_supports_angstrom_and_percent():
    ureg = units.create_unit_registry()
    assert str(ureg.Unit("Angstrom")) == "Angstrom"
    assert str(ureg.Unit("percent")) == "percent"

    value = (1 * ureg.Unit("1/nm")).to("1/Angstrom").magnitude
    assert np.isfinite(value)

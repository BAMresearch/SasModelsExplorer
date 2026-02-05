from ModelExplorer.sasmodels_adapter import (
    build_parameter_list,
    load_model_and_info,
    split_magnetic_parameters,
)

PD_TYPES = ["uniform", "rectangle", "gaussian", "lognormal", "schulz", "boltzmann"]


def test_build_parameter_list_deduplicates_common_parameters():
    model, info = load_model_and_info("sphere@hardsphere+cylinder")
    parameters = build_parameter_list(model, info, PD_TYPES)
    names = [param.name for param in parameters]

    assert names.count("scale") == 1
    assert names.count("background") == 1
    assert len(names) == len(set(names))


def test_build_parameter_list_adds_polydisperse_parameters():
    model, info = load_model_and_info("sphere")
    parameters = build_parameter_list(model, info, PD_TYPES)
    names = {param.name for param in parameters}

    assert "radius_pd" in names
    assert "radius_pd_type" in names


def test_split_magnetic_parameters_hides_magnetic_defaults():
    model, info = load_model_and_info("sphere")
    parameters = build_parameter_list(model, info, PD_TYPES)
    visible, hidden_defaults = split_magnetic_parameters(parameters, include_magnetic=False)
    visible_names = {param.name for param in visible}

    assert "sld_M0" not in visible_names
    assert "sld_mtheta" not in visible_names
    assert "sld_mphi" not in visible_names
    assert "sld_M0" in hidden_defaults

# SasModelsExplorer HDF5 State Schema

Related design:
- `docs/design/internal_maintainability_foundation.md`
- `docs/design/processingdata_migration_plan.md`

## Scope

This document defines the persisted HDF5 state format used by SasModelsExplorer for
save/load workflows.

As of **April 1, 2026**, the application supports only this schema and no longer reads
the short-lived legacy flat-attribute format.

## Root Layout

All state content is stored under:

- `/analyses/SasModelsExplorer`

Tree overview:

```text
/analyses/SasModelsExplorer
  model_name                          (dataset, UTF-8 string)
  /model
    Q                                 (dataset, float[])
      attrs: units
    I                                 (dataset, float[])
      attrs: units
  /parameters
    <parameter_name>                  (dataset, scalar string|int|float|bool)
      attrs: units[, min][, max]
  /data
    /sample_raw|sample_clipped|sample_binned
      Q                               (dataset, float[])
        attrs: units[, uncertainties="QSigma"]
      I                               (dataset, float[])
        attrs: units[, uncertainties="ISigma"]
      QSigma                          (optional dataset, float[])
        attrs: units
      ISigma                          (optional dataset, float[])
        attrs: units
      <additional_frame_columns>      (optional datasets)
      attrs: description (optional)
  /SME_settings
    schema_version                    (dataset, int; current = 2)
    q_unit                            (dataset, UTF-8 string)
    intensity_unit                    (dataset, UTF-8 string)
    q_min                             (dataset, float)
    q_max                             (dataset, float)
    selected_data_stage               (dataset, UTF-8 string)
    fit_parameter_names_json          (dataset, UTF-8 string JSON list)
    hidden_parameter_defaults_json    (dataset, UTF-8 string JSON mapping)
    data_loading_yaml                 (dataset, UTF-8 string)
    data_loading_preset               (optional dataset, UTF-8 string)
    data_source_file                  (optional dataset, UTF-8 string)
    data_mode_label                   (dataset, UTF-8 string)
    exporter                          (dataset, UTF-8 string)
```

## Parameter Dataset Rules

- Parameter names are encoded directly as dataset names in `/parameters`.
- No redundant `parameter_name` attribute is written.
- `units` is always written (empty string if unknown).
- `min`/`max` are written when limits are known.

## Data Dataset Rules

- `Q`, `I`, `QSigma`, and `ISigma` carry dataset-level `units` attributes.
- If `ISigma` exists, `I.attrs["uncertainties"] = "ISigma"`.
- If `QSigma` exists, `Q.attrs["uncertainties"] = "QSigma"`.
- Stage groups may include extra columns from McSAS3 `frame_from_bundle` output.

## Load Behavior

- Loader requires `/analyses/SasModelsExplorer`; missing root is treated as unsupported schema.
- `selected_data_stage` chooses the active stage if present; otherwise first available stage is used.
- If no stage data is present, model/session state still loads and overlay data is cleared.
- YAML text and optional preset name are restored from `SME_settings`.

## Contract Tests

Schema behavior is enforced by:

- `tests/test_hdf5_state_contract.py`

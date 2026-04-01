# Internal Maintainability Foundation

## Purpose

This document captures the internal refactor foundation completed before implementing export UI
features (McSAS3 run-template export, HDF5 save/load, and CSV export).

For the concrete persisted HDF5 contract, see `docs/design/hdf5_state_schema.md`.

The goals are:

- keep current behavior stable while reducing maintenance cost,
- make typing and internal interfaces explicit,
- prepare backend-only export contracts for the next feature phase.

## Pain Points Addressed

- logic duplication between plotting and fitting parameter preparation,
- logic duplication between plotting and fitting data-overlay conversion,
- ad-hoc McSAS3 integration/import handling in UI code,
- large fallback copy of McSAS3GUI YAML editor code,
- missing static type-check gate for core service modules,
- fragile Git pre-commit hook interpreter pinning.

## Target Architecture

- `modelexplorer.py` remains orchestration-focused.
- reusable backend logic lives in `ModelExplorer/services/*`.
- canonical McSAS3 data integration is isolated in `services/mcsas3_backend.py`.
- parameter preparation is shared via `services/model_parameters.py`.
- overlay conversion + chi-square helpers are shared via `services/overlay_processing.py`.
- export contracts and payload transforms are defined in:
  - `services/export_contracts.py`
  - `services/export_snapshot_builder.py`

## Typed Contracts Introduced

- session/state dataclasses in `ModelExplorer/types.py`:
  - `ModelSessionState`
  - `DataSelectionState`
- export dataclasses:
  - `ExportSnapshot`
  - `RunConfigTemplatePayload`
  - `CsvModelParameterTable`
  - `Hdf5StatePayload`
- exporter protocols:
  - `RunConfigTemplateExporter`
  - `CsvModelExporter`
  - `Hdf5StateExporter`

These contracts are backend-only in this phase; no export UI actions are added yet.

## Export-Prep Data Flow

1. GUI state is normalized into typed session/data state.
2. `build_export_snapshot()` gathers model curve, parameters, units, and overlay metadata.
3. Pure payload builders transform snapshot data to:
   - run-template payload,
   - CSV table payload,
   - HDF5 payload.

The next feature phase can wire these payloads to concrete file writers and UI actions without
reworking core state extraction.

## Dependency and Tooling Policy

- runtime baseline: Python `>=3.12`.
- required runtime deps: `mcsas3`, `modacor`.
- current transition pin: `mcsas3` installs from branch `in_depth_upgrades` until the next canonical release lands on PyPI.
- optional integration: `mcsas3gui` via `gui-interop` extra.
- type-check gate: mypy on core modules configured in `pyproject.toml`.
- expanded typed UI modules in gate:
  - `ModelExplorer/export_panel.py`
  - `ModelExplorer/fitting_panel.py`
  - `ModelExplorer/parameter_panel.py`
  - `ModelExplorer/data_loading_panel.py`
  - `ModelExplorer/modelbrowser.py`
  - `ModelExplorer/yaml_editor_widget.py`
  - `ModelExplorer/modelexplorer.py`
- pre-commit reliability:
  - repo-managed hooks via `.githooks`,
  - stable wrapper hook at `.githooks/pre-commit`,
  - bootstrap command: `./scripts/bootstrap_dev.sh`.

## Follow-Up (Optional)

`modelexplorer.py` is now part of the mypy gate and uses typed state/protocol helpers for HDF5 IO.

Recommended next cleanup:

1. Move remaining HDF5 read/write orchestration from `modelexplorer.py` to a dedicated typed persistence service.

## Checklist for Future Developers

- After recreating `.venv`, always run `./scripts/bootstrap_dev.sh`.
- Keep non-UI domain logic in `services/*`, not in QWidget classes.
- Extend export behavior by adding payload mappers/writers first, UI actions second.
- Keep new service functions fully typed and documented with concise docstrings.
- Keep `ruff`, `mypy`, and `pytest` passing before merging.

# ProcessingData Migration Plan

## Context

`McSAS3`/`McSAS3GUI` moved to the canonical `ProcessingData` workflow model backed by MoDaCor
(`ProcessingData`, `DataBundle`, `BaseData`).
`SasModelsExplorer` previously loaded data through the removed `mcsas3.mc_data_1d.McData1D` path and
manually rebuilt bundles.

This document defines what needed to change, what was simplified, and how to keep the integration
lean and maintainable.

## 1. Required Updates In SasModelsExplorer

### Data-loading integration

- Replace `McData1D` construction with canonical workflow ingestion:
  - `prepare_1d_processing_data_from_file(...)`
  - `selected_bundle_from_processing(...)`
- Select stage bundles using canonical stage names:
  - `sample_raw`
  - `sample_clipped`
  - `sample_binned`
- Keep backward compatibility for legacy stage aliases (`rawData`, `clippedData`, `binnedData`) and
  legacy YAML keys (`Q_unit`, `I_unit`).

### Bundle contract alignment

- Stop constructing app-local `DataBundle`/`BaseData` wrappers in `SasModelsExplorer`.
- Consume canonical bundles from McSAS3 directly and only transform for plotting/fitting.
- Support both canonical signal key (`signal`) and legacy key (`I`) during the transition period.

### Runtime dependency loading

- Keep local sibling-checkout fallback (`../McSAS3/src`) for development workflows.
- Prefer canonical McSAS3 API imports; avoid importing removed/internal modules.

## 2. Implementation Plan

### Phase 1 (done in this update)

1. Migrate loader service to canonical workflow APIs.
2. Migrate data-loading panel import checks and stage keys.
3. Update overlay extraction to read canonical signal bundles.
4. Update tests to the canonical stage/bundle model.

### Phase 2 (next, optional hardening)

1. Add UI validation hints for YAML keys that are deprecated aliases.
2. Add an integration test that exercises real McSAS3 + MoDaCor objects (not only dummies).
3. Add a narrow compatibility shim module so data-model assumptions are centralized.

### Phase 3 (if dependency policy allows)

1. Convert McSAS3 and MoDaCor from optional runtime presence to explicit project dependencies for
   the data-loading feature profile.
2. Gate packaging profiles (minimal vs full-feature) with explicit extras.

## 3. Simplification Opportunities

### Reuse McSAS3GUI YAML editor widget

Status in this update:

- `ModelExplorer/yaml_editor_widget.py` now prefers importing
  `mcsas3gui.gui.yaml_editor_widget` directly.
- A local fallback implementation remains for environments where `mcsas3gui` is not installed.

Why this is leaner:

- Avoids maintaining a forked copy of the same widget behavior.
- Keeps UX consistency between McSAS3GUI and SasModelsExplorer.
- Reduces divergence risk when editor behavior evolves upstream.

### Keep business logic in McSAS3/MoDaCor

- Data ingestion and canonical carrier handling should stay in McSAS3/MoDaCor.
- SasModelsExplorer should remain a thin consumer for plotting/fitting UX.

This minimizes duplicated domain logic and keeps maintenance centralized in the shared core suite.

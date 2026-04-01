# ModelExplorer/modelexplorer.py

import csv
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .data_loading_panel import DataLoadingPanel
from .export_panel import ExportPanel, RunConfigExportDialog
from .fitting_panel import FittingPanel
from .modelbrowser import ModelBrowser
from .parameter_panel import ParameterPanel
from .plotting import PlotManager
from .sasmodels_adapter import (
    SupportsModelInfo,
    build_parameter_list,
    compute_intensity,
    generate_model_info_text,
    load_model_and_info,
    split_magnetic_parameters,
)
from .services.export_contracts import ExportSnapshot
from .services.export_snapshot_builder import (
    SnapshotInputs,
    build_csv_model_parameter_table,
)
from .services.export_snapshot_builder import (
    build_export_snapshot as build_export_snapshot_payload,
)
from .services.fitting_engine import fit_model
from .services.model_parameters import ensure_pd_parameter_defaults, merge_parameter_values
from .services.overlay_processing import BaseDataLike, overlay_from_bundle, reduced_chi_square
from .services.run_config_export import build_mcsas3_run_configuration
from .types import DataSelectionState, ModelSessionState, ParameterMapping, ParameterValue
from .utils.units import MODEL_INTENSITY_SCALE, create_unit_registry, normalize_unit_label

ureg = create_unit_registry()


class H5DatasetLike(Protocol):
    """Minimal HDF5 dataset surface used by save/load helpers."""

    attrs: "H5AttrsLike"

    def __getitem__(self, key: object) -> object:
        """Return dataset contents for key/index payloads."""


class H5AttrsLike(Protocol):
    """Minimal mapping-like surface of HDF5 attributes."""

    def get(self, key: str, default: object | None = None) -> object:
        """Return attribute value with fallback default."""

    def __contains__(self, key: object) -> bool:
        """Return whether an attribute key exists."""

    def __setitem__(self, key: str, value: object) -> None:
        """Set an attribute key/value pair."""


class H5GroupLike(Protocol):
    """Minimal HDF5 group/file surface used by save/load helpers."""

    attrs: H5AttrsLike

    def create_group(self, name: str) -> "H5GroupLike":
        """Create and return a nested group."""

    def create_dataset(
        self,
        name: str,
        data: object,
        dtype: object | None = None,
    ) -> H5DatasetLike:
        """Create and return a dataset."""

    def get(self, name: str) -> object | None:
        """Return nested node by key, or ``None`` when missing."""

    def __getitem__(self, key: str) -> object:
        """Return a nested node by key."""

    def __contains__(self, key: object) -> bool:
        """Return whether a key exists."""

    def keys(self) -> Iterable[str]:
        """Return group keys."""


class BaseDataFactory(Protocol):
    """Callable factory protocol for constructing BaseData-like payloads."""

    def __call__(
        self,
        *,
        signal: object,
        units: str,
        uncertainties: Mapping[str, object] | None = None,
    ) -> BaseDataLike:
        """Build a BaseData-like object."""


class DataBundleLike(Protocol):
    """Mapping-like DataBundle with writable description."""

    description: str

    def __getitem__(self, key: str) -> BaseDataLike:
        """Return entry by key."""

    def __iter__(self) -> Iterable[str]:
        """Iterate over available keys."""

    def __len__(self) -> int:
        """Return number of keys."""

    def get(self, key: str, default: BaseDataLike | None = None) -> BaseDataLike | None:
        """Return entry by key with fallback default."""


class DataBundleFactory(Protocol):
    """Callable factory protocol for constructing DataBundle-like payloads."""

    def __call__(self, values: Mapping[str, BaseDataLike]) -> DataBundleLike:
        """Build a DataBundle-like mapping."""


class FrameLike(Protocol):
    """Minimal dataframe-like protocol used for HDF5 stage export."""

    columns: Iterable[str]

    def __getitem__(self, key: str) -> object:
        """Return the selected column payload."""


StageBundleMap: TypeAlias = dict[str, DataBundleLike]


@dataclass(slots=True)
class LoadedExplorerState:
    """Typed state payload loaded from an explorer HDF5 file."""

    model_expression: str
    q_unit: str
    intensity_unit: str
    q_min: float
    q_max: float
    parameters: ParameterMapping
    hidden_defaults: ParameterMapping
    fit_names: list[str]
    source_file: str
    stage_name: str
    yaml_text: str
    yaml_preset: str
    stage_bundles: StageBundleMap

    def __getitem__(self, key: str) -> object:
        """Provide backward-compatible dict-like state access by key."""

        return getattr(self, key)

    def get(self, key: str, default: object | None = None) -> object | None:
        """Provide backward-compatible ``dict.get`` style state access."""

        return getattr(self, key, default)


class SasModelApp(QMainWindow):
    """Main PyQt window that wires model inputs, parameter panel, and plotting."""

    pd_types: list[str] = ["uniform", "rectangle", "gaussian", "lognormal", "schulz", "boltzmann"]
    q_units: list[str] = ["1/nm", "1/Ångström", "1/m"]
    i_units: list[str] = ["1/(m sr)", "1/(cm sr)"]

    def __init__(self, modelName: str = "sphere") -> None:
        """Initialize the UI, wire signals, and load the initial model."""
        super().__init__()
        self.q: NDArray[np.float64] = np.geomspace(0.01, 1.0, 250)
        self.model: SupportsModelInfo | None = None
        self.kernel: object | None = None
        self.model_info: object | None = None
        self.model_parameters: Mapping[str, object] | None = None
        self.qunit: str = normalize_unit_label(self.q_units[0])
        self.infoText: str = ""
        self.hidden_parameter_defaults: ParameterMapping = {}

        self.setWindowTitle("SasModels Explorer")
        self.resize(1440, 720)

        # generate the infoText:
        self.infoText = generate_model_info_text()

        # Left layout for controls
        self.parameter_panel = ParameterPanel(on_change=self.update_plot, width=540)

        # Text input for model
        self.model_input = QLineEdit(modelName)
        self.model_input.setFixedWidth(350)
        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.model_input)

        self.parameter_panel.add_header_row("Model:", model_row)
        self.model_input.returnPressed.connect(self.load_model_parameters)
        self.show_magnetic_checkbox = QCheckBox("Show")
        self.show_magnetic_checkbox.setChecked(False)
        self.show_magnetic_checkbox.stateChanged.connect(self.load_model_parameters)
        self.parameter_panel.add_header_row("Magnetic:", self.show_magnetic_checkbox)

        # Right layout for plot
        self.plot_manager = PlotManager(figsize=(6, 4))
        self.data_panel = DataLoadingPanel()
        self.data_panel.dataChanged.connect(self.update_plot)

        # qmin and qmax inputs below the plot
        self.q_min_input = QLineEdit("0.01")
        self.q_max_input = QLineEdit("10.0")
        self.q_min_input.setFixedWidth(80)
        self.q_max_input.setFixedWidth(80)
        self.q_min_input.editingFinished.connect(self.update_kernel_and_plot)
        self.q_max_input.editingFinished.connect(self.update_kernel_and_plot)
        self.q_unit_input = QComboBox()
        for unit in self.q_units:
            self.q_unit_input.addItem(unit)
        self.q_unit_input.setFixedWidth(150)
        self.q_unit_input.currentIndexChanged.connect(self.update_kernel_and_plot)

        # Layout for q range inputs
        q_range_layout = QHBoxLayout()
        q_range_layout.addStretch(1)
        q_range_layout.addWidget(QLabel("Q Min:"))
        q_range_layout.addWidget(self.q_min_input)
        q_range_layout.addWidget(self.q_unit_input)
        q_range_layout.addWidget(QLabel("Q Max:"))
        q_range_layout.addWidget(self.q_max_input)
        q_range_layout.addStretch(1)

        # Vertical layout for plot and q range controls
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.plot_manager.canvas)
        plot_layout.addLayout(q_range_layout)

        plot_container = QWidget()
        plot_container.setLayout(plot_layout)

        self.side_panel_button = QPushButton("◀")
        self.side_panel_button.clicked.connect(self._toggle_side_panel)
        self.side_panel_button.setStyleSheet(
            "QPushButton { background-color: #f4f4f4; border: 0px solid #dddddd; }"
            "QPushButton:pressed { background-color: #e9e9e9; }"
        )

        toggle_layout = QVBoxLayout()
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.addWidget(self.side_panel_button)

        toggle_container = QWidget()
        toggle_container.setLayout(toggle_layout)
        toggle_container.setFixedWidth(13)
        self.side_panel_button.setSizePolicy(
            self.side_panel_button.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Expanding,
        )

        plot_with_toggle_layout = QHBoxLayout()
        plot_with_toggle_layout.setContentsMargins(0, 0, 0, 0)
        plot_with_toggle_layout.addWidget(plot_container)
        plot_with_toggle_layout.addWidget(toggle_container)

        plot_with_toggle = QWidget()
        plot_with_toggle.setLayout(plot_with_toggle_layout)

        self.fit_panel = FittingPanel()
        self.fit_panel.fitRequested.connect(self._run_fit)
        self.export_panel = ExportPanel()
        self.export_panel.save_state_requested.connect(self._save_state_hdf5)
        self.export_panel.load_state_requested.connect(self._load_state_hdf5)
        self.export_panel.export_csv_requested.connect(self._export_csv_parameters)
        self.export_panel.export_run_config_requested.connect(self._export_mcsas3_run_configuration)

        # Model browser (now embedded as a tab)
        self.model_browser = ModelBrowser(parent=self)
        self.model_browser.model_selected.connect(self.append_model_text)

        self.side_tabs = QTabWidget()
        self.side_tabs.addTab(self.model_browser, "Models")
        self.side_tabs.addTab(self.data_panel, "Data")
        self.side_tabs.addTab(self.fit_panel, "Fitting")
        self.side_tabs.addTab(self.export_panel, "Export/Import")
        self.side_tabs.setMinimumWidth(320)

        self.right_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.right_splitter.addWidget(plot_with_toggle)
        self.right_splitter.addWidget(self.side_tabs)
        self.right_splitter.setStretchFactor(0, 3)
        self.right_splitter.setStretchFactor(1, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.parameter_panel)
        splitter.addWidget(self.right_splitter)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self.setCentralWidget(splitter)

        # Load initial model
        self.model_info = None
        self.model_parameters = None
        self.load_model_parameters()
        self._set_side_panel_visible(False)

    # only one structure factro per form factor allowed...
    def append_model_text(self, model_name: str, is_structure: bool = False) -> None:
        current = self.model_input.text().strip()

        if not current:
            # First model
            self.model_input.setText(model_name)
            self.load_model_parameters()
            return

        # Split into additive terms
        terms = current.split("+")

        if is_structure:
            # Apply to last term only
            last_term = terms[-1]

            if "@" in last_term:
                # Replace structure factor
                base, _ = last_term.split("@", 1)
                terms[-1] = f"{base}@{model_name}"
            else:
                # Add structure factor to last term
                terms[-1] = f"{last_term}@{model_name}"

        else:
            # Add new form factor as new term
            terms.append(model_name)

        new_expression = "+".join(terms)

        self.model_input.setText(new_expression)
        self.load_model_parameters()

    def generate_infotext(self) -> str:
        """Return the help text shown when a model name is invalid."""
        return generate_model_info_text()

    def load_model_parameters(self, *_: object) -> None:
        """Load model info, rebuild parameter controls, and trigger a plot refresh."""
        previous_values: ParameterMapping = {}
        if self.parameter_panel is not None:
            previous_values = self.parameter_panel.get_values()
            previous_values.update(self.hidden_parameter_defaults)

        previous_fit_selection = self.fit_panel.get_selected_parameters()
        model_name = self.model_input.text().strip()

        try:
            # Attempt model load
            model, model_info = load_model_and_info(model_name)

            # Only update state after successful load
            self.model = model
            self.model_info = model_info
            self.model_parameters = model_info.parameters.defaults.copy()

            parameters = build_parameter_list(self.model, self.model_info, self.pd_types)
            visible_parameters, hidden_defaults = split_magnetic_parameters(
                parameters,
                self.show_magnetic_checkbox.isChecked(),
            )

            # Preserve previous values where possible
            for name in hidden_defaults:
                if name in previous_values:
                    hidden_defaults[name] = previous_values[name]

            self.hidden_parameter_defaults = hidden_defaults

            self.parameter_panel.set_parameters(visible_parameters)
            self.parameter_panel.set_values(
                {name: value for name, value in previous_values.items() if name in self.parameter_panel.parameters},
                emit_change=False,
            )

            self.fit_panel.set_parameters(self.parameter_panel.parameters)
            self.fit_panel.set_selected_parameters(previous_fit_selection)

            # Build kernel + refresh plot
            self.update_model_and_plot()

        except Exception as e:
            logging.warning(f"Error loading model '{model_name}': {e}", exc_info=True)

            # Reset state to avoid poisoned session
            self.model = None
            self.model_info = None
            self.kernel = None
            self.hidden_parameter_defaults = {}
            self._set_side_panel_visible(True)
            self.side_tabs.setCurrentWidget(self.model_browser)

            # ---------- Error Dialog ----------
            dialog = QDialog(self)
            dialog.setWindowTitle("Invalid Model Name")

            layout = QVBoxLayout(dialog)

            # Error label at top
            error_text = QTextEdit()
            error_text.setReadOnly(True)
            error_text.setMaximumHeight(80)
            error_text.setPlainText(f"Could not load model '{model_name}'.\n\n{type(e).__name__}: {e}")
            layout.addWidget(error_text)

            # Available models list (monospaced + scrollable)
            model_list = QTextEdit()
            model_list.setReadOnly(True)
            model_list.setPlainText(self.infoText)

            fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            model_list.setFont(fixed_font)
            # increase font size:
            font = model_list.font()
            font.setPointSize(font.pointSize() + 2)
            model_list.setFont(font)

            layout.addWidget(model_list)

            dialog.resize(900, 550)
            dialog.exec()

    def update_model_and_plot(self) -> None:
        """Rebuild the kernel (if needed) and refresh the plot."""
        self.update_kernel_and_plot()

    def update_kernel_and_plot(self) -> None:
        """Update the sasmodels kernel using the current q range/units."""
        # Retrieve and validate q range and units
        if self.model is None:
            return
        logging.info("updating kernel")
        try:
            qmin = float(self.q_min_input.text())
            qmax = float(self.q_max_input.text())
            qunit = normalize_unit_label(self.q_unit_input.currentText())
        except ValueError:
            qmin, qmax = 0.01, 1.0  # Default values in case of error
            qunit = "1/nm"

        # Prepare parameters for sasmodel
        self.q = np.geomspace(qmin, qmax, 250)
        self.qunit = qunit

        kernel_q = np.asarray(self.q * ureg.Quantity(1, qunit).to("1/Ang").magnitude, dtype=float)
        self.kernel = self.model.make_kernel([kernel_q])
        self.update_plot()

    def update_plot(self) -> None:
        """Compute intensity from current parameters and redraw the plot."""

        if self.model is None or self.kernel is None:
            return

        logging.info("updating plot")

        parameters = self._current_parameter_values()

        # Compute intensity
        Q = self.q
        Q_unit = self.qunit
        kernel = self.kernel
        logging.info("calling sasmodels with %s parameters", len(parameters))
        I_model = compute_intensity(kernel, parameters)

        data_bundle = cast(Mapping[str, BaseDataLike] | None, self.data_panel.get_data_bundle())
        overlay_data = overlay_from_bundle(data_bundle, Q_unit, self.i_units[0])
        chi2, dof, points = reduced_chi_square(
            np.asarray(Q, dtype=float),
            np.asarray(I_model, dtype=float),
            overlay_data,
            n_parameters=len(self.parameter_panel.get_values()),
            intensity_scale=self.plot_manager.scale,
        )
        chi_text = f"chi^2_red={chi2:.4g}" if chi2 is not None else None
        self.data_panel.set_chi_square(chi2, dof, points)
        self.plot_manager.plot(Q, I_model, Q_unit, data=overlay_data, chi_square_text=chi_text)

    def _toggle_side_panel(self) -> None:
        self._set_side_panel_visible(not self.side_tabs.isVisible())

    def _set_side_panel_visible(self, visible: bool) -> None:
        self.side_tabs.setVisible(visible)
        self.side_panel_button.setText("▶" if visible else "◀")
        if visible:
            self.right_splitter.setSizes([3, 1])
        else:
            self.right_splitter.setSizes([1, 0])

    def _run_fit(self) -> None:
        if self.model is None:
            return
        fit_names = self.fit_panel.get_selected_parameters()
        data_bundle = cast(Mapping[str, BaseDataLike] | None, self.data_panel.get_data_bundle())
        overlay_data = overlay_from_bundle(data_bundle, self.qunit, self.i_units[0])
        if overlay_data is None:
            self.fit_panel.set_status("Load data before fitting.")
            return

        parameters = self._current_parameter_values()
        self.fit_panel.set_status("Fitting...")
        result = fit_model(
            model=self.model,
            model_info=self.model.info,
            parameters=parameters,
            fit_names=fit_names,
            parameter_defs=self.parameter_panel.parameters,
            data=overlay_data,
            q_unit=self.qunit,
            max_nfev=self.fit_panel.get_max_iterations(),
            intensity_scale=MODEL_INTENSITY_SCALE,
        )

        self.parameter_panel.set_values(result.parameters)
        self.fit_panel.set_status(result.message)

    def _current_parameter_values(self) -> ParameterMapping:
        """Return merged visible+hidden model parameters with required PD helper keys."""

        parameters = merge_parameter_values(self.parameter_panel.get_values(), self.hidden_parameter_defaults)
        if self.model is None:
            return parameters
        pd_names = list(getattr(self.model.info.parameters, "pd_1d", []))
        ensure_pd_parameter_defaults(parameters, pd_names)
        return parameters

    def _set_q_unit_selection(self, target_q_unit: str) -> None:
        """Select the matching Q unit item from the dropdown by normalized comparison."""

        target_norm = normalize_unit_label(target_q_unit)
        for index in range(self.q_unit_input.count()):
            display_unit = self.q_unit_input.itemText(index)
            if normalize_unit_label(display_unit) == target_norm:
                self.q_unit_input.setCurrentIndex(index)
                return

    def _float_parameter_limits(self, parameters: ParameterMapping) -> dict[str, tuple[float, float]]:
        """Return numeric parameter names and effective slider limits for run-config export."""

        limits: dict[str, tuple[float, float]] = {}
        for name in self.parameter_panel.parameter_sliders:
            value = parameters.get(name)
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                limits[name] = self.parameter_panel.get_effective_limits(name)
        return limits

    def _set_export_status(self, message: str) -> None:
        self.export_panel.set_status(message)

    def _parameter_export_metadata(self, param_name: str) -> tuple[str, float | None, float | None]:
        """Return export metadata (units, min, max) for a model parameter."""

        parameter_def = self.parameter_panel.parameters.get(param_name)
        units = str(getattr(parameter_def, "units", "") or "")

        if param_name in self.parameter_panel.parameter_sliders:
            min_val, max_val = self.parameter_panel.get_effective_limits(param_name)
            return units, float(min_val), float(max_val)

        limits = getattr(parameter_def, "limits", None)
        if isinstance(limits, (list, tuple)) and len(limits) >= 2:
            try:
                low = float(limits[0])
                high = float(limits[1])
                if np.isfinite(low) and np.isfinite(high) and high > low:
                    return units, low, high
            except (TypeError, ValueError):
                pass

        return units, None, None

    @staticmethod
    def _dataset_scalar_to_python(value: object) -> ParameterValue:
        """Normalize HDF5 scalar payload values to Python scalars."""

        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, (str, bool, int, float)):
            return value
        return str(value)

    @staticmethod
    def _dataset_from_group(group: H5GroupLike | None, name: str) -> H5DatasetLike | None:
        """Return a named dataset from a group, or ``None`` when missing."""

        if group is None:
            return None
        node = group.get(name)
        if node is None:
            return None
        return cast(H5DatasetLike, node)

    @classmethod
    def _dataset_text(cls, group: H5GroupLike | None, name: str, default: str = "") -> str:
        """Read a string-like dataset value from a group with fallback default."""

        dataset = cls._dataset_from_group(group, name)
        if dataset is None:
            return default
        try:
            raw = dataset[()]
        except Exception:
            return default
        if isinstance(raw, np.ndarray) and raw.shape == ():
            raw = raw.item()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

    @classmethod
    def _dataset_float(cls, group: H5GroupLike | None, name: str, default: float) -> float:
        """Read a float dataset value with fallback default."""

        dataset = cls._dataset_from_group(group, name)
        if dataset is None:
            return default
        try:
            return float(cls._dataset_scalar_to_python(dataset[()]))
        except Exception:
            return default

    @classmethod
    def _parse_parameter_json(cls, payload: str) -> ParameterMapping:
        """Parse JSON mapping payload and coerce values to export-safe scalars."""

        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        if not isinstance(raw, dict):
            return {}

        parsed: ParameterMapping = {}
        for key, value in raw.items():
            parsed[str(key)] = cls._dataset_scalar_to_python(value)
        return parsed

    @staticmethod
    def _parse_string_list_json(payload: str) -> list[str]:
        """Parse a JSON list payload into a list of strings."""

        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw]

    def _write_model_group(self, sme_group: H5GroupLike, snapshot: ExportSnapshot) -> None:
        """Write model Q/I datasets with dataset-level units metadata."""

        model_group = sme_group.create_group("model")
        model_q_ds = model_group.create_dataset("Q", data=np.array(snapshot.q_values, copy=True, dtype=float))
        model_i_ds = model_group.create_dataset("I", data=np.array(snapshot.model_intensity, copy=True, dtype=float))
        model_q_ds.attrs["units"] = snapshot.q_unit
        model_i_ds.attrs["units"] = snapshot.intensity_unit

    def _write_parameters_group(self, sme_group: H5GroupLike, snapshot: ExportSnapshot, text_dtype: object) -> None:
        """Write parameter datasets with units and optional min/max attrs."""

        parameters_group = sme_group.create_group("parameters")
        for name, value in sorted(snapshot.parameters.items()):
            if isinstance(value, str):
                ds = parameters_group.create_dataset(name, data=value, dtype=text_dtype)
            else:
                ds = parameters_group.create_dataset(name, data=self._dataset_scalar_to_python(value))

            units, min_val, max_val = self._parameter_export_metadata(name)
            ds.attrs["units"] = units
            if min_val is not None:
                ds.attrs["min"] = float(min_val)
            if max_val is not None:
                ds.attrs["max"] = float(max_val)

    def _write_data_group(self, sme_group: H5GroupLike, snapshot: ExportSnapshot, text_dtype: object) -> None:
        """Write all loaded stage dataframes with units/uncertainty metadata."""

        data_group = sme_group.create_group("data")
        stage_bundles = self.data_panel.get_stage_bundles()
        stage_frames = self.data_panel.get_stage_frames()

        for stage_name, frame_obj in sorted(stage_frames.items()):
            frame = cast(FrameLike, frame_obj)
            stage_group = data_group.create_group(stage_name)
            stage_q_unit = snapshot.q_unit
            stage_i_unit = snapshot.intensity_unit
            stage_bundle_obj = stage_bundles.get(stage_name)
            if stage_bundle_obj is not None:
                stage_bundle = cast(Mapping[str, BaseDataLike], stage_bundle_obj)
                try:
                    q_node = stage_bundle.get("Q")
                    if q_node is not None:
                        stage_q_unit = str(getattr(q_node, "units", stage_q_unit))
                except Exception:
                    pass
                try:
                    i_node = stage_bundle.get("signal")
                    if i_node is None:
                        i_node = stage_bundle.get("I")
                    if i_node is not None:
                        stage_i_unit = str(getattr(i_node, "units", stage_i_unit))
                except Exception:
                    pass

            for column_name in frame.columns:
                values = np.asarray(frame[column_name])
                if np.issubdtype(values.dtype, np.number):
                    ds = stage_group.create_dataset(column_name, data=np.asarray(values, dtype=float))
                else:
                    text_values = np.asarray([str(item) for item in values], dtype=object)
                    ds = stage_group.create_dataset(column_name, data=text_values, dtype=text_dtype)

                if column_name in {"Q", "QSigma"}:
                    ds.attrs["units"] = stage_q_unit
                elif column_name in {"I", "ISigma"}:
                    ds.attrs["units"] = stage_i_unit

            if "I" in stage_group and "ISigma" in stage_group:
                cast(H5DatasetLike, stage_group["I"]).attrs["uncertainties"] = "ISigma"
            if "Q" in stage_group and "QSigma" in stage_group:
                cast(H5DatasetLike, stage_group["Q"]).attrs["uncertainties"] = "QSigma"

            if stage_bundle_obj is not None:
                description = str(getattr(stage_bundle_obj, "description", "") or "")
                if description:
                    stage_group.attrs["description"] = description

    def _write_settings_group(self, sme_group: H5GroupLike, snapshot: ExportSnapshot, text_dtype: object) -> None:
        """Write non-model settings payload under SME_settings."""

        settings_group = sme_group.create_group("SME_settings")
        settings_group.create_dataset("schema_version", data=2)
        settings_group.create_dataset("q_unit", data=snapshot.q_unit, dtype=text_dtype)
        settings_group.create_dataset("intensity_unit", data=snapshot.intensity_unit, dtype=text_dtype)
        settings_group.create_dataset("q_min", data=float(self.q_min_input.text()))
        settings_group.create_dataset("q_max", data=float(self.q_max_input.text()))
        settings_group.create_dataset(
            "selected_data_stage",
            data=str(self.data_panel.data_mode_combo.currentData()),
            dtype=text_dtype,
        )
        settings_group.create_dataset(
            "fit_parameter_names_json",
            data=json.dumps(self.fit_panel.get_selected_parameters()),
            dtype=text_dtype,
        )
        settings_group.create_dataset(
            "hidden_parameter_defaults_json",
            data=json.dumps(self.hidden_parameter_defaults),
            dtype=text_dtype,
        )
        settings_group.create_dataset(
            "data_loading_yaml",
            data=self.data_panel.get_yaml_config_text(),
            dtype=text_dtype,
        )

        preset_name = self.data_panel.get_selected_preset_name()
        if preset_name:
            settings_group.create_dataset("data_loading_preset", data=preset_name, dtype=text_dtype)
        source_file = self.data_panel.file_path_line.text().strip()
        if source_file:
            settings_group.create_dataset("data_source_file", data=source_file, dtype=text_dtype)
        data_mode_text = self.data_panel.data_mode_combo.currentText()
        settings_group.create_dataset("data_mode_label", data=data_mode_text, dtype=text_dtype)
        settings_group.create_dataset("exporter", data="SasModelsExplorer", dtype=text_dtype)

    def _write_state_hdf5_tree(self, h5f: H5GroupLike, snapshot: ExportSnapshot) -> None:
        """Write the complete /analyses/SasModelsExplorer state tree."""

        import h5py

        text_dtype = h5py.string_dtype(encoding="utf-8")
        analyses_group = h5f.create_group("analyses")
        sme_group = analyses_group.create_group("SasModelsExplorer")
        sme_group.create_dataset("model_name", data=snapshot.model_expression, dtype=text_dtype)

        self._write_model_group(sme_group, snapshot)
        self._write_parameters_group(sme_group, snapshot, text_dtype)
        self._write_data_group(sme_group, snapshot, text_dtype)
        self._write_settings_group(sme_group, snapshot, text_dtype)

    def _save_state_hdf5(self) -> None:
        """Save current model/session state (and optional overlay data) to an HDF5 file."""

        if self.model is None or self.kernel is None:
            QMessageBox.warning(self, "Save State", "Load a model before saving state.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Explorer State",
            "",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith((".h5", ".hdf5")):
            file_path = f"{file_path}.h5"

        snapshot = self.build_export_snapshot()

        try:
            import h5py

            with h5py.File(file_path, "w") as h5f:
                self._write_state_hdf5_tree(h5f, snapshot)
        except Exception as exc:
            QMessageBox.critical(self, "Save State", f"Failed to save state: {exc}")
            self._set_export_status(f"Save failed: {exc}")
            return

        self._set_export_status(f"Saved state to {file_path}")

    def _read_new_state_hdf5(
        self,
        h5f: H5GroupLike,
        file_path: str,
        base_data_cls: BaseDataFactory,
        data_bundle_cls: DataBundleFactory,
    ) -> LoadedExplorerState:
        """Read state payload from the /analyses/SasModelsExplorer tree."""

        sme_group = cast(H5GroupLike, h5f["/analyses/SasModelsExplorer"])
        settings_group = cast(H5GroupLike | None, sme_group.get("SME_settings"))
        model_group = cast(H5GroupLike | None, sme_group.get("model"))

        model_expression = self._dataset_text(sme_group, "model_name", "").strip()
        if not model_expression:
            raise ValueError("State file does not contain /analyses/SasModelsExplorer/model_name.")

        model_q_unit = ""
        model_i_unit = ""
        if model_group is not None:
            if "Q" in model_group:
                model_q_unit = str(cast(H5DatasetLike, model_group["Q"]).attrs.get("units", ""))
            if "I" in model_group:
                model_i_unit = str(cast(H5DatasetLike, model_group["I"]).attrs.get("units", ""))

        q_unit = self._dataset_text(settings_group, "q_unit", model_q_unit or "1/nm")
        intensity_unit = self._dataset_text(settings_group, "intensity_unit", model_i_unit or self.i_units[0])
        q_min = self._dataset_float(settings_group, "q_min", 0.01)
        q_max = self._dataset_float(settings_group, "q_max", 10.0)

        parameters: ParameterMapping = {}
        parameters_group = cast(H5GroupLike | None, sme_group.get("parameters"))
        if parameters_group is not None:
            for item_name in parameters_group.keys():
                dataset = cast(H5DatasetLike, parameters_group[item_name])
                raw_value = self._dataset_scalar_to_python(dataset[()])
                parameters[str(item_name)] = raw_value

        hidden_defaults = SasModelApp._parse_parameter_json(
            self._dataset_text(settings_group, "hidden_parameter_defaults_json", "{}")
        )
        fit_names = SasModelApp._parse_string_list_json(
            self._dataset_text(settings_group, "fit_parameter_names_json", "[]")
        )
        source_file = self._dataset_text(settings_group, "data_source_file", "").strip()
        stage_name = self._dataset_text(settings_group, "selected_data_stage", "").strip()
        yaml_text = self._dataset_text(settings_group, "data_loading_yaml", "")
        yaml_preset = self._dataset_text(settings_group, "data_loading_preset", "")

        stage_bundles: StageBundleMap = {}
        data_group = cast(H5GroupLike | None, sme_group.get("data"))
        if data_group is not None:
            for candidate_stage in data_group.keys():
                stage_group = cast(H5GroupLike, data_group[candidate_stage])
                if "Q" not in stage_group or "I" not in stage_group:
                    continue
                q_dataset = cast(H5DatasetLike, stage_group["Q"])
                i_dataset = cast(H5DatasetLike, stage_group["I"])
                q_data = np.asarray(q_dataset[()], dtype=float)
                i_data = np.asarray(i_dataset[()], dtype=float)
                if q_data.size == 0 or i_data.size == 0:
                    continue
                i_sigma = (
                    np.asarray(cast(H5DatasetLike, stage_group["ISigma"])[()], dtype=float)
                    if "ISigma" in stage_group
                    else None
                )
                stage_q_unit = str(q_dataset.attrs.get("units", q_unit))
                stage_i_unit = str(i_dataset.attrs.get("units", intensity_unit))
                q_obj = base_data_cls(signal=q_data, units=normalize_unit_label(stage_q_unit))
                i_unc = {} if i_sigma is None else {"ISigma": i_sigma}
                i_obj = base_data_cls(signal=i_data, units=stage_i_unit, uncertainties=i_unc)
                bundle = data_bundle_cls({"Q": q_obj, "signal": i_obj})
                description = str(stage_group.attrs.get("description", "")).strip()
                if description:
                    bundle.description = description  # type: ignore[attr-defined]
                else:
                    bundle.description = f"Imported from {Path(file_path).name} ({candidate_stage})"  # type: ignore[attr-defined]
                stage_bundles[str(candidate_stage)] = bundle

        return LoadedExplorerState(
            model_expression=model_expression,
            q_unit=q_unit,
            intensity_unit=intensity_unit,
            q_min=q_min,
            q_max=q_max,
            parameters=parameters,
            hidden_defaults=hidden_defaults,
            fit_names=fit_names,
            source_file=source_file,
            stage_name=stage_name,
            yaml_text=yaml_text,
            yaml_preset=yaml_preset,
            stage_bundles=stage_bundles,
        )

    def _read_state_hdf5(
        self,
        h5f: H5GroupLike,
        file_path: str,
        base_data_cls: BaseDataFactory,
        data_bundle_cls: DataBundleFactory,
    ) -> LoadedExplorerState:
        """Read state payload from the current HDF5 schema."""

        if "/analyses/SasModelsExplorer" not in h5f:
            raise ValueError(
                "Unsupported state schema. Expected group '/analyses/SasModelsExplorer'. "
                "Please re-export state files with the current SasModelsExplorer version."
            )
        return self._read_new_state_hdf5(h5f, file_path, base_data_cls, data_bundle_cls)

    def _load_state_hdf5(self) -> None:
        """Load model/session state from an HDF5 file and update the current UI."""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Explorer State",
            "",
            "HDF5 Files (*.h5 *.hdf5 *.nxs *.nx);;All Files (*)",
        )
        if not file_path:
            return

        try:
            import h5py
            from mcsas3.data_model import BaseData, DataBundle

            with h5py.File(file_path, "r") as h5f:
                state = self._read_state_hdf5(
                    cast(H5GroupLike, h5f),
                    file_path,
                    cast(BaseDataFactory, BaseData),
                    cast(DataBundleFactory, DataBundle),
                )
        except Exception as exc:
            QMessageBox.critical(self, "Load State", f"Failed to load state: {exc}")
            self._set_export_status(f"Load failed: {exc}")
            return

        self.model_input.setText(state.model_expression)
        self.load_model_parameters()
        self.q_min_input.setText(f"{state.q_min:.6g}")
        self.q_max_input.setText(f"{state.q_max:.6g}")
        self._set_q_unit_selection(state.q_unit)

        self.hidden_parameter_defaults = dict(state.hidden_defaults)

        visible_values = {
            name: value for name, value in state.parameters.items() if name in self.parameter_panel.parameters
        }
        self.parameter_panel.set_values(visible_values, emit_change=False)

        selected_names = [name for name in state.fit_names if name in self.parameter_panel.parameters]
        self.fit_panel.set_selected_parameters(selected_names)

        if state.source_file:
            self.data_panel.file_path_line.setText(state.source_file)
        if state.yaml_text:
            self.data_panel.set_yaml_config_text(
                state.yaml_text,
                mark_custom=True,
                prefer_matching_preset=True,
                preferred_preset_name=state.yaml_preset or None,
            )

        if state.stage_bundles:
            selected_stage = (
                state.stage_name if state.stage_name in state.stage_bundles else next(iter(state.stage_bundles))
            )
            self.data_panel.set_stage_bundles(
                cast(dict[str, object], state.stage_bundles),
                selected_stage=selected_stage,
                message="Loaded stage data from state file.",
            )
        else:
            self.data_panel.set_data_bundle(None, message="Loaded state without overlay data.")

        self.update_kernel_and_plot()
        self._set_export_status(f"Loaded state from {file_path}")

    def _export_csv_parameters(self) -> None:
        """Export current model parameter table to CSV."""

        if self.model is None or self.kernel is None:
            QMessageBox.warning(self, "Export CSV", "Load a model before exporting.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Parameters to CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".csv"):
            file_path = f"{file_path}.csv"

        snapshot = self.build_export_snapshot()
        table = build_csv_model_parameter_table(snapshot)
        try:
            with Path(file_path).open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([f"# model_name: {snapshot.model_expression}"])
                writer.writerow(table.headers)
                writer.writerows(table.rows)
        except Exception as exc:
            QMessageBox.critical(self, "Export CSV", f"Failed to export CSV: {exc}")
            self._set_export_status(f"CSV export failed: {exc}")
            return

        self._set_export_status(f"Exported parameters to {file_path}")

    def _export_mcsas3_run_configuration(self) -> None:
        """Export a McSAS3 run-configuration YAML file from current model/parameter state."""

        if self.model is None:
            QMessageBox.warning(self, "Export Run Configuration", "Load a model before exporting.")
            return

        parameters = self._current_parameter_values()
        float_limits = self._float_parameter_limits(parameters)
        if not float_limits:
            QMessageBox.warning(
                self,
                "Export Run Configuration",
                "No float parameters available for optimization.",
            )
            return

        dialog = RunConfigExportDialog(float_limits, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        choice = dialog.selected_choice()
        try:
            run_config = build_mcsas3_run_configuration(
                model_expression=self.model_input.text().strip(),
                parameters=parameters,
                fit_parameter=choice.fit_parameter,
                fit_limits=(choice.lower_limit, choice.upper_limit),
                log_random=choice.log_random,
                n_rep=10,
                max_iter=100000,
                max_accept=10000,
                n_contrib=300,
                conv_crit=1.0,
                n_cores=5,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Export Run Configuration", str(exc))
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export McSAS3 Run Configuration",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith((".yaml", ".yml")):
            file_path = f"{file_path}.yaml"

        try:
            import yaml

            Path(file_path).write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Export Run Configuration", f"Failed to export run configuration: {exc}")
            self._set_export_status(f"Run configuration export failed: {exc}")
            return

        self._set_export_status(f"Exported run configuration to {file_path}")

    def build_export_snapshot(self) -> ExportSnapshot:
        """Build a backend export snapshot from the current UI state."""

        if self.model is None or self.kernel is None:
            raise ValueError("Model must be loaded before creating an export snapshot.")

        model_expression = self.model_input.text().strip()
        parameters = self._current_parameter_values()
        q_values = np.asarray(self.q, dtype=float)
        model_intensity = np.asarray(compute_intensity(self.kernel, parameters), dtype=float)
        data_bundle = cast(Mapping[str, BaseDataLike] | None, self.data_panel.get_data_bundle())
        overlay_data = overlay_from_bundle(data_bundle, self.qunit, self.i_units[0])

        session = ModelSessionState(
            model_expression=model_expression,
            q_unit=self.qunit,
            q_min=float(self.q_min_input.text()),
            q_max=float(self.q_max_input.text()),
            parameters=parameters,
            hidden_defaults=dict(self.hidden_parameter_defaults),
        )
        data_state = DataSelectionState(
            source_file=Path(self.data_panel.file_path_line.text()).expanduser()
            if self.data_panel.file_path_line.text().strip()
            else None,
            stage_name=str(self.data_panel.data_mode_combo.currentData()),
            points_loaded=int(0 if overlay_data is None else overlay_data.Q.size),
            description=str("" if overlay_data is None else overlay_data.label),
        )
        return build_export_snapshot_payload(
            SnapshotInputs(
                model_expression=session.model_expression,
                q_unit=session.q_unit,
                intensity_unit=self.i_units[0],
                parameters=session.parameters,
                fit_parameter_names=self.fit_panel.get_selected_parameters(),
                q_values=q_values,
                model_intensity=model_intensity,
                overlay_data=overlay_data,
                metadata={
                    "q_min": session.q_min,
                    "q_max": session.q_max,
                    "data_stage": data_state.stage_name,
                    "data_points_loaded": data_state.points_loaded,
                },
            )
        )

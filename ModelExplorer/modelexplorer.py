# ModelExplorer/modelexplorer.py

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .data_loading_panel import DataLoadingPanel
from .fitting_panel import FittingPanel
from .modelbrowser import ModelBrowser
from .parameter_panel import ParameterPanel
from .plotting import PlotManager
from .sasmodels_adapter import (
    build_parameter_list,
    compute_intensity,
    generate_model_info_text,
    load_model_and_info,
    split_magnetic_parameters,
)
from .services.export_snapshot_builder import (
    SnapshotInputs,
)
from .services.export_snapshot_builder import (
    build_export_snapshot as build_export_snapshot_payload,
)
from .services.fitting_engine import fit_model
from .services.model_parameters import ensure_pd_parameter_defaults, merge_parameter_values
from .services.overlay_processing import overlay_from_bundle, reduced_chi_square
from .types import DataSelectionState, ModelSessionState
from .utils.units import MODEL_INTENSITY_SCALE, create_unit_registry, normalize_unit_label

ureg = create_unit_registry()


class SasModelApp(QMainWindow):
    """Main PyQt window that wires model inputs, parameter panel, and plotting."""

    q: np.ndarray = None
    model = None
    kernel = None
    model_info = None
    model_parameters = None
    pd_types: list[str] = ["uniform", "rectangle", "gaussian", "lognormal", "schulz", "boltzmann"]
    q_units: list[str] = ["1/nm", "1/Ångström", "1/m"]
    i_units: list[str] = ["1/(m sr)", "1/(cm sr)"]
    qunit: str = None
    infoText: str = None

    def __init__(self, modelName: str = "sphere") -> None:
        """Initialize the UI, wire signals, and load the initial model."""
        super().__init__()
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
        self.hidden_parameter_defaults = {}

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

        # Model browser (now embedded as a tab)
        self.model_browser = ModelBrowser(parent=self)
        self.model_browser.model_selected.connect(self.append_model_text)

        self.side_tabs = QTabWidget()
        self.side_tabs.addTab(self.model_browser, "Models")
        self.side_tabs.addTab(self.data_panel, "Data")
        self.side_tabs.addTab(self.fit_panel, "Fitting")
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

    def load_model_parameters(self, *_) -> None:
        """Load model info, rebuild parameter controls, and trigger a plot refresh."""
        previous_values = {}
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
            self.model_parameters = self.model_info.parameters.defaults.copy()

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

        self.kernel = self.model.make_kernel([self.q * ureg.Quantity(1, qunit).to("1/Ang").magnitude])
        self.update_plot()

    def update_plot(self) -> None:
        """Compute intensity from current parameters and redraw the plot."""

        if self.model is None:
            return

        logging.info("updating plot")

        parameters = merge_parameter_values(self.parameter_panel.get_values(), self.hidden_parameter_defaults)
        ensure_pd_parameter_defaults(parameters, getattr(self.model.info.parameters, "pd_1d", []))

        # Compute intensity
        Q = self.q
        Q_unit = self.qunit
        kernel = self.kernel
        logging.info("calling sasmodels with %s parameters", len(parameters))
        I_model = compute_intensity(kernel, parameters)

        data_bundle = self.data_panel.get_data_bundle()
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
        data_bundle = self.data_panel.get_data_bundle()
        overlay_data = overlay_from_bundle(data_bundle, self.qunit, self.i_units[0])
        if overlay_data is None:
            self.fit_panel.set_status("Load data before fitting.")
            return

        parameters = merge_parameter_values(self.parameter_panel.get_values(), self.hidden_parameter_defaults)
        ensure_pd_parameter_defaults(parameters, getattr(self.model.info.parameters, "pd_1d", []))
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

    def build_export_snapshot(self) -> object:
        """Build a backend export snapshot from the current UI state."""

        if self.model is None or self.kernel is None:
            raise ValueError("Model must be loaded before creating an export snapshot.")

        model_expression = self.model_input.text().strip()
        parameters = merge_parameter_values(self.parameter_panel.get_values(), self.hidden_parameter_defaults)
        ensure_pd_parameter_defaults(parameters, getattr(self.model.info.parameters, "pd_1d", []))
        q_values = np.asarray(self.q, dtype=float)
        model_intensity = np.asarray(compute_intensity(self.kernel, parameters), dtype=float)
        overlay_data = overlay_from_bundle(self.data_panel.get_data_bundle(), self.qunit, self.i_units[0])

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

# ModelExplorer/modelexplorer.py

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .data_loading_panel import DataLoadingPanel
from .fitting_panel import FittingPanel
from .parameter_panel import ParameterPanel
from .plotting import PlotManager
from .sasmodels_adapter import (
    build_parameter_list,
    compute_intensity,
    generate_model_info_text,
    load_model_and_info,
    split_magnetic_parameters,
)
from .services.fitting_engine import fit_model
from .types import OverlayData
from .utils.units import MODEL_INTENSITY_SCALE, create_unit_registry, normalize_unit_label

ureg = create_unit_registry()


class SasModelApp(QMainWindow):
    """Main PyQt window that wires model inputs, parameter panel, and plotting."""

    q: np.ndarray = None
    model = None
    kernel = None
    model_info = None
    model_parameters = None
    pd_types: List = ["uniform", "rectangle", "gaussian", "lognormal", "schulz", "boltzmann"]
    q_units: List = ["1/nm", "1/Ångström", "1/m"]
    i_units: List = ["1/(m sr)", "1/(cm sr)"]
    qunit: str = None
    infoText: str = None

    def __init__(self, modelName: str = "sphere") -> None:
        """Initialize the UI, wire signals, and load the initial model."""
        super().__init__()
        self.setWindowTitle("SasModels Explorer")

        # generate the infoText:
        self.infoText = generate_model_info_text()

        # Left layout for controls
        self.parameter_panel = ParameterPanel(on_change=self.update_plot, width=450)

        # Text input for model
        self.model_input = QLineEdit(modelName)
        self.model_input.setFixedWidth(300)
        self.parameter_panel.add_header_row("Model:", self.model_input)
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
        self.q_max_input = QLineEdit("1.0")
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

        self.side_tabs = QTabWidget()
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

    def generate_infotext(self) -> str:
        """Return the help text shown when a model name is invalid."""
        return generate_model_info_text()

    def load_model_parameters(self) -> None:
        """Load model info, rebuild parameter controls, and trigger a plot refresh."""
        previous_values = {}
        if self.parameter_panel is not None:
            previous_values = self.parameter_panel.get_values()
            previous_values.update(self.hidden_parameter_defaults)
        previous_fit_selection = self.fit_panel.get_selected_parameters()
        model_name = self.model_input.text()
        try:
            # Load model info from sasmodels
            self.model, self.model_info = load_model_and_info(model_name)
            self.model_parameters = self.model_info.parameters.defaults.copy()

            parameters = build_parameter_list(self.model, self.model_info, self.pd_types)
            visible_parameters, hidden_defaults = split_magnetic_parameters(
                parameters, self.show_magnetic_checkbox.isChecked()
            )
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
            # Initial plot
            self.update_model_and_plot()

        except Exception as e:
            logging.warning(f"Error loading model '{model_name}': {e}")
            self.hidden_parameter_defaults = {}

            # let's create a dialog box to inform the user of the error
            # Create a custom font
            # ---------------------
            font = QFont()
            font.setFamily("Courier")
            font.setPointSize(9)

            dialog = QMessageBox(self)
            dialog.setWindowTitle("Invalid model")
            dialog.setText(f"Could not load model '{model_name}': {e}")
            dialog.setIcon(QMessageBox.Warning)
            dialog.setStandardButtons(QMessageBox.Ok)
            dialog.setInformativeText(self.infoText)

            button = dialog.exec()
            if button:
                dialog.close()
            else:
                dialog.close()
            # if button == QMessageBox.Help:

    def update_model_and_plot(self) -> None:
        """Rebuild the kernel (if needed) and refresh the plot."""
        self.update_kernel_and_plot()

    def update_kernel_and_plot(self) -> None:
        """Update the sasmodels kernel using the current q range/units."""
        # Retrieve and validate q range and units
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
        logging.info("updating plot")

        parameters = self.parameter_panel.get_values()
        parameters.update(self.hidden_parameter_defaults)

        # Update the model parameters
        # find names that are in the polydisperse parameter list
        pd_params = [param for param in parameters.keys() if param in self.model.info.parameters.pd_1d]

        for param in pd_params:
            # for each parameter, add a _pd_type and a _pd_n
            pd_n = param + "_pd_n"
            parameters[pd_n] = 35

        # Compute intensity
        Q = self.q
        Q_unit = self.qunit
        # model = self.model
        kernel = self.kernel
        logging.info(f"calling sasmodels with {[{p: v} for p, v in parameters.items()]}")
        I_model = compute_intensity(kernel, parameters)

        data_bundle = self.data_panel.get_data_bundle()
        overlay_data = self._prepare_overlay_data(data_bundle, Q_unit, self.i_units[0])
        chi2, dof, points = self._compute_reduced_chi_square(Q, I_model, overlay_data)
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

    def _prepare_overlay_data(
        self, data_bundle: Optional[Any], target_Q_unit: str, target_I_unit: str
    ) -> Optional[OverlayData]:
        if data_bundle is None:
            return None
        data_I = data_bundle.get("I")
        data_Q = data_bundle.get("Q")
        if data_I is None or data_Q is None:
            return None

        I_copy = self._copy_basedata(data_I)
        Q_copy = self._copy_basedata(data_Q)

        target_Q_unit = normalize_unit_label(target_Q_unit)
        try:
            Q_copy.to_units(target_Q_unit)
            I_copy.to_units(target_I_unit)
        except Exception:
            pass

        Q_vals = np.asarray(Q_copy.signal, dtype=float)
        I_vals = np.asarray(I_copy.signal, dtype=float)
        sigma = None
        if "ISigma" in I_copy.uncertainties:
            sigma = np.asarray(I_copy.uncertainties["ISigma"], dtype=float)

        mask = np.isfinite(Q_vals) & np.isfinite(I_vals)
        if sigma is not None:
            mask &= np.isfinite(sigma)

        Q_vals = Q_vals[mask]
        I_vals = I_vals[mask]
        if sigma is not None:
            sigma = sigma[mask]

        if Q_vals.size == 0:
            return None

        order = np.argsort(Q_vals)
        Q_vals = Q_vals[order]
        I_vals = I_vals[order]
        if sigma is not None:
            sigma = sigma[order]

        label = getattr(data_bundle, "description", None) or "Data"
        return OverlayData(Q=Q_vals, I=I_vals, ISigma=sigma, label=label)

    def _copy_basedata(self, source: Any) -> Any:
        uncertainties = {key: np.array(val, copy=True) for key, val in source.uncertainties.items()}
        weights = np.array(getattr(source, "weights", 1.0), copy=True)
        return source.__class__(
            signal=np.array(source.signal, copy=True),
            units=source.units,
            uncertainties=uncertainties,
            weights=weights,
            rank_of_data=getattr(source, "rank_of_data", 0),
        )

    def _compute_reduced_chi_square(
        self,
        Q: np.ndarray,
        I: np.ndarray,  # noqa: E741
        data: Optional[OverlayData],
    ) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        if data is None:
            return None, None, None
        if data.ISigma is None:
            return None, None, None

        try:
            model_I = self._interpolate_model(Q, I, data.Q) * self.plot_manager.scale
        except Exception:
            return None, None, None

        mask = np.isfinite(model_I) & np.isfinite(data.I) & np.isfinite(data.ISigma) & (data.ISigma > 0)
        if mask.sum() < 2:
            return None, None, None

        n_params = len(self.parameter_panel.get_values())
        dof = max(int(mask.sum()) - n_params, 1)
        chi2 = np.sum(((data.I[mask] - model_I[mask]) / data.ISigma[mask]) ** 2) / dof
        return float(chi2), dof, int(mask.sum())

    def _interpolate_model(self, Q_model: np.ndarray, I_model: np.ndarray, Q_data: np.ndarray) -> np.ndarray:
        Q_model = np.asarray(Q_model, dtype=float)
        I_model = np.asarray(I_model, dtype=float)
        Q_data = np.asarray(Q_data, dtype=float)
        if np.any(Q_model <= 0) or np.any(I_model <= 0) or np.any(Q_data <= 0):
            return np.interp(Q_data, Q_model, I_model, left=np.nan, right=np.nan)
        log_Q = np.log10(Q_model)
        log_I = np.log10(I_model)
        log_Qd = np.log10(Q_data)
        log_I_interp = np.interp(log_Qd, log_Q, log_I, left=np.nan, right=np.nan)
        return 10**log_I_interp

    def _run_fit(self) -> None:
        fit_names = self.fit_panel.get_selected_parameters()
        data_bundle = self.data_panel.get_data_bundle()
        overlay_data = self._prepare_overlay_data(data_bundle, self.qunit, self.i_units[0])
        if overlay_data is None:
            self.fit_panel.set_status("Load data before fitting.")
            return

        parameters = self.parameter_panel.get_values()
        parameters.update(self.hidden_parameter_defaults)
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

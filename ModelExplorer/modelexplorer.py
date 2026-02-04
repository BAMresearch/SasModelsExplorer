# ModelExplorer/modelexplorer.py

import logging
from typing import List
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox,
                             QLabel, QLineEdit, QComboBox, QSplitter, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np
import pint
ureg = pint.UnitRegistry(auto_reduce_dimensions=True)
ureg.define(r"percent = 0.01 = %")
ureg.define(r"Ångström = 1e-10*m = Å = Ang = Angstrom")
ureg.define(r"item = 1")
from .parameter_panel import ParameterPanel
from .plotting import PlotManager
from .sasmodels_adapter import (
    build_parameter_list,
    compute_intensity,
    generate_model_info_text,
    load_model_and_info,
    split_magnetic_parameters,
)

class SasModelApp(QMainWindow):
    """Main PyQt window that wires model inputs, parameter panel, and plotting."""
    q:np.ndarray = None
    model = None
    kernel = None
    model_info = None
    model_parameters = None
    pd_types:List = ['uniform', 'rectangle', 'gaussian', 'lognormal', 'schulz', 'boltzmann']
    q_units:List = ['1/nm', '1/Ångström', '1/m']
    i_units:List = ['1/(m sr)', '1/(cm sr)']
    qunit:str = None
    infoText:str = None
    
    def __init__(self, modelName:str="sphere") -> None:
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
        self.show_magnetic_checkbox.setChecked(True)
        self.show_magnetic_checkbox.stateChanged.connect(self.load_model_parameters)
        self.parameter_panel.add_header_row("Magnetic:", self.show_magnetic_checkbox)
        self.hidden_parameter_defaults = {}

        # Right layout for plot
        self.plot_manager = PlotManager(figsize=(6, 4))

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
        q_range_layout.addWidget(QLabel("Q Min:"))
        q_range_layout.addWidget(self.q_min_input)
        q_range_layout.addWidget(self.q_unit_input)
        q_range_layout.addWidget(QLabel("Q Max:"))
        q_range_layout.addWidget(self.q_max_input)

        # Vertical layout for plot and q range controls
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.plot_manager.canvas)
        plot_layout.addLayout(q_range_layout)

        plot_container = QWidget()
        plot_container.setLayout(plot_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.parameter_panel)
        splitter.addWidget(plot_container)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self.setCentralWidget(splitter)

        # Load initial model
        self.model_info = None
        self.model_parameters = None
        self.load_model_parameters()


    def generate_infotext(self)->str:
        """Return the help text shown when a model name is invalid."""
        return generate_model_info_text()


    def load_model_parameters(self) -> None:
        """Load model info, rebuild parameter controls, and trigger a plot refresh."""
        model_name = self.model_input.text()
        try:
            # Load model info from sasmodels
            self.model, self.model_info = load_model_and_info(model_name)
            self.model_parameters = self.model_info.parameters.defaults.copy()

            parameters = build_parameter_list(self.model, self.model_info, self.pd_types)
            visible_parameters, hidden_defaults = split_magnetic_parameters(
                parameters, self.show_magnetic_checkbox.isChecked()
            )
            self.hidden_parameter_defaults = hidden_defaults
            self.parameter_panel.set_parameters(visible_parameters)
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
        logging.info(f'updating kernel')
        try:
            qmin = float(self.q_min_input.text())
            qmax = float(self.q_max_input.text())
            qunit = self.q_unit_input.currentText()
        except ValueError:
            qmin, qmax = 0.01, 1.0  # Default values in case of error
            qunit = '1/nm'
        
        # Prepare parameters for sasmodel
        self.q = np.geomspace(qmin, qmax, 250)
        self.qunit = qunit
        self.kernel = self.model.make_kernel([self.q * ureg.Quantity(1, qunit).to('1/Ang').magnitude])
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
        q=self.q
        qunit=self.qunit
        # model = self.model
        kernel = self.kernel
        logging.info(f'calling sasmodels with {[{p: v} for p, v in parameters.items()]}')
        intensity = compute_intensity(kernel, parameters)

        self.plot_manager.plot(q, intensity, qunit)

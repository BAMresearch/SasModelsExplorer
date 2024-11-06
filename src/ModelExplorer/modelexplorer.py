import logging
import re
from typing import List
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox,
                             QLabel, QLineEdit, QPushButton, QFormLayout, QSlider, QScrollArea)
from PyQt5.QtCore import Qt
import sasmodels.core
import sasmodels.direct_model
import numpy as np
import matplotlib.pyplot as plt
# import sip
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pint
ureg = pint.UnitRegistry(auto_reduce_dimensions=True)
ureg.define(r"percent = 0.01 = %")
ureg.define(r"Ångström = 1e-10*m = Å = Ang = Angstrom")
ureg.define(r"item = 1")

class SasModelApp(QMainWindow):
    model = None
    model_info = None
    model_parameters = None
    parameters = dict() # the actual parameter object references in the UI
    parameter_choosers = None # for pulldown box choices
    parameter_sliders = None # for sliders
    parameter_inputs = None # sliders have a linked text input box 
    # Pattern list to exclude specific parameters
    exclude_patterns = [r'up_.*', r'.*_M0', r'.*_mtheta', r'.*_mphi']
    pd_types = ['uniform', 'rectangle', 'gaussian', 'lognormal', 'schulz', 'boltzmann']
    q_units = ['1/nm', '1/Ångström']
    i_units = ['1/(m sr)', '1/(cm sr)']
    
    def __init__(self, modelName="sphere"):
        super().__init__()
        self.setWindowTitle("SasModels Interactive App")

        # Main layout
        main_layout = QHBoxLayout()

        # Left layout for controls
        self.control_layout = QFormLayout()

        # Text input for model
        self.model_input = QLineEdit(modelName)
        self.model_input.setFixedWidth(300)
        self.control_layout.addRow("Model:", self.model_input)
        self.model_input.returnPressed.connect(self.load_model_parameters)

        # Scroll area for parameters
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.control_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(460)

        # Placeholder for dynamically added sliders and input boxes
        self.parameter_sliders = {}
        self.parameter_choosers = {}
        self.parameter_inputs = {}

        # Right layout for plot
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)

        # qmin and qmax inputs below the plot
        self.q_min_input = QLineEdit("0.01")
        self.q_max_input = QLineEdit("1.0")
        self.q_min_input.setFixedWidth(80)
        self.q_max_input.setFixedWidth(80)
        self.q_min_input.editingFinished.connect(self.update_plot)
        self.q_max_input.editingFinished.connect(self.update_plot)
        self.q_unit_input = self.create_pulldown_menu_elements(self.q_units)[0]

        # Layout for q range inputs
        q_range_layout = QHBoxLayout()
        q_range_layout.addWidget(QLabel("Q Min:"))
        q_range_layout.addWidget(self.q_min_input)
        q_range_layout.addWidget(self.q_unit_input)
        q_range_layout.addWidget(QLabel("Q Max:"))
        q_range_layout.addWidget(self.q_max_input)

        # Vertical layout for plot and q range controls
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)
        plot_layout.addLayout(q_range_layout)

        # Add widgets to the main layout
        main_layout.addWidget(scroll_area, 1)
        main_layout.addLayout(plot_layout, 3)

        # Central widget to hold the main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Load initial model
        self.model_info = None
        self.model_parameters = {}
        self.load_model_parameters()

    def remove_layout_and_widgets(self, item, starting_index:int=0):
        # Clear only parameter-specific widgets in the control layout
        if hasattr(item, "layout"):
            if callable(item.layout):
                layout = item.layout()
        else:
            layout = None

        if hasattr(item, "widget"):
            if callable(item.widget):
                widget = item.widget()
        else:
            widget = None

        if widget:
            widget.deleteLater()
            widget=None
            # widget.setParent(None)
        elif layout:
            for i in reversed(range(layout.count())):
                if i>=starting_index:
                    self.remove_layout_and_widgets(layout.itemAt(i))

    def load_model_parameters(self):
        model_name = self.model_input.text()
        try:
            # Load model info from sasmodels
            self.model = sasmodels.core.load_model(model_name)
            self.model_info = sasmodels.core.load_model_info(model_name)
            self.model_parameters = self.model_info.parameters.defaults.copy()

            self.remove_layout_and_widgets(self.control_layout, starting_index=2)
            # Reset the parameter-specific dictionaries to clear any previous model data
            self.parameter_sliders.clear()
            self.parameter_choosers.clear()
            self.parameter_inputs.clear()
            self.parameters.clear()

            # Dynamically add sliders and input boxes for each model parameter
            for parameter in self.model.info.parameters.common_parameters+self.model.info.parameters.kernel_parameters:
                self.parameters[parameter.name] = parameter
                # Add the parameter layout for the current parameter
                param_layout = self.create_parameter_input_element(parameter)
                self.control_layout.addRow(param_layout)

                # Check if the current parameter is a polydisperse parameter and, if so, add the relevant polydisperse choices
                if parameter.name in self.model.info.parameters.pd_1d: # polydisperse parameters
                    new_param = sasmodels.modelinfo.Parameter(
                        parameter.name + "_pd",
                        units='',
                        default=0,
                        limits=(0, 1),
                        description=f'relative polydispersity of parameter {parameter.name}',
                    )
                    radius_pd_layout = self.create_parameter_input_element(new_param)
                    self.control_layout.addRow(radius_pd_layout)
                    self.parameters[new_param.name] = new_param
                    new_param = sasmodels.modelinfo.Parameter(
                        parameter.name + "_pd_type",
                        limits = [[self.pd_types]],
                        units='',
                        default='gaussian',
                        description=f'polydispersity distribution shape for parameter {parameter.name}',
                    )
                    new_param.choices = self.pd_types
                    radius_pd_layout = self.create_parameter_input_element(new_param)
                    self.control_layout.addRow(radius_pd_layout)
                    self.parameters[new_param.name] = new_param


            logging.info(f'parameters listed in self.parameters: {self.parameters.keys()}')
            # Initial plot
            self.update_plot()
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")

    def create_parameter_input_element(self, parameter: sasmodels.modelinfo.Parameter):
        assert isinstance(parameter, sasmodels.modelinfo.Parameter), 'parameter supplied to create_parameter must be a sasmodels parameter'
        logging.info(f"Creating parameter input element for {parameter}")
        # depending on the parameter choices and limits, create a pulldown box or slider
        param_layout = QHBoxLayout()

        label = QLabel(parameter.name)
        label.setToolTip(parameter.description)
        label.setFixedWidth(100)

        if len(parameter.choices)>0: # create a pulldown menu with the choices
            logging.debug(f"Creating pulldown for parameter {parameter}")
            adjustment_elements = self.create_pulldown_menu_elements(parameter.choices)
            self.parameter_choosers[parameter.name] = adjustment_elements[0]
        else: # create a log slider adhering to the limits if not -inf, inf
            logging.debug(f"Creating slider for parameter {parameter}")
            adjustment_elements = self.create_log_slider_and_input_elements(parameter)
            self.parameter_sliders[parameter.name] = adjustment_elements[0]
            self.parameter_inputs[parameter.name] = adjustment_elements[1]
        
        param_layout.addWidget(label)
        for element in adjustment_elements:
            param_layout.addWidget(element)
        
        return param_layout

    def create_pulldown_menu_elements(self, choices:List):
        """create a pulldown menu with the parameter choices, a linked input box, and return it as a two-element list, total width = 500"""
        pulldown = QComboBox()
        for choice in choices:
            pulldown.addItem(choice)
        pulldown.setFixedWidth(150)
        pulldown.currentIndexChanged.connect(lambda: self.update_plot())
        return [pulldown]

    def create_log_slider_and_input_elements(self, parameter:sasmodels.modelinfo.Parameter):
        """create a log-slider, input box and units text, return a three-elememnt list, total width = 500"""
        # Create a logarithmic slider for adjusting values
        slider = QSlider(Qt.Horizontal)
        slider.setFixedWidth(150)
        slider.setMinimum(0) #, np.max(0, parameter.limits[0]))
        slider.setMaximum(1000) #, np.min(1000, parameter.limits[1]))
        slider.setValue(self.value_to_log_slider(parameter.default, parameter))
        slider.valueChanged.connect(lambda: self.update_input_box(parameter.name))
        
        # Create an input box for exact value input
        input_box = QLineEdit(str(parameter.default))
        input_box.setFixedWidth(80)
        input_box.editingFinished.connect(lambda: self.update_slider(parameter.name))

        #unit text
        unit_text = QLabel(parameter.units)

        return [slider, input_box, unit_text]
    

    def value_to_log_slider(self, value, parameter:sasmodels.modelinfo.Parameter=None):
        """Convert a parameter value to a log slider position."""
        # Adjust range if necessary
        min_val, max_val = 1e-6, 1e3
        if parameter is not None:
            min_val = np.maximum(min_val, parameter.limits[0])
            max_val = np.minimum(parameter.limits[1], max_val)
        logging.debug('value_to_log_slider: %s', (min_val, max_val))

        if value == 0:
            log_pos = 0
        else:
            log_pos = int(1000 * (np.log10(value) - np.log10(min_val)) / (np.log10(max_val) - np.log10(min_val)))
        return log_pos

    def log_slider_to_value(self, slider_pos, parameter:sasmodels.modelinfo.Parameter=None):
        """Convert a log slider position back to a parameter value."""
        min_val, max_val = 1e-6, 1e3
        # Adjust range if necessary
        if parameter is not None:
            min_val = np.maximum(min_val, parameter.limits[0])
            max_val = np.minimum(parameter.limits[1], max_val)
        logging.debug('log_slider_to_value: %s', (min_val, max_val))

        if slider_pos == 0:
            value = 0
        else:
            value = 10 ** (np.log10(min_val) + slider_pos / 1000 * (np.log10(max_val) - np.log10(min_val)))
        return value

    def update_input_box(self, param_name):
        # Get the slider value, convert it back to the original scale, and update the input box
        slider = self.parameter_sliders[param_name]
        value = self.log_slider_to_value(slider.value(), parameter=self.parameters[param_name])
        input_box = self.parameter_inputs[param_name]
        input_box.setText(f"{value:.6f}")
        self.update_plot()

    def update_slider(self, param_name):
        # Get the value from the input box and update the corresponding slider
        input_box = self.parameter_inputs[param_name]
        try:
            value = float(input_box.text())
            slider = self.parameter_sliders[param_name]
            slider.setValue(self.value_to_log_slider(value, self.parameters[param_name]))
            self.update_plot()
        except ValueError:
            # Reset input box to slider value if the input is invalid
            slider = self.parameter_sliders[param_name]
            input_box.setText(f"{self.log_slider_to_value(slider.value(), self.parameters[param_name]):.6f}")

    def get_slider_values(self):
        # Retrieve values from sliders, adjusting them back to the parameter scale
        values = {param: self.log_slider_to_value(slider.value(), self.parameters[param]) for param, slider in self.parameter_sliders.items()}

        # Ensure that "radius_pd" is included in the parameters if "radius" was present
        pd_params = [param for param in values.keys() if param in self.model.info.parameters.pd_1d]

        for param in pd_params:
            # for each parameter, add a _pd_type and a _pd_n
            pname = param + "_pd"
            values[param + "_pd"] = self.log_slider_to_value(self.parameter_sliders[pname].value(), self.parameters[pname])

        return values
    
    def get_pulldown_values(self):
        # Retrieve values from pulldowns, adjusting them back to the parameter scale
        values = {}
        for param, chooser in self.parameter_choosers.items():
            parameter = self.parameters[param]
            if "_pd_type" in parameter.name:
                values[param] = parameter.choices[chooser.currentIndex()]
            else:
                values[param] = chooser.currentIndex()

        return values

    def update_plot(self):
        # Clear the current plot
        self.ax.clear()

        # Get values from sliders
        parameters = self.get_slider_values()
        # retrieve values from pulldown boxes
        parameters.update(self.get_pulldown_values())

        # Retrieve and validate q range and units
        try:
            qmin = float(self.q_min_input.text())
            qmax = float(self.q_max_input.text())
            qunit = self.q_unit_input.currentText()
        except ValueError:
            qmin, qmax = 0.01, 1.0  # Default values in case of error
            qunit = '1/nm'
        
        # Prepare parameters for sasmodel
        q = np.geomspace(qmin, qmax, 250)

        # Update the model parameters
        # find names that are in the polydisperse parameter list
        pd_params = [param for param in parameters.keys() if param in self.model.info.parameters.pd_1d]

        for param in pd_params:
            # for each parameter, add a _pd_type and a _pd_n
            pd_n = param + "_pd_n"
            parameters[pd_n] = 35

        # Compute intensity
        model = sasmodels.core.load_model(self.model_input.text())
        kernel = model.make_kernel([q * ureg.Quantity(1, qunit).to('1/Ang').magnitude])
        logging.info(f'calling sasmodels with {[{p: v} for p, v in parameters.items()]}')
        intensity = sasmodels.direct_model.call_kernel(kernel, parameters)

        # Plot
        self.ax.plot(q, intensity*100., '-')
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel(f"q ({qunit})")
        self.ax.set_ylabel("I (1/(m sr))")

        # Refresh the canvas
        self.canvas.draw()


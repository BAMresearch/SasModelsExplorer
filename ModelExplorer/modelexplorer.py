# ModelExplorer/modelexplorer.py

import logging
import re
from typing import List
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QCheckBox, QMessageBox,
                             QLabel, QLineEdit, QPushButton, QFormLayout, QSlider, QScrollArea)
# from PyQt5.QtWidgets import QBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase
import sasmodels.core
import sasmodels.direct_model
import numpy as np
import matplotlib.pyplot as plt
# import sip
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import collections
import pint
ureg = pint.UnitRegistry(auto_reduce_dimensions=True)
ureg.define(r"percent = 0.01 = %")
ureg.define(r"Ångström = 1e-10*m = Å = Ang = Angstrom")
ureg.define(r"item = 1")
from attrs import define, field
from .utils.list_to_columnar_string import list_to_columnar_string

class SasModelApp(QMainWindow):
    q:np.ndarray = None
    model = None
    kernel = None
    model_info:sasmodels.modelinfo.ModelInfo = None
    model_parameters = None
    parameters:dict = dict() # the actual parameter object references in the UI
    parameter_choosers:List = None # for pulldown box choices
    parameter_sliders:List = None # for sliders
    parameter_inputs:List = None # sliders have a linked text input box 
    parameter_checkboxes:List = None # checkboxes for future fitting
    # Pattern list to exclude specific parameters
    pd_types:List = ['uniform', 'rectangle', 'gaussian', 'lognormal', 'schulz', 'boltzmann']
    q_units:List = ['1/nm', '1/Ångström', '1/m']
    i_units:List = ['1/(m sr)', '1/(cm sr)']
    qunit:str = None
    infoText:str = None
    
    def __init__(self, modelName:str="sphere") -> None:
        super().__init__()
        self.setWindowTitle("SasModels Explorer")
        
        # generate the infoText:
        self.infoText = self.generate_infotext()

        # Main layout
        main_layout = QHBoxLayout()

        # Left layout for controls
        self.control_layout = QFormLayout()

        # Text input for model
        self.model_input = QLineEdit(modelName)
        self.model_input.setFixedWidth(300)
        self.control_layout.addRow("Model:", self.model_input)
        self.model_input.returnPressed.connect(self.load_model_parameters)
        self._parameter_row_start = self.control_layout.rowCount()

        # Scroll area for parameters
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.control_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(450)

        # Placeholder for dynamically added sliders and input boxes
        self.parameter_sliders = {}
        self.parameter_choosers = {}
        self.parameter_inputs = {}
        self.parameter_checkboxes = {}

        # Right layout for plot
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)

        # qmin and qmax inputs below the plot
        self.q_min_input = QLineEdit("0.01")
        self.q_max_input = QLineEdit("1.0")
        self.q_min_input.setFixedWidth(80)
        self.q_max_input.setFixedWidth(80)
        self.q_min_input.editingFinished.connect(self.update_kernel_and_plot)
        self.q_max_input.editingFinished.connect(self.update_kernel_and_plot)
        self.q_unit_input = self.create_pulldown_menu_elements(self.q_units, connected_function=self.update_kernel_and_plot)[0]

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
        self.model_parameters = None
        self.load_model_parameters()


    def generate_infotext(self)->str:
        """Generate the help text presented if an uninterpretable model is entered"""
        ncols = 3 # Number of columns for the model listing.       
        padding = "   " 
        categories = [sasmodels.core.load_model_info(model).category for model in sasmodels.core.list_models()]
        
        len([i for i in categories if i.startswith('shape:')])
        len(categories)
        groupings = [i.split(':')[0] for i in categories]
        infoText = 'Sasmodels can be specified as one of the following. They can also be composed by multiplication/division or addition/subtraction.\n'
        infoText += 'For example: "cylinder+sphere" will add these two models. A structure factor can be applied with the @-operator, e.g. "sphere@hardsphere".\n\n'
        for cat in list(collections.Counter(groupings).keys()):
            infoText += f"Available {cat} Models:\n"
            infoText += " -- \n"
            modelList = [sasmodels.core.load_model_info(model).id for model in sasmodels.core.list_models() if sasmodels.core.load_model_info(model).category.startswith(cat)]
            infoText += list_to_columnar_string(modelList, ncols=ncols, padding = padding, ordering='cols')
            
        print(infoText)
        return infoText


    def _delete_layout_item(self, item) -> None:
        if item is None:
            return
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
            return
        layout = item.layout()
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                self._delete_layout_item(child)
            layout.deleteLater()

    def clear_parameter_rows(self) -> None:
        while self.control_layout.rowCount() > self._parameter_row_start:
            row = self.control_layout.takeRow(self.control_layout.rowCount() - 1)
            if row is None:
                continue
            if isinstance(row, (tuple, list)):
                label_item, field_item = row
            else:
                label_item_attr = getattr(row, "labelItem", None)
                field_item_attr = getattr(row, "fieldItem", None)
                label_item = label_item_attr() if callable(label_item_attr) else label_item_attr
                field_item = field_item_attr() if callable(field_item_attr) else field_item_attr
            self._delete_layout_item(label_item)
            self._delete_layout_item(field_item)

    def load_model_parameters(self) -> None:
        model_name = self.model_input.text()
        try:
            # Load model info from sasmodels
            self.model = sasmodels.core.load_model(model_name)
            self.model_info = sasmodels.core.load_model_info(model_name)
            self.model_parameters = self.model_info.parameters.defaults.copy()

            self.clear_parameter_rows()
            # Reset the parameter-specific dictionaries to clear any previous model data
            self.parameter_sliders.clear()
            self.parameter_choosers.clear()
            self.parameter_inputs.clear()
            self.parameter_checkboxes.clear()
            self.parameters.clear()

            # Dynamically add sliders and input boxes for each model parameter
            # XSB: 5.5.2025 change to iterate through model_parameters written in call_parameters instead of the info parameters. allows to load 'core_multi_shell'
            # for parameter in self.model.info.parameters.common_parameters+self.model.info.parameters.kernel_parameters:
            seen_parameters = set()
            for parameter in self.model.info.parameters.common_parameters+self.model_info.parameters.call_parameters:
                if parameter.name in seen_parameters:
                    continue
                seen_parameters.add(parameter.name)
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
            self.update_model_and_plot()

        except Exception as e:
            logging.warning(f"Error loading model '{model_name}': {e}")

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
            

    def create_parameter_input_element(self, parameter: sasmodels.modelinfo.Parameter) -> QHBoxLayout:
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

    def create_pulldown_menu_elements(self, choices:List, connected_function=None) -> List:
        """create a pulldown menu with the parameter choices, a linked input box, and return it as a two-element list, total width = 500"""
        pulldown = QComboBox()
        for choice in choices:
            pulldown.addItem(choice)
        pulldown.setFixedWidth(150)
        if connected_function is None:
            pulldown.currentIndexChanged.connect(lambda: self.update_plot())
        else: 
            pulldown.currentIndexChanged.connect(lambda: connected_function())
        return [pulldown]

    def create_log_slider_and_input_elements(self, parameter:sasmodels.modelinfo.Parameter) -> List:
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

        # Checkbox for parameter fit
        # fit_checkbox = QCheckBox("Fit")
        # fit_checkbox.setChecked(False)
        # fit_checkbox.stateChanged.connect(self.update_plot)
        # self.parameter_checkboxes[parameter.name] = fit_checkbox

        return [slider, input_box, unit_text] #, fit_checkbox]
    

    def value_to_log_slider(self, value:float, parameter:sasmodels.modelinfo.Parameter=None) -> int:
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

    def log_slider_to_value(self, slider_pos:int, parameter:sasmodels.modelinfo.Parameter=None)->float:
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

    def update_input_box(self, param_name:str)->None:
        # Get the slider value, convert it back to the original scale, and update the input box
        slider = self.parameter_sliders[param_name]
        value = self.log_slider_to_value(slider.value(), parameter=self.parameters[param_name])
        input_box = self.parameter_inputs[param_name]
        input_box.setText(f"{value:.6f}")
        self.update_plot()

    def update_slider(self, param_name:str)->None:
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

    def get_slider_values(self)->dict:
        # Retrieve values from sliders, adjusting them back to the parameter scale
        values = {param: self.log_slider_to_value(slider.value(), self.parameters[param]) for param, slider in self.parameter_sliders.items()}

        # Ensure that "radius_pd" is included in the parameters if "radius" was present
        pd_params = [param for param in values.keys() if param in self.model.info.parameters.pd_1d]

        for param in pd_params:
            # for each parameter, add a _pd_type and a _pd_n
            pname = param + "_pd"
            values[param + "_pd"] = self.log_slider_to_value(self.parameter_sliders[pname].value(), self.parameters[pname])

        return values
    
    def get_pulldown_values(self)->dict:
        # Retrieve values from pulldowns, adjusting them back to the parameter scale
        values = {}
        for param, chooser in self.parameter_choosers.items():
            parameter = self.parameters[param]
            if "_pd_type" in parameter.name:
                values[param] = parameter.choices[chooser.currentIndex()]
            else:
                values[param] = chooser.currentIndex()

        return values

    def update_model_and_plot(self) -> None:
        # Update the model and kernel, then plot the results
        logging.info(f'loading model {self.model_input.text()}')
        # try:
        self.model = sasmodels.core.load_model(self.model_input.text())
        self.update_kernel_and_plot()

    def update_kernel_and_plot(self) -> None:
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
        # Clear the current plot
        logging.info(f'updating plot')
        self.ax.clear()

        # Get values from sliders
        parameters = self.get_slider_values()
        # retrieve values from pulldown boxes
        parameters.update(self.get_pulldown_values())

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
        intensity = sasmodels.direct_model.call_kernel(kernel, parameters)

        # Plot
        self.ax.plot(q, intensity*100., '-')
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel(f"q ({qunit})")
        self.ax.set_ylabel("I (1/(m sr))")

        # Refresh the canvas
        self.canvas.draw()

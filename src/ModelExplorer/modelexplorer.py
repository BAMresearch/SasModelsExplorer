import re
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QLabel, QLineEdit, QPushButton, QFormLayout, QSlider, QScrollArea)
from PyQt5.QtCore import Qt
import sasmodels.core
import sasmodels.direct_model
import numpy as np
import matplotlib.pyplot as plt
# import sip
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class SasModelApp(QMainWindow):
    model = None
    model_info = None
    model_parameters = None
    parameter_sliders = None
    parameter_inputs = None
    # Pattern list to exclude specific parameters
    exclude_patterns = [r'up_.*', r'.*_M0', r'.*_mtheta', r'.*_mphi']
    
    def __init__(self, modelName="sphere"):
        super().__init__()
        self.setWindowTitle("SasModels Interactive App")

        # Main layout
        main_layout = QHBoxLayout()

        # Left layout for controls
        self.control_layout = QFormLayout()

        # Text input for model
        self.model_input = QLineEdit(modelName)
        self.control_layout.addRow("Model:", self.model_input)
        self.model_input.returnPressed.connect(self.load_model_parameters)

        # # Button to reload model based on input
        # load_button = QPushButton("Load Model")
        # load_button.clicked.connect(self.load_model_parameters)
        # self.control_layout.addWidget(load_button)

        # Scroll area for parameters
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.control_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(400)

        # Placeholder for dynamically added sliders and input boxes
        self.parameter_sliders = {}
        self.parameter_inputs = {}

        # Right layout for plot
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)

        # qmin and qmax inputs below the plot
        self.q_min_input = QLineEdit("0.01")
        self.q_max_input = QLineEdit("1.0")
        self.q_min_input.setFixedWidth(80)
        self.q_max_input.setFixedWidth(80)
        self.q_min_input.editingFinished.connect(self.update_plot)
        self.q_max_input.editingFinished.connect(self.update_plot)

        # Layout for q range inputs
        q_range_layout = QHBoxLayout()
        q_range_layout.addWidget(QLabel("Q Min:"))
        q_range_layout.addWidget(self.q_min_input)
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
                    
        # while layout.count() > starting_index:  # Start clearing after model_input
        #     print('ping')
        #     item = layout.takeAt(starting_index)  # Always take the second item since index 0 is preserved
        #     print(item) # these are layout objects or widgets. if layout objects, go deeper
        #     if isinstance(item, QHBoxLayout):
        #         # hopefully we'll hit widgets at some point. 
        #         self.remove_layout_and_widgets(item, starting_index=0)
            
        #     widget = item.widget()
        #     if widget:
        #         layout.removeWidget(widget)
        #         # sip.delete(widget)
        #         # widget.deleteLater()
        #         widget.setParent(None)
        #         widget = None

    def load_model_parameters(self):
        model_name = self.model_input.text()
        try:
            # Load model info from sasmodels
            self.model_info = sasmodels.core.load_model_info(model_name)
            self.model_parameters = self.model_info.parameters.defaults.copy()

            self.remove_layout_and_widgets(self.control_layout, starting_index=2)
            # Reset the parameter-specific dictionaries to clear any previous model data
            self.parameter_sliders.clear()
            self.parameter_inputs.clear()

            # Dynamically add sliders and input boxes for each model parameter
            for param, default_value in self.model_parameters.items():
                # Skip parameters matching exclusion patterns
                if any(re.match(pattern, param) for pattern in self.exclude_patterns):
                    continue
                
                # Add the parameter layout for the current parameter
                param_layout = self.create_log_slider_and_input(param, default_value)
                self.control_layout.addRow(param_layout)

                # Check if the current parameter is "radius" and, if so, add "radius_pd" with a default value if it exists
                if param.endswith("radius"):
                    new_param = param + "_pd"
                    radius_pd_default = self.model_parameters.get(new_param, 0.1)  # Set to 0.1 or any fallback default
                    radius_pd_layout = self.create_log_slider_and_input(new_param, radius_pd_default)
                    self.control_layout.addRow(radius_pd_layout)

            # Initial plot
            self.update_plot()
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")


    def create_log_slider_and_input(self, param_name, default_value):
        # Horizontal layout for the parameter row
        param_layout = QHBoxLayout()

        # Create the label
        label = QLabel(param_name)
        
        # Create a logarithmic slider for adjusting values
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(1000)
        slider.setValue(self.value_to_log_slider(default_value))
        slider.valueChanged.connect(lambda: self.update_input_box(param_name))
        
        # Create an input box for exact value input
        input_box = QLineEdit(str(default_value))
        input_box.setFixedWidth(60)
        input_box.editingFinished.connect(lambda: self.update_slider(param_name))

        # Store slider and input box for later access
        self.parameter_sliders[param_name] = slider
        self.parameter_inputs[param_name] = input_box

        # Add widgets to the horizontal layout
        param_layout.addWidget(label)
        param_layout.addWidget(slider)
        param_layout.addWidget(input_box)

        return param_layout

    def value_to_log_slider(self, value):
        """Convert a parameter value to a log slider position."""
        # Adjust range if necessary
        min_val, max_val = 1e-6, 1e3
        log_pos = int(1000 * (np.log10(value) - np.log10(min_val)) / (np.log10(max_val) - np.log10(min_val)))
        return log_pos

    def log_slider_to_value(self, slider_pos):
        """Convert a log slider position back to a parameter value."""
        min_val, max_val = 1e-6, 1e3
        value = 10 ** (np.log10(min_val) + slider_pos / 1000 * (np.log10(max_val) - np.log10(min_val)))
        return value

    def update_input_box(self, param_name):
        # Get the slider value, convert it back to the original scale, and update the input box
        slider = self.parameter_sliders[param_name]
        value = self.log_slider_to_value(slider.value())
        input_box = self.parameter_inputs[param_name]
        input_box.setText(f"{value:.6f}")
        self.update_plot()

    def update_slider(self, param_name):
        # Get the value from the input box and update the corresponding slider
        input_box = self.parameter_inputs[param_name]
        try:
            value = float(input_box.text())
            slider = self.parameter_sliders[param_name]
            slider.setValue(self.value_to_log_slider(value))
            self.update_plot()
        except ValueError:
            # Reset input box to slider value if the input is invalid
            slider = self.parameter_sliders[param_name]
            input_box.setText(f"{self.log_slider_to_value(slider.value()):.6f}")

    def get_slider_values(self):
        # Retrieve values from sliders, adjusting them back to the parameter scale
        values = {param: self.log_slider_to_value(slider.value()) for param, slider in self.parameter_sliders.items()}

        # Ensure that "radius_pd" is included in the parameters if "radius" was present
        # find a parameter ending in radius, and add one tacking on _pd 
        radius_params = [param for param in values.keys() if param.endswith('radius')]

        # radius_params = [param for param in values.keys() if "radius" in param and not "_pd" in param]
        for param in radius_params:
            # for each parameter, add a _pd_type and a _pd_n
            values[param + "_pd"] = self.log_slider_to_value(self.parameter_sliders[param + "_pd"].value())

        return values
    
    def update_plot(self):
        # Clear the current plot
        self.ax.clear()

        # Get values from sliders
        parameters = self.get_slider_values()

        # Retrieve and validate q range
        try:
            qmin = float(self.q_min_input.text())
            qmax = float(self.q_max_input.text())
        except ValueError:
            qmin, qmax = 0.01, 1.0  # Default values in case of error

        # Prepare parameters for sasmodel
        q = np.geomspace(qmin, qmax, 250)

        # Update the model parameters
        # find names with radius in them
        radius_params = [param for param in parameters if param.endswith('radius')]

        # radius_params = [param for param in parameters if "radius" in param and not "_pd" in param]
        for param in radius_params:
            # for each parameter, add a _pd_type and a _pd_n
            pd_type = param + "_pd_type"
            pd_n = param + "_pd_n"
            parameters[pd_type] = "gaussian"
            parameters[pd_n] = 35

        parameters.update({
            'scale': parameters.get('scale', 1.0),
            'background': parameters.get('background', 0.001),
        })

        # Compute intensity
        model = sasmodels.core.load_model(self.model_input.text())
        kernel = model.make_kernel([q])
        intensity = sasmodels.direct_model.call_kernel(kernel, parameters)

        # Plot
        self.ax.plot(q, intensity, '-')
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel("q (1/nm)")
        self.ax.set_ylabel("I (1/(m sr))")

        # Refresh the canvas
        self.canvas.draw()


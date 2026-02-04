# ModelExplorer/parameter_panel.py

from typing import Callable, Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSlider,
    QWidget,
)


class ParameterPanel(QScrollArea):
    """Scrollable UI panel that renders sasmodels parameters and reads user values."""
    def __init__(self, on_change: Optional[Callable[[], None]] = None, width: int = 450) -> None:
        """Initialize the panel with optional change callback and fixed width."""
        super().__init__()
        self._on_change = on_change
        self._header_rows = 0

        self._container = QWidget()
        self._layout = QFormLayout()
        self._container.setLayout(self._layout)
        self.setWidget(self._container)
        self.setWidgetResizable(True)
        self.setMinimumWidth(width)

        self.parameters: Dict[str, object] = {}
        self.parameter_sliders: Dict[str, QSlider] = {}
        self.parameter_choosers: Dict[str, QComboBox] = {}
        self.parameter_inputs: Dict[str, QLineEdit] = {}
        self.parameter_checkboxes: Dict[str, object] = {}

    def add_header_row(self, label: str, widget: QWidget) -> None:
        """Add a fixed header row that is not cleared with model changes."""
        self._layout.addRow(label, widget)
        self._header_rows = self._layout.rowCount()

    def set_parameters(self, parameters: List[object]) -> None:
        """Clear existing parameter rows and rebuild inputs for new parameters."""
        self.clear_parameter_rows()
        self.parameters.clear()
        self.parameter_sliders.clear()
        self.parameter_choosers.clear()
        self.parameter_inputs.clear()
        self.parameter_checkboxes.clear()

        for parameter in parameters:
            self.parameters[parameter.name] = parameter
            param_layout = self.create_parameter_input_element(parameter)
            self._layout.addRow(param_layout)

    def clear_parameter_rows(self) -> None:
        """Remove all parameter rows while keeping header rows intact."""
        while self._layout.rowCount() > self._header_rows:
            row = self._layout.takeRow(self._layout.rowCount() - 1)
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

    def _delete_layout_item(self, item) -> None:
        """Recursively delete widgets/layouts contained in a layout item."""
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

    def create_parameter_input_element(self, parameter) -> QHBoxLayout:
        """Create a horizontal row for a parameter (label + controls)."""
        param_layout = QHBoxLayout()

        label = QLabel(parameter.name)
        label.setToolTip(parameter.description)
        label.setFixedWidth(100)

        choices = getattr(parameter, "choices", None) or []
        if len(choices) > 0:
            adjustment_elements = self.create_pulldown_menu_elements(choices)
            self.parameter_choosers[parameter.name] = adjustment_elements[0]
        else:
            adjustment_elements = self.create_log_slider_and_input_elements(parameter)
            self.parameter_sliders[parameter.name] = adjustment_elements[0]
            self.parameter_inputs[parameter.name] = adjustment_elements[1]

        param_layout.addWidget(label)
        for element in adjustment_elements:
            param_layout.addWidget(element)

        return param_layout

    def create_pulldown_menu_elements(self, choices: List[str]) -> List[QComboBox]:
        """Build a dropdown control for parameters with choice lists."""
        pulldown = QComboBox()
        for choice in choices:
            pulldown.addItem(choice)
        pulldown.setFixedWidth(150)
        pulldown.currentIndexChanged.connect(self._trigger_change)
        return [pulldown]

    def create_log_slider_and_input_elements(self, parameter) -> List[object]:
        """Build a log slider, input box, and unit label for numeric parameters."""
        slider = QSlider(Qt.Horizontal)
        slider.setFixedWidth(150)
        slider.setMinimum(0)
        slider.setMaximum(1000)
        slider.setValue(self.value_to_log_slider(parameter.default, parameter))
        slider.valueChanged.connect(lambda: self.update_input_box(parameter.name))

        input_box = QLineEdit(str(parameter.default))
        input_box.setFixedWidth(80)
        input_box.editingFinished.connect(lambda: self.update_slider(parameter.name))

        unit_text = QLabel(parameter.units)

        return [slider, input_box, unit_text]

    def value_to_log_slider(self, value: float, parameter=None) -> int:
        """Map a parameter value to its log slider position."""
        min_val, max_val = 1e-6, 1e3
        if parameter is not None:
            min_val = np.maximum(min_val, parameter.limits[0])
            max_val = np.minimum(parameter.limits[1], max_val)

        if value == 0:
            return 0
        return int(
            1000
            * (np.log10(value) - np.log10(min_val))
            / (np.log10(max_val) - np.log10(min_val))
        )

    def log_slider_to_value(self, slider_pos: int, parameter=None) -> float:
        """Map a log slider position back to the parameter value."""
        min_val, max_val = 1e-6, 1e3
        if parameter is not None:
            min_val = np.maximum(min_val, parameter.limits[0])
            max_val = np.minimum(parameter.limits[1], max_val)

        if slider_pos == 0:
            return 0
        return 10 ** (
            np.log10(min_val) + slider_pos / 1000 * (np.log10(max_val) - np.log10(min_val))
        )

    def update_input_box(self, param_name: str) -> None:
        """Sync the text input when a slider changes and trigger redraw."""
        slider = self.parameter_sliders[param_name]
        value = self.log_slider_to_value(slider.value(), parameter=self.parameters[param_name])
        input_box = self.parameter_inputs[param_name]
        input_box.setText(f"{value:.6f}")
        self._trigger_change()

    def update_slider(self, param_name: str) -> None:
        """Sync the slider when a text input changes and trigger redraw."""
        input_box = self.parameter_inputs[param_name]
        try:
            value = float(input_box.text())
            slider = self.parameter_sliders[param_name]
            slider.setValue(self.value_to_log_slider(value, self.parameters[param_name]))
            self._trigger_change()
        except ValueError:
            slider = self.parameter_sliders[param_name]
            input_box.setText(
                f"{self.log_slider_to_value(slider.value(), self.parameters[param_name]):.6f}"
            )

    def get_values(self) -> Dict[str, float]:
        """Return a dict of current parameter values from all controls."""
        values = {
            param: self.log_slider_to_value(slider.value(), self.parameters[param])
            for param, slider in self.parameter_sliders.items()
        }

        for param, chooser in self.parameter_choosers.items():
            parameter = self.parameters[param]
            if "_pd_type" in parameter.name:
                values[param] = parameter.choices[chooser.currentIndex()]
            else:
                values[param] = chooser.currentIndex()

        return values

    def _trigger_change(self) -> None:
        """Invoke the change callback if provided."""
        if self._on_change is not None:
            self._on_change()

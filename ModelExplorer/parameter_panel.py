# ModelExplorer/parameter_panel.py

from numbers import Real
from typing import Callable, Optional, Protocol

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLayoutItem,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSlider,
    QWidget,
)

from .types import ParameterValue


class ParameterDefinition(Protocol):
    """Protocol for parameter objects rendered in the panel."""

    name: str
    default: object
    units: str
    description: str
    limits: object
    choices: list[str]


class ClickableLabel(QLabel):
    """QLabel that emits ``clicked`` when left-clicked."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


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

        self.parameters: dict[str, ParameterDefinition] = {}
        self.parameter_sliders: dict[str, QSlider] = {}
        self.parameter_choosers: dict[str, QComboBox] = {}
        self.parameter_inputs: dict[str, QLineEdit] = {}
        self.parameter_checkboxes: dict[str, object] = {}
        self.parameter_limit_overrides: dict[str, tuple[float, float]] = {}

    def add_header_row(self, label: str, widget: QWidget) -> None:
        """Add a fixed header row that is not cleared with model changes."""
        self._layout.addRow(label, widget)
        self._header_rows = self._layout.rowCount()

    def set_parameters(self, parameters: list[ParameterDefinition]) -> None:
        """Clear existing parameter rows and rebuild inputs for new parameters."""
        self.clear_parameter_rows()
        self.parameters.clear()
        self.parameter_sliders.clear()
        self.parameter_choosers.clear()
        self.parameter_inputs.clear()
        self.parameter_checkboxes.clear()
        self.parameter_limit_overrides.clear()

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

    def _delete_layout_item(self, item: QLayoutItem | None) -> None:
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

    def create_parameter_input_element(self, parameter: ParameterDefinition) -> QHBoxLayout:
        """Create a horizontal row for a parameter (label + controls)."""
        param_layout = QHBoxLayout()

        label = self._create_parameter_label(parameter)
        label.setFixedWidth(100)

        choices = self._parameter_choices(parameter)
        elements: list[QWidget]
        if choices:
            chooser = self.create_pulldown_menu_elements(choices)
            self.parameter_choosers[parameter.name] = chooser
            elements = [chooser]
        else:
            slider, input_box, unit_label = self.create_log_slider_and_input_elements(parameter)
            self.parameter_sliders[parameter.name] = slider
            self.parameter_inputs[parameter.name] = input_box
            elements = [slider, input_box, unit_label]

        param_layout.addWidget(label)
        for element in elements:
            param_layout.addWidget(element)

        return param_layout

    def create_pulldown_menu_elements(self, choices: list[str]) -> QComboBox:
        """Build a dropdown control for parameters with choice lists."""
        pulldown = QComboBox()
        for choice in choices:
            pulldown.addItem(choice)
        pulldown.setFixedWidth(150)
        pulldown.currentIndexChanged.connect(self._trigger_change)
        return pulldown

    def create_log_slider_and_input_elements(self, parameter: ParameterDefinition) -> tuple[QSlider, QLineEdit, QLabel]:
        """Build a log slider, input box, and unit label for numeric parameters."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setFixedWidth(150)
        slider.setMinimum(0)
        slider.setMaximum(1000)
        default_value = self._coerce_float(parameter.default, default=0.0)
        slider.setValue(self.value_to_log_slider(default_value, parameter, parameter.name))
        slider.valueChanged.connect(lambda: self.update_input_box(parameter.name))

        input_box = QLineEdit(str(default_value))
        input_box.setFixedWidth(80)
        input_box.editingFinished.connect(lambda: self.update_slider(parameter.name))

        unit_text = QLabel(parameter.units)

        return slider, input_box, unit_text

    def _create_parameter_label(self, parameter: ParameterDefinition) -> QLabel:
        """Create either a plain or clickable parameter label."""

        description = str(getattr(parameter, "description", ""))
        if not self._is_numeric_parameter(parameter):
            label = QLabel(parameter.name)
            label.setToolTip(description)
            return label

        label = ClickableLabel(parameter.name)
        label.setCursor(Qt.CursorShape.PointingHandCursor)
        font = label.font()
        font.setUnderline(True)
        label.setFont(font)
        label.setStyleSheet("color: #1f6fb7;")
        suffix = "\nClick to override slider min/max."
        label.setToolTip(f"{description}{suffix}" if description else suffix.strip())
        label.clicked.connect(lambda name=parameter.name: self._edit_parameter_limits(name))
        return label

    def _is_numeric_parameter(self, parameter: ParameterDefinition) -> bool:
        """Return ``True`` for parameters that use numeric sliders."""

        choices = self._parameter_choices(parameter)
        if choices:
            return False
        default = getattr(parameter, "default", None)
        return isinstance(default, Real) and not isinstance(default, bool)

    def _effective_limits(
        self,
        param_name: str | None,
        parameter: ParameterDefinition | None = None,
    ) -> tuple[float, float]:
        """Return effective min/max slider limits for a parameter."""

        if param_name is not None and param_name in self.parameter_limit_overrides:
            return self.parameter_limit_overrides[param_name]

        min_val, max_val = 1e-6, 1e3
        if parameter is not None:
            limits = getattr(parameter, "limits", None)
            if isinstance(limits, (list, tuple)) and len(limits) >= 2:
                try:
                    low = float(limits[0])
                    if np.isfinite(low) and low > 0:
                        min_val = max(min_val, low)
                except (TypeError, ValueError):
                    pass
                try:
                    high = float(limits[1])
                    if np.isfinite(high) and high > min_val:
                        max_val = high
                except (TypeError, ValueError):
                    pass

        if not np.isfinite(min_val) or min_val <= 0:
            min_val = 1e-6
        if not np.isfinite(max_val) or max_val <= min_val:
            max_val = min_val * 1e3
        return float(min_val), float(max_val)

    def get_effective_limits(self, param_name: str) -> tuple[float, float]:
        """Return active slider limits for a parameter name."""

        parameter = self.parameters.get(param_name)
        return self._effective_limits(param_name, parameter)

    def value_to_log_slider(
        self,
        value: float | int,
        parameter: ParameterDefinition | None = None,
        param_name: str | None = None,
    ) -> int:
        """Map a parameter value to its log slider position."""
        min_val, max_val = self._effective_limits(param_name, parameter)
        value_float = self._coerce_float(value, default=0.0)
        if not np.isfinite(value_float) or value_float <= 0:
            return 0
        if value_float < min_val:
            value_float = min_val
        if value_float > max_val:
            value_float = max_val
        return int(1000 * (np.log10(value_float) - np.log10(min_val)) / (np.log10(max_val) - np.log10(min_val)))

    def log_slider_to_value(
        self,
        slider_pos: int,
        parameter: ParameterDefinition | None = None,
        param_name: str | None = None,
    ) -> float:
        """Map a log slider position back to the parameter value."""
        min_val, max_val = self._effective_limits(param_name, parameter)

        if slider_pos == 0:
            return 0
        return float(10 ** (np.log10(min_val) + slider_pos / 1000 * (np.log10(max_val) - np.log10(min_val))))

    def update_input_box(self, param_name: str) -> None:
        """Sync the text input when a slider changes and trigger redraw."""
        slider = self.parameter_sliders[param_name]
        value = self.log_slider_to_value(
            slider.value(),
            parameter=self.parameters[param_name],
            param_name=param_name,
        )
        input_box = self.parameter_inputs[param_name]
        input_box.setText(f"{value:.6f}")
        self._trigger_change()

    def update_slider(self, param_name: str) -> None:
        """Sync the slider when a text input changes and trigger redraw."""
        input_box = self.parameter_inputs[param_name]
        try:
            value = float(input_box.text())
            slider = self.parameter_sliders[param_name]
            slider.setValue(
                self.value_to_log_slider(
                    value,
                    self.parameters[param_name],
                    param_name=param_name,
                )
            )
            self._trigger_change()
        except ValueError:
            slider = self.parameter_sliders[param_name]
            input_box.setText(
                f"{self.log_slider_to_value(slider.value(), self.parameters[param_name], param_name=param_name):.6f}"
            )

    def get_values(self) -> dict[str, ParameterValue]:
        """Return a dict of current parameter values from all controls."""
        values: dict[str, ParameterValue] = {
            param: self.log_slider_to_value(
                slider.value(),
                self.parameters[param],
                param_name=param,
            )
            for param, slider in self.parameter_sliders.items()
        }

        for param, chooser in self.parameter_choosers.items():
            parameter = self.parameters[param]
            if "_pd_type" in parameter.name:
                choices = self._parameter_choices(parameter)
                values[param] = choices[chooser.currentIndex()]
            else:
                values[param] = chooser.currentIndex()

        return values

    def set_values(self, values: dict[str, ParameterValue], emit_change: bool = True) -> None:
        """Set current parameter values for sliders and dropdowns."""
        for param_name, value in values.items():
            if param_name in self.parameter_sliders:
                slider = self.parameter_sliders[param_name]
                input_box = self.parameter_inputs[param_name]
                numeric_value = self._coerce_float(value, default=self.log_slider_to_value(slider.value()))
                slider.blockSignals(True)
                input_box.blockSignals(True)
                slider.setValue(
                    self.value_to_log_slider(
                        numeric_value,
                        self.parameters[param_name],
                        param_name=param_name,
                    )
                )
                input_box.setText(f"{numeric_value:.6g}")
                slider.blockSignals(False)
                input_box.blockSignals(False)
            elif param_name in self.parameter_choosers:
                chooser = self.parameter_choosers[param_name]
                chooser.blockSignals(True)
                if isinstance(value, str):
                    parameter = self.parameters.get(param_name)
                    choices = self._parameter_choices(parameter)
                    if value in choices and choices:
                        chooser.setCurrentIndex(choices.index(value))
                else:
                    chooser.setCurrentIndex(int(value))
                chooser.blockSignals(False)

        if emit_change:
            self._trigger_change()

    def _edit_parameter_limits(self, param_name: str) -> None:
        """Show a dialog to override slider min/max for a numeric parameter."""

        parameter = self.parameters.get(param_name)
        if parameter is None or param_name not in self.parameter_sliders:
            return

        current_min, current_max = self._effective_limits(param_name, parameter)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"{param_name}: Slider Limits")
        layout = QFormLayout(dialog)
        min_input = QLineEdit(f"{current_min:.6g}")
        max_input = QLineEdit(f"{current_max:.6g}")
        layout.addRow("Min (> 0):", min_input)
        layout.addRow("Max (> Min):", max_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            new_min = float(min_input.text())
            new_max = float(max_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Limits", "Slider limits must be numeric values.")
            return

        if not np.isfinite(new_min) or not np.isfinite(new_max):
            QMessageBox.warning(self, "Invalid Limits", "Slider limits must be finite.")
            return
        if new_min <= 0:
            QMessageBox.warning(self, "Invalid Limits", "Minimum must be greater than 0 for log sliders.")
            return
        if new_max <= new_min:
            QMessageBox.warning(self, "Invalid Limits", "Maximum must be greater than minimum.")
            return

        self.parameter_limit_overrides[param_name] = (new_min, new_max)
        self.update_slider(param_name)

    def _trigger_change(self) -> None:
        """Invoke the change callback if provided."""
        if self._on_change is not None:
            self._on_change()

    @staticmethod
    def _parameter_choices(parameter: ParameterDefinition | None) -> list[str]:
        """Return normalized parameter choices list."""

        if parameter is None:
            return []
        choices = getattr(parameter, "choices", None)
        if isinstance(choices, list):
            return [str(choice) for choice in choices]
        if isinstance(choices, tuple):
            return [str(choice) for choice in choices]
        return []

    @staticmethod
    def _coerce_float(value: object, default: float) -> float:
        """Coerce scalar-like values to ``float`` with fallback default."""

        if isinstance(value, bool):
            return default
        if isinstance(value, Real):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

"""UI panel for selecting fit parameters and running optimization."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class FittingPanel(QWidget):
    """Panel showing fit-parameter toggles and fit status controls."""

    fitRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._checkboxes: dict[str, QCheckBox] = {}

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Parameters to fit:"))

        self._param_container = QWidget()
        self._param_layout = QVBoxLayout()
        self._param_layout.addStretch(1)
        self._param_container.setLayout(self._param_layout)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._param_container)
        self._scroll.setMinimumHeight(260)
        layout.addWidget(self._scroll)

        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Max iterations:"))
        self.max_iter_input = QSpinBox()
        self.max_iter_input.setMinimum(10)
        self.max_iter_input.setMaximum(100000)
        self.max_iter_input.setValue(300)
        iter_layout.addWidget(self.max_iter_input)
        iter_layout.addStretch(1)
        layout.addLayout(iter_layout)

        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.fitRequested.emit)
        layout.addWidget(self.fit_button)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self.setLayout(layout)

    def set_parameters(self, parameters: dict[str, object]) -> None:
        """Rebuild the checkbox list for currently visible numeric parameters."""

        for checkbox in self._checkboxes.values():
            checkbox.setParent(None)
        self._checkboxes.clear()

        while self._param_layout.count() > 1:
            item = self._param_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for name, param in parameters.items():
            choices = getattr(param, "choices", None) or []
            if choices:
                continue
            checkbox = QCheckBox(name)
            self._checkboxes[name] = checkbox
            self._param_layout.insertWidget(self._param_layout.count() - 1, checkbox)

    def get_selected_parameters(self) -> list[str]:
        """Return names of parameters selected for fitting."""

        return [name for name, box in self._checkboxes.items() if box.isChecked()]

    def set_selected_parameters(self, names: list[str]) -> None:
        """Set selected fit parameter names from persisted selection."""

        for name in names:
            checkbox = self._checkboxes.get(name)
            if checkbox is not None:
                checkbox.setChecked(True)

    def get_max_iterations(self) -> int:
        """Return requested fit iteration limit."""

        return int(self.max_iter_input.value())

    def set_status(self, message: str) -> None:
        """Update fit status text."""

        self.status_label.setText(message)

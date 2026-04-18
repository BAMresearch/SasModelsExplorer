"""Export/import tab widgets for HDF5, CSV, and McSAS3 run-configuration workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True, slots=True)
class RunConfigExportChoice:
    """Choices collected from the run-configuration export dialog."""

    fit_parameter: str
    lower_limit: float
    upper_limit: float
    log_random: bool


class RunConfigExportDialog(QDialog):
    """Dialog that collects fit parameter and range for McSAS3 run-config export."""

    def __init__(
        self, float_parameter_limits: Mapping[str, tuple[float, float]], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export McSAS3 Run Configuration")
        self._limits_by_name = dict(float_parameter_limits)

        layout = QFormLayout(self)

        self.parameter_combo = QComboBox()
        for name in self._limits_by_name:
            self.parameter_combo.addItem(name)
        self.parameter_combo.currentIndexChanged.connect(self._apply_selected_limits)
        layout.addRow("Optimize parameter:", self.parameter_combo)

        self.min_input = QLineEdit()
        self.max_input = QLineEdit()
        layout.addRow("Min limit:", self.min_input)
        layout.addRow("Max limit:", self.max_input)

        self.log_random_checkbox = QCheckBox("logRandom")
        self.log_random_checkbox.setChecked(True)
        layout.addRow("", self.log_random_checkbox)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self._apply_selected_limits()

    def _apply_selected_limits(self) -> None:
        name = self.parameter_combo.currentText()
        limits = self._limits_by_name.get(name)
        if limits is None:
            return
        low, high = limits
        self.min_input.setText(f"{float(low):.6g}")
        self.max_input.setText(f"{float(high):.6g}")

    def selected_choice(self) -> RunConfigExportChoice:
        """Return the selected export options."""

        return RunConfigExportChoice(
            fit_parameter=self.parameter_combo.currentText(),
            lower_limit=float(self.min_input.text()),
            upper_limit=float(self.max_input.text()),
            log_random=self.log_random_checkbox.isChecked(),
        )


class ExportPanel(QWidget):
    """Tab panel with import/export actions for model state and run templates."""

    save_state_requested = pyqtSignal()
    load_state_requested = pyqtSignal()
    export_csv_requested = pyqtSignal()
    export_run_config_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout()

        state_button_row = QHBoxLayout()
        self.load_state_button = QPushButton("Load State (HDF5)")
        self.save_state_button = QPushButton("Save State (HDF5)")
        self.load_state_button.clicked.connect(self.load_state_requested.emit)
        self.save_state_button.clicked.connect(self.save_state_requested.emit)
        state_button_row.addWidget(self.load_state_button)
        state_button_row.addWidget(self.save_state_button)
        layout.addLayout(state_button_row)

        self.export_csv_button = QPushButton("Export Model Parameters to CSV")
        self.export_csv_button.clicked.connect(self.export_csv_requested.emit)
        layout.addWidget(self.export_csv_button)

        self.export_run_config_button = QPushButton("Export McSAS3 Run Configuration")
        self.export_run_config_button.clicked.connect(self.export_run_config_requested.emit)
        layout.addWidget(self.export_run_config_button)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch(1)
        self.setLayout(layout)

    def set_status(self, message: str) -> None:
        """Set panel status text."""

        self.status_label.setText(message)

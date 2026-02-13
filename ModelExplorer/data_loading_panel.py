# ModelExplorer/data_loading_panel.py

import sys
from pathlib import Path
from typing import Any, List, Optional

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .services.data_loader import load_data_bundle
from .yaml_editor_widget import YAMLEditorWidget

DEFAULT_YAML = """# Units are assumed to be 1/(m sr) for I and 1/nm for Q
Q_unit: "1/nm"
I_unit: "1/(m sr)"
nbins: 100
dataRange:
  - 0.0
  - .inf
csvargs:
  sep: ";"
  header: null
  names:
    - "Q"
    - "I"
    - "ISigma"
"""


class FileDropLineEdit(QLineEdit):
    fileDropped = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            event.ignore()
            return
        local_path = urls[0].toLocalFile()
        if local_path:
            self.setText(local_path)
            self.fileDropped.emit(local_path)
            event.acceptProposedAction()
        else:
            event.ignore()


class DataLoadingPanel(QWidget):
    """Panel for loading experimental data using a YAML configuration."""

    dataChanged = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mds = None
        self._data_bundle = None
        self._missing_deps: Optional[str] = None

        self._McData1D = None
        self._BaseData = None
        self._DataBundle = None
        self._modacor_ureg = None

        self._config_dir = self._find_default_config_dir()
        self._config_files: List[Path] = []

        self._suppress_yaml_change = False
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._load_data)

        layout = QVBoxLayout()

        self.config_combo = QComboBox()
        self.config_combo.currentIndexChanged.connect(self._on_config_selected)
        layout.addWidget(QLabel("Default YAML configuration:"))
        layout.addWidget(self.config_combo)

        self.yaml_editor_widget = YAMLEditorWidget(directory=self._config_dir, parent=self, multipart=False)
        self.yaml_editor_widget.set_yaml_content(DEFAULT_YAML)
        self.yaml_editor_widget.yaml_editor.textChanged.connect(self._on_yaml_changed)
        self.yaml_editor_widget.fileSaved.connect(self._refresh_config_list)
        layout.addWidget(QLabel("Data loading configuration (YAML):"))
        layout.addWidget(self.yaml_editor_widget)

        file_layout = QHBoxLayout()
        self.file_path_line = FileDropLineEdit()
        self.file_path_line.fileDropped.connect(self._schedule_load)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_path_line)
        file_layout.addWidget(browse_button)
        layout.addWidget(QLabel("Data file:"))
        layout.addLayout(file_layout)

        self.data_mode_combo = QComboBox()
        self.data_mode_combo.addItem("Binned data", "binnedData")
        self.data_mode_combo.addItem("Clipped data", "clippedData")
        self.data_mode_combo.addItem("Raw data", "rawData")
        self.data_mode_combo.currentIndexChanged.connect(self._schedule_load)
        layout.addWidget(QLabel("Overlay data source:"))
        layout.addWidget(self.data_mode_combo)

        self.message_box = QPlainTextEdit()
        self.message_box.setReadOnly(True)
        self.message_box.setPlaceholderText("Messages will appear here.")
        layout.addWidget(self.message_box)

        self.chi_square_label = QLabel("Reduced chi-square: --")
        layout.addWidget(self.chi_square_label)

        layout.addStretch(1)
        self.setLayout(layout)

        self._refresh_config_list()

    def _find_default_config_dir(self) -> Optional[Path]:
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root.parent / "McSAS3GUI" / "src" / "mcsas3gui" / "configurations" / "readdata"
        if candidate.is_dir():
            return candidate
        return None

    def _refresh_config_list(self) -> None:
        self.config_combo.blockSignals(True)
        self.config_combo.clear()
        self._config_files = []
        if self._config_dir and self._config_dir.exists():
            self._config_files = sorted(self._config_dir.glob("*.yaml"))
        for path in self._config_files:
            self.config_combo.addItem(path.name, path)
        self.config_combo.addItem("<Custom...>", None)
        self.config_combo.blockSignals(False)

        if self._config_files:
            self.config_combo.setCurrentIndex(0)
            self._load_yaml_from_path(self._config_files[0])
        else:
            self.config_combo.setCurrentText("<Custom...>")

    def _on_config_selected(self) -> None:
        path = self.config_combo.currentData()
        if isinstance(path, Path):
            self._load_yaml_from_path(path)

    def _load_yaml_from_path(self, path: Path) -> None:
        try:
            content = path.read_text()
        except Exception as exc:
            self._set_message(f"Failed to read YAML file: {exc}")
            return
        self._suppress_yaml_change = True
        self.yaml_editor_widget.set_yaml_content(content)
        self._suppress_yaml_change = False
        self._schedule_load()

    def _on_yaml_changed(self) -> None:
        if self._suppress_yaml_change:
            return
        if self.config_combo.currentText() != "<Custom...>":
            self.config_combo.blockSignals(True)
            self.config_combo.setCurrentText("<Custom...>")
            self.config_combo.blockSignals(False)
        self._schedule_load()

    def _browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select data file", "", "All Files (*.*)")
        if file_path:
            self.file_path_line.setText(file_path)
            self._schedule_load()

    def _schedule_load(self) -> None:
        self._debounce_timer.start(300)

    def _ensure_mcsas3(self) -> bool:
        if self._McData1D is not None:
            return True
        try:
            from mcsas3.mc_data_1d import McData1D

            self._McData1D = McData1D
            return True
        except Exception:
            pass

        repo_root = Path(__file__).resolve().parents[1]
        local_src = repo_root.parent / "McSAS3" / "src"
        if local_src.is_dir():
            sys.path.append(str(local_src))
            try:
                from mcsas3.mc_data_1d import McData1D

                self._McData1D = McData1D
                return True
            except Exception as exc:
                self._missing_deps = f"Failed to import McSAS3: {exc}"
                return False

        self._missing_deps = "McSAS3 could not be imported. Install it or clone it next to this repo."
        return False

    def _ensure_modacor(self) -> bool:
        if self._BaseData is not None and self._DataBundle is not None:
            return True
        try:
            from modacor import ureg as modacor_ureg
            from modacor.dataclasses.basedata import BaseData
            from modacor.dataclasses.databundle import DataBundle

            self._BaseData = BaseData
            self._DataBundle = DataBundle
            self._modacor_ureg = modacor_ureg
            self._ensure_modacor_units()
            return True
        except Exception:
            pass

        repo_root = Path(__file__).resolve().parents[1]
        local_src = repo_root.parent / "MoDaCor" / "src"
        if local_src.is_dir():
            sys.path.append(str(local_src))
            try:
                from modacor import ureg as modacor_ureg
                from modacor.dataclasses.basedata import BaseData
                from modacor.dataclasses.databundle import DataBundle

                self._BaseData = BaseData
                self._DataBundle = DataBundle
                self._modacor_ureg = modacor_ureg
                self._ensure_modacor_units()
                return True
            except Exception as exc:
                self._missing_deps = f"Failed to import MoDaCor: {exc}"
                return False

        self._missing_deps = "MoDaCor could not be imported. Install it or clone it next to this repo."
        return False

    def _ensure_modacor_units(self) -> None:
        if self._modacor_ureg is None:
            return
        try:
            self._modacor_ureg.Unit("Angstrom")
        except Exception:
            try:
                self._modacor_ureg.define("Angstrom = 1e-10*m = Ang = angstrom")
            except Exception:
                pass
        try:
            self._modacor_ureg.Unit("percent")
        except Exception:
            try:
                self._modacor_ureg.define("percent = 0.01 = %")
            except Exception:
                pass

    def _load_data(self) -> None:
        self._clear_message()
        self._data_bundle = None

        file_path = self.file_path_line.text().strip()
        if not file_path:
            self.dataChanged.emit()
            return

        data_path = Path(file_path)
        if not data_path.exists():
            self._set_message(f"File not found: {data_path}")
            self.dataChanged.emit()
            return

        if not self._ensure_mcsas3() or not self._ensure_modacor():
            self._set_message(self._missing_deps or "Missing dependencies for data loading.")
            self.dataChanged.emit()
            return

        yaml_text = self.yaml_editor_widget.yaml_editor.toPlainText()
        data_kind = self.data_mode_combo.currentData()
        try:
            bundle, used_kind, count = load_data_bundle(
                data_path,
                data_kind,
                yaml_text,
                self._McData1D,
                self._BaseData,
                self._DataBundle,
            )
        except ValueError as exc:
            self._set_message(str(exc))
            self.dataChanged.emit()
            return
        except Exception as exc:
            self._set_message(f"Error loading data: {exc}")
            self.dataChanged.emit()
            return

        self._data_bundle = bundle
        self._set_message(f"Loaded {count} points from {used_kind}.")
        self._maybe_list_hdf5_paths(data_path)
        self.dataChanged.emit()

    def _maybe_list_hdf5_paths(self, data_path: Path) -> None:
        if data_path.suffix.lower() not in {".h5", ".hdf5", ".nxs", ".nx"}:
            return
        try:
            import h5py
        except Exception:
            return

        lines: List[str] = []
        try:
            with h5py.File(data_path, "r") as h5f:

                def _visit(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Dataset):
                        lines.append(f"{name}: {obj.shape}")

                h5f.visititems(_visit)
        except Exception as exc:
            self.message_box.appendPlainText(f"HDF5 read error: {exc}")
            return

        if lines:
            self.message_box.appendPlainText("Available datasets:")
            for line in lines[:50]:
                self.message_box.appendPlainText(line)
            if len(lines) > 50:
                self.message_box.appendPlainText("... (truncated)")

    def _set_message(self, message: str) -> None:
        self.message_box.setPlainText(message)

    def _clear_message(self) -> None:
        self.message_box.clear()

    def set_chi_square(self, value: Optional[float], dof: Optional[int], points: Optional[int]) -> None:
        if value is None or dof is None or points is None:
            self.chi_square_label.setText("Reduced chi-square: --")
            return
        self.chi_square_label.setText(f"Reduced chi-square: {value:.4g} (dof={dof}, N={points})")

    def get_data_bundle(self) -> Optional[Any]:
        return self._data_bundle

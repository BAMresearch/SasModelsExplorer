# ModelExplorer/data_loading_panel.py

from pathlib import Path
from typing import Any, Optional

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

from .services.data_loader import load_data_selection
from .services.mcsas3_backend import ProcessingBackend, load_mcsas3_backend
from .yaml_editor_widget import YAMLEditorWidget

STAGE_RAW = "sample_raw"
STAGE_CLIPPED = "sample_clipped"
STAGE_BINNED = "sample_binned"

DEFAULT_YAML = """# Units below are interpreted as source file units.
sourceQUnits: "1/nm"
sourceIntensityUnits: "1/(m sr)"
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

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if event is None:
            return
        mime_data = event.mimeData()
        if mime_data is not None and mime_data.hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is None:
            return
        mime_data = event.mimeData()
        if mime_data is None:
            return
        urls = mime_data.urls()
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
        self._data_bundle: object | None = None
        self._data_bundles_by_stage: dict[str, object] = {}
        self._missing_deps: Optional[str] = None

        self._backend: ProcessingBackend | None = None

        self._config_dir = self._find_default_config_dir()
        self._config_files: list[Path] = []

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
        self.data_mode_combo.addItem("Binned data", STAGE_BINNED)
        self.data_mode_combo.addItem("Clipped data", STAGE_CLIPPED)
        self.data_mode_combo.addItem("Raw data", STAGE_RAW)
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
        if self._backend is not None:
            return True
        try:
            self._backend = load_mcsas3_backend()
            return True
        except ImportError as exc:
            self._missing_deps = str(exc)
            return False

    def _load_data(self) -> None:
        self._clear_message()
        self._data_bundle = None
        self._data_bundles_by_stage = {}

        file_path = self.file_path_line.text().strip()
        if not file_path:
            self.dataChanged.emit()
            return

        data_path = Path(file_path)
        if not data_path.exists():
            self._set_message(f"File not found: {data_path}")
            self.dataChanged.emit()
            return

        if not self._ensure_mcsas3():
            self._set_message(self._missing_deps or "Missing dependencies for data loading.")
            self.dataChanged.emit()
            return

        yaml_text = self.yaml_editor_widget.yaml_editor.toPlainText()
        data_kind = str(self.data_mode_combo.currentData())
        if self._backend is None:
            self._set_message("Missing McSAS3 backend.")
            self.dataChanged.emit()
            return
        try:
            selection = load_data_selection(
                data_path,
                data_kind,
                yaml_text,
                self._backend,
            )
        except ValueError as exc:
            self._set_message(str(exc))
            self.dataChanged.emit()
            return
        except Exception as exc:
            self._set_message(f"Error loading data: {exc}")
            self.dataChanged.emit()
            return

        self._data_bundle = selection.bundle
        self._data_bundles_by_stage = dict(selection.stage_bundles)

        if selection.used_kind != data_kind:
            for index in range(self.data_mode_combo.count()):
                if str(self.data_mode_combo.itemData(index)) == selection.used_kind:
                    self.data_mode_combo.blockSignals(True)
                    self.data_mode_combo.setCurrentIndex(index)
                    self.data_mode_combo.blockSignals(False)
                    break

        self._set_message(f"Loaded {selection.count} points from {selection.used_kind}.")
        self._maybe_list_hdf5_paths(data_path)
        self.dataChanged.emit()

    def _maybe_list_hdf5_paths(self, data_path: Path) -> None:
        if data_path.suffix.lower() not in {".h5", ".hdf5", ".nxs", ".nx"}:
            return
        try:
            import h5py
        except Exception:
            return

        lines: list[str] = []
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

    def get_data_bundle(self) -> object | None:
        return self._data_bundle

    def get_yaml_config_text(self) -> str:
        """Return current YAML configuration text from the editor."""

        return str(self.yaml_editor_widget.yaml_editor.toPlainText())

    def get_selected_preset_name(self) -> str | None:
        """Return selected preset filename, or ``None`` for custom/unknown selection."""

        current_data = self.config_combo.currentData()
        if isinstance(current_data, Path):
            return current_data.name
        return None

    def set_yaml_config_text(
        self,
        yaml_text: str,
        *,
        mark_custom: bool = True,
        prefer_matching_preset: bool = False,
        preferred_preset_name: str | None = None,
    ) -> None:
        """Set YAML configuration text without triggering an automatic reload."""

        self._suppress_yaml_change = True
        self.yaml_editor_widget.set_yaml_content(yaml_text)
        self._suppress_yaml_change = False

        if prefer_matching_preset:
            selected_index = self._matching_preset_index(yaml_text, preferred_preset_name)
            if selected_index is not None:
                self.config_combo.blockSignals(True)
                self.config_combo.setCurrentIndex(selected_index)
                self.config_combo.blockSignals(False)
                return

        if mark_custom:
            custom_index = self.config_combo.findText("<Custom...>")
            if custom_index >= 0:
                self.config_combo.blockSignals(True)
                self.config_combo.setCurrentIndex(custom_index)
                self.config_combo.blockSignals(False)

    def _matching_preset_index(self, yaml_text: str, preferred_preset_name: str | None = None) -> int | None:
        """Return preset combo index if YAML matches a known preset, else ``None``."""

        normalized_target = self._normalize_yaml_text(yaml_text)
        if not normalized_target:
            return None

        if preferred_preset_name:
            preferred_index = self.config_combo.findText(preferred_preset_name)
            if preferred_index >= 0:
                preferred_path = self.config_combo.itemData(preferred_index)
                if isinstance(preferred_path, Path):
                    try:
                        preferred_text = preferred_path.read_text(encoding="utf-8")
                    except Exception:
                        preferred_text = ""
                    if self._normalize_yaml_text(preferred_text) == normalized_target:
                        return int(preferred_index)

        for index in range(self.config_combo.count()):
            preset_path = self.config_combo.itemData(index)
            if not isinstance(preset_path, Path):
                continue
            try:
                preset_text = preset_path.read_text(encoding="utf-8")
            except Exception:
                continue
            if self._normalize_yaml_text(preset_text) == normalized_target:
                return index
        return None

    @staticmethod
    def _normalize_yaml_text(yaml_text: str) -> str:
        """Normalize YAML text for robust preset matching."""

        lines = [line.rstrip() for line in yaml_text.replace("\r\n", "\n").split("\n")]
        return "\n".join(lines).strip()

    def get_stage_bundles(self) -> dict[str, object]:
        """Return loaded stage bundles keyed by canonical stage name."""

        return dict(self._data_bundles_by_stage)

    def get_stage_frames(self) -> dict[str, object]:
        """Return per-stage dataframe projections using the active backend adapter."""

        if self._backend is None:
            return {}
        frames: dict[str, object] = {}
        for stage_name, bundle in self._data_bundles_by_stage.items():
            try:
                frames[stage_name] = self._backend.frame_from_bundle(bundle)
            except Exception:
                continue
        return frames

    def set_stage_bundles(
        self,
        stage_bundles: dict[str, object],
        selected_stage: str | None = None,
        message: str | None = None,
    ) -> None:
        """Set all loaded stage bundles and update currently active bundle."""

        self._data_bundles_by_stage = dict(stage_bundles)
        if selected_stage and selected_stage in self._data_bundles_by_stage:
            self._data_bundle = self._data_bundles_by_stage[selected_stage]
            for index in range(self.data_mode_combo.count()):
                if str(self.data_mode_combo.itemData(index)) == selected_stage:
                    self.data_mode_combo.blockSignals(True)
                    self.data_mode_combo.setCurrentIndex(index)
                    self.data_mode_combo.blockSignals(False)
                    break
        elif self._data_bundles_by_stage:
            first_stage = next(iter(self._data_bundles_by_stage))
            self._data_bundle = self._data_bundles_by_stage[first_stage]
        else:
            self._data_bundle = None

        if message is not None:
            self._set_message(message)
        self.dataChanged.emit()

    def set_data_bundle(self, bundle: object | None, message: str | None = None) -> None:
        """Set the active overlay data bundle from an external source."""

        self._data_bundle = bundle
        stage_name = str(self.data_mode_combo.currentData())
        if bundle is None:
            self._data_bundles_by_stage = {}
        else:
            self._data_bundles_by_stage = {stage_name: bundle}
        if message is not None:
            self._set_message(message)
        self.dataChanged.emit()

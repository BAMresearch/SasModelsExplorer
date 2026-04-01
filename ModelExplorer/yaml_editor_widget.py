"""YAML editor widget adapter with optional McSAS3GUI integration."""

from __future__ import annotations

try:
    from mcsas3gui.gui.yaml_editor_widget import CustomDumper, YAMLEditorWidget, YAMLErrorHighlighter
except Exception:
    import logging
    from pathlib import Path

    import yaml
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtGui import QSyntaxHighlighter
    from PyQt6.QtWidgets import (
        QFileDialog,
        QHBoxLayout,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    logger = logging.getLogger(__name__)

    class CustomDumper(yaml.Dumper):
        """Fallback dumper that keeps nested lists readable."""

        def increase_indent(self, flow: bool = False, indentless: bool = False):
            return super().increase_indent(flow, False)

        def represent_list(self, data: list[object]):
            if any(isinstance(item, (list, dict)) for item in data):
                return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)
            return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

        def represent_dict(self, data: dict[str, object]):
            return self.represent_mapping("tag:yaml.org,2002:map", data, flow_style=False)

    CustomDumper.add_representer(list, CustomDumper.represent_list)
    CustomDumper.add_representer(dict, CustomDumper.represent_dict)

    class YAMLErrorHighlighter(QSyntaxHighlighter):
        """No-op fallback highlighter for API compatibility."""

        def highlightBlock(self, text: str) -> None:  # noqa: N802
            _ = text

    class YAMLEditorWidget(QWidget):
        """Minimal YAML editor fallback used when McSAS3GUI is unavailable."""

        fileSaved = pyqtSignal(str)

        def __init__(self, directory, parent=None, multipart: bool = False):
            super().__init__(parent)
            self.directory = str(directory) if directory else ""
            self.multipart = multipart

            layout = QVBoxLayout()
            self.yaml_editor = QTextEdit()
            self.yaml_editor.setAcceptDrops(False)
            layout.addWidget(self.yaml_editor)

            button_layout = QHBoxLayout()
            load_button = QPushButton("Load Configuration")
            save_button = QPushButton("Save Configuration")
            load_button.clicked.connect(self.load_yaml)
            save_button.clicked.connect(self.save_yaml)
            button_layout.addWidget(load_button)
            button_layout.addWidget(save_button)
            layout.addLayout(button_layout)
            self.setLayout(layout)

        def load_yaml(self) -> None:
            """Load YAML content from disk into the editor."""

            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Configuration",
                self.directory,
                "YAML Files (*.yaml)",
            )
            if file_name:
                self.set_yaml_content(Path(file_name).read_text())

        def save_yaml(self) -> None:
            """Validate and save YAML content from the editor."""

            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Configuration",
                self.directory,
                "YAML Files (*.yaml)",
            )
            if not file_name:
                return

            yaml_content = self.yaml_editor.toPlainText()
            try:
                if self.multipart:
                    parsed = [item for item in list(yaml.safe_load_all(yaml_content)) if item]
                    serialized = yaml.dump_all(parsed, Dumper=CustomDumper, default_flow_style=None, sort_keys=False)
                else:
                    parsed = yaml.safe_load(yaml_content)
                    serialized = yaml.dump(parsed, Dumper=CustomDumper, default_flow_style=None, sort_keys=False)
            except yaml.YAMLError as exc:
                logger.error("Error validating YAML before save: %s", exc)
                return

            Path(file_name).write_text(serialized)
            self.fileSaved.emit(file_name)

        def get_yaml_content(self) -> list[object]:
            """Return parsed YAML documents from current editor text."""

            try:
                return list(yaml.safe_load_all(self.yaml_editor.toPlainText()))
            except yaml.YAMLError as exc:
                logger.error("YAML parsing error: %s", exc)
                return []

        def set_yaml_content(self, yaml_content: object) -> None:
            """Set editor content from serialized YAML or parsed objects."""

            if isinstance(yaml_content, list):
                text = "---\n".join(
                    yaml.dump(item, Dumper=CustomDumper, default_flow_style=None, sort_keys=False)
                    for item in yaml_content
                )
            elif isinstance(yaml_content, dict):
                text = yaml.dump(yaml_content, Dumper=CustomDumper, default_flow_style=None, sort_keys=False)
            else:
                text = str(yaml_content)
            self.yaml_editor.setPlainText(text)

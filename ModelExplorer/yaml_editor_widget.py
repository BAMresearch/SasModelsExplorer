# ModelExplorer/yaml_editor_widget.py

import logging
import re

import yaml
from PyQt6.QtCore import QEvent, QRegularExpression, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class CustomDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(CustomDumper, self).increase_indent(flow, False)

    def represent_list(self, data):
        if any(isinstance(item, (list, dict)) for item in data):
            return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    def represent_dict(self, data):
        return self.represent_mapping("tag:yaml.org,2002:map", data, flow_style=False)


CustomDumper.add_representer(list, CustomDumper.represent_list)
CustomDumper.add_representer(dict, CustomDumper.represent_dict)


class YAMLErrorHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.error_line = None
        self.error_message = None

        self.key_format = QTextCharFormat()
        self.key_format.setForeground(QColor("blue"))
        self.key_format.setFontWeight(QFont.Weight.Bold)

        self.value_format = QTextCharFormat()
        self.value_format.setForeground(QColor("darkgreen"))

        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("gray"))
        self.comment_format.setFontItalic(True)

        self.rules = [
            (QRegularExpression(r"^\s*[^#]*:"), self.key_format),
            (QRegularExpression(r":\s*[^#]*"), self.value_format),
            (QRegularExpression(r"#.*$"), self.comment_format),
        ]

    def set_error(self, line, message):
        self.error_line = line
        self.error_message = message
        self.rehighlight()

    def clear_error(self):
        self.error_line = None
        self.error_message = None
        self.rehighlight()

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)

        if self.error_line is not None and self.currentBlock().blockNumber() == self.error_line - 1:
            error_format = QTextCharFormat()
            error_format.setBackground(QColor("lightcoral"))
            self.setFormat(0, len(text), error_format)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.ToolTip and self.error_line is not None:
            cursor = obj.cursorForPosition(event.pos())
            if cursor.blockNumber() == self.error_line - 1:
                QToolTip.showText(event.globalPos(), self.error_message, obj)
            else:
                QToolTip.hideText()
            return True
        return super().eventFilter(obj, event)


class YAMLEditorWidget(QWidget):
    fileSaved = pyqtSignal(str)

    def __init__(self, directory, parent=None, multipart=False):
        super().__init__(parent)
        self.directory = str(directory) if directory else ""
        self.multipart = multipart
        layout = QVBoxLayout()

        self.yaml_editor = QTextEdit()
        self.yaml_editor.setStyleSheet(
            """
                QTextEdit, QPlainTextEdit {
                    background-color: #f7f7ff;
                    border: 1px solid #cccccc;
                    padding: 6px;
                    font-family: "Courier New", monospace;
                }
            """
        )
        self.yaml_editor.setAcceptDrops(False)
        self.error_highlighter = YAMLErrorHighlighter(self.yaml_editor.document())
        self.yaml_editor.textChanged.connect(self.validate_yaml)
        self.yaml_editor.installEventFilter(self.error_highlighter)
        layout.addWidget(self.yaml_editor)

        button_layout = QHBoxLayout()
        load_button = QPushButton("Load Configuration")
        load_button.clicked.connect(self.load_yaml)
        save_button = QPushButton("Save Configuration")
        save_button.clicked.connect(self.save_yaml)
        button_layout.addWidget(load_button)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def validate_yaml(self):
        yaml_content = self.yaml_editor.toPlainText()
        self.yaml_editor.textChanged.disconnect(self.validate_yaml)

        try:
            list(yaml.safe_load_all(yaml_content))
            self.error_highlighter.clear_error()
        except yaml.YAMLError as e:
            error_message = str(e)
            line_number = self.extract_error_line(error_message)
            if line_number is not None:
                self.error_highlighter.set_error(line_number, error_message)
        finally:
            self.yaml_editor.textChanged.connect(self.validate_yaml)

    def extract_error_line(self, error_message):
        match = re.search(r"line (\d+)", error_message)
        return int(match.group(1)) if match else None

    def load_yaml(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Configuration", self.directory, "YAML Files (*.yaml)")
        if file_name:
            logger.debug("Loading YAML configuration from file: %s", file_name)
            with open(file_name, "r") as file:
                try:
                    yaml_content = list(yaml.safe_load_all(file))
                    yaml_text = "---\n".join(
                        yaml.dump(doc, Dumper=CustomDumper, default_flow_style=None, sort_keys=False)
                        for doc in yaml_content
                        if doc
                    )
                    self.yaml_editor.setPlainText(yaml_text)
                except yaml.YAMLError as e:
                    logger.error("Error loading YAML file %s: %s", file_name, e)
                    self.yaml_editor.setPlainText("Error loading YAML file.")

    def save_yaml(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Configuration", self.directory, "YAML Files (*.yaml)")
        if file_name:
            yaml_content = self.yaml_editor.toPlainText()
            try:
                with open(file_name, "w") as file:
                    if self.multipart:
                        parsed_content = [i for i in list(yaml.safe_load_all(yaml_content)) if i]
                        yaml.dump_all(
                            parsed_content,
                            file,
                            Dumper=CustomDumper,
                            default_flow_style=None,
                            sort_keys=False,
                        )
                    else:
                        parsed_content = yaml.safe_load(yaml_content)
                        yaml.dump(
                            parsed_content,
                            file,
                            Dumper=CustomDumper,
                            default_flow_style=None,
                            sort_keys=False,
                        )

                logger.debug("Saved YAML configuration to file: %s", file_name)
                self.fileSaved.emit(file_name)

            except yaml.YAMLError as e:
                logger.error("Error saving YAML to file %s: %s", file_name, e)

    def get_yaml_content(self):
        try:
            return list(yaml.safe_load_all(self.yaml_editor.toPlainText()))
        except yaml.YAMLError as e:
            logger.error("YAML parsing error: %s", e)
            return []

    def set_yaml_content(self, yaml_content):
        if isinstance(yaml_content, list):
            yaml_text = "---\n".join(
                yaml.dump(doc, Dumper=CustomDumper, default_flow_style=None, sort_keys=False) for doc in yaml_content
            )
        elif isinstance(yaml_content, dict):
            yaml_text = yaml.dump(yaml_content, Dumper=CustomDumper, default_flow_style=None, sort_keys=False)
        else:
            yaml_text = yaml_content
        self.yaml_editor.setPlainText(yaml_text)

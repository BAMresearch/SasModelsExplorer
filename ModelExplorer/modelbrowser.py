"""Tree-based model browser for selecting sasmodels model names."""

from __future__ import annotations

import sasmodels.core
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ModelBrowser(QWidget):
    """Widget that groups and filters available sasmodels models."""

    model_selected = pyqtSignal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Browser")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(500, 700)

        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search models...")
        layout.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        layout.addWidget(self.tree)

        self._model_metadata: dict[str, bool] = {}
        self._populate()
        self.tree.itemDoubleClicked.connect(self._handle_click)
        self.search.textChanged.connect(self._filter_tree)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        """Focus the search box when the browser becomes visible."""

        super().showEvent(event)
        self.search.setFocus()
        self.search.selectAll()

    def _populate(self) -> None:
        """Populate grouped model names into the tree widget."""

        models = sasmodels.core.list_models()
        groups: dict[str, list[str]] = {}

        for model_name in models:
            info = sasmodels.core.load_model_info(model_name)
            group = info.category.split(":")[0]
            groups.setdefault(group, []).append(info.id)
            self._model_metadata[info.id] = info.category.lower().startswith("structure")

        for group_name, model_list in sorted(groups.items()):
            parent_item = QTreeWidgetItem([group_name])
            parent_item.setFlags(parent_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.tree.addTopLevelItem(parent_item)

            for child_name in sorted(model_list):
                parent_item.addChild(QTreeWidgetItem([child_name]))

        self.tree.expandAll()

    def _handle_click(self, item: QTreeWidgetItem, column: int) -> None:
        """Emit selected model name and whether it is a structure factor."""

        _ = column
        if item.parent() is None:
            return
        model_name = item.text(0)
        is_structure = self._model_metadata.get(model_name, False)
        self.model_selected.emit(model_name, is_structure)

    def _filter_tree(self, text: str) -> None:
        """Filter displayed models according to case-insensitive search text."""

        filter_text = text.lower().strip()
        for index in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(index)
            parent_visible = False
            for child_index in range(parent.childCount()):
                child = parent.child(child_index)
                match = filter_text in child.text(0).lower()
                child.setHidden(not match)
                if match:
                    parent_visible = True
            parent.setHidden(not parent_visible)

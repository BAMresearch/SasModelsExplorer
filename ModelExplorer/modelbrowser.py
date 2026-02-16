import sasmodels.core
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ModelBrowser(QWidget):
    model_selected = pyqtSignal(str, bool)  # (model_name, is_structure_factor)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Browser")
        self.setWindowFlags(Qt.WindowType.Window)

        # ✅ Make window larger by default
        self.resize(500, 700)

        layout = QVBoxLayout(self)

        # 🔍 Search field
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search models...")
        layout.addWidget(self.search)

        # 🌳 Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        layout.addWidget(self.tree)

        self._populate()

        self.tree.itemDoubleClicked.connect(self._handle_click)
        self.search.textChanged.connect(self._filter_tree)

    # ------------------------
    # Focus behavior
    # ------------------------

    def showEvent(self, event):
        super().showEvent(event)

        # ✅ Auto-focus search field
        self.search.setFocus()

        # Optional: select existing text so typing replaces it
        self.search.selectAll()

    # ------------------------
    # Populate tree
    # ------------------------

    def _populate(self):
        models = sasmodels.core.list_models()

        groups = {}
        self._model_metadata = {}

        for m in models:
            info = sasmodels.core.load_model_info(m)
            group = info.category.split(":")[0]
            groups.setdefault(group, []).append(info.id)

            is_structure = info.category.lower().startswith("structure")
            self._model_metadata[info.id] = is_structure

        for group_name, model_list in sorted(groups.items()):
            parent_item = QTreeWidgetItem([group_name])
            parent_item.setFlags(parent_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.tree.addTopLevelItem(parent_item)

            for model_name in sorted(model_list):
                child = QTreeWidgetItem([model_name])
                parent_item.addChild(child)

        self.tree.expandAll()

    # ------------------------
    # Handle selection
    # ------------------------

    def _handle_click(self, item, column):
        if item.parent() is None:
            return

        model_name = item.text(0)
        is_structure = self._model_metadata.get(model_name, False)

        self.model_selected.emit(model_name, is_structure)

    # ------------------------
    # Live filtering
    # ------------------------

    def _filter_tree(self, text: str):
        text = text.lower().strip()

        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            parent_visible = False

            for j in range(parent.childCount()):
                child = parent.child(j)
                match = text in child.text(0).lower()
                child.setHidden(not match)

                if match:
                    parent_visible = True

            parent.setHidden(not parent_visible)

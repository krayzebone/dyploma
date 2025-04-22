import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QTabWidget, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox,
    QGroupBox, QFormLayout
)

class OptimizeRectSection(QWidget):
    """
    A new tab to collect inputs for:
      - MEd, b, h (QLineEdits)
      - figl (checkboxes)
      - fck (checkboxes)
    """
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # --- Inputs for MEd, b, h ---
        input_group = QGroupBox("Inputs")
        form_layout = QFormLayout()

        self.mEd_input = QLineEdit()
        self.b_input = QLineEdit()
        self.h_input = QLineEdit()

        form_layout.addRow("MEd:", self.mEd_input)
        form_layout.addRow("b:", self.b_input)
        form_layout.addRow("h:", self.h_input)
        input_group.setLayout(form_layout)
        layout.addWidget(input_group)

        # --- Checkboxes for figl ---
        self.figl_values = [8, 10, 12, 16, 20, 25, 28, 32]
        self.figl_checkboxes = []
        figl_group = QGroupBox("Średnice prętów")
        figl_layout = QVBoxLayout()

        for val in self.figl_values:
            cb = QCheckBox(str(val))
            figl_layout.addWidget(cb)
            self.figl_checkboxes.append(cb)

        figl_group.setLayout(figl_layout)
        layout.addWidget(figl_group)

        # --- Checkboxes for fck ---
        self.fck_values = [25, 30, 35, 40, 45]
        self.fck_checkboxes = []
        fck_group = QGroupBox("Klasy Betonu")
        fck_layout = QVBoxLayout()

        for val in self.fck_values:
            cb = QCheckBox(str(val))
            fck_layout.addWidget(cb)
            self.fck_checkboxes.append(cb)

        fck_group.setLayout(fck_layout)
        layout.addWidget(fck_group)

        # --- Button to gather the data ---
        get_values_btn = QPushButton("Oblicz")
        get_values_btn.clicked.connect(self.get_values)
        layout.addWidget(get_values_btn)

        self.setLayout(layout)

    def get_values(self):
        """
        Collects the inputs from line edits and checkboxes, then prints them.
        Replace print statements with your own logic as needed.
        """
        # Get MEd, b, h
        MEd = self.mEd_input.text()
        b = self.b_input.text()
        h = self.h_input.text()

        # Collect selected figl
        selected_figl = []
        for i, cb in enumerate(self.figl_checkboxes):
            if cb.isChecked():
                selected_figl.append(self.figl_values[i])

        # Collect selected fck
        selected_fck = []
        for i, cb in enumerate(self.fck_checkboxes):
            if cb.isChecked():
                selected_fck.append(self.fck_values[i])

        print(f"MEd = {MEd}, b = {b}, h = {h}")
        print(f"Selected figl: {selected_figl}")
        print(f"Selected fck: {selected_fck}")
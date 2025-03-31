from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QGridLayout, QHBoxLayout, QSizePolicy, QPushButton

class RectSectionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.moment_value = None
        self.setup_ui()

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout

class MainTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        info_label = QLabel(
            "Instrukcja korzystania z programu.\n\n"
            "Program ma za zadanie optymalizację przekroju żelbetowego..."
        )
        layout.addWidget(info_label)
        
        self.input_layout = QGridLayout()
        self.next_button = QPushButton("Dalej")
        self.input_layout.addWidget(self.next_button, 0, 0, 1, 2)
        layout.addLayout(self.input_layout)

        self.setLayout(layout)

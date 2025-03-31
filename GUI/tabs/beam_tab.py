import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QLineEdit, \
    QGridLayout, QPushButton, QTextEdit, QHBoxLayout, QSizePolicy
from PyQt6.QtGui import QPixmap
from calculations.beams.beam_calcs import calculate_moments
from GUI.resource_path import resource_path

class BeamTab(QWidget):
    def __init__(self):
        super().__init__()
        self.moment_max = None
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Dropdown
        self.dropdown = QComboBox()
        self.dropdown.addItems(["Przegub-przegub", "Przegub-sztywny", "Sztywny-sztywny", "Wspornik"])
        self.dropdown.currentIndexChanged.connect(self.update_image)
        self.dropdown.setFixedSize(200, 25)
        main_layout.addWidget(self.dropdown)

        # Image label
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(200, 125)
        main_layout.addWidget(self.image_label)

        # Input layout
        self.input_layout = QGridLayout()

        # length
        self.length_label = QLabel("Długość [m]:")
        self.length_input = QLineEdit()
        self.length_input.setPlaceholderText("Wprowadź długość")
        self.input_layout.addWidget(self.length_label, 0, 0)
        self.input_layout.addWidget(self.length_input, 0, 1)

        # load
        self.load_label = QLabel("Obciążenie [kN/m]:")
        self.load_input = QLineEdit()
        self.load_input.setPlaceholderText("Wprowadź obciążenie")
        self.input_layout.addWidget(self.load_label, 1, 0)
        self.input_layout.addWidget(self.load_input, 1, 1)

        # calculate button
        self.calculate_button = QPushButton("Oblicz")
        self.calculate_button.clicked.connect(self.calculate_beam_moment)
        self.input_layout.addWidget(self.calculate_button, 2, 0, 1, 2)

        # result
        self.result_label = QLabel("Max Moment [kNm]:")
        self.result_output = QLineEdit()
        self.result_output.setReadOnly(True)
        self.input_layout.addWidget(self.result_label, 3, 0)
        self.input_layout.addWidget(self.result_output, 3, 1)

        # next (save) button
        self.next_button = QPushButton("Zapisz moment")
        self.input_layout.addWidget(self.next_button, 4, 0, 1, 2)

        hbox = QHBoxLayout()
        hbox.addLayout(self.input_layout)
        hbox.addStretch()
        main_layout.addLayout(hbox)

        # log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        main_layout.addWidget(self.log)

        # default image load
        self.update_image(0)

    def update_image(self, index):
        base_dir = resource_path("GUI\resources\images")
        images = {
            0: os.path.join(base_dir, "przegub_przegub_belka.png"),
            1: os.path.join(base_dir, "przegub_sztywny_belka.png"),
            2: os.path.join(base_dir, "sztywny_sztywny_belka.png"),
            3: os.path.join(base_dir, "wspornik_belka.png")
        }
        pixmap = QPixmap(images.get(index, images[0]))
        if pixmap.isNull():
            self.log.append(f"Error: Could not load image for index {index}")
        else:
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))
            self.log.append(f"Loaded image: {self.dropdown.currentText()}")

    def calculate_beam_moment(self):
        try:
            beam_type = self.dropdown.currentText()
            length = float(self.length_input.text())
            load = float(self.load_input.text())
            moment_max = calculate_moments(beam_type, length, load)
            self.moment_max = moment_max
            self.result_output.setText(f"{moment_max:.3f}")
            self.log.append(f"Wyznaczony moment maksymalny = {moment_max:.3f} kNm")
        except ValueError:
            self.log.append("Error: Invalid numerical input.")
        except Exception as e:
            self.log.append(f"Calculation error: {e}")

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QGridLayout, QHBoxLayout, QSizePolicy, QPushButton
from calculations.section_calcs.T_section_calc import wymiarowanie_przekroju_teowego

class SectionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.moment_value = None
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.moment_label = QLabel("No moment has been passed yet.")
        main_layout.addWidget(self.moment_label)

        # Input grid
        self.input_layout = QGridLayout()

        # Max Moment
        self.result_label = QLabel("Max Moment [kNm]:")
        self.result_output = QLineEdit()
        self.result_output.setReadOnly(True)
        self.input_layout.addWidget(self.result_label, 0, 0)
        self.input_layout.addWidget(self.result_output, 0, 1)

        # Input fields
        self.inputs = {}  # Store references to input fields
        labels = ["b_eff [mm]", "b_w [mm]", "h [mm]", "h_f [mm]", "f_ck [MPa]", 
                  "f_yk [MPa]", "c_nom [mm]", "fi_gł [mm]", "fi_str [mm]"]
        
        for i, label_text in enumerate(labels, start=1):
            label = QLabel(label_text)
            input_field = QLineEdit()
            input_field.setPlaceholderText(f"Wprowadź {label_text}")
            self.input_layout.addWidget(label, i, 0)
            self.input_layout.addWidget(input_field, i, 1)
            self.inputs[label_text] = input_field  # Save input field reference

        # Calculate button
        self.calculate_button = QPushButton("Oblicz")
        self.calculate_button.clicked.connect(self.calculate_results)
        self.input_layout.addWidget(self.calculate_button, len(labels) + 1, 0, 1, 2)

        # Output fields for the function results
        self.result_1_label = QLabel("A_s1:")
        self.result_1_output = QLineEdit()
        self.result_1_output.setReadOnly(True)
        self.input_layout.addWidget(self.result_1_label, len(labels) + 2, 0)
        self.input_layout.addWidget(self.result_1_output, len(labels) + 2, 1)

        self.result_2_label = QLabel("A_s2:")
        self.result_2_output = QLineEdit()
        self.result_2_output.setReadOnly(True)
        self.input_layout.addWidget(self.result_2_label, len(labels) + 3, 0)
        self.input_layout.addWidget(self.result_2_output, len(labels) + 3, 1)

        hbox = QHBoxLayout()
        hbox.addLayout(self.input_layout)
        hbox.addStretch()
        main_layout.addLayout(hbox)

    def calculate_results(self):
        try:
            param_map = {
                "b_eff [mm]": "b_eff",
                "b_w [mm]":  "b_w",
                "h [mm]":    "h",
                "h_f [mm]":  "h_f",
                "f_ck [MPa]": "f_ck",
                "f_yk [MPa]": "f_yk",
                "c_nom [mm]": "c_nom",
                "fi_gł [mm]": "fi_gl",
                "fi_str [mm]": "fi_str",
            }

            # Collect user inputs
            inputs = {
                param_map[label_text]: float(field.text())
                for label_text, field in self.inputs.items()
                if field.text().strip()
            }

            # Make sure everything is filled out
            if len(inputs) != len(param_map):
                self.moment_label.setText("❌ Wprowadź wszystkie wartości wejściowe!")
                return

            # Check that we also have a valid moment
            if self.moment_value is None:
                self.moment_label.setText("❌ Brak momentu z poprzedniej zakładki!")
                return

            # Include the moment in the parameters if your function requires it, e.g. M_ed
            inputs["M_Ed"] = self.moment_value

            # Now call the function with the geometry, material data + moment
            result1, result2 = wymiarowanie_przekroju_teowego(**inputs)

            # Update output fields
            self.result_1_output.setText(f"{result1:.3f}")
            self.result_2_output.setText(f"{result2:.3f}")

        except ValueError:
            self.moment_label.setText("❌ Błąd: Wprowadź poprawne liczby!")

    def display_moment(self, moment_value):
        """Display moment passed from BeamTab."""
        if moment_value is None:
            self.moment_label.setText("No moment has been calculated yet.")
            self.result_output.setText("")
            self.moment_value = None
        else:
            self.moment_label.setText(f"Calculated Moment: {moment_value:.3f} kNm")
            self.result_output.setText(f"{moment_value:.3f}")
            self.moment_value = moment_value  # <-- Store it

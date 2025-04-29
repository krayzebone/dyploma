from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QGridLayout, QHBoxLayout,
    QPushButton, QCheckBox, QGroupBox, QScrollArea
)
from PyQt6.QtCore import Qt

from GUI.T_section_calc.T_section_plus import find_optimal_scenario


class SectionTab(QWidget):
    """One tab that collects geometry + material data,
    calls `find_optimal_scenario`, and shows the best result."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------ UI ----
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.status_label = QLabel()                 # ← red/green status line
        root.addWidget(self.status_label)

        grid = QGridLayout()                         # everything else
        self._add_input_boxes(grid)                  # geometry & cover
        self._add_fck_group(grid)                    # concrete classes
        self._add_fi_group(grid)                     # main-bar diameters
        self._add_results_group(grid)                # read-only results
        self._add_calculate_button(grid)             # runs the optimiser

        # put the grid in a scroll-area so the tab works on small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        wrap = QWidget()
        wrap.setLayout(grid)
        scroll.setWidget(wrap)
        root.addWidget(scroll)

    # -------------------------- helpers to build each chunk of the grid -------
    def _add_input_boxes(self, grid):
        labels = ["MEd [kNm]", "beff [mm]", "bw [mm]", "h [mm]", "hf [mm]",
                  "c_nom [mm]", "fi_str [mm]"]
        self.inputs = {}
        for row, txt in enumerate(labels):
            grid.addWidget(QLabel(txt), row, 0)
            le = QLineEdit()
            le.setPlaceholderText(f"Enter {txt}")
            le.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(le, row, 1)
            self.inputs[txt] = le

    def _add_fck_group(self, grid):
        self.fck_options = {
            "C12/15": 12, "C16/20": 16, "C20/25": 20, "C25/30": 25,
            "C30/37": 30, "C35/45": 35, "C40/50": 40, "C45/55": 45,
            "C50/60": 50,
        }
        self.fck_check = {}
        box = QGroupBox("Concrete class fck [MPa]")
        lay = QVBoxLayout(box)
        for k in self.fck_options:
            cb = QCheckBox(k)
            lay.addWidget(cb)
            self.fck_check[k] = cb
        grid.addWidget(box, 7, 0, 1, 2)

    def _add_fi_group(self, grid):
        self.fi_options = {
            "Ø6": 6, "Ø8": 8, "Ø10": 10, "Ø12": 12, "Ø14": 14,
            "Ø16": 16, "Ø18": 18, "Ø20": 20, "Ø22": 22, "Ø25": 25,
            "Ø28": 28, "Ø32": 32,
        }
        self.fi_check = {}
        box = QGroupBox("Main bar ∅ [mm]")
        lay = QVBoxLayout(box)
        for k in self.fi_options:
            cb = QCheckBox(k)
            lay.addWidget(cb)
            self.fi_check[k] = cb
        grid.addWidget(box, 8, 0, 1, 2)

    def _add_calculate_button(self, grid):
        self.calc_btn = QPushButton("Calculate optimal solution")
        self.calc_btn.clicked.connect(self._calculate_results)
        grid.addWidget(self.calc_btn, 9, 0, 1, 2)

    def _add_results_group(self, grid):
        labels = [
            "Concrete Class", "Rebar Diameter", "Moment Region",
            "Layers", "Reinforcement Type", "As1 [mm²]", "As2 [mm²]",
            "Number of rods (As1)", "Number of rods (As2)",
            "Actual As1 [mm²]", "Actual As2 [mm²]", "Rods Fit?",
            "Total Cost [zł]"
        ]
        self.result_fields = {}
        box = QGroupBox("Optimal solution")
        lay = QVBoxLayout(box)

        # optional: echo the applied moment on the very first line
        top = QHBoxLayout()
        top.addWidget(QLabel("Applied moment MEd [kNm]"))
        self.result_output = QLineEdit()         # ← missing in original code
        self.result_output.setReadOnly(True)
        top.addWidget(self.result_output)
        lay.addLayout(top)

        for txt in labels:
            row = QHBoxLayout()
            row.addWidget(QLabel(txt))
            out = QLineEdit()
            out.setReadOnly(True)
            out.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(out)
            lay.addLayout(row)
            self.result_fields[txt] = out
        grid.addWidget(box, 10, 0, 1, 2)

    # ----------------------------------------------------------------- logic --
    def _float(self, line_edit: QLineEdit, name: str) -> float | None:
        """Return the value or None and colour the box red when invalid."""
        try:
            val = float(line_edit.text())
            line_edit.setStyleSheet("")          # clear error colour
            return val
        except ValueError:
            line_edit.setStyleSheet("background:#ffdddd")
            self.status_label.setText(f"❌ ‘{name}’ is not a number.")
            return None

    def _selected(self, mapping, checks) -> list[int]:
        """Return all numbers where the corresponding checkbox is ticked."""
        return [mapping[k] for k, cb in checks.items() if cb.isChecked()]

    def _calculate_results(self):
        self.status_label.clear()

        # 1. collect lists from the check-boxes ------------------------------
        fck_list = self._selected(self.fck_options, self.fck_check)
        fi_list  = self._selected(self.fi_options,  self.fi_check)
        if not fck_list or not fi_list:
            self.status_label.setText("❌ Tick at least one fck and one ∅.")
            return

        # 2. read the numeric input boxes ------------------------------------
        raw = {}
        for key, le in self.inputs.items():
            val = self._float(le, key)
            if val is None:
                return                          # error already shown
            raw[key] = val

        inputs = {                # rename to exactly what the backend expects
            "MEd":   raw["MEd [kNm]"],
            "beff":  raw["beff [mm]"],
            "bw":    raw["bw [mm]"],
            "h":     raw["h [mm]"],
            "hf":    raw["hf [mm]"],
            "cnom":  raw["c_nom [mm]"],
            "fi_str": raw["fi_str [mm]"],
        }

        # 3. run optimiser ----------------------------------------------------
        try:
            best = find_optimal_scenario(inputs, fi_list, fck_list)
        except Exception as exc:
            self.status_label.setText(f"❌ Optimiser error: {exc}")
            return

        if not best or best.get("cost", float("inf")) == float("inf"):
            self.status_label.setText("❌ No valid solution for those choices.")
            return

        # 4. show the results -------------------------------------------------
        self.result_output.setText(f"{inputs['MEd']:.2f}")
        self.result_fields["Concrete Class"].setText(
            f"C{best['fck']}/{best['fck']+5}"
        )
        self.result_fields["Rebar Diameter"].setText(f"Ø{best['fi']} mm")
        self.result_fields["Moment Region"].setText(best["type"])
        self.result_fields["Layers"].setText(str(best["layers"]))
        self.result_fields["Reinforcement Type"].setText(best["reinforcement_type"])
        self.result_fields["As1 [mm²]"].setText(f"{best['As1']:.1f}")
        self.result_fields["As2 [mm²]"].setText(f"{best['As2']:.1f}")
        self.result_fields["Number of rods (As1)"].setText(str(best["num_rods_As1"]))
        self.result_fields["Number of rods (As2)"].setText(str(best["num_rods_As2"]))
        self.result_fields["Actual As1 [mm²]"].setText(f"{best['actual_As1']:.1f}")
        self.result_fields["Actual As2 [mm²]"].setText(f"{best['actual_As2']:.1f}")
        self.result_fields["Rods Fit?"].setText("Yes" if best["fit_check"] else "No")
        self.result_fields["Total Cost [zł]"].setText(f"{best['cost']:.2f}")

        self.status_label.setText("✓ Optimal solution found.")


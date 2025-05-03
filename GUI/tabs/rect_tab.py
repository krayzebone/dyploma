"""
T-section optimiser – one‑file version (fixed)
---------------------------------------------

*Click the check‑boxes, fill in the geometry/material values,
then hit “Calculate optimal solution”.*
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

# --------------------  CALCULATION ENGINE  ----------------------------------


def quadratic_equation(a: float, b: float, c: float, limit: float) -> float | None:
    """Solve ax²+bx+c = 0 and return the root that lies in (0, limit)."""
    if a == 0:
        return None

    delta = b ** 2 - 4 * a * c
    if delta < 0:
        return None

    sqrt_delta = math.sqrt(delta)
    x1 = (-b - sqrt_delta) / (2 * a)
    x2 = (-b + sqrt_delta) / (2 * a)
    valid = [x for x in (x1, x2) if 0 < x < limit]
    return min(valid) if valid else None


def calc_cost(
    b: float,
    h: float,
    fck: float,
    As1: float,
    As2: float,
) -> float:
    """Concrete + steel cost in PLN."""
    concrete_cost_by_class = {
        8: 230,
        12: 250,
        16: 300,
        20: 350,
        25: 400,
        30: 450,
        35: 500,
        40: 550,
        45: 600,
        50: 650,
        55: 700,
        60: 800,
    }

    steel_cost = (As1 + As2) / 1_000_000 * 7_900 * 5  # mm²→m² * ρ * price
    conc_area = ((b * h) / 1_000_000 - (As1 + As2)) / 1_000_000
    conc_cost = conc_area * concrete_cost_by_class[int(fck)]
    return steel_cost + conc_cost


def calc_rect_section(MEd, b, h, fck, fi_gl, c_nom):

    # Parametry materiałowe
    fyk = 500
    fcd = fck / 1.4  # wytrzymałość obliczeniowa betonu na ściskanie [MPa]
    fyd = fyk / 1.15  # wytrzymałość obliczeniowa stali na rozciąganie [MPa]
    E_s = 200_000.0  # moduł Younga stali [MPa]

    # Wyznaczenie współczynnika względnej wysokości strefy ściskanej
    ksi_eff_lim = 0.8 * 0.0035 / (0.0035 + fyd / E_s)

    # Podstawowe wymiary przekroju
    fi_str = 8
    a_1 = c_nom + fi_gl / 2 + fi_str
    d = h - a_1

    aq = -0.5 * b * fcd
    bq = b * fcd * d
    cq = -MEd * 1e6

    xeff = quadratic_equation(aq, bq, cq, h)
    if xeff is None:
        # brak rozwiązania, zwróć nieskończony koszt aby odrzucić wariant
        return float("inf"), float("inf"), float("inf"), "brak rozwiązania"

    ksieff = xeff / d

    if ksieff > ksi_eff_lim:
        reinforcement_type = "podwójnie zbrojony"

        x_eff = ksi_eff_lim * d
        As2 = (
            -x_eff * b * fcd * (d - 0.5 * x_eff) + MEd * 1e6
        ) / (fyd * (d - a_1))
        As1 = (As2 * fyd + x_eff * b * fcd) / fyd
        cost = calc_cost(b, h, fck, As1, As2)
        return As1, As2, cost, reinforcement_type

    reinforcement_type = "pojedyńczo zbrojony"
    As1 = xeff * b * fcd / fyd
    As2 = 0.0
    cost = calc_cost(b, h, fck, As1, As2)
    return As1, As2, cost, reinforcement_type


def calculate_number_of_rods(As: float, fi: float) -> tuple[int, float]:
    """Return number of bars and the provided reinforcement area."""
    if As <= 0 or math.isinf(As):
        return 0, 0.0
    area_bar = math.pi * fi ** 2 / 4
    n = math.ceil(As / area_bar)
    return n, n * area_bar


def check_rods_fit(b: float, cnom: float, num_rods: int, fi: float, smax: float) -> bool:
    """Check clear spacing rules in *one* reinforcement layer."""
    if num_rods == 0:
        return True
    required = 2 * cnom + num_rods * fi + smax * (num_rods - 1)
    return required <= b


@dataclass
class Inputs:
    MEd: float
    b: float
    h: float
    cnom: float
    fi_str: float


def find_optimal_scenario(
    inputs: dict[str, float], possible_fi: list[int], possible_fck: list[int]
) -> dict:
    """Search all (fck, fi, layout) combinations – return the cheapest fit."""
    MEd, b, h, cnom, fi_str = (
        inputs[k] for k in ("MEd", "b", "h", "cnom", "fi_str")
    )
    fyk = 500.0
    best = {"cost": float("inf")}

    for fck in possible_fck:
        fcd = fck / 1.4
        fyd = fyk / 1.15
        for fi in possible_fi:
            smax = max(20, fi)
            As1, As2, cost, rtype = calc_rect_section(MEd, b, h, fck, fi, cnom)
            n1, act1 = calculate_number_of_rods(As1, fi)
            n2, act2 = calculate_number_of_rods(As2, fi)
            fits = check_rods_fit(b, cnom, n1, fi, smax) and check_rods_fit(
                b, cnom, n2, fi, smax
            )

            if fits and cost < best["cost"]:
                best = {
                    "cost": cost,
                    "fck": fck,
                    "fi": fi,
                    "reinforcement_type": rtype,
                    "As1": As1,
                    "As2": As2,
                    "num_rods_As1": n1,
                    "num_rods_As2": n2,
                    "actual_As1": act1,
                    "actual_As2": act2,
                    "fit_check": fits,
                }
    return best


# ---------------------------------------------------------------------------
#                           P Y Q T   G U I
# ---------------------------------------------------------------------------
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QCheckBox,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QDialog,
)
from PyQt6.QtGui import QFontMetrics

# ---------------------------------------------------------------------------
#  T-section drawing widget – placed above SectionTab
# ---------------------------------------------------------------------------


class RectSectionTab(QWidget):
    """One tab that collects geometry + material data and shows the best result."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._best: dict | None = None
        self._last_params: dict | None = None

    # ---------- UI ----------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.status = QLabel()
        root.addWidget(self.status)

        grid = QGridLayout()

        # --- top row: three group boxes side‑by‑side ------------------------
        box_params = self._create_param_box()
        box_fck = self._create_fck_box()
        box_fi = self._create_fi_box()

        # size policies – let material boxes grow equally; parameters fixed min width
        box_fck.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        box_fi.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        hbox_top = QHBoxLayout()
        hbox_top.addWidget(box_params)
        hbox_top.addWidget(box_fck)
        hbox_top.addWidget(box_fi)
        hbox_top.setStretchFactor(box_fck, 1)
        hbox_top.setStretchFactor(box_fi, 1)

        grid.addLayout(hbox_top, 0, 0, 1, 2)

        # --- calculate button ----------------------------------------------
        self._add_button(grid, row=1)

        # -------- NEW: extra button just under the calculate one ------------
        self.btn_draw = QPushButton("Pokaż przekrój")
        self.btn_draw.setEnabled(False)
        self.btn_draw.clicked.connect(self._open_section_win)
        grid.addWidget(self.btn_draw, 2, 0, 1, 2)

        # --- results section -----------------------------------------------
        self._add_results_group(grid, row=3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        wrap = QWidget()
        wrap.setLayout(grid)
        scroll.setWidget(wrap)
        root.addWidget(scroll)

    # ---- helpers -----------------------------------------------------------
    def _create_param_box(self) -> QGroupBox:
        labels = [
            "MEd [kNm]",
            "b [mm]",
            "h [mm]",
            "c_nom [mm]",
            "fi_str [mm]",
        ]
        self.inputs: dict[str, QLineEdit] = {}
        box = QGroupBox("Parametry przekroju")
        lay = QGridLayout(box)
        for row, txt in enumerate(labels):
            lay.addWidget(QLabel(txt), row, 0)
            le = QLineEdit(alignment=Qt.AlignmentFlag.AlignRight)
            le.setPlaceholderText("…")
            lay.addWidget(le, row, 1)
            self.inputs[txt] = le
        return box

    # --- Concrete class group ----------------------------------------------
    def _create_fck_box(self) -> QGroupBox:
        self.fck_opts = {
            "C12/15": 12,
            "C16/20": 16,
            "C20/25": 20,
            "C25/30": 25,
            "C30/37": 30,
            "C35/45": 35,
            "C40/50": 40,
            "C45/55": 45,
            "C50/60": 50,
        }
        self.fck_check: dict[str, QCheckBox] = {}

        box = QGroupBox("Klasa betonu")
        lay = QVBoxLayout(box)
        for label in self.fck_opts:
            cb = QCheckBox(label)
            self.fck_check[label] = cb
            lay.addWidget(cb)
        lay.addStretch()
        btn_all = QPushButton("Zaznacz wszystkie")
        btn_all.clicked.connect(
            lambda _=None: [cb.setChecked(True) for cb in self.fck_check.values()]
        )
        lay.addWidget(btn_all)
        return box

    # --- Main‑bar diameter group -------------------------------------------
    def _create_fi_box(self) -> QGroupBox:
        self.fi_opts = {
            "Ø6": 6,
            "Ø8": 8,
            "Ø10": 10,
            "Ø12": 12,
            "Ø14": 14,
            "Ø16": 16,
            "Ø18": 18,
            "Ø20": 20,
            "Ø22": 22,
            "Ø25": 25,
            "Ø28": 28,
            "Ø32": 32,
        }
        self.fi_check: dict[str, QCheckBox] = {}

        box = QGroupBox("Średnica zbrojenia")
        lay = QVBoxLayout(box)
        for label in self.fi_opts:
            cb = QCheckBox(label)
            self.fi_check[label] = cb
            lay.addWidget(cb)
        lay.addStretch()
        btn_all = QPushButton("Zaznacz wszystkie")
        btn_all.clicked.connect(
            lambda _=None: [cb.setChecked(True) for cb in self.fi_check.values()]
        )
        lay.addWidget(btn_all)
        return box

    # --- Calculate button ---------------------------------------------------
    def _add_button(self, grid: QGridLayout, *, row: int) -> None:
        self.btn = QPushButton("Oblicz optymalne rozwiązanie")
        self.btn.clicked.connect(self._run_optimiser)
        grid.addWidget(self.btn, row, 0, 1, 2)

    # --- Results ------------------------------------------------------------
    def _add_results_group(self, grid: QGridLayout, *, row: int) -> None:
        labels = [
            "Klasa betonu: ",
            "Średnica zbrojenia: ",
            "Liczba warstw zbrojenia głównego: ",
            "As1_req [mm²]: ",
            "As2_req [mm²]: ",
            "Liczba prętów rozciąganych: ",
            "Liczba prętów ściskanych: ",
            "As1_prov [mm²]",
            "As2_prov [mm²]",
            "Rozstaw prętów: ",
            "Całkowity koszt [zł/mb]",
        ]
        self.result: dict[str, QLineEdit] = {}
        box = QGroupBox("Optymalny przekrój")
        lay = QVBoxLayout(box)
        for txt in labels:
            row_h = QHBoxLayout()
            row_h.addWidget(QLabel(txt))
            out = QLineEdit(readOnly=True, alignment=Qt.AlignmentFlag.AlignRight)
            row_h.addWidget(out)
            lay.addLayout(row_h)
            self.result[txt] = out
        grid.addWidget(box, row, 0, 1, 2)

    # ---------- logic -------------------------------------------------------
    def _float(self, le: QLineEdit, name: str) -> float | None:
        try:
            val = float(le.text())
            le.setStyleSheet("")
            return val
        except ValueError:
            le.setStyleSheet("background:#ffdddd")
            self.status.setText(f"❌ “{name}” is not a number.")
            return None

    def _chosen(self, mapping: dict[str, int], checks: dict[str, QCheckBox]) -> list[int]:
        return [mapping[k] for k, cb in checks.items() if cb.isChecked()]

    def _run_optimiser(self) -> None:
        self.status.clear()

        fck_list = self._chosen(self.fck_opts, self.fck_check)
        fi_list = self._chosen(self.fi_opts, self.fi_check)
        if not fck_list or not fi_list:
            self.status.setText("❌ Tick at least one fck and one ∅.")
            return

        raw: dict[str, float] = {}
        for key, le in self.inputs.items():
            v = self._float(le, key)
            if v is None:
                return
            raw[key] = v

        params = {
            "MEd": raw["MEd [kNm]"],
            "b": raw["b [mm]"],
            "h": raw["h [mm]"],
            "cnom": raw["c_nom [mm]"],
            "fi_str": raw["fi_str [mm]"],
        }

        try:
            best = find_optimal_scenario(params, fi_list, fck_list)
            self._best = best
        except Exception as exc:  # pragma: no cover
            self.status.setText(f"❌ optimiser error: {exc}")
            return

        if not best or best["cost"] == float("inf"):
            self.status.setText("❌ Brak rozwiązań.")
            return

        # Store the parameters for the drawing
        self._last_params = params
        self.btn_draw.setEnabled(True)  # Enable the draw button

        self._show(best)
        self.status.setText("✓ Optimalny przekrój znaleziony.")

    def _show(self, b: dict) -> None:
        self.result["Klasa betonu: "].setText(f"C{b['fck']}/{b['fck'] + 5}")
        self.result["Średnica zbrojenia: "].setText(f"Ø{b['fi']} mm")
        self.result["Liczba warstw zbrojenia głównego: "].setText(b["reinforcement_type"])
        self.result["As1_req [mm²]: "].setText(f"{b['As1']:.1f}")
        self.result["As2_req [mm²]: "].setText(f"{b['As2']:.1f}")
        self.result["Liczba prętów rozciąganych: "].setText(str(b["num_rods_As1"]))
        self.result["Liczba prętów ściskanych: "].setText(str(b["num_rods_As2"]))
        self.result["As1_prov [mm²]"].setText(f"{b['actual_As1']:.1f}")
        self.result["As2_prov [mm²]"].setText(f"{b['actual_As2']:.1f}")
        self.result["Rozstaw prętów: "].setText("✔" if b["fit_check"] else "✘")
        self.result["Całkowity koszt [zł/mb]"].setText(f"{b['cost']:.2f}")

    # ---------- drawing -----------------------------------------------------
    def _open_section_win(self):
        """Very simple placeholder – opens a dialog until proper drawing is added."""
        if not self._best:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Podgląd przekroju – w przygotowaniu")
        lay = QVBoxLayout(dlg)
        txt = QLabel("(Podgląd graficzny przekroju nie został jeszcze zaimplementowany.)")
        txt.setWordWrap(True)
        lay.addWidget(txt)
        dlg.exec()


# ---------------------------------------------------------------------------
#  Convenience alias – the original code referred to SectionTab
# ---------------------------------------------------------------------------
SectionTab = RectSectionTab

# -----------------------  run it  -------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    win = SectionTab()
    win.setWindowTitle("T-Section optimiser")
    win.resize(480, 640)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

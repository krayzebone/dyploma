"""
T-section optimiser – one-file version
-------------------------------------

*Click the check-boxes, fill in the geometry/material values,
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

    delta = b**2 - 4 * a * c
    if delta < 0:
        return None

    sqrt_delta = math.sqrt(delta)
    x1 = (-b - sqrt_delta) / (2 * a)
    x2 = (-b + sqrt_delta) / (2 * a)
    valid = [x for x in (x1, x2) if 0 < x < limit]
    return min(valid) if valid else None


def calc_cost(
    beff: float,
    bw: float,
    h: float,
    hf: float,
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
    conc_area = ((beff * hf) + (h - hf) * bw) / 1_000_000 - (As1 + As2) / 1_000_000
    conc_cost = conc_area * concrete_cost_by_class[int(fck)]
    return steel_cost + conc_cost

def calc_PT_1r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    a1 = cnom + fi / 2 + fi_str
    a2 = cnom + fi / 2 + fi_str
    d = h - a1
    
    xeff = quadratic_equation(-0.5 * beff * fcd, beff * fcd * d, -MEd * 1e6, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf'), None

    ksieff = xeff / d
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / 200_000)

    if ksieff <= ksiefflim:
        As1 = xeff * beff * fcd / fyd
        As2 = 0
        reinforcement_type = '1r'
    else:
        xeff = ksiefflim * d
        As2 = (MEd * 1e6 - xeff * beff * fcd * (d - 0.5 * xeff)) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * beff * fcd) / fyd
        reinforcement_type = '2r'

    return As1, As2, calc_cost(beff, bw, h, hf, fck, As1, As2), reinforcement_type


def calc_PT_2r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    smax = max(20, fi)
    a1 = cnom + fi_str + fi + smax / 2
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    xeff = quadratic_equation(-0.5 * beff * fcd, beff * fcd * d, -MEd * 1e6, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf'), None

    ksieff = xeff / d
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / 200_000)

    if ksieff <= ksiefflim:
        As1 = xeff * beff * fcd / fyd
        As2 = 0
        reinforcement_type = '1r'
    else:
        xeff = ksiefflim * d
        As2 = (MEd * 1e6 - xeff * beff * fcd * (d - 0.5 * xeff)) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * beff * fcd) / fyd
        reinforcement_type = '2r'

    return As1, As2, calc_cost(beff, bw, h, hf, fck, As1, As2), reinforcement_type


def calc_PT_3r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    smax = max(20, fi)
    a1 = cnom + fi_str + fi * 3/2 + smax
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    xeff = quadratic_equation(-0.5 * beff * fcd, beff * fcd * d, -MEd * 1e6, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf'), None

    ksieff = xeff / d
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / 200_000)

    if ksieff <= ksiefflim:
        As1 = xeff * beff * fcd / fyd
        As2 = 0
        reinforcement_type = '1r'
    else:
        xeff = ksiefflim * d
        As2 = (MEd * 1e6 - xeff * beff * fcd * (d - 0.5 * xeff)) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * beff * fcd) / fyd
        reinforcement_type = '2r'

    return As1, As2, calc_cost(beff, bw, h, hf, fck, As1, As2), reinforcement_type


def calc_RZT_1r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    a1 = cnom + fi / 2 + fi_str
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    xeff = quadratic_equation(-0.5 * bw * fcd, bw * fcd * d, 
                             hf * (beff - bw) * fcd * (d - 0.5 * hf) - MEd * 1e6, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf'), None

    ksieff = xeff / d
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / 200_000)

    if ksieff <= ksiefflim:
        As1 = (xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        As2 = 0
        reinforcement_type = '1r'
    else:
        xeff = ksiefflim * d
        As2 = (-xeff * bw * fcd * (d - 0.5 * xeff) - hf * (beff - bw) * fcd * (d - 0.5 * hf) + MEd * 1e6) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        reinforcement_type = '2r'

    return As1, As2, calc_cost(beff, bw, h, hf, fck, As1, As2), reinforcement_type


def calc_RZT_2r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    smax = max(20, fi)
    a1 = cnom + fi_str + fi + smax / 2
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    xeff = quadratic_equation(-0.5 * bw * fcd, bw * fcd * d, 
                             hf * (beff - bw) * fcd * (d - 0.5 * hf) - MEd * 1e6, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf'), None

    ksieff = xeff / d
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / 200_000)

    if ksieff <= ksiefflim:
        As1 = (xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        As2 = 0
        reinforcement_type = '1r'
    else:
        xeff = ksiefflim * d
        As2 = (-xeff * bw * fcd * (d - 0.5 * xeff) - hf * (beff - bw) * fcd * (d - 0.5 * hf) + MEd * 1e6) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        reinforcement_type = '2r'

    return As1, As2, calc_cost(beff, bw, h, hf, fck, As1, As2), reinforcement_type


def calc_RZT_3r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    smax = max(20, fi)
    a1 = cnom + fi_str + fi *3/2 + smax
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    xeff = quadratic_equation(-0.5 * bw * fcd, bw * fcd * d, 
                             hf * (beff - bw) * fcd * (d - 0.5 * hf) - MEd * 1e6, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf'), None

    ksieff = xeff / d
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / 200_000)

    if ksieff <= ksiefflim:
        As1 = (xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        As2 = 0
        reinforcement_type = '1r'
    else:
        xeff = ksiefflim * d
        As2 = (-xeff * bw * fcd * (d - 0.5 * xeff) - hf * (beff - bw) * fcd * (d - 0.5 * hf) + MEd * 1e6) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        reinforcement_type = '2r'

    return As1, As2, calc_cost(beff, bw, h, hf, fck, As1, As2), reinforcement_type


def calculate_number_of_rods(As: float, fi: float) -> tuple[int, float]:
    """Return number of bars and the provided reinforcement area."""
    if As <= 0 or math.isinf(As):
        return 0, 0.0
    area_bar = math.pi * fi**2 / 4
    n = math.ceil(As / area_bar)
    return n, n * area_bar


def check_rods_fit(
    bw: float, cnom: float, num_rods: int, fi: float, smax: float, layers: int = 1
) -> bool:
    """Check clear spacing rules in *one* reinforcement layer."""
    if num_rods == 0:
        return True
    required = 2 * cnom + num_rods * fi + smax * (num_rods - 1)
    return required <= layers * bw


@dataclass
class Inputs:
    MEd: float
    beff: float
    bw: float
    h: float
    hf: float
    cnom: float
    fi_str: float


def find_optimal_scenario(
    inputs: dict[str, float], possible_fi: list[int], possible_fck: list[int]
) -> dict:
    """Search all (fck, fi, layout) combinations – return the cheapest fit."""
    MEd, beff, bw, h, hf, cnom, fi_str = (
        inputs[k] for k in ("MEd", "beff", "bw", "h", "hf", "cnom", "fi_str")
    )
    fyk = 500.0
    best = {"cost": float("inf")}

    for fck in possible_fck:
        fcd = fck / 1.4
        fyd = fyk / 1.15
        for fi in possible_fi:
            smax = max(20, fi)
            a1 = {
                1: cnom + fi_str + fi / 2,
                2: cnom + fi_str + fi + smax / 2,
                3: cnom + fi_str + 1.5 * fi + smax,
            }
            for layers, _ in a1.items():
                # decide “pure-T” or “Ribbed T”
                d = h - a1[layers]
                MRd = (beff * hf * fcd * (d - 0.5 * hf)) / 1e6
                t_or_r = "PT" if MEd < MRd else "RZT"
                func = globals()[f"calc_{t_or_r}_{layers}r_plus"]

                As1, As2, cost, rtype = func(
                    MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck
                )
                n1, act1 = calculate_number_of_rods(As1, fi)
                n2, act2 = calculate_number_of_rods(As2, fi)
                fits = check_rods_fit(bw, cnom, n1, fi, smax, layers) and check_rods_fit(
                    bw, cnom, n2, fi, smax, layers
                )

                if fits and cost < best["cost"]:
                    best = {
                        "cost": cost,
                        "fck": fck,
                        "fi": fi,
                        "type": t_or_r,
                        "layers": layers,
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
    QVBoxLayout,
    QWidget,
)


class SectionTab(QWidget):
    """One tab that collects geometry + material data and shows the best result."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ---------- UI ----------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.status = QLabel()
        root.addWidget(self.status)

        grid = QGridLayout()
        self._add_inputs(grid)
        self._add_fck_group(grid)
        self._add_fi_group(grid)
        self._add_results_group(grid)
        self._add_button(grid)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        wrap = QWidget()
        wrap.setLayout(grid)
        scroll.setWidget(wrap)
        root.addWidget(scroll)

    # ---- helpers -----------------------------------------------------------
    def _add_inputs(self, grid: QGridLayout) -> None:
        labels = [
            "MEd [kNm]",
            "beff [mm]",
            "bw [mm]",
            "h [mm]",
            "hf [mm]",
            "c_nom [mm]",
            "fi_str [mm]",
        ]
        self.inputs: dict[str, QLineEdit] = {}
        for row, txt in enumerate(labels):
            grid.addWidget(QLabel(txt), row, 0)
            le = QLineEdit(alignment=Qt.AlignmentFlag.AlignRight)
            le.setPlaceholderText("…")
            grid.addWidget(le, row, 1)
            self.inputs[txt] = le

    def _add_fck_group(self, grid: QGridLayout) -> None:
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
        box = QGroupBox("Concrete class fck")
        lay = QVBoxLayout(box)
        for k in self.fck_opts:
            cb = QCheckBox(k)
            lay.addWidget(cb)
            self.fck_check[k] = cb
        grid.addWidget(box, 7, 0, 1, 2)

    def _add_fi_group(self, grid: QGridLayout) -> None:
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
        box = QGroupBox("Main-bar diameter")
        lay = QVBoxLayout(box)
        for k in self.fi_opts:
            cb = QCheckBox(k)
            lay.addWidget(cb)
            self.fi_check[k] = cb
        grid.addWidget(box, 8, 0, 1, 2)

    def _add_button(self, grid: QGridLayout) -> None:
        self.btn = QPushButton("Calculate optimal solution")
        self.btn.clicked.connect(self._run_optimiser)
        grid.addWidget(self.btn, 9, 0, 1, 2)

    def _add_results_group(self, grid: QGridLayout) -> None:
        labels = [
            "Concrete Class",
            "Rebar Diameter",
            "Moment Region",
            "Layers",
            "Reinforcement Type",
            "As1 [mm²]",
            "As2 [mm²]",
            "Number of rods (As1)",
            "Number of rods (As2)",
            "Actual As1 [mm²]",
            "Actual As2 [mm²]",
            "Rods Fit?",
            "Total Cost [zł]",
        ]
        self.result: dict[str, QLineEdit] = {}
        box = QGroupBox("Optimal solution")
        lay = QVBoxLayout(box)

        for txt in labels:
            row = QHBoxLayout()
            row.addWidget(QLabel(txt))
            out = QLineEdit(readOnly=True, alignment=Qt.AlignmentFlag.AlignRight)
            row.addWidget(out)
            lay.addLayout(row)
            self.result[txt] = out

        grid.addWidget(box, 10, 0, 1, 2)

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

        raw = {}
        for key, le in self.inputs.items():
            v = self._float(le, key)
            if v is None:
                return
            raw[key] = v

        params = {
            "MEd": raw["MEd [kNm]"],
            "beff": raw["beff [mm]"],
            "bw": raw["bw [mm]"],
            "h": raw["h [mm]"],
            "hf": raw["hf [mm]"],
            "cnom": raw["c_nom [mm]"],
            "fi_str": raw["fi_str [mm]"],
        }

        try:
            best = find_optimal_scenario(params, fi_list, fck_list)
        except Exception as exc:  # pragma: no cover
            self.status.setText(f"❌ optimiser error: {exc}")
            return

        if not best or best["cost"] == float("inf"):
            self.status.setText("❌ No valid solution.")
            return

        self._show(best)
        self.status.setText("✓ Optimal solution found.")

    def _show(self, b: dict) -> None:
        self.result["Concrete Class"].setText(f"C{b['fck']}/{b['fck']+5}")
        self.result["Rebar Diameter"].setText(f"Ø{b['fi']} mm")
        self.result["Moment Region"].setText(b["type"])
        self.result["Layers"].setText(str(b["layers"]))
        self.result["Reinforcement Type"].setText(b["reinforcement_type"])
        self.result["As1 [mm²]"].setText(f"{b['As1']:.1f}")
        self.result["As2 [mm²]"].setText(f"{b['As2']:.1f}")
        self.result["Number of rods (As1)"].setText(str(b["num_rods_As1"]))
        self.result["Number of rods (As2)"].setText(str(b["num_rods_As2"]))
        self.result["Actual As1 [mm²]"].setText(f"{b['actual_As1']:.1f}")
        self.result["Actual As2 [mm²]"].setText(f"{b['actual_As2']:.1f}")
        self.result["Rods Fit?"].setText("Yes" if b["fit_check"] else "No")
        self.result["Total Cost [zł]"].setText(f"{b['cost']:.2f}")


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

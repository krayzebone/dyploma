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
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QPainter, QPen, QColor, QFontMetrics, QBrush
from PyQt6.QtWidgets import QDialog 

# ---------------------------------------------------------------------------
# 2)  T-section drawing widget – add anywhere above SectionTab
# ---------------------------------------------------------------------------
class SectionView(QDialog):
    """
    Window that draws the optimal T-section with dimensions and reinforcement.
    """
    def __init__(
        self,
        beff: float, bw: float, h: float, hf: float,                 # geometry
        fi: float, n_bars: int, layers: int,                        # reinforcement
        cnom: float, fi_str: float,                                 # cover / stirrup
        parent=None
    ):
        super().__init__(parent)
        self.beff, self.bw, self.h, self.hf = beff, bw, h, hf
        self.fi, self.n_bars, self.layers = fi, n_bars, layers
        self.cnom, self.fi_str = cnom, fi_str
        self.smax = max(20, fi)

        self.setWindowTitle("Optimal T-section")
        self.resize(600, 500)  # Slightly larger for better visibility

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)

        # ---------- overall scale ----------
        margin = 0.10
        w_pix, h_pix = self.width(), self.height()
        scale = min(
            (1 - 2 * margin) * w_pix / max(self.beff, self.bw),
            (1 - 2 * margin) * h_pix / self.h
        )

        # --------------------------- geometry in pixels ----------------------
        x_center = int(w_pix / 2)
        y_top = int(margin * h_pix)

        flange_w = int(self.beff * scale)
        flange_h = int(self.hf * scale)
        web_w = int(self.bw * scale)
        web_h = int((self.h - self.hf) * scale)

        flange_x = x_center - flange_w // 2
        flange_y = y_top
        web_x = x_center - web_w // 2
        web_y = flange_y + flange_h

        # ------------------ draw concrete outline ----------------------------
        qp.setPen(QPen(QColor("black"), 2))
        qp.drawRect(flange_x, flange_y, flange_w, flange_h)
        qp.drawRect(web_x, web_y, web_w, web_h)

        # ------------------ draw reinforcement bars --------------------------
        self._draw_bars(qp, scale, web_x, web_y, web_w, web_h)

        # ------------------ draw dimension strings --------------------------
        qp.setPen(QPen(QColor("blue"), 1))
        off = 20
        self._dim_line(qp, flange_x, flange_y - off, flange_w,
                      f"beff = {self.beff:.0f} mm", horizontal=True)
        self._dim_line(qp, web_x, web_y + web_h + off, web_w,
                      f"bw = {self.bw:.0f} mm", horizontal=True)
        self._dim_line(qp, flange_x + flange_w + off, flange_y,
                      web_h + flange_h, f"h = {self.h:.0f} mm", horizontal=False)
        self._dim_line(qp, flange_x - off - 30, flange_y, flange_h,
                      f"hf = {self.hf:.0f} mm", horizontal=False)

    def _draw_bars(self, qp: QPainter, scale: float,
                  web_x: int, web_y: int, web_w: int, web_h: int):
        if self.n_bars == 0:
            return

        # Calculate a1 (distance from edge to first bar center)
        if self.layers == 1:
            a1_mm = self.cnom + self.fi_str + self.fi / 2
        elif self.layers == 2:
            a1_mm = self.cnom + self.fi_str + self.fi + self.smax / 2
        else:  # 3 layers
            a1_mm = self.cnom + self.fi_str + 1.5 * self.fi + self.smax

        a1_px = int(a1_mm * scale)
        fi_px = int(self.fi * scale)

        # Calculate available width between bar centers
        available_width = web_w - 2 * a1_px
        if available_width <= 0:
            return  # Not enough space

        # Calculate bar positions
        if self.n_bars == 1:
            xs = [web_x + web_w // 2]  # Single bar in center
        else:
            spacing = available_width / (self.n_bars - 1)
            xs = [web_x + a1_px + i * spacing for i in range(self.n_bars)]

        # Calculate vertical positions for each layer
        cover_bottom = (self.cnom + self.fi_str + self.fi / 2) * scale
        y_first = int(web_y + web_h - cover_bottom)
        pitch_v = int((self.fi + self.smax) * scale)
        y_rows = [y_first - i * pitch_v for i in range(min(self.layers, 3))]

        # Distribute bars to layers
        bars_per_layer = [self.n_bars // self.layers] * self.layers
        bars_per_layer[0] += self.n_bars - sum(bars_per_layer)  # Add remainder to first layer

        qp.setBrush(QBrush(QColor("#4444ff")))
        qp.setPen(QPen(QColor("black"), 1))

        bar_index = 0
        for layer, n_bars in enumerate(bars_per_layer):
            if n_bars == 0:
                continue
            y = int(y_rows[layer])
            layer_xs = xs[bar_index:bar_index + n_bars]
            bar_index += n_bars
            
            for x in layer_xs:
                x_pos = int(x - fi_px / 2)
                y_pos = int(y - fi_px / 2)
                qp.drawEllipse(x_pos, y_pos, fi_px, fi_px)

    def _dim_line(self, qp: QPainter, x: int, y: int, length_px: int,
                 text: str, *, horizontal: bool) -> None:
        fm = QFontMetrics(qp.font())
        arrow = 5

        if horizontal:
            qp.drawLine(x, y, x + length_px, y)
            qp.drawLine(x, y, x, y - arrow)
            qp.drawLine(x + length_px, y, x + length_px, y - arrow)
            text_w = fm.horizontalAdvance(text)
            qp.drawText(x + (length_px - text_w) // 2, y - arrow - 4, text)
        else:
            qp.drawLine(x, y, x, y + length_px)
            qp.drawLine(x, y, x - arrow, y)
            qp.drawLine(x, y + length_px, x - arrow, y + length_px)
            qp.save()
            qp.translate(x + arrow + 4, y + length_px / 2)
            qp.rotate(-90)
            text_w = fm.horizontalAdvance(text)
            qp.drawText(-text_w // 2, 0, text)
            qp.restore()

# ---------------------------------------------------------------------------
# 3)  SectionTab – NEW “Show section” button & plumbing
#    (only the changed/added lines are shown, context lines are kept
#     so you can find the spots quickly)
# ---------------------------------------------------------------------------

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

        # --- top row: three group boxes side‑by‑side ------------------------
        box_params = self._create_param_box()
        box_fck    = self._create_fck_box()
        box_fi     = self._create_fi_box()

        # size policies – let material boxes grow equally; parameters fixed min width
        box_fck.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        box_fi.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        hbox_top = QHBoxLayout()
        hbox_top.addWidget(box_params)
        hbox_top.addWidget(box_fck)
        hbox_top.addWidget(box_fi)
        hbox_top.setStretchFactor(box_fck, 1)
        hbox_top.setStretchFactor(box_fi,  1)

        grid.addLayout(hbox_top, 0, 0, 1, 2)

        # --- calculate button ----------------------------------------------
        self._add_button(grid, row=1)

        # -------- NEW: extra button just under the calculate one ------------
        self.btn_draw = QPushButton("Pokaż przekrój")          # NEW
        self.btn_draw.setEnabled(False)                        # NEW – enabled once we have a result
        self.btn_draw.clicked.connect(self._open_section_win)  # NEW
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
            "beff [mm]",
            "bw [mm]",
            "h [mm]",
            "hf [mm]",
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
        btn_all.clicked.connect(lambda _=None: [cb.setChecked(True) for cb in self.fck_check.values()])
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
        btn_all.clicked.connect(lambda _=None: [cb.setChecked(True) for cb in self.fi_check.values()])
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
            "Rzędy zbrojenia: ",
            "Typ zbrojenia przekroju: ",
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
    def _open_section_win(self):
        if not hasattr(self, "_last_params") or not hasattr(self, "_best"):
            self.status.setText("❌ Najpierw wykonaj obliczenia.")
            return

        p = self._last_params
        b = self._best
        dlg = SectionView(
            p["beff"], p["bw"], p["h"], p["hf"],     # geometry
            b["fi"], b["num_rods_As1"], b["layers"], # reinforcement
            p["cnom"], p["fi_str"],                  # cover / stirrup
            self
        )
        dlg.exec()   

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
        self.result["Klasa betonu: "].setText(f"C{b['fck']}/{b['fck']+5}")
        self.result["Średnica zbrojenia: "].setText(f"Ø{b['fi']} mm")
        self.result["Rzędy zbrojenia: "].setText(b["type"])
        self.result["Typ zbrojenia przekroju: "].setText(str(b["layers"]))
        self.result["Liczba warstw zbrojenia głównego: "].setText(b["reinforcement_type"])
        self.result["As1_req [mm²]: "].setText(f"{b['As1']:.1f}")
        self.result["As2_req [mm²]: "].setText(f"{b['As2']:.1f}")
        self.result["Liczba prętów rozciąganych: "].setText(str(b["num_rods_As1"]))
        self.result["Liczba prętów ściskanych: "].setText(str(b["num_rods_As2"]))
        self.result["As1_prov [mm²]"].setText(f"{b['actual_As1']:.1f}")
        self.result["As2_prov [mm²]"].setText(f"{b['actual_As2']:.1f}")
        self.result["Rozstaw prętów: "].setText("Yes" if b["fit_check"] else "No")
        self.result["Całkowity koszt [zł/mb]"].setText(f"{b['cost']:.2f}")

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

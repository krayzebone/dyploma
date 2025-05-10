"""
T-section optimiser – one-file version
-------------------------------------

*Click the check-boxes, fill in the geometry/material values,
then hit "Calculate optimal solution".*
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

# --------------------  CALCULATION ENGINE  ----------------------------------
CONCRETE_COST_BY_CLASS = {
    8: 230, 12: 250, 16: 300, 20: 350, 25: 400, 30: 450,
    35: 500, 40: 550, 45: 600, 50: 650, 55: 700, 60: 800
}
POSSIBLE_FCKS = [30]
POSSIBLE_FIS = [10, 12, 14, 16, 20, 25, 28, 32]
STEEL_DENSITY = 7900  # kg/m³
STEEL_PRICE = 5  # PLN/kg


def quadratic_equation(a: float, b: float, c: float, limit: float) -> Optional[float]:
    """Solve quadratic equation and return the smallest root within (0, limit)."""
    if a == 0:
        return None
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    
    sqrt_discriminant = math.sqrt(discriminant)
    denominator = 2*a
    r1 = (-b - sqrt_discriminant) / denominator
    r2 = (-b + sqrt_discriminant) / denominator
    
    valid_roots = [x for x in (r1, r2) if 0 < x < limit]
    return min(valid_roots) if valid_roots else None

def calculate_number_of_rods(As: float, fi: float) -> Tuple[int, float]:
    """Calculate minimum bar count and provided area."""
    if As <= 0:  # no reinforcement required
        return 0, 0.0
    
    Aφ = math.pi * fi**2 / 4
    n = math.ceil(As / Aφ)
    return n, n * Aφ

def distribute_rods_2_layers(bw: float, cnom: float, n: int, fi: float) -> Tuple[bool, List[int]]:
    """Distribute rods between two layers.
    
    Returns:
        Tuple of (fits?, [n_bottom, n_second])
    """
    if n == 0:
        return True, [0, 0]
    
    smax = max(20.0, fi)
    pitch = fi + smax
    available_space = bw - 2*cnom + smax
    n_max = math.floor(available_space / pitch)
    
    if n_max <= 0:
        return False, [0, 0]
    
    n_bottom = min(n, n_max)
    n_second = n - n_bottom
    return n_second <= n_max, [n_bottom, n_second]

def centroid_shift(rods: List[int], fi: float) -> float:
    """Calculate yc above bottom layer for two-layer pack."""
    n1, n2 = rods
    if n1 + n2 == 0:
        return 0.0
    
    Aφ = math.pi * fi**2 / 4
    dy = fi + max(20.0, fi)
    return (n2 * Aφ * dy) / ((n1 + n2) * Aφ)

def calc_cost(
    beff: float,
    bw: float,
    h: float,
    hf: float,
    fck: float,
    As1: float,
    As2: float,
) -> float:
    """Calculate concrete + steel cost in PLN."""
    # Steel cost: mm²→m² * ρ * price
    steel_cost = (As1 + As2) / 1_000_000 * STEEL_DENSITY * STEEL_PRICE
    
    # Concrete area minus steel area
    conc_area = ((beff * hf) + (h - hf) * bw) / 1_000_000 - (As1 + As2) / 1_000_000
    conc_cost = conc_area * CONCRETE_COST_BY_CLASS[int(fck)]
    
    return steel_cost + conc_cost

from typing import Optional, Dict, Any
import math

def evaluate_section(
    MEd: float,
    beff: float,
    bw: float,
    h: float,
    hf: float,
    cnom: float,
    fi_str: float,
    fck: float,
    fi: float
) -> Optional[Dict[str, Any]]:
    """Evaluate a specific combination of fi and fck for a given section."""
    # Calculate effective depth
    a1 = cnom + fi/2 + fi_str
    d0 = h - a1
    
    # Material properties
    fcd = fck / 1.4
    fyd = 500 / 1.15  # fyk is always 500

    # Check section type (T-beam or rectangular)
    MRd_p = beff * hf * fcd * (d0 - 0.5 * hf) / 1e6  # Fixed: used d0 instead of undefined d

    if MEd > MRd_p:
        typ = 'RZT'
    else:
        typ = 'PT'
    
    if typ == 'PT':
        # Solve for neutral axis position in T-beam
        xeff = quadratic_equation(-0.5*beff*fcd, beff*fcd*d0, -MEd*1e6, h)  # Convert MEd to Nmm
        if xeff is None:
            return None
            
        ξ = xeff / d0
        ξ_lim = 0.8*0.0035 / (0.0035 + fyd/200_000)
        
        # Pozornie teowy pojedyńczo zbrojony
        if ξ <= ξ_lim:
            typ = 'pT1Z1r'
            As1 = (xeff * beff * fcd) / fyd
            n1, As1_prov = calculate_number_of_rods(As1, fi)
            fit, r1 = distribute_rods_2_layers(bw, cnom, n1, fi)
            
            if not fit:
                return None
                
            # Adjust for centroid shift
            yc = centroid_shift(r1, fi)
            d = d0 - yc
            a1p = a1 + yc
            
            # Recalculate with adjusted depth
            x2 = quadratic_equation(-0.5*beff*fcd, beff*fcd*d, -MEd*1e6, h)
            if x2 is None:
                return None
                
            if x2/d > ξ_lim:
                typ = 'pT2Z2r'  
                x2 = ξ_lim * d
                As2 = (MEd*1e6 - x2*beff*fcd*(d - 0.5*x2)) / (fyd*(d - a1p))
                As1 = (As2*fyd + x2*beff*fcd) / fyd
                
                n1, As1_prov = calculate_number_of_rods(As1, fi)
                n2, As2_prov = calculate_number_of_rods(As2, fi)
                
                fit1, r1 = distribute_rods_2_layers(bw, cnom, n1, fi)
                fit2, _ = distribute_rods_2_layers(bw, cnom, n2, fi)
                
                if fit1 and fit2:
                    cost = calc_cost(beff, bw, h, hf, fck, As1_prov, As2_prov)
                    return {
                        "MEd": MEd, "beff": beff, "bw": bw, "h": h, "hf": hf,
                        "fi": fi, "fck": fck, "fi_str": fi_str, "cnom": cnom,
                        "rods_layer1": r1[0], "rods_layer2": r1[1],
                        "rods_compression": n2, "cost": cost,
                        "As1": As1, "As2": As2,
                        "actual_As1": As1_prov, "actual_As2": As2_prov,
                        "type": "Doubly reinforced",
                        "layers": 2 if r1[1] > 0 else 1,
                        "reinforcement_type": "Tension and compression" if n2 > 0 else "Tension only",
                        "num_rods_As1": n1,
                        "num_rods_As2": n2,
                        "fit_check": fit1 and fit2
                    }
                else:
                    return None
            else:  # Tension reinforcement only
                cost = calc_cost(beff, bw, h, hf, fck, As1_prov, 0)
                return {
                    "MEd": MEd, "beff": beff, "bw": bw, "h": h, "hf": hf,
                    "fi": fi, "fck": fck, "fi_str": fi_str, "cnom": cnom,
                    "rods_layer1": r1[0], "rods_layer2": r1[1],
                    "rods_compression": 0, "cost": cost,
                    "As1": As1, "As2": 0,
                    "actual_As1": As1_prov, "actual_As2": 0,
                    "type": "Singly reinforced",
                    "layers": 2 if r1[1] > 0 else 1,
                    "reinforcement_type": "Tension only",
                    "num_rods_As1": n1,
                    "num_rods_As2": 0,
                    "fit_check": fit
                }
        # Case 2: Directly doubly reinforced
        else:
            typ = "PT2Z1r"
            xeff = ξ_lim * d0
            As2 = (MEd*1e6 - xeff*beff*fcd*(d0 - 0.5*xeff)) / (fyd*(d0 - a1))
            As1 = (As2*fyd + xeff*beff*fcd) / fyd
            
            n1, As1_prov = calculate_number_of_rods(As1, fi)
            n2, As2_prov = calculate_number_of_rods(As2, fi)
            
            fit1, r1 = distribute_rods_2_layers(bw, cnom, n1, fi)
            fit2, _ = distribute_rods_2_layers(bw, cnom, n2, fi)
            
            if fit1 and fit2:
                cost = calc_cost(beff, bw, h, hf, fck, As1_prov, As2_prov)
                return {
                    "MEd": MEd, "beff": beff, "bw": bw, "h": h, "hf": hf,
                    "fi": fi, "fck": fck, "fi_str": fi_str, "cnom": cnom,
                    "rods_layer1": r1[0], "rods_layer2": r1[1],
                    "rods_compression": n2, "cost": cost,
                    "As1": As1, "As2": As2,
                    "actual_As1": As1_prov, "actual_As2": As2_prov,
                    "type": "Doubly reinforced",
                    "layers": 2 if r1[1] > 0 else 1,
                    "reinforcement_type": "Tension and compression",
                    "num_rods_As1": n1,
                    "num_rods_As2": n2,
                    "fit_check": fit1 and fit2
                }
            else:
                return None
    else:
        typ = "RZT"
        # Solve for neutral axis position
        xeff = quadratic_equation(-0.5*bw*fcd, bw*fcd*d0, hf*(beff-bw)*fcd*(d0-0.5*hf)-MEd*1e6, h)
        if xeff is None:
            return None
            
        ξ = xeff / d0
        ξ_lim = 0.8*0.0035 / (0.0035 + fyd/200_000)
        print(ξ)
        
        
        # Case 1: Single reinforcement first
        if ξ <= ξ_lim:
            typ = "RZT1Z1r"
            As1 = (xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd  # Fixed: removed flange contribution for rectangular section
            n1, As1_prov = calculate_number_of_rods(As1, fi)
            fit, r1 = distribute_rods_2_layers(bw, cnom, n1, fi)
            
            if not fit:
                return None
                
            # Adjust for centroid shift
            yc = centroid_shift(r1, fi)
            d = d0 - yc
            a1p = a1 + yc

            # Recalculate with adjusted depth
            x2 = quadratic_equation(-0.5*bw*fcd, bw*fcd*d, hf*(beff-bw)*fcd*(d-0.5*hf)-MEd*1e6, h)
            if x2 is None:
                return None
                
            if x2/d > ξ_lim:  # Need compression reinforcement
                typ = "RZT2Z1r"
                x2 = ξ_lim * d
                As2 = (MEd*1e6 - x2*bw*fcd*(d - 0.5*x2) - hf * (beff - bw) * fcd * (d - 0.5 * hf)) / (fyd * (d - a1p))
                As1 = (As2*fyd + x2*bw*fcd + hf * (beff - bw) * fcd) / fyd  
                
                n1, As1_prov = calculate_number_of_rods(As1, fi)
                n2, As2_prov = calculate_number_of_rods(As2, fi)
                
                fit1, r1 = distribute_rods_2_layers(bw, cnom, n1, fi)
                fit2, _ = distribute_rods_2_layers(bw, cnom, n2, fi)

                if fit1 and fit2:
                    cost = calc_cost(beff, bw, h, hf, fck, As1_prov, As2_prov)
                    return {
                        "MEd": MEd, "beff": beff, "bw": bw, "h": h, "hf": hf,
                        "fi": fi, "fck": fck, "fi_str": fi_str, "cnom": cnom,
                        "rods_layer1": r1[0], "rods_layer2": r1[1],
                        "rods_compression": n2, "cost": cost,
                        "As1": As1, "As2": As2,
                        "actual_As1": As1_prov, "actual_As2": As2_prov,
                        "type": "Doubly reinforced",
                        "layers": 2 if r1[1] > 0 else 1,
                        "reinforcement_type": "Tension and compression" if n2 > 0 else "Tension only",
                        "num_rods_As1": n1,
                        "num_rods_As2": n2,
                        "fit_check": fit1 and fit2
                    }
                else:
                    return None
            else:  # Tension reinforcement only
                cost = calc_cost(beff, bw, h, hf, fck, As1_prov, 0)
                return {
                    "MEd": MEd, "beff": beff, "bw": bw, "h": h, "hf": hf,
                    "fi": fi, "fck": fck, "fi_str": fi_str, "cnom": cnom,
                    "rods_layer1": r1[0], "rods_layer2": r1[1],
                    "rods_compression": 0, "cost": cost,
                    "As1": As1, "As2": 0,
                    "actual_As1": As1_prov, "actual_As2": 0,
                    "type": "Singly reinforced",
                    "layers": 2 if r1[1] > 0 else 1,
                    "reinforcement_type": "Tension only",
                    "num_rods_As1": n1,
                    "num_rods_As2": 0,
                    "fit_check": fit
                }
        # Case 2: Directly doubly reinforced
        else:
            # First try with original depth
            xeff = ξ_lim * d0
            As2 = (MEd*1e6 - xeff*bw*fcd*(d0 - 0.5*xeff) - hf*(beff - bw)*fcd*(d0 - 0.5*hf)) / (fyd*(d0 - a1))
            As1 = (As2*fyd + xeff*bw*fcd + hf*(beff - bw)*fcd) / fyd
            
            # Calculate reinforcement distribution
            n1, As1_prov = calculate_number_of_rods(As1, fi)
            n2, As2_prov = calculate_number_of_rods(As2, fi)
            fit1, r1 = distribute_rods_2_layers(bw, cnom, n1, fi)
            fit2, _ = distribute_rods_2_layers(bw, cnom, n2, fi)
            
            if not (fit1 and fit2):
                return None
            
            # Adjust for centroid shift
            yc = centroid_shift(r1, fi)
            d = d0 - yc
            a1p = a1 + yc
            
            # Check if compression steel is actually needed
            x2 = ξ_lim * d
            Mrd = x2*bw*fcd*(d - 0.5*x2) + hf*(beff-bw)*fcd*(d-0.5*hf)
            
            if MEd*1e6 > Mrd:  # Doubly reinforced
                As2 = (MEd*1e6 - Mrd) / (fyd*(d - a1p))
                As1 = (As2*fyd + x2*bw*fcd + hf*(beff - bw)*fcd) / fyd
                
                n1, As1_prov = calculate_number_of_rods(As1, fi)
                n2, As2_prov = calculate_number_of_rods(As2, fi)
                fit1, r1 = distribute_rods_2_layers(bw, cnom, n1, fi)
                fit2, _ = distribute_rods_2_layers(bw, cnom, n2, fi)
                
                if not (fit1 and fit2):
                    return None
                    
                cost = calc_cost(beff, bw, h, hf, fck, As1_prov, As2_prov)
                return {
                        "MEd": MEd, "beff": beff, "bw": bw, "h": h, "hf": hf,
                        "fi": fi, "fck": fck, "fi_str": fi_str, "cnom": cnom,
                        "rods_layer1": r1[0], "rods_layer2": r1[1],
                        "rods_compression": n2, "cost": cost,
                        "As1": As1, "As2": As2,
                        "actual_As1": As1_prov, "actual_As2": As2_prov,
                        "type": "Doubly reinforced",
                        "layers": 2 if r1[1] > 0 else 1,
                        "reinforcement_type": "Tension and compression" if n2 > 0 else "Tension only",
                        "num_rods_As1": n1,
                        "num_rods_As2": n2,
                        "fit_check": fit1 and fit2
                    }
            
            else:  # Singly reinforced
                cost = calc_cost(beff, bw, h, hf, fck, As1_prov, 0)
                return {
                        "MEd": MEd, "beff": beff, "bw": bw, "h": h, "hf": hf,
                        "fi": fi, "fck": fck, "fi_str": fi_str, "cnom": cnom,
                        "rods_layer1": r1[0], "rods_layer2": r1[1],
                        "rods_compression": n2, "cost": cost,
                        "As1": As1, "As2": As2,
                        "actual_As1": As1_prov, "actual_As2": As2_prov,
                        "type": "Doubly reinforced",
                        "layers": 2 if r1[1] > 0 else 1,
                        "reinforcement_type": "Tension and compression" if n2 > 0 else "Tension only",
                        "num_rods_As1": n1,
                        "num_rods_As2": n2,
                        "fit_check": fit1 and fit2
                    }
    
    

def find_optimal_solution(params: Dict[str, float], fi_list: List[int], fck_list: List[int]) -> Optional[Dict[str, Any]]:
    """Find the lowest cost solution across all combinations."""
    best_solution = None
    min_cost = float('inf')
    
    for fck in fck_list:
        for fi in fi_list:
            solution = evaluate_section(
                MEd=params["MEd"],
                beff=params["beff"],
                bw=params["bw"],
                h=params["h"],
                hf=params["hf"],
                cnom=params["cnom"],
                fi_str=params["fi_str"],
                fck=fck,
                fi=fi
            )
            
            if solution and solution["cost"] < min_cost:
                min_cost = solution["cost"]
                best_solution = solution
    
    return best_solution


@dataclass
class Inputs:
    MEd: float
    beff: float
    bw: float
    h: float
    hf: float
    cnom: float
    fi_str: float

class CalculationDataT:
    def __init__(self):
        self.MEd = None
        self.beff = None
        self.bw = None
        self.h = None
        self.hf = None
        self.fi = None
        self.fck = None
        self.As1 = None
        self.As2 = None
        self.num_rods_As1 = None  # Number of tension rods
        self.num_rods_As2 = None  # Number of compression rods
        self.act1 = None
        self.act2 = None

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


class TSectionTabSGN(QWidget):
    """One tab that collects geometry + material data and shows the best result."""

    def __init__(self, parent: QWidget | None = None, data_storeT: CalculationDataT = None) -> None:
        super().__init__(parent)
        self.data_storeT = data_storeT if data_storeT else CalculationDataT()
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

        # --- results section -----------------------------------------------
        self._add_results_group(grid, row=2)

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
        
        # Set some default values for testing
        self.inputs["MEd [kNm]"].setText("550")
        self.inputs["beff [mm]"].setText("900")
        self.inputs["bw [mm]"].setText("800")
        self.inputs["h [mm]"].setText("300")
        self.inputs["hf [mm]"].setText("100")
        self.inputs["c_nom [mm]"].setText("30")
        self.inputs["fi_str [mm]"].setText("8")
        
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
        
        # Check C30/37 by default
        self.fck_check["C30/37"].setChecked(True)
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
        
        # Check Ø20 by default
        self.fi_check["Ø20"].setChecked(True)
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
            "Zbrojenie rozciągane w pierwszej warstwie: ",
            "Zbrojenie rozciągane w drugiej warstwie: ",
            "Zbrojenie ściskane: ",
            "As1_req [mm²]: ",
            "As2_req [mm²]: ",
            "Liczba prętów rozciąganych: ",
            "Liczba prętów ściskanych: ",
            "As1_prov [mm²]: ",
            "As2_prov [mm²]: ",
            "Rozstaw prętów: ",
            "Całkowity koszt [zł/mb]: ",
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

    def _float(self, le: QLineEdit, name: str) -> float | None:
        try:
            val = float(le.text())
            le.setStyleSheet("")
            return val
        except ValueError:
            le.setStyleSheet("background:#ffdddd")
            self.status.setText(f"❌ '{name}' is not a number.")
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
            best = find_optimal_solution(params, fi_list, fck_list)
        except Exception as exc:  # pragma: no cover
            self.status.setText(f"❌ optimiser error: {exc}")
            return

        if not best:
            self.status.setText("❌ Brak rozwiązań.")
            return
        
        self._show(best)
        self.status.setText("✓ Optimalny przekrój znaleziony.")

        # Store the values in the data store
        self.data_storeT.MEd = params["MEd"]
        self.data_storeT.beff = params["beff"]
        self.data_storeT.bw = params["bw"]
        self.data_storeT.h = params["h"]
        self.data_storeT.hf = params["hf"]
        self.data_storeT.fi = best["fi"]
        self.data_storeT.fck = best["fck"]
        self.data_storeT.As1 = best["As1"]
        self.data_storeT.As2 = best["As2"]
        self.data_storeT.num_rods_As1 = best["num_rods_As1"]  # Store number of tension rods
        self.data_storeT.num_rods_As2 = best["num_rods_As2"]  # Store number of compression rods
        self.data_storeT.act1 = best["num_rods_As1"] * math.pi * best["fi"]**2 /4
        self.data_storeT.act2 = best["num_rods_As2"] * math.pi * best["fi"]**2 /4

    def _show(self, b: dict) -> None:
        self.result["Klasa betonu: "].setText(f"C{b['fck']}/{b['fck']+5}")
        self.result["Średnica zbrojenia: "].setText(f"Ø{b['fi']} mm")
        self.result["Zbrojenie rozciągane w pierwszej warstwie: "].setText(str(b["rods_layer1"]))
        self.result["Zbrojenie rozciągane w drugiej warstwie: "].setText(str(b["rods_layer2"]))
        self.result["Zbrojenie ściskane: "].setText(str(b["rods_compression"]))
        self.result["As1_req [mm²]: "].setText(f"{b['As1']:.1f}")
        self.result["As2_req [mm²]: "].setText(f"{b['As2']:.1f}" if b["As2"] > 0 else "0.0")
        self.result["Liczba prętów rozciąganych: "].setText(str(b["num_rods_As1"]))
        self.result["Liczba prętów ściskanych: "].setText(str(b["num_rods_As2"]))
        self.result["As1_prov [mm²]: "].setText(f"{b['actual_As1']:.1f}")
        self.result["As2_prov [mm²]: "].setText(f"{b['actual_As2']:.1f}" if b["actual_As2"] > 0 else "0.0")
        self.result["Rozstaw prętów: "].setText("Tak" if b["fit_check"] else "Nie")
        self.result["Całkowity koszt [zł/mb]: "].setText(f"{b['cost']:.2f}")

# -----------------------  run it  -------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    win = TSectionTabSGN()
    win.setWindowTitle("T-Section optimiser")
    win.resize(600, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
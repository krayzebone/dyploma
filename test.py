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

    # check 
    
    # Solve for neutral axis position
    xeff = quadratic_equation(-0.5*beff*fcd, beff*fcd*d0, -MEd*1e6, h)  # Convert MEd to Nmm
    if xeff is None:
        return None
        
    ξ = xeff / d0
    ξ_lim = 0.8*0.0035 / (0.0035 + fyd/200_000)
    
    # Case 1: Single reinforcement first
    if ξ <= ξ_lim:
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
            
        if x2/d > ξ_lim:  # Need compression reinforcement
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
    
    return None

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

result = evaluate_section(2000, 1000, 800, 600, 150, 30, 8, 35, 20)

print(result)
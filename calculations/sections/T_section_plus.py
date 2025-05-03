import math
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.text as mtext

def quadratic_equation(a: float, b: float, c: float, limit: float) -> float:
    """Solve quadratic equation and return solution within (0, limit) if exists."""
    if a == 0:
        return None
        
    delta = b**2 - 4*a*c

    if delta < 0:
        return None
    elif delta == 0:
        x = -b / (2*a)
        return x if 0 < x < limit else None
    else:
        sqrt_delta = math.sqrt(delta)
        x1 = (-b - sqrt_delta) / (2*a)
        x2 = (-b + sqrt_delta) / (2*a)
        valid_solutions = [x for x in (x1, x2) if 0 < x < limit]
        return min(valid_solutions) if valid_solutions else None

def calc_cost(beff: float, bw: float, h: float, hf: float, fck: float, As1: float, As2: float) -> float:
    concrete_cost_by_class = {
        8: 230, 12: 250, 16: 300, 20: 350, 25: 400, 30: 450, 35: 500, 40: 550, 
        45: 600, 50: 650, 55: 700, 60: 800
    }
    
    steel_cost = (As1 + As2) / 1_000_000 * 7900 * 5  # mm²->m² * density * cost
    concrete_area = ((beff * hf) + (h - hf) * bw) / 1_000_000 - (As1 + As2)/1_000_000
    concrete_cost = concrete_area * concrete_cost_by_class[int(fck)]
    
    return steel_cost + concrete_cost

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

def calculate_number_of_rods(As: float, fi: float) -> tuple:
    if As <= 0 or math.isinf(As):
        return 0, 0
    rod_area = math.pi * (fi ** 2) / 4
    number_of_rods = math.ceil(As / rod_area)
    return number_of_rods, number_of_rods * rod_area

def check_rods_fit(bw: float, cnom: float, num_rods: int, fi: float, smax: float, layers: int = 1) -> bool:
    if num_rods == 0:
        return True
    required_width = 2 * cnom + num_rods * fi + smax * (num_rods - 1)
    available_width = layers * bw
    return required_width <= available_width

def find_optimal_scenario(inputs, possible_fi, possible_fck):
    MEd = inputs['MEd']
    beff = inputs['beff']
    bw = inputs['bw']
    h = inputs['h']
    hf = inputs['hf']
    cnom = inputs['cnom']
    fi_str = inputs['fi_str']
    fyk = 500.0

    best_section = {
        'cost': float('inf'),
        'fck': None,
        'fi': None,
        'type': None,
        'layers': None,
        'reinforcement_type': None,
        'As1': None,
        'As2': None,
        'num_rods_As1': None,
        'num_rods_As2': None,
        'actual_As1': None,
        'actual_As2': None,
        'fit_check': None,
        'parameters': {
            'MEd': MEd,
            'beff': beff,
            'bw': bw,
            'h': h,
            'hf': hf,
            'cnom': cnom,
            'fi_str': fi_str
        }
    }

    for fck in possible_fck:
        fcd = fck / 1.4
        fyd = fyk / 1.15
        for fi in possible_fi:
            smax = max(20, fi)
            a1_vals = {
                1: cnom + fi_str + fi/2,
                2: cnom + fi_str + fi + smax/2,
                3: cnom + fi_str + 1.5*fi + smax
            }
            
            for layers, a1 in a1_vals.items():
                d = h - a1
                MRd = (beff * hf * fcd * (d - 0.5*hf)) / 1e6
                scenario_type = 'PT' if MEd < MRd else 'RZT'
                func_name = f'calc_{scenario_type}_{layers}r_plus'
                
                try:
                    # Assuming these functions are defined elsewhere
                    func = globals().get(func_name)
                    if not func:
                        raise ValueError(f"Function {func_name} not found")
                    
                    As1, As2, cost, reinforcement_type = func(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck)
                    num_rods_As1, actual_As1 = calculate_number_of_rods(As1, fi)
                    num_rods_As2, actual_As2 = calculate_number_of_rods(As2, fi)
                    rods_fit = (check_rods_fit(bw, cnom, num_rods_As1, fi, smax, layers) and 
                               check_rods_fit(bw, cnom, num_rods_As2, fi, smax, layers))
                    
                    if rods_fit and cost < best_section['cost']:
                        best_section.update({
                            'cost': cost,
                            'fck': fck,
                            'fi': fi,
                            'type': scenario_type,
                            'layers': layers,
                            'reinforcement_type': reinforcement_type,
                            'As1': As1,
                            'As2': As2,
                            'num_rods_As1': num_rods_As1,
                            'num_rods_As2': num_rods_As2,
                            'actual_As1': actual_As1,
                            'actual_As2': actual_As2,
                            'fit_check': rods_fit
                        })
                        
                except Exception as e:
                    print(f"Error processing fck={fck}, fi={fi}, layers={layers}: {str(e)}")
                    continue

    # Clean up None values for better JSON serialization if needed
    best_section = {k: v for k, v in best_section.items() if v is not None}
    return best_section

inputs = {
    'MEd': 500,
    'beff': 1000,
    'bw': 300,
    'h': 500,
    'hf': 120,
    'cnom': 30,
    'fi_str': 8
}
possible_fi = [8, 10, 12, 16, 20, 25, 28, 32]
possible_fck = [20, 25, 30, 35, 40, 45, 50, 55]
optimal_section = find_optimal_scenario(inputs, possible_fi, possible_fck)
print(optimal_section)





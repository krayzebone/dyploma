import math
import tqdm
import pandas as pd
import numpy as np

def quadratic_equation(
    a: float, 
    b: float, 
    c: float, 
    limit: float,
) -> float:
    """
    Rozwiązanie równania kwadratowego a*x^2 + b*x + c = 0
    Zwraca jedno z rozwiązań (x1 lub x2), o ile leży w przedziale (0, limit).
    Jeśli żadne rozwiązanie nie mieści się w tym przedziale, zwraca None.
    """
    delta = b**2 - 4*a*c

    if delta < 0:
        return None
    elif delta == 0:
        # jedno rozwiązanie
        x = -b / (2*a)
        if 0 < x < limit:
            return x
        return None
    else:
        # Dwa rozwiązania
        sqrt_delta = math.sqrt(delta)
        x1 = (-b - sqrt_delta) / (2*a)
        x2 = (-b + sqrt_delta) / (2*a)

        # Przekrój może mieć 2 rozwiązania które należą do (0,h),wybieramy mniejszą z nich
        valid_solutions = []
        if 0 < x1 < limit:
            valid_solutions.append(x1)
        if 0 < x2 < limit:
            valid_solutions.append(x2)

        # jeśli jedna, tę jedną, w przeciwnym razie None
        if not valid_solutions:
            return None
        
        # Jeśli obie leżą w przedziale, zwracamy mniejszą,
        if valid_solutions:
            return min(valid_solutions)
        
def check_section_type_2r(MEd: float, beff: float, h: float, hf: float, figl: float, fck: float):

    cnom = 48
    a1 = cnom + figl * 3 / 2
    d = h - a1
    fcd = fck / 1.4

    # Flange load capacity
    MRd = beff * hf * fcd * (d - 0.5 * hf) / 1e6

    if MRd < MEd:
        section_type = 'RZT'
    else:
        section_type = 'PT'
    
    return section_type

def PT_Section_2r(MEd: float, beff: float, h: float, figl: float, fck: float):

    fyk = 500
    fyd = fyk / 1.15
    fcd = fck / 1.4
    Es = 200_000.0

    cnom = 48
    a1 = cnom + figl * 3 / 2
    a2 = cnom + figl / 2
    d = h - a1

    a = (-0.5) * beff * fcd
    b = beff * fcd * d
    c = -MEd * 1e6
    xeff = quadratic_equation(a, b, c, h)

    if xeff == None:
        return None
    
    ksieff = xeff / d
    ksieff_lim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff < ksieff_lim:
        section_type = 'PT1'
    else:
        section_type = 'PT2'

    if section_type == 'PT1':
        As1 = xeff * beff * fcd / fyd
        As2 = 0
        return section_type, As1, As2

    else:
        xeff = ksieff_lim * d
        As2 = (MEd * 1e6 - xeff * beff * fcd * (d - 0.5 * xeff)) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * beff * fcd) / fyd
        return section_type, As1, As2

def RZT_Section_2r(MEd: float, beff: float, bw: float, h: float, hf: float, figl: float, fck: float):

    fyk = 500
    fyd = fyk / 1.15
    fcd = fck / 1.4
    Es = 200_000.0
    cnom = 48
    
    a1 = cnom + figl * 3 / 2
    a2 = cnom + figl / 2
    d = h - a1

    a = (-0.5 * bw * fcd)
    b = bw * fcd * d
    c = hf * (beff-bw) * fcd * (d - 0.5 * hf) - MEd * 1e6
    xeff = quadratic_equation(a, b, c, h)

    if xeff == None:
        return None

    ksieff = xeff / d
    ksieff_lim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff < ksieff_lim:
        section_type = 'RZT1'
    else:
        section_type = 'RZT2'
    
    if section_type == 'RZT1':
        As1 = (xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        As2 = 0
        return section_type, As1, As2
    
    else:
        xeff = ksieff_lim * d
        As2 = (-xeff * bw * fcd * (d - 0.5 * xeff) - hf * (beff - bw) * fcd * (d - 0.5 * hf) + MEd * 1e6) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        return section_type, As1, As2






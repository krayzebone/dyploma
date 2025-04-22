import math
from typing import Optional, Tuple
import numpy as np

def rownanie_kwadratowe(
    a: float, 
    b: float, 
    c: float, 
    limit: float,
) -> Optional[float]:
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
        
def sprawdzenie_ksi_eff(x_eff: float, d: float, ksi_eff_lim: float) -> bool:
    """
    Sprawdza warunek x_eff / d <= ksi_eff_lim.
    """
    ksi_eff = x_eff / d
    return ksi_eff <= ksi_eff_lim

def rect_section_design(
    b: float,
    h: float,
    f_ck: float,
    f_yk: float,
    c_nom: float,
    fi_gl: float,
    fi_str: float,
    M_Ed: float,
) -> list:
    """
    Funkcja wymiaruje przekrój T belki żelbetowej
    zgodnie z podstawowymi założeniami normowymi (EC2).
    
    Parametry:
    -----------
    b      : szerokość przekroju (mm)
    h      : wysokosć przekroju (mm)
    f_ck   : charakterystyczna wytrzymałość betonu na ściskanie (MPa)
    f_yk   : charakterystyczna granica plastyczności stali (MPa)
    c_nom  : otulina nominalna (mm)
    fi_gl  : średnica głównego zbrojenia (mm)
    fi_str : średnica strzemion (mm)
    M_Ed   : moment obliczeniowy (Nmm) - należy zwrócić uwagę na jednostki!
    """

    # Parametry materiałowe
    f_cd = f_ck / 1.4     # wytrzymałość obliczeniowa betonu na ściskanie [MPa]
    f_yd = f_yk / 1.15    # wytrzymałość obliczeniowa stali na rozciąganie [MPa]
    E_s = 200000.0        # moduł Younga stali [MPa]

    # Wyznaczenie współczynnika względnej wysokości strefy ściskanej
    ksi_eff_lim = 0.8 * 0.0035 / (0.0035 + f_yk / E_s)

    # Podstawowe wymiary przekroju
    a_1 = c_nom + fi_gl / 2 + fi_str  
    d = h - a_1

    x_eff = rownanie_kwadratowe(
            a = -0.5 * b * f_cd,
            b = b * f_cd * d,
            c = -M_Ed * 1_000_000,
            limit=h,
        )

    if x_eff is not None and sprawdzenie_ksi_eff(x_eff, d, ksi_eff_lim):
            A_s1 = x_eff * b * f_cd / f_yd

            return [
                A_s1,
                0,
            ]
        
        # Delta jest mniejsza od 0
    elif x_eff is None:
        return None
        
        # Przekrój pozornie teowy, podwójnie zbrojony
    else:
        x_eff = ksi_eff_lim * d
        A_s2 = (
            (- x_eff * b * f_cd * (d - 0.5 * x_eff) + M_Ed * 1e6)
            / (f_yd * (d - a_1))
        )
        A_s1 = (A_s2 * f_yd + x_eff * b * f_cd) / f_yd

        return [
            A_s1,
            A_s2,
        ]

def rect_section_strain(
    b: float,
    h: float,
    f_ck: float,
    f_yk: float,
    c_nom: float,
    fi_gl: float,
    fi_str: float,
    M_Ed: float,
    A_s1: float,
    A_s2: float,  
)-> list:
    """
    Funkcja wyznacza wytężenie przekroju M_Rd / M_Ed
    zgodnie z podstawowymi założeniami normowymi (EC2).
    
    Parametry:
    -----------
    b      : szerokość przekroju (mm)
    h      : wysokosć przekroju (mm)
    f_ck   : charakterystyczna wytrzymałość betonu na ściskanie (MPa)
    f_yk   : charakterystyczna granica plastyczności stali (MPa)
    c_nom  : otulina nominalna (mm)
    fi_gl  : średnica głównego zbrojenia (mm)
    fi_str : średnica strzemion (mm)
    M_Ed   : moment obliczeniowy (Nmm)
    """
    # Parametry materiałowe
    f_cd = f_ck / 1.4     # wytrzymałość obliczeniowa betonu na ściskanie [MPa]
    f_yd = f_yk / 1.15    # wytrzymałość obliczeniowa stali na rozciąganie [MPa]
    E_s = 200000.0        # moduł Younga stali [MPa]

    # Wyznaczenie współczynnika względnej wysokości strefy ściskanej
    ksi_eff_lim = 0.8 * 0.0035 / (0.0035 + f_yk / E_s)

    # Podstawowe wymiary przekroju
    a_1 = c_nom + fi_gl / 2 + fi_str  
    d = h - a_1

    x_eff = (A_s1 * f_yd - A_s2 * f_yd) / (b * f_cd)
    M_Rd = x_eff * b * f_cd * (d - 0.5 * x_eff) + A_s2 * f_yd * (d - a_1)
    strain = M_Ed / M_Rd

    return strain
    
def calc_reinforcement_bars(A_s1, A_s2, b, c_nom):

    possible_fi_gl = sorted([6, 8, 10, 12, 14, 16, 20, 22, 24, 25, 26, 28, 30, 32])

    for fi in possible_fi_gl:
        pole_preta = math.pi * (fi**2) / 4
        n1 = max(2, math.ceil(A_s1 / pole_preta))
        n2 = max(2, math.ceil(max(A_s2, 0) / pole_preta))

        s_min = max(fi, 20)
        total_width_n1 = 2 * c_nom + (n1 * fi) + ((n1 - 1) * s_min)
        total_width_n2 = 2 * c_nom + (n2 * fi) + ((n2 - 1) * s_min) if n2 > 0 else 0

        if total_width_n1 <= b and total_width_n2 <= b:
            return n1, n2, fi  # Return the first valid (smallest possible) fi
        
    return None  # No valid solution

def calc_min_and_max_reinfocement(b, h, f_ck, f_yk, c_nom, fi_gl, fi_str):

    a_1 = c_nom + fi_str + fi_gl/2
    d = h - a_1

    if f_ck <= 50:
        f_ctm = 0.3 * (f_ck)**(2/3)
        A_s_min = max(0.26*(f_ctm / f_yk) * b * d, 0.0013 * b * d)
        A_s_max = 0.04 * b * h

        return A_s_min, A_s_max
    
    else:
        f_cm = f_ck + 8
        f_ctm = 2.12 * np.log(1 + (f_cm / 10))
        A_s_min = max(0.26*(f_ctm / f_yk) * b * d, 0.0013 * b * d)
        A_s_max = 0.04 * b * h

        return A_s_min, A_s_max

def find_minimum_bar_diameter(A_s_req, b, c_nom, s_min):
    """
    Finds the smallest bar diameter from a predefined list of diameters
    that satisfies both the required steel area A_s_req
    and the geometric spacing constraint.

    Parameters
    ----------
    A_s_req : float
        Required steel area (mm^2).
    b : float
        Total width available for the rebar (mm).
    c_nom : float
        Nominal cover (mm).
    s_min : float
        Minimum clear spacing between bars (mm).

    Returns
    -------
    (diameter, n_bars) : tuple
        The selected bar diameter (mm) and the number of bars required.
        Returns (None, None) if no diameter can satisfy the requirements.
    """
    possible_diams = [6, 8, 10, 12, 14, 16, 20, 22, 24, 25, 26, 28, 30, 32]

    for fi in possible_diams:
        # Area of a single bar
        A_bar = math.pi * (fi ** 2) / 4.0

        # Number of bars needed
        n_bars = math.ceil(A_s_req / A_bar)

        # Check the spacing constraint
        total_space_needed = 2 * c_nom + n_bars * fi + (n_bars - 1) * s_min

        if total_space_needed <= b:
            # As soon as we find the first diameter that meets the requirement, return it
            return fi, n_bars

    # If no diameter satisfies the requirement, return None
    return None, None

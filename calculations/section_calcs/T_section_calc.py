import math
from typing import Optional, Tuple

def rownanie_kwadratowe(
    a: float, 
    b: float, 
    c: float, 
    limit: float,
    typ_przekroju: str
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

        # Jeśli obie leżą w przedziale, zwracamy mniejszą,
        # jeśli jedna, tę jedną, w przeciwnym razie None
        if not valid_solutions:
            return None
        
        elif typ_przekroju == "pozornie_teowy":
            return min(valid_solutions)

        elif typ_przekroju == "rzeczywiscie_teowy":
            return max(valid_solutions)
        
        else:
            return None


def sprawdzenie_ksi_eff(x_eff: float, d: float, ksi_eff_lim: float) -> bool:
    """
    Sprawdza warunek x_eff / d <= ksi_eff_lim.
    """
    ksi_eff = x_eff / d
    return ksi_eff <= ksi_eff_lim


def wymiarowanie_przekroju_teowego(
    b_eff: float,
    b_w: float,
    h: float,
    h_f: float,
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
    b_eff  : efektywna szerokość półki (mm)
    b_w    : szerokość środnika (mm)
    h      : całkowita wysokość przekroju (mm)
    h_f    : grubość półki (mm)
    f_ck   : charakterystyczna wytrzymałość betonu na ściskanie (MPa)
    f_yk   : charakterystyczna granica plastyczności stali (MPa)
    c_nom  : otulina nominalna (mm)
    fi_gl  : średnica głównego zbrojenia (mm)
    fi_str : średnica strzemion (mm)
    M_Ed   : moment obliczeniowy (Nmm) - należy zwrócić uwagę na jednostki!
    """
    # Sprawdzenie warunków geometrycznych.
    if h < h_f or b_eff < b_w:
        return None

    # Parametry materiałowe
    f_cd = f_ck / 1.4     # wytrzymałość obliczeniowa betonu na ściskanie [MPa]
    f_yd = f_yk / 1.15    # wytrzymałość obliczeniowa stali na rozciąganie [MPa]
    E_s = 200000.0        # moduł Younga stali [MPa]

    # Wyznaczenie współczynnika względnej wysokości strefy ściskanej
    ksi_eff_lim = 0.8 * 0.0035 / (0.0035 + f_yk / E_s)

    # Podstawowe wymiary przekroju
    a_1 = c_nom + fi_gl / 2 + fi_str  
    d = h - a_1                       

    # Nośność samej półki
    M_Rd = b_eff * h_f * f_cd * (d - 0.5 * h_f) / 1_000_000

    # Określenie czy przekrój jest pozornie teowy czy rzeczywiście teowy
    if M_Rd >= M_Ed:
        typ_przekroju = "pozornie_teowy"
    else:
        typ_przekroju = "rzeczywiscie_teowy"


    ########################################################
    # Przekrój pozornie teowy
    ########################################################

    if typ_przekroju == "pozornie_teowy":
        x_eff = rownanie_kwadratowe(
            a=-0.5 * b_eff * f_cd,
            b=b_eff * f_cd * d,
            c=-M_Ed * 1_000_000,
            limit=h,
            typ_przekroju=typ_przekroju
        )

        # Przekrój pozornie teowy, pojedynczo zbrojony
        if x_eff is not None and sprawdzenie_ksi_eff(x_eff, d, ksi_eff_lim):
            A_s1 = x_eff * b_eff * f_cd / f_yd

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
                (M_Ed * 1_000_000 - x_eff * b_eff * f_cd * (d - 0.5 * x_eff))
                / (f_yd * (d - a_1))
            )
            A_s1 = (A_s2 * f_yd + x_eff * b_eff * f_cd) / f_yd

            return [
                A_s1,
                A_s2,
            ]
        
    ########################################################
    # Przekrój rzeczywiście teowy
    ########################################################

    else:

        x_eff = rownanie_kwadratowe(
            a=-0.5 * b_w * f_cd,
            b=b_w * f_cd * d,
            c=-(M_Ed * 1_000_000 - h_f * (b_eff - b_w) * f_cd * (d - 0.5 * h_f)),
            limit=h,
            typ_przekroju=typ_przekroju
        )

        # Przekrój pojedynczo zbrojony, rzeczywiście teowy
        if x_eff is not None and sprawdzenie_ksi_eff(x_eff, d, ksi_eff_lim):

            A_s1 = (x_eff * b_w * f_cd + h_f * (b_eff - b_w) * f_cd) / f_yd

            return [
                A_s1,
                0,
            ]
        
        # Delta jest mniejsza od 0
        elif x_eff is None:
            return None

        # Przekrój podwójnie zbrojony, rzeczywiście teowy
        else:
            x_eff = ksi_eff_lim * d
            A_s2 = (
                -x_eff * b_w * f_cd * (d - 0.5 * x_eff)
                - h_f * (b_eff - b_w) * f_cd * (d - 0.5 * h_f)
                + M_Ed * 1_000_000
            ) / (f_yd * (d - a_1))

            A_s1 = (A_s2 * f_yd + x_eff * b_w * f_cd + h_f * (b_eff - b_w) * f_cd) / f_yd

            return [
                A_s1,
                A_s2,
            ]


# wyznaczenie liczby prętów zbrojeniowych na podstawie pola zbrojenia
def wyznaczenie_zbrojenia_rzeczywistego(A_s1, A_s2, fi_gl) -> Optional[Tuple[int, int]]:

    if A_s1 is None or A_s2 is None:
        return None
    
    pole_preta = math.pi * (fi_gl**2) / 4

    n1 = math.ceil(A_s1/pole_preta)

    n2 = math.ceil(max(A_s2,0)/pole_preta)

    return n1, n2

def warunek_rozmieszczenia_zbrojenia(A_s1, A_s2, b_w, c_nom):

    possible_fi_gl = sorted([6, 8, 10, 12, 14, 16, 20, 22, 24, 25, 26, 28, 30, 32])

    for fi in possible_fi_gl:
        pole_preta = math.pi * (fi**2) / 4
        n1 = max(2, math.ceil(A_s1 / pole_preta))
        n2 = max(2, math.ceil(max(A_s2, 0) / pole_preta))

        s_min = max(fi, 20)
        total_width_n1 = 2 * c_nom + (n1 * fi) + ((n1 - 1) * s_min)
        total_width_n2 = 2 * c_nom + (n2 * fi) + ((n2 - 1) * s_min) if n2 > 0 else 0

        if total_width_n1 <= b_w and total_width_n2 <= b_w:
            return n1, n2, fi  # Return the first valid (smallest possible) fi
        
    return None  # No valid solution

#wynik = wymiarowanie_przekroju_teowego(b_eff=100, b_w=100, h=423, h_f=405, f_ck=30, f_yk=500, c_nom=30, fi_gl=20, fi_str=8, M_Ed=150)

#wynik = wymiarowanie_przekroju_teowego(b_eff=2800, b_w=1250, h=120, h_f=110, f_ck=45, f_yk=500, c_nom=30, fi_gl=28, fi_str=8, M_Ed=100)

#print(wynik)

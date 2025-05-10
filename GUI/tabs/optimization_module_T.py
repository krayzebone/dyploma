# calculations.py
import math
from dataclasses import dataclass


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
                # decide "pure-T" or "Ribbed T"
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
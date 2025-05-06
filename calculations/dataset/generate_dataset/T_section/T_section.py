import math
import tqdm
import pandas as pd
import numpy as np

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


def calculate_number_of_rods(As: float, fi: float) -> tuple[int, float]:
    """Return number of bars and the provided reinforcement area."""
    if As <= 0 or math.isinf(As):
        return 0, 0.0
    area_bar = math.pi * fi**2 / 4
    n = math.ceil(As / area_bar)
    return n, n * area_bar


def check_rods_fit(bw: float, cnom: float, num_rods: int, fi: float, smax: float = 25, layers: int = 1) -> bool:
    """Check clear spacing rules in *one* reinforcement layer."""
    if num_rods == 0:
        return True
    required = 2 * cnom + num_rods * fi + smax * (num_rods - 1)
    return required <= layers * bw


def smallest_layer_count(
    bw: float,
    cnom: float,
    num_rods: int,
    fi: float,
    smax: float = 25,
    max_layers: int = 3,
) -> int:
    """
    Return the *minimum* number of reinforcement layers (≥ 1) that allows
    `num_rods` bars Ø fi to fit in a web of width `bw`, honouring the clear
    spacing rule in `check_rods_fit`.  Raises `ValueError` if `max_layers`
    is exceeded (practically, > 6 layers is already outside common detailing
    practice, but raise the limit if you need more).
    """
    for layers in range(1, max_layers + 1):
        if check_rods_fit(bw, cnom, num_rods, fi, smax=smax, layers=layers):
            return layers
    raise ValueError(
        f"{num_rods} ø{fi} bars will not fit in {max_layers} layers within bw = {bw} mm"
    )


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



num_iterations = 100000
data_list = []

for _ in tqdm.tqdm(range(num_iterations), desc="Running simulations"):

    #####################################################################
    #   1. Parametry wejściowe
    #####################################################################

    # External moment
    MEd = np.random.uniform(low=10, high=10000) * 1e6

    # Geometry of section
    beff = np.random.uniform(low=100, high=2000)
    bw = np.random.uniform(low=100, high=2000)
    h = np.random.uniform(low=100, high=1500)
    hf = np.random.uniform(low=100, high=1500)
    cnom = np.random.uniform(low=20, high=60)

    if hf > h:
        continue

    if bw > beff:
        continue
    
    # Concrete choice
    fck = np.random.choice([16, 20, 25, 30, 35, 40, 45, 50])
    fyk = 500
    
    # Choose bar diameter from discrete set
    fi = np.random.choice([8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32])
    fi_str = np.random.choice([8, 10, 12, 14, 16, 18, 20])

    
    a1 = cnom + fi / 2 + fi_str
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    fcd = fck / 1.4
    fyd = fyk / 1.15
    
    xeff = quadratic_equation(-0.5 * beff * fcd, beff * fcd * d, -MEd, h)

    if xeff is None:
        continue

    ksieff = xeff / d
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / 200_000)

    if ksieff <= ksiefflim:
        continue

    if ksieff > ksiefflim:
        As1 = xeff * beff * fcd / fyd
        As2 = 0
    else:
        xeff = ksiefflim * d
        As2 = (MEd - xeff * beff * fcd * (d - 0.5 * xeff)) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * beff * fcd) / fyd

    n_bars, As1_provided = calculate_number_of_rods(As1, fi)
    try:
        layers = smallest_layer_count(bw, cnom, n_bars, fi)
    except ValueError:
        continue        # skip cases that simply cannot be detailed

    cost = calc_cost(beff, bw, h, hf, fck, As1, As2)
    
    # Store final data
    data_entry = {
        'MEd': MEd / 1e6,
        'beff': beff,
        'bw': bw,
        'h': h,
        'hf': hf,
        'fi': fi,
        'fck': fck,
        'cnom': cnom,
        'fistr': fi_str,
        'a1': a1,
        'd': d,
        'cost': cost,
        'As1': As1,
        'As2': As2,
        "n_bars": n_bars,
        "layers": layers,
    }

    data_list.append(data_entry)

# Save results
if data_list:
    df = pd.DataFrame(data_list)
    df.to_csv("Tsection222.csv", index=False)
    print(f"\nSaved {len(data_list)} valid results to 'Tsection.parquet'")
else:
    print("\nNo valid cases found.")

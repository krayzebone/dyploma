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

def calculate_section_cost(beff: float, bw: float, h: float, hf: float, f_ck: float, As1: float, As2: float) -> float:
    concrete_cost_by_class = {
        8: 230, 12: 250, 16: 300, 20: 350, 25: 400, 30: 450, 35: 500, 40: 550, 45: 600, 50: 650, 55: 700, 60: 800
    }
    
    steel_cost_by_weight = 5  # zł/kg
    steel_density = 7900      # kg/m3
    
    steel_area = (As1 + As2) / 1_000_000  # mm^2 -> m^2
    steel_weight = steel_area * steel_density
    steel_cost = steel_weight * steel_cost_by_weight
    
    concrete_area = ((beff * hf) + (h - hf) * bw) / 1_000_000 - steel_area
    f_ck_int = int(f_ck)
    concrete_cost = concrete_area * concrete_cost_by_class[f_ck_int]
    
    total_cost = steel_cost + concrete_cost
    
    return total_cost

num_iterations = 10000000
data_list = []

for _ in tqdm.tqdm(range(num_iterations), desc="Running simulations"):

    MEd = np.random.uniform(low=100, high=15_000)
    beff = np.random.uniform(low=100, high=4000)
    bw = np.random.uniform(low=100, high=1500)
    h = np.random.uniform(low=300, high=2000)
    hf = np.random.uniform(low=100, high=500)

    fck = np.random.choice([20, 25, 30, 35, 40, 45, 50, 55])
    fi = np.random.choice([8, 10, 12, 16, 20, 25, 28, 32])

    Es = 200_000.0
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    fyk = 500
    fyd = fyk / 1.15
    fcd = fck / 1.4
    
    cnom = 48
    a1 = cnom + fi/2
    a2 = cnom + fi/2
    d = h - a1

    a = (-0.5) * beff * fcd
    b = beff * fcd * d
    c = -MEd * 1e6
    xeff = quadratic_equation(a, b, c, h)

    ksieff = xeff / d

    As1 = xeff * beff * fcd / fyd
    As2 = 0

    MRd = xeff * beff * fcd * (d - 0.5 * xeff) / 1e6

    cost = calculate_section_cost(b, h, f_ck, A_s1, A_s2)
    # Store final data
    data_entry = {
        'MEd': MEd,
        'MRd': MRd,
        'beff': beff,
        'bw': bw,
        'h': h,
        'hf': hf,
        'fi': fi,
        'fck': fck,
        'As1': As1,
        'As2': As2,
        'cost': cost
    }
    data_list.append(data_entry)

    if ksieff > ksiefflim:
        continue

    if xeff == None:
        continue
    
    if beff <= bw:
        continue

    if h < hf:
        continue

    if d < 2 * a1:
        continue

# Save results
if data_list:
    df = pd.DataFrame(data_list)
    df.to_excel("dataset.xlsx", index=False)
    print(f"\nSaved {len(data_list)} valid results to 'dataset.parquet'")
else:
    print("\nNo valid cases found.")
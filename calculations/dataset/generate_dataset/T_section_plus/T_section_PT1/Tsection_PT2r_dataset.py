import math
import tqdm
import pandas as pd
import numpy as np
import os

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
    if a == 0:
        return None  # Not a quadratic equation
        
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

        if not valid_solutions:
            return None
        
        # Jeśli obie leżą w przedziale, zwracamy mniejszą
        return min(valid_solutions)   

def calculate_section_cost(beff: float, bw: float, h: float, hf: float, fck: float, As1: float, As2: float) -> float:
    concrete_cost_by_class = {
        8: 230, 12: 250, 16: 300, 20: 350, 25: 400, 30: 450, 35: 500, 40: 550, 45: 600, 50: 650, 55: 700, 60: 800
    }
    
    steel_cost_by_weight = 5  # zł/kg
    steel_density = 7900      # kg/m3
    
    steel_area = (As1 + As2) / 1_000_000  # mm^2 -> m^2
    steel_weight = steel_area * steel_density
    steel_cost = steel_weight * steel_cost_by_weight
    
    concrete_area = ((beff * hf) + (h - hf) * bw) / 1_000_000 - steel_area
    fck_int = int(fck)
    concrete_cost = concrete_area * concrete_cost_by_class[fck_int]
    
    total_cost = steel_cost + concrete_cost
    
    return total_cost

num_iterations = 250000
data_list = []

for _ in tqdm.tqdm(range(num_iterations), desc="Running simulations"):
    MEd = np.random.uniform(low=100, high=15_000)
    beff = np.random.uniform(low=100, high=4000)
    bw = np.random.uniform(low=100, high=1500)
    h = np.random.uniform(low=300, high=2000)
    hf = np.random.uniform(low=100, high=500)

    fck = np.random.choice([20, 25, 30, 35, 40, 45, 50, 55])
    fi = np.random.choice([8, 10, 12, 16, 20, 25, 28, 32])

    fyk = 500
    fyd = fyk / 1.15
    fcd = fck / 1.4

    Es = 200_000.0
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)
    
    cnom = 48
    a1 = cnom + fi * 3 / 2
    a2 = cnom + fi/2
    d = h - a1

    # Validation checks before calculations
    if beff <= bw:
        continue

    if h < hf:
        continue

    if d <= 2 * a1:
        continue

    a = (-0.5) * beff * fcd
    b = beff * fcd * d
    c = -MEd * 1e6
    xeff = quadratic_equation(a, b, c, h)

    if xeff is None:
        continue
        
    ksieff = xeff / d
    if ksieff > ksiefflim:
        continue

    As1 = xeff * beff * fcd / fyd
    As2 = 0

    MRd = xeff * beff * fcd * (d - 0.5 * xeff) / 1e6

    n1 = math.ceil(As1 / (fi**2 * math.pi / 4))
    n2 = math.ceil(As2 / (fi**2 * math.pi / 4))

    As1_prov = n1 *( fi**2 * math.pi / 4) 
    As2_prov = n2 *( fi**2 * math.pi / 4) 

    xeff1 = As1_prov * fyd / (beff * fcd)
    ksieff1 = xeff1 / d

    if ksieff1 <= ksiefflim:
        MRd = xeff1 * beff * fcd * (d - 0.5 * xeff1) / 1e6
    else:
        continue

    Fc = hf * beff / fcd
    Fs = As1_prov * fyd

    #if Fc < Fs:
        #continue

    smax = max(20, fi)

    limit = 2 * cnom + fi * n1 + smax * (n1 - 1)

    if 2 * bw < limit:
        continue

    cost = calculate_section_cost(beff, bw, h, hf, fck, As1, As2)
    
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
        'n1': n1,
        'n2': n2,
        'As2': As2,
        'cost': cost
    }
    data_list.append(data_entry)

# Save results
if data_list:
    df = pd.DataFrame(data_list)
    csv_path = os.path.join(r'C:\Users\marci\Documents\GitHub\dyploma\datasets', 'PT2r_dataset.parquet')
    df.to_parquet(csv_path, index=False)
    print(f"\nSaved {len(data_list)} valid results to 'dataset.parquet'")
else:
    print("\nNo valid cases found.")
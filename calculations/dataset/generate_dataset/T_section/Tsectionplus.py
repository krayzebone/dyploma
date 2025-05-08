# ───── Imports ───────────────────────────────────────────────────────────────
import math, tqdm, numpy as np, pandas as pd
from typing import List, Tuple, Dict, Any

# ───── Helper functions ──────────────────────────────────────────────────────
def quadratic_equation(a: float, b: float, c: float, limit: float) -> float | None:
    if a == 0:
        return None
    Δ = b**2 - 4*a*c
    if Δ < 0:
        return None
    r1 = (-b - math.sqrt(Δ)) / (2*a)
    r2 = (-b + math.sqrt(Δ)) / (2*a)
    roots = [x for x in (r1, r2) if 0 < x < limit]
    return min(roots) if roots else None

def calculate_number_of_rods(As: float, fi: float) -> Tuple[int, float]:
    """Minimum bar count + provided area."""
    if As <= 0:           # no reinforcement required
        return 0, 0.0
    Aφ = math.pi * fi**2 / 4
    n   = math.ceil(As / Aφ)
    return n, n*Aφ

def distribute_rods_2_layers(bw: float, cnom: float,
                             n: int, fi: float) -> Tuple[bool, List[int]]:
    """Return (fits?, [n_bottom, n_second])."""
    if n == 0:
        return True, [0, 0]
    smax  = max(20., fi)
    pitch = fi + smax
    n_max = math.floor((bw - 2*cnom + smax) / pitch)
    if n_max <= 0:
        return False, [0, 0]
    n_bottom = min(n, n_max)
    n_second = n - n_bottom
    return n_second <= n_max, [n_bottom, n_second]

def centroid_shift(rods: List[int], fi: float) -> float:
    """yc above bottom layer for two‑layer pack."""
    n1, n2 = rods
    if n1 + n2 == 0:
        return 0.0
    Aφ  = math.pi * fi**2 / 4
    dy  = fi + max(20., fi)
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

def evaluate_section(
    MEd: float,
    beff: float,
    bw: float,
    h: float,
    hf: float,
    cnom: float,
    fi_str: float
) -> Dict[str, Any] | None:
    """Evaluate all combinations of fi and fck for a given section and return the lowest cost option."""
    possible_fcks = [16, 20, 25, 30, 35, 40, 45, 50]
    possible_fis = [8, 10, 12, 14, 16, 20, 25, 28, 32]
    
    best_solution = None
    min_cost = float('inf')
    
    for fck in possible_fcks:
        for fi in possible_fis:
            a1 = cnom + fi/2 + fi_str
            d0 = h - a1
            fcd = fck / 1.4
            fyd = 500 / 1.15  # fyk is always 500
            
            xeff = quadratic_equation(-0.5*beff*fcd, beff*fcd*d0, -MEd, h)
            if xeff is None:
                continue
                
            ξ = xeff / d0
            ξ_lim = 0.8*0.0035 / (0.0035 + fyd/200_000)
            
            # --------- single reinforcement first ----------------------------------
            if ξ <= ξ_lim:
                As1 = (xeff * beff * fcd) / fyd
                n1, As1_prov = calculate_number_of_rods(As1, fi)
                fit, r1 = distribute_rods_2_layers(bw, cnom, n1, fi)
                if not fit:
                    continue
                    
                # depth shift
                yc = centroid_shift(r1, fi)
                d = d0 - yc
                a1p = a1 + yc
                
                x2 = quadratic_equation(-0.5*beff*fcd, beff*fcd*d, -MEd, h)
                if x2 is None:
                    continue
                    
                if x2/d > ξ_lim:  # add compression bars
                    x2 = ξ_lim * d
                    As2 = (MEd - x2*beff*fcd*(d - 0.5*x2)) / (fyd*(d - a1p))
                    As1 = (As2*fyd + x2*beff*fcd) / fyd
                    n1, As1_prov = calculate_number_of_rods(As1, fi)
                    n2, As2_prov = calculate_number_of_rods(As2, fi)
                    fit1, r1 = distribute_rods_2_layers(bw, cnom, n1, fi)
                    fit2, _ = distribute_rods_2_layers(bw, cnom, n2, fi)
                    if not (fit1 and fit2):
                        continue
                        
                    cost = calc_cost(beff, bw, h, hf, fck, As1_prov, As2_prov)
                    if cost < min_cost:
                        min_cost = cost
                        best_solution = dict(
                            MEd=MEd/1e6, beff=beff, bw=bw, h=h, hf=hf,
                            fi=fi, fck=fck, fi_str=fi_str, cnom=cnom,
                            rods_layer1=r1[0], rods_layer2=r1[1],
                            rods_compression=n2, cost=cost)
                else:  # tension only
                    cost = calc_cost(beff, bw, h, hf, fck, As1_prov, 0)
                    if cost < min_cost:
                        min_cost = cost
                        best_solution = dict(
                            MEd=MEd/1e6, beff=beff, bw=bw, h=h, hf=hf,
                            fi=fi, fck=fck, fi_str=fi_str, cnom=cnom,
                            rods_layer1=r1[0], rods_layer2=r1[1],
                            rods_compression=0, cost=cost)
            # --------- directly doubly‑reinforced -----------------------------------
            else:
                xeff = ξ_lim * d0
                As2 = (MEd - xeff*beff*fcd*(d0 - 0.5*xeff)) / (fyd*(d0 - a1))
                As1 = (As2*fyd + xeff*beff*fcd) / fyd
                n1, As1_prov = calculate_number_of_rods(As1, fi)
                n2, As2_prov = calculate_number_of_rods(As2, fi)
                fit1, r1 = distribute_rods_2_layers(bw, cnom, n1, fi)
                fit2, _ = distribute_rods_2_layers(bw, cnom, n2, fi)
                if not (fit1 and fit2):
                    continue
                    
                cost = calc_cost(beff, bw, h, hf, fck, As1_prov, As2_prov)
                if cost < min_cost:
                    min_cost = cost
                    best_solution = dict(
                        MEd=MEd/1e6, beff=beff, bw=bw, h=h, hf=hf,
                        fi=fi, fck=fck, fi_str=fi_str, cnom=cnom,
                        rods_layer1=r1[0], rods_layer2=r1[1],
                        rods_compression=n2, cost=cost)
    
    return best_solution

# ───── Monte‑Carlo loop ──────────────────────────────────────────────────────
N = 1000_000
rows = []

for _ in tqdm.tqdm(range(N), desc="simulations"):
    # random section / material data
    MEd = np.random.uniform(10, 10_000) * 1e6        # [Nmm]
    beff = np.random.uniform(100, 2_000)
    bw = np.random.uniform(100, 2_000)
    h = np.random.uniform(100, 1_500)
    hf = np.random.uniform(100, 1_500)
    cnom = np.random.uniform(20, 60)
    if hf > h or bw > beff:
        continue

    fi_str = np.random.choice([8, 10, 12, 14, 16])
    
    # Evaluate all combinations for this section
    solution = evaluate_section(MEd, beff, bw, h, hf, cnom, fi_str)
    if solution:
        rows.append(solution)

# ───── Export ────────────────────────────────────────────────────────────────
if rows:
    df = pd.DataFrame(rows, columns=[
        "MEd", "beff", "bw", "h", "hf", "fi", "fck", "fi_str", "cnom",
        "rods_layer1", "rods_layer2", "rods_compression", "cost"])
    df.to_parquet("Tsection_optimized.parquet", index=False)
    print(f"\nSaved {len(df)} rows to 'Tsection_optimized.csv'")
else:
    print("\nNo valid cases found.")
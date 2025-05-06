# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import math
import tqdm
import numpy as np
import pandas as pd
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def quadratic_equation(a: float, b: float, c: float, limit: float) -> float | None:
    if a == 0:                         # linear or degenerate
        return None
    delta = b**2 - 4 * a * c
    if delta < 0:
        return None
    sqrt_d = math.sqrt(delta)
    roots  = [(-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)]
    valid  = [x for x in roots if 0 < x < limit]
    return min(valid) if valid else None

def calculate_number_of_rods(As: float, fi: float) -> Tuple[int, float]:
    if As <= 0:
        return 0, 0.0
    area_bar = math.pi * fi**2 / 4
    n = math.ceil(As / area_bar)
    return n, n * area_bar

# ── two‑layer helpers ────────────────────────────────────────────────────────
def distribute_rods_2_layers(bw: float, cnom: float,
                             num_rods: int, fi: float) -> Tuple[bool, List[int]]:
    if num_rods == 0:
        return True, [0, 0]
    smax          = max(20.0, fi)
    pitch         = fi + smax
    max_per_layer = math.floor((bw - 2*cnom + smax) / pitch)
    if max_per_layer <= 0:
        return False, [0, 0]
    n_bottom = min(num_rods, max_per_layer)
    n_second = num_rods - n_bottom
    return n_second <= max_per_layer, [n_bottom, n_second]

def centroid_shift_2_layers(rods_per_layer: List[int], fi: float, cnom: float) -> float:
    n1, n2 = rods_per_layer
    if n1 + n2 == 0:
        return 0.0
    area_bar = math.pi * fi**2 / 4
    dy       = fi + max(20.0, fi)
    return (n2 * area_bar * dy) / ((n1 + n2) * area_bar)

def update_effective_depth_two_layers(d: float, a1: float,
                                      rods_per_layer: List[int],
                                      fi: float, cnom: float) -> Tuple[float, float, float]:
    yc = centroid_shift_2_layers(rods_per_layer, fi, cnom)
    return yc, d - yc, a1 + yc

# ──────────────────────────────────────────────────────────────────────────────
# Monte‑Carlo simulation
# ──────────────────────────────────────────────────────────────────────────────
num_iterations = 100_000
data_list      = []

for _ in tqdm.tqdm(range(num_iterations), desc="Running simulations"):
    # — random geometry & material —
    MEd  = np.random.uniform(10, 10_000) * 1e6
    beff = np.random.uniform(100, 2_000)
    bw   = np.random.uniform(100, 2_000)
    h    = np.random.uniform(100, 1_500)
    hf   = np.random.uniform(100, 1_500)
    cnom = np.random.uniform(20, 60)
    if hf > h or bw > beff:
        continue

    fck   = np.random.choice([16, 20, 25, 30, 35, 40, 45, 50])
    fyk   = 500
    fi    = np.random.choice([ 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32])
    fi_st = np.random.choice([ 8, 10, 12, 14, 16, 18, 20])

    # — section parameters —
    a1        = cnom + fi/2 + fi_st
    d_initial = h - a1                       # store the *original* effective depth
    fcd       = fck / 1.4
    fyd       = fyk / 1.15

    # — neutral‑axis depth from quadratic equilibrium —
    xeff = quadratic_equation(-0.5 * beff * fcd, beff * fcd * d_initial, -MEd, h)
    if xeff is None:
        continue
    ksieff     = xeff / d_initial
    ksieff_lim = 0.8 * 0.0035 / (0.0035 + fyd / 200_000)

    # ───────── Case 1 – single reinforcement first ──────────────────────────
    if ksieff <= ksieff_lim:
        As1, As2 = (xeff * beff * fcd) / fyd, 0.0
        n1, _ = calculate_number_of_rods(As1, fi)
        fits, rp = distribute_rods_2_layers(bw, cnom, n1, fi)
        if not fits:
            continue
        yc, d_p, a1_p = update_effective_depth_two_layers(d_initial, a1, rp, fi, cnom)

        # re‑check with shifted depth
        xeff_p = quadratic_equation(-0.5 * beff * fcd, beff * fcd * d_p, -MEd, h)
        if xeff_p is None:
            continue
        ksieff_p = xeff_p / d_p
        if ksieff_p > ksieff_lim:            # switch to doubly reinforced
            xeff_p = ksieff_lim * d_p
            As2    = (MEd - xeff_p*beff*fcd*(d_p - 0.5*xeff_p)) / (fyd*(d_p - a1_p))
            As1    = (As2*fyd + xeff_p*beff*fcd) / fyd
            n1, _  = calculate_number_of_rods(As1, fi)
            n2, _  = calculate_number_of_rods(As2, fi)
            fits1, rp1 = distribute_rods_2_layers(bw, cnom, n1, fi)
            fits2, rp2 = distribute_rods_2_layers(bw, cnom, n2, fi)
            if not (fits1 and fits2):
                continue
            yc1 = centroid_shift_2_layers(rp1, fi, cnom)
            yc2 = centroid_shift_2_layers(rp2, fi, cnom)
            d_f  = d_initial - (yc1 + yc2)/2
            a1_f = a1 + (yc1 + yc2)/2
            xeff_f = ksieff_lim * d_f
            As2 = (MEd - xeff_f*beff*fcd*(d_f - 0.5*xeff_f)) / (fyd*(d_f - a1_f))
            As1 = (As2*fyd + xeff_f*beff*fcd) / fyd
            data_list.append(dict(
                MEd=MEd/1e6, beff=beff, bw=bw, h=h, hf=hf,
                fck=fck, fi=fi, fi_str=fi_st, cnom=cnom,
                As1=As1, As2=As2, xeff=xeff_f,
                d_initial=d_initial, d_prime=d_f,
                a1_prime=a1_f, yc=(yc1+yc2)/2,
                rods_per_layer_tension=rp1,
                rods_per_layer_compression=rp2,
                num_rods_tension=n1, num_rods_compression=n2,
                case=2))
        else:                                # still single reinforcement
            As1 = (xeff_p * beff * fcd) / fyd
            data_list.append(dict(
                MEd=MEd/1e6, beff=beff, bw=bw, h=h, hf=hf,
                fck=fck, fi=fi, fi_str=fi_st, cnom=cnom,
                As1=As1, As2=0.0, xeff=xeff_p,
                d_initial=d_initial, d_prime=d_p,
                a1_prime=a1_p, yc=yc,
                rods_per_layer_tension=rp,
                rods_per_layer_compression=[0,0],
                num_rods_tension=n1, num_rods_compression=0,
                case=1))

    # ───────── Case 2 – doubly reinforced from the start ────────────────────
    else:
        xeff    = ksieff_lim * d_initial
        As2     = (MEd - xeff*beff*fcd*(d_initial - 0.5*xeff)) / (fyd * (d_initial - a1))
        As1     = (As2*fyd + xeff*beff*fcd) / fyd
        n1, _   = calculate_number_of_rods(As1, fi)
        n2, _   = calculate_number_of_rods(As2, fi)
        fits1, rp1 = distribute_rods_2_layers(bw, cnom, n1, fi)
        fits2, rp2 = distribute_rods_2_layers(bw, cnom, n2, fi)
        if not (fits1 and fits2):
            continue
        yc1 = centroid_shift_2_layers(rp1, fi, cnom)
        yc2 = centroid_shift_2_layers(rp2, fi, cnom)
        d_p  = d_initial - (yc1 + yc2)/2
        a1_p = a1 + (yc1 + yc2)/2
        xeff_p = ksieff_lim * d_p
        As2 = (MEd - xeff_p*beff*fcd*(d_p - 0.5*xeff_p)) / (fyd*(d_p - a1_p))
        As1 = (As2*fyd + xeff_p*beff*fcd) / fyd
        data_list.append(dict(
            MEd=MEd/1e6, beff=beff, bw=bw, h=h, hf=hf,
            fck=fck, fi=fi, fi_str=fi_st, cnom=cnom,
            As1=As1, As2=As2, xeff=xeff_p,
            d_initial=d_initial, d_prime=d_p,
            a1_prime=a1_p, yc=(yc1+yc2)/2,
            rods_per_layer_tension=rp1,
            rods_per_layer_compression=rp2,
            num_rods_tension=n1, num_rods_compression=n2,
            case=2))

# ──────────────────────────────────────────────────────────────────────────────
# Results and CSV export
# ──────────────────────────────────────────────────────────────────────────────
if data_list:
    results_df = pd.DataFrame(data_list)
    results_df.to_csv("Tsection222.csv", index=False)
    print(f"\nSaved {len(data_list)} valid results to 'Tsection222.csv'")
else:
    print("\nNo valid cases found.")

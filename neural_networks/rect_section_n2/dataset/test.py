import math
import tqdm
import pandas as pd
import numpy as np

num_iterations = 100000 #20min
data_list = []

def quadratic_equation(a: float, b: float, c: float, limit: float) -> float | None:
    """Solve ax²+bx+c = 0 and return the root that lies in (0, limit)."""
    if a == 0:
        return None

    delta = b ** 2 - 4 * a * c
    if delta < 0:
        return None

    sqrt_delta = math.sqrt(delta)
    x1 = (-b - sqrt_delta) / (2 * a)
    x2 = (-b + sqrt_delta) / (2 * a)
    valid = [x for x in (x1, x2) if 0 < x < limit]
    return min(valid) if valid else None

def calc_cost(b: float, h: float, f_ck: float, A_s1: float, A_s2: float) -> float:
    concrete_cost_by_class = {
        8: 230, 12: 250, 16: 300, 20: 350, 25: 400, 30: 450, 35: 500, 40: 550, 45: 600, 50: 650, 55: 700, 60: 800
    }
    
    steel_cost_by_weight = 5  # zł/kg
    steel_density = 7900      # kg/m3
    
    steel_area = (A_s1 + A_s2) / 1_000_000  # mm^2 -> m^2
    steel_weight = steel_area * steel_density
    steel_cost = steel_weight * steel_cost_by_weight
    
    concrete_area = (b * h) / 1_000_000 - steel_area
    f_ck_int = int(f_ck)
    concrete_cost = concrete_area * concrete_cost_by_class[f_ck_int]
    
    total_cost = steel_cost + concrete_cost
    
    return total_cost

def find_valid_ro(b, h, fi_gl, n_min=2, n_max=200):
    """Find all reinforcement ratios that result in integer bar counts"""
    area_one_bar = math.pi * (fi_gl**2) / 4.0
    valid_ro = []
    for n in range(n_min, n_max + 1):
        ro = (n * area_one_bar) / (b * h)
        if 0.001 <= ro <= 0.04:  # Practical limits
            valid_ro.append(ro)
    return np.array(valid_ro)

def calc_rect_section(MEd, b, h, fck, fi_gl, c_nom, fi_str):

    # Parametry materiałowe
    fyk = 500
    fcd = fck / 1.4  # wytrzymałość obliczeniowa betonu na ściskanie [MPa]
    fyd = fyk / 1.15  # wytrzymałość obliczeniowa stali na rozciąganie [MPa]
    E_s = 200_000.0  # moduł Younga stali [MPa]

    # Wyznaczenie współczynnika względnej wysokości strefy ściskanej
    ksi_eff_lim = 0.8 * 0.0035 / (0.0035 + fyd / E_s)

    # Podstawowe wymiary przekroj
    a_1 = c_nom + fi_gl / 2 + fi_str
    d = h - a_1

    aq = -0.5 * b * fcd
    bq = b * fcd * d
    cq = -MEd

    xeff = quadratic_equation(aq, bq, cq, h)
    if xeff is None:
        # brak rozwiązania, zwróć nieskończony koszt aby odrzucić wariant
        return float("inf"), float("inf"), float("inf"), "brak rozwiązania"

    ksieff = xeff / d

    if ksieff < ksi_eff_lim:
        reinforcement_type = "pojedyńczo zbrojony"
        As1 = xeff * b * fcd / fyd
        As2 = 0.0
        cost = calc_cost(b, h, fck, As1, As2)
        return As1, As2, cost, reinforcement_type
    
    else: 
        reinforcement_type = "podwójnie zbrojony"
        x_eff = ksi_eff_lim * d
        As2 = (-x_eff * b * fcd * (d - 0.5 * x_eff) + MEd) / (fyd * (d - a_1))
        As1 = (As2 * fyd + x_eff * b * fcd) / fyd
        cost = calc_cost(b, h, fck, As1, As2)
        return As1, As2, cost, reinforcement_type




for _ in tqdm.tqdm(range(num_iterations), desc="Running simulations"):
    #####################################################################
    #   1. Parametry wejściowe
    #####################################################################
    # Geometry of section
    b = np.random.normal(loc=1000, scale=300)
    h = np.random.normal(loc=300, scale=200)
    
    # Concrete choice
    f_ck = np.random.choice([16, 20, 25, 30, 35, 40, 45, 50])
    f_yk = 500
    
    # Choose bar diameter from discrete set
    fi_gl = np.random.choice([8, 10, 12, 14, 16, 20, 25, 28, 32])
    fi_str = np.random.choice([8, 10, 12, 14, 16])
    c_nom = np.random.uniform(low=10, high=60)
    
    # External moment
    M_Ed = np.random.uniform(low=10, high=2000) * 1e6
    MEqp = np.random.uniform(low=10, high=2000) * 1e6

    # Material constants
    E_s = 200_000       # MPa (steel)
    E_cm = 31_000       # MPa (concrete)
    RH = 70             # relative humidity in %
    f_cm_map = {16: 20, 20: 28, 25: 33, 30: 38, 35: 43, 40: 48, 45: 53, 50: 58, 55: 63, 60: 68}
    t_0 = 28            # days, age of concrete at loading
    t   = 8             # days
    f_ctm = 0.3 * f_ck**(2/3)

    A_s1, A_s2, cost, reinforcement_type = calc_rect_section(M_Ed, b, h, f_ck, fi_gl, c_nom, fi_str)


    # Calculate exact integer bar counts
    area_one_bar = math.pi * (fi_gl**2) / 4.0

    # Recalculate exact reinforcement ratios from integer bar counts
    ro_s1 = A_s1 / (b * h)
    ro_s2 = A_s2 / (b * h)

    # Effective depth
    a_1 = c_nom + fi_gl / 2
    d = h - a_1
    
    # Skip if d is invalid
    if d <= 0:
        continue

    #####################################################################
    #   3. Check section capacity
    #####################################################################
    f_cd = f_ck / 1.4
    f_yd = f_yk / 1.15

    x_eff = (A_s1*f_yd - A_s2*f_yd) / (b * f_cd)
    ksi_eff = x_eff / d
    ksi_eff_lim = 0.8 * 0.0035 / (0.0035 + f_yd / E_s)

    if ksi_eff <= ksi_eff_lim:
        continue
        M_Rd = (x_eff * b * f_cd * (d - 0.5 * x_eff))
    
    else:
        x_eff = ksi_eff_lim * d

        if x_eff < 2 * a_1:
            M_Rd = (A_s1 * f_yd * (d - a_1))
        else:
            M_Rd = (x_eff * b * f_cd * (d - 0.5 * x_eff) + A_s2 * f_yd * (d - a_1))

    

    #####################################################################
    #   4. Cracking checks
    #####################################################################

    # 4.1 Creep coefficient
    A_c = b * h
    u   = 2*(b + h)
    h_0 = 2*A_c / u

    alpha_1 = (35 / f_cm_map[f_ck])**0.7
    alpha_2 = (35 / f_cm_map[f_ck])**0.2
    alpha_3 = (35 / f_cm_map[f_ck])**0.5
    alpha   = 0  # for cement N

    if f_ck <= 35:
        fi_RH = 1 + ((1 - RH / 100) / (0.1 * h_0**(1/3)))
    else:
        fi_RH = (1 + (1 - RH/100) / (0.1*h_0**(1/3)*alpha_1)) * alpha_2

    Beta_t0  = 1 / (0.1 + t_0**0.2)
    Beta_fcm = 16.8 / math.sqrt(f_cm_map[f_ck])
    fi_0 = fi_RH * Beta_fcm * Beta_t0

    # 4.2 Stage I analysis
    E_c_eff = E_cm  # or E_cm / (1 + fi_0) if you want to include creep
    alpha_cs = E_s / E_c_eff

    # Transformed area
    A_cs = A_c + alpha_cs*(A_s1 + A_s2)

    # First moment
    S_cs = A_c*(h/2) + alpha_cs*(A_s1*d + A_s2*a_1)

    # Centroid from the bottom
    x_I = S_cs / A_cs

    # Moment of inertia (uncracked)
    I_I = b*(h**3)/12 + b*h*(h/2 - x_I)**2 + alpha_cs*(A_s1*(d - x_I)**2 + A_s2*(x_I - a_1)**2)

    # Section modulus for tension
    W_cs = I_I / (h - x_I)

    # Cracking moment
    M_cr = f_ctm * W_cs

    # 4.3 Stage II analysis
    ro_s  = ro_s1 + ro_s2
    delta_ul = a_1 / d
    k_lu = (ro_s1 + delta_ul * ro_s2) / ro_s if ro_s > 0 else 0

    ro_cs = ro_s * alpha_cs
    sqrt_arg = ro_cs * (ro_cs + 2*k_lu)
    if sqrt_arg < 0:
        continue

    ksi_II = math.sqrt(sqrt_arg) - ro_cs
    x_II   = ksi_II * d

    # Moment of inertia (cracked)
    I_II = b*(x_II**3)/3 + alpha_cs*(A_s1*(d - x_II)**2 + A_s2*(x_II - a_1)**2)

    k_t = 0.4
    sigma_s_cr = k_t * alpha_cs * f_ctm * (d - x_I)/(h - x_I)
    sigma_s = (alpha_cs * MEqp / I_II) * (d - x_II)
    delta_sigma = max(sigma_s - sigma_s_cr, 0.6 * sigma_s)
    epsilon_cr = delta_sigma / E_s

    k_1 = 0.8
    k_2 = 0.5
    k_3 = 3.4
    k_4 = 0.425

    A_c_eff = b * min(2.5*(h - d), (h - x_I)/3)
    if A_c_eff <= 0:
        continue

    ro_p_eff = (A_s1 + A_s2) / A_c_eff
    if ro_p_eff <= 0:
        continue

    s_r_max = k_3*c_nom + k_1*k_2*k_4*fi_gl / ro_p_eff
    w_k = delta_sigma * epsilon_cr
    w_k_max = 0.3

    #####################################################################
    #   5. Filtering / constraints
    #####################################################################
    if d < 2*a_1:
        continue

    if w_k > 5:
        continue

    # Min & max area constraints
    A_s_min = max(0.26*(f_ctm/f_yk)*b*d, 0.0013*b*d)
    A_s_max = 0.04 * b * h
    if (A_s1 + A_s2) < A_s_min or (A_s1 + A_s2) > A_s_max:
        continue


    cost = calc_cost(b, h, f_ck, A_s1, A_s2)

    # Store final data
    data_entry = {
        'MRd': M_Rd / 1e6,
        'MEd': M_Ed / 1e6,
        'MEqp':MEqp /1e6,
        'b': b,
        'h': h,
        'fi': fi_gl,
        'fck': f_ck,
        'd': d,
        'cnom': c_nom,
        'ro1': ro_s1,
        'ro2': ro_s2,
        'Wk': w_k,
        'Mcr': M_cr / 1e6,
        'Cost': cost,
    }
    data_list.append(data_entry)

# Save results
if data_list:
    df = pd.DataFrame(data_list)
    df.to_parquet(r"neural_networks\rect_section_n2\dataset\dataset_rect_n2test.parquet", index=False)
    print(f"\nSaved {len(data_list)} valid results to 'dataset.parquet'")
else:
    print("\nNo valid cases found.")

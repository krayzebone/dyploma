import math

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

def calc_cost(beff: float, bw: float, h: float, hf: float, fck: float, As1: float, As2: float) -> float:
    concrete_cost_by_class = {
        8: 230, 12: 250, 16: 300, 20: 350, 25: 400, 30: 450, 35: 500, 40: 550, 
        45: 600, 50: 650, 55: 700, 60: 800
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

def calc_PT_1r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    a1 = cnom + fi / 2 + fi_str
    a2 = cnom + fi / 2 + fi_str
    d = h - a1
    
    # Define coefficients for the quadratic equation
    a = (-0.5) * beff * fcd
    b = beff * fcd * d
    c = -MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf')

    ksieff = xeff / d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        # Calculate reinforcement area
        As1 = xeff * beff * fcd / fyd
        As2 = 0
    else:
        As2 = (MEd * 1e6 - xeff * beff * fcd * (d - 0.5 * xeff)) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * beff * fcd) / fyd

    cost = calc_cost(beff, bw, h, hf, fck, As1, As2)
    return As1, As2, cost

def calc_PT_2r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    smax = max(20, fi)
    a1 = cnom + fi_str + fi + smax / 2
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    # Define coefficients for the quadratic equation
    a = (-0.5) * beff * fcd
    b = beff * fcd * d
    c = -MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf')

    ksieff = xeff / d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        As1 = xeff * beff * fcd / fyd
        As2 = 0
    else:
        As2 = (MEd * 1e6 - xeff * beff * fcd * (d - 0.5 * xeff)) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * beff * fcd) / fyd

    cost = calc_cost(beff, bw, h, hf, fck, As1, As2)
    return As1, As2, cost

def calc_PT_3r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    smax = max(20, fi)
    a1 = cnom + fi_str + fi * 3/2 + smax
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    # Define coefficients for the quadratic equation
    a = (-0.5) * beff * fcd
    b = beff * fcd * d
    c = -MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf')

    ksieff = xeff / d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        As1 = xeff * beff * fcd / fyd
        As2 = 0
    else:
        As2 = (MEd * 1e6 - xeff * beff * fcd * (d - 0.5 * xeff)) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * beff * fcd) / fyd

    cost = calc_cost(beff, bw, h, hf, fck, As1, As2)
    return As1, As2, cost

def calc_RZT_1r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    a1 = cnom + fi / 2 + fi_str
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    # Define coefficients for the quadratic equation
    a = (-0.5) * bw * fcd
    b = bw * fcd * d
    c = hf * (beff - bw) * fcd * (d - 0.5 * hf) - MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf')

    ksieff = xeff / d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        As1 = (xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        As2 = 0
    else:
        As2 = (-xeff * bw * fcd * (d - 0.5 * xeff) - hf * (beff - bw) * fcd * (d - 0.5 * hf) + MEd * 1e6) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd

    cost = calc_cost(beff, bw, h, hf, fck, As1, As2)
    return As1, As2, cost

def calc_RZT_2r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    smax = max(20, fi)
    a1 = cnom + fi_str + fi + smax / 2
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    # Define coefficients for the quadratic equation
    a = (-0.5) * bw * fcd
    b = bw * fcd * d
    c = hf * (beff - bw) * fcd * (d - 0.5 * hf) - MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf')

    ksieff = xeff / d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        As1 = (xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        As2 = 0
    else:
        As2 = (-xeff * bw * fcd * (d - 0.5 * xeff) - hf * (beff - bw) * fcd * (d - 0.5 * hf) + MEd * 1e6) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd

    cost = calc_cost(beff, bw, h, hf, fck, As1, As2)
    return As1, As2, cost

def calc_RZT_3r_plus(MEd, beff, bw, h, hf, fi, fi_str, cnom, fcd, fyd, fck):
    smax = max(20, fi)
    a1 = cnom + fi_str + fi *3/2 + smax
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    # Define coefficients for the quadratic equation
    a = (-0.5) * bw * fcd
    b = bw * fcd * d
    c = hf * (beff - bw) * fcd * (d - 0.5 * hf) - MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)
    if xeff is None:
        return float('inf'), float('inf'), float('inf')

    ksieff = xeff / d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        As1 = (xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        As2 = 0
    else:
        As2 = (-xeff * bw * fcd * (d - 0.5 * xeff) - hf * (beff - bw) * fcd * (d - 0.5 * hf) + MEd * 1e6) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd

    cost = calc_cost(beff, bw, h, hf, fck, As1, As2)
    return As1, As2, cost

def calculate_number_of_rods(As: float, fi: float) -> tuple:
    """
    Calculate the number of steel rods needed for a given reinforcement area.
    
    Args:
        As: Required reinforcement area in mm²
        fi: Diameter of the steel rod in mm
        
    Returns:
        A tuple containing:
        - number_of_rods: The minimum number of rods needed (rounded up)
        - actual_As: The actual provided reinforcement area
    """
    if As <= 0 or math.isinf(As):
        return 0, 0
    
    # Calculate area of one rod
    rod_area = math.pi * (fi ** 2) / 4
    
    # Calculate number of rods needed (round up)
    number_of_rods = math.ceil(As / rod_area)
    
    # Calculate actual provided area
    actual_As = number_of_rods * rod_area
    
    return number_of_rods, actual_As

def check_rods_fit(bw: float, cnom: float, num_rods: int, fi: float, smax: float, layers: int = 1) -> bool:
    """
    Check if the number of rods can fit within the available section width for a given number of layers.
    Args:
        bw: Section width (mm)
        cnom: Concrete cover (mm)
        num_rods: Number of reinforcement bars
        fi: Bar diameter (mm)
        smax: Maximum spacing between bars (mm)
        layers: Number of layers of rods (1, 2, or 3)
    Returns:
        True if the bars fit, False otherwise
    """
    if num_rods == 0:
        return True  # No rods always fits

    # Total required width for one row of rods
    required_width = 2 * cnom + num_rods * fi + smax * (num_rods - 1)
    # Total available width increases with number of layers
    available_width = layers * bw
    return required_width <= available_width

def find_optimal_scenario(inputs):
    MEd   = inputs['MEd']
    beff  = inputs['beff']
    bw    = inputs['bw']
    h     = inputs['h']
    hf    = inputs['hf']
    cnom  = inputs['cnom']
    fi_str= inputs['fi_str']
    fyk   = 500.0

    best = {
        'cost': float('inf'),
        'fck': None,
        'fi': None,
        'type': None,        # 'PT' or 'RZT'
        'layers': None,      # 1,2 or 3
        'As1': None,
        'As2': None,
        'num_rods_As1': None,
        'num_rods_As2': None,
        'actual_As1': None,
        'actual_As2': None,
        'fit_check': None   # Whether the rods fit in the section
    }

    for fck in possible_fck:
        fcd = fck / 1.4
        fyd = fyk / 1.15
        for fi in possible_fi:
            smax = max(20, fi)

            # compute the flange moment capacity MRd for each layer-count
            a1_vals = {
                1: cnom + fi_str + fi/2,
                2: cnom + fi_str + fi + smax/2,
                3: cnom + fi_str + 1.5*fi + smax
            }
            for layers, a1 in a1_vals.items():
                d   = h - a1
                MRd = beff * hf * fcd * (d - 0.5*hf)

                if MEd < MRd:
                    # positive moment (PT) case
                    func = globals()[f'calc_PT_{layers}r_plus']
                    scenario_type = 'PT'
                else:
                    # negative moment (RZT) case
                    func = globals()[f'calc_RZT_{layers}r_plus']
                    scenario_type = 'RZT'

                As1, As2, cost = func(
                    MEd, beff, bw, h, hf,
                    fi, fi_str, cnom, fcd, fyd, fck
                )

                # Calculate number of rods needed
                num_rods_As1, actual_As1 = calculate_number_of_rods(As1, fi)
                num_rods_As2, actual_As2 = calculate_number_of_rods(As2, fi)

                # Check if rods fit in the section for the given number of layers
                rods_fit = (check_rods_fit(bw, cnom, num_rods_As1, fi, smax, layers) and 
                           check_rods_fit(bw, cnom, num_rods_As2, fi, smax, layers))
                
                if not rods_fit:
                    continue

                # Only consider solutions where rods fit and cost is lower
                if cost < best['cost']:
                    best.update({
                        'cost': cost,
                        'fck': fck,
                        'fi': fi,
                        'type': scenario_type,
                        'layers': layers,
                        'As1': As1,
                        'As2': As2,
                        'num_rods_As1': num_rods_As1,
                        'num_rods_As2': num_rods_As2,
                        'actual_As1': actual_As1,
                        'actual_As2': actual_As2,
                        'fit_check': rods_fit
                    })

    return best

if __name__ == '__main__':
    inputs = {
        'MEd': 600,
        'beff': 1570,
        'bw': 600,
        'h': 300,
        'hf': 300,
        'cnom': 30,
        'fi_str': 8
    }

    possible_fi  = [8, 10, 12, 16, 20, 25, 28, 32]
    possible_fck = [20, 25, 30, 35, 40, 45, 50, 55]

    best = find_optimal_scenario(inputs)
    print("Cheapest design:")
    print(f"  concrete fck = C{best['fck']}")
    print(f"  bar ∅ = {best['fi']} mm")
    print(f"  moment region = {best['type']} with {best['layers']} layer(s)")
    print(f"  Required As1 = {best['As1']:.1f} mm², As2 = {best['As2']:.1f} mm²")
    print(f"  Number of rods: As1 = {best['num_rods_As1']}, As2 = {best['num_rods_As2']}")
    print(f"  Actual provided: As1 = {best['actual_As1']:.1f} mm², As2 = {best['actual_As2']:.1f} mm²")
    print(f"  Do rods fit in section? {'Yes' if best['fit_check'] else 'No'}")
    print(f"  total cost = {best['cost']:.2f} zł")
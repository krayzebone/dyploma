import math

inputs = {
    'MEd': 700,
    'beff': 1570,
    'bw': 1400,
    'h':300,
    'hf': 300,
    'cnom': 30,
    'fi_str': 8
}

MEd = inputs['MEd']
beff = inputs['beff']
bw = inputs['bw']
h = inputs['h']
hf = inputs['hf']
cnom = inputs['cnom']
fi_str = inputs['fi_str']

fi = 32
fck = 25
fyk = 500
smax = max(20, fi)

fcd = fck / 1.4
fyd = fyk / 1.15

possible_fi = [8, 10, 12, 16, 20, 25, 28, 32]
possible_fck = [20, 25, 30, 35, 40, 45, 50, 55]

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


def calc_PT_1r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd):

    a1 = cnom + fi / 2 + fi_str
    a2 = cnom + fi / 2 + fi_str
    d = h - a1
    
    # Define coefficients for the quadratic equation
    a = (-0.5) * beff * fcd
    b = beff * fcd * d
    c = -MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)

    ksieff = xeff * d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        # Calculate reinforcement area
        As1 = xeff * beff * fcd / fyd
        As2 = 0

    else:
        As2 = (MEd * 1e6 - xeff * beff * fcd * (d - 0.5 * xeff)) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * beff * fcd) / fyd

    cost = calc_cost(beff, bw ,h, hf, fck, As1, As2)

    return As1, As2, cost

def calc_PT_2r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd):

    a1 = cnom + fi_str + fi + smax / 2
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    # Define coefficients for the quadratic equation
    a = (-0.5) * beff * fcd
    b = beff * fcd * d
    c = -MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)

    ksieff = xeff * d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        # Calculate reinforcement area
        As1 = xeff * beff * fcd / fyd
        As2 = 0
    else:
        As2 = (MEd * 1e6 - xeff * beff * fcd * (d - 0.5 * xeff)) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * beff * fcd) / fyd

    cost = calc_cost(beff, bw ,h, hf, fck, As1, As2)

    return As1, As2, cost

def calc_PT_3r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd):

    a1 = cnom + fi_str + fi * 3/2 + smax
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    # Define coefficients for the quadratic equation
    a = (-0.5) * beff * fcd
    b = beff * fcd * d
    c = -MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)

    ksieff = xeff * d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        # Calculate reinforcement area
        As1 = xeff * beff * fcd / fyd
        As2 = 0
    else:
        As2 = (MEd * 1e6 - xeff * beff * fcd * (d - 0.5 * xeff)) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * beff * fcd) / fyd


    cost = calc_cost(beff, bw ,h, hf, fck, As1, As2)

    return As1, As2, cost


def calc_RZT_1r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd):

    a1 = cnom + fi / 2 + fi_str
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    # Define coefficients for the quadratic equation
    a = (-0.5) * bw * fcd
    b = bw * fcd * d
    c = hf * (beff - bw) * fcd * (d - 0.5 * hf) - MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)

    ksieff = xeff * d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        # Calculate reinforcement area
        As1 = (xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        As2 = 0
    else:
        As2 = (-xeff * bw * fcd * (d - 0.5 * xeff) - hf * (beff - bw) * fcd * (d - 0.5 * hf) + MEd * 1e6) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd

    cost = calc_cost(beff, bw ,h, hf, fck, As1, As2)

    return As1, As2, cost

def calc_RZT_2r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd):

    a1 = cnom + fi_str + fi + smax / 2
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    # Define coefficients for the quadratic equation
    a = (-0.5) * bw * fcd
    b = bw * fcd * d
    c = hf * (beff - bw) * fcd * (d - 0.5 * hf) - MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)

    ksieff = xeff * d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        # Calculate reinforcement area
        As1 = (xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        As2 = 0
    else:
        As2 = (-xeff * bw * fcd * (d - 0.5 * xeff) - hf * (beff - bw) * fcd * (d - 0.5 * hf) + MEd * 1e6) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd


    cost = calc_cost(beff, bw ,h, hf, fck, As1, As2)

    return As1, As2, cost

def calc_RZT_3r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd):

    a1 = cnom + fi_str + fi *3/2 + smax
    a2 = cnom + fi / 2 + fi_str
    d = h - a1

    # Define coefficients for the quadratic equation
    a = (-0.5) * bw * fcd
    b = bw * fcd * d
    c = hf * (beff - bw) * fcd * (d - 0.5 * hf) - MEd * 1e6

    # Calculate height of the compression area
    xeff = quadratic_equation(a, b, c, h)

    ksieff = xeff * d
    Es = 200_000
    ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    if ksieff <= ksiefflim:
        # Calculate reinforcement area
        As1 = (xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd
        As2 = 0
    else:
        As2 = (-xeff * bw * fcd * (d - 0.5 * xeff) - hf * (beff - bw) * fcd * (d - 0.5 * hf) + MEd * 1e6) / (fyd * (d - a2))
        As1 = (As2 * fyd + xeff * bw * fcd + hf * (beff - bw) * fcd) / fyd

    cost = calc_cost(beff, bw ,h, hf, fck, As1, As2)

    return As1, As2, cost


if MEd > 0:
    # Calculate flange load capacity
    a1_1r = cnom + fi_str + fi / 2
    a1_2r = cnom + fi_str + fi + smax / 2
    a1_3r = cnom + fi_str + fi * 3/2 + smax 
    a1_list = [a1_1r, a1_2r, a1_3r]

    for a1 in a1_list:

        d = h - a1
        MRd = beff * hf * fcd * (d - 0.5 * hf)

        if MEd < MRd:
            PT_1r_plus = calc_PT_1r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd)
            PT_2r_plus = calc_PT_2r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd)
            PT_3r_plus = calc_PT_3r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd)
        
        else:
            RZT_1r_plus = calc_RZT_1r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd)
            RZT_2r_plus = calc_RZT_2r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd)
            RZT_3r_plus = calc_RZT_3r_plus(MEd, beff, bw , h, hf, fi, fi_str, cnom, fcd, fyd)






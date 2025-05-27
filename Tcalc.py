import math

def quadratic_equation(a: float, b: float, c: float, limit: float) -> float:
    """Solve quadratic equation and return solution within (0, limit) if exists."""
    if a == 0:
        return None
        
    delta = b**2 - 4*a*c

    if delta < 0:
        return None
    elif delta == 0:
        x = -b / (2*a)
        return x if 0 < x < limit else None
    else:
        sqrt_delta = math.sqrt(delta)
        x1 = (-b - sqrt_delta) / (2*a)
        x2 = (-b + sqrt_delta) / (2*a)
        valid_solutions = [x for x in (x1, x2) if 0 < x < limit]
        return min(valid_solutions) if valid_solutions else None

def calc_cost(beff: float, bw: float, h: float, hf: float, fck: float, As1: float, As2: float) -> float:
    concrete_cost_by_class = {
        8: 230, 12: 250, 16: 300, 20: 350, 25: 400, 30: 450, 35: 500, 40: 550, 
        45: 600, 50: 650, 55: 700, 60: 800
    }
    
    steel_cost = (As1 + As2) / 1_000_000 * 7900 * 5  # mm²->m² * density * cost
    concrete_area = ((beff * hf) + (h - hf) * bw) / 1_000_000 - (As1 + As2)/1_000_000
    concrete_cost = concrete_area * concrete_cost_by_class[int(fck)]
    
    return steel_cost + concrete_cost


##########################
### Inputs ###############
##########################

MEd = 1180 * 1e6
beff = 1000
bw = 500
h = 600
hf = 250

fi = 20
fi_str = 8
cnom = 30

fck = 25
fyk = 500

fcd = fck / 1.4
fyd = fyk / 1.15

ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / 200_000)

a1 = cnom + fi / 2 + fi_str

d = h - a1

Ms = bw * (h - hf) * fcd * (d - 0.5 * (h - hf))

aq = -0.5 * bw * fcd 
bq = bw * fcd * d
cq = -MEd
delta = bq**2 - 4 * aq * cq
sqrtd = math.sqrt(delta)
xeff1 = (-bq - sqrtd) / (2 * aq)
xeff2 = (-bq + sqrtd) / (2 * aq)
xeff = min (xeff1, xeff2)
print(f" xeff={xeff}")

if MEd <= Ms:
    typ = "Pozornie_Teowy"
    aq = -0.5 * bw * fcd 
    bq = bw * fcd * d
    cq = -MEd
    xeff = quadratic_equation(aq, bq, cq, h)
    print(f" xeff={xeff}")
    ksieff = xeff / d

    if ksieff <= ksiefflim:
        typ="Pozornie_Teowy_Pojedyńczo_Zbrojony"
        As1 = xeff * bw * fcd / fyd
        As2 = 0
    else:
        typ="Pozornie_Teowy_Podwójnie_Zbrojony"
        xeff = ksiefflim * d
        As2 = (MEd - xeff * bw * fcd * (d - 0.5 * xeff)) / (fyd * (d - a1))
        As1 = (As2 * fyd + xeff * bw * fcd) / fyd

else:
    typ = "Rzeczywiście_Teowy"
    aq = -0.5 * beff * fcd
    bq = beff * fcd * (hf - a1)
    cq = bw * (h-hf) * fcd * (d - 0.5 * (h - hf)) - MEd
    delta = bq**2 - 4 * aq * cq
    sqrtd = math.sqrt(delta)
    xeff1 = (-bq - sqrtd) / (2 * aq)
    xeff2 = (-bq + sqrtd) / (2 * aq)
    print(f" xeff1={xeff1}\n xeff2={xeff2}")
    xeff = min(xeff1, xeff2)
    ksieff = xeff / d

    if ksieff <= ksiefflim:
        typ="Rzeczywiście_Teowy_Pojedyńczo_Zbrojony"
        As1 = (xeff * beff * fcd + bw * (h - hf) * fcd ) / fyd
        As2 = 0
    else:
        typ="Rzeczywiście_Teowy_Podwójnie_Zbrojony"
        xeff = ksiefflim * d
        As2 = (MEd - (bw * (h - hf) * fcd * (d - 0.5 * (h - hf))) - (xeff * beff * fcd * (hf - 0.5 * xeff))) / fyd
        As1 = (xeff * beff * fcd + bw * (h - hf) * fcd + As2 * fyd) / fyd




#print(f" Ms={Ms/1e6}\n typ={typ}\n xeff={xeff}\n As1={As1}\n As2={As2}")


###########################
#Pozornie teowy


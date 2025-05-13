import math
import pandas as pd
import numpy as np
import tqdm

Es  = 200_000   # steel modulus
Ecm = 31_000    # concrete secant modulus

def calc_capacity(beff: float, 
                 bw: float, 
                 h: float, 
                 hf: float, 
                 fck: float, 
                 fyk: float, 
                 fi: float, 
                 fistr: float,
                 cnom: float,
                 n1: float,
                 n2: float,) -> float:
    
    # Input validation
    if any(v <= 0 for v in [beff, bw, h, hf, fck, fyk, fi]) or any(v < 0 for v in [n1, n2]):
        return float('nan')
    
    Es  = 200_000   # steel modulus
    Ecm = 31_000    # concrete secant modulus

    try:
        a1 = cnom + fi / 2 + fistr
        a2 = cnom + fi / 2 + fistr
        d = h - a1
        
        # Validate effective depth
        if d <= 0:
            return float('nan')

        fcd = fck / 1.4
        fyd = fyk / 1.15
        ksiefflim = 0.8 * (0.0035/(0.0035 + fyd / Es))

        As1 = n1 * math.pi * fi**2 / 4
        As2 = n2 * math.pi * fi**2 / 4

        # Case 1: Only tension reinforcement
        if n1 > 0 and n2 == 0:
            Fc = hf * beff * fcd
            Fs = As1 * fyd

            if Fc >= Fs:  # Rectangular behavior
                xeff = As1 * fyd / (beff * fcd)
                ksieff = xeff / d
                if ksieff > ksiefflim:
                    xeff = ksiefflim * d
                return xeff * beff * fcd * (d - 0.5 * xeff)
            else:  # T-shaped behavior
                xeff = (As1 * fyd - hf * (beff - bw) * fcd) / (bw * fcd)
                ksieff = xeff / d
                if ksieff > ksiefflim:
                    xeff = ksiefflim * d
                return (hf * (beff - bw) * fcd * (d - 0.5 * hf) + xeff * bw * fcd * (d - 0.5 * xeff))

        # Case 2: Both tension and compression reinforcement
        elif n1 > 0 and n2 > 0:
            Fc = hf * beff * fcd
            Fs1 = As1 * fyd
            Fs2 = As2 * fyd
            F = Fc + Fs2

            if F >= Fs1:  # Neutral axis in flange
                xeff = (As1 * fyd - As2 * fyd) / (beff * fcd)
                ksieff = xeff / d
                if ksieff > ksiefflim:
                    xeff = ksiefflim * d
                
                if xeff >= 2 * a2 and xeff <= hf:
                    return xeff * beff * fcd * (d - 0.5 * xeff) + As2 * fyd * (d - a2)
                elif xeff < 2 * a2:
                    return As1 * fyd * (d - a2)
            
            else:  # Neutral axis in web
                xeff = (As1 * fyd - As2 * fyd - hf * (beff - bw) * fcd) / (bw * fcd)
                ksieff = xeff / d
                if ksieff > ksiefflim:
                    xeff = ksiefflim * d
                
                if xeff >= 2 * a2 and xeff <= h:
                    return (As2 * fyd * (d - a2) + hf * (beff - bw) * fcd * (d - 0.5 * hf) + xeff * bw * fcd * (d - 0.5 * xeff))
                else:
                    return As1 * fyd * (d - a2)

    except Exception as e:
        print(f"Error in calc_capacity: {e}")
    
    return float('nan')  # Return NaN for all invalid cases


def calc_creep(beff, bw, h, hf, fck):

    f_cm_map = {16: 20, 20: 28, 25: 33, 30: 38, 35: 43, 40: 48, 45: 53, 50: 58, 55: 63, 60: 68}

    t0 = 28

    RH = 70
    
    Ac = beff * hf + bw * (h - hf)

    u = beff + 2 * (h - hf) * bw

    h0 = 2 * Ac / u

    alpha1 = (35 / f_cm_map[fck])**0.7
    alpha2 = (35 / f_cm_map[fck])**0.2
    alpha3 = (35 / f_cm_map[fck])**0.5

    if fck <= 35:
        fiRH = 1 + (1 - RH / 100) / (0.1 * h0**(1/3))
    
    else:
        fiRH = 1 + (1 - RH / 100) / (0.1 * h0**(1/3) * alpha1) * alpha2
    
    Bt0 = 1 / (0.1 + t0**0.2)
    Bfcm = 16.8 / math.sqrt(f_cm_map[fck])

    fi0 = fiRH * Bt0 * Bfcm

    return fi0


def calc_crack(MEqp: float, 
               beff: float, 
               bw: float, 
               h: float, 
               hf: float, 
               fck: float, 
               fyk: float,
               fi: float,
               fistr: float,
               cnom: float,
               As1: float,
               As2: float,
               ):
    
    a1 = cnom + fi/2 + fistr
    a2 = cnom + fi/2 + fistr
    d = h - a1

    fi0 = calc_creep(beff, bw, h, hf, fck)
    Eceff = Ecm / (1 + fi0)
    acs = Es / Eceff

    # Neutral axis calculation for uncracked section
    Acs = beff * hf + bw * (h - hf) + acs * (As1 + As2)
    Scs = beff * hf * (h - 0.5 * hf) + (h - hf) * bw * 0.5 * (h - hf) + acs * (As1 * d + As2 * a2)
    xI = Scs / Acs
    
    JI = ((beff - bw) * hf**3 / 12 + (beff - bw) * hf * (h - xI - 0.5 * hf)**2 + 
          bw * h**3 / 12 + bw * h * (0.5 * h - xI)**2 + 
          acs * (As1 * (d - xI)**2 + As2 * (xI - a2)**2))

    Wcs = JI / (h - xI) 
    fctm = 0.3 * fck**(2/3)  # Note: Changed from 0.3*fck^0.3 to match Eurocode
    Mcr = fctm * Wcs

    # Neutral axis calculation for cracked section
    discriminant = acs * (2 * beff * As2 * a2 + 2 * beff * As1 * d + acs * (As1 + As2)**2)
    
    # Handle negative discriminant (invalid case)
    if discriminant < 0:
        return float('nan'), float('nan')  # Return NaN values to be filtered out later

    sqrt_discriminant = math.sqrt(discriminant)
    xII1 = (- (sqrt_discriminant + acs * (As1 + As2))) / beff
    xII2 = (sqrt_discriminant - acs * (As1 + As2)) / beff

    # Choose valid root (must be real number between 0 and h)
    xII = None
    for candidate in [xII1, xII2]:
        if isinstance(candidate, complex):  # Skip complex roots
            continue
        if 0 <= candidate <= h:
            xII = candidate
            break

    if xII is None:
        return float('nan'), float('nan')
        
    # Moment of inertia for cracked section
    if xII <= hf:
        JII = beff * xII**3 / 3 + acs * (As1 * (d - xII)**2 + As2 * (xII - a2)**2)
    else:
        JII = beff * hf**3 / 12 + beff * hf * (xII - hf/2)**2 + bw * (xII - hf)**3 / 3 + acs * (As1 * (d - xII)**2 + As2 * (xII - a2)**2)

    # Crack width calculation
    kt = 0.4
    sigmas = (acs * MEqp / JII) * (d - xII)
    Aceff = bw * min(2.5 * (h - d), (h - xI) / 3)
    roeff = As1 / Aceff
    

    k1 = 0.8
    k2 = 0.5
    k3 = 3.4
    k4 = 0.425

    srmax = k3 * cnom + k1 * k2 * k4 * fi / roeff
    depsilon = max((sigmas - kt * fctm / roeff * (1 + acs * roeff)) / Es, 
                   0.6 * sigmas / Es)
    Wk = srmax * depsilon

    return Mcr, Wk


MEd = 1050 * 1e6
MEqp = 1500 * 1e6

beff = 3000
bw = 800
h = 1200
hf=250

fi = 10
fistr = 8
cnom = 30

n1=27
n2=0

As1 = n1 * math.pi * fi**2 / 4
As2 = n1 * math.pi * fi**2 / 4

fck = 16
fyk = 500
Es = 200_000
Ecm = 31_000

MRd=calc_capacity(beff, 
                 bw, 
                 h, 
                 hf, 
                 fck, 
                 fyk, 
                 fi, 
                 fistr,
                 cnom,
                 n1,
                 n2,)

Mcr, Wk = calc_crack(MEqp, 
               beff, 
               bw, 
               h, 
               hf, 
               fck, 
               fyk,
               fi,
               fistr,
               cnom,
               As1,
               As2,
               )

print(f" MRd={MRd / 1e6}")
print(f" Mcr={Mcr}")
print(f" Wk={Wk}")
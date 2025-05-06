import math

def crack(MEqp, beff, bw, h, hf, fi, fck, As1, As2):

    # Material constants
    Es = 200_000       # MPa (steel)
    Ecm = 31_000       # MPa (concrete)
    RH = 70             # relative humidity in %
    fcm_map = {16: 20, 20: 28, 25: 33, 30: 38, 35: 43, 40: 48, 45: 53, 50: 58, 55: 63, 60: 68}
    t0 = 28            # days, age of concrete at loading
    t   = 8             # days
    fctm = 0.3 * fck**(2/3)
    c_nom = 38
    fyk = 500

    u = beff + 2 * (h - hf)
    Ac = beff * hf + bw * (h - hf)
    h0 = 2 * Ac / u

    # Calculate fiRH
    alpha_1 = (35 / int(fcm_map[fck]))**0.7
    alpha_2 = (35 / int(fcm_map[fck]))**0.2
    alpha_3 = (35 / int(fcm_map[fck]))**0.5
    alpha   = 0  # for cement N

    if fck <= 35:
        fi_RH = 1 + ((1 - RH / 100) / (0.1 * h0**(1/3)))
    else:
        fi_RH = (1 + (1 - RH/100) / (0.1*h0**(1/3)*alpha_1)) * alpha_2

    Beta_t0  = 1 / (0.1 + t0**0.2)
    Beta_fcm = 16.8 / math.sqrt(fcm_map[fck])
    #fi_0 = fi_RH * Beta_fcm * Beta_t0
    fi_0 = 1.871

    # Effective depth
    a1 = c_nom + fi / 2
    d = h - a1

    if d <= 0:
        return None
    
    E_c_eff = Ecm / (1 + fi_0)  # or E_cm / (1 + fi_0) if you want to include creep
    alpha_cs = Es / E_c_eff

    # Pole przekroju betonu
    Ac = beff * hf + bw * (h - hf)
    print(Ac)

    # Obwód części przekroju wystawionej na wysychanie
    u = beff + 2 * (h - hf)

    # Miarodajny wymair przekroju
    h0 = 2 * Ac / u

    # Pole sprowadzonego pola przekroju
    Acs = beff * hf + bw * (h - hf) + (As1 + As2) * alpha_cs

    # Moment statyczny przekroju
    Scs = beff * hf * (h - 0.5 * hf) + (h - hf) * bw * 0.5 * (h - hf)

    # Odległość osi centralnej przekroju od osi przyjętej
    xI = Scs / Acs

    I_I = ((beff - bw) * hf**3) / 12 + (beff - bw) * hf *(h - xI - 0.5 * hf)**2 
    + (bw * h**3) / 12 + bw * h * (0.5 * h - xI)
    + alpha_cs * (As1 * (d - xI)**2 + As2 * (xI - a1)**2)

    # Wskaźnik na zginanie przekroju
    Wcs = I_I / (h - xI)

    # Moment krytyczny
    Mcr = fctm * Wcs

    return Mcr / 1e6

MEqp = 117.789
beff = 1636
bw = 300
h = 500
hf = 150
fi = 20
fck = 25
As1 = 942.5
As2 = 0
    
result = crack(MEqp, beff, bw, h, hf, fi, fck, As1, As2)

print(result)
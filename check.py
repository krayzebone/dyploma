import math

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

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


def predict_sectionn1(MEqp: float, beff: float, bw:float, h: float, hf: float, fck: float, fi: float, cnom: float, As1: float, As2: float):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"neural_networks\Tsectionn1\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn1\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn1\models\Mcr_model\scaler_y.pkl"
        },
        'MRd': {
            'model': r"neural_networks\Tsectionn1\models\MRd_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn1\models\MRd_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn1\models\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"neural_networks\Tsectionn1\models\Wk_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn1\models\Wk_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn1\models\Wk_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr': ["beff", "bw", "h", "hf", "fi", "fck", "ro1"],
        'MRd': ["beff", "bw", "h", "hf", "cnom", "d", "fi", "fck", "ro1"],
        'Wk': ["MEqp", "beff", "bw", "h", "hf", 'cnom', 'd', "fi", "fck", "ro1"]
    }
    
    # Calculate derived parameters
    d = h - cnom - fi / 2 - 8
    ro1 = As1 / ((beff * hf) + (bw * (h-hf))) if ((beff * hf) + (bw * (h-hf))) > 0 else 0
    print(ro1)
    
    feature_values = {
        'MEqp': float(MEqp),
        'beff': float(beff),
        'bw': float(bw),
        'h': float(h),
        'hf': float(hf),
        'd': float(d),
        'cnom': float(cnom),
        'fi': float(fi),
        'fck': float(fck),
        'ro1': float(ro1),
    }
    
    results = {}
    for model_name in ['Mcr', 'MRd', 'Wk', 'Cost']:
        try:
            model_info = MODEL_PATHS[model_name]
            model = tf.keras.models.load_model(model_info['model'], compile=False)
            X_scaler = joblib.load(model_info['scaler_X'])
            y_scaler = joblib.load(model_info['scaler_y'])

            X_values = [feature_values[f] for f in MODEL_FEATURES[model_name]]
            X = pd.DataFrame([X_values], columns=MODEL_FEATURES[model_name])

            X_scaled = X_scaler.transform(np.log(X + 1e-8))
            pred_scaled = model.predict(X_scaled)
            pred = np.exp(y_scaler.inverse_transform(pred_scaled))[0][0]
            results[model_name] = pred
        except Exception as e:
            print(f"⚠️ Error in {model_name}: {e}")
            results[model_name] = None
    
    return results

def predict_sectionn2(MEqp: float, beff: float, bw:float, h: float, hf: float, fck: float, fi: float, cnom: float, As1: float, As2: float):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"neural_networks\Tsectionn2\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn2\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn2\models\Mcr_model\scaler_y.pkl"
        },
        'MRd': {
            'model': r"neural_networks\Tsectionn2\models\MRd_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn2\models\MRd_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn2\models\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"neural_networks\Tsectionn2\models\Wk_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn2\models\Wk_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn2\models\Wk_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr': ["beff", "bw", "h", "hf", "fi", "fck", "ro1", "ro2"],
        'MRd': ["beff", "bw", "h", "hf", "cnom", "d", "fi", "fck", "ro1", "ro2"],
        'Wk': ["MEqp", "beff", "bw", "h", "hf", 'cnom', 'd', "fi", "fck", "ro1", "ro2"]
    }
    
    # Calculate derived parameters
    d = h - cnom - fi / 2 - 8
    ro1 = As1 / ((beff * hf) + (bw * (h-hf))) if ((beff * hf) + (bw * (h-hf))) > 0 else 0
    ro2 = As2 / ((beff * hf) + (bw * (h-hf))) if ((beff * hf) + (bw * (h-hf))) > 0 else 0

    print(ro1, ro2)
    
    feature_values = {
        'MEqp': float(MEqp),
        'beff': float(beff),
        'bw': float(bw),
        'h': float(h),
        'hf': float(hf),
        'd': float(d),
        'cnom': float(cnom),
        'fi': float(fi),
        'fck': float(fck),
        'ro1': float(ro1),
        'ro2': float(ro2)
    }
    
    results = {}
    for model_name in ['Mcr', 'MRd', 'Wk', 'Cost']:
        try:
            model_info = MODEL_PATHS[model_name]
            model = tf.keras.models.load_model(model_info['model'], compile=False)
            X_scaler = joblib.load(model_info['scaler_X'])
            y_scaler = joblib.load(model_info['scaler_y'])

            X_values = [feature_values[f] for f in MODEL_FEATURES[model_name]]
            X = pd.DataFrame([X_values], columns=MODEL_FEATURES[model_name])

            X_scaled = X_scaler.transform(np.log(X + 1e-8))
            pred_scaled = model.predict(X_scaled)
            pred = np.exp(y_scaler.inverse_transform(pred_scaled))[0][0]
            results[model_name] = pred
        except Exception as e:
            print(f"⚠️ Error in {model_name}: {e}")
            results[model_name] = None
    
    return results


MEd = 550 * 1e6
MEqp = 600 * 1e6

beff = 900
bw = 800
h = 300
hf=100

fi = 20
fistr = 8
cnom = 30

n1=22
n2=5

As1 = n1 * math.pi * fi**2 / 4
As2 = n2 * math.pi * fi**2 / 4

fck = 30
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

if n2>0:
    pred2 = predict_sectionn2(MEqp/1e6, beff, bw, h, hf, fck, fi, cnom, As1, As2)
    Mcr_pred = pred2.get('Mcr')
    MRd_pred = pred2.get('MRd')
    Wk_pred  = pred2.get('Wk')
    Cost_pred = pred2.get('Cost')  # Might be None if not used
else:
    pred = predict_sectionn1(MEqp/1e6, beff, bw, h, hf, fck, fi, cnom, As1, As2)

    Mcr_pred = pred.get('Mcr')
    MRd_pred = pred.get('MRd')
    Wk_pred  = pred.get('Wk')
    Cost_pred = pred.get('Cost')  # Might be None if not used

d = h - cnom - fi / 2 - 8
ro1 = As1 / ((beff * hf) + (bw * (h-hf))) if ((beff * hf) + (bw * (h-hf))) > 0 else 0
ro2 = As2 / ((beff * hf) + (bw * (h-hf))) if ((beff * hf) + (bw * (h-hf))) > 0 else 0



print(f" MRd_real={MRd / 1e6}")
print(f" Mcr_real={Mcr / 1e6}")
print(f" Wk_real={Wk}")

print(f" MRd_pred={MRd_pred}")
print(f" Mcr_pred={Mcr_pred}")
print(f" Wk_pred={Wk_pred}")


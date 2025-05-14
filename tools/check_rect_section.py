import math

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

Es  = 200_000   # steel modulus
Ecm = 31_000    # concrete secant modulus

def calc_capacity(b: float,  
                 h: float, 
                 fck: float, 
                 fyk: float, 
                 fi: float, 
                 fistr: float,
                 cnom: float,
                 n1: float,
                 n2: float,) -> float:
    
    a1 = cnom + fi / 2 + fistr
    d = h - a1

    fcd = fck / 1.4
    fyd = fyk / 1.15
    Es = 200_000
    ksi_eff_lim = 0.8 * 0.0035 / (0.0035 + fyd / Es)

    As1 = n1 * math.pi * fi**2 / 4
    As2 = n2 * math.pi * fi**2 / 4

    if n1 > 0 and n2 == 0:
        xeff = (As1 * fyd) / (b * fcd)
        ksieff = xeff / d
        if ksieff > ksi_eff_lim:
            xeff = ksi_eff_lim * d
            MRd = xeff * b * fcd * (d - 0.5 * xeff)
        else:
            MRd = xeff * b * fcd * (d - 0.5 * xeff)
    else:
        xeff = (As1 * fyd - As2 * fyd) / (b * fcd)
        ksieff = xeff / d
        if ksieff > ksi_eff_lim:
            xeff = ksi_eff_lim * d
            if xeff < 2 * a1:
                MRd = (xeff * b * fcd * (d - 0.5 * xeff) + As2 * fyd * (d - a1))
            else:
                MRd = (As1 * fyd * (d - a1))
        else:
            if xeff < 2 * a1:
                MRd = (xeff * b * fcd * (d - 0.5 * xeff) + As2 * fyd * (d - a1))
            else:
                MRd = (As1 * fyd * (d - a1))

    return MRd

def calc_creep(b, h, fck):

    f_cm_map = {16: 20, 20: 28, 25: 33, 30: 38, 35: 43, 40: 48, 45: 53, 50: 58, 55: 63, 60: 68}

    t0 = 28

    RH = 70
    
    A_c = b * h
    u   = 2*(b + h)
    h_0 = 2*A_c / u

    alpha_1 = (35 / f_cm_map[fck])**0.7
    alpha_2 = (35 / f_cm_map[fck])**0.2
    alpha_3 = (35 / f_cm_map[fck])**0.5
    alpha   = 0  # for cement N

    if fck <= 35:
        fi_RH = 1 + ((1 - RH / 100) / (0.1 * h_0**(1/3)))
    else:
        fi_RH = (1 + (1 - RH/100) / (0.1*h_0**(1/3)*alpha_1)) * alpha_2

    Beta_t0  = 1 / (0.1 + t0**0.2)
    Beta_fcm = 16.8 / math.sqrt(f_cm_map[fck])
    fi_0 = fi_RH * Beta_fcm * Beta_t0

    return fi_0


def calc_crack(MEqp, b, h, f_ck, fi_gl, fistr, c_nom, A_s1, A_s2):

    E_cm = 31000
    E_s = 200_000
    a_1 = c_nom + fi_gl / 2
    d = h - a_1
    A_c = b * h
    f_ctm = 0.3 * f_ck**(2/3)

    ro_s1 = A_s1 / (b * h)
    ro_s2 = A_s2 / (b * h)
   
    # 4.2 Stage I analysis
    E_c_eff = E_cm   # or E_cm / (1 + fi_0) if you want to include creep
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

    ro_p_eff = (A_s1 + A_s2) / A_c_eff

    s_r_max = k_3*c_nom + k_1*k_2*k_4*fi_gl / ro_p_eff
    w_k = delta_sigma * epsilon_cr

    return M_cr, w_k
 

def predict_sectionn1(MEqp: float, b: float, h: float, fck: float, fi: float, cnom: float, As1: float):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"neural_networks\rect_section_n1\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_y.pkl"
        },
        'MRd': {
            'model': r"neural_networks\rect_section_n1\models\MRd_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\MRd_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"neural_networks\rect_section_n1\models\Wk_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\Wk_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\Wk_model\scaler_y.pkl"
        },
        'Cost': {
            'model': r"neural_networks\rect_section_n1\models\Cost_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\Cost_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\Cost_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr':  ["b", "h", "d", "fi", "fck", "ro1"],
        'MRd':  ["b", "h", "d", "fi", "fck", "ro1"],
        'Wk':   ["MEqp", "b", "h", "d", "fi", "fck", "ro1"],
        'Cost': ["b", "h", "d", "fi", "fck", "ro1"]
    }
    
    # Calculate derived parameters
    d = h - cnom - fi / 2 - 8
    ro1 = As1 / (b * h)
    
    feature_values = {
        'MEqp': float(MEqp),
        'b': float(b),
        'h': float(h),
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

def predict_sectionn2(MEqp: float, b: float, h: float, fck: float, fi: float, cnom: float, As1: float, As2: float):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"neural_networks\rect_section_n2\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n2\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n2\models\Mcr_model\scaler_y.pkl"
        },
        'MRd': {
            'model': r"neural_networks\rect_section_n2\models\MRd_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n2\models\MRd_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n2\models\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"neural_networks\rect_section_n2\models\Wk_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n2\models\Wk_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n2\models\Wk_model\scaler_y.pkl"
        },
        'Cost': {
            'model': r"neural_networks\rect_section_n2\models\Cost_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n2\models\Cost_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n2\models\Cost_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr':  ["b", "h", "d", "fi", "fck", "ro1", "ro2"],
        'MRd':  ["b", "h", "d", "fi", "fck", "ro1", "ro2"],
        'Wk':   ["MEqp", "b", "h", "d", "fi", "fck", "ro1", "ro2"],
        'Cost': ["b", "h", "d", "fi", "fck", "ro1", "ro2"]
    }
    
    # Calculate derived parameters
    d = h - cnom - fi / 2 - 8
    ro1 = As1 / (b * h)
    ro2 = As2 / (b * h)

    
    feature_values = {
        'MEqp': float(MEqp),
        'b': float(b),
        'h': float(h),
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


MEd = 360 * 1e6
MEqp = 300 * 1e6

#200 - Mcr 42.62, MRd 378.04, Wk 0.22, cost 329
#300 - Mcr 42.62, MRd 378.04, Wk 0.5095, cost 329

# opt fi 20, fck 25, n1 23, n2 0, MRd 425.16, Mcr 36.04, Wk 0.2995, Cost 382.52

b=1000
h = 250

fi = 20
fistr = 8
cnom = 30

n1=23
n2=0

As1 = n1 * math.pi * fi**2 / 4
As2 = n2 * math.pi * fi**2 / 4

fck = 25
fyk = 500
Es = 200_000
Ecm = 31_000

MRd=calc_capacity(b, h, fck, fyk, fi, fistr, cnom, n1, n2,)

Mcr, Wk = calc_crack(MEqp, b, h, fck, fi, fistr, cnom, As1, As2,)


if n2>0:
    pred2 = predict_sectionn2(MEqp/1e6, b, h, fck, fi, cnom, As1, As2)
    Mcr_pred = pred2.get('Mcr')
    MRd_pred = pred2.get('MRd')
    Wk_pred  = pred2.get('Wk')
    Cost_pred = pred2.get('Cost')  # Might be None if not used
else:
    pred = predict_sectionn1(MEqp/1e6, b, h, fck, fi, cnom, As1)

    Mcr_pred = pred.get('Mcr')
    MRd_pred = pred.get('MRd')
    Wk_pred  = pred.get('Wk')
    Cost_pred = pred.get('Cost')  # Might be None if not used

d = h - cnom - fi / 2 - 8
ro1 = As1 / (b * h)
ro2 = As2 / (b * h)



print(f" MRd_real={MRd / 1e6}")
print(f" Mcr_real={Mcr / 1e6}")
print(f" Wk_real={Wk}")

print(f" MRd_pred={MRd_pred}")
print(f" Mcr_pred={Mcr_pred}")
print(f" Wk_pred={Wk_pred}")


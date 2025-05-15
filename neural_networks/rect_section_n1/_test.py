import math
import tqdm
import pandas as pd
import numpy as np

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
import matplotlib.pyplot as plt
from scipy.stats import boxcox_normmax
from functools import partial
import random


def calculate_section_cost(b: float, h: float, f_ck: float, A_s1: float, A_s2: float) -> float:
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

def predict_section_n1(MEqp: float, b: float, h: float, d:float, fck: float, fi: float, cnom: float, As1: float):
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
    ro1 = As1 / (b * h) if (b * h) > 0 else 0
    
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
            
            # Load scalers - X_scaler is a dictionary, y_scaler is a single scaler
            X_scalers_dict = joblib.load(model_info['scaler_X'])
            y_scaler = joblib.load(model_info['scaler_y'])

            # Prepare input data
            X_values = [feature_values[f] for f in MODEL_FEATURES[model_name]]
            X_df = pd.DataFrame([X_values], columns=MODEL_FEATURES[model_name])
            
            # Apply log transform to each feature (as done in training)
            X_log = np.log(X_df + 1e-8)
            
            # Scale each feature with its respective scaler
            X_scaled = np.zeros_like(X_log)
            for i, feature in enumerate(MODEL_FEATURES[model_name]):
                scaler = X_scalers_dict[feature]  # Get the specific scaler for this feature
                X_scaled[:, i] = scaler.transform(X_log[feature].values.reshape(-1, 1)).flatten()
            
            # Make prediction
            pred_scaled = model.predict(X_scaled)
            
            # Inverse transform the prediction
            pred = np.exp(y_scaler.inverse_transform(pred_scaled))[0][0]
            results[model_name] = pred
            
        except Exception as e:
            print(f"⚠️ Error in {model_name}: {e}")
            results[model_name] = None
    
    return results





MEqp=200*1e6
M_Ed=360*1e6
b=1000
h=250
fi_gl=16
f_ck=35
f_yk=500
n1=26
n2=0
c_nom=30
fi_str=8

# Material constants
E_s = 200_000       # MPa (steel)
E_cm = 31_000       # MPa (concrete)
RH = 70             # relative humidity in %
f_cm_map = {16: 20, 20: 28, 25: 33, 30: 38, 35: 43, 40: 48, 45: 53, 50: 58, 55: 63, 60: 68}
t_0 = 28            # days, age of concrete at loading
t   = 8             # days
f_ctm = 0.3 * f_ck**(2/3)

area_one_bar=math.pi*fi_gl**2/4

A_s1 = n1 * area_one_bar
A_s2 = n2 * area_one_bar
ro_s1 = A_s1 / (b * h)
ro_s2 = A_s2 / (b * h)


# Effective depth
a_1 = c_nom + fi_gl / 2 - fi_str
d = h - a_1
    
f_cd = f_ck / 1.4
f_yd = f_yk / 1.15

x_eff = (A_s1*f_yd) / (b * f_cd)
ksi_eff = x_eff / d
ksi_eff_lim = 0.8 * 0.0035 / (0.0035 + f_yd / E_s)

M_Rd = (x_eff * b * f_cd * (d - 0.5 * x_eff))

    

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
E_c_eff = E_cm / (1 + fi_0)  # or E_cm / (1 + fi_0) if you want to include creep
alpha_cs = E_s / E_c_eff

    # Transformed area
A_cs = A_c + alpha_cs*(A_s1)

    # First moment
S_cs = A_c*(h/2) + alpha_cs*(A_s1*d)

    # Centroid from the bottom
x_I = S_cs / A_cs

    # Moment of inertia (uncracked)
I_I = b*(h**3)/12 + b*h*(h/2 - x_I)**2 + alpha_cs*(A_s1*(d - x_I)**2)

    # Section modulus for tension
W_cs = I_I / (h - x_I)

    # Cracking moment
M_cr = f_ctm * W_cs

    # 4.3 Stage II analysis
ro_s  = ro_s1
delta_ul = a_1 / d
k_lu = (ro_s1) / ro_s if ro_s > 0 else 0

ro_cs = ro_s * alpha_cs
sqrt_arg = ro_cs * (ro_cs + 2*k_lu)

ksi_II = math.sqrt(sqrt_arg) - ro_cs
x_II   = ksi_II * d

    # Moment of inertia (cracked)
I_II = b*(x_II**3)/3 + alpha_cs*(A_s1*(d - x_II)**2)

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


ro_p_eff = (A_s1) / A_c_eff


s_r_max = k_3*c_nom + k_1*k_2*k_4*fi_gl / ro_p_eff
w_k = delta_sigma * epsilon_cr
w_k_max = 0.3

results = predict_section_n1(MEqp / 1e6, b, h, d, f_ck, fi_gl, c_nom, A_s1)

print(f" MRd_real={M_Rd / 1e6:.2f} kNm")
print(f" Mcr_real={M_cr / 1e6:.2f} kNm")
print(f" Wk_real={w_k:.4f} mm")

print("\nPredicted values:")
print(f" MRd_pred={results['MRd']:.2f} kNm")
print(f" Mcr_pred={results['Mcr']:.2f} kNm")
print(f" Wk_pred={results['Wk']:.4f} mm")
print(f" Cost_pred={results['Cost']:.2f} zł")
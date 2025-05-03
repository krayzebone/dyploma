import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import math

def quadratic_equation(a: float, b: float, c: float, limit: float) -> float | None:
    """Solve ax²+bx+c = 0 and return the root that lies in (0, limit)."""
    if a == 0:
        return None

    delta = b**2 - 4 * a * c
    if delta < 0:
        return None

    sqrt_delta = math.sqrt(delta)
    x1 = (-b - sqrt_delta) / (2 * a)
    x2 = (-b + sqrt_delta) / (2 * a)
    valid = [x for x in (x1, x2) if 0 < x < limit]
    return min(valid) if valid else None


def calc_rect_section(MEd, b, h, fck, fi_gl, c_nom):

    # Parametry materiałowe
    fyk = 500
    fcd = fck / 1.4     # wytrzymałość obliczeniowa betonu na ściskanie [MPa]
    fyd = fyk / 1.15    # wytrzymałość obliczeniowa stali na rozciąganie [MPa]
    E_s = 200000.0      # moduł Younga stali [MPa]

    # Wyznaczenie współczynnika względnej wysokości strefy ściskanej
    ksi_eff_lim = 0.8 * 0.0035 / (0.0035 + fyd / E_s)

    # Podstawowe wymiary przekroju
    fi_str = 8
    a_1 = c_nom + fi_gl / 2 + fi_str  
    d = h - a_1

    aq = (-0.5) * b * fcd
    bq = b * fcd * d
    cq = -MEd * 1e6

    xeff = quadratic_equation(aq, bq, cq, h)
    print(f" xeff={xeff}")

    ksieff = xeff / d

    if ksieff > ksi_eff_lim:

        x_eff = ksi_eff_lim * d
        As2 = (- x_eff * b * fcd * (d - 0.5 * x_eff) + MEd * 1e6) / (fyd * (d - a_1))
        As1 = (As2 * fyd + x_eff * b * fcd) / fyd    
        return As1, As2
    
    else:
        As1 = xeff * b * fcd / fyd
        As2 = 0
        return As1, As2


def predict_section(input_dict):
    """
    Predict and print results for a concrete section.
    
    Args:
        input_dict (dict): Must contain:
            MEd, b, h, fck, fi_gl, c_nom, ro1, ro2
    """
    # Calculate derived parameters
    input_dict['d'] = input_dict['h'] - input_dict['c_nom'] - input_dict['fi_gl']/2
    input_dict['fi'] = input_dict['fi_gl']
    
    results = {}
    for model_name in ['Mcr', 'MRd', 'Wk', 'Cost']:
        try:
            # Load model and scalers
            model = tf.keras.models.load_model(MODEL_PATHS[model_name]['model'], compile=False)
            X_scaler = joblib.load(MODEL_PATHS[model_name]['scaler_X'])
            y_scaler = joblib.load(MODEL_PATHS[model_name]['scaler_y'])
            
            # Prepare input
            X = pd.DataFrame([input_dict])[MODEL_FEATURES[model_name]]
            X_scaled = X_scaler.transform(np.log(X + 1e-8))
            
            # Predict and inverse transform
            pred = np.exp(y_scaler.inverse_transform(model.predict(X_scaled)))[0][0]
            results[model_name] = pred
        except Exception as e:
            print(f"⚠️ Error in {model_name}: {str(e)}")
            results[model_name] = None
    
    return results


# Model configuration
MODEL_PATHS = {
    'Mcr': {
        'model': r"nn_models\rect_section\Mcr_model\model.keras",
        'scaler_X': r"nn_models\rect_section\Mcr_model\scaler_X.pkl",
        'scaler_y': r"nn_models\rect_section\Mcr_model\scaler_y.pkl"
    },
    'MRd': {
        'model': r"nn_models\rect_section\MRd_model\model.keras",
        'scaler_X': r"nn_models\rect_section\MRd_model\scaler_X.pkl",
        'scaler_y': r"nn_models\rect_section\MRd_model\scaler_y.pkl"
    },
    'Wk': {
        'model': r"nn_models\rect_section\Wk_model\model.keras",
        'scaler_X': r"nn_models\rect_section\Wk_model\scaler_X.pkl",
        'scaler_y': r"nn_models\rect_section\Wk_model\scaler_y.pkl"
    },
    'Cost': {
        'model': r"nn_models\rect_section\cost_model\model.keras",
        'scaler_X': r"nn_models\rect_section\cost_model\scaler_X.pkl",
        'scaler_y': r"nn_models\rect_section\cost_model\scaler_y.pkl"
    }
}

MODEL_FEATURES = {
    'Mcr': ['b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
    'MRd': ['b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
    'Wk': ['MEd', 'b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
    'Cost': ['MEd', 'b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2']
}


# Input data
input_data = {
    'MEd': 409,
    'b': 400,
    'h': 500,
    'fck': 30,
    'fi_gl': 16,
    'c_nom': 30,
}

# Pass the parameters correctly
As1, As2 = calc_rect_section(
    input_data["MEd"],
    input_data["b"],
    input_data["h"],
    input_data["fck"],
    input_data["fi_gl"],
    input_data["c_nom"]
)

# Compute ro1, ro2
ro1 = As1 / (input_data['b'] * input_data['h'])
ro2 = As2 / (input_data['b'] * input_data['h']) if As2 > 0 else (2 * input_data["fi_gl"]**2 * 3.14159 / 4) / (input_data["b"] * input_data["h"])

input_data['ro1'] = ro1
input_data['ro2'] = ro2
print(ro1,ro2)
print(As1, As2)

# Calculate the number of steel rods using ro1, ro2 (NOT input_data['ro1'], etc.)
n1 = (ro1 * input_data['b'] * input_data['h']) / (input_data['fi_gl']**2 * 3.14159 / 4)
n2 = (ro2 * input_data['b'] * input_data['h']) / (input_data['fi_gl']**2 * 3.14159 / 4)

# Predict MRd, Mcr, Wk and cost for the input_data
predictions = predict_section(input_data)

print(f"Returned predictions: {predictions}")
print(f"Number of rods n1 = {n1}, n2 = {n2}")

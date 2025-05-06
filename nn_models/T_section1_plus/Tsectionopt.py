import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def predict_section1(MEd: float, beff: float, bw: float, h: float, hf: float, fi: float, fck: float, cnom: float, fistr: float, a1: float, d: float):

    MODEL_PATHS = {
        'As1': {
            'model': r"nn_models\T_section1_plus\As1\model.keras",
            'scaler_X': r"nn_models\T_section1_plus\As1\scaler_X.pkl",
            'scaler_y': r"nn_models\T_section1_plus\As1\scaler_y.pkl"
        },

        'cost': {
            'model': r"nn_models\T_section1_plus\cost\model.keras",
            'scaler_X': r"nn_models\T_section1_plus\cost\scaler_X.pkl",
            'scaler_y': r"nn_models\T_section1_plus\cost\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'As1': ['MEd', 'beff', 'bw', 'h', 'hf', 'fi', 'fck', 'cnom', 'fistr', 'a1', 'd'],
        'cost': ['MEd', 'beff', 'bw', 'h', 'hf', 'fi', 'fck', 'cnom', 'fistr', 'a1', 'd']
    }
    
    # Calculate derived parameters
    d = h - cnom - fi / 2  # Effective depth
    
    # Create a dictionary of all possible features
    feature_values = {
        'MEd': MEd,
        'beff': beff,
        'bw': bw,
        'h': h,
        'hf': hf,
        'fi': fi,
        'fck': fck,
        'cnom': cnom,
        'fistr': fistr,
        'a1': a1,
        'd': d,
    }
    
    results = {}
    for model_name in ['As1', 'cost']:
        try:
            # Load model and scalers
            model = tf.keras.models.load_model(MODEL_PATHS[model_name]['model'], compile=False)
            X_scaler = joblib.load(MODEL_PATHS[model_name]['scaler_X'])
            y_scaler = joblib.load(MODEL_PATHS[model_name]['scaler_y'])
            
            # Prepare input - select only the needed features and create a DataFrame row
            X_values = [feature_values[feature] for feature in MODEL_FEATURES[model_name]]
            X = pd.DataFrame([X_values], columns=MODEL_FEATURES[model_name])
            
            # Apply log transform and scale
            X_scaled = X_scaler.transform(np.log(X + 1e-8))
            
            # Predict and inverse transform
            pred_scaled = model.predict(X_scaled)
            pred = np.exp(y_scaler.inverse_transform(pred_scaled))[0][0]
            results[model_name] = pred
        except Exception as e:
            print(f"⚠️ Error in {model_name}: {str(e)}")
            results[model_name] = None
    
    return results


MEd = 1600
beff = 2000
bw = 800
h = 1200
hf = 250
fi = 28
fck = 16
cnom = 30
fistr = 8

a1 = cnom + fistr + fi / 2
d = h - a1

results = predict_section1(MEd, beff, bw, h, hf, fi, fck, cnom, fistr, a1, d)

print(results)

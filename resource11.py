import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from itertools import product

def predict_section_batch(MEd: float, b: float, h: float, fck: float, fi: float, cnom: float, 
                         n1_list: list, n2_list: list):
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
    
    # Prepare batch data
    batch_results = []
    batch_data = []
    
    for n1, n2 in zip(n1_list, n2_list):
        As1 = n1 * (fi**2) * math.pi / 4
        As2 = n2 * (fi**2) * math.pi / 4
        
        # Calculate derived parameters
        d = h - cnom - fi / 2
        ro1 = As1 / (b * d) if (b * d) > 0 else 0
        ro2 = As2 / (b * d) if (b * d) > 0 else 0
        
        feature_values = {
            'MEd': float(MEd),
            'b': float(b),
            'h': float(h),
            'd': float(d),
            'fi': float(fi),
            'fck': float(fck),
            'ro1': float(ro1),
            'ro2': float(ro2),
            'n1': n1,
            'n2': n2
        }
        batch_data.append(feature_values)
    
    # Batch prediction for each model
    for model_name in ['Mcr', 'MRd', 'Wk', 'Cost']:
        try:
            model_info = MODEL_PATHS[model_name]
            model = tf.keras.models.load_model(model_info['model'], compile=False)
            X_scaler = joblib.load(model_info['scaler_X'])
            y_scaler = joblib.load(model_info['scaler_y'])

            # Prepare batch input
            X_values = [[data[f] for f in MODEL_FEATURES[model_name]] for data in batch_data]
            X = pd.DataFrame(X_values, columns=MODEL_FEATURES[model_name])

            X_scaled = X_scaler.transform(np.log(X + 1e-8))
            pred_scaled = model.predict(X_scaled)
            preds = np.exp(y_scaler.inverse_transform(pred_scaled)).flatten()
            
            # Store predictions
            for i, pred in enumerate(preds):
                if model_name not in batch_data[i]:
                    batch_data[i][model_name] = {}
                batch_data[i][model_name] = pred
        except Exception as e:
            print(f"⚠️ Error in {model_name}: {e}")
            for data in batch_data:
                data[model_name] = None
    
    # Calculate cost for each combination
    for data in batch_data:
        As1 = data['n1'] * (fi**2) * math.pi / 4
        As2 = data['n2'] * (fi**2) * math.pi / 4
        data['Cost'] = calculate_section_cost(b, h, fck, As1, As2)
    
    return batch_data

def predict_section(MEd: float, b: float, h: float, fck: float, fi: float, cnom: float, As1: float, As2: float):
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
    
    # Calculate derived parameters
    d = h - cnom - fi / 2
    ro1 = As1 / (b * d) if (b * d) > 0 else 0
    ro2 = As2 / (b * d) if (b * d) > 0 else 0
    
    feature_values = {
        'MEd': float(MEd),
        'b': float(b),
        'h': float(h),
        'd': float(d),
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

MEd = 200
b = 1000
h = 250
cnom = 30
fi = 12
fck = 20
wk_max = 0.3

n1 = 25
n2 = 0

As1 = n1 * (fi**2) * math.pi / 4
As2 = n2 * (fi**2) * math.pi / 4

results = predict_section(MEd, b, h, fck, fi, cnom, As1, As2)

print("Mcr:", results['Mcr'])
print("MRd:", results['MRd'])
print("Wk:", results['Wk'])
print("Cost:", results['Cost'])

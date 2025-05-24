import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import math

def predict_section_batch(input_data: pd.DataFrame, model_name: str):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"nn_models\Tsectionplus\Mcr_model\model.keras",
            'scaler_X': r"nn_models\Tsectionplus\Mcr_model\scaler_X.pkl",
            'scaler_y': r"nn_models\Tsectionplus\Mcr_model\scaler_y.pkl"
        },
        'MRd': {
            'model': r"nn_models\Tsectionplus\MRd_model\model.keras",
            'scaler_X': r"nn_models\Tsectionplus\MRd_model\scaler_X.pkl",
            'scaler_y': r"nn_models\Tsectionplus\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"nn_models\Tsectionplus\Wk_model\model.keras",
            'scaler_X': r"nn_models\Tsectionplus\Wk_model\scaler_X.pkl",
            'scaler_y': r"nn_models\Tsectionplus\Wk_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr': ["beff", "bw", "h", "hf", "fi", "fck", "ro1", "ro2"],
        'MRd': ["beff", "bw", "h", "hf", "fi", 'cnom', 'd', "fck", "ro1", "ro2"],
        'Wk': ["MEqp", "beff", "bw", "h", "hf", 'cnom', 'd', "fi", "fck", "ro1", "ro2"]
    }

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model name: {model_name}")
    
    try:
        model_info = MODEL_PATHS[model_name]
        features = MODEL_FEATURES[model_name]

        missing = set(features) - set(input_data.columns)
        if missing:
            raise ValueError(f"Missing columns for {model_name}: {missing}")

        model = tf.keras.models.load_model(model_info['model'], compile=False)
        X_scaler = joblib.load(model_info['scaler_X'])
        y_scaler = joblib.load(model_info['scaler_y'])

        X = input_data[features].copy()
        epsilon = 1e-6
        if (X <= 0).any().any():
            raise ValueError(f"Non-positive values found in features for log transform in model: {model_name}")
        
        X_log = np.log(X + epsilon)
        X_scaled = X_scaler.transform(X_log)
        pred_scaled = model.predict(X_scaled, batch_size=1024)
        return np.exp(y_scaler.inverse_transform(pred_scaled)).flatten()

    except Exception as e:
        print(f"⚠️ Critical error in {model_name}: {str(e)}")
        raise


def predict_section(MEqp: float, beff: float, bw:float, h: float, hf: float, fck: float, fi: float, cnom: float, As1: float, As2: float):
    MODEL_PATHS = {
        'MRd': {
            'model': r"nn_models\Tsectionplus\MRd_model\model.keras",
            'scaler_X': r"nn_models\Tsectionplus\MRd_model\scaler_X.pkl",
            'scaler_y': r"nn_models\Tsectionplus\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"nn_models\Tsectionplus\Wk_model\model.keras",
            'scaler_X': r"nn_models\Tsectionplus\Wk_model\scaler_X.pkl",
            'scaler_y': r"nn_models\Tsectionplus\Wk_model\scaler_y.pkl"
        },
        'Cost': {
            'model': r"nn_models\Tsectionplus\cost_model\model.keras",
            'scaler_X': r"nn_models\Tsectionplus\cost_model\scaler_X.pkl",
            'scaler_y': r"nn_models\Tsectionplus\cost_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr': ['beff', 'bw', 'h', 'hf', 'fi', 'fck', 'ro1', 'ro2'],
        'MRd': ['beff', 'bw', 'h', 'hf', 'cnom', 'd', 'fi', 'fck', 'ro1', 'ro2'],
        'Wk': ['MEqp', 'beff', 'bw', 'h', 'hf', 'cnom', 'd', 'fi', 'fck', 'ro1', 'ro2'],
        'Cost': ['beff', 'bw', 'h', 'hf', 'fi', 'fck', 'ro1', 'ro2']
    }
    
    # Calculate derived parameters
    d = h - cnom - fi / 2 - 8
    ro1 = As1 / ((beff * hf) + (bw * (h-hf))) if ((beff * hf) + (bw * (h-hf))) > 0 else 0
    ro2 = As2 / ((beff * hf) + (bw * (h-hf))) if ((beff * hf) + (bw * (h-hf))) > 0 else 0
    
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


MEqp=300
beff=900
bw=800
h=300
hf=100
cnom=30
fck=40
fi=28
n1=5
n2=1
d=h-cnom-fi/2-8

As1 = n1 * math.pi * (fi ** 2) / 4
As2 = n2 * math.pi * (fi ** 2) / 4

ro1 = As1 / ((beff * hf) + (bw * (h-hf))) if ((beff * hf) + (bw * (h-hf))) > 0 else 0
ro2 = As2 / ((beff * hf) + (bw * (h-hf))) if ((beff * hf) + (bw * (h-hf))) > 0 else 0

print(f" As1={As1}")
print(f" As2={As2}")

print(f" ro1={ro1}")
print(f" ro2={ro2}")

result = predict_section(MEqp, beff, bw, h, hf, fck, fi, cnom, As1, As2)

print(result)
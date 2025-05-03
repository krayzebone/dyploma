import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

def predict_section(MEd: float, b: float, h: float, fck: float, fi: float, cnom: float, As1: float, As2: float):
    """
    Predict and print results for a concrete section.
    
    Args:
        MEd: Design moment (kNm)
        b: Section width (mm)
        h: Section height (mm)
        fck: Concrete characteristic strength (MPa)
        fi: Rebar diameter (mm)
        cnom: Concrete cover (mm)
        As1: Area of tension reinforcement (mm²)
        As2: Area of compression reinforcement (mm²)
    """
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
    d = h - cnom - fi / 2  # Effective depth
    ro1 = As1 / (b * h)     # Reinforcement ratio (should use d, not h)
    ro2 = As2 / (b * h)      # Reinforcement ratio (should use d, not h)
    
    # Create a dictionary of all possible features
    feature_values = {
        'MEd': MEd,
        'b': b,
        'h': h,
        'd': d,
        'fi': fi,
        'fck': fck,
        'ro1': ro1,
        'ro2': ro2
    }
    
    results = {}
    for model_name in ['Mcr', 'MRd', 'Wk', 'Cost']:
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

# Example usage
MEd = 404
b = 975
h = 348
fi = 32
fck = 30
cnom = 38
As1 = 9 * np.pi * (fi/2)**2  # Area of 9 bars with diameter fi
As2 = 5 * np.pi * (fi/2)**2  # Area of 5 bars with diameter fi

results = predict_section(MEd, b, h, fck, fi, cnom, As1, As2)
print(results)
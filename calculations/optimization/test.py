import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Model configuration
MODEL_PATHS = {
    'Mcr': {
        'model': r"nn_models/nn_models_rect_section/Mcr_model/model.keras",
        'scaler_X': r"nn_models/nn_models_rect_section/Mcr_model/scaler_X.pkl",
        'scaler_y': r"nn_models/nn_models_rect_section/Mcr_model/scaler_y.pkl"
    },
    'MRd': {
        'model': r"nn_models/nn_models_rect_section/MRd_model/model.keras",
        'scaler_X': r"nn_models/nn_models_rect_section/MRd_model/scaler_X.pkl",
        'scaler_y': r"nn_models/nn_models_rect_section/MRd_model/scaler_y.pkl"
    },
    'Wk': {
        'model': r"nn_models/nn_models_rect_section/Wk_model/model.keras",
        'scaler_X': r"nn_models/nn_models_rect_section/Wk_model/scaler_X.pkl",
        'scaler_y': r"nn_models/nn_models_rect_section/Wk_model/scaler_y.pkl"
    },
    'Cost': {
        'model': r"nn_models/nn_models_rect_section/Cost_model/model.keras",
        'scaler_X': r"nn_models/nn_models_rect_section/Cost_model/scaler_X.pkl",
        'scaler_y': r"nn_models/nn_models_rect_section/Cost_model/scaler_y.pkl"
    }
}

MODEL_FEATURES = {
    'Mcr': ['b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
    'MRd': ['b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
    'Wk': ['MEd', 'b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
    'Cost': ['MEd', 'b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2']
}

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
    
    # Print formatted results
    print("\n=== Prediction Results ===")
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if value is not None else f"{key}: Prediction failed")
    
    return results
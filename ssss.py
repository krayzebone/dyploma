import math
import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

def predict_section(MEd: float, b: float, h: float, fck: float, fi: float, cnom: float, As1: float, As2: float ):
    """
    Predict and print results for a concrete section.
    
    Args:
        input_dict (dict): Must contain:
            MEd, b, h, fck, fi_gl, c_nom, ro1, ro2
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
    d = h - cnom - fi / 2
    
    ro1 = As1 / (b * h)
    ro2 = As2 / (b * h)
    
    results = {}
    for model_name in ['Mcr', 'MRd', 'Wk', 'Cost']:
        try:
            # Load model and scalers
            model = tf.keras.models.load_model(MODEL_PATHS[model_name]['model'], compile=False)
            X_scaler = joblib.load(MODEL_PATHS[model_name]['scaler_X'])
            y_scaler = joblib.load(MODEL_PATHS[model_name]['scaler_y'])
            
            # Prepare input
            X = pd.DataFrame([MEd, b, h, d, fi, fck, ro1, ro2 ])[MODEL_FEATURES[model_name]]
            X_scaled = X_scaler.transform(np.log(X + 1e-8))
            
            # Predict and inverse transform
            pred = np.exp(y_scaler.inverse_transform(model.predict(X_scaled)))[0][0]
            results[model_name] = pred
        except Exception as e:
            print(f"⚠️ Error in {model_name}: {str(e)}")
            results[model_name] = None
    
    return results

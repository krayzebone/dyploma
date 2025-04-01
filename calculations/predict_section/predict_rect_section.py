import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

def predict_rect_section_properties(MEd, b, h, d, fi, fck, ro1, ro2):
    """
    Predict beam properties using trained models.
    
    Args:
        MEd: Design moment (kNm)
        b: Beam width (mm)
        h: Beam height (mm)
        d: Effective depth (mm)
        fi: Bar diameter (mm)
        fck: Concrete strength (MPa)
        ro1: Bottom reinforcement ratio
        ro2: Top reinforcement ratio
        
    Returns:
        Dictionary with numpy.float32 predictions for:
        - 'Mcr': Cracking moment
        - 'MRd': Moment capacity
        - 'Wk': Crack width
        - 'Cost': Construction cost
    """
    # Configuration - Model Paths
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

    # Transformation Functions
    def log_transform(x):
        return np.log(x)

    def log_inverse(x):
        return np.exp(x)

    # Fallback transformation configs
    FALLBACK_TRANSFORMATION_CONFIG = {
        'Mcr': {
            'features': {
                'b':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'h':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'd':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'fi':  {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'fck': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'ro1': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'ro2': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8}
            },
            'target': {
                'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8
            }
        },
        'MRd': {
            'features': {
                'b':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'h':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'd':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'fi':  {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'fck': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'ro1': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'ro2': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8}
            },
            'target': {
                'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8
            }
        },
        'Wk': {
            'features': {
                'MEd': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'b':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'h':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'd':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'fi':  {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'fck': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'ro1': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'ro2': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8}
            },
            'target': {
                'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8
            }
        },
        'Cost': {
            'features': {
                'MEd': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'b':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'h':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'd':   {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'fi':  {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'fck': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'ro1': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
                'ro2': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8}
            },
            'target': {
                'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8
            }
        }
    }

    # Features required by each model
    MODEL_FEATURES = {
        'Mcr': ['b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
        'MRd': ['b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
        'Wk': ['MEd', 'b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
        'Cost': ['MEd', 'b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2']
    }

    def load_model_and_scalers(model_name):
        """Load model and scalers for the specified prediction type."""
        paths = MODEL_PATHS[model_name]
        model = tf.keras.models.load_model(paths['model'], compile=False)
        X_scaler = joblib.load(paths['scaler_X'])
        y_scaler = joblib.load(paths['scaler_y'])
        
        transformers_path = os.path.join(os.path.dirname(paths['model']), "transformers_config.pkl")
        if os.path.exists(transformers_path):
            transformation_config = joblib.load(transformers_path)
        else:
            transformation_config = FALLBACK_TRANSFORMATION_CONFIG[model_name]
        
        return model, X_scaler, y_scaler, transformation_config

    def transform_input(input_dict, features, transformation_config, X_scaler):
        """Transform and scale input features for prediction."""
        df = pd.DataFrame([input_dict])
        X_transformed = np.zeros((1, len(features)), dtype=float)
        for i, feature in enumerate(features):
            tcfg = transformation_config['features'][feature]
            epsilon = tcfg.get('epsilon', 0.0)
            X_transformed[0, i] = tcfg['transform'](df[feature].values[0] + epsilon)
        return X_scaler.transform(X_transformed)

    def inverse_transform_output(y_scaled, y_scaler, transformation_config):
        """Invert scaling and transformation on predictions."""
        y_unscaled_transformed = y_scaler.inverse_transform(y_scaled)
        return transformation_config['target']['inverse_transform'](y_unscaled_transformed)

    def predict_with_model(model_name, input_dict):
        """Make prediction using specified model."""
        model, X_scaler, y_scaler, transformation_config = load_model_and_scalers(model_name)
        features = MODEL_FEATURES[model_name]
        
        X_scaled = transform_input(
            input_dict=input_dict,
            features=features,
            transformation_config=transformation_config,
            X_scaler=X_scaler
        )
        
        y_pred_scaled = model.predict(X_scaled)
        prediction = inverse_transform_output(
            y_scaled=y_pred_scaled,
            y_scaler=y_scaler,
            transformation_config=transformation_config
        ).flatten()[0]
        
        return prediction

    # Prepare input dictionary
    input_dict = {
        'MEd': float(MEd),
        'b': float(b),
        'h': float(h),
        'd': float(d),
        'fi': float(fi),
        'fck': float(fck),
        'ro1': float(ro1),
        'ro2': float(ro2)
    }
    
    # Make predictions
    results = {}
    for model_name in ['Mcr', 'MRd', 'Wk', 'Cost']:
        try:
            prediction = predict_with_model(model_name, input_dict)
            results[model_name] = np.float32(prediction)
        except Exception as e:
            print(f"Error predicting {model_name}: {str(e)}")
            results[model_name] = np.float32(np.nan)
    
    return results
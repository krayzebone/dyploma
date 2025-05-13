import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

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

def log_transform(x):
    return np.log(x)

def log_inverse(x):
    return np.exp(x)

# Fallback transformation config
fallback_TRANSFORMATION_CONFIG = {
    'features': {
        'MEd': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'b': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'h': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'd': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'fi': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'fck': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'ro1': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'ro2': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
    },
    'target': {
        'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8
    }
}

def load_model_and_scalers(model_folder):
    """Load a Keras model, plus X_scaler, y_scaler, and transformation config."""
    # Load model
    model_path = os.path.join(model_folder, "model.keras")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load scalers
    scaler_X_path = os.path.join(model_folder, "scaler_X.pkl")
    scaler_y_path = os.path.join(model_folder, "scaler_y.pkl")
    X_scaler = joblib.load(scaler_X_path)
    y_scaler = joblib.load(scaler_y_path)
    
    # Load transformation config if present
    transformers_path = os.path.join(model_folder, "transformers_config.pkl")
    if os.path.exists(transformers_path):
        transformation_config = joblib.load(transformers_path)
    else:
        transformation_config = fallback_TRANSFORMATION_CONFIG
    
    return model, X_scaler, y_scaler, transformation_config

def transform_features(df, features, transformation_config, X_scaler):
    """Transform and scale features for model prediction."""
    X_transformed = np.zeros((len(df), len(features)), dtype=float)
    for i, feature in enumerate(features):
        tcfg = transformation_config['features'][feature]
        epsilon = tcfg.get('epsilon', 0.0)
        X_transformed[:, i] = tcfg['transform'](df[feature].values + epsilon)
    
    X_scaled = X_scaler.transform(X_transformed)
    return X_scaled

def inverse_transform_target(y_scaled, y_scaler, transformation_config):
    """Invert scaling and transformation to get predictions in original scale."""
    y_unscaled_transformed = y_scaler.inverse_transform(y_scaled)
    return transformation_config['target']['inverse_transform'](y_unscaled_transformed)

# Input parameters
MEd = 360
b = 500
h = 250
cnom = 30
fi_str = 8
fi = 16
fck = 35
As1 = 5227.6
As2 = 0

# Calculate d (effective depth)
d = h - cnom - fi_str - fi/2

# Calculate reinforcement ratios
ro1 = As1 / (b * h)
ro2 = As2 / (b * h)

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'MEd': [MEd],
    'b': [b],
    'h': [h],
    'd': [d],
    'fi': [fi],
    'fck': [fck],
    'ro1': [ro1],
    'ro2': [ro2]
})

# Load the model and scalers
MODEL_FOLDER = r"nn_models\rect_section\cost_model"
model, X_scaler, y_scaler, transformation_config = load_model_and_scalers(MODEL_FOLDER)

# Define the features in the same order as during training
features = ['MEd', 'b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2']

# Transform and scale the input features
X_scaled = transform_features(input_data, features, transformation_config, X_scaler)

# Make prediction
y_pred_scaled = model.predict(X_scaled)
predicted_cost = inverse_transform_target(y_pred_scaled, y_scaler, transformation_config)[0][0]

# Calculate actual cost
actual_cost = calculate_section_cost(b, h, fck, As1, As2)

# Print results
print(f"Predicted cost: {predicted_cost:.2f} zł")
print(f"Actual calculated cost: {actual_cost:.2f} zł")
print(f"Difference: {abs(predicted_cost - actual_cost):.2f} zł ({abs(predicted_cost - actual_cost)/actual_cost*100:.1f}%)")
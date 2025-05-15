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

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# Define named functions for transformations (pickle-friendly)
# ============================================

from functools import partial

def box_cox_transform(x, lmbda):
    """
    Box–Cox transform:
      y = (x**λ - 1) / λ      if λ != 0
      y = log(x)             if λ == 0
    Requires x > 0.
    """
    x = np.asarray(x, dtype=float)
    if lmbda == 0:
        return np.log(x)
    return (np.power(x, lmbda) - 1.0) / lmbda

def box_cox_inverse(y, lmbda):
    """
    Inverse Box–Cox:
      x = (λ * y + 1)**(1/λ)  if λ != 0
      x = exp(y)             if λ == 0
    """
    y = np.asarray(y, dtype=float)
    if lmbda == 0:
        return np.exp(y)
    return np.power(y * lmbda + 1.0, 1.0 / lmbda)

def yeo_johnson_transform(x, lmbda):
    """
    Yeo–Johnson transform (handles x >= 0 and x < 0):
      for x >= 0:
        y = ((x + 1)**λ - 1) / λ        if λ != 0
        y = log(x + 1)                 if λ == 0
      for x <  0:
        y = - [(-x + 1)**(2-λ) - 1] / (2-λ)   if λ != 2
        y = - log(-x + 1)                    if λ == 2
    """
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    pos = x >= 0
    neg = ~pos

    # positive part
    if lmbda == 0:
        y[pos] = np.log(x[pos] + 1)
    else:
        y[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda

    # negative part
    if lmbda == 2:
        y[neg] = -np.log(-x[neg] + 1)
    else:
        y[neg] = - (np.power(-x[neg] + 1, 2 - lmbda) - 1) / (2 - lmbda)

    return y

def yeo_johnson_inverse(y, lmbda):
    """
    Inverse Yeo–Johnson:
      for y >= 0:
        x = (λ * y + 1)**(1/λ) - 1         if λ != 0
        x = exp(y) - 1                    if λ == 0
      for y <  0:
        x = 1 - (-(2-λ) * y + 1)**(1/(2-λ))   if λ != 2
        x = 1 - exp(-y)                       if λ == 2
    """
    y = np.asarray(y, dtype=float)
    x = np.empty_like(y)
    pos = y >= 0
    neg = ~pos

    # inverse positive part
    if lmbda == 0:
        x[pos] = np.exp(y[pos]) - 1
    else:
        x[pos] = np.power(lmbda * y[pos] + 1, 1.0 / lmbda) - 1

    # inverse negative part
    if lmbda == 2:
        x[neg] = 1 - np.exp(-y[neg])
    else:
        x[neg] = 1 - np.power(-(2 - lmbda) * y[neg] + 1, 1.0 / (2 - lmbda))

    return x

def no_transform(x):
    return x

def log_transform(x):
    return np.log(x)

def log_inverse(x):
    return np.exp(x)

def sqrt_transform(x):
    return np.sqrt(x)

def sqrt_inverse(x):
    return x**2

# ============================================
# CENTRALIZED CONFIGURATION (MODIFY EVERYTHING HERE)
# ============================================

# 1. Data Configuration
DATA_CONFIG = {
    'filepath': r"neural_networks\rect_section_n2\dataset\dataset_rect_n2.parquet",
    'features': ["b", "h", "d", "cnom", "fi", "fck", "ro1", "ro2"],
    'target': "Cost",
    'test_size': 0.3,
    'random_state': 42
}

from sklearn.preprocessing import PowerTransformer

df = pd.read_parquet(DATA_CONFIG['filepath'])
df = df.iloc[:100000].copy()

x_bx = df['b'].values + 1e-8
bx_lambda = boxcox_normmax(x_bx)

pt = PowerTransformer(method='yeo-johnson', standardize=False)
pt.fit(df[['b']])
yj_lambda = pt.lambdas_[0]

b_boxcox = {
    'transform': partial(box_cox_transform, lmbda=bx_lambda),
    'inverse_transform': partial(box_cox_inverse, lmbda=bx_lambda),
    'scaler': StandardScaler(),
    'epsilon': 1e-8
}

b_yj = {
    'transform': partial(yeo_johnson_transform, lmbda=yj_lambda),
    'inverse_transform': partial(yeo_johnson_inverse, lmbda=yj_lambda),
    'scaler': StandardScaler(),
    'epsilon': 1e-8
}

# 2. Transformation Configuration
# Now using named functions instead of lambdas
TRANSFORMATION_CONFIG = {
    'features': {
        # Format: 'feature_name': {'transform': func, 'inverse_transform': func, 'epsilon': value}
        'b': {'transform': log_transform, 'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-8},
        'h': {'transform': log_transform, 'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-8},
        'd': {'transform': log_transform, 'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-8},
        'cnom': {'transform': log_transform, 'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-8},
        'fi': {'transform': log_transform, 'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-8},
        'fck': {'transform': log_transform, 'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-8},
        'ro1': {'transform': log_transform, 'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-8},
        'ro2': {'transform': log_transform, 'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-8},
    },

    'target': {
        'transform': log_transform, 'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-8
    }
}

#RANSFORMATION_CONFIG['features']['b'] = b_boxcox
#TRANSFORMATION_CONFIG['features']['b']     = b_yj

MODEL_CONFIG = {
    'hidden_layers': [
        {'units': 66, 'activation': 'relu', 'dropout': 0.0025092782108507294},
    ],
    'output_activation': 'linear'
}

TRAINING_CONFIG = {
    'optimizer': Adam(learning_rate=7.919543594369374e-05),
    'loss': 'mse',
    'metrics': ['mse', 'mae'],
    'batch_size': 51,
    'epochs': 200,
    'callbacks': [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=1e-8),
    ]
}

OUTPUT_CONFIG = {
    'save_path': r"neural_networks\rect_section_n2\models\Cost_model",
    'visualization': {
        'max_samples': 100000,
        'histogram_bins': 100
    },
    'save_transformers': True
}

# ============================================
# Data Loading and Preprocessing
# ============================================
def load_and_preprocess_data():
    """Load data with centralized configuration."""
    df = pd.read_parquet(DATA_CONFIG['filepath'])
    df = df.iloc[:100000].copy()

    # Apply feature-specific transformations and scaling
    X_transformed = np.zeros_like(df[DATA_CONFIG['features']].values)
    X_scaled = np.zeros_like(df[DATA_CONFIG['features']].values)
    
    for i, feature in enumerate(DATA_CONFIG['features']):
        transform_config = TRANSFORMATION_CONFIG['features'].get(feature, {'transform', 'inverse_transform', 'scaler', 'epsilon'})
        # Apply transformation
        transformed = transform_config['transform'](df[feature].values + transform_config['epsilon'])
        # Fit and transform the scaler
        scaler = transform_config['scaler']
        X_scaled[:, i] = scaler.fit_transform(transformed.reshape(-1, 1)).flatten()
    
    y = df[DATA_CONFIG['target']].values.reshape(-1, 1)
    
    # Apply target transformation and scaling
    target_config = TRANSFORMATION_CONFIG['target']
    y_transformed = target_config['transform'](y + target_config['epsilon'])
    y_scaled = target_config['scaler'].fit_transform(y_transformed)

    # Train-validation split
    X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
        X_scaled, y_scaled, 
        test_size=DATA_CONFIG['test_size'], 
        random_state=DATA_CONFIG['random_state']
    )
    
    return X_train, X_val, y_train_scaled, y_val_scaled, df, X_scaled

# ============================================
# Inverse Transformation Helpers
# ============================================
def inverse_transform_features(X_scaled):
    """Inverse transform features from scaled to original space."""
    X_original = np.zeros_like(X_scaled)
    for i, feature in enumerate(DATA_CONFIG['features']):
        transform_config = TRANSFORMATION_CONFIG['features'].get(feature, {'inverse_transform', 'scaler'})
        # Inverse scale
        unscaled = transform_config['scaler'].inverse_transform(X_scaled[:, i].reshape(-1, 1)).flatten()
        # Inverse transform
        X_original[:, i] = transform_config['inverse_transform'](unscaled)
    return X_original

def inverse_transform_target(y_scaled):
    """Inverse transform target from scaled to original space."""
    target_config = TRANSFORMATION_CONFIG['target']
    # Inverse scale
    y_transformed = target_config['scaler'].inverse_transform(y_scaled)
    # Inverse transform
    y_original = target_config['inverse_transform'](y_transformed)
    return y_original

# ============================================
# Model Building (Now uses MODEL_CONFIG)
# ============================================
def build_model(input_shape):
    """Build model with centralized configuration."""
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    
    # Add hidden layers from config
    for layer in MODEL_CONFIG['hidden_layers']:
        model.add(Dense(layer['units'], activation=layer['activation']))
        if layer['dropout'] > 0:
            model.add(Dropout(layer['dropout']))
    
    # Output layer
    model.add(Dense(1, activation=MODEL_CONFIG['output_activation']))
    
    # Compile with training config
    model.compile(
        optimizer=TRAINING_CONFIG['optimizer'],
        loss=TRAINING_CONFIG['loss'],
        metrics=TRAINING_CONFIG['metrics']
    )
    
    model.summary()
    return model

# ============================================
# Training (Now uses TRAINING_CONFIG)
# ============================================
def train_model(model, X_train, y_train_scaled, X_val, y_val_scaled):
    """Train with centralized configuration."""
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        callbacks=TRAINING_CONFIG['callbacks'],
        verbose=1
    )
    return history

# ============================================
# Evaluation (Updated for new config)
# ============================================
def evaluate_model(model, X_val, y_val_scaled):
    """Evaluate with centralized configuration."""
    # Evaluation in transformed (scaled) space
    val_loss, val_mse_scaled, val_mae_scaled = model.evaluate(X_val, y_val_scaled, verbose=0)
    print(f"\nMetrics in transformed space:")
    print(f"  - {TRAINING_CONFIG['metrics'][0]}: {val_mse_scaled}")
    print(f"  - {TRAINING_CONFIG['metrics'][1]}: {val_mae_scaled}")

    # Convert predictions from scaled -> transformed -> original
    val_pred_scaled = model.predict(X_val)
    val_pred = inverse_transform_target(val_pred_scaled)

    # Convert ground truth back to real scale as well
    y_val_unscaled = inverse_transform_target(y_val_scaled)

    # Compute real-scale metrics
    mse_unscaled = np.mean((val_pred - y_val_unscaled) ** 2)
    mae_unscaled = np.mean(np.abs(val_pred - y_val_unscaled))
    r2 = 1 - np.sum((y_val_unscaled - val_pred) ** 2) / np.sum((y_val_unscaled - np.mean(y_val_unscaled)) ** 2)

    print("\nReal-Scale Metrics:")
    print("  - MSE:", mse_unscaled)
    print("  - MAE:", mae_unscaled)
    print("  - R²:", r2)

    return val_pred, y_val_unscaled

# ============================================
# Visualization (Updated for new config)
# ============================================
def plot_histograms(predicted_all, actual_all):
    """Plot histograms with centralized configuration."""
    squared_errors = (predicted_all.flatten() - actual_all.flatten()) ** 2
    abs_errors = np.abs(predicted_all.flatten() - actual_all.flatten())

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(squared_errors, bins=OUTPUT_CONFIG['visualization']['histogram_bins'], edgecolor='black')
    plt.title("Histogram of Squared Errors")
    plt.xlabel("Squared Error")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(abs_errors, bins=OUTPUT_CONFIG['visualization']['histogram_bins'], edgecolor='black')
    plt.title("Histogram of Absolute Errors")
    plt.xlabel("Absolute Error")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

def plot_scatter(actual_all, predicted_all):
    """Scatter plot with centralized configuration."""
    n_samples = min(OUTPUT_CONFIG['visualization']['max_samples'], len(actual_all))
    indices = np.random.choice(range(len(actual_all)), size=n_samples, replace=False)

    actual = actual_all[indices]
    predicted = predicted_all[indices]

    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, s=1, alpha=0.5)
    # 1:1 line
    _min, _max = min(actual), max(actual)
    plt.plot([_min, _max], [_min, _max], 'r--')
    plt.title("Predicted vs. Actual Values")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.show()

# ============================================
# Main Execution (Simplified with new config)
# ============================================
def main():
    # 1) Load and preprocess data
    X_train, X_val, y_train_scaled, y_val_scaled, df, X_scaled = load_and_preprocess_data()

    # 2) Build model
    model = build_model(input_shape=X_train.shape[1])

    # 3) Train model
    history = train_model(model, X_train, y_train_scaled, X_val, y_val_scaled)

    # 4) Evaluate model
    val_pred, y_val_unscaled = evaluate_model(model, X_val, y_val_scaled)

    # Generate full predictions on all data
    full_pred_scaled = model.predict(X_scaled)
    full_pred = inverse_transform_target(full_pred_scaled)
    actual_all = df[DATA_CONFIG['target']].values

    # 1) Add predictions and errors to the DataFrame
    df['predicted_Wk'] = full_pred
    df['error'] = df[DATA_CONFIG['target']] - df['predicted_Wk']
    df['abs_error'] = np.abs(df['error'])

    # 2) Filter for absolute error > 50
    large_error_df = df[df['abs_error'] > 50]
    large_error_df.to_csv("large_error_rows.csv", index=False)

    # 3) Display them
    print(f"Found {len(large_error_df)} rows with |actual – predicted| > 50:\n")
    print(large_error_df)

    # 5) Visualizations
    plot_histograms(full_pred, actual_all)
    plot_scatter(actual_all, full_pred)

    # 6) Save outputs
    os.makedirs(OUTPUT_CONFIG['save_path'], exist_ok=True)
    model.save(os.path.join(OUTPUT_CONFIG['save_path'], "model.keras"))

    X_scalers = {feature: TRANSFORMATION_CONFIG['features'][feature]['scaler'] for feature in DATA_CONFIG['features']}
    y_scaler = TRANSFORMATION_CONFIG['target']['scaler']

    # Save them separately
    joblib.dump(X_scalers, os.path.join(OUTPUT_CONFIG['save_path'], "scaler_X.pkl"))
    joblib.dump(y_scaler, os.path.join(OUTPUT_CONFIG['save_path'], "scaler_y.pkl"))
        
    # Save all scalers and transformation config
    joblib.dump(TRANSFORMATION_CONFIG, os.path.join(OUTPUT_CONFIG['save_path'], "transformers_config.pkl"))

    print("\n✅ Model and transformation parameters saved successfully.")

if __name__ == "__main__":
    main()
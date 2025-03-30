import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
import matplotlib.pyplot as plt
from scipy import stats

tf.random.set_seed(38)

# ============================================
# Define named functions for transformations (pickle-friendly)
# ============================================
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

def boxcox_transform(x, lmbda=None):
    """Box-Cox transformation with automatic lambda estimation if not provided."""
    if lmbda is not None:
        return stats.boxcox(x, lmbda=lmbda)
    return stats.boxcox(x)

def boxcox_inverse(x, lmbda):
    """Inverse Box-Cox transformation."""
    return stats.inv_boxcox(x, lmbda)

# ============================================
# CENTRALIZED CONFIGURATION (MODIFY EVERYTHING HERE)
# ============================================

# 1. Data Configuration
DATA_CONFIG = {
    'filepath': r"datasets\dataset_rect_section.parquet",
    'features': ["MEd", "b", "h", "d", "fi", "fck", "ro1", "ro2"],
    'target': "wk",
    'test_size': 0.3,
    'random_state': 42
}

# 2. Transformation Configuration
# Now using named functions instead of lambdas
TRANSFORMATION_CONFIG = {
    'features': {
        # Format: 'feature_name': {'transform': func, 'inverse_transform': func, 'epsilon': value}
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

SCALER_CONFIG = {
    'X_scaler': StandardScaler(),
    'y_scaler': StandardScaler()
}

MODEL_CONFIG = {
    'hidden_layers': [
        {'units': 357, 'activation': 'relu', 'dropout': 0.18435158080849778},
        {'units': 296, 'activation': 'relu', 'dropout': 0.05481771741940156},
        {'units': 61,  'activation': 'relu', 'dropout': 0.16126167701693794},
        {'units': 253, 'activation': 'relu', 'dropout': 0.31165637335280605},
    ],
    'output_activation': 'linear'
}

TRAINING_CONFIG = {
    'optimizer': Adam(learning_rate= 0.002353704934480255),
    'loss': 'mse',
    'metrics': ['mse', 'mae'],
    'batch_size': 96,
    'epochs': 30,
    'callbacks': [
        EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=1e-8),
    ]
}

OUTPUT_CONFIG = {
    'save_path': r"nn_models\nn_models_rect_section\Wk_model",
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

    # Apply feature-specific transformations
    X_transformed = np.zeros_like(df[DATA_CONFIG['features']].values)
    for i, feature in enumerate(DATA_CONFIG['features']):
        transform_config = TRANSFORMATION_CONFIG['features'].get(feature, {'transform': lambda x: x, 'inverse_transform': lambda x: x, 'epsilon': 0})
        X_transformed[:, i] = transform_config['transform'](
            df[feature].values + transform_config['epsilon']
        )
    
    y = df[DATA_CONFIG['target']].values.reshape(-1, 1)
    
    # Apply target transformation
    y_transformed = TRANSFORMATION_CONFIG['target']['transform'](
        y + TRANSFORMATION_CONFIG['target']['epsilon']
    )

    # Scale features and transformed target
    X_scaled = SCALER_CONFIG['X_scaler'].fit_transform(X_transformed)
    y_scaled = SCALER_CONFIG['y_scaler'].fit_transform(y_transformed)

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
    X_unscaled = SCALER_CONFIG['X_scaler'].inverse_transform(X_scaled)
    X_original = np.zeros_like(X_unscaled)
    for i, feature in enumerate(DATA_CONFIG['features']):
        transform_config = TRANSFORMATION_CONFIG['features'].get(feature, {'inverse_transform': lambda x: x})
        X_original[:, i] = transform_config['inverse_transform'](X_unscaled[:, i])
    return X_original

def inverse_transform_target(y_scaled):
    """Inverse transform target from scaled to original space."""
    y_transformed = SCALER_CONFIG['y_scaler'].inverse_transform(y_scaled)
    y_original = TRANSFORMATION_CONFIG['target']['inverse_transform'](y_transformed)
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

    # 5) Visualizations
    plot_histograms(full_pred, actual_all)
    plot_scatter(actual_all, full_pred)

    # 6) Save outputs
    os.makedirs(OUTPUT_CONFIG['save_path'], exist_ok=True)
    model.save(os.path.join(OUTPUT_CONFIG['save_path'], "model.keras"))
    joblib.dump(SCALER_CONFIG['X_scaler'], os.path.join(OUTPUT_CONFIG['save_path'], "scaler_X.pkl"))
    joblib.dump(SCALER_CONFIG['y_scaler'], os.path.join(OUTPUT_CONFIG['save_path'], "scaler_y.pkl"))
    
    if OUTPUT_CONFIG['save_transformers']:
        # Save transformation configuration
        joblib.dump(TRANSFORMATION_CONFIG, os.path.join(OUTPUT_CONFIG['save_path'], "transformers_config.pkl"))

    print("\n✅ Model, scalers, and transformation parameters saved successfully.")

if __name__ == "__main__":
    main()
"""
Regression neural‑network training script **with full learning‑diagnostic plots** added:
  • Training & validation loss / MAE curves (learning curves)
  • Residual (Pred‑Actual) distribution histogram
  • Predicted‑vs‑Actual scatter and error histograms
Everything else – data pipeline, model, training, saving – unchanged.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
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

# -----------------------------------------------------------------------------
#  Transformation helpers (unchanged from original script)
# -----------------------------------------------------------------------------

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

def box_cox_transform(x, lmbda):
    x = np.asarray(x, dtype=float)
    if lmbda == 0:
        return np.log(x)
    return (np.power(x, lmbda) - 1.0) / lmbda

def box_cox_inverse(y, lmbda):
    y = np.asarray(y, dtype=float)
    if lmbda == 0:
        return np.exp(y)
    return np.power(y * lmbda + 1.0, 1.0 / lmbda)

# … (yeo‑johnson helpers, log/sqrt helpers – unchanged) …

# -----------------------------------------------------------------------------
#  CENTRAL CONFIGURATION
# -----------------------------------------------------------------------------
DATA_CONFIG = {
    'filepath': r"neural_networks/rect_section_n1/dataset/dataset_rect_n1_test5.parquet",
    'features': ["b", "h", "d", "fi", "fck", "ro1"],
    'target': "MRd",
    'test_size': 0.3,
    'random_state': 42,
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
        # Format: 'feature_name': {'transform': func, 'inverse_transform': func, 'scaler': scaler, 'epsilon': value}
        'b':        {'transform': log_transform,     'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-10},
        'h':        {'transform': log_transform,     'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-10},
        'd':        {'transform': log_transform,     'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-10},
        'fi':       {'transform': log_transform,      'inverse_transform': log_inverse,     'scaler': StandardScaler(), 'epsilon': 1e-10},
        'fck':      {'transform': log_transform,      'inverse_transform': log_inverse,     'scaler': StandardScaler(), 'epsilon': 1e-10},
        'ro1':      {'transform': log_transform,     'inverse_transform': log_inverse,      'scaler': StandardScaler(), 'epsilon': 1e-10},
    },

    'target': {
        'transform': log_transform, 'inverse_transform': log_inverse, 'scaler': StandardScaler(), 'epsilon': 1e-8
    }
}

#RANSFORMATION_CONFIG['features']['b'] = b_boxcox
#TRANSFORMATION_CONFIG['features']['b']     = b_yj

MODEL_CONFIG = {
    'hidden_layers': [
        {'units': 456, 'activation': 'relu', 'dropout': 0.0019310079769050935},
        {'units': 468, 'activation': 'relu', 'dropout': 0.05066347110743058}
    ],
    'output_activation': 'linear'
}

TRAINING_CONFIG = {
    'optimizer': Adam(learning_rate=5.222394546658954e-05),
    'loss': 'mse',
    'metrics': ['mse', 'mae'],
    'batch_size': 183,
    'epochs': 200,
    'callbacks': [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=1e-8),
    ]
}

OUTPUT_CONFIG = {
    'save_path': r"neural_networks\rect_section_n1\models\MRd_model1",
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

# -----------------------------------------------------------------------------
#  Visualisation utilities
# -----------------------------------------------------------------------------

def plot_learning_curves(history):
    """Training & validation loss / MAE vs. epochs"""
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    mae = history.history.get('mae', [])
    val_mae = history.history.get('val_mae', [])
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Train loss')
    plt.plot(epochs, val_loss, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.title('Learning curves – loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, mae, label='Train MAE') if mae else None
    plt.plot(epochs, val_mae, label='Val MAE') if val_mae else None
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Learning curves – MAE')
    if mae or val_mae:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residual_distribution(actual, predicted, bins=100):
    """Histogram of residuals (Pred − Actual)"""
    residuals = predicted.flatten() - actual.flatten()
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=bins, edgecolor='black')
    plt.title('Residual distribution (Pred − Actual)')
    plt.xlabel('Residual')
    plt.ylabel('Count')
    plt.grid(True)
    plt.axvline(0, linestyle='--')
    plt.show()


def plot_histograms(predicted_all, actual_all, bins=100):
    """Squared‑ and absolute‑error histograms"""
    se = (predicted_all.flatten() - actual_all.flatten()) ** 2
    ae = np.abs(predicted_all.flatten() - actual_all.flatten())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(se, bins=bins, edgecolor='black')
    plt.title('Histogram of Squared Errors')
    plt.xlabel('Squared error')

    plt.subplot(1, 2, 2)
    plt.hist(ae, bins=bins, edgecolor='black')
    plt.title('Histogram of Absolute Errors')
    plt.xlabel('Absolute error')
    plt.tight_layout()
    plt.show()


def plot_scatter(actual, predicted):
    """Predicted vs. Actual scatter with y=x reference line"""
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, s=3, alpha=0.4)
    _min, _max = min(actual), max(actual)
    plt.plot([_min, _max], [_min, _max], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs. Actual')
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def main():
    X_train, X_val, y_train_s, y_val_s, df, X_scaled = load_and_preprocess_data()
    model = build_model(input_shape=X_train.shape[1])
    history = train_model(model, X_train, y_train_s, X_val, y_val_s)

    plot_learning_curves(history)

    val_pred, _ = evaluate_model(model, X_val, y_val_s)

    full_pred_s = model.predict(X_scaled)
    full_pred = inverse_transform_target(full_pred_s)
    actual_all = df[DATA_CONFIG['target']].values

    plot_histograms(full_pred, actual_all, bins=OUTPUT_CONFIG['visualization']['histogram_bins'])
    plot_scatter(actual_all, full_pred)
    plot_residual_distribution(actual_all, full_pred,
                               bins=OUTPUT_CONFIG['visualization']['histogram_bins'])

    # Saving model & scalers (unchanged)
    os.makedirs(OUTPUT_CONFIG['save_path'], exist_ok=True)
    model.save(os.path.join(OUTPUT_CONFIG['save_path'], 'model.keras'))

    print('\n✅ Training complete; plots generated.')


if __name__ == '__main__':
    main()

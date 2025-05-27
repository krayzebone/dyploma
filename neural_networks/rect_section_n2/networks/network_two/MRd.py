# ============================================
# Disable oneDNN optimisations (unchanged)
# ============================================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ============================================
# Core imports (unchanged)
# ============================================
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox_normmax
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# 1.  Transformation helpers  (unchanged)
# ============================================
def box_cox_transform(x, lmbda):
    x = np.asarray(x, dtype=float)
    return np.log(x) if lmbda == 0 else (np.power(x, lmbda) - 1.0) / lmbda

def box_cox_inverse(y, lmbda):
    y = np.asarray(y, dtype=float)
    return np.exp(y) if lmbda == 0 else np.power(y * lmbda + 1.0, 1.0 / lmbda)

def yeo_johnson_transform(x, lmbda):
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    pos, neg = x >= 0, x < 0
    y[pos] = np.log(x[pos] + 1) if lmbda == 0 else (np.power(x[pos] + 1, lmbda) - 1) / lmbda
    y[neg] = -np.log(-x[neg] + 1) if lmbda == 2 else - (np.power(-x[neg] + 1, 2 - lmbda) - 1) / (2 - lmbda)
    return y

def yeo_johnson_inverse(y, lmbda):
    y = np.asarray(y, dtype=float)
    x = np.empty_like(y)
    pos, neg = y >= 0, y < 0
    x[pos] = np.exp(y[pos]) - 1 if lmbda == 0 else np.power(lmbda * y[pos] + 1, 1.0 / lmbda) - 1
    x[neg] = 1 - np.exp(-y[neg]) if lmbda == 2 else 1 - np.power(-(2 - lmbda) * y[neg] + 1, 1.0 / (2 - lmbda))
    return x

def log_transform(x): return np.log(x)

def log_inverse(x):   return np.exp(x)

def no_transform(x): return x

# ============================================
# 2.  Central configuration  (unchanged)
# ============================================
DATA_CONFIG = {
    'filepath': r"neural_networks\rect_section_n2\dataset\dataset_rect_n2.parquet_100k",
    'features': ["b", "h", "d", "fi", "fck", "ro1", "ro2"],
    'target': "MRd",
    'test_size': 0.3,
    'random_state': 42
}

# derive Box-Cox / Y-J lambdas for optional use
df_lambda_probe = pd.read_parquet(DATA_CONFIG['filepath']).iloc[:1000]
bx_lambda = boxcox_normmax(df_lambda_probe['b'].values + 1e-8)
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson', standardize=False).fit(df_lambda_probe[['b']])
yj_lambda = pt.lambdas_[0]

# transformations (unchanged)
TRANSFORMATION_CONFIG = {
    'features': {
        f: {'transform': log_transform, 'inverse_transform': log_inverse,
            'scaler': StandardScaler(), 'epsilon': 1e-10}
        for f in DATA_CONFIG['features']
    },
    'target': {
        'transform': log_transform, 'inverse_transform': log_inverse,
        'scaler': StandardScaler(), 'epsilon': 1e-8
    }
}

MODEL_CONFIG = {
    'hidden_layers': [
        {'units': 490, 'activation': 'relu', 'dropout': 0.111126125826379},
    ],
    'output_activation': 'linear'
}

TRAINING_CONFIG = {
    'optimizer': Adam(learning_rate=0.000860434271563372),
    'loss': 'mse',
    'metrics': ['mse', 'mae'],
    'batch_size': 256,
    'epochs': 300,
    'callbacks': [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-8),
    ]
}

OUTPUT_CONFIG = {
    'save_path': r"neural_networks\rect_section_n2\network_two\models\MRd_model",
    'visualization': {'max_samples': 100_000, 'histogram_bins': 100},
    'save_transformers': True
}

# ============================================
# 3.  Data loading / preprocessing  (unchanged)
# ============================================
def load_and_preprocess_data():
    df = pd.read_parquet(DATA_CONFIG['filepath']).iloc[:100_000].copy()

    X_scaled = np.zeros_like(df[DATA_CONFIG['features']].values)
    for i, feature in enumerate(DATA_CONFIG['features']):
        cfg = TRANSFORMATION_CONFIG['features'][feature]
        transformed = cfg['transform'](df[feature].values + cfg['epsilon'])
        X_scaled[:, i] = cfg['scaler'].fit_transform(transformed[:, None]).ravel()

    y = df[DATA_CONFIG['target']].values[:, None]
    tcfg = TRANSFORMATION_CONFIG['target']
    y_scaled = tcfg['scaler'].fit_transform(tcfg['transform'](y + tcfg['epsilon']))

    X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
        X_scaled, y_scaled,
        test_size=DATA_CONFIG['test_size'],
        random_state=DATA_CONFIG['random_state']
    )
    return X_train, X_val, y_train_scaled, y_val_scaled, df, X_scaled

# ============================================
# 4.  Inverse-transform helpers  (unchanged)
# ============================================
def inverse_transform_features(X_scaled):
    X_orig = np.zeros_like(X_scaled)
    for i, f in enumerate(DATA_CONFIG['features']):
        cfg = TRANSFORMATION_CONFIG['features'][f]
        unscaled = cfg['scaler'].inverse_transform(X_scaled[:, i][:, None]).ravel()
        X_orig[:, i] = cfg['inverse_transform'](unscaled)
    return X_orig

def inverse_transform_target(y_scaled):
    tcfg = TRANSFORMATION_CONFIG['target']
    y_trans = tcfg['scaler'].inverse_transform(y_scaled)
    return tcfg['inverse_transform'](y_trans)

# ============================================
# 5.  Model-building  (unchanged)
# ============================================
def build_model(input_shape):
    m = Sequential([Input(shape=(input_shape,))])
    for layer in MODEL_CONFIG['hidden_layers']:
        m.add(Dense(layer['units'], activation=layer['activation']))
        if layer['dropout'] > 0:
            m.add(Dropout(layer['dropout']))
    m.add(Dense(1, activation=MODEL_CONFIG['output_activation']))
    m.compile(optimizer=TRAINING_CONFIG['optimizer'],
              loss=TRAINING_CONFIG['loss'],
              metrics=TRAINING_CONFIG['metrics'])
    m.summary()
    return m

# ============================================
# 6.  REAL-SCALE metrics callback  (NEW)
# ============================================
class RealScaleMetrics(tf.keras.callbacks.Callback):
    """Tracks MAE & MSE on the validation set in original (non-log) units."""
    def __init__(self, X_val, y_val_scaled):
        super().__init__()
        self.X_val, self.y_val_scaled = X_val, y_val_scaled
        self.val_mae_real, self.val_mse_real = [], []

    def on_epoch_end(self, epoch, logs=None):
        pred_scaled = self.model.predict(self.X_val, verbose=0)
        pred_real  = inverse_transform_target(pred_scaled)
        y_real     = inverse_transform_target(self.y_val_scaled)

        mae_real = np.mean(np.abs(pred_real - y_real))
        mse_real = np.mean((pred_real - y_real) ** 2)

        # store
        self.val_mae_real.append(mae_real)
        self.val_mse_real.append(mse_real)

        # also inject into Keras logs so History keeps them
        if logs is not None:
            logs['val_mae_real'] = mae_real
            logs['val_mse_real'] = mse_real

# ============================================
# 7.  Training  (unchanged params, NEW callback)
# ============================================
def train_model(model, X_train, y_train_scaled, X_val, y_val_scaled):
    real_cb = RealScaleMetrics(X_val, y_val_scaled)
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        callbacks=TRAINING_CONFIG['callbacks'] + [real_cb],
        verbose=1
    )
    return history, real_cb

# ============================================
# 8.  Evaluation (unchanged)
# ============================================
def evaluate_model(model, X_val, y_val_scaled):
    val_pred_scaled = model.predict(X_val, verbose=0)
    val_pred = inverse_transform_target(val_pred_scaled)
    y_val_real = inverse_transform_target(y_val_scaled)

    mse = np.mean((val_pred - y_val_real) ** 2)
    mae = np.mean(np.abs(val_pred - y_val_real))
    r2  = 1 - np.sum((y_val_real - val_pred) ** 2) / np.sum((y_val_real - np.mean(y_val_real)) ** 2)

    print("\nReal-scale metrics on validation set:")
    print(f"  MAE  : {mae:.8f}")
    print(f"  MSE  : {mse:.8f}")
    print(f"  R²   : {r2 :.8f}")
    return val_pred, y_val_real

# ============================================
# 9.  Plotting utilities  (NEW & updated)
# ============================================
def plot_training_curves(history, real_cb):
    """Line plots of training dynamics in *real* units."""
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # (a) Scaled-space loss for reference
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Train MSE (scaled)')
    plt.plot(epochs, history.history['val_loss'], label='Val MSE (scaled)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (scaled)')
    plt.title('Training & Validation Loss (scaled)')
    plt.legend()
    plt.grid(True)

    # (b) Real-scale MAE & RMSE
    val_mae_real = real_cb.val_mae_real
    val_rmse_real = np.sqrt(real_cb.val_mse_real)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_mae_real, label='Val MAE (real)')
    plt.plot(epochs, val_rmse_real, label='Val RMSE (real)')
    plt.xlabel('Epoch')
    plt.ylabel('Error in original units')
    plt.title('Validation Error in Real Scale')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_histograms(pred, actual):
    se = (pred.flatten() - actual.flatten()) ** 2
    ae = np.abs(pred.flatten() - actual.flatten())
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(se, bins=OUTPUT_CONFIG['visualization']['histogram_bins'], edgecolor='black')
    plt.title('Squared Error distribution')
    plt.xlabel('Squared Error'); plt.ylabel('Count')
    plt.subplot(1, 2, 2)
    plt.hist(ae, bins=OUTPUT_CONFIG['visualization']['histogram_bins'], edgecolor='black')
    plt.title('Absolute Error distribution')
    plt.xlabel('Absolute Error'); plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_scatter(actual, pred):
    n = min(OUTPUT_CONFIG['visualization']['max_samples'], len(actual))
    idx = np.random.choice(len(actual), n, replace=False)
    a, p = actual[idx], pred[idx]
    plt.figure(figsize=(6, 6))
    plt.scatter(a, p, s=4, alpha=0.5)
    _min, _max = min(a.min(), p.min()), max(a.max(), p.max())
    plt.plot([_min, _max], [_min, _max], 'r--', linewidth=1)
    plt.xlabel('Actual'); plt.ylabel('Predicted')
    plt.title('Predicted vs Actual (real scale)')
    plt.grid(True)
    plt.show()

# ============================================
# 10.  Main workflow  (rewritten)
# ============================================
def main():
    # (1) data
    X_train, X_val, y_train_scaled, y_val_scaled, df_raw, X_scaled = load_and_preprocess_data()

    # (2) model
    model = build_model(input_shape=X_train.shape[1])

    # (3) training
    history, real_cb = train_model(model, X_train, y_train_scaled, X_val, y_val_scaled)

    # (4) curves
    plot_training_curves(history, real_cb)

    # (5) evaluation
    val_pred, y_val_real = evaluate_model(model, X_val, y_val_scaled)

    # (6) full-dataset predictions (for hist/scatter)
    full_pred_real = inverse_transform_target(model.predict(X_scaled, verbose=0))
    actual_all = df_raw[DATA_CONFIG['target']].values

    # (7) error inspection
    df_raw['predicted_MRd'] = full_pred_real
    df_raw['abs_error'] = np.abs(df_raw['predicted_MRd'] - df_raw[DATA_CONFIG['target']])
    big_err = df_raw[df_raw['abs_error'] > 50]
    big_err.to_csv("large_error_rows.csv", index=False)
    print(f"\n⚠️  {len(big_err)} rows with |error| > 50 written to large_error_rows.csv")

    # (8) visualisations
    plot_histograms(full_pred_real, actual_all)
    plot_scatter(actual_all, full_pred_real)

    # (9) save artefacts
    os.makedirs(OUTPUT_CONFIG['save_path'], exist_ok=True)
    model.save(os.path.join(OUTPUT_CONFIG['save_path'], "model.keras"))

    X_scalers = {f: TRANSFORMATION_CONFIG['features'][f]['scaler'] for f in DATA_CONFIG['features']}
    y_scaler  = TRANSFORMATION_CONFIG['target']['scaler']
    joblib.dump(X_scalers, os.path.join(OUTPUT_CONFIG['save_path'], "scaler_X.pkl"))
    joblib.dump(y_scaler,  os.path.join(OUTPUT_CONFIG['save_path'], "scaler_y.pkl"))
    joblib.dump(TRANSFORMATION_CONFIG, os.path.join(OUTPUT_CONFIG['save_path'], "transformers_config.pkl"))

    print("\n✅ Model, scalers and config saved to", OUTPUT_CONFIG['save_path'])

if __name__ == "__main__":
    main()

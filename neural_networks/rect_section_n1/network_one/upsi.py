"""
Neural‐Network–based design of rectangular reinforced‐concrete sections
===============================================================================
This script automates the prediction of bending resistance (MRd) using a
feed‐forward neural network and compares its performance against a classical
Linear Regression baseline. Results can be exported for integration with
structural‐engineering programs (SOFiSTiK, Autodesk Robot, MIDAS, etc.).
Multi‐criteria optimization and alternative architectures (CNN/LSTM) are
outlined as future extensions.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import boxcox_normmax
from functools import partial

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# ==============================================================================
# Reproducibility
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==============================================================================
# 1.  Helpers for transforms (unchanged)
# ==============================================================================
def box_cox_transform(x, lmbda):
    x = np.asarray(x, dtype=float)
    return np.log(x) if lmbda == 0 else (np.power(x, lmbda) - 1.0) / lmbda

def box_cox_inverse(y, lmbda):
    y = np.asarray(y, dtype=float)
    return np.exp(y) if lmbda == 0 else np.power(y * lmbda + 1.0, 1.0 / lmbda)

def log_transform(x): return np.log(x)
def log_inverse(x):   return np.exp(x)
def no_transform(x): return x

# ==============================================================================
# 2.  Configuration (unchanged paths + sizes; added CV folds)
# ==============================================================================
DATA_CONFIG = {
    'filepath': r"neural_networks\rect_section_n1\dataset\dataset_rect_n1_test5_100k.parquet",
    'features': ["b", "h", "d", "fi", "fck", "ro1"],
    'target': "MRd",
    'test_size': 0.3,
    'random_state': SEED
}
CV_FOLDS = 5

# estimate lambdas
df_probe = pd.read_parquet(DATA_CONFIG['filepath']).iloc[:1000]
bx_lambda = boxcox_normmax(df_probe['b'].values + 1e-8)

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
        {'units': 429, 'activation': 'relu', 'dropout': 0.00083916825453903},
    ],
    'output_activation': 'linear'
}

TRAINING_CONFIG = {
    'optimizer': Adam(learning_rate=0.00645375648436954),
    'loss': 'mse',
    'metrics': ['mse', 'mae'],
    'batch_size': 160,
    'epochs': 300,
    'callbacks': [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-8),
    ]
}

OUTPUT_CONFIG = {
    'save_path': r"neural_networks\rect_section_n1\network_one\models\MRd_modelk_fold",
    'visualization': {'max_samples': 100_000, 'histogram_bins': 100},
    'save_transformers': True,
    'export_csv': r"neural_networks\rect_section_n1\exports\predictions_for_FEA.csv"
}

# ==============================================================================
# 3.  Data loading / preprocessing (unchanged)
# ==============================================================================
def load_and_preprocess_data():
    df = pd.read_parquet(DATA_CONFIG['filepath']).iloc[:100_000].copy()
    X_scaled = np.zeros_like(df[DATA_CONFIG['features']].values)
    for i, feat in enumerate(DATA_CONFIG['features']):
        cfg = TRANSFORMATION_CONFIG['features'][feat]
        t = cfg['transform'](df[feat].values + cfg['epsilon'])
        X_scaled[:, i] = cfg['scaler'].fit_transform(t[:,None]).ravel()
    y = df[DATA_CONFIG['target']].values[:, None]
    tc = TRANSFORMATION_CONFIG['target']
    y_scaled = tc['scaler'].fit_transform(tc['transform'](y + tc['epsilon']))
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled,
        test_size=DATA_CONFIG['test_size'],
        random_state=DATA_CONFIG['random_state']
    )
    return X_train, X_val, y_train, y_val, df, X_scaled

def inverse_transform_target(y_scaled):
    tc = TRANSFORMATION_CONFIG['target']
    y_t = tc['scaler'].inverse_transform(y_scaled)
    return tc['inverse_transform'](y_t)

# ==============================================================================
# 4.  Model‐building & real‐scale metrics callback (unchanged)
# ==============================================================================
class RealScaleMetrics(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val_scaled):
        super().__init__()
        self.X_val, self.y_val_scaled = X_val, y_val_scaled
        self.val_mae_real, self.val_mse_real = [], []
    def on_epoch_end(self, epoch, logs=None):
        pred_s = self.model.predict(self.X_val, verbose=0)
        pred_r = inverse_transform_target(pred_s)
        y_r    = inverse_transform_target(self.y_val_scaled)
        mae = np.mean(np.abs(pred_r - y_r))
        mse = np.mean((pred_r - y_r)**2)
        self.val_mae_real.append(mae)
        self.val_mse_real.append(mse)
        if logs is not None:
            logs['val_mae_real'], logs['val_mse_real'] = mae, mse

def build_model(input_shape run_eagerly=False):
    m = Sequential([Input(shape=(input_shape,))])
    for L in MODEL_CONFIG['hidden_layers']:
        m.add(Dense(L['units'], activation=L['activation']))
        if L['dropout']>0:
            m.add(Dropout(L['dropout']))
    m.add(Dense(1, activation=MODEL_CONFIG['output_activation']))
    m.compile(
        optimizer=TRAINING_CONFIG['optimizer'],
        loss=TRAINING_CONFIG['loss'],
        metrics=TRAINING_CONFIG['metrics'],
        run_eagerly=run_eagerly
    )
    return m

# ==============================================================================
# 5.  Baseline classical method: Linear Regression
# ==============================================================================
def evaluate_linear_baseline(X_train, y_train, X_val, y_val):
    lr = LinearRegression()
    t0 = time.time()
    lr.fit(X_train, y_train)
    fit_time = time.time() - t0

    t1 = time.time()
    pred_s = lr.predict(X_val)
    pred_r = inverse_transform_target(pred_s)
    y_r    = inverse_transform_target(y_val)
    pred_time = time.time() - t1

    mse = np.mean((pred_r - y_r)**2)
    mae = np.mean(np.abs(pred_r - y_r))
    print("\n[Baseline: LinearRegression]")
    print(f"  Fit time:    {fit_time:.4f}s")
    print(f"  Predict time:{pred_time:.4f}s")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    return {'fit_time': fit_time, 'pred_time': pred_time, 'mae':mae, 'mse':mse}

# ==============================================================================
# 6.  K-Fold CV for NN
# ==============================================================================
def cross_validate_nn(X, y, n_splits=CV_FOLDS):
    """
    K-Fold cross-validation for the neural network, limited to 50 epochs
    per fold for faster feedback and with per-epoch verbose output.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    metrics = []
    cv_epochs = min(50, TRAINING_CONFIG['epochs'])

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n=== Starting CV fold {fold}/{n_splits} (epochs={cv_epochs}) ===")
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = build_model(X.shape[1], run_eagerly=False)
        cb = RealScaleMetrics(X_va, y_va)

        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=cv_epochs,
            batch_size=TRAINING_CONFIG['batch_size'],
            callbacks=TRAINING_CONFIG['callbacks'] + [cb],
            verbose=1
        )

        pred_s = model.predict(X_va, verbose=0)
        pred_r = inverse_transform_target(pred_s)
        y_r    = inverse_transform_target(y_va)

        mse = np.mean((pred_r - y_r)**2)
        mae = np.mean(np.abs(pred_r - y_r))
        metrics.append({'fold': fold, 'mae': mae, 'mse': mse})
        print(f"Fold {fold} results → MAE: {mae:.4f}, MSE: {mse:.4f}")

    avg_mae = np.mean([m['mae'] for m in metrics])
    avg_mse = np.mean([m['mse'] for m in metrics])
    print(f"\n[NN CV average over {n_splits} folds] MAE={avg_mae:.4f}, MSE={avg_mse:.4f}")

    return metrics
# ==============================================================================
# 7.  Permutation‐based sensitivity analysis
# ==============================================================================
def sensitivity_analysis(model, X_val, y_val_scaled, feature_names):
    baseline_s = model.predict(X_val, verbose=0)
    baseline_r = inverse_transform_target(baseline_s)
    y_r         = inverse_transform_target(y_val_scaled)
    baseline_mse = np.mean((baseline_r-y_r)**2)

    importances = {}
    for i, name in enumerate(feature_names):
        Xp = X_val.copy()
        np.random.shuffle(Xp[:, i])
        ps = model.predict(Xp, verbose=0)
        pr = inverse_transform_target(ps)
        mse = np.mean((pr - y_r)**2)
        importances[name] = mse - baseline_mse

    print("\n[Permutation Sensitivity (∆MSE)]")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:>5}: {imp:.4f}")
    return importances

# ==============================================================================
# 8.  Export for FEA/MES software
# ==============================================================================
def export_for_engineering(df):
    os.makedirs(os.path.dirname(OUTPUT_CONFIG['export_csv']), exist_ok=True)
    cols = DATA_CONFIG['features'] + ['predicted_MRd']
    df[cols].to_csv(OUTPUT_CONFIG['export_csv'], index=False)
    print(f"\n✅ Predictions exported to {OUTPUT_CONFIG['export_csv']}")

# ==============================================================================
# 9.  Future‐extension stubs
# ==============================================================================
def multi_criteria_optimization_stub():
    # TODO: implement weighted optimization (mass, cost, capacity, env. impact)
    pass

def alternative_architectures_stub():
    # TODO: build & compare CNN, LSTM, transfer‐learning models
    pass

# ==============================================================================
# 10. Main workflow
# ==============================================================================
def main():
    # load
    X_tr, X_va, y_tr, y_va, df_raw, X_all = load_and_preprocess_data()

    # baseline
    baseline_metrics = evaluate_linear_baseline(X_tr, y_tr, X_va, y_va)

    # NN CV
    _ = cross_validate_nn(np.vstack([X_tr,X_va]), np.vstack([y_tr,y_va]))

    # train final NN
    model = build_model(X_tr.shape[1] run_eagerly=True)
    real_cb = RealScaleMetrics(X_va, y_va)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va,y_va),
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        callbacks=TRAINING_CONFIG['callbacks']+[real_cb],
        verbose=1
    )

    # final eval
    pred_s = model.predict(X_va, verbose=0)
    pred_r = inverse_transform_target(pred_s)
    y_r    = inverse_transform_target(y_va)
    print("\n[Final model real‐scale eval]")
    print(f"  MAE: {np.mean(np.abs(pred_r-y_r)):.4f}")
    print(f"  MSE: {np.mean((pred_r-y_r)**2):.4f}")

    # sensitivity
    _ = sensitivity_analysis(model, X_va, y_va, DATA_CONFIG['features'])

    # full‐dataset predictions & export
    full_pred = inverse_transform_target(model.predict(X_all, verbose=0))
    df_raw['predicted_MRd'] = full_pred
    export_for_engineering(df_raw)

    # save
    os.makedirs(OUTPUT_CONFIG['save_path'], exist_ok=True)
    model.save(os.path.join(OUTPUT_CONFIG['save_path'], "model.keras"))
    joblib.dump(
        {f:TRANSFORMATION_CONFIG['features'][f]['scaler'] for f in DATA_CONFIG['features']},
        os.path.join(OUTPUT_CONFIG['save_path'], "scaler_X.pkl")
    )
    joblib.dump(TRANSFORMATION_CONFIG['target']['scaler'],
                os.path.join(OUTPUT_CONFIG['save_path'], "scaler_y.pkl"))
    joblib.dump(TRANSFORMATION_CONFIG,
                os.path.join(OUTPUT_CONFIG['save_path'], "transformers_config.pkl"))
    print(f"\n✅ All artifacts saved under {OUTPUT_CONFIG['save_path']}")

if __name__=="__main__":
    main()

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Helpers ===
def log_inverse(x):
    return np.exp(x)

def log_transform(x):
    return np.log(x)

def log_inverse(x):
    return np.exp(x)

def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return {'MSE': mse, 'MAE': mae, 'R2': r2}

def plot_errors(y_true, y_pred, title=""):
    errors_sq = (y_true - y_pred) ** 2
    errors_abs = np.abs(y_true - y_pred)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(errors_sq, bins=100, edgecolor='black')
    plt.title(f"{title} – Squared Errors")
    plt.xlabel("Squared Error")

    plt.subplot(1, 2, 2)
    plt.hist(errors_abs, bins=100, edgecolor='black')
    plt.title(f"{title} – Absolute Errors")
    plt.xlabel("Absolute Error")

    plt.tight_layout()
    plt.show()

def plot_scatter(y_true, y_pred, title=""):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=1, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f"{title} – Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.show()

# === Config ===
BASE_PATH = r"neural_networks/rect_section_n1/models"
TARGETS = {
    "Mcr": {
        "features": ["b", "h", "d", "fi", "fck", "ro1"],
    },
    "MRd": {
        "features": ["b", "h", "d", "fi", "fck", "ro1"],
    },
    "Wk": {
        "features": ["MEqp", "b", "h", "d", "fi", "fck", "ro1"],
    }
}
DATASET_PATH = r"neural_networks/rect_section_n1/dataset/dataset_rect_n1.parquet"

# === Load dataset and preprocess ===
df = pd.read_parquet(DATASET_PATH).head(100_000)
df['d'] = df['h'] - df['fi'] / 2   # recalculate depth

for target, config in TARGETS.items():
    print(f"\n🔹 Evaluating model: {target}")
    
    # Load model and scalers
    model = load_model(os.path.join(BASE_PATH, f"{target}_model/model.keras"))
    scaler_X = joblib.load(os.path.join(BASE_PATH, f"{target}_model/scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(BASE_PATH, f"{target}_model/scaler_y.pkl"))
    transformers = joblib.load(os.path.join(BASE_PATH, f"{target}_model/transformers_config.pkl"))

    features = config["features"]
    X_raw = df[features].copy()

    # Apply log transformation
    X_transformed = np.zeros_like(X_raw.values)
    for i, feat in enumerate(features):
        eps = transformers["features"][feat]["epsilon"]
        X_transformed[:, i] = np.log(X_raw[feat].values + eps)

    # Scale input
    X_scaled = scaler_X.transform(X_transformed)

    # Predict
    y_scaled = model.predict(X_scaled)
    y_transformed = scaler_y.inverse_transform(y_scaled)
    y_pred = log_inverse(y_transformed).flatten()

    y_true = df[target].values

    # Metrics
    metrics = compute_metrics(y_true, y_pred)
    print(f"  MSE:  {metrics['MSE']:.4f}")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  R²:   {metrics['R2']:.4f}")

    # Plots
    plot_errors(y_true, y_pred, title=target)
    plot_scatter(y_true, y_pred, title=target)

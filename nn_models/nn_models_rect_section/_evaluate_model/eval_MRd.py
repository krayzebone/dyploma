#!/usr/bin/env python
# predict_100k_samples.py
"""
Script to load a trained model (and scalers) for MRd prediction,
take 100,000 random samples from the dataset, transform them,
perform predictions, and visualize the results.

Author: Your Name
Date:   2025-03-30
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# 1. Configuration (Adjust paths accordingly)
# --------------------------------------------------------------------------------

DATA_FILEPATH = r"datasets\dataset_rect_section.parquet"
MODEL_FOLDER  = r"nn_models\nn_models_rect_section\MRd_model"

# Name of the target column in the dataset
TARGET_COLUMN = "MRd"

# Maximum number of samples to predict
MAX_SAMPLES = 100_000

# --------------------------------------------------------------------------------
# 2. Load Transformation Functions & Config
#    (Make sure these match exactly how you trained the model)
# --------------------------------------------------------------------------------

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

# We will load the TRANSFORMATION_CONFIG from disk
# but here is the fallback structure:
fallback_TRANSFORMATION_CONFIG = {
    'features': {
        'b':   {'transform': log_transform,  'inverse_transform': log_inverse,  'epsilon': 1e-8},
        'h':   {'transform': log_transform,  'inverse_transform': log_inverse,  'epsilon': 1e-8},
        'd':   {'transform': log_transform,  'inverse_transform': log_inverse,  'epsilon': 1e-8},
        'fi':  {'transform': log_transform,  'inverse_transform': log_inverse,  'epsilon': 1e-8},
        'fck': {'transform': log_transform,  'inverse_transform': log_inverse,  'epsilon': 1e-8},
        'ro1': {'transform': log_transform,  'inverse_transform': log_inverse,  'epsilon': 1e-8},
        'ro2': {'transform': log_transform,  'inverse_transform': log_inverse,  'epsilon': 1e-8},
    },
    'target': {
        'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8
    }
}

# --------------------------------------------------------------------------------
# 3. Utility Functions
# --------------------------------------------------------------------------------

def load_model_and_scalers(model_folder):
    """
    Load a Keras model, plus X_scaler, y_scaler, and (optionally) transformation config.
    """
    # Load model (model.keras) 
    model_path = os.path.join(model_folder, "model.keras")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Load scalers (scaler_X.pkl, scaler_y.pkl)
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
    """
    For each feature, apply the same transform used during training,
    then apply the fitted scaler.
    Returns a NumPy array of scaled features.
    """
    # Transform features
    X_transformed = np.zeros((len(df), len(features)), dtype=float)
    for i, feature in enumerate(features):
        tcfg = transformation_config['features'][feature]
        epsilon = tcfg.get('epsilon', 0.0)
        X_transformed[:, i] = tcfg['transform'](df[feature].values + epsilon)

    # Scale
    X_scaled = X_scaler.transform(X_transformed)
    return X_scaled


def inverse_transform_target(y_scaled, y_scaler, transformation_config):
    """
    Invert the scaling on the predictions and then invert the transform 
    (e.g., log, sqrt) to get back to the original scale of the target.
    """
    y_unscaled_transformed = y_scaler.inverse_transform(y_scaled)  # from scaled space back to "log" or "sqrt"
    # Now use the inverse_transform from transformation_config
    return transformation_config['target']['inverse_transform'](y_unscaled_transformed)


def compute_metrics(y_true, y_pred):
    """
    Compute various regression metrics:
    MSE, MAE, RMSE, R².
    Returns a dictionary of metric values.
    """
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    # R^2 = 1 - SS_res / SS_tot
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }


def plot_histogram_of_difference(y_true, y_pred, bins=100):
    """
    Plot a histogram of the differences (y_true - y_pred).
    """
    differences = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(differences, bins=bins, edgecolor='black')
    plt.title("Histogram of (Actual - Predicted)")
    plt.xlabel("Difference")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_scatter_actual_vs_predicted(y_true, y_pred):
    """
    Scatter plot of Actual vs. Predicted.
    Shows a 1:1 reference line.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, s=1, alpha=0.5)
    # 1:1 line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title("Actual vs. Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.show()


# --------------------------------------------------------------------------------
# 4. Main Execution
# --------------------------------------------------------------------------------

def main():
    # ------------------------------------------------------
    # 4.1 Load the dataset
    # ------------------------------------------------------
    if not os.path.exists(DATA_FILEPATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_FILEPATH}")

    print("Loading dataset...")
    df = pd.read_parquet(DATA_FILEPATH)

    # Make sure the target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
    
    # List out features you used during training:
    # (Adjust if your features differ; 
    #  must match the original training 'features' exactly)
    features_used = ["b", "h", "d", "fi", "fck", "ro1", "ro2"]

    # ------------------------------------------------------
    # 4.2 Randomly sample 100,000 rows (or all if smaller)
    # ------------------------------------------------------
    total_rows = len(df)
    sample_size = min(MAX_SAMPLES, total_rows)
    print(f"Total rows in dataset: {total_rows}")
    print(f"Sampling {sample_size} rows for prediction...")
    sample_df = df.sample(n=sample_size, random_state=42).copy()

    # ------------------------------------------------------
    # 4.3 Load model & scalers & transformation config
    # ------------------------------------------------------
    print("Loading trained model and scalers...")
    model, X_scaler, y_scaler, transformation_config = load_model_and_scalers(MODEL_FOLDER)

    # ------------------------------------------------------
    # 4.4 Transform the sample's features
    # ------------------------------------------------------
    print("Transforming sample features...")
    X_scaled = transform_features(
        df=sample_df,
        features=features_used,
        transformation_config=transformation_config,
        X_scaler=X_scaler
    )

    # ------------------------------------------------------
    # 4.5 Predict using the loaded model
    # ------------------------------------------------------
    print("Predicting on sample...")
    y_pred_scaled = model.predict(X_scaled)
    # Convert from scaled => original scale
    y_pred = inverse_transform_target(
        y_scaled=y_pred_scaled, 
        y_scaler=y_scaler, 
        transformation_config=transformation_config
    ).flatten()

    # ------------------------------------------------------
    # 4.6 Compare with actual target in original scale
    # ------------------------------------------------------
    y_true = sample_df[TARGET_COLUMN].values

    # ------------------------------------------------------
    # 4.7 Compute Metrics
    # ------------------------------------------------------
    print("Computing metrics...")
    metrics_dict = compute_metrics(y_true, y_pred)
    print("Metrics on the 100k sample:")
    for k, v in metrics_dict.items():
        print(f"  {k}: {v:.6f}")

    # ------------------------------------------------------
    # 4.8 Visualize: Scatter + Histogram of difference
    # ------------------------------------------------------
    print("Generating plots...")
    plot_scatter_actual_vs_predicted(y_true, y_pred)
    plot_histogram_of_difference(y_true, y_pred, bins=100)

    print("\n✅ Prediction script completed successfully!")


if __name__ == "__main__":
    main()

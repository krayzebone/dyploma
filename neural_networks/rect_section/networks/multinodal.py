import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
import matplotlib.pyplot as plt

tf.random.set_seed(38)

# --------------------------------------------------------------------------------
# Named transformations (so they can be easily pickled if needed)
# --------------------------------------------------------------------------------
def no_transform(x):
    return x

def log_transform(x):
    return np.log(x)

def log_inverse(x):
    return np.exp(x)

# --------------------------------------------------------------------------------
# We will define a helper function that:
#  1) Reads the dataset
#  2) Subsets the required features & target
#  3) Log-transforms them
#  4) Scales them
#  5) Returns train/val splits + the scalers
# Because we have 3 different sets of inputs & 3 different targets, we’ll call this
# helper 3 times with different arguments.
# --------------------------------------------------------------------------------
def prepare_data(
    df,
    input_features,  # list of strings for features
    target_col,      # string name of the target
    test_size=0.3,
    random_state=42
):
    """
    1) Subset df to the chosen features (X) and chosen target (y).
    2) Log-transform X columns and the target column.
    3) Scale X columns and y column.
    4) Return X_train, X_val, y_train, y_val, plus the scalers for X and y.
    """
    # Subset
    df_sub = df[input_features + [target_col]].copy()

    # (Optional) Drop any rows with NaNs if your dataset might have them
    df_sub.dropna(inplace=True)

    # Separate X and y
    X_raw = df_sub[input_features].values
    y_raw = df_sub[target_col].values.reshape(-1, 1)

    # Log-transform inputs
    X_log = np.log(X_raw + 1e-8)
    # Log-transform target
    y_log = np.log(y_raw + 1e-8)

    # Scale inputs
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_log)

    # Scale outputs
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_log)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, y_train, y_val, X_scaler, y_scaler

# --------------------------------------------------------------------------------
# Function to build a Sequential model given a list of hidden-layer specs:
# Example hidden layer config:
# [
#    {'units': 289, 'activation': 'relu', 'dropout': 0.00017},
#    {'units': 84,  'activation': 'relu', 'dropout': 0.00737}
# ]
# --------------------------------------------------------------------------------
def build_sequential_model(input_dim, hidden_config, output_dim=1, output_activation='linear'):
    """Builds a simple Sequential MLP with the specified hidden layers."""
    inputs = Input(shape=(input_dim,))
    x = inputs
    for layer_spec in hidden_config:
        x = Dense(layer_spec['units'], activation=layer_spec['activation'])(x)
        if layer_spec['dropout'] > 0:
            x = Dropout(layer_spec['dropout'])(x)
    outputs = Dense(output_dim, activation=output_activation)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --------------------------------------------------------------------------------
# A helper to train/evaluate a single model with the provided config.
# We'll do early stopping & reduce-on-plateau as in your example.
# --------------------------------------------------------------------------------
def train_and_evaluate_model(
    model,
    optimizer,
    loss,
    metrics,
    X_train, y_train,
    X_val,   y_val,
    batch_size,
    max_epochs=3000
):
    # Compile
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Summaries
    model.summary()

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=1e-8)
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate in scaled space
    val_loss, val_mse_scaled, val_mae_scaled = model.evaluate(X_val, y_val, verbose=0)
    print("\nValidation metrics in scaled (log) space:")
    print(f"  - {metrics[0]}: {val_mse_scaled}")
    print(f"  - {metrics[1]}: {val_mae_scaled}")

    return history

# --------------------------------------------------------------------------------
# Real-scale evaluation helper
#   We pass in the model + X_scaler, y_scaler
#   We'll predict on the entire dataset or just the val set, as desired,
#   invert back to real space, then compute MSE, MAE, R^2.
# --------------------------------------------------------------------------------
def evaluate_on_real_scale(model, X, y_true, X_scaler, y_scaler):
    """Compute predictions in real scale (undoing the log + standard scaling)."""
    # Invert the X-scaling
    # (But typically, you only need to invert the Y for error metrics.)
    # Predict in scaled space:
    y_pred_scaled = model.predict(X)
    # Convert from scaled -> log space
    y_pred_log = y_scaler.inverse_transform(y_pred_scaled)
    # Then from log space to real space
    y_pred = np.exp(y_pred_log)

    # Do the same for y_true
    y_true_log = y_scaler.inverse_transform(y_true)
    y_true_real = np.exp(y_true_log)

    # MSE, MAE, R^2
    mse_unscaled = np.mean((y_pred.flatten() - y_true_real.flatten()) ** 2)
    mae_unscaled = np.mean(np.abs(y_pred.flatten() - y_true_real.flatten()))
    ss_res = np.sum((y_true_real.flatten() - y_pred.flatten()) ** 2)
    ss_tot = np.sum((y_true_real.flatten() - np.mean(y_true_real.flatten())) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print("\nReal-Scale Evaluation:")
    print("  - MSE:", mse_unscaled)
    print("  - MAE:", mae_unscaled)
    print("  - R^2:", r2)

    return y_pred, y_true_real

# --------------------------------------------------------------------------------
# Main script that trains 3 separate models: Mcr, MRd, Wk
# --------------------------------------------------------------------------------
def main():
    # ----------------------------------------------------------------
    # 1) Read the parquet dataset once
    # ----------------------------------------------------------------
    data_path = r"datasets\dataset_rect_section.parquet"
    df = pd.read_parquet(data_path)
    
    # ----------------------------------------------------------------
    # 2) Prepare data for each target using each target's input features
    #    (log transform of both features & target, then scale)
    # ----------------------------------------------------------------
    # Mcr
    mcr_features = ["b", "h", "d", "fi", "fck", "ro1", "ro2"]
    X_train_mcr, X_val_mcr, y_train_mcr, y_val_mcr, X_scaler_mcr, y_scaler_mcr = prepare_data(
        df=df,
        input_features=mcr_features,
        target_col="Mcr",
        test_size=0.3,
        random_state=42
    )

    # MRd
    mrd_features = ["b", "h", "d", "fi", "fck", "ro1", "ro2"]
    X_train_mrd, X_val_mrd, y_train_mrd, y_val_mrd, X_scaler_mrd, y_scaler_mrd = prepare_data(
        df=df,
        input_features=mrd_features,
        target_col="MRd",
        test_size=0.3,
        random_state=42
    )

    # wk
    wk_features = ["MEd", "b", "h", "d", "fi", "fck", "ro1", "ro2"]
    X_train_wk, X_val_wk, y_train_wk, y_val_wk, X_scaler_wk, y_scaler_wk = prepare_data(
        df=df,
        input_features=wk_features,
        target_col="wk",
        test_size=0.3,
        random_state=42
    )

    # ----------------------------------------------------------------
    # 3) Build each model with its own hyperparams
    # ----------------------------------------------------------------

    # -------------------------------------
    # Mcr model hyperparameters
    # -------------------------------------
    mcr_hidden_layers = [
        {'units': 289, 'activation': 'relu', 'dropout': 0.00017408447601856974},
        {'units': 84,  'activation': 'relu', 'dropout': 0.0073718573268978585}
    ]
    mcr_optimizer = Adam(learning_rate=6.509132184030181e-05)
    mcr_loss = 'mse'
    mcr_metrics = ['mse', 'mae']
    mcr_batch_size = 148

    mcr_model = build_sequential_model(
        input_dim=len(mcr_features),
        hidden_config=mcr_hidden_layers,
        output_dim=1,
        output_activation='linear'
    )

    # -------------------------------------
    # MRd model hyperparameters
    # -------------------------------------
    mrd_hidden_layers = [
        {'units': 315, 'activation': 'relu', 'dropout': 0.00463261993637544}
    ]
    mrd_optimizer = Adam(learning_rate=0.00018775216843688273)
    mrd_loss = 'mse'
    mrd_metrics = ['mse', 'mae']
    mrd_batch_size = 82

    mrd_model = build_sequential_model(
        input_dim=len(mrd_features),
        hidden_config=mrd_hidden_layers,
        output_dim=1,
        output_activation='linear'
    )

    # -------------------------------------
    # Wk model hyperparameters
    # (4 hidden layers)
    # -------------------------------------
    # We’ll reconstruct from your dictionary:
    # {'n_layers': 4, 'n_units_l0': 357, 'dropout_l0': 0.1843, 'n_units_l1': 296, 'dropout_l1': 0.0548,
    #  'n_units_l2': 61, 'dropout_l2': 0.1612, 'n_units_l3': 253, 'dropout_l3': 0.3116, 'lr': 0.0023537, 'batch_size': 96}
    wk_hidden_layers = [
        {'units': 357, 'activation': 'relu', 'dropout': 0.18435158080849778},
        {'units': 296, 'activation': 'relu', 'dropout': 0.05481771741940156},
        {'units': 61,  'activation': 'relu', 'dropout': 0.16126167701693794},
        {'units': 253, 'activation': 'relu', 'dropout': 0.31165637335280605},
    ]
    wk_optimizer = Adam(learning_rate=0.002353704934480255)
    wk_loss = 'mse'
    wk_metrics = ['mse', 'mae']
    wk_batch_size = 96

    wk_model = build_sequential_model(
        input_dim=len(wk_features),
        hidden_config=wk_hidden_layers,
        output_dim=1,
        output_activation='linear'
    )

    # ----------------------------------------------------------------
    # 4) Train each model separately, with a large epoch limit
    # ----------------------------------------------------------------
    max_epochs = 30

    print("===== Training Mcr model =====")
    train_and_evaluate_model(
        model=mcr_model,
        optimizer=mcr_optimizer,
        loss=mcr_loss,
        metrics=mcr_metrics,
        X_train=X_train_mcr, y_train=y_train_mcr,
        X_val=X_val_mcr,     y_val=y_val_mcr,
        batch_size=mcr_batch_size,
        max_epochs=max_epochs
    )

    print("\n===== Training MRd model =====")
    train_and_evaluate_model(
        model=mrd_model,
        optimizer=mrd_optimizer,
        loss=mrd_loss,
        metrics=mrd_metrics,
        X_train=X_train_mrd, y_train=y_train_mrd,
        X_val=X_val_mrd,     y_val=y_val_mrd,
        batch_size=mrd_batch_size,
        max_epochs=max_epochs
    )

    print("\n===== Training Wk model =====")
    train_and_evaluate_model(
        model=wk_model,
        optimizer=wk_optimizer,
        loss=wk_loss,
        metrics=wk_metrics,
        X_train=X_train_wk, y_train=y_train_wk,
        X_val=X_val_wk,     y_val=y_val_wk,
        batch_size=wk_batch_size,
        max_epochs=max_epochs
    )

    # ----------------------------------------------------------------
    # 5) Evaluate real-scale performance on validation set (or entire df)
    # ----------------------------------------------------------------
    # For demonstration, let's do real-scale evaluation on each val set
    print("\n\n===== Real-Scale Evaluation: Mcr =====")
    _ = evaluate_on_real_scale(mcr_model, X_val_mcr, y_val_mcr, X_scaler_mcr, y_scaler_mcr)

    print("\n\n===== Real-Scale Evaluation: MRd =====")
    _ = evaluate_on_real_scale(mrd_model, X_val_mrd, y_val_mrd, X_scaler_mrd, y_scaler_mrd)

    print("\n\n===== Real-Scale Evaluation: Wk =====")
    _ = evaluate_on_real_scale(wk_model, X_val_wk, y_val_wk, X_scaler_wk, y_scaler_wk)

    # ----------------------------------------------------------------
    # 6) Save all models & scalers
    # ----------------------------------------------------------------
    output_dir = r"nn_models\nn_models_rect_section"
    os.makedirs(output_dir, exist_ok=True)

    # Save the Keras models in standard .keras format
    mcr_model.save(os.path.join(output_dir, "Mcr_model.keras"))
    mrd_model.save(os.path.join(output_dir, "MRd_model.keras"))
    wk_model.save(os.path.join(output_dir, "Wk_model.keras"))

    # Save the scalers (X_scaler_* and y_scaler_*) using joblib
    joblib.dump(X_scaler_mcr, os.path.join(output_dir, "X_scaler_mcr.pkl"))
    joblib.dump(y_scaler_mcr, os.path.join(output_dir, "y_scaler_mcr.pkl"))

    joblib.dump(X_scaler_mrd, os.path.join(output_dir, "X_scaler_mrd.pkl"))
    joblib.dump(y_scaler_mrd, os.path.join(output_dir, "y_scaler_mrd.pkl"))

    joblib.dump(X_scaler_wk, os.path.join(output_dir, "X_scaler_wk.pkl"))
    joblib.dump(y_scaler_wk, os.path.join(output_dir, "y_scaler_wk.pkl"))

    print("\n✅ All three models and their scalers have been saved successfully!")

if __name__ == "__main__":
    main()

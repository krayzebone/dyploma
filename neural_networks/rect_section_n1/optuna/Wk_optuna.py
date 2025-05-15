import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

tf.random.set_seed(38)

# ============================================
# Data Loading and Preprocessing
# ============================================
df = pd.read_parquet(r"neural_networks\rect_section_n1\dataset\dataset_rect_n1_test5.parquet")

features = ["MEqp", "b", "h", "cnom", "d", "fi", "fck", "ro1"]
target = ["Wk"]

X = df[features].values   # shape: (n_samples, 8)
y = df[target].values.reshape(-1, 1)     # shape: (n_samples, 1)

# Apply log transformation to features and target
X_log = np.log1p(X)  # Using log1p to handle zero values safely
y_log = np.log1p(y)

# Standardize the log-transformed features and targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_log)
y_scaled = scaler_y.fit_transform(y_log)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42
)

# ============================================
# Model Building
# ============================================
def create_model(trial):
    """
    Build a neural network model for single output prediction.
    Includes Batch Normalization after each Dense layer.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    
    # Number of hidden layers
    n_layers = trial.suggest_int("n_layers", 1, 8)
    
    for i in range(n_layers):
        n_units = trial.suggest_int(f"n_units_l{i}", 16, 600)
        dropout_rate = trial.suggest_float(f"dropout_l{i}", 0.0, 0.6)

        model.add(layers.Dense(n_units, activation=None))
        model.add(layers.Activation('relu'))
        
        if dropout_rate > 0:
            model.add(layers.Dropout(rate=dropout_rate))
    
    # Final layer with 1 output
    model.add(layers.Dense(1, activation='linear'))
    
    # Learning rate
    lr = trial.suggest_float("lr", 1e-7, 1e-1, log=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mse']    
    )
    return model

# ============================================
# Objective Function for Optuna
# ============================================
def objective(trial):
    model = create_model(trial)
    batch_size = trial.suggest_int("batch_size", 16, 256)
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    val_loss, _ = model.evaluate(X_val, y_val, verbose=0)
    return val_loss

# ============================================
# Run the Optuna Study
# ============================================
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

# Print best trial results
print("Best trial:")
trial = study.best_trial
print("  MSE: ", trial.value)
print("  Best hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
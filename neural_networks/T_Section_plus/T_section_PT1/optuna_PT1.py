import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(38)

# ============================================
# Data Loading and Preprocessing
# ============================================
df = pd.read_parquet(r"dataset_files\T_section_plus\T_section_PT1\PT1r_dataset.parquet")

features = ["MEd", "beff", "bw", "h", "hf", "fi", "fck"]
target = ["cost", "MRd", "As1"]

X = df[features].values
y = df[target].values

# Add a small epsilon to avoid log(0) issues
epsilon = 1e-8
X = np.log(X + epsilon)
y = np.log(y + epsilon)

# Standardize the features and targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42
)

# ============================================
# Model Building
# ============================================
def create_model(trial):
    """
    Build a neural network model for triple output prediction.
    """
    input_layer = layers.Input(shape=(X_train.shape[1],))
    
    # Number of hidden layers
    n_layers = trial.suggest_int("n_layers", 1, 6)
    
    x = input_layer
    for i in range(n_layers):
        n_units = trial.suggest_int(f"n_units_l{i}", 16, 400)
        dropout_rate = trial.suggest_float(f"dropout_l{i}", 0.0, 0.5)

        x = layers.Dense(n_units)(x)
        x = layers.Activation('relu')(x)
        
        if dropout_rate > 0:
            x = layers.Dropout(rate=dropout_rate)(x)
    
    # Create three output layers
    output1 = layers.Dense(1, activation='linear', name='cost')(x)
    output2 = layers.Dense(1, activation='linear', name='MRd')(x)
    output3 = layers.Dense(1, activation='linear', name='As1')(x)
    
    model = keras.Model(inputs=input_layer, outputs=[output1, output2, output3])
    
    # Learning rate
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    
    # Compile with metrics for each output
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics={
            'cost': 'mse',
            'MRd': 'mse',
            'As1': 'mse'
        }
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
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, 
        {'cost': y_train[:,0], 'MRd': y_train[:,1], 'As1': y_train[:,2]},
        validation_data=(
            X_val, 
            {'cost': y_val[:,0], 'MRd': y_val[:,1], 'As1': y_val[:,2]}
        ),
        epochs=150,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    val_loss = model.evaluate(
        X_val, 
        {'cost': y_val[:,0], 'MRd': y_val[:,1], 'As1': y_val[:,2]}, 
        verbose=0
    )
    return val_loss[0]  # Return the total loss

# ============================================
# Run the Optuna Study
# ============================================
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Print best trial results
print("Best trial:")
trial = study.best_trial
print("  MSE: ", trial.value)
print("  Best hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
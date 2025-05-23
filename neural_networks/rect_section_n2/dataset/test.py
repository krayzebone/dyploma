import os
import pandas as pd
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(38)
np.random.seed(38)

# ============================================
# Data Loading and Preprocessing
# ============================================
df = pd.read_parquet(r"neural_networks\rect_section_n2\dataset\dataset_rect_n2.parquet_40k")
features = ["b", "h", "d", "fi", "fck", "ro1", "ro2"]
targets = ["Wk", "MRd", "Cost"]  # All target variables

# ============================================
# Model Building
# ============================================
def create_model(trial, input_shape):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    n_layers = trial.suggest_int("n_layers", 1, 6)
    use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
    
    for i in range(n_layers):
        n_units = trial.suggest_int(f"n_units_l{i}", 32, 512)
        dropout_rate = trial.suggest_float(f"dropout_l{i}", 0.0, 0.5)
        
        model.add(layers.Dense(
            n_units,
            activation=None
        ))
        
        if use_batchnorm:
            model.add(layers.BatchNormalization())
        
        model.add(layers.Activation('relu'))
        
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation='linear'))
    
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam"])
    
    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# ============================================
# Objective Function with Target-specific Processing
# ============================================
def create_objective(target_name, X_train, X_val, y_train, y_val, results_file):
    def objective(trial):
        model = create_model(trial, X_train.shape[1])
        batch_size = trial.suggest_int("batch_size", 32, 256, step=8)
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        pruning = optuna.integration.TFKerasPruningCallback(
            trial,
            monitor='val_loss'
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, pruning],
            verbose=0
        )
        
        val_loss, _ = model.evaluate(X_val, y_val, verbose=0)
        
        # Log trial results to DataFrame
        trial_results = {
            "trial_number": trial.number,
            "val_loss": val_loss,
            "best_epoch": len(history.history['val_loss']) - early_stop.patience,
            "target_variable": target_name,
            **trial.params  # Unpacks all hyperparameters
        }
        
        # Append to Excel file (creates if not exists)
        results_df = pd.DataFrame([trial_results])
        if not os.path.exists(results_file):
            results_df.to_excel(results_file, index=False)
        else:
            existing_df = pd.read_excel(results_file)
            updated_df = pd.concat([existing_df, results_df], ignore_index=True)
            updated_df.to_excel(results_file, index=False)
        
        return val_loss
    return objective

# ============================================
# Run Studies for Each Target
# ============================================
def run_study_for_target(target_name, n_trials=100):
    print(f"\nStarting study for target: {target_name}")
    
    # Prepare data for this specific target
    X = df[features].values
    y = df[[target_name]].values.reshape(-1, 1)
    
    # Log-transform and standardize
    X_log = np.log(X+1e-9)
    y_log = np.log(y+1e-9)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_log)
    y_scaled = scaler_y.fit_transform(y_log)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42
    )
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=38),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Create results filename
    results_file = f"optuna_results_{target_name.lower()}.xlsx"
    
    # Optimize
    study.optimize(
        create_objective(target_name, X_train, X_val, y_train, y_val, results_file),
        n_trials=n_trials,
        timeout=3600
    )
    
    # Print best results
    print(f"\nBest trial for {target_name}:")
    trial = study.best_trial
    print(f"  MSE: {trial.value:.4f}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save visualizations
    fig_history = optuna.visualization.plot_optimization_history(study)
    fig_history.write_image(f"optimization_history_{target_name.lower()}.png")
    
    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance.write_image(f"param_importances_{target_name.lower()}.png")
    
    return study

# ============================================
# Main Execution
# ============================================
if __name__ == "__main__":
    # Run studies for each target
    studies = {}
    for target in targets:
        studies[target] = run_study_for_target(target, n_trials=100)
    
    print("\nAll studies completed!")
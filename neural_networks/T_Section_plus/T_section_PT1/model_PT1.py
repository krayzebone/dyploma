import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
import matplotlib.pyplot as plt

tf.random.set_seed(38)

# ============================================
# Define named functions for transformations
# ============================================

def log_transform(x):
    return np.log(x)

def log_inverse(x):
    return np.exp(x)

# ============================================
# CENTRALIZED CONFIGURATION
# ============================================

# 1. Data Configuration
DATA_CONFIG = {
    'filepath': r"dataset_files\T_section_plus\T_section_PT1\PT1r_dataset.parquet",
    'features': ["MEd", "beff", "bw", "h", "hf", "fi", "fck"],
    'targets': ["cost", "MRd", "As1"],  # Changed from 'target' to 'targets' for clarity
    'test_size': 0.3,
    'random_state': 42
}

# 2. Transformation Configuration
TRANSFORMATION_CONFIG = {
    'features': {
        # Feature transformations
        'MEd': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'beff': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'bw': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'h': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'hf': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'fi': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'fck': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
    },
    'targets': {
        # Target transformations (one per output)
        'cost': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'MRd': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'As1': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
    }
}

SCALER_CONFIG = {
    'X_scaler': StandardScaler(),
    'y_scalers': {  # Separate scaler for each target
        'cost': StandardScaler(),
        'MRd': StandardScaler(),
        'As1': StandardScaler()
    }
}

MODEL_CONFIG = {
    'hidden_layers': [
        {'units': 397, 'activation': 'relu', 'dropout': 0.14053617874853672}
    ],
    'output_activation': 'linear'
}

TRAINING_CONFIG = {
    'optimizer': Adam(learning_rate=1.5805718329232507e-05),
    'loss': 'mse',
    'metrics': ['mse', 'mae'],
    'batch_size': 50,
    'epochs': 3000,
    'callbacks': [
        EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=1e-8),
    ]
}

OUTPUT_CONFIG = {
    'save_path': r"nn_models\nn_models_rect_section\cost_model",
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
    """Load and preprocess data with proper multi-target handling"""
    df = pd.read_parquet(DATA_CONFIG['filepath'])

    # Apply feature transformations
    X_transformed = np.zeros_like(df[DATA_CONFIG['features']].values)
    for i, feature in enumerate(DATA_CONFIG['features']):
        transform_config = TRANSFORMATION_CONFIG['features'].get(feature)
        X_transformed[:, i] = transform_config['transform'](
            df[feature].values + transform_config['epsilon']
        )
    
    # Apply target transformations to each output
    y_transformed = np.zeros_like(df[DATA_CONFIG['targets']].values)
    for i, target in enumerate(DATA_CONFIG['targets']):
        transform_config = TRANSFORMATION_CONFIG['targets'].get(target)
        y_transformed[:, i] = transform_config['transform'](
            df[target].values + transform_config['epsilon']
        )

    # Scale features and targets
    X_scaled = SCALER_CONFIG['X_scaler'].fit_transform(X_transformed)
    
    # Scale each target separately
    y_scaled = np.zeros_like(y_transformed)
    for i, target in enumerate(DATA_CONFIG['targets']):
        y_scaled[:, i] = SCALER_CONFIG['y_scalers'][target].fit_transform(
            y_transformed[:, i].reshape(-1, 1)
        ).flatten()

    # Train-validation split
    X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
        X_scaled, y_scaled, 
        test_size=DATA_CONFIG['test_size'], 
        random_state=DATA_CONFIG['random_state']
    )
    
    return X_train, X_val, y_train_scaled, y_val_scaled, df, X_scaled

# ============================================
# Model Building (Updated for multiple outputs)
# ============================================
def build_model(input_shape):
    """Build model with multiple outputs"""
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    
    # Add hidden layers from config
    for layer in MODEL_CONFIG['hidden_layers']:
        model.add(Dense(layer['units'], activation=layer['activation']))
        if layer['dropout'] > 0:
            model.add(Dropout(layer['dropout']))
    
    # Output layers (one per target)
    outputs = []
    for target in DATA_CONFIG['targets']:
        outputs.append(Dense(1, activation=MODEL_CONFIG['output_activation'], name=target))
    
    # Create multi-output model
    model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    
    # Compile with training config
    model.compile(
        optimizer=TRAINING_CONFIG['optimizer'],
        loss={target: TRAINING_CONFIG['loss'] for target in DATA_CONFIG['targets']},
        metrics={target: TRAINING_CONFIG['metrics'] for target in DATA_CONFIG['targets']},
        loss_weights={target: 1.0 for target in DATA_CONFIG['targets']}  # Equal weights
    )
    
    model.summary()
    return model

# ============================================
# Training (Updated for multiple outputs)
# ============================================
def train_model(model, X_train, y_train_scaled, X_val, y_val_scaled):
    """Train with multiple outputs"""
    # Prepare target dictionary
    train_targets = {target: y_train_scaled[:, i] for i, target in enumerate(DATA_CONFIG['targets'])}
    val_targets = {target: y_val_scaled[:, i] for i, target in enumerate(DATA_CONFIG['targets'])}
    
    history = model.fit(
        X_train, train_targets,
        validation_data=(X_val, val_targets),
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        callbacks=TRAINING_CONFIG['callbacks'],
        verbose=1
    )
    return history

# [Rest of the functions would need similar updates for multi-output handling]
# [Evaluation, visualization, and inverse transformations need to handle each target separately]

if __name__ == "__main__":
    main()
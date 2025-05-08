import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# === Configuration ===
MODEL_DIR = 'nn_models/Tsection'
MODEL_PATH = os.path.join(MODEL_DIR, 'concrete_properties_classifier.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_FCK_PATH = os.path.join(MODEL_DIR, 'label_encoder_fck.pkl')
ENCODER_FI_PATH = os.path.join(MODEL_DIR, 'label_encoder_fi.pkl')

FEATURES = ['MEd', 'beff', 'bw', 'h', 'hf', 'cnom', 'fi_str']

# === Load model and scalers ===
model = tf.keras.models.load_model(MODEL_PATH)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

with open(ENCODER_FCK_PATH, 'rb') as f:
    le_fck = pickle.load(f)

with open(ENCODER_FI_PATH, 'rb') as f:
    le_fi = pickle.load(f)

# === Example input (you can replace this with your own data) ===
# Example: MEd, beff, bw, h, hf, cnom, fi_str
input_data = pd.DataFrame([{
    'MEd': 120.0,
    'beff': 1000,
    'bw': 300,
    'h': 550,
    'hf': 80,
    'cnom': 25,
    'fi_str': 16
}])[FEATURES]

# === Scale input ===
input_scaled = scaler.transform(input_data)

# === Make predictions ===
predictions = model.predict(input_scaled)
fck_probs, fi_probs = predictions

fck_pred = np.argmax(fck_probs, axis=1)
fi_pred = np.argmax(fi_probs, axis=1)

# === Decode to original class labels ===
fck_class = le_fck.inverse_transform(fck_pred)
fi_class = le_fi.inverse_transform(fi_pred)

# === Output ===
print(f"Predicted fck: {fck_class[0]}")
print(f"Predicted fi: {fi_class[0]}")

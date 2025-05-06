# append_predictions.py
# ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras import models

# ── paths (edit if yours differ) ───────────────────────────────
CSV_PATH   = r"Tsection222.csv"           # will be OVERWRITTEN
MODEL_PATH = r"C:\Users\marci\Documents\GitHub\dyploma\best_layers_nbars.keras"
SCAL_PATH  = r"C:\Users\marci\Documents\GitHub\dyploma\layers_nbars_scaler.pkl"

# ── load artefacts once ────────────────────────────────────────
model  = models.load_model(MODEL_PATH)
scaler = joblib.load(SCAL_PATH)

# ── 1. read csv ────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# ensure correct column order for the model
feature_cols = ["MEd", "beff", "bw", "h", "hf", "fi", "fck", "d"]
X = df[feature_cols].values.astype(float)

# ── 2. scale and predict ───────────────────────────────────────
X_scaled           = scaler.transform(X)
nb_pred, ly_probs  = model.predict(X_scaled, verbose=0)

df["n_bars1"] = nb_pred.flatten()          # regression head
df["layers1"] = ly_probs.argmax(axis=1) + 1  # classes start at 1

# ── 3. overwrite csv (or change name to keep original) ─────────
df.to_csv(CSV_PATH, index=False)
print(f"✅ Appended predictions and saved back to {CSV_PATH}")

# layers_nbars_model.py
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks, utils

# 1. ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_parquet(r"Tsection2.parquet")

X = df[["MEd", "beff", "bw", "h", "hf", "fi", "fck", "d"]].values
y_nbars  = df["n_bars"].values.reshape(-1, 1)          # regression target
y_layers = df["layers"].astype(int).values             # classification target

# 2. ── Train/val split (stratify on layer classes) ────────────────────────────
X_tr, X_val, nb_tr, nb_val, ly_tr_i, ly_val_i = train_test_split(
    X, y_nbars, y_layers,
    test_size=0.2, random_state=42, stratify=y_layers
)

# 3. ── One‑hot encode layer labels AFTER split ───────────────────────────────
n_classes = ly_tr_i.max() + 1
ly_tr = utils.to_categorical(ly_tr_i, num_classes=n_classes)
ly_val = utils.to_categorical(ly_val_i, num_classes=n_classes)

# 4. ── Scale numeric inputs ──────────────────────────────────────────────────
scaler = StandardScaler()
X_tr  = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)

# 5. ── Build two‑head network ────────────────────────────────────────────────
inputs = layers.Input(shape=(X_tr.shape[1],))
x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(16, activation="relu")(x)

out_nbars  = layers.Dense(1,  name="nbars")(x)                 # regression
out_layers = layers.Dense(n_classes, activation="softmax",
                          name="layers")(x)                    # classification

model = models.Model(inputs, [out_nbars, out_layers])
model.compile(
    optimizer="adam",
    loss={"nbars": "mse", "layers": "categorical_crossentropy"},
    metrics={"nbars": "mae", "layers": "accuracy"},
)

# 6. ── Callbacks ─────────────────────────────────────────────────────────────
chkpt_path = "best_layers_nbars.keras"
checkpoint_cb = callbacks.ModelCheckpoint(
    chkpt_path, 
    monitor="val_layers_accuracy",
    mode = "max",
    save_best_only=True
)
early_cb = callbacks.EarlyStopping(
    patience=15, 
    restore_best_weights=True,
    mode = "max",
    monitor="val_layers_accuracy"
)

# 7. ── Train ─────────────────────────────────────────────────────────────────
history = model.fit(
    X_tr, {"nbars": nb_tr, "layers": ly_tr},
    validation_data=(X_val, {"nbars": nb_val, "layers": ly_val}),
    epochs=300, batch_size=32,
    callbacks=[checkpoint_cb, early_cb],
    verbose=2
)

# 8. ── Persist artefacts ─────────────────────────────────────────────────────
model.save("final_layers_nbars.keras")       # consolidated final
joblib.dump(scaler, "layers_nbars_scaler.pkl")

print("✅ Saved: model (.keras) and scaler (.pkl)")

# 9. ── Reload & predict helper ───────────────────────────────────────────────
def load_and_predict(X_sample):
    """
    Parameters
    ----------
    X_sample : array‑like, shape (n_samples, 8)
        Columns in the same order as used for training.

    Returns
    -------
    nbars_pred : ndarray, shape (n_samples, 1)
    layers_pred : ndarray[str], shape (n_samples,)
    probs : ndarray, shape (n_samples, n_classes)
    """
    scaler = joblib.load("layers_nbars_scaler.pkl")
    model  = models.load_model("best_layers_nbars.keras")

    X_scaled = scaler.transform(X_sample)
    nbars_pred, layer_probs = model.predict(X_scaled)
    layer_classes = layer_probs.argmax(axis=1) + 1   # classes start at 1
    return nbars_pred.squeeze(), layer_classes, layer_probs

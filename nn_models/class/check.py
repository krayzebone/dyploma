import numpy as np
import pandas as pd
import joblib
from tensorflow.keras import models

# ── absolute paths to your artefacts ───────────────────────────
MODEL_PATH  = r"C:\Users\marci\Documents\GitHub\dyploma\best_layers_nbars.keras"
SCALER_PATH = r"C:\Users\marci\Documents\GitHub\dyploma\layers_nbars_scaler.pkl"

# ── lazy singletons: load once, reuse forever ─────────────────
_model  = None
_scaler = None

def _lazy_load():
    global _model, _scaler
    if _model is None:
        _model  = models.load_model(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)

def predict_section(X):
    """
    Parameters
    ----------
    X : array‑like of shape (n_samples, 8)
        Feature order: [MEd, beff, bw, h, hf, fi, fck, d]

    Returns
    -------
    n_bars   : ndarray shape (n_samples,)
    layers   : ndarray shape (n_samples,)  (integer classes ≥ 1)
    probs    : ndarray shape (n_samples, n_classes)
    """
    _lazy_load()
    X_arr = np.asarray(X, dtype=float)
    X_scaled = _scaler.transform(X_arr)
    nb_pred, layer_probs = _model.predict(X_scaled, verbose=0)

    # regression → 1‑D vector; classification → argmax + 1 (because classes started at 1)
    nbars   = nb_pred.flatten()
    layers  = layer_probs.argmax(axis=1)
    return nbars, layers, layer_probs

# ── example usage ─────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame(
        [[8685, 1947, 1339, 1004, 348, 32, 16, 938]],
        columns=["MEd", "beff", "bw", "h", "hf", "fi", "fck", "d"]
    )
    n_bars, layers, probs = predict_section(sample)
    print(f"Predicted n_bars ≈ {n_bars[0]:.1f}")
    print(f"Predicted layers = {layers[0]}")
    print("Class probabilities:", probs[0])

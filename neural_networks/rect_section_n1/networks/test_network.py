
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Model artefact locations and input feature definitions
# --------------------------------------------------------------------------------------
MODEL_PATHS = {
    "MRd": {
        "model": r"neural_networks\rect_section_n1\network_one\models\MRd_model\model.keras",
        "scaler_X": r"neural_networks\rect_section_n1\network_one\models\MRd_model\scaler_X.pkl",
        "scaler_y": r"neural_networks\rect_section_n1\network_one\models\MRd_model\scaler_y.pkl",
    },
    "Wk": {
        "model": r"neural_networks/rect_section_n1/models/Wk_model/model.keras",
        "scaler_X": r"neural_networks/rect_section_n1/models/Wk_model/scaler_X.pkl",
        "scaler_y": r"neural_networks/rect_section_n1/models/Wk_model/scaler_y.pkl",
    },
    "Cost": {
        "model": r"neural_networks/rect_section_n1/models/Cost_model/model.keras",
        "scaler_X": r"neural_networks/rect_section_n1/models/Cost_model/scaler_X.pkl",
        "scaler_y": r"neural_networks/rect_section_n1/models/Cost_model/scaler_y.pkl",
    },
}

MODEL_FEATURES = {
    "MRd": ["b", "h", "d", "fi", "fck", "ro1"],
    "Wk": ["MEqp", "b", "h", "d", "fi", "fck", "ro1"],
    "Cost": ["b", "h", "d", "fi", "fck", "ro1"],
}


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def _predict_single_target(model_name: str, input_df: pd.DataFrame) -> np.ndarray:
    """Load model + scalers and return predictions for *model_name* on *input_df*.

    Parameters
    ----------
    model_name : {"Mcr", "MRd", "Wk", "Cost"}
        Target variable to predict.
    input_df : pandas.DataFrame
        DataFrame containing at least the features required for the model.

    Returns
    -------
    numpy.ndarray, shape (n_samples,)
        Predictions in the original (non‑log) scale.
    """
    try:
        paths = MODEL_PATHS[model_name]
        model = tf.keras.models.load_model(paths["model"], compile=False)
        X_scalers: dict[str, object] = joblib.load(paths["scaler_X"])  # feature‑wise scalers
        y_scaler = joblib.load(paths["scaler_y"])  # target scaler

        # ------------------------------------------------------------------
        # Prepare feature matrix (log‑transform → feature‑wise scaling)
        # ------------------------------------------------------------------
        feat_cols = MODEL_FEATURES[model_name]
        X_raw = input_df[feat_cols].copy()
        X_log = np.log(X_raw + 1e-8)

        X_scaled = np.empty((len(X_log), len(feat_cols)), dtype=float)
        for idx, feat in enumerate(feat_cols):
            scaler = X_scalers[feat]
            X_scaled[:, idx] = scaler.transform(X_log[feat].values.reshape(-1, 1)).ravel()

        # ------------------------------------------------------------------
        # Predict & inverse‑transform back to original scale
        # ------------------------------------------------------------------
        pred_scaled = model.predict(X_scaled, verbose=0)
        pred = np.exp(y_scaler.inverse_transform(pred_scaled)).ravel()
        return pred

    except Exception as exc:  # pylint: disable=broad‑except
        print(f"⚠️  Error while predicting {model_name}: {exc}")
        return np.full(len(input_df), np.nan)


# --------------------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------------------

def main(data_path: str) -> None:
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset from: {data_path}")
    df = pd.read_parquet(data_path)

    # ------------------------------------------------------------------
    # 2. Run predictions for each target
    # ------------------------------------------------------------------
    targets = ["Wk", "MRd", "Cost"]
    for tgt in targets:
        print(f"Predicting {tgt} …")
        df[f"{tgt}_pred"] = _predict_single_target(tgt, df)

    # ------------------------------------------------------------------
    # 3. Plot Actual vs. Predicted scatter plots
    # ------------------------------------------------------------------
    _plot_results(df, targets)


def _plot_results(df: pd.DataFrame, targets: list[str]) -> None:
    """Create scatter plots: Actual vs. Predicted for each target variable."""
    n = len(targets)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(5 * n, 4), sharex=False, sharey=False)

    if n == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, tgt in zip(axes, targets):
        ax.scatter(df[tgt], df[f"{tgt}_pred"], s=1, alpha=0.6)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "--", linewidth=1)
        ax.set_xlabel(f"Actual {tgt}")
        ax.set_ylabel(f"Predicted {tgt}")
        ax.set_title(f"{tgt}: Actual vs. Predicted")
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------------
# Script entry point
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Wk, MRd, and Cost then plot A vs. P.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"neural_networks/rect_section_n1/dataset/dataset_rect_n1_test.parquet",
        help="Path to the test parquet dataset.",
    )

    args = parser.parse_args()
    if not Path(args.data_path).is_file():
        raise FileNotFoundError(f"Dataset not found at {args.data_path}")

    # Silence excessive TF logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa: RUF012
    main(args.data_path)
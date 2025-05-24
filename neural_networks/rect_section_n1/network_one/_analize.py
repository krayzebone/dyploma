# ──────────────────────────────────────────────────────────────────────────────
#  ALL-IN-ONE  ▸  Evaluate the three N-1 rectangular-section models
#               – predicts MRd, Wk, Cost on a test set
#               – prints MAE / RMSE
#               – shows ONE scatter-plot figure (1×3)
#               – shows ONE error-histogram figure (2×3)
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from pathlib import Path


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ 1.  Batch-prediction helper                                             │
# ╰──────────────────────────────────────────────────────────────────────────╯
def predict_section_batch_n1(input_data: pd.DataFrame, model_name: str) -> np.ndarray:
    """Return model predictions on *real scale* for one of the three targets."""
    model_paths = {
        "MRd":  Path(r"neural_networks\rect_section_n1\network_one\models\MRd_model"),
        "Wk":   Path(r"neural_networks\rect_section_n1\network_one\models\Wk_model"),
        "Cost": Path(r"neural_networks\rect_section_n1\network_one\models\Cost_model"),
    }
    model_features = {
        "MRd":  ["b", "h", "d", "fi", "fck", "ro1"],
        "Wk":   ["MEqp", "b", "h", "d", "fi", "fck", "ro1"],
        "Cost": ["b", "h", "d", "fi", "fck", "ro1"],
    }

    try:
        base = model_paths[model_name]
        model = tf.keras.models.load_model(base / "model.keras", compile=False)
        X_scalers = joblib.load(base / "scaler_X.pkl")  # dict: one scaler per feature
        y_scaler  = joblib.load(base / "scaler_y.pkl")

        # orderly feature matrix → log-transform → feature-wise scaling
        feats = model_features[model_name]
        X_log = np.log(input_data[feats].values + 1e-8)          # (n, d)
        X_scaled = np.empty_like(X_log)
        for j, feat in enumerate(feats):
            X_scaled[:, j] = X_scalers[feat].transform(X_log[:, [j]]).ravel()

        # predict (scaled) → inverse y-scaler → exp → flat 1-D
        pred_scaled = model.predict(X_scaled, verbose=0)
        return np.exp(y_scaler.inverse_transform(pred_scaled)).ravel()

    except Exception as exc:
        print(f"⚠️  {model_name} prediction failed:", exc)
        return np.full(len(input_data), np.nan)


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ 2.  Evaluation & combined plotting                                      │
# ╰──────────────────────────────────────────────────────────────────────────╯
def evaluate_models(dataset_path: str | Path) -> None:
    """Main routine: loads test data, predicts, prints metrics, plots."""
    test_data = pd.read_parquet(dataset_path)

    # — create two multi-axes figures —
    fig_scatter, axs_scatter = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    fig_error,   axs_error   = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)

    for pos, target in enumerate(["MRd", "Wk", "Cost"]):
        print(f"\nEvaluating {target} model…")

        y_true = test_data[target].values
        y_pred = predict_section_batch_n1(test_data, target)

        mask   = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_true, y_pred = y_true[mask], y_pred[mask]

        mae  = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        print(f"  MAE : {mae:,.4f}")
        print(f"  RMSE: {rmse:,.4f}")

        # ─ scatter subplot ─
        ax_s = axs_scatter[pos]
        ax_s.scatter(y_true, y_pred, s=4, alpha=0.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax_s.plot(lims, lims, "r--", linewidth=1)
        ax_s.set_title(target)
        ax_s.set_xlabel("Actual");  ax_s.set_ylabel("Predicted")
        ax_s.grid(True)

        # ─ error histograms ─
        abs_err = np.abs(y_pred - y_true)
        pct_err = 100 * abs_err / y_true

        axs_error[0, pos].hist(abs_err, bins=50, edgecolor="black")
        axs_error[0, pos].set_title(f"{target} | Absolute error")
        axs_error[0, pos].set_xlabel("|error|"); axs_error[0, pos].set_ylabel("Count")

        axs_error[1, pos].hist(pct_err, bins=50, edgecolor="black")
        axs_error[1, pos].set_title(f"{target} | % error")
        axs_error[1, pos].set_xlabel("% error"); axs_error[1, pos].set_ylabel("Count")

    # — figure titles & show —
    fig_scatter.suptitle("Predicted vs. Actual (all models)", fontsize=16, y=1.02)
    fig_error.suptitle("Error distributions (all models)",    fontsize=16, y=1.02)
    plt.show()


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ 3.  Run it!                                                              │
# ╰──────────────────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    evaluate_models(r"neural_networks/rect_section_n1/dataset/dataset_rect_n1_test.parquet")

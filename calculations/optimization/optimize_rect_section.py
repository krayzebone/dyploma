import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Example: disable OneDNN optimizations if needed

import numpy as np
import math
import joblib
from tensorflow.keras.models import load_model  # type: ignore
from itertools import product
from tqdm import tqdm  # Import tqdm for progress bar

###############################################################################
# 1. LOADING + PREDICTION UTILITIES
###############################################################################

def load_nn_model(model_path, scaler_X_path, scaler_y_path):
    """
    Loads one neural network model and its corresponding scalers
    from the given paths.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_X_path):
        raise FileNotFoundError(f"Scaler X file not found: {scaler_X_path}")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scaler y file not found: {scaler_y_path}")

    model = load_model(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    return model, scaler_X, scaler_y


def predict_value(model, scaler_X, scaler_y, input_list):
    """
    Generic function to:
      1. log-transform the inputs (must be > 0),
      2. scale them,
      3. run the neural network prediction,
      4. inverse the scaling of outputs,
      5. exponentiate to get back to original scale if outputs were trained in log-space.
    """
    # Convert to numpy array and check positivity
    arr = np.array(input_list, dtype=float).reshape(1, -1)
    if np.any(arr <= 0.0):
        raise ValueError("All input features must be > 0 for the log-transform approach.")
    
    # Log-transform inputs (if your model was trained on log of X)
    arr_log = np.log(arr)

    # Scale
    arr_scaled = scaler_X.transform(arr_log)

    # Predict
    pred_scaled = model.predict(arr_scaled)
    
    # Invert scaling of output
    pred_log = scaler_y.inverse_transform(pred_scaled)

    # Exponentiate the result (assuming output was also in log scale)
    pred = np.exp(pred_log)

    return float(pred[0, 0])  # If your model has a single output


###############################################################################
# 2. LOAD ALL FOUR MODELS: Mcr, MRd, Wk, Cost
###############################################################################
# Adjust these paths to match your local directory structure
Mcr_model_path    = r"nn_models/nn_models_rect_section/Mcr_model/model.keras"
Mcr_scaler_X_path = r"nn_models/nn_models_rect_section/Mcr_model/scaler_X.pkl"
Mcr_scaler_y_path = r"nn_models/nn_models_rect_section/Mcr_model/scaler_y.pkl"

MRd_model_path    = r"nn_models/nn_models_rect_section/MRd_model/model.keras"
MRd_scaler_X_path = r"nn_models/nn_models_rect_section/MRd_model/scaler_X.pkl"
MRd_scaler_y_path = r"nn_models/nn_models_rect_section/MRd_model/scaler_y.pkl"

Wk_model_path     = r"nn_models/nn_models_rect_section/Wk_model/model.keras"
Wk_scaler_X_path  = r"nn_models/nn_models_rect_section/Wk_model/scaler_X.pkl"
Wk_scaler_y_path  = r"nn_models/nn_models_rect_section/Wk_model/scaler_y.pkl"

Cost_model_path    = r"nn_models/nn_models_rect_section/Cost_model/model.keras"   # Example
Cost_scaler_X_path = r"nn_models/nn_models_rect_section/Cost_model/scaler_X.pkl"  # Example
Cost_scaler_y_path = r"nn_models/nn_models_rect_section/Cost_model/scaler_y.pkl"  # Example

# Load them
Mcr_model,  Mcr_sX,  Mcr_sY  = load_nn_model(Mcr_model_path,  Mcr_scaler_X_path,  Mcr_scaler_y_path)
MRd_model,  MRd_sX,  MRd_sY  = load_nn_model(MRd_model_path,  MRd_scaler_X_path,  MRd_scaler_y_path)
Wk_model,   Wk_sX,   Wk_sY   = load_nn_model(Wk_model_path,   Wk_scaler_X_path,   Wk_scaler_y_path)
Cost_model, Cost_sX, Cost_sY = load_nn_model(Cost_model_path, Cost_scaler_X_path, Cost_scaler_y_path)


###############################################################################
# 3. MAIN OPTIMIZATION LOGIC
###############################################################################
def main():

    # Fixed input values (the user states these are inputs)
    M_Ed = 100.0   # [kNm]
    b    = 1000.0  # [mm]
    h    = 250.0   # [mm]
    
    # Other needed parameters
    c_nom = 40.0   # nominal cover [mm]
    f_yk  = 500.0  # yield strength of steel [MPa]
    
    # Ranges to iterate
    fi_gl_values = [8, 10, 12, 16, 20, 24, 28, 32]  # [mm]
    f_ck_values  = [40] #[25, 30, 35, 40, 45]
    
    # Decide on how to sweep over rho1, rho2 (example discrete steps)
    # Adjust these as needed for your design space:
    rho1_values = np.linspace(0.001, 0.03, 10)  # example range
    rho2_values = np.linspace(0.001, 0.01, 5)   # example range

    # Define acceptance criteria
    wk_limit = 0.3   # [mm], for instance

    best_cost = float("inf")
    best_solution = None

    # Combine all parameter ranges using product and add tqdm progress bar
    total_iterations = len(f_ck_values) * len(fi_gl_values) * len(rho1_values) * len(rho2_values)
    for f_ck, fi_gl, rho1, rho2 in tqdm(product(f_ck_values, fi_gl_values, rho1_values, rho2_values),
                                        total=total_iterations,
                                        desc="Optimizing"):
        d = h - (c_nom + fi_gl / 2.0)

        # 1. Predict Mcr using [b, h, d, rho1, rho2, fi_gl, f_ck]
        Mcr_input = [b, h, d, rho1, rho2, fi_gl, f_ck]
        Mcr_pred  = predict_value(Mcr_model, Mcr_sX, Mcr_sY, Mcr_input)

        # Check Mcr >= M_Ed
        if Mcr_pred < M_Ed:
            continue

        # 2. Predict MRd using [b, h, d, rho1, rho2, fi_gl, f_ck]
        MRd_input = [b, h, d, rho1, rho2, fi_gl, f_ck]
        MRd_pred  = predict_value(MRd_model, MRd_sX, MRd_sY, MRd_input)

        # Check MRd >= M_Ed
        if MRd_pred < M_Ed:
            continue

        # 3. Predict Wk using [M_Ed, b, d, h, rho1, rho2, fi_gl, f_ck]
        Wk_input = [M_Ed, b, d, h, rho1, rho2, fi_gl, f_ck]
        Wk_pred  = predict_value(Wk_model, Wk_sX, Wk_sY, Wk_input)

        # Check wk <= limit
        if Wk_pred > wk_limit:
            continue

        # 4. Predict cost using [M_Ed, b, h, d, rho1, rho2, fi_gl, f_ck]
        Cost_input = [M_Ed, b, h, d, rho1, rho2, fi_gl, f_ck]
        cost_pred  = predict_value(Cost_model, Cost_sX, Cost_sY, Cost_input)

        # ----------------------------------------------------------------
        # NOW CHECK BAR-FITTING + REINFORCEMENT LIMIT CONSTRAINTS
        # ----------------------------------------------------------------
        A_s1 = rho1 * b * h
        A_s2 = rho2 * b * h

        # Minimum and maximum steel areas
        f_ctm  = 0.3 * f_ck ** (2.0/3.0)
        A_s_min = max(0.26 * (f_ctm / f_yk) * b * d, 0.0013 * b * d)
        A_s_max = 0.04 * b * h

        total_As = A_s1 + A_s2
        if not (A_s_min <= total_As <= A_s_max):
            continue

        # Check if the bars can actually fit
        area_one_bar = math.pi * (fi_gl**2) / 4.0
        n_bars_1 = math.ceil(A_s1 / area_one_bar)
        n_bars_2 = math.ceil(A_s2 / area_one_bar)

        # Example spacing requirement: minimal spacing (e.g., max(20 mm, fi_gl))
        min_spacing = max(20, fi_gl)
        width_needed_tension     = n_bars_1 * fi_gl + (n_bars_1 - 1) * min_spacing
        width_needed_compression = n_bars_2 * fi_gl + (n_bars_2 - 1) * min_spacing

        if (width_needed_tension > b) or (width_needed_compression > b):
            continue

        # If we get here, the design is feasible
        if cost_pred < best_cost:
            best_cost = cost_pred
            best_solution = {
                "f_ck":         f_ck,
                "fi_gl":        fi_gl,
                "rho1":         rho1,
                "rho2":         rho2,
                "Mcr":          Mcr_pred,
                "MRd":          MRd_pred,
                "wk":           Wk_pred,
                "cost":         cost_pred,
                "A_s1":         A_s1,
                "A_s2":         A_s2,
                "num_bars_1":   n_bars_1,
                "num_bars_2":   n_bars_2
            }

    # Finally, report the best solution found
    if best_solution is None:
        print("No solution satisfies all constraints in the given search space.")
    else:
        print("Best solution found:")
        for k, v in best_solution.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

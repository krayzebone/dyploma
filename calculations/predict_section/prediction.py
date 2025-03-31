import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import numpy as np
import joblib
import math
from tensorflow.keras.models import load_model  # type: ignore

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
      1. Log-transform the inputs (all values must be > 0),
      2. Scale them,
      3. Run the neural network prediction,
      4. Invert the scaling,
      5. Exponentiate to return to the original scale (assuming the outputs were log-transformed).
    """
    # Convert to numpy array and check for positivity
    arr = np.array(input_list, dtype=float).reshape(1, -1)
    if np.any(arr <= 0.0):
        raise ValueError("All input features must be > 0 for the log-transform approach.")
    
    # Log-transform inputs
    arr_log = np.log(arr)
    # Scale the inputs
    arr_scaled = scaler_X.transform(arr_log)
    # Run prediction
    pred_scaled = model.predict(arr_scaled)
    # Invert scaling of output
    pred_log = scaler_y.inverse_transform(pred_scaled)
    # Return to original scale
    pred = np.exp(pred_log)
    
    return float(pred[0, 0])


###############################################################################
# 2. LOAD MODELS AND SCALERS
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

# Load the models and scalers
Mcr_model,  Mcr_sX,  Mcr_sY  = load_nn_model(Mcr_model_path,  Mcr_scaler_X_path,  Mcr_scaler_y_path)
MRd_model,  MRd_sX,  MRd_sY  = load_nn_model(MRd_model_path,  MRd_scaler_X_path,  MRd_scaler_y_path)
Wk_model,   Wk_sX,   Wk_sY   = load_nn_model(Wk_model_path,   Wk_scaler_X_path,   Wk_scaler_y_path)
Cost_model, Cost_sX, Cost_sY = load_nn_model(Cost_model_path, Cost_scaler_X_path, Cost_scaler_y_path)


###############################################################################
# 3. PREDICTION FUNCTION
###############################################################################
def predict_all(MEd, b, h, d, f_ck, fi_gl, ro1, ro2):
    """
    Predicts Mcr, MRd, Wk, and Cost given the following inputs:
      - MEd: Applied moment [kNm]
      - b: Section width [mm]
      - h: Section overall depth [mm]
      - d: Effective depth [mm]
      - f_ck: Concrete strength [MPa]
      - fi_gl: Bar diameter [mm]
      - ro1: Reinforcement ratio in tension
      - ro2: Reinforcement ratio in compression
    """
    # For Mcr and MRd, the expected input order is: [b, h, d, ro1, ro2, fi_gl, f_ck]
    input_Mcr = [b, h, d, ro1, ro2, fi_gl, f_ck]
    Mcr_pred = predict_value(Mcr_model, Mcr_sX, Mcr_sY, input_Mcr)
    
    input_MRd = [b, h, d, ro1, ro2, fi_gl, f_ck]
    MRd_pred = predict_value(MRd_model, MRd_sX, MRd_sY, input_MRd)
    
    # For Wk, the expected input order is: [MEd, b, d, h, ro1, ro2, fi_gl, f_ck]
    input_Wk = [MEd, b, d, h, ro1, ro2, fi_gl, f_ck]
    Wk_pred = predict_value(Wk_model, Wk_sX, Wk_sY, input_Wk)
    
    # For Cost, the expected input order is: [MEd, b, h, d, ro1, ro2, fi_gl, f_ck]
    input_Cost = [MEd, b, h, d, ro1, ro2, fi_gl, f_ck]
    Cost_pred = predict_value(Cost_model, Cost_sX, Cost_sY, input_Cost)
    
    return Mcr_pred, MRd_pred, Wk_pred, Cost_pred


###############################################################################
# 4. MAIN EXECUTION
###############################################################################
def main():
    # Example input values - adjust these as needed
    MEd   = 1174   # [kNm]
    b     = 1974  # [mm]
    h     = 466   # [mm]
    f_ck  = 40      # [MPa]
    fi_gl = 32      # [mm]
    ro1   = 0.0157   # reinforcement ratio in tension
    ro2   = 0.01113   # reinforcement ratio in compression

    c_nom = 40
    d = h - c_nom - fi_gl / 2



    # Predict outputs
    Mcr, MRd, Wk, Cost = predict_all(MEd, b, h, d, f_ck, fi_gl, ro1, ro2)

    print("Predicted outputs:")
    print(f"Mcr:  {Mcr:.2f}")
    print(f"MRd:  {MRd:.2f}")
    print(f"Wk:   {Wk:.2f}")
    print(f"Cost: {Cost:.2f}")

if __name__ == "__main__":
    main()

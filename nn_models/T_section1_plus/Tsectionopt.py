import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import math
from itertools import product

def quadratic_equation(a: float, b: float, c: float, limit: float) -> float | None:
    """Solve ax²+bx+c = 0 and return the root that lies in (0, limit)."""
    if a == 0:
        return None

    delta = b**2 - 4 * a * c
    if delta < 0:
        return None

    sqrt_delta = math.sqrt(delta)
    x1 = (-b - sqrt_delta) / (2 * a)
    x2 = (-b + sqrt_delta) / (2 * a)
    valid = [x for x in (x1, x2) if 0 < x < limit]
    return min(valid) if valid else None

def calculate_number_of_rods(As: float, fi: float) -> tuple[int, float]:
    """Return number of bars and the provided reinforcement area."""
    if As <= 0 or math.isinf(As):
        return 0, 0.0
    area_bar = math.pi * fi**2 / 4
    n = math.ceil(As / area_bar)
    return n, n * area_bar

def check_rods_fit(bw: float, cnom: float, num_rods: int, fi: float, smax: float = 25, layers: int = 1) -> bool:
    """Check clear spacing rules in *one* reinforcement layer."""
    if num_rods == 0:
        return True
    required = 2 * cnom + num_rods * fi + smax * (num_rods - 1)
    return required <= layers * bw

def batch_predict(model_dict, X_batch):
    """Make batch predictions using pre-loaded models and scalers"""
    try:
        # Apply log transform and scale
        X_scaled = model_dict['scaler_X'].transform(np.log(X_batch + 1e-8))
        
        # Predict and inverse transform
        pred_scaled = model_dict['model'].predict(X_scaled)
        pred = np.exp(model_dict['scaler_y'].inverse_transform(pred_scaled))
        return pred.flatten()
    except Exception as e:
        print(f"⚠️ Error in batch prediction: {str(e)}")
        return None

def check_section_type_batch(fck_values, fyk, beff, d_values, h, MEd):
    results = []
    for fck, d in zip(fck_values, d_values):
        fcd = fck / 1.4
        fyd = fyk / 1.15
        xeff = quadratic_equation(-0.5 * beff * fcd, beff * fcd * d, -MEd * 1e6, h)
        
        if xeff is None:
            results.append(None)
            continue

        ksieff = xeff / d
        ksiefflim = 0.8 * 0.0035 / (0.0035 + fyd / 200_000)
        results.append(1 if ksieff <= ksiefflim else 2)
    return results

def find_optimal_combination(MEd, beff, bw, h, hf, fi_values, fck_values, cnom, fistr, fyk=500, smax=25):
    # Load all models and scalers once
    model_dict = {
        'section1': {
            'As1': {
                'model': tf.keras.models.load_model(r"nn_models\T_section1_plus\As1\model.keras", compile=False),
                'scaler_X': joblib.load(r"nn_models\T_section1_plus\As1\scaler_X.pkl"),
                'scaler_y': joblib.load(r"nn_models\T_section1_plus\As1\scaler_y.pkl")
            },
            'cost': {
                'model': tf.keras.models.load_model(r"nn_models\T_section1_plus\cost\model.keras", compile=False),
                'scaler_X': joblib.load(r"nn_models\T_section1_plus\cost\scaler_X.pkl"),
                'scaler_y': joblib.load(r"nn_models\T_section1_plus\cost\scaler_y.pkl")
            }
        },
        'section2': {
            'As1': {
                'model': tf.keras.models.load_model(r"nn_models\T_section2_plus\As1\model.keras", compile=False),
                'scaler_X': joblib.load(r"nn_models\T_section2_plus\As1\scaler_X.pkl"),
                'scaler_y': joblib.load(r"nn_models\T_section2_plus\As1\scaler_y.pkl")
            },
            'As2': {
                'model': tf.keras.models.load_model(r"nn_models\T_section2_plus\As2\model.keras", compile=False),
                'scaler_X': joblib.load(r"nn_models\T_section2_plus\As2\scaler_X.pkl"),
                'scaler_y': joblib.load(r"nn_models\T_section2_plus\As2\scaler_y.pkl")
            },
            'cost': {
                'model': tf.keras.models.load_model(r"nn_models\T_section2_plus\cost\model.keras", compile=False),
                'scaler_X': joblib.load(r"nn_models\T_section2_plus\cost\scaler_X.pkl"),
                'scaler_y': joblib.load(r"nn_models\T_section2_plus\cost\scaler_y.pkl")
            }
        }
    }

    # Generate all combinations
    combinations = list(product(fi_values, fck_values))
    n_combinations = len(combinations)
    
    # Prepare batch data
    fi_batch = np.array([c[0] for c in combinations])
    fck_batch = np.array([c[1] for c in combinations])
    a1_batch = cnom + fistr + fi_batch / 2
    d_batch = h - a1_batch
    
    # Check section types for all combinations
    section_types = check_section_type_batch(fck_batch, fyk, beff, d_batch, h, MEd)
    
    # Prepare DataFrames for batch prediction
    common_data = {
        'MEd': np.full(n_combinations, MEd),
        'beff': np.full(n_combinations, beff),
        'bw': np.full(n_combinations, bw),
        'h': np.full(n_combinations, h),
        'hf': np.full(n_combinations, hf),
        'cnom': np.full(n_combinations, cnom),
        'fistr': np.full(n_combinations, fistr)
    }
    
    # Initialize results
    results = {
        'fi': fi_batch,
        'fck': fck_batch,
        'type': section_types,
        'As1': np.full(n_combinations, np.nan),
        'As2': np.full(n_combinations, np.nan),
        'cost': np.full(n_combinations, np.nan),
        'valid': np.full(n_combinations, False)
    }
    
    # Process section1 and section2 separately
    for typ in [1, 2]:
        mask = np.array([t == typ for t in section_types])
        if not np.any(mask):
            continue
            
        # Prepare batch data for this type
        typ_data = {
            **common_data,
            'fi': fi_batch[mask],
            'fck': fck_batch[mask],
            'a1': a1_batch[mask],
            'd': d_batch[mask]
        }
        
        # Create DataFrame
        features = ['MEd', 'beff', 'bw', 'h', 'hf', 'fi', 'fck', 'cnom', 'fistr', 'a1', 'd']
        X_df = pd.DataFrame({k: typ_data[k] for k in features}, columns=features)
        
        if typ == 1:
            # Predict As1 and cost for section1
            As1_pred = batch_predict(model_dict['section1']['As1'], X_df)
            cost_pred = batch_predict(model_dict['section1']['cost'], X_df)
            
            results['As1'][mask] = As1_pred
            results['cost'][mask] = cost_pred
            
            # Check if rods fit for valid predictions
            for i in np.where(mask)[0]:
                if not np.isnan(As1_pred[mask][i]):
                    num_rods, _ = calculate_number_of_rods(As1_pred[mask][i], fi_batch[i])
                    rods_fit = check_rods_fit(bw, cnom, num_rods, fi_batch[i], smax)
                    results['valid'][i] = rods_fit and not np.isnan(cost_pred[mask][i])
        else:
            # Predict As1, As2 and cost for section2
            As1_pred = batch_predict(model_dict['section2']['As1'], X_df)
            As2_pred = batch_predict(model_dict['section2']['As2'], X_df)
            cost_pred = batch_predict(model_dict['section2']['cost'], X_df)
            
            results['As1'][mask] = As1_pred
            results['As2'][mask] = As2_pred
            results['cost'][mask] = cost_pred
            
            # Check if rods fit for valid predictions
            for i in np.where(mask)[0]:
                if not np.isnan(As1_pred[mask][i]) and not np.isnan(As2_pred[mask][i]):
                    num_rods_tension, _ = calculate_number_of_rods(As1_pred[mask][i], fi_batch[i])
                    num_rods_compression, _ = calculate_number_of_rods(As2_pred[mask][i], fi_batch[i])
                    
                    # Check both tension and compression rods fit
                    tension_fit = check_rods_fit(bw, cnom, num_rods_tension, fi_batch[i], smax)
                    compression_fit = check_rods_fit(bw, cnom, num_rods_compression, fi_batch[i], smax)
                    
                    results['valid'][i] = tension_fit and compression_fit and not np.isnan(cost_pred[mask][i])
    
    # Find the combination with minimum valid cost
    valid_mask = results['valid']
    if np.any(valid_mask):
        min_idx = np.nanargmin(results['cost'][valid_mask])
        optimal_idx = np.where(valid_mask)[0][min_idx]
        
        # Calculate number of rods for the optimal solution
        optimal_fi = results['fi'][optimal_idx]
        optimal_As1 = results['As1'][optimal_idx]
        num_rods_tension, provided_As1 = calculate_number_of_rods(optimal_As1, optimal_fi)
        
        if results['type'][optimal_idx] == 2:
            optimal_As2 = results['As2'][optimal_idx]
            num_rods_compression, provided_As2 = calculate_number_of_rods(optimal_As2, optimal_fi)
        else:
            optimal_As2 = None
            num_rods_compression = 0
            provided_As2 = 0.0
        
        optimal_params = {
            'fck': results['fck'][optimal_idx],
            'fi': optimal_fi,
            'cost': results['cost'][optimal_idx],
            'type': results['type'][optimal_idx],
            'As1': optimal_As1,
            'As2': optimal_As2,
            'num_rods_tension': num_rods_tension,
            'provided_As1': provided_As1,
            'num_rods_compression': num_rods_compression,
            'provided_As2': provided_As2 if results['type'][optimal_idx] == 2 else None
        }
        return optimal_params
    else:
        return None

# Input parameters
MEd = 6000
beff = 2000
bw = 800
h = 1200
hf = 250
fi_values = [8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32]
fck_values = [12, 16, 20, 25, 30, 35, 40, 45, 50]
cnom = 30
fistr = 8
smax = 20  # maximum spacing between bars (mm)

# Find optimal combination
optimal = find_optimal_combination(MEd, beff, bw, h, hf, fi_values, fck_values, cnom, fistr, smax=smax)

if optimal:
    print("\nOptimal combination found:")
    print(f"Concrete strength (fck): {optimal['fck']} MPa")
    print(f"Rebar diameter (fi): {optimal['fi']} mm")
    print(f"Section type: {optimal['type']}")
    print(f"Minimum cost: {optimal['cost']}")
    print(f"\nTension reinforcement:")
    print(f"Required As1: {optimal['As1']:.2f} mm²")
    print(f"Number of bars: {optimal['num_rods_tension']}")
    print(f"Provided As1: {optimal['provided_As1']:.2f} mm²")
    
    if optimal['type'] == 2:
        print(f"\nCompression reinforcement:")
        print(f"Required As2: {optimal['As2']:.2f} mm²")
        print(f"Number of bars: {optimal['num_rods_compression']}")
        print(f"Provided As2: {optimal['provided_As2']:.2f} mm²")
else:
    print("No valid combination found with the given parameters.")
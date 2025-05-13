import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from itertools import product
from tqdm import tqdm

def calculate_section_cost(b: float, h: float, f_ck: float, A_s1: float, A_s2: float) -> float:
    concrete_cost_by_class = {
        8: 230, 12: 250, 16: 300, 20: 350, 25: 400, 30: 450, 35: 500, 40: 550, 45: 600, 50: 650, 55: 700, 60: 800
    }
    
    steel_cost_by_weight = 5  # zł/kg
    steel_density = 7900      # kg/m3
    
    steel_area = (A_s1 + A_s2) / 1_000_000  # mm^2 -> m^2
    steel_weight = steel_area * steel_density
    steel_cost = steel_weight * steel_cost_by_weight
    
    concrete_area = (b * h) / 1_000_000 - steel_area
    f_ck_int = int(f_ck)
    concrete_cost = concrete_area * concrete_cost_by_class[f_ck_int]
    
    total_cost = steel_cost + concrete_cost
    
    return total_cost

def predict_section_batch(input_data: pd.DataFrame, model_name: str):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"nn_models\rect_section\Mcr_model\model.keras",
            'scaler_X': r"nn_models\rect_section\Mcr_model\scaler_X.pkl",
            'scaler_y': r"nn_models\rect_section\Mcr_model\scaler_y.pkl"
        },
        'MRd': {
            'model': r"nn_models\rect_section\MRd_model\model.keras",
            'scaler_X': r"nn_models\rect_section\MRd_model\scaler_X.pkl",
            'scaler_y': r"nn_models\rect_section\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"nn_models\rect_section\Wk_model\model.keras",
            'scaler_X': r"nn_models\rect_section\Wk_model\scaler_X.pkl",
            'scaler_y': r"nn_models\rect_section\Wk_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr': ['b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
        'MRd': ['b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
        'Wk': ['MEd', 'b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2']
    }
    
    try:
        model_info = MODEL_PATHS[model_name]
        model = tf.keras.models.load_model(model_info['model'], compile=False)
        X_scaler = joblib.load(model_info['scaler_X'])
        y_scaler = joblib.load(model_info['scaler_y'])

        X = input_data[MODEL_FEATURES[model_name]]
        X_scaled = X_scaler.transform(np.log(X + 1e-8))
        pred_scaled = model.predict(X_scaled)
        return np.exp(y_scaler.inverse_transform(pred_scaled)).flatten()
    except Exception as e:
        print(f"⚠️ Error in {model_name}: {e}")
        return np.full(len(input_data), np.nan)

def calc_max_rods(b: float, fi: float, cnom: float) -> int:
    smin = max(20, fi)
    usable_width = b - 2 * cnom
    return math.floor((usable_width + smin) / (fi + smin))

def generate_all_combinations(MEd: float, b: float, h: float, cnom: float):
    possible_fck = [16, 20, 25, 30, 35, 40, 45, 50]
    possible_fi = [8, 10, 12, 14, 16, 20, 25, 28, 32]
    
    all_combinations = []
    for fck, fi in product(possible_fck, possible_fi):
        n_max = calc_max_rods(b, fi, cnom)
        for n1, n2 in product(range(n_max), range(n_max)):
            all_combinations.append({
                'MEd': MEd,
                'b': b,
                'h': h,
                'cnom': cnom,
                'fck': fck,
                'fi': fi,
                'n1': n1,
                'n2': n2
            })
    return pd.DataFrame(all_combinations)

def process_combinations_batch(combinations_df: pd.DataFrame, wk_max: float, MEd: float):
    # Calculate derived parameters
    combinations_df['d'] = combinations_df['h'] - combinations_df['cnom'] - combinations_df['fi'] / 2
    combinations_df['As1'] = combinations_df['n1'] * (combinations_df['fi']**2) * math.pi / 4
    combinations_df['As2'] = combinations_df['n2'] * (combinations_df['fi']**2) * math.pi / 4
    combinations_df['ro1'] = combinations_df['As1'] / (combinations_df['b'] * combinations_df['d']).replace(np.inf, 0)
    combinations_df['ro2'] = combinations_df['As2'] / (combinations_df['b'] * combinations_df['d']).replace(np.inf, 0)
    
    # Batch predictions
    print("Running batch predictions...")
    combinations_df['Mcr'] = predict_section_batch(combinations_df, 'Mcr')
    combinations_df['MRd'] = predict_section_batch(combinations_df, 'MRd')
    combinations_df['Wk'] = predict_section_batch(combinations_df, 'Wk')
    
    # Calculate costs
    combinations_df['Cost'] = combinations_df.apply(
        lambda row: calculate_section_cost(row['b'], row['h'], row['fck'], row['As1'], row['As2']), 
        axis=1
    )
    
    # Filter valid solutions
    valid_solutions = combinations_df[
        (combinations_df['Wk'] < wk_max) & 
        (combinations_df['MRd'] > MEd)
    ].copy()
    
    return valid_solutions

def find_optimal_solution(MEqp: float, MEd: float, b: float, h: float, cnom: float, wk_max: float):
    # Generate all possible combinations
    print("Generating all combinations...")
    all_combinations = generate_all_combinations(MEqp, b, h, cnom)
    
    # Process in batches
    valid_solutions = process_combinations_batch(all_combinations, wk_max, MEd)
    
    if valid_solutions.empty:
        print("No valid solutions found that satisfy wk < wk_max and MRd > MEd")
        return None
    
    # Find optimal solution
    optimal_idx = valid_solutions['Cost'].idxmin()
    optimal_solution = valid_solutions.loc[optimal_idx].to_dict()
    
    return optimal_solution

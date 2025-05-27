import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from itertools import product
from tqdm import tqdm

def calc_cost_n1(b: float, h: float, f_ck: float, A_s1: float) -> float:
    concrete_cost_by_class = {
        8: 230, 12: 250, 16: 300, 20: 350, 25: 400, 30: 450, 35: 500, 40: 550, 45: 600, 50: 650, 55: 700, 60: 800
    }
    
    steel_cost_by_weight = 5  # zł/kg
    steel_density = 7900      # kg/m3
    
    steel_area = (A_s1) / 1_000_000  # mm^2 -> m^2
    steel_weight = steel_area * steel_density
    steel_cost = steel_weight * steel_cost_by_weight
    
    concrete_area = (b * h) / 1_000_000 - steel_area
    f_ck_int = int(f_ck)
    concrete_cost = concrete_area * concrete_cost_by_class[f_ck_int]
    
    total_cost = steel_cost + concrete_cost
    
    return total_cost

def calc_cost_n2(b: float, h: float, f_ck: float, A_s1: float, A_s2: float) -> float:
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

def predict_section_batch_n1(input_data: pd.DataFrame, model_name: str):
    MODEL_PATHS = {
        'MRd': {
            'model': r"neural_networks\rect_section_n1\models\MRd_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\MRd_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"neural_networks\rect_section_n1\models\Wk_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\Wk_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\Wk_model\scaler_y.pkl"
        },
        'Cost': {
            'model': r"neural_networks\rect_section_n1\models\Cost_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\Cost_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\Cost_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'MRd':  ["b", "h", "d", "fi", "fck", "ro1"],
        'Wk':   ["MEqp", "b", "h", "d", "fi", "fck", "ro1"],
        'Cost': ["b", "h", "d", "fi", "fck", "ro1"]
    }
    
    try:
        model_info = MODEL_PATHS[model_name]
        model = tf.keras.models.load_model(model_info['model'], compile=False)
        X_scalers_dict = joblib.load(model_info['scaler_X'])  # This is a dictionary of scalers
        y_scaler = joblib.load(model_info['scaler_y'])

        # Get the features in the correct order
        features = MODEL_FEATURES[model_name]
        X = input_data[features]
        
        # Apply log transform to each feature
        X_log = np.log(X + 1e-8)
        
        # Scale each feature with its respective scaler
        X_scaled = np.zeros_like(X_log)
        for i, feature in enumerate(features):
            scaler = X_scalers_dict[feature]  # Get the specific scaler for this feature
            X_scaled[:, i] = scaler.transform(X_log[feature].values.reshape(-1, 1)).flatten()
        
        # Make prediction
        pred_scaled = model.predict(X_scaled)
        
        # Inverse transform the prediction
        return np.exp(y_scaler.inverse_transform(pred_scaled)).flatten()
    except Exception as e:
        print(f"⚠️ Error in {model_name}: {e}")
        return np.full(len(input_data), np.nan)

def predict_section_batch_n2(input_data: pd.DataFrame, model_name: str):
    MODEL_PATHS = {
        'MRd': {
            'model': r"neural_networks\rect_section_n2\models\MRd_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n2\models\MRd_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n2\models\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"neural_networks\rect_section_n2\models\Wk_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n2\models\Wk_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n2\models\Wk_model\scaler_y.pkl"
        },
        'Cost': {
            'model': r"neural_networks\rect_section_n2\models\Cost_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n2\models\Cost_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n2\models\Cost_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'MRd':  ["b", "h", "d", "fi", "fck", "ro1", "ro2"],
        'Wk':   ["MEqp", "b", "h", "d", "fi", "fck", "ro1", "ro2"],
        'Cost': ["b", "h", "d", "fi", "fck", "ro1", "ro2"]
    }
    
    try:
        model_info = MODEL_PATHS[model_name]
        model = tf.keras.models.load_model(model_info['model'], compile=False)
        X_scalers_dict = joblib.load(model_info['scaler_X'])  # This is a dictionary of scalers
        y_scaler = joblib.load(model_info['scaler_y'])

        # Get the features in the correct order
        features = MODEL_FEATURES[model_name]
        X = input_data[features]
        
        # Apply log transform to each feature
        X_log = np.log(X + 1e-8)
        
        # Scale each feature with its respective scaler
        X_scaled = np.zeros_like(X_log)
        for i, feature in enumerate(features):
            scaler = X_scalers_dict[feature]  # Get the specific scaler for this feature
            X_scaled[:, i] = scaler.transform(X_log[feature].values.reshape(-1, 1)).flatten()
        
        # Make prediction
        pred_scaled = model.predict(X_scaled)
        
        # Inverse transform the prediction
        return np.exp(y_scaler.inverse_transform(pred_scaled)).flatten()
    except Exception as e:
        print(f"⚠️ Error in {model_name}: {e}")
        return np.full(len(input_data), np.nan)

def calc_max_rods(b: float, fi: float, cnom: float) -> int:
    smin = max(20, fi)
    usable_width = b - 2 * cnom
    return math.floor((usable_width + smin) / (fi + smin))

def generate_all_combinations_n1(MEd: float, b: float, h: float, cnom: float):
    possible_fck = [16, 20, 25, 30, 35, 40, 45, 50]
    possible_fi = [8, 10, 12, 14, 16, 20, 25, 28, 32]
    
    all_combinations = []
    for fck, fi in product(possible_fck, possible_fi):
        n_max = calc_max_rods(b, fi, cnom)
        for n1 in range(0, n_max):
            all_combinations.append({
                'MEqp': MEd,
                'b': b,
                'h': h,
                'fck': fck,
                'fi': fi,
                'n1': n1,
                'cnom': cnom  # Store cnom but won't be used in predictions
            })
    return pd.DataFrame(all_combinations)

def generate_all_combinations_n2(MEd: float, b: float, h: float, cnom: float):
    possible_fck = [16, 20, 25, 30, 35, 40, 45, 50]
    possible_fi = [8, 10, 12, 14, 16, 20, 25, 28, 32]
    
    all_combinations = []
    for fck, fi in product(possible_fck, possible_fi):
        n_max = calc_max_rods(b, fi, cnom)
        for n1, n2 in product(range(n_max), range(n_max)):
            all_combinations.append({
                'MEqp': MEd,
                'b': b,
                'h': h,
                'fck': fck,
                'fi': fi,
                'n1': n1,
                'n2': n2,
                'cnom': cnom  # Store cnom but won't be used in predictions
            })
    return pd.DataFrame(all_combinations)

def process_combinations_batch_n1(combinations_df: pd.DataFrame, wk_max: float, MEd: float):
    # Calculate derived parameters (using cnom for d calculation)
    combinations_df['d'] = combinations_df['h'] - combinations_df['cnom'] - combinations_df['fi'] / 2
    combinations_df['As1'] = combinations_df['n1'] * (combinations_df['fi']**2) * math.pi / 4
    combinations_df['As2'] = 0
    combinations_df['ro1'] = combinations_df['As1'] / (combinations_df['b'] * combinations_df['h']).replace(np.inf, 0)
    combinations_df['ro2'] = 0
    combinations_df['n2'] = 0
    
    # Batch predictions (cnom is not in MODEL_FEATURES so it won't be used)
    print("Running batch predictions...")
    combinations_df['MRd'] = predict_section_batch_n1(combinations_df, 'MRd')
    combinations_df['Wk'] = predict_section_batch_n1(combinations_df, 'Wk')
    
    # Calculate costs
    combinations_df['Cost'] = combinations_df.apply(
        lambda row: calc_cost_n1(row['b'], row['h'], row['fck'], row['As1']), 
        axis=1
    )
    
    # Filter valid solutions
    valid_solutions = combinations_df[
        (combinations_df['Wk'] < wk_max) & 
        (combinations_df['MRd'] > MEd)
    ].copy()
    
    return valid_solutions

def process_combinations_batch_n2(combinations_df: pd.DataFrame, wk_max: float, MEd: float):
    # Calculate derived parameters (using cnom for d calculation)
    combinations_df['d'] = combinations_df['h'] - combinations_df['cnom'] - combinations_df['fi'] / 2
    combinations_df['As1'] = combinations_df['n1'] * (combinations_df['fi']**2) * math.pi / 4
    combinations_df['As2'] = combinations_df['n2'] * (combinations_df['fi']**2) * math.pi / 4
    combinations_df['ro1'] = combinations_df['As1'] / (combinations_df['b'] * combinations_df['h']).replace(np.inf, 0)
    combinations_df['ro2'] = combinations_df['As2'] / (combinations_df['b'] * combinations_df['h']).replace(np.inf, 0)
    
    # Batch predictions (cnom is not in MODEL_FEATURES so it won't be used)
    print("Running batch predictions...")
    combinations_df['MRd'] = predict_section_batch_n2(combinations_df, 'MRd')
    combinations_df['Wk'] = predict_section_batch_n2(combinations_df, 'Wk')
    
    # Calculate costs
    combinations_df['Cost'] = combinations_df.apply(
        lambda row: calc_cost_n2(row['b'], row['h'], row['fck'], row['As1'], row['As2']), 
        axis=1
    )
    
    # Filter valid solutions
    valid_solutions = combinations_df[
        (combinations_df['Wk'] < wk_max) & 
        (combinations_df['MRd'] > MEd)
    ].copy()
    
    return valid_solutions

def find_optimal_solution_n1(MEqp: float, MEd: float, b: float, h: float, cnom: float, wk_max: float):
    # Generate all possible combinations
    print("Generating all combinations...")
    all_combinations = generate_all_combinations_n1(MEqp, b, h, cnom)
    
    # Process in batches
    valid_solutions = process_combinations_batch_n1(all_combinations, wk_max, MEd)
    
    if valid_solutions.empty:
        print("No valid solutions found that satisfy wk < wk_max and MRd > MEd")
        return None
    
    # Find optimal solution
    optimal_idx = valid_solutions['Cost'].idxmin()
    optimal_solution = valid_solutions.loc[optimal_idx].to_dict()
    
    return optimal_solution

def find_optimal_solution_n2(MEqp: float, MEd: float, b: float, h: float, cnom: float, wk_max: float):
    # Generate all possible combinations
    print("Generating all combinations...")
    all_combinations = generate_all_combinations_n2(MEqp, b, h, cnom)
    
    # Process in batches
    valid_solutions = process_combinations_batch_n2(all_combinations, wk_max, MEd)
    
    if valid_solutions.empty:
        print("No valid solutions found that satisfy wk < wk_max and MRd > MEd")
        return None
    
    # Find optimal solution
    optimal_idx = valid_solutions['Cost'].idxmin()
    optimal_solution = valid_solutions.loc[optimal_idx].to_dict()
    
    return optimal_solution

def find_best_solution(MEqp: float, MEd: float, b: float, h: float, wk_max: float, cnom: float):
    """Find the best solution between both models (with and without ro2)"""
    print("=== Evaluating solutions without ro2 ===")
    solution_n1 = find_optimal_solution_n1(MEqp, MEd, b, h, cnom, wk_max)
    
    print("\n=== Evaluating solutions with ro2 ===")
    solution_n2 = find_optimal_solution_n2(MEqp, MEd, b, h, cnom, wk_max)
    
    if solution_n1 is None and solution_n2 is None:
        print("No valid solutions found in either model")
        return None
    elif solution_n1 is None:
        print("Only found valid solution with ro2")
        return solution_n2
    elif solution_n2 is None:
        print("Only found valid solution without ro2")
        return solution_n1
    else:
        # Compare costs
        if solution_n1['Cost'] < solution_n2['Cost']:
            print(f"Solution without ro2 is cheaper ({solution_n1['Cost']} vs {solution_n2['Cost']})")
            print(f" n1={solution_n1}")
            print(f" n2={solution_n2}")
            return solution_n1
        else:
            print(f" n1={solution_n1}")
            print(f" n2={solution_n2}")
            print(f"Solution with ro2 is cheaper ({solution_n2['Cost']} vs {solution_n1['Cost']})")
            return solution_n2
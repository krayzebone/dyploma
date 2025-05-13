import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from itertools import product
from tqdm import tqdm

def calc_cost(
    beff: float,
    bw: float,
    h: float,
    hf: float,
    fck: float,
    As1: float,
    As2: float = 0,  # Make As2 optional with default 0
) -> float:
    """Concrete + steel cost in PLN."""
    concrete_cost_by_class = {
        8: 230,
        12: 250,
        16: 300,
        20: 350,
        25: 400,
        30: 450,
        35: 500,
        40: 550,
        45: 600,
        50: 650,
        55: 700,
        60: 800,
    }

    steel_cost = (As1 + As2) / 1_000_000 * 7_900 * 5  # mm²→m² * ρ * price
    conc_area = ((beff * hf) + (h - hf) * bw) / 1_000_000 - (As1 + As2) / 1_000_000
    conc_cost = conc_area * concrete_cost_by_class[int(fck)]
    return steel_cost + conc_cost

def predict_section_batchn1(input_data: pd.DataFrame, model_name: str):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"neural_networks\Tsectionn1\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn1\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn1\models\Mcr_model\scaler_y.pkl"
        },
        'MRd': {
            'model': r"neural_networks\Tsectionn1\models\MRd_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn1\models\MRd_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn1\models\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"neural_networks\Tsectionn1\models\Wk_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn1\models\Wk_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn1\models\Wk_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr': ["beff", "bw", "h", "hf", "fi", "fck", "ro1"],
        'MRd': ["beff", "bw", "h", "hf", "cnom", "d", "fi", "fck", "ro1"],
        'Wk': ["MEd", "beff", "bw", "h", "hf", 'cnom', 'd', "fi", "fck", "ro1"]
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

def predict_section_batchn2(input_data: pd.DataFrame, model_name: str):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"neural_networks\Tsectionn2\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn2\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn2\models\Mcr_model\scaler_y.pkl"
        },
        'MRd': {
            'model': r"neural_networks\Tsectionn2\models\MRd_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn2\models\MRd_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn2\models\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"neural_networks\Tsectionn2\models\Wk_model\model.keras",
            'scaler_X': r"neural_networks\Tsectionn2\models\Wk_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\Tsectionn2\models\Wk_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr': ["beff", "bw", "h", "hf", "fi", "fck", "ro1", "ro2"],
        'MRd': ["beff", "bw", "h", "hf", "cnom", "d", "fi", "fck", "ro1", "ro2"],
        'Wk': ["MEd", "beff", "bw", "h", "hf", 'cnom', 'd', "fi", "fck", "ro1", "ro2"]
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


def calc_max_rodsn1(bw: float, fi: float, cnom: float) -> int:
    smax = max(20, fi)
    pitch = fi + smax
    available_space = bw - 2*cnom + smax
    n_max = math.floor(available_space / pitch)
    return n_max

def calc_max_rodsn2(bw: float, fi: float, cnom: float) -> int:
    smax = max(20, fi)
    pitch = fi + smax
    available_space = bw - 2*cnom + smax
    n_max = 2 * math.floor(available_space / pitch)
    return n_max


def generate_all_combinationsn1(MEd: float, beff: float, bw: float, h: float, hf: float, cnom: float):
    possible_fck = [16, 20, 25, 30, 35, 40, 45, 50]
    possible_fi = [8, 10, 12, 14, 16, 20, 25, 28, 32]
    
    all_combinations = []
    for fck, fi in product(possible_fck, possible_fi):
        n_max = calc_max_rodsn1(bw, fi, cnom)
        for n1 in range(1, n_max + 1):  # Fixed range iteration
            all_combinations.append({
                'MEd': MEd,
                'beff': beff,
                'bw': bw,
                'h': h,
                'hf': hf,
                'cnom': cnom,
                'fck': fck,
                'fi': fi,
                'n1': n1,
            })
    return pd.DataFrame(all_combinations)

def generate_all_combinationsn2(MEd: float, beff: float, bw: float, h: float, hf: float, cnom: float):
    possible_fck = [16, 20, 25, 30, 35, 40, 45, 50]
    possible_fi = [8, 10, 12, 14, 16, 20, 25, 28, 32]
    
    all_combinations = []
    for fck, fi in product(possible_fck, possible_fi):
        n_max = calc_max_rodsn1(bw, fi, cnom)
        for n1 in range(1, n_max + 1):  # Fixed range iteration
            for n2 in range(2, n_max + 1):  # Assuming minimum 2 bars for compression reinforcement
                all_combinations.append({
                    'MEd': MEd,
                    'beff': beff,
                    'bw': bw,
                    'h': h,
                    'hf': hf,
                    'cnom': cnom,
                    'fck': fck,
                    'fi': fi,
                    'n1': n1,
                    'n2': n2,
                })
    return pd.DataFrame(all_combinations)


def process_combinations_batchn1(combinations_df: pd.DataFrame, wk_max: float, MEd: float):
    # Calculate derived parameters
    combinations_df['d'] = combinations_df['h'] - combinations_df['cnom'] - combinations_df['fi'] / 2
    combinations_df['As1'] = combinations_df['n1'] * (combinations_df['fi']**2) * math.pi / 4
    combinations_df['ro1'] = combinations_df['As1'] / (combinations_df['beff'] * combinations_df['hf'] + combinations_df['bw'] * (combinations_df['h'] - combinations_df['hf']))
    
    # Batch predictions
    print("Running batch predictions for model without ro2...")
    combinations_df['Mcr'] = predict_section_batchn1(combinations_df, 'Mcr')
    combinations_df['MRd'] = predict_section_batchn1(combinations_df, 'MRd')
    combinations_df['Wk'] = predict_section_batchn1(combinations_df, 'Wk')
    
    # Calculate costs
    combinations_df['Cost'] = combinations_df.apply(
        lambda row: calc_cost(row['beff'], row['bw'], row['h'], row['hf'], row['fck'], row['As1']), 
        axis=1
    )

    combinations_df.to_csv("combinations_results_n1.csv", index=False)
    print("✅ Saved all combinations to 'combinations_results_n1.csv'")
    
    # Filter valid solutions
    valid_solutions = combinations_df[
        (combinations_df['Wk'] < wk_max) &
        (combinations_df['MRd'] > MEd)
    ].copy()
    
    return valid_solutions

def process_combinations_batchn2(combinations_df: pd.DataFrame, wk_max: float, MEd: float):
    # Calculate derived parameters
    combinations_df['d'] = combinations_df['h'] - combinations_df['cnom'] - combinations_df['fi'] / 2
    combinations_df['As1'] = combinations_df['n1'] * (combinations_df['fi']**2) * math.pi / 4
    combinations_df['As2'] = combinations_df['n2'] * (combinations_df['fi']**2) * math.pi / 4
    combinations_df['ro1'] = combinations_df['As1'] / (combinations_df['beff'] * combinations_df['hf'] + combinations_df['bw'] * (combinations_df['h'] - combinations_df['hf']))
    combinations_df['ro2'] = combinations_df['As2'] / (combinations_df['beff'] * combinations_df['hf'] + combinations_df['bw'] * (combinations_df['h'] - combinations_df['hf']))
    
    # Batch predictions
    print("Running batch predictions for model with ro2...")
    combinations_df['Mcr'] = predict_section_batchn2(combinations_df, 'Mcr')
    combinations_df['MRd'] = predict_section_batchn2(combinations_df, 'MRd')
    combinations_df['Wk'] = predict_section_batchn2(combinations_df, 'Wk')
    
    # Calculate costs
    combinations_df['Cost'] = combinations_df.apply(
        lambda row: calc_cost(row['beff'], row['bw'], row['h'], row['hf'], row['fck'], row['As1'], row['As2']), 
        axis=1
    )

    combinations_df.to_csv("combinations_results_n2.csv", index=False)
    print("✅ Saved all combinations to 'combinations_results_n2.csv'")
    
    # Filter valid solutions
    valid_solutions = combinations_df[
        (combinations_df['Wk'] < wk_max) &
        (combinations_df['MRd'] > MEd)
    ].copy()
    
    return valid_solutions


def find_optimal_solutionn1(MEqp: float, MEd: float, beff: float, bw: float, h: float, hf: float, cnom: float, wk_max: float):
    # Generate all possible combinations
    print("Generating all combinations for model without ro2...")
    all_combinations = generate_all_combinationsn1(MEqp, beff, bw, h, hf, cnom)
    
    # Process in batches
    valid_solutions = process_combinations_batchn1(all_combinations, wk_max, MEd)
    
    if valid_solutions.empty:
        print("No valid solutions found for model without ro2")
        return None
    
    # Find optimal solution
    optimal_idx = valid_solutions['Cost'].idxmin()
    optimal_solution = valid_solutions.loc[optimal_idx].to_dict()
    
    return optimal_solution

def find_optimal_solutionn2(MEqp: float, MEd: float, beff: float, bw: float, h: float, hf: float, cnom: float, wk_max: float):
    # Generate all possible combinations
    print("Generating all combinations for model with ro2...")
    all_combinations = generate_all_combinationsn2(MEqp, beff, bw, h, hf, cnom)
    
    # Process in batches
    valid_solutions = process_combinations_batchn2(all_combinations, wk_max, MEd)
    
    if valid_solutions.empty:
        print("No valid solutions found for model with ro2")
        return None
    
    # Find optimal solution
    optimal_idx = valid_solutions['Cost'].idxmin()
    optimal_solution = valid_solutions.loc[optimal_idx].to_dict()
    
    return optimal_solution

def find_best_solution(MEqp: float, MEd: float, beff: float, bw: float, h: float, hf: float, cnom: float, wk_max: float):
    """Find the best solution between both models (with and without ro2)"""
    print("=== Evaluating solutions without ro2 ===")
    solution_n1 = find_optimal_solutionn1(MEqp, MEd, beff, bw, h, hf, cnom, wk_max)
    
    print("\n=== Evaluating solutions with ro2 ===")
    solution_n2 = find_optimal_solutionn2(MEqp, MEd, beff, bw, h, hf, cnom, wk_max)
    
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
            return solution_n1
        else:
            print(f"Solution with ro2 is cheaper ({solution_n2['Cost']} vs {solution_n1['Cost']})")
            return solution_n2

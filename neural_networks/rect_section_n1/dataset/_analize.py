

def predict_section(MEqp: float, b: float, h: float, fck: float, fi: float, cnom: float, As1: float, As2: float):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"neural_networks\rect_section_n1\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_y.pkl"
        },
        'MRd': {
            'model': r"neural_networks\rect_section_n1\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"neural_networks\rect_section_n1\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_y.pkl"
        },
        'Cost': {
            'model': r"neural_networks\rect_section_n1\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr':  ['b', 'h', 'd', 'fi', 'fck', 'ro1'],
        'MRd':  ['b', 'h', 'd', 'fi', 'fck', 'ro1'],
        'Wk':   ['MEqp', 'b', 'h', 'd', 'fi', 'fck', 'ro1'],
        'Cost': ['b', 'h', 'd', 'fi', 'fck', 'ro1']
    }
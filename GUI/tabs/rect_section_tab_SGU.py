import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QTextEdit
)
from PyQt6.QtGui import QFontMetrics

def predict_section(MEd: float, b: float, h: float, fck: float, fi: float, cnom: float, As1: float, As2: float):
    """
    Predict and print results for a concrete section.
    
    Args:
        MEd: Design moment (kNm)
        b: Section width (mm)
        h: Section height (mm)
        fck: Concrete characteristic strength (MPa)
        fi: Rebar diameter (mm)
        cnom: Concrete cover (mm)
        As1: Area of tension reinforcement (mm²)
        As2: Area of compression reinforcement (mm²)
    """
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
        },
        'Cost': {
            'model': r"nn_models\rect_section\cost_model\model.keras",
            'scaler_X': r"nn_models\rect_section\cost_model\scaler_X.pkl",
            'scaler_y': r"nn_models\rect_section\cost_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr': ['b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
        'MRd': ['b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
        'Wk': ['MEd', 'b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2'],
        'Cost': ['MEd', 'b', 'h', 'd', 'fi', 'fck', 'ro1', 'ro2']
    }
    
    # Calculate derived parameters
    d = h - cnom - fi / 2  # Effective depth
    ro1 = As1 / (b * d) if (b * d) > 0 else 0  # Reinforcement ratio (using d)
    ro2 = As2 / (b * d) if (b * d) > 0 else 0   # Reinforcement ratio (using d)
    
    # Create a dictionary of all possible features
    feature_values = {
        'MEd': float(MEd),
        'b': float(b),
        'h': float(h),
        'd': float(d),
        'fi': float(fi),
        'fck': float(fck),
        'ro1': float(ro1),
        'ro2': float(ro2)
    }
    
    results = {}
    for model_name in ['Mcr', 'MRd', 'Wk', 'Cost']:
        try:
            # Load model and scalers
            model = tf.keras.models.load_model(MODEL_PATHS[model_name]['model'], compile=False)
            X_scaler = joblib.load(MODEL_PATHS[model_name]['scaler_X'])
            y_scaler = joblib.load(MODEL_PATHS[model_name]['scaler_y'])
            
            # Prepare input - select only the needed features and create a DataFrame row
            X_values = [feature_values[feature] for feature in MODEL_FEATURES[model_name]]
            X = pd.DataFrame([X_values], columns=MODEL_FEATURES[model_name])
            
            # Apply log transform and scale
            X_scaled = X_scaler.transform(np.log(X + 1e-8))
            
            # Predict and inverse transform
            pred_scaled = model.predict(X_scaled)
            pred = np.exp(y_scaler.inverse_transform(pred_scaled))[0][0]
            results[model_name] = pred
        except Exception as e:
            print(f"⚠️ Error in {model_name}: {str(e)}")
            results[model_name] = None
    
    return results

class RectSectionTabSGU(QObject):
    data_updated = pyqtSignal()  # Signal for data changes
    
    def __init__(self):
        super().__init__()
        self.MEd = None
        self.b = None
        self.h = None
        self.fi = None
        self.fck = None
        self.As1 = None
        self.As2 = None
        self.cnom = None  # Concrete cover
    
    def update_data(self, **kwargs):
        """Update multiple values at once and emit signal"""
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)
        self.data_updated.emit()

class RectSectionTabSGU(QWidget):
    def __init__(self, data_store: CalculationData, parent=None):
        super().__init__(parent)
        self.data_store = data_store
        self._build_ui()
        self._update_display()
        
        # Connect to data updates
        self.data_store.data_updated.connect(self._update_display)
        
    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Neural Network Predictions")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Input parameters display
        input_group = QGroupBox("Input Parameters")
        input_layout = QVBoxLayout()
        self.input_display = QTextEdit()
        self.input_display.setReadOnly(True)
        self.input_display.setMinimumHeight(150)
        input_layout.addWidget(self.input_display)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Prediction results
        result_group = QGroupBox("Prediction Results")
        result_layout = QVBoxLayout()
        self.prediction_display = QTextEdit()
        self.prediction_display.setReadOnly(True)
        self.prediction_display.setMinimumHeight(200)
        result_layout.addWidget(self.prediction_display)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Run Predictions")
        self.predict_btn.clicked.connect(self._run_predictions)
        btn_layout.addWidget(self.predict_btn)
        
        self.update_btn = QPushButton("Refresh Inputs")
        self.update_btn.clicked.connect(self._update_display)
        btn_layout.addWidget(self.update_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def _update_display(self):
        """Update the input parameters display"""
        missing_data = []
        for attr in ['MEd', 'b', 'h', 'fi', 'fck', 'As1', 'cnom']:
            if getattr(self.data_store, attr, None) is None:
                missing_data.append(attr)
        
        if missing_data:
            self.input_display.setPlainText(
                f"Waiting for data from Optimizer tab...\n\n"
                f"Missing parameters: {', '.join(missing_data)}"
            )
            self.predict_btn.setEnabled(False)
            return
        
        text = f"""MEd = {self.data_store.MEd:.2f} kNm
b = {self.data_store.b:.1f} mm
h = {self.data_store.h:.1f} mm
fi = {self.data_store.fi:.1f} mm
fck = {self.data_store.fck:.1f} MPa
As1 = {self.data_store.As1:.1f} mm²
As2 = {getattr(self.data_store, 'As2', 0):.1f} mm²
cnom = {self.data_store.cnom:.1f} mm
"""
        self.input_display.setPlainText(text.strip())
        self.predict_btn.setEnabled(True)
    
    def _run_predictions(self):
        """Run the neural network predictions"""
        try:
            # Get all required parameters
            required_params = {
                'MEd': self.data_store.MEd,
                'b': self.data_store.b,
                'h': self.data_store.h,
                'fck': self.data_store.fck,
                'fi': self.data_store.fi,
                'cnom': self.data_store.cnom,
                'As1': self.data_store.As1,
                'As2': getattr(self.data_store, 'As2', 0)
            }
            
            if None in required_params.values():
                missing = [k for k, v in required_params.items() if v is None]
                self.prediction_display.setPlainText(
                    f"Error: Missing required parameters:\n{', '.join(missing)}"
                )
                return
            
            results = predict_section(**required_params)
            
            if not results:
                self.prediction_display.setPlainText("Error: No results returned from prediction")
                return
            
            # Format results
            result_text = "Neural Network Predictions:\n\n"
            result_text += f"Cracking Moment (Mcr): {results.get('Mcr', 'N/A'):.2f} kNm\n"
            result_text += f"Design Moment (MRd): {results.get('MRd', 'N/A'):.2f} kNm\n"
            result_text += f"Crack Width (Wk): {results.get('Wk', 'N/A'):.4f} mm\n"
            result_text += f"Estimated Cost: {results.get('Cost', 'N/A'):.2f} PLN/m\n"
            
            # Add safety checks if we have MRd and MEd
            if 'MRd' in results and self.data_store.MEd is not None:
                try:
                    safety_factor = results['MRd'] / self.data_store.MEd
                    result_text += f"\nSafety Factor (MRd/MEd): {safety_factor:.2f}"
                    if safety_factor < 1.0:
                        result_text += " ⚠️ (UNSAFE - MRd < MEd)"
                    else:
                        result_text += " ✓ (Safe)"
                except ZeroDivisionError:
                    result_text += "\nSafety Factor: N/A (MEd is zero)"
            
            self.prediction_display.setPlainText(result_text)
            
        except Exception as e:
            self.prediction_display.setPlainText(f"Prediction Error:\n{str(e)}")
from __future__ import annotations
import math
import os
import sys
from dataclasses import dataclass

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

from PyQt6.QtCore import Qt
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
)
from PyQt6.QtGui import QFontMetrics
from GUI.tabs.rect_section_tab_SGN import CalculationData

def predict_section(MEd: float, b: float, h: float, fck: float, fi: float, cnom: float, As1: float, As2: float):

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

class RectSectionTabSGU(QWidget):
    """Tab that displays the stored calculation results from CalculationData."""
    
    def __init__(self, parent: QWidget | None = None, data_store: CalculationData = None) -> None:
        super().__init__(parent)
        self.data_store = data_store if data_store else CalculationData()
        self._build_ui()
        self._update_display()
        
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        
        # Title label
        title = QLabel("Stored Calculation Results")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Add some spacing
        layout.addSpacing(20)
        
        # Create the results display group
        group = QGroupBox("Calculation Parameters and Results")
        group_layout = QVBoxLayout(group)
        
        # Create the display fields
        self.result_fields = {}
        fields = [
            ("Moment [kNm]", "MEd"),
            ("Width [mm]", "b"),
            ("Height [mm]", "h"),
            ("Reinforcement diameter [mm]", "fi"),
            ("Concrete class", "fck"),
            ("Required As1 [mm²]", "As1"),
            ("Required As2 [mm²]", "As2"),
            ("Number of tension rods", "num_rods_As1"),
            ("Number of compression rods", "num_rods_As2"),
        ]
        
        for label_text, data_key in fields:
            hbox = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(250)
            output = QLineEdit()
            output.setReadOnly(True)
            output.setAlignment(Qt.AlignmentFlag.AlignRight)
            hbox.addWidget(label)
            hbox.addWidget(output)
            group_layout.addLayout(hbox)
            self.result_fields[data_key] = output
        
        # Add stretch to push everything up
        group_layout.addStretch()
        
        # Add the group to the main layout
        layout.addWidget(group)
        
        # Add refresh button
        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.clicked.connect(self._update_display)
        layout.addWidget(self.refresh_btn)
        
        # Add stretch to push everything up
        layout.addStretch()
    
    def _update_display(self) -> None:
        """Update all fields with current data from CalculationData."""
        # Display basic parameters
        self.result_fields["MEd"].setText(
            f"{self.data_store.MEd:.2f}" if self.data_store.MEd is not None else "N/A"
        )
        self.result_fields["b"].setText(
            f"{self.data_store.b:.1f}" if self.data_store.b is not None else "N/A"
        )
        self.result_fields["h"].setText(
            f"{self.data_store.h:.1f}" if self.data_store.h is not None else "N/A"
        )
        self.result_fields["fi"].setText(
            f"{self.data_store.fi:.1f}" if self.data_store.fi is not None else "N/A"
        )
        
        # Display concrete class if available
        fck_text = "N/A"
        if self.data_store.fck is not None:
            fck_text = f"C{self.data_store.fck}/{self.data_store.fck + 5}"
        self.result_fields["fck"].setText(fck_text)
        
        # Display reinforcement areas
        self.result_fields["As1"].setText(
            f"{self.data_store.As1:.1f}" if self.data_store.As1 is not None else "N/A"
        )
        self.result_fields["As2"].setText(
            f"{self.data_store.As2:.1f}" if self.data_store.As2 is not None else "N/A"
        )

        # Display number of rods
        self.result_fields["num_rods_As1"].setText(
            str(self.data_store.num_rods_As1) if self.data_store.num_rods_As1 is not None else "N/A"
        )
        self.result_fields["num_rods_As2"].setText(
            str(self.data_store.num_rods_As2) if self.data_store.num_rods_As2 is not None else "N/A"
        )
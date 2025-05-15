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

from .optimization_module_rect import find_best_solution

def predict_section_n1(MEqp: float, b: float, h: float, fck: float, fi: float, cnom: float, As1: float):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"neural_networks\rect_section_n1\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n1\models\Mcr_model\scaler_y.pkl"
        },
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
        'Mcr':  ["b", "h", "d", "cnom", "fi", "fck", "ro1"],
        'MRd':  ["b", "h", "d", "cnom", "fi", "fck", "ro1"],
        'Wk':   ["MEqp", "b", "h", "d", "cnom", "fi", "fck", "ro1"],
        'Cost': ["b", "h", "d", "fi", "cnom", "fck", "ro1"]
    }
    
    # Calculate derived parameters
    d = h - cnom - fi / 2 - 8
    ro1 = As1 / (b * h) if (b * h) > 0 else 0
    
    feature_values = {
        'MEqp': float(MEqp),
        'b': float(b),
        'h': float(h),
        'd': float(d),
        'cnom': float(cnom),
        'fi': float(fi),
        'fck': float(fck),
        'ro1': float(ro1),
    }
    
    results = {}
    for model_name in ['Mcr', 'MRd', 'Wk', 'Cost']:
        try:
            model_info = MODEL_PATHS[model_name]
            model = tf.keras.models.load_model(model_info['model'], compile=False)
            
            # Load scalers - X_scaler is a dictionary, y_scaler is a single scaler
            X_scalers_dict = joblib.load(model_info['scaler_X'])
            y_scaler = joblib.load(model_info['scaler_y'])

            # Prepare input data
            X_values = [feature_values[f] for f in MODEL_FEATURES[model_name]]
            X_df = pd.DataFrame([X_values], columns=MODEL_FEATURES[model_name])
            
            # Apply log transform to each feature (as done in training)
            X_log = np.log(X_df + 1e-8)
            
            # Scale each feature with its respective scaler
            X_scaled = np.zeros_like(X_log)
            for i, feature in enumerate(MODEL_FEATURES[model_name]):
                scaler = X_scalers_dict[feature]  # Get the specific scaler for this feature
                X_scaled[:, i] = scaler.transform(X_log[feature].values.reshape(-1, 1)).flatten()
            
            # Make prediction
            pred_scaled = model.predict(X_scaled)
            
            # Inverse transform the prediction
            pred = np.exp(y_scaler.inverse_transform(pred_scaled))[0][0]
            results[model_name] = pred
            
        except Exception as e:
            print(f"⚠️ Error in {model_name}: {e}")
            results[model_name] = None
    
    return results


def predict_section_n2(MEqp: float, b: float, h: float, fck: float, fi: float, cnom: float, As1: float, As2: float):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"neural_networks\rect_section_n2\models\Mcr_model\model.keras",
            'scaler_X': r"neural_networks\rect_section_n2\models\Mcr_model\scaler_X.pkl",
            'scaler_y': r"neural_networks\rect_section_n2\models\Mcr_model\scaler_y.pkl"
        },
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
        'Mcr':  ["b", "h", "d", "cnom", "fi", "fck", "ro1", "ro2"],
        'MRd':  ["b", "h", "d", "cnom", "fi", "fck", "ro1", "ro2"],
        'Wk':   ["MEqp", "b", "cnom", "h", "d", "fi", "fck", "ro1", "ro2"],
        'Cost': ["b", "h", "d", "cnom", "fi", "fck", "ro1", "ro2"]
    }
    
    # Calculate derived parameters
    d = h - cnom - fi / 2 - 8
    ro1 = As1 / (b * h) if (b * h) > 0 else 0
    ro2 = As2 / (b * h) if (b * h) > 0 else 0
    
    feature_values = {
        'MEqp': float(MEqp),
        'b': float(b),
        'h': float(h),
        'd': float(d),
        'cnom': float(cnom),
        'fi': float(fi),
        'fck': float(fck),
        'ro1': float(ro1),
        'ro2': float(ro2)
    }
    
    results = {}
    for model_name in ['Mcr', 'MRd', 'Wk', 'Cost']:
        try:
            model_info = MODEL_PATHS[model_name]
            model = tf.keras.models.load_model(model_info['model'], compile=False)
            
            # Load scalers - X_scaler is a dictionary, y_scaler is a single scaler
            X_scalers_dict = joblib.load(model_info['scaler_X'])
            y_scaler = joblib.load(model_info['scaler_y'])

            # Prepare input data
            X_values = [feature_values[f] for f in MODEL_FEATURES[model_name]]
            X_df = pd.DataFrame([X_values], columns=MODEL_FEATURES[model_name])
            
            # Apply log transform to each feature (as done in training)
            X_log = np.log(X_df + 1e-8)
            
            # Scale each feature with its respective scaler
            X_scaled = np.zeros_like(X_log)
            for i, feature in enumerate(MODEL_FEATURES[model_name]):
                scaler = X_scalers_dict[feature]  # Get the specific scaler for this feature
                X_scaled[:, i] = scaler.transform(X_log[feature].values.reshape(-1, 1)).flatten()
            
            # Make prediction
            pred_scaled = model.predict(X_scaled)
            
            # Inverse transform the prediction
            pred = np.exp(y_scaler.inverse_transform(pred_scaled))[0][0]
            results[model_name] = pred
            
        except Exception as e:
            print(f"⚠️ Error in {model_name}: {e}")
            results[model_name] = None
    
    return results


class RectSectionTabSGU(QWidget):
    """Tab that displays the stored calculation results from CalculationData."""
    
    def __init__(self, parent: QWidget | None = None, data_store: CalculationData = None) -> None:
        super().__init__(parent)
        self.data_store = data_store or CalculationData()
        self._build_ui()
        self._update_display()
        
    def _build_ui(self) -> None:
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(scroll_area)

        title = QLabel("Stored Calculation Results")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title.font()
        font.setBold(True)
        font.setPointSize(14)
        title.setFont(font)
        layout.addWidget(title)
        layout.addSpacing(20)

        # Add refresh button at the top of the content (before the calculation group)
        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.clicked.connect(self._update_display)
        layout.addWidget(self.refresh_btn)
        layout.addSpacing(10)  # Add some spacing between button and group

        # Calculation group
        group = QGroupBox("Calculation Parameters and Results")
        group_layout = QVBoxLayout(group)

        # Add MEqp input box at the top of calculation group
        hbox = QHBoxLayout()
        lbl = QLabel("MEqp for prediction [kNm]")
        lbl.setFixedWidth(250)
        self.MEqp_input = QLineEdit()
        self.MEqp_input.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.MEqp_input.setFixedHeight(28)
        hbox.addWidget(lbl)
        hbox.addWidget(self.MEqp_input)
        group_layout.addLayout(hbox)

        self.result_fields = {}
        fields = [
            ("Moment [kNm]", "MEd"),
            ("Width [mm]", "b"),
            ("Height [mm]", "h"),
            ("Reinforcement diameter [mm]", "fi"),
            ("Concrete class", "fck"),
            ("Provided As1 [mm²]", "act1"),
            ("Provided As2 [mm²]", "act2"),
            ("Number of tension rods", "num_rods_As1"),
            ("Number of compression rods", "num_rods_As2"),
        ]
        for label_text, key in fields:
            hbox = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(250)
            out = QLineEdit()
            out.setReadOnly(True)
            out.setAlignment(Qt.AlignmentFlag.AlignRight)
            out.setFixedHeight(28)
            hbox.addWidget(lbl)
            hbox.addWidget(out)
            group_layout.addLayout(hbox)
            self.result_fields[key] = out

        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self._on_predict)
        group_layout.addWidget(self.predict_btn)

        preds = [
            ("Predicted Mcr [kNm]", "pred_Mcr"),
            ("Predicted MRd [kNm]", "pred_MRd"),
            ("Predicted Wk [mm]", "pred_Wk"),
            ("Predicted Cost", "pred_Cost"),
        ]
        for txt, key in preds:
            hbox = QHBoxLayout()
            lbl = QLabel(txt)
            lbl.setFixedWidth(250)
            out = QLineEdit()
            out.setReadOnly(True)
            out.setAlignment(Qt.AlignmentFlag.AlignRight)
            out.setFixedHeight(28)
            hbox.addWidget(lbl)
            hbox.addWidget(out)
            group_layout.addLayout(hbox)
            self.result_fields[key] = out

        group_layout.addStretch()
        layout.addWidget(group)

        # Optimization section below calculation group
        self._build_optimization_group(layout)
        layout.addStretch()

    def _build_optimization_group(self, layout: QVBoxLayout) -> None:
        opt_group = QGroupBox("Optimized Section Parameters")
        opt_layout = QVBoxLayout(opt_group)

        self.optimize_btn = QPushButton("Optimize Section")
        self.optimize_btn.clicked.connect(self._on_optimize)
        opt_layout.addWidget(self.optimize_btn)

        opt_fields = [
            ("Optimized Ø [mm]", "opt_fi"),
            ("Optimized Concrete Class", "opt_fck"),
            ("Optimized n1", "opt_n1"),
            ("Optimized n2", "opt_n2"),
            ("Optimized MRd [kNm]", "opt_MRd"),
            ("Optimized Mcr [kNm]", "opt_Mcr"),
            ("Optimized Wk [mm]", "opt_Wk"),
            ("Optimized Cost", "opt_Cost"),
        ]
        self.opt_result_fields = {}

        for label_text, key in opt_fields:
            hbox = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(250)
            out = QLineEdit()
            out.setReadOnly(True)
            out.setAlignment(Qt.AlignmentFlag.AlignRight)
            out.setFixedHeight(28)
            hbox.addWidget(lbl)
            hbox.addWidget(out)
            opt_layout.addLayout(hbox)
            self.opt_result_fields[key] = out

        layout.addWidget(opt_group)

    def _update_display(self) -> None:
        ds = self.data_store
        self.result_fields["MEd"].setText(f"{ds.MEd:.2f}" if ds.MEd is not None else "N/A")
        self.result_fields["b"].setText(f"{ds.b:.1f}" if ds.b is not None else "N/A")
        self.result_fields["h"].setText(f"{ds.h:.1f}" if ds.h is not None else "N/A")
        self.result_fields["fi"].setText(f"{ds.fi:.1f}" if ds.fi is not None else "N/A")
        fck_txt = f"C{ds.fck}/{ds.fck+5}" if ds.fck is not None else "N/A"
        self.result_fields["fck"].setText(fck_txt)
        self.result_fields["act1"].setText(f"{ds.act1:.1f}" if ds.act1 is not None else "N/A")
        self.result_fields["act2"].setText(f"{ds.act2:.1f}" if ds.act2 is not None else "N/A")
        self.result_fields["num_rods_As1"].setText(str(ds.num_rods_As1) if ds.num_rods_As1 is not None else "N/A")
        self.result_fields["num_rods_As2"].setText(str(ds.num_rods_As2) if ds.num_rods_As2 is not None else "N/A")

    def _on_predict(self) -> None:
        ds = self.data_store
        try:
            n1_txt = self.result_fields["num_rods_As1"].text()
            n2_txt = self.result_fields["num_rods_As2"].text()
            n1 = int(n1_txt) if n1_txt not in ("N/A", "") else 0
            n2 = int(n2_txt) if n2_txt not in ("N/A", "") else 0

            # Use MEqp input if provided, otherwise use MEd
            MEqp_text = self.MEqp_input.text()
            MEd = float(MEqp_text) if MEqp_text else float(self.result_fields["MEd"].text())
            
            b = float(self.result_fields["b"].text())
            h = float(self.result_fields["h"].text())
            cnom = float(ds.cnom)
            fi = float(self.result_fields["fi"].text())
            fck_txt = self.result_fields["fck"].text()
            fck = float(fck_txt.split('/')[0][1:])

            As1 = n1 * math.pi * (fi ** 2) / 4
            As2 = n2 * math.pi * (fi ** 2) / 4

            if As2 == 0:
                results = predict_section_n1(MEd, b, h, fck, fi, cnom, As1)
                self.result_fields["pred_Mcr"].setText(f"{results['Mcr']:.2f}" if results['Mcr'] is not None else "N/A")
                self.result_fields["pred_MRd"].setText(f"{results['MRd']:.2f}" if results['MRd'] is not None else "N/A")
                self.result_fields["pred_Wk"].setText(f"{results['Wk']:.4f}" if results['Wk'] is not None else "N/A")
                self.result_fields["pred_Cost"].setText(f"{results['Cost']:.2f}" if results['Cost'] is not None else "N/A")
            else:
                results = predict_section_n2(MEd, b, h, fck, fi, cnom, As1, As2)
                self.result_fields["pred_Mcr"].setText(f"{results['Mcr']:.2f}" if results['Mcr'] is not None else "N/A")
                self.result_fields["pred_MRd"].setText(f"{results['MRd']:.2f}" if results['MRd'] is not None else "N/A")
                self.result_fields["pred_Wk"].setText(f"{results['Wk']:.4f}" if results['Wk'] is not None else "N/A")
                self.result_fields["pred_Cost"].setText(f"{results['Cost']:.2f}" if results['Cost'] is not None else "N/A")
        except Exception as e:
            print(f"Error in prediction: {e}")
            for key in ["pred_Mcr", "pred_MRd", "pred_Wk", "pred_Cost"]:
                self.result_fields[key].setText("Invalid input")

    def _on_optimize(self) -> None:
        ds = self.data_store
        try:
            # Get both MEqp and MEd values
            MEqp_text = self.MEqp_input.text()
            MEd = float(self.result_fields["MEd"].text())
            MEqp = float(MEqp_text) if MEqp_text else MEd  # Use MEqp if provided, else MEd
            
            b = float(self.result_fields["b"].text())
            h = float(self.result_fields["h"].text())
            cnom = float(ds.cnom)
            wk_max = 0.3

            # Find optimal solution based on MEqp (or MEd if MEqp not provided)
            optimal = find_best_solution(MEqp, MEd, b, h, cnom, wk_max)
            
            if not optimal:
                for field in self.opt_result_fields.values():
                    field.setText("No valid solution")
                return
                
            # Verify that MRd > MEd (the actual requirement)
            if optimal['MRd'] <= MEd:
                for field in self.opt_result_fields.values():
                    field.setText("MRd ≤ MEd - unsafe")
                return

            # If all checks passed, display the results
            self.opt_result_fields["opt_fi"].setText(f"{optimal['fi']:.0f}")
            self.opt_result_fields["opt_fck"].setText(f"C{optimal['fck']}/{int(optimal['fck'])+5}")
            self.opt_result_fields["opt_n1"].setText(str(int(optimal['n1'])))
            self.opt_result_fields["opt_n2"].setText(str(int(optimal['n2'])))
            self.opt_result_fields["opt_MRd"].setText(f"{optimal['MRd']:.2f}")
            self.opt_result_fields["opt_Mcr"].setText(f"{optimal['Mcr']:.2f}")
            self.opt_result_fields["opt_Wk"].setText(f"{optimal['Wk']:.4f}")
            self.opt_result_fields["opt_Cost"].setText(f"{optimal['Cost']:.2f}")
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            for field in self.opt_result_fields.values():
                field.setText("Error")
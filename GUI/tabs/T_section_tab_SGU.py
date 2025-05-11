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
from GUI.tabs.T_section_tab_SGN import CalculationDataT

from .optimization_module_T import predict_section_batch
from .optimization_module_T import calc_max_rods
from .optimization_module_T import generate_all_combinations
from .optimization_module_T import process_combinations_batch
from .optimization_module_T import find_optimal_solution

def predict_section(MEqp: float, beff: float, bw:float, h: float, hf: float, fck: float, fi: float, cnom: float, As1: float, As2: float):
    MODEL_PATHS = {
        'Mcr': {
            'model': r"nn_models\Tsectionplus\Mcr_model\model.keras",
            'scaler_X': r"nn_models\Tsectionplus\Mcr_model\scaler_X.pkl",
            'scaler_y': r"nn_models\Tsectionplus\Mcr_model\scaler_y.pkl"
        },
        'MRd': {
            'model': r"nn_models\Tsectionplus\MRd_model\model.keras",
            'scaler_X': r"nn_models\Tsectionplus\MRd_model\scaler_X.pkl",
            'scaler_y': r"nn_models\Tsectionplus\MRd_model\scaler_y.pkl"
        },
        'Wk': {
            'model': r"nn_models\Tsectionplus\Wk_model\model.keras",
            'scaler_X': r"nn_models\Tsectionplus\Wk_model\scaler_X.pkl",
            'scaler_y': r"nn_models\Tsectionplus\Wk_model\scaler_y.pkl"
        },
        'Cost': {
            'model': r"nn_models\Tsectionplus\cost_model\model.keras",
            'scaler_X': r"nn_models\Tsectionplus\cost_model\scaler_X.pkl",
            'scaler_y': r"nn_models\Tsectionplus\cost_model\scaler_y.pkl"
        }
    }

    MODEL_FEATURES = {
        'Mcr': ['beff', 'bw', 'h', 'hf', 'fi', 'fck', 'ro1', 'ro2'],
        'MRd': ['beff', 'bw', 'h', 'hf', 'fi', 'fck', 'ro1', 'ro2'],
        'Wk': ['MEqp', 'beff', 'bw', 'h', 'hf', 'fi', 'fck', 'ro1', 'ro2'],
        'Cost': ['beff', 'bw', 'h', 'hf', 'fi', 'fck', 'ro1', 'ro2']
    }
    
    # Calculate derived parameters
    d = h - cnom - fi / 2
    ro1 = As1 / (beff * d) if (beff * d) > 0 else 0
    ro2 = As2 / (beff * d) if (beff * d) > 0 else 0
    
    feature_values = {
        'MEqp': float(MEqp),
        'beff': float(beff),
        'bw': float(bw),
        'h': float(h),
        'hf': float(hf),
        'd': float(d),
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
            X_scaler = joblib.load(model_info['scaler_X'])
            y_scaler = joblib.load(model_info['scaler_y'])

            X_values = [feature_values[f] for f in MODEL_FEATURES[model_name]]
            X = pd.DataFrame([X_values], columns=MODEL_FEATURES[model_name])

            X_scaled = X_scaler.transform(np.log(X + 1e-8))
            pred_scaled = model.predict(X_scaled)
            pred = np.exp(y_scaler.inverse_transform(pred_scaled))[0][0]
            results[model_name] = pred
        except Exception as e:
            print(f"⚠️ Error in {model_name}: {e}")
            results[model_name] = None
    
    return results

class TSectionTabSGU(QWidget):
    """Tab that displays the stored calculation results from CalculationData."""
    
    def __init__(self, parent: QWidget | None = None, data_storeT: CalculationDataT = None) -> None:
        super().__init__(parent)
        self.data_storeT = data_storeT or CalculationDataT()
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

        self.result_fields = {}
        fields = [
            ("MEqp for prediction [kNm]", "MEqp"),  # Added MEqp input box
            ("Moment [kNm]", "MEd"),
            ("Flange width [mm]", "beff"),
            ("Web width [mm]", "bw"),
            ("Height [mm]", "h"),
            ("Flange height [mm]", "hf"),
            ("Reinforcement diameter [mm]", "fi"),
            ("Concrete class", "fck"),
            ("Provided As1 [mm²]", "act1"),
            ("Provided As2 [mm²]", "act2"),
            ("Number of tension rods", "num_rods_As1"),
            ("Number of compression rods", "num_rods_As2"),
            ("Pręty rozciągane w 1 warstwie", "rods_layer1"),
            ("Pręty rozciągane w 2 warstwie", "rods_layer2")
        ]
        for label_text, key in fields:
            hbox = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(250)
            out = QLineEdit()
            out.setReadOnly(key != "MEqp")  # Make MEqp editable, others read-only
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
            #("Predicted MRd [kNm]", "pred_MRd"),
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
        ds = self.data_storeT
        self.result_fields["MEd"].setText(f"{ds.MEd:.2f}" if ds.MEd is not None else "N/A")
        self.result_fields["beff"].setText(f"{ds.beff:.1f}" if ds.beff is not None else "N/A")
        self.result_fields["bw"].setText(f"{ds.bw:.1f}" if ds.bw is not None else "N/A")
        self.result_fields["h"].setText(f"{ds.h:.1f}" if ds.h is not None else "N/A")
        self.result_fields["hf"].setText(f"{ds.hf:.1f}" if ds.hf is not None else "N/A")
        self.result_fields["fi"].setText(f"{ds.fi:.1f}" if ds.fi is not None else "N/A")
        fck_txt = f"C{ds.fck}/{ds.fck+5}" if ds.fck is not None else "N/A"
        self.result_fields["fck"].setText(fck_txt)
        self.result_fields["act1"].setText(f"{ds.act1:.1f}" if ds.act1 is not None else "N/A")
        self.result_fields["act2"].setText(f"{ds.act2:.1f}" if ds.act2 is not None else "N/A")
        self.result_fields["num_rods_As1"].setText(str(ds.num_rods_As1) if ds.num_rods_As1 is not None else "N/A")
        self.result_fields["num_rods_As2"].setText(str(ds.num_rods_As2) if ds.num_rods_As2 is not None else "N/A")
        self.result_fields["rods_layer1"].setText(str(ds.num_rods_As1) if ds.num_rods_As1 is not None else "N/A")
        self.result_fields["rods_layer2"].setText(str(ds.num_rods_As2) if ds.num_rods_As2 is not None else "N/A")
        
        # Initialize MEqp with MEd value if not set
        if "MEqp" not in self.result_fields or not self.result_fields["MEqp"].text():
            self.result_fields["MEqp"].setText(f"{ds.MEd:.2f}" if ds.MEd is not None else "N/A")

    def _on_predict(self) -> None:
        try:
            n1_txt = self.result_fields["num_rods_As1"].text()
            n2_txt = self.result_fields["num_rods_As2"].text()
            n1 = int(n1_txt) if n1_txt not in ("N/A", "") else 0
            n2 = int(n2_txt) if n2_txt not in ("N/A", "") else 0

            # Get MEqp value from input box
            MEqp_text = self.result_fields["MEqp"].text()
            if MEqp_text in ("N/A", ""):
                MEqp = float(self.result_fields["MEd"].text())
            else:
                MEqp = float(MEqp_text)
            
            beff = float(self.result_fields["beff"].text())
            bw = float(self.result_fields["bw"].text())
            h = float(self.result_fields["h"].text())
            hf = float(self.result_fields["hf"].text())
            fi = float(self.result_fields["fi"].text())
            fck_txt = self.result_fields["fck"].text()
            fck = float(fck_txt.split('/')[0][1:])
            cnom = 30

            As1 = n1 * math.pi * (fi ** 2) / 4
            As2 = n2 * math.pi * (fi ** 2) / 4

            results = predict_section(MEqp, beff, bw, h, hf, fck, fi, cnom, As1, As2)
            self.result_fields["pred_Mcr"].setText(f"{results['Mcr']:.2f}" if results['Mcr'] is not None else "N/A")
            #self.result_fields["pred_MRd"].setText(f"{results['MRd']:.2f}" if results['MRd'] is not None else "N/A")
            self.result_fields["pred_Wk"].setText(f"{results['Wk']:.4f}" if results['Wk'] is not None else "N/A")
            self.result_fields["pred_Cost"].setText(f"{results['Cost']:.2f}" if results['Cost'] is not None else "N/A")
        except Exception as e:
            print(f"Error in prediction: {e}")
            for key in ["pred_Mcr", "pred_MRd", "pred_Wk", "pred_Cost"]:
                self.result_fields[key].setText("Invalid input")

    def _on_optimize(self) -> None:
        try:
            MEqp_text = self.result_fields["MEqp"].text()
            MEd = float(MEqp_text) if MEqp_text else float(self.result_fields["MEd"].text())
            beff = float(self.result_fields["beff"].text())
            bw = float(self.result_fields["bw"].text())
            h = float(self.result_fields["h"].text())
            hf = float(self.result_fields["hf"].text())
            cnom = 30
            wk_max = 0.3

            optimal = find_optimal_solution(MEd, beff, bw, h, hf, cnom, wk_max)
            if not optimal:
                for field in self.opt_result_fields.values():
                    field.setText("No valid solution")
                return

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
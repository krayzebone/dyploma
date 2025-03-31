import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import numpy as np
import math

from model_result import predict_section

Mcr_model_path = r"nn_models\nn_models_rect_section\Mcr_model\model.keras"
Mcr_scaler_X_path = r"nn_models\nn_models_rect_section\Mcr_model\scaler_X.pkl"
Mcr_scaler_y_path = r"nn_models\nn_models_rect_section\Mcr_model\scaler_y.pkl"

MRd_model_path = r"nn_models\nn_models_rect_section\MRd_model\model.keras"
MRd_scaler_X_path = r"nn_models\nn_models_rect_section\MRd_model\scaler_X.pkl"
MRd_scaler_y_path = r"nn_models\nn_models_rect_section\MRd_model\scaler_y.pkl"

Wk_model_path = r"nn_models\nn_models_rect_section\Wk_model\model.keras"
Wk_scaler_X_path = r"nn_models\nn_models_rect_section\Wk_model\scaler_X.pkl"
Wk_scaler_y_path = r"nn_models\nn_models_rect_section\Wk_model\scaler_y.pkl"

# Fixed input values
M_Ed = 346           # [kNm]
b = 1000             # [mm] 
 

c_nom = 30           # [mm]
fi_str = 8           # [mm]
f_yk = 500           # [MPa]


fi_gl_values = [8, 10, 12, 16, 20, 24, 28, 32]  # [mm]
f_ck_values = [25, 30, 35, 40, 45]

best_cost = float("inf")
best_solution = None


for f_ck in f_ck_values

    for fi_gl in fi_gl_values:

        f_cd = f_ck / 1.4 
        a_1 = c_nom + fi_str + fi_gl / 2.0
        h = float(a_1 + np.sqrt((2.0 * M_Ed * 1e6) / (b * f_cd)))
        d = h - a_1

        f_ctm = 0.3 * f_ck ** (2 / 3)
        A_s_min = max(0.26 * (f_ctm / f_yk) * b * d, 0.0013 * b * d)
        A_s_max = 0.04 * b * h

        A_s1, A_s2, cost = predict_section(M_Ed, b, h, d, a_1, fi_gl, f_ck, model_path, scaler_X_path, scaler_y_path)

        area_one_bar = math.pi * (fi_gl ** 2) / 4.0
        n1 = math.ceil(A_s1 / area_one_bar)
        n2 = math.ceil(A_s2 / area_one_bar)

        spacing = min(20, fi_gl)

        width_needed_tension = n1 * fi_gl + (n1 - 1) * spacing
        width_needed_compression = n2 * fi_gl + (n2 - 1) * spacing

        A_s1_new = n1 * math.pi * fi_gl**2 / 4
        A_s2_new = n2 * math.pi * fi_gl**2 / 4

        if width_needed_tension > b or width_needed_compression > b:
            continue

        if A_s1_new + A_s2_new  < A_s_min or A_s1_new + A_s2_new  > A_s_max:
            continue

        ro = (A_s1_new + A_s2_new) / (b * h)

        if cost < best_cost:
            best_cost = cost
            best_solution = {
                "f_ck": f_ck,
                "fi_gl": fi_gl,
                "n_bars_tension": n1,
                "n_bars_compression": n2,
                "h": h,
                "A_s1": A_s1,
                "A_s2": A_s2,
                "cost": cost,
                'ro': ro
            }

    if best_solution is None:
        print("No solution found that satisfies the bar-fitting constraints.")
    else:
        print("Best solution found:")
        for k, v in best_solution.items():
            print(f"  {k}: {v}")
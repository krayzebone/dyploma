from predict_rect_section import predict_section
import math
import numpy as np

def calc_rect_section(MEd, b, h, fck, fi_gl, c_nom):

    # Parametry materiałowe
    fyk = 500
    fcd = fck / 1.4     # wytrzymałość obliczeniowa betonu na ściskanie [MPa]
    fyd = fyk / 1.15    # wytrzymałość obliczeniowa stali na rozciąganie [MPa]
    E_s = 200000.0      # moduł Younga stali [MPa]

    # Wyznaczenie współczynnika względnej wysokości strefy ściskanej
    ksi_eff_lim = 0.8 * 0.0035 / (0.0035 + fyk / E_s)

    # Podstawowe wymiary przekroju
    fi_str = 8
    a_1 = c_nom + fi_gl / 2 + fi_str  
    d = h - a_1

    x_eff = ksi_eff_lim * d
    As2 = (
        (- x_eff * b * fcd * (d - 0.5 * x_eff) + MEd * 1e6)
        / (fyd * (d - a_1))
    )
    As1 = (As2 * fyd + x_eff * b * fcd) / fyd    
    return As1, As2

# Input data
input_data = {
    'MEd': 1580,
    'b': 873,
    'h': 711,
    'fck': 35,
    'fi_gl': 32,
    'c_nom': 40,
}

# Pass the parameters correctly
As1, As2 = calc_rect_section(
    input_data["MEd"],
    input_data["b"],
    input_data["h"],
    input_data["fck"],
    input_data["fi_gl"],
    input_data["c_nom"]
)

# Compute ro1, ro2
ro1 = As1 / (input_data['b'] * input_data['h'])
ro2 = As2 / (input_data['b'] * input_data['h'])

input_data['ro1'] = ro1
input_data['ro2'] = ro2
print(ro1,ro2)

# Calculate the number of steel rods using ro1, ro2 (NOT input_data['ro1'], etc.)
n1 = (ro1 * input_data['b'] * input_data['h']) / (input_data['fi_gl']**2 * 3.14159 / 4)
n2 = (ro2 * input_data['b'] * input_data['h']) / (input_data['fi_gl']**2 * 3.14159 / 4)

# Predict MRd, Mcr, Wk and cost for the input_data
predictions = predict_section(input_data)

print(f"Returned predictions: {predictions}")
print(f"Number of rods n1 = {n1}, n2 = {n2}")

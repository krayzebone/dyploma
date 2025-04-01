from predict_rect_section import predict_section
import math

# Your input data
input_data = {
    'MEd': 1580,
    'b': 873,
    'h': 711,
    'fck': 35,
    'fi_gl': 32,
    'c_nom': 40,
    'ro1': 0.020722,
    'ro2': 0.016836,
}

n1 = (input_data['ro1'] * input_data['b']* input_data['h']) / (input_data['fi_gl']**2 * 3.141 / 4)
n2 = (input_data['ro2'] * input_data['b']* input_data['h']) / (input_data['fi_gl']**2 * 3.141 / 4)

# Call the function (prints results automatically)
predictions = predict_section(input_data)

# You can also access the returned dict:
print(f"\nReturned predictions: {predictions}")
print(n1)
print(n2)
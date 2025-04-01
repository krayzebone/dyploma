from test import predict_section

# Your input data
input_data = {
    'MEd': 157.163915,
    'b': 1306.223854,
    'h': 629.038043,
    'fck': 35,
    'fi_gl': 28,
    'c_nom': 40,
    'ro1': 0.016487,
    'ro2': 0.010492,
}

# Call the function (prints results automatically)
predictions = predict_section(input_data)

# You can also access the returned dict:
print(f"\nReturned predictions: {predictions}")
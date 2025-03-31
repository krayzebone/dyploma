import numpy as np
import matplotlib.pyplot as plt

def calculate_moments(beam_type, length, load):

    if beam_type == "Przegub-przegub":
        M_max_plus = abs(load * length**2 / 8)
        M_max_minus = 0
        return max(M_max_plus, M_max_minus)
    
    elif beam_type == "Przegub-sztywny":
        M_max_plus = 0
        M_max_minus = abs(load * length**2 / 8)
        return max(M_max_plus, M_max_minus)
    
    elif beam_type == "Sztywny-sztywny":
        M_max_plus = abs(load * length**2 / 24)
        M_max_minus = abs(load * length**2 / 12)
        return max(M_max_plus, M_max_minus)
    
    elif beam_type == "Wspornik":
        M_max_plus = 0
        M_max_minus = abs(load * length**2 / 2)
        return max(M_max_plus, M_max_minus)
    
    else:
        print("Error")


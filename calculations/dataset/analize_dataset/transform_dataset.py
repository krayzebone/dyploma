import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
import warnings

# ----- Transformation Functions -----

# Log Transformation and Inverse
def log_transform(x, epsilon=1e-8):
    """Applies a log transform with a small epsilon offset for numerical stability."""
    return np.log(x + epsilon)

def log_inverse(y, epsilon=1e-8):
    """Inverse of the log transform."""
    return np.exp(y) - epsilon

# Yeo-Johnson Transformation and Inverse
def yeo_johnson_transform(x, lmbda):
    """
    Applies the Yeo-Johnson transformation.
    This implementation applies the transformation elementwise.
    """
    x = np.array(x, dtype=np.float64)
    transformed = np.empty_like(x)
    
    # For nonnegative values: 
    pos = x >= 0
    if lmbda != 0:
        transformed[pos] = ((x[pos] + 1) ** lmbda - 1) / lmbda
    else:
        transformed[pos] = np.log(x[pos] + 1)
        
    # For negative values:
    neg = ~pos
    if lmbda != 2:
        transformed[neg] = - ((-x[neg] + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
    else:
        transformed[neg] = -np.log(-x[neg] + 1)
    
    return transformed

def yeo_johnson_inverse(y, lmbda):
    """
    Computes the inverse of the Yeo-Johnson transformation.
    Note: The inverse is computed by using the fact that the transformation is monotonic.
    For y >= 0, the inverse corresponds to the nonnegative branch.
    For y < 0, the inverse uses the negative branch.
    """
    y = np.array(y, dtype=np.float64)
    x = np.empty_like(y)
    
    # For nonnegative branch (y corresponds to x >= 0)
    pos = y >= 0
    if lmbda != 0:
        x[pos] = np.power(lmbda * y[pos] + 1, 1/lmbda) - 1
    else:
        x[pos] = np.exp(y[pos]) - 1
        
    # For negative branch (y corresponds to x < 0)
    neg = ~pos
    if lmbda != 2:
        x[neg] = 1 - np.power(-(2 - lmbda) * y[neg] + 1, 1/(2 - lmbda))
    else:
        x[neg] = 1 - np.exp(-y[neg])
        
    return x

# Box-Cox Transformation and Inverse
def boxcox_transform(x, lmbda, epsilon=1e-8):
    """
    Applies the Box-Cox transformation.
    Note: Input values must be strictly positive; we add epsilon to ensure this.
    """
    x = np.array(x, dtype=np.float64) + epsilon
    if lmbda != 0:
        return (np.power(x, lmbda) - 1) / lmbda
    else:
        return np.log(x)

def boxcox_inverse(y, lmbda, epsilon=1e-8):
    """
    Computes the inverse of the Box-Cox transformation.
    """
    if lmbda != 0:
        return np.power(y * lmbda + 1, 1 / lmbda) - epsilon
    else:
        return np.exp(y) - epsilon

# Square Root Transformation and Inverse
def sqrt_transform(x, epsilon=1e-8):
    """Applies a square root transformation (offset by epsilon)."""
    return np.sqrt(x + epsilon)

def sqrt_inverse(y, epsilon=1e-8):
    """Inverse of the square root transform."""
    return np.power(y, 2) - epsilon

# ----- Main Processing Pipeline -----
def main():
    # Load the Parquet dataset (adjust the path as needed)
    df = pd.read_parquet(r"neural_networks\rect_section_n1\dataset\dataset_rect_n1_test5.parquet")
    
    # Define a dictionary for feature-specific transformations.
    # You can extend this dictionary to use any of the transformation functions above.
    transformations = {
        'b': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'h': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'd': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'fi': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'cnom': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'fck': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'ro1': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'MRd': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'Wk': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'Mcr': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},
        'Cost': {'transform': log_transform, 'inverse_transform': log_inverse, 'epsilon': 1e-8},

        # For example, if you want to use Yeo-Johnson on a feature 'x':
        # 'x': {'transform': yeo_johnson_transform, 'inverse_transform': yeo_johnson_inverse, 'lmbda': 1.0},
        # And for Box-Cox on a feature 'y':
        # 'y': {'transform': boxcox_transform, 'inverse_transform': boxcox_inverse, 'lmbda': 0.5, 'epsilon': 1e-8},
    }
    
    # Make a copy of the dataset to store the transformed features
    df_transformed = df.copy()
    
    # Apply the specified transformation for each feature if it exists in the dataframe.
    for feature, funcs in transformations.items():
        if feature in df_transformed.columns:
            # Collect additional keyword parameters (e.g., epsilon or lmbda)
            params = {key: val for key, val in funcs.items() if key not in ['transform', 'inverse_transform']}
            try:
                df_transformed[feature] = funcs['transform'](df_transformed[feature], **params)
            except Exception as e:
                warnings.warn(f"Transformation for feature '{feature}' failed: {e}")
    
    # List of features that were transformed (keys of the transformations dict)
    transformed_features = list(transformations.keys())
    
    # Apply Standard Scaling to the transformed features
    scaler = StandardScaler()
    df_transformed[transformed_features] = scaler.fit_transform(df_transformed[transformed_features])
    
    # Save the new dataset to a Parquet file
    df_transformed.to_parquet(r"your_dataset_transformed.parquet")
    print("Dataset transformed and saved to 'your_dataset_transformed.parquet'.")

if __name__ == "__main__":
    main()

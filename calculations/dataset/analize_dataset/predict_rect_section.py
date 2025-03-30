# ============================================
# Import Libraries
# ============================================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import joblib
import matplotlib.pyplot as plt

# ============================================
# Load the Preprocessed Data
# ============================================
file_path = r"C:\Users\marci\Desktop\projekt_marcin\dataset\files\dataset.parquet"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_parquet(file_path)

# Check for required columns
expected_columns = ["MEd", "MRd", "b", "d", "h", 'a1', "fi", "fck", "n1", "n2", "ro1", "ro2", "wk", "Mcr", "cost"]
if not all(col in df.columns for col in expected_columns):
    raise ValueError("The dataset is missing one or more required columns.")

# ============================================
# Load the Trained Model and Scalers
# ============================================
model_path = r"C:\Users\marci\Desktop\projekt_marcin\models\MRd_model\model.keras"
scaler_X_path = r"C:\Users\marci\Desktop\projekt_marcin\models\MRd_model\scaler_X.pkl"
scaler_y_path = r"C:\Users\marci\Desktop\projekt_marcin\models\MRd_model\scaler_y.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(scaler_X_path):
    raise FileNotFoundError(f"Scaler X file not found: {scaler_X_path}")
if not os.path.exists(scaler_y_path):
    raise FileNotFoundError(f"Scaler y file not found: {scaler_y_path}")

model = load_model(model_path)
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# ============================================
# Select Random Samples
# ============================================
sample_df = df.sample(50000, random_state=42)

# Define the features used as inputs
features = ["b", "h", "d", "fi", "fck", "ro1", "ro2"]

# Extract the input features (raw values)
X_sample = sample_df[features].values

# Check for non-positive values before log transformation
if np.any(X_sample <= 0):
    raise ValueError("All values in X_sample must be greater than 0 for logarithmic transformation.")

# Extract the actual target values in the original scale
actual_MRd = sample_df[["MRd"]].values  # shape (50000, 1)
actual_MRd = actual_MRd[:, 0]

# ============================================
# Scale Input Features (with Log Transformation)
# ============================================
# Apply log transformation to the input features
X_sample_log = X_sample
# Then scale the log-transformed features
X_sample_scaled = scaler_X.transform(X_sample_log)

# ============================================
# Make Predictions and Invert Transformations
# ============================================
# Predict using the model on the scaled (log-transformed) data
predictions = model.predict(X_sample_scaled)

# Check predictions shape
if predictions.shape != (50000, 1):
    raise ValueError(f"Unexpected shape for predictions: {predictions.shape}. Expected (50000, 3).")

# Invert scaling for predictions
predictions_log = scaler_y.inverse_transform(predictions)
# Invert the logarithmic transformation using np.exp to obtain predictions in the original scale
predicted_MRd = predictions_log[:, 0]  # First output is A_s1


# ============================================
# Compute Ratios (Predicted / Actual)
# ============================================
ratio_MRd = predicted_MRd / actual_MRd


# -- Helper function for printing stats
def print_stats(name, ratio_array):
    mean_ratio = np.mean(ratio_array)
    median_ratio = np.median(ratio_array)
    std_ratio = np.std(ratio_array)
    min_ratio = np.min(ratio_array)
    max_ratio = np.max(ratio_array)
    within_1_percent = np.sum((ratio_array >= 0.99) & (ratio_array <= 1.01)) / len(ratio_array) * 100

    print(f"=== {name} Ratio Stats ===")
    print(f" Mean   : {mean_ratio:.4f}")
    print(f" Median : {median_ratio:.4f}")
    print(f" Std    : {std_ratio:.4f}")
    print(f" Min    : {min_ratio:.4f}")
    print(f" Max    : {max_ratio:.4f}")
    print(f" % within 1% of 1.0: {within_1_percent:.2f}%")
    print("")

# Print stats for A_s1 and A_s2
print_stats("MRd", ratio_MRd)


# ============================================
# Plot the Comparisons for MRd
# ============================================
plt.figure(figsize=(18, 6))

# Plot for MRd
plt.subplot(1, 1, 1)  # 1 row, 3 columns, first plot
plt.scatter(actual_MRd, predicted_MRd, s=1, label='MRd')
plt.plot([actual_MRd.min(), actual_MRd.max()],
         [actual_MRd.min(), actual_MRd.max()],
         'r--', label='y = x')
plt.xlabel('Actual MRd')
plt.ylabel('Predicted MRd')
plt.title('Actual vs. Predicted MRd')
plt.grid(True)
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
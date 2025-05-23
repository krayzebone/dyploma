import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data from the Parquet file
file_path = r"neural_networks\rect_section_n1\dataset\dataset_rect_n1_test5_100k.parquet"
df = pd.read_parquet(file_path)

# Specify the features to transform and plot
features = ["MRd", "b", "d", "h", "fi", "fck", "ro1", "Wk", "Cost"]

# Apply log1p transformation to handle skewness and avoid log(0)
df_log = df[features].apply(lambda x: np.log(x))

# Standardize the log-transformed data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_log), columns=features)

# Plot histograms of transformed and scaled features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 5, i)
    sns.histplot(df_scaled[feature], bins=100, color='green')
    plt.title(f'Log Scaled: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

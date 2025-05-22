import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Load the data from the Parquet file
file_path = r"neural_networks\rect_section_n1\dataset\dataset_rect_n1_test5_100k.parquet"
df = pd.read_parquet(file_path)

# Specify the features to analyze
features = ["b", "h", "d", "fi","fck", "ro1", "MRd", "Wk", "Mcr", "Cost"]

# Create a DataFrame to store the statistics
stats_df = pd.DataFrame(index=features, 
                       columns=['Skewness', 'Kurtosis', 'IQR', 'Outliers Count', 'Outliers Percentage'])

# Calculate statistics for each feature
for feature in features:
    # Basic statistics
    skew = df[feature].skew()
    kurt = df[feature].kurtosis()
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate z-scores
    z_scores = np.abs(stats.zscore(df[feature]))
    
    # Identify outliers (using both IQR and z-score methods)
    outliers_iqr = ((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR)))
    outliers_zscores = (z_scores > 3)
    outliers = outliers_iqr | outliers_zscores
    
    # Store results
    stats_df.loc[feature, 'Skewness'] = skew
    stats_df.loc[feature, 'Kurtosis'] = kurt
    stats_df.loc[feature, 'IQR'] = IQR
    stats_df.loc[feature, 'Outliers Count'] = outliers.sum()
    stats_df.loc[feature, 'Outliers Percentage'] = f"{outliers.mean() * 100:.2f}%"
    
    # Add z-scores as a new column in the original DataFrame
    df[f'{feature}_zscore'] = z_scores

# Save statistics to Excel
output_excel_path = r"neural_networks\rect_section_n2\dataset\dataset_statistics.xlsx"
with pd.ExcelWriter(output_excel_path) as writer:
    stats_df.to_excel(writer, sheet_name='Summary Statistics')
    
    # Save z-scores for all features to a separate sheet
    zscore_cols = [col for col in df.columns if '_zscore' in col]
    df[features + zscore_cols].to_excel(writer, sheet_name='Data with Z-scores', index=False)

print(f"Statistics saved to {output_excel_path}")

# Plot histograms for each feature
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 5, i)  # Adjust the grid size based on the number of features
    sns.histplot(df[feature], kde=False, color='blue', bins=100)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
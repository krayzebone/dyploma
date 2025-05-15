import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Parquet file
file_path = r"neural_networks\rect_section_n1\dataset\dataset_rect_n1_test5.parquet"
df = pd.read_parquet(file_path)

# Specify the features to plot
features = ["b", "h", "d", "fi", "cnom", "fck", "ro1", "MRd", "Wk", "Mcr", "Cost"]

# Set up the matplotlib figure
plt.figure(figsize=(15, 10))

# Plot histograms for each feature
for i, feature in enumerate(features, 1):
    plt.subplot(3, 5, i)  # Adjust the grid size based on the number of features
    sns.histplot(df[feature], kde=False, color='blue', bins=100)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
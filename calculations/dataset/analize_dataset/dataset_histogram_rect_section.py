import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Parquet file
file_path = r"your_dataset_transformed.parquet"
df = pd.read_parquet(file_path)

# Specify the features to plot
features = ["MEd", "MRd", "b", "d", "h", 'a1', "fi", "fck", "n1", "n2", "ro1", "ro2", "wk", "Mcr", "cost"]

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
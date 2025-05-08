import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load dataset ===
file_path = 'dataset_files\Tsection\Tsection_balanced.parquet'
df = pd.read_parquet(file_path)

# === Basic info ===
print("\n--- Dataset Overview ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# === Check for missing values ===
print("\n--- Missing Values ---")
print(df.isnull().sum())

# === Check class distributions ===
print("\n--- fck Distribution ---")
print(df['fck'].value_counts())
print("\n--- fi Distribution ---")
print(df['fi'].value_counts())

# === Plot class distributions ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='fck', data=df)
plt.title('fck Distribution')

plt.subplot(1, 2, 2)
sns.countplot(x='fi', data=df)
plt.title('fi Distribution')

plt.tight_layout()
plt.savefig('class_distributions.png')
plt.show()

# === Plot histograms for features ===
numerical_features = ['MEd', 'beff', 'bw', 'h', 'hf', 'cnom', 'fi_str']
df[numerical_features].hist(bins=30, figsize=(14, 10))
plt.suptitle("Feature Histograms")
plt.tight_layout()
plt.savefig('feature_histograms.png')
plt.show()

# === Correlation matrix ===
corr_matrix = df[numerical_features + ['fi_str']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig('correlation_heatmap.png')
plt.show()

# === Imbalance Warnings ===
fck_counts = df['fck'].value_counts(normalize=True) * 100
fi_counts = df['fi'].value_counts(normalize=True) * 100

print("\n--- Imbalance Check ---")
print("Classes with <5% representation:")
print("fck:")
print(fck_counts[fck_counts < 5])
print("fi:")
print(fi_counts[fi_counts < 5])

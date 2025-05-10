import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Specify the path to your .parquet file
#parquet_file_path_preprocessed_data = r"C:\Users\marci\Desktop\Lagrange\dataset_files\preprocessed_files\preprocessed_results.parquet"
parquet_file_path = r"dataset_files\Tsection\datasetSGU.parquet"

# Read the .parquet file into a DataFrame
#df1 = pd.read_parquet(parquet_file_path_preprocessed_data)
df = pd.read_parquet(parquet_file_path)

# Filter rows where f_ck equals 25
#filtered_df = df[df["wk"] > 0.3]

# Count the number of rows that match the condition
#row_count = len(filtered_df)

#print(f"Number of rows with wk = 25: {row_count}")

# Display the first few rows of the DataFrame
print("First 5 rows of the DataFrame:")
#print(df1.head())
print(df.head())
print(len(df))

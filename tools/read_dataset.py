import pandas as pd
import numpy as np

# Path to your .parquet file
parquet_file_path = r"neural_networks\rect_section_n1\dataset\dataset_rect_n1_test2.parquet"

# Read the .parquet file into a DataFrame
df = pd.read_parquet(parquet_file_path)

# Select only the numeric columns (or replace this with your own list)
num_cols = df.select_dtypes(include=[np.number]).columns

# Find rows where any of those numeric columns is < 0
neg_rows = df[(df[num_cols] < 0).any(axis=1)]

print(f"Total rows in dataset: {len(df)}")
print(f"Rows with at least one feature < 0: {len(neg_rows)}\n")

if not neg_rows.empty:
    print("Here are the rows with negative values:")
    print(neg_rows)
else:
    print("✔️ No negative values found in any numeric column.")

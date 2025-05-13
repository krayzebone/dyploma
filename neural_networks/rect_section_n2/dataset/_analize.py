import pandas as pd

# Load the Parquet file
file_path = r"neural_networks/rect_sectionn1/dataset/dataset_rect_n1.parquet"
df = pd.read_parquet(file_path)

# Filter rows where MRd > 10000 or MRd < 0
filtered_df = df[(df['MRd'] > 10000) | (df['MRd'] < 0)]

# Print the filtered rows
print("Rows where MRd > 10000 or MRd < 0:")
print(filtered_df)
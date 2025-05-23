import pandas as pd

# Paths
input_path  = r"neural_networks\rect_section_n2\dataset\dataset_rect_n2.parquet"
output_path = r"neural_networks\rect_section_n2\dataset\dataset_rect_n2.parquet_100k"

# Read the full dataset
df = pd.read_parquet(input_path)

# Take a random 100_000-row sample (reproducible)
df_trim = df.sample(n=100_000, random_state=42)

# (Alternatively, to just take the first 100k rows: df_trim = df.iloc[:100_000] )

# Write the trimmed dataset back out
df_trim.to_parquet(output_path, index=False)

print(f"Trimmed dataset saved to {output_path} ({len(df_trim)} rows).")
